import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import batch_evaluate
import evaluate_checkpoint
from segmentation_metrics import DatasetSegmentationMetrics


class BatchEvaluationTests(unittest.TestCase):
    def test_discovery_uses_best_only(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'last.pth').touch()
            with self.assertRaises(FileNotFoundError):batch_evaluate.discover_checkpoints(root)
            (root/'one').mkdir();(root/'one/best.pth').touch()
            self.assertEqual(batch_evaluate.discover_checkpoints(root),[(root/'one/best.pth').resolve()])

    def fake_evaluation(self, command, log):
        self.assertIn('evaluate_checkpoint.py',command[2])
        self.assertNotIn('--epochs',command)
        weights=Path(command[command.index('--weights')+1])
        destination=Path(command[command.index('--output')+1]);destination.mkdir(parents=True)
        meter=DatasetSegmentationMetrics(4)
        meter.update(np.arange(4),np.arange(4))
        (destination/'metrics.json').write_text(json.dumps(dict(arch='Unet',encoder='resnet34',seed=42,
            best_epoch=4,parameters=20,evaluation_dataset='/dataset',test=meter.compute())))
        Path(log).write_text('mock evaluation completed')
        self.assertTrue(weights.is_file())
        return 0

    def test_batch_creates_all_tables_and_preserves_checkpoints(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'runs'
            for name in ('one_seed42','two_seed42'):
                p=root/name;p.mkdir(parents=True);(p/'best.pth').write_bytes(b'unchanged checkpoint')
            output=Path(d)/'evaluation'
            with patch('batch_evaluate.run_logged',side_effect=self.fake_evaluation) as run:
                batch_evaluate.main(['--runs-root',str(root),'--output',str(output),'--no-benchmark'])
            self.assertEqual(run.call_count,2)
            self.assertEqual(len(json.loads((output/'evaluation_status.json').read_text())),2)
            for name in ('results.csv','all_class_metrics.csv','test_class_iou.csv','val_class_iou.csv'):
                self.assertTrue((output/name).is_file())
            for path in root.rglob('best.pth'):self.assertEqual(path.read_bytes(),b'unchanged checkpoint')

    def test_one_failed_model_does_not_stop_remaining_models(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'runs';root.mkdir()
            for name in ('one','two'):
                (root/name).mkdir();(root/name/'best.pth').touch()
            output=Path(d)/'evaluation'
            calls=[]
            def run(command, log):
                calls.append(command)
                return 7 if len(calls)==1 else self.fake_evaluation(command,log)
            with patch('batch_evaluate.run_logged',side_effect=run):
                with self.assertRaises(SystemExit) as error:
                    batch_evaluate.main(['--runs-root',str(root),'--output',str(output)])
            self.assertEqual(error.exception.code,1);self.assertEqual(len(calls),2)
            manifest=json.loads((output/'summary_manifest.json').read_text())
            self.assertEqual(manifest['models'],1);self.assertEqual(len(manifest['failures']),1)

    def test_existing_output_rejected_before_worker(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'run';root.mkdir();(root/'best.pth').touch()
            with patch('batch_evaluate.run_logged') as worker:
                with self.assertRaises(FileExistsError):
                    batch_evaluate.main(['--runs-root',str(root),'--output',str(root)])
                worker.assert_not_called()

    def test_parser_needs_no_training_options(self):
        args=evaluate_checkpoint.parse_args(['--weights','one/best.pth','--output','new'])
        self.assertEqual(args.splits,['test']);self.assertEqual(args.batch_size,8)
        self.assertIsNone(args.data_root)

    def test_mixed_dataset_configs_rejected_before_workers(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for k in (4,6):
                model=root/f'model{k}';model.mkdir();(model/'best.pth').touch()
                (model/'config.json').write_text(json.dumps(dict(num_classes=k,size=512,
                    class_names=[f'class_{i}' for i in range(k)],data_root=f'/data{k}')))
            with patch('batch_evaluate.run_logged') as worker:
                with self.assertRaises(ValueError):
                    batch_evaluate.main(['--runs-root',str(root),'--output',str(root/'evaluation')])
                worker.assert_not_called()


HAS_TORCH = importlib.util.find_spec('torch') is not None
HAS_SMP = importlib.util.find_spec('segmentation_models_pytorch') is not None


@unittest.skipUnless(HAS_TORCH and HAS_SMP,'Requires torch and segmentation_models_pytorch')
class CheckpointCPUIntegration(unittest.TestCase):
    def test_actual_load_and_test_only_inference_never_updates_weights(self):
        import torch
        from PIL import Image
        for classes in (4,6):
            with self.subTest(classes=classes),tempfile.TemporaryDirectory() as d:
                root=Path(d);data=root/'data'
                for folder in ('JPEGImages','SegmentationClass','ImageSets'):(data/folder).mkdir(parents=True)
                Image.new('RGB',(32,32),(60,100,80)).save(data/'JPEGImages/one.png')
                Image.fromarray((np.indices((32,32))[1]%classes).astype(np.uint8)).save(data/'SegmentationClass/one.png')
                (data/'ImageSets/test.txt').write_text('one\n')
                model=torch.nn.Conv2d(3,classes,1)
                config=dict(arch='Unet',encoder='resnet34',num_classes=classes,size=32,
                    class_names=[f'class_{i}' for i in range(classes)],seed=42,
                    data_root='/moved/old/path',ignore_index=255,background_index=0,test_split='test')
                weights=root/'best.pth';torch.save(dict(config=config,model=model.state_dict(),epoch=9),weights)
                original=weights.read_bytes(); output=root/'output'
                with patch('segmentation_models_pytorch.create_model',return_value=model) as create, \
                     patch('torch.optim.AdamW',side_effect=AssertionError('Must not create an optimizer')):
                    result=evaluate_checkpoint.main(['--weights',str(weights),'--output',str(output),
                        '--data-root',str(data),'--device','cpu','--workers','0','--no-benchmark'])
                self.assertIsNone(create.call_args.kwargs['encoder_weights'])
                self.assertEqual(result['best_epoch'],10)
                self.assertEqual(result['test']['num_classes'],classes)
                self.assertEqual(weights.read_bytes(),original)
                self.assertTrue((output/'class_metrics.csv').is_file())
                self.assertFalse(list(output.rglob('*.pth')))


if __name__ == '__main__':unittest.main()
