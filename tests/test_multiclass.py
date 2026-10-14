import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from metrics import MulticlassSegmentationMetrics, export_metrics

class MetricsTests(unittest.TestCase):
    def test_hand_calculated_dataset_confusion(self):
        m=MulticlassSegmentationMetrics(3)
        m.update(np.array([0,1,2,0,2]),np.array([0,0,1,1,2]))
        r=m.compute()
        np.testing.assert_array_equal(r['confusion_matrix'],[[1,1,0],[1,0,1],[0,0,1]])
        self.assertAlmostEqual(r['miou'],(1/3+0+1/2)/3)
        self.assertAlmostEqual(r['macc'],(.5+0+1)/3)
        self.assertAlmostEqual(r['pixacc'],2/5)
        with tempfile.TemporaryDirectory() as d:
            export_metrics(r,d,'test')
            self.assertAlmostEqual(r['miou_fg'],.25)

    def test_absent_and_false_positive_classes(self):
        m=MulticlassSegmentationMetrics(4)
        m.update(np.array([0,1]),np.array([0,0]))
        r=m.compute()
        self.assertEqual(r['per_class'][1]['iou'],0)
        self.assertIsNone(r['per_class'][1]['acc'])
        self.assertIsNone(r['per_class'][2]['iou'])
        self.assertEqual(r['evaluated_iou_classes'],2)
        self.assertAlmostEqual(r['miou'],.25)
        json.dumps(r,allow_nan=False)

    def test_ignore_and_batch_invariance(self):
        a=MulticlassSegmentationMetrics(3); b=MulticlassSegmentationMetrics(3)
        pred=np.array([0,2,1,0]); truth=np.array([0,1,255,2])
        a.update(pred,truth)
        for i in range(4): b.update(pred[i:i+1],truth[i:i+1])
        self.assertEqual(a.compute(),b.compute())
        self.assertEqual(a.compute()['valid_pixels'],3)
        with self.assertRaises(ValueError): a.update(np.array([0]),np.array([3]))

    def test_palette_mask_and_multiclass_loss(self):
        import torch
        from train import VOC,loss_fn
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            for folder in ['JPEGImages','SegmentationClass','ImageSets/Main']: (p/folder).mkdir(parents=True)
            (p/'ImageSets/Main/train.txt').write_text('one\n')
            Image.new('RGB',(32,32)).save(p/'JPEGImages/one.jpg')
            mask=np.tile(np.array([0,1,2,255],dtype=np.uint8),(32,8))
            im=Image.fromarray(mask).convert('P'); im.save(p/'SegmentationClass/one.png')
            x,y=VOC(d,'train',32,3)[0]
            self.assertEqual(set(y.unique().tolist()),{0,1,2,255})
            self.assertEqual(y.dtype,torch.int64)
            logits=torch.randn(1,3,32,32,requires_grad=True)
            loss=loss_fn(logits,y[None]); loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(logits.grad.abs().sum().item(),0)
            empty=loss_fn(logits,torch.full_like(y[None],255)); self.assertEqual(empty.item(),0)

    def test_configured_model_output_channels(self):
        import torch
        import segmentation_models_pytorch as smp
        torch.set_num_threads(1)
        jobs=json.loads((Path(__file__).resolve().parents[1]/'experiments.json').read_text())
        for job in jobs:
            with self.subTest(model=job['name']):
                model=smp.create_model(job['arch'],encoder_name=job['encoder'],encoder_weights=None,classes=6).eval()
                with torch.inference_mode(): output=model(torch.zeros(1,3,64,64))
                self.assertEqual(tuple(output.shape),(1,6,64,64))
                del model, output

if __name__=='__main__': unittest.main()
