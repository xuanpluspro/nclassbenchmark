import csv
import json
from pathlib import Path
import tempfile
import unittest

from result_tables import read_record, export_tables
from segmentation_metrics import DatasetSegmentationMetrics
from summarize_results import summarize
import numpy as np


class ResultTableTests(unittest.TestCase):
    def make_legacy(self, root, name="unet_seed42", classes=4):
        directory = Path(root)/name; directory.mkdir(parents=True)
        rows = [dict(class_id=i, class_name=f"class_{i}", iou=.2+i*.1,
            dice=.3+i*.1, acc=.4+i*.1, precision=.5, support=10, predicted_pixels=10)
            for i in range(classes)]
        metrics = dict(arch="Unet", encoder="resnet34", seed=42, best_epoch=3, parameters=10,
            test=dict(miou=.5, mdice=.99, mdice_fg=.4, macc=.6, pixacc=.8,
                      miou_fg=.3, per_class=rows))
        path = directory/'metrics.json'; path.write_text(json.dumps(metrics))
        (directory/'config.json').write_text(json.dumps(dict(data_root='/dataset',background_index=0)))
        return path

    def test_legacy_dice_never_uses_all_class_value(self):
        with tempfile.TemporaryDirectory() as d:
            record = read_record(self.make_legacy(d))
            self.assertEqual(record['test']['dice'],.4)
            self.assertEqual(record['test']['dice_all'],.99)
            self.assertEqual(record['test']['metric_protocol'],'legacy-unversioned')
            self.assertIsNone(record['test']['boundary_f1'])
            self.assertEqual(record['test']['per_class'][1]['recall'],.5)

    def test_legacy_missing_foreground_dice_derived_from_classes(self):
        with tempfile.TemporaryDirectory() as d:
            path=self.make_legacy(d)
            value=json.loads(path.read_text()); del value['test']['mdice_fg']
            path.write_text(json.dumps(value))
            self.assertAlmostEqual(read_record(path)['test']['dice'],.5)

    def test_two_models_all_classes_and_wide_tables(self):
        with tempfile.TemporaryDirectory() as d:
            records=[read_record(self.make_legacy(d,name)) for name in ('one_seed42','two_seed42')]
            output=Path(d)/'summary';export_tables(records,output)
            with (output/'all_class_metrics.csv').open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),8)
            with (output/'test_class_iou.csv').open(encoding='utf-8-sig') as f: wide=list(csv.DictReader(f))
            self.assertEqual(len(wide),2)
            self.assertEqual(float(wide[0]['iou_3_class_3']),.5)
            self.assertEqual(wide[0]['dice'],'0.4')
            self.assertEqual(wide[0]['fps'],'')
            with (output/'results.csv').open(encoding='utf-8-sig') as f: self.assertEqual(len(list(csv.DictReader(f))),2)

    def test_six_classes_and_distinct_seeds(self):
        with tempfile.TemporaryDirectory() as d:
            p1=self.make_legacy(d,'model_seed42',6)
            p2=self.make_legacy(d,'model_seed43',6)
            data=json.loads(p2.read_text());data['seed']=43;p2.write_text(json.dumps(data))
            output=Path(d)/'summary'
            export_tables([read_record(p1,name='model'),read_record(p2,name='model')],output)
            with (output/'test_class_iou.csv').open(encoding='utf-8-sig') as f:
                reader=csv.DictReader(f);rows=list(reader)
                self.assertEqual(len([x for x in reader.fieldnames if x.startswith('iou_')]),6)
                self.assertEqual([r['seed'] for r in rows],['42','43'])

    def test_inconsistent_class_maps_and_duplicate_runs_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            a=read_record(self.make_legacy(d,'one'))
            b=read_record(self.make_legacy(d,'two'))
            b['test']['per_class'][1]['class_name']='another_label'
            with self.assertRaises(ValueError):export_tables([a,b],Path(d)/'summary')
            with self.assertRaises(ValueError):export_tables([a,a],Path(d)/'summary')

    def test_different_dataset_roots_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            a=read_record(self.make_legacy(d,'one'))
            b=read_record(self.make_legacy(d,'two'));b['evaluation_dataset']='/another_dataset'
            with self.assertRaises(ValueError):export_tables([a,b],Path(d)/'summary')

    def test_summary_does_not_touch_inputs_or_load_models(self):
        with tempfile.TemporaryDirectory() as d:
            path=self.make_legacy(d);original=path.read_bytes()
            summarize(d,Path(d)/'summary')
            self.assertEqual(path.read_bytes(),original)
            # Repeated summaries must not discover summary CSVs as extra runs.
            self.assertEqual(len(summarize(d,Path(d)/'summary')),1)

    def test_v2_boundary_scores_and_class_data_are_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            path=self.make_legacy(d)
            value=json.loads(path.read_text())
            meter=DatasetSegmentationMetrics(4)
            labels=np.zeros((1,4,4),int);labels[0,1,1]=1
            meter.update(labels,labels)
            value['test']=meter.compute();path.write_text(json.dumps(value))
            record=read_record(path)
            self.assertEqual(record['test']['boundary_f1'],1)
            self.assertEqual(record['test']['per_class'][1]['hd95'],0)
            self.assertEqual(record['test']['metric_protocol'],'bimseg-segmentation-v2')

    def test_failures_are_explicit_and_no_fake_zero_scores(self):
        with tempfile.TemporaryDirectory() as d:
            export_tables([],d,[dict(name='failed_model',status=1)])
            with (Path(d)/'results.csv').open(encoding='utf-8-sig') as f: row=next(csv.DictReader(f))
            self.assertEqual(row['status'],'1');self.assertEqual(row['miou'],'')
            self.assertEqual(json.loads((Path(d)/'summary_manifest.json').read_text())['models'],0)


if __name__ == '__main__': unittest.main()
