"""Dataset-level multiclass metrics; rows=true, columns=prediction; no TN-based class accuracy."""
import csv
from pathlib import Path
import numpy as np


def _divide(a, b):
    return np.divide(a, b, out=np.full_like(a, np.nan, dtype=float), where=b != 0)


def _mean(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else None


def _values(x):
    return [float(v) if np.isfinite(v) else None for v in x]


class MulticlassSegmentationMetrics:
    def __init__(self, num_classes, class_names=None, ignore_index=255):
        self.num_classes = num_classes
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]
        self.ignore_index = ignore_index
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, prediction, target):
        if hasattr(prediction, 'detach'): prediction = prediction.detach().cpu().numpy()
        if hasattr(target, 'detach'): target = target.detach().cpu().numpy()
        pred, truth = np.asarray(prediction), np.asarray(target)
        if pred.shape != truth.shape: raise ValueError('Prediction/target shape mismatch')
        valid = truth != self.ignore_index
        pred, truth = pred[valid], truth[valid]
        c = self.num_classes
        if not np.issubdtype(pred.dtype,np.integer) or not np.issubdtype(truth.dtype,np.integer):
            raise ValueError('Metrics require integer class IDs')
        if ((truth < 0)|(truth >= c)|(pred < 0)|(pred >= c)).any():
            raise ValueError('Out-of-range class ID')
        self.confusion += np.bincount(c*truth.astype(np.int64)+pred.astype(np.int64), minlength=c*c).reshape(c,c)

    def compute(self):
        cm = self.confusion
        support, predicted = cm.sum(1), cm.sum(0)
        tp = np.diag(cm).astype(float)
        iou = _divide(tp, support+predicted-tp)
        acc = _divide(tp, support)  # Per-class recall, as used in semantic segmentation mAcc.
        precision = _divide(tp, predicted)
        dice = _divide(2*tp, support+predicted)
        per_class = [dict(class_id=i, class_name=self.class_names[i], iou=_values(iou)[i],
                          acc=_values(acc)[i], dice=_values(dice)[i], precision=_values(precision)[i],
                          support=int(support[i]), predicted_pixels=int(predicted[i])) for i in range(self.num_classes)]
        return dict(miou=_mean(iou), macc=_mean(acc), mdice=_mean(dice),
                    pixacc=float(tp.sum()/cm.sum()) if cm.sum() else None,
                    per_class=per_class, confusion_matrix=cm.tolist(), valid_pixels=int(cm.sum()),
                    evaluated_iou_classes=int(np.isfinite(iou).sum()),
                    evaluated_acc_classes=int(np.isfinite(acc).sum()))


def export_metrics(result, output, split, background_index=0):
    output = Path(output)
    with (output/f'{split}_per_class.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w = csv.DictWriter(f,fieldnames=list(result['per_class'][0]))
        w.writeheader(); w.writerows(result['per_class'])
    with (output/f'{split}_confusion_matrix.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); names=[r['class_name'] for r in result['per_class']]
        w.writerow(['true / predicted',*names])
        for name,row in zip(names,result['confusion_matrix']): w.writerow([name,*row])
    # Exclude background only from the mean: retain all background-related FP/FN.
    fg=[r for r in result['per_class'] if r['class_id'] != background_index]
    result['miou_fg']=_mean([r['iou'] if r['iou'] is not None else np.nan for r in fg])
    result['macc_fg']=_mean([r['acc'] if r['acc'] is not None else np.nan for r in fg])
    result['mdice_fg']=_mean([r['dice'] if r['dice'] is not None else np.nan for r in fg])
