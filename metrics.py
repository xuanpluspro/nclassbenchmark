"""Multiclass adapter and compatibility exports for shared metric protocol v2."""
import csv
from pathlib import Path
from segmentation_metrics import DatasetSegmentationMetrics


class MulticlassSegmentationMetrics(DatasetSegmentationMetrics):
    def __init__(self, num_classes, class_names=None, ignore_index=255,
                 background_index=0, boundary_tolerance=2.0):
        super().__init__(num_classes, class_names, ignore_index, background_index, boundary_tolerance)

    def compute(self):
        result = super().compute()
        # mdice retains its former all-class meaning; canonical dice is foreground-only.
        result.update(mdice=result["dice_all"], mdice_fg=result["dice"],
                      miou_fg=result["foreground_miou"])
        for row in result["per_class"]:
            row["acc"] = row["recall"]
        return result


def export_metrics(result, output, split, background_index=0):
    """Keep legacy split CSVs alongside the new standardized exports."""
    if result["background_index"] != background_index:
        raise ValueError("Background index differs from the metric configuration")
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    with (output/f"{split}_per_class.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["per_class"][0]))
        writer.writeheader(); writer.writerows(result["per_class"])
    with (output/f"{split}_confusion_matrix.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true / predicted", *[r["class_name"] for r in result["per_class"]]])
        for row, values in zip(result["per_class"], result["confusion_matrix"]):
            writer.writerow([row["class_name"], *values])
