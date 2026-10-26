"""Normalize saved results and write cross-model overall/per-class tables.

No checkpoint loading, model execution or training. Legacy JSON remains marked
legacy-unversioned; missing boundary values are never fabricated.
"""
import csv
import json
from pathlib import Path

from segmentation_metrics import CLASS_FIELDS, RESULT_FIELDS, result_rows


def _first(mapping, *keys):
    for key in keys:
        if mapping.get(key) is not None:
            return mapping[key]
    return None


def _mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def read_record(path, name=None, source_run=None):
    path = Path(path).resolve()
    result = json.loads(path.read_text(encoding="utf-8-sig"))
    config_path = path.parent / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8-sig")) if config_path.is_file() else {}
    result = dict(result)
    result.update(name=name or result.get("name") or path.parent.name,
                  seed=result.get("seed", config.get("seed")),
                  arch=result.get("arch", config.get("arch", "")),
                  encoder=result.get("encoder", config.get("encoder", "")),
                  source_run=source_run or str(path.parent), metrics_file=str(path))
    suffix = f"_seed{result['seed']}"
    if result.get("seed") is not None and result["name"].endswith(suffix):
        result["name"] = result["name"][:-len(suffix)]
    result["evaluation_dataset"] = result.get("evaluation_dataset") or config.get("data_root")
    result["evaluation_size"] = result.get("evaluation_size") or result.get("source_config", {}).get("size") or config.get("size")
    if not any(split in result for split in ("val", "test")):
        raise ValueError(f"No val/test metrics in {path}")
    for split in ("val", "test"):
        if split not in result:
            continue
        raw = result[split]
        classes = raw.get("per_class")
        if not classes:
            raise ValueError(f"Missing per_class in {path}: {split}; re-evaluate the checkpoint")
        bg = raw.get("background_index", config.get("background_index", 0))
        rows = []
        for item in classes:
            row = dict(item)
            row.update(class_id=int(item["class_id"]),
                class_name=item.get("class_name", f"class_{item['class_id']}"),
                is_background=int(item["class_id"]) == bg,
                iou=_first(item, "iou", "IoU"), dice=_first(item, "dice", "Dice"),
                precision=_first(item, "precision", "Precision"),
                recall=_first(item, "recall", "acc", "Recall", "Accuracy"),
                support=_first(item, "support", "pixels"))
            rows.append(row)
        rows.sort(key=lambda row: row["class_id"])
        if [row["class_id"] for row in rows] != list(range(len(rows))):
            raise ValueError(f"Class IDs must be unique and contiguous in {path}: {split}")
        fg = [row for row in rows if not row["is_background"]]
        normalized = dict(raw, per_class=rows, num_classes=len(rows), background_index=bg)
        aliases = {
            "miou": ("miou", "Mean IoU"), "macc": ("macc", "Mean Accuracy"),
            "pixacc": ("pixacc", "accuracy", "Pixel Accuracy"),
            "foreground_miou": ("foreground_miou", "miou_fg"),
            "dice": ("dice", "mdice_fg", "Dice"),
            "dice_all": ("dice_all", "mdice"),
            "precision": ("precision", "Precision"), "recall": ("recall", "Recall"),
            "specificity": ("specificity", "Specificity"),
            "fwiou": ("fwiou", "Frequency Weighted IoU"),
            "hd95": ("hd95", "HD95"), "assd": ("assd", "ASSD"),
            "boundary_f1": ("boundary_f1", "Boundary F1"),
        }
        for key, candidates in aliases.items():
            normalized[key] = _first(raw, *candidates)
        # Never use legacy mdice (includes background) as foreground dice.
        fallbacks = {"miou": _mean([r["iou"] for r in rows]),
            "foreground_miou": _mean([r["iou"] for r in fg]),
            "dice": _mean([r["dice"] for r in fg]),
            "dice_all": _mean([r["dice"] for r in rows]),
            "macc": _mean([r["recall"] for r in rows]),
            "macc_fg": _mean([r["recall"] for r in fg]),
            "background_iou": rows[bg]["iou"]}
        for key, value in fallbacks.items():
            if normalized.get(key) is None:
                normalized[key] = value
        normalized["metric_protocol"] = raw.get("metric_protocol", result.get("metric_protocol", "legacy-unversioned"))
        result[split] = normalized
    return result


def _write_csv(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def export_tables(records, output, failures=()):
    """Write stable long tables and class-IoU wide tables for each split."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    seen = set(); mapping = None; datasets = set(); sizes = set()
    for record in records:
        key = (record["name"], record.get("seed"))
        if key in seen:
            raise ValueError(f"Duplicate model/seed {key}; choose one experiment/evaluation directory")
        seen.add(key)
        if record.get("evaluation_dataset"):
            datasets.add(str(Path(record["evaluation_dataset"]).resolve()))
        if record.get("evaluation_size"):
            sizes.add(record["evaluation_size"])
        for split in ("val", "test"):
            if split not in record:
                continue
            m = record[split]
            current = (m["background_index"], tuple((r["class_id"], r["class_name"]) for r in m["per_class"]))
            if mapping is not None and current != mapping:
                raise ValueError("Class mappings differ; summarize a single dataset with consistent labels")
            mapping = current
    if len(datasets) > 1:
        raise ValueError("Different dataset roots detected; summarize one dataset at a time")
    if len(sizes) > 1:
        raise ValueError("Evaluation sizes differ; use matching input resolutions for a comparison")
    overall, all_classes = [], []
    meta_fields = ("name", "arch", "encoder", "seed", "metric_protocol", "best_epoch")
    for record in records:
        for row in result_rows(record):
            row.update(source_run=record["source_run"], metrics_file=record["metrics_file"])
            overall.append(row)
        for split in ("val", "test"):
            if split not in record:
                continue
            meta = {key: record.get(key) for key in meta_fields}
            meta["metric_protocol"] = record[split]["metric_protocol"]
            for row in record[split]["per_class"]:
                all_classes.append(dict(row, **meta, split=split,
                    source_run=record["source_run"], metrics_file=record["metrics_file"]))
    overall.extend(dict(failure) for failure in failures)
    _write_csv(output / "results.csv", (*RESULT_FIELDS, "source_run", "metrics_file"), overall)
    _write_csv(output / "all_class_metrics.csv",
        (*meta_fields, *CLASS_FIELDS, "source_run", "metrics_file"), all_classes)
    class_columns = [(cid, f"iou_{cid}_{name}") for cid, name in mapping[1]] if mapping else []
    wide_fields = (*meta_fields, "split", "miou", "foreground_miou", "dice", "macc", "pixacc",
                   *[column for _, column in class_columns], "parameters", "params_m", "latency_ms", "fps",
                   "source_run", "metrics_file")
    for split in ("val", "test"):
        wide = []
        for record in records:
            if split not in record:
                continue
            row = next(row for row in result_rows(record) if row["split"] == split)
            per_class = {r["class_id"]: r["iou"] for r in record[split]["per_class"]}
            row.update({column: per_class[cid] for cid, column in class_columns})
            row.update(source_run=record["source_run"], metrics_file=record["metrics_file"])
            wide.append(row)
        _write_csv(output / f"{split}_class_iou.csv", wide_fields, wide)
    manifest = dict(models=len(records), failures=list(failures),
        evaluation_sizes=sorted(sizes),
        metric_protocols=sorted({r[s]["metric_protocol"] for r in records for s in ("val", "test") if s in r}),
        class_mapping=[dict(class_id=cid, class_name=name) for cid, name in mapping[1]] if mapping else [],
        units="scores 0-1; HD95/ASSD pixels; parameters count; params_m millions; latency_ms milliseconds",
        source_files=[r["metrics_file"] for r in records],
        note="Legacy scores are retained with their original protocol; missing boundary scores stay blank. No averaging across seeds or models.")
    (output / "summary_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
