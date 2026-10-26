"""Sequentially evaluate existing best.pth checkpoints; never train models."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

from batch_train import run_logged
from result_tables import read_record, export_tables


def discover_checkpoints(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths = sorted(root.rglob("best.pth"))
    if not paths:
        raise FileNotFoundError(f"No best.pth found under {root}")
    return paths


def validate_known_configs(checkpoints, data_root_override=None):
    """Reject obvious mixed datasets/settings before launching GPU workers."""
    signatures, roots = set(), set()
    for weights in checkpoints:
        path = weights.parent / "config.json"
        if not path.is_file():
            continue
        config = json.loads(path.read_text(encoding="utf-8-sig"))
        signatures.add((config.get("num_classes"), tuple(config.get("class_names") or []),
                        config.get("background_index", 0), config.get("ignore_index", 255), config.get("size")))
        if config.get("data_root"):
            roots.add(str(Path(config["data_root"]).resolve()))
    if len(signatures) > 1 or (not data_root_override and len(roots) > 1):
        raise ValueError("Mixed datasets/class mappings/input sizes found. Set --runs-root to ONE dataset/experiment directory")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True, help="Directory holding the trained model folders")
    parser.add_argument("--output", help="New directory; defaults to RUNS_ROOT/evaluation_DATE_TIME")
    parser.add_argument("--data-root", help="Target data root; otherwise use each checkpoint's saved path")
    parser.add_argument("--splits", nargs="+", choices=("val", "test"), default=["test"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--benchmark", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--benchmark-warmup", type=int, default=50)
    parser.add_argument("--benchmark-iterations", type=int, default=200)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.workers < 0 or args.benchmark_warmup < 0 or args.benchmark_iterations < 1:
        parser.error("Invalid batch size, workers or benchmark repetitions")
    root = Path(args.runs_root).resolve()
    checkpoints = discover_checkpoints(root)
    validate_known_configs(checkpoints, args.data_root)
    output = Path(args.output).resolve() if args.output else root / f"evaluation_{datetime.now():%Y%m%d_%H%M%S_%f}"
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Use a new or empty --output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    log_dir = output / "logs"; log_dir.mkdir()
    print(f"Found {len(checkpoints)} checkpoints. Evaluation output: {output}", flush=True)
    records, failures, statuses = [], [], []
    for index, weights in enumerate(checkpoints, 1):
        relative = weights.parent.relative_to(root)
        destination = output / "models" / (relative if relative.parts else Path("model"))
        command = [sys.executable, "-u", str(Path(__file__).with_name("evaluate_checkpoint.py")),
            "--weights", str(weights), "--output", str(destination), "--splits", *args.splits,
            "--batch-size", str(args.batch_size), "--workers", str(args.workers), "--device", args.device,
            "--benchmark" if args.benchmark else "--no-benchmark",
            "--benchmark-warmup", str(args.benchmark_warmup), "--benchmark-iterations", str(args.benchmark_iterations)]
        if args.data_root:
            command += ["--data-root", args.data_root]
        print(f"\nEvaluate {index}/{len(checkpoints)}: {weights}", flush=True)
        status = run_logged(command, log_dir / f"{index:03d}.log")
        entry = dict(checkpoint=str(weights), source_run=str(relative), status=status,
                     output=str(destination), log=str(log_dir / f"{index:03d}.log"))
        if status == 0:
            try:
                records.append(read_record(destination / "metrics.json", name=weights.parent.name,
                                           source_run=str(relative)))
            except Exception as exc:
                entry.update(status=1, error=f"{type(exc).__name__}: {exc}")
        if entry["status"]:
            failures.append(dict(name=weights.parent.name, source_run=str(relative), status=entry["status"]))
        statuses.append(entry)
        (output / "evaluation_status.json").write_text(json.dumps(statuses, indent=2, ensure_ascii=False), encoding="utf-8")
        export_tables(records, output, failures)
    print(f"Finished: {len(records)} successful, {len(failures)} failed; tables: {output}", flush=True)
    if failures:
        raise SystemExit(1)
    return output


if __name__ == "__main__":
    main()
