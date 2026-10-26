"""Collect per-model metrics.json into overall and per-class cross-model tables.

This command does not open checkpoints or run models. Works with legacy nclass
results and v2 results; missing legacy boundary metrics remain blank.
"""
import argparse
from pathlib import Path

from result_tables import read_record, export_tables


def summarize(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths = sorted(root.rglob("metrics.json"))
    if not paths:
        raise FileNotFoundError(f"No metrics.json found under {root}")
    records = [read_record(path, source_run=str(path.parent.relative_to(root))) for path in paths]
    export_tables(records, output)
    print(f"Summarized {len(records)} model runs to {output}", flush=True)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True, help="One dataset/experiment directory")
    parser.add_argument("--output", help="Defaults to RUNS_ROOT/summary")
    args = parser.parse_args(argv)
    summarize(args.runs_root, args.output or Path(args.runs_root) / "summary")


if __name__ == "__main__":
    main()
