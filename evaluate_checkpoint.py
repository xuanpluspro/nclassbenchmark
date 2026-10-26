"""Evaluate one trained best.pth without creating an optimizer or training.

Only load checkpoints produced by this repository (trusted pickle files).
Model configuration is read from the checkpoint; no pretrained download occurs.
"""
import argparse
import hashlib
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True, help="New or empty result directory")
    parser.add_argument("--data-root", help="Override saved dataset root, e.g. after moving the dataset")
    parser.add_argument("--splits", nargs="+", choices=("val", "test"), default=["test"])
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size; independent of training")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--benchmark", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--benchmark-warmup", type=int, default=50)
    parser.add_argument("--benchmark-iterations", type=int, default=200)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.workers < 0 or args.benchmark_warmup < 0 or args.benchmark_iterations < 1:
        parser.error("Invalid batch size, workers or benchmark repetitions")
    return args


def run(args):
    import torch
    import segmentation_models_pytorch as smp
    from torch.utils.data import DataLoader
    from train import VOC, evaluate
    from benchmark import benchmark_model
    from segmentation_metrics import write_run_results

    weights = Path(args.weights).resolve()
    output = Path(args.output).resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Use a new or empty --output: {output}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Use auto, cpu, cuda, or cuda:N")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    # Original checkpoints contain NumPy RNG/optimizer objects: weights_only=False
    # is necessary for these trusted, locally trained files. Discard training state.
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "config" not in checkpoint or "model" not in checkpoint:
        raise ValueError("Expected this repository's checkpoint with config and model")
    config = dict(checkpoint["config"])
    state = checkpoint["model"]
    best_epoch = int(checkpoint["epoch"]) + 1
    del checkpoint
    for key in ("arch", "encoder", "num_classes", "size", "class_names"):
        if key not in config:
            raise ValueError(f"Checkpoint config is missing {key}")
    data_root = args.data_root or config.get("data_root")
    if not data_root:
        raise ValueError("Supply --data-root: checkpoint has no dataset root")
    data_root = Path(data_root).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset not found: {data_root}; supply --data-root after moving it")
    model = smp.create_model(config["arch"], encoder_name=config["encoder"],
        encoder_weights=None, in_channels=3, classes=config["num_classes"])
    model.load_state_dict(state, strict=True)
    del state
    model.to(device).eval()
    parameters = sum(p.numel() for p in model.parameters())
    result = dict(arch=config["arch"], encoder=config["encoder"], seed=config.get("seed"),
        best_epoch=best_epoch, checkpoint=str(weights), parameters=parameters, params_m=parameters / 1e6,
        evaluation_type="trained checkpoint only; no training or optimizer updates",
        evaluation_dataset=str(data_root), source_config=config,
        evaluation_config=vars(args), evaluation_splits={}, benchmark=None)
    print(f"Evaluating {weights}\nModel={config['arch']}/{config['encoder']}; "
          f"classes={config['num_classes']}; input={config['size']}; device={device}", flush=True)
    with torch.inference_mode():
        for split in dict.fromkeys(args.splits):
            saved_split = config.get(f"{split}_split", split)
            if str(saved_split).lower() == "none":
                raise ValueError(f"Checkpoint config disables {split}; choose another --splits value")
            dataset = VOC(data_root, saved_split, config["size"], config["num_classes"],
                          augment=False, ignore_index=config.get("ignore_index", 255))
            loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                                shuffle=False, pin_memory=device.type == "cuda")
            result["evaluation_splits"][split] = dict(manifest_name=saved_split, samples=len(dataset),
                ids=dataset.ids, ids_sha256=hashlib.sha256("\n".join(dataset.ids).encode()).hexdigest())
            result[split] = evaluate(model, loader, device, config["num_classes"], config["class_names"],
                config.get("ignore_index", 255), desc=f"{weights.parent.name} {split}",
                background_index=config.get("background_index", 0))
            scores = result[split]
            print(f"{split}: mIoU={scores['miou']:.4f}; Dice={scores['dice']}; "
                  f"samples={len(dataset)}", flush=True)
    # Persist accuracy first: profiling failures must not erase evaluated metrics.
    write_run_results(result, output)
    if args.benchmark:
        try:
            result["benchmark"] = benchmark_model(model, config["size"], args.benchmark_warmup, args.benchmark_iterations)
        except Exception as exc:
            result["benchmark"] = dict(parameters=parameters, params_m=parameters / 1e6,
                gflops=None, fps=None, latency_ms=None, benchmark_error=f"{type(exc).__name__}: {exc}")
        write_run_results(result, output)
    print(f"Saved metrics only: {output}", flush=True)
    return result


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
