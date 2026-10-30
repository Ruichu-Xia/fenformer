"""Score a trained model on benchmark directories with the full inference pipeline.

    python -m fenformer.evaluate --output results/fenformer data/benchmarks/*

Each image goes through live U2NETP framing, the model, and constrained decoding, exactly
as in `predict`. Metrics are reported for both plain argmax and constrained decoding.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .board import fen_to_labels, labels_to_fen
from .data import load_benchmark
from .decode import decode
from .localize import SIZE, download_u2netp, frame_board, load_image
from .predict import download_weights, load_model, logits


class FramedImages(Dataset):
    """Frame benchmark images with live U2NETP inside loader workers.

    With `u2netp=None` the whole letterboxed image is used instead (framing ablation).
    """

    def __init__(self, samples: list[dict], u2netp: Path | None):
        """Store (path, fen) records and the localizer weights."""
        self.samples = samples
        self.u2netp = u2netp

    def __len__(self):
        """Return the number of images."""
        return len(self.samples)

    def __getitem__(self, index):
        """Return framed uint8 pixels and the target labels."""
        sample = self.samples[index]
        image = load_image(sample["path"])
        if self.u2netp is None:
            pixels = frame_board(image, box=(0, 0, SIZE, SIZE))
        else:
            pixels = frame_board(image, weights=self.u2netp)
        return pixels, torch.tensor(fen_to_labels(sample["fen"]))


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    p = successes / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return center - half, center + half


def board_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict:
    """Exact-board accuracy (with 95% CI), square accuracy and mean wrong squares."""
    wrong = (predictions != targets).sum(1)
    exact = int((wrong == 0).sum())
    n = len(targets)
    return {
        "images": n,
        "board_accuracy": exact / n,
        "board_accuracy_ci95": wilson_interval(exact, n),
        "square_accuracy": 1 - wrong.sum().item() / (64 * n),
        "mean_wrong_squares": wrong.float().mean().item(),
    }


def evaluate(
    model, benchmark: Path, u2netp: Path | None, *, batch_size: int = 16, workers: int = 4
):
    """Return metrics and per-image predictions for one benchmark directory."""
    samples = load_benchmark(benchmark)
    loader = DataLoader(FramedImages(samples, u2netp), batch_size=batch_size, num_workers=workers)
    scores, targets = [], []
    started = time.monotonic()
    for pixels, labels in loader:
        scores.append(logits(model, pixels).cpu())
        targets.append(labels)
    seconds = time.monotonic() - started
    scores, targets = torch.cat(scores), torch.cat(targets)
    argmax, constrained = scores.argmax(-1), decode(scores)
    records = [
        {
            "image": str(sample["path"].relative_to(benchmark)),
            "target": sample["fen"],
            "prediction": labels_to_fen(constrained[i].tolist()),
            "wrong_squares": int((constrained[i] != targets[i]).sum()),
        }
        for i, sample in enumerate(samples)
    ]
    metrics = {
        "argmax": board_metrics(argmax, targets),
        "constrained": board_metrics(constrained, targets),
        "images_per_second": len(samples) / seconds,  # decode, frame and score, end to end
    }
    return metrics, records


def main():
    """Evaluate every benchmark and write `<output>/<name>.json` plus a summary."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("benchmarks", nargs="+", type=Path)
    parser.add_argument("--weights", type=Path, help="default: released weights")
    parser.add_argument("--u2netp", type=Path, default=Path("weights/u2netp.onnx"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    weights = args.weights or download_weights(Path("weights/fenformer.safetensors"))
    model = load_model(weights, args.device)
    u2netp = download_u2netp(args.u2netp)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for benchmark in args.benchmarks:
        metrics, records = evaluate(model, benchmark, u2netp, workers=args.workers)
        summary[benchmark.name] = metrics
        result = {"metrics": metrics, "predictions": records}
        (args.output / f"{benchmark.name}.json").write_text(json.dumps(result, indent=2) + "\n")
        m = metrics["constrained"]
        print(
            f"{benchmark.name}: boards {m['board_accuracy']:.2%} "
            f"squares {m['square_accuracy']:.3%} ({m['images']} images)",
            flush=True,
        )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
