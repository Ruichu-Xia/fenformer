"""Score the released ChessQueries model on fenformer benchmarks, unmodified.

ChessQueries (github.com/JSeytre/chessqueries, weights: huggingface.co/joelseytre/chessqueries)
is a ViT-L/14 square-query model trained on ChessReD, ChessCog and SLCC. This script uses
the author's loader and preprocessing (whole-image resize to 644 px) and reports the same
metrics as `fenformer.evaluate`, for its stock argmax and with our constrained decoding.

    python scripts/baseline_chessqueries.py --source chessqueries/ --weights joint.safetensors \
        --output results/chessqueries data/benchmarks/samryan-500 data/benchmarks/chesscog-transfer

Run it in an environment with the ChessQueries dependencies installed.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fenformer.board import fen_to_labels, labels_to_fen  # noqa: E402
from fenformer.data import load_benchmark  # noqa: E402
from fenformer.decode import decode  # noqa: E402
from fenformer.evaluate import board_metrics  # noqa: E402


def main():
    """Evaluate each benchmark directory and write `<output>/<name>.json`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("benchmarks", nargs="+", type=Path)
    parser.add_argument("--source", type=Path, required=True, help="ChessQueries checkout")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source))
    from chessqueries.models.predictor import Predictor

    torch.set_num_threads(args.threads)
    predictor = Predictor.from_checkpoint(args.weights, resolution=644, device="cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    for benchmark in args.benchmarks:
        samples = load_benchmark(benchmark)
        scores = []
        with torch.inference_mode():
            for sample in samples:
                pixels = predictor._load(sample["path"]).unsqueeze(0)
                scores.append(predictor.model(pixels).float())
        scores = torch.cat(scores)
        targets = torch.tensor([fen_to_labels(s["fen"]) for s in samples])
        constrained = decode(scores)
        metrics = {
            "argmax": board_metrics(scores.argmax(-1), targets),
            "constrained": board_metrics(constrained, targets),
        }
        records = [
            {
                "image": str(s["path"].relative_to(benchmark)),
                "target": s["fen"],
                "prediction": labels_to_fen(scores[i].argmax(-1).tolist()),
                "wrong_squares": int((scores[i].argmax(-1) != targets[i]).sum()),
            }
            for i, s in enumerate(samples)
        ]
        result = {"metrics": metrics, "predictions": records}
        (args.output / f"{benchmark.name}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(benchmark.name, json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
