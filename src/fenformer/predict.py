"""Predict the FEN piece placement of chess board images.

python -m fenformer.predict board.jpg --weights weights/fenformer.pt
"""

import argparse
from pathlib import Path

import torch

from .board import labels_to_fen
from .decode import decode
from .localize import download_u2netp, frame_board, load_image
from .model import FENFormer


def load_model(weights: Path, device: str) -> FENFormer:
    """Load a trained model from `best.pt` (or a bare state dict) for inference."""
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model = FENFormer()
    model.load_state_dict(state.get("model", state))
    return model.to(device).eval()


@torch.inference_mode()
def logits(model: FENFormer, pixels: torch.Tensor) -> torch.Tensor:
    """Score framed uint8 images [B, 3, 644, 644]; return float32 logits [B, 64, 13]."""
    device = next(model.parameters()).device
    pixels = pixels.to(device).float().div_(255)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        return model(pixels).float()


def main():
    """Print one FEN per input image."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--weights", type=Path, default=Path("weights/fenformer.pt"))
    parser.add_argument("--u2netp", type=Path, default=Path("weights/u2netp.onnx"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    model = load_model(args.weights, args.device)
    u2netp = download_u2netp(args.u2netp)
    for path in args.images:
        pixels = frame_board(load_image(path), weights=u2netp)
        labels = decode(logits(model, pixels[None]))[0]
        print(f"{labels_to_fen(labels.tolist())}\t{path}")


if __name__ == "__main__":
    main()
