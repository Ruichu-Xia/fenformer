"""Predict the FEN piece placement of chess board images.

python -m fenformer.predict board.jpg

Released weights are downloaded on first use; pass `--weights` for a training checkpoint.
"""

import argparse
import hashlib
from pathlib import Path

import torch
from safetensors.torch import load_file

from .board import labels_to_fen
from .decode import decode
from .localize import download_u2netp, frame_board, load_image
from .model import FENFormer

WEIGHTS_URL = (
    "https://huggingface.co/Eric-Xia/fenformer/resolve/"
    "4cdd976f92f46a037fbeca5eacdfefcdf1bf9aae/fenformer.safetensors"
)
WEIGHTS_SHA256 = "12a6de0931fe97b74a7c673499d4a97c8b749ba12a3b95c89f2221bc76c746c0"


def download_weights(path: Path) -> Path:
    """Fetch the released fenformer weights once and verify their digest."""
    import urllib.request

    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(WEIGHTS_URL, path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != WEIGHTS_SHA256:
        raise ValueError(f"Weights digest differs from the pinned release: {path}")
    return path


def load_model(weights: Path, device: str) -> FENFormer:
    """Load released `.safetensors` weights or a training checkpoint (`best.pt`)."""
    if Path(weights).suffix == ".safetensors":
        state = load_file(weights)
    else:
        state = torch.load(weights, map_location="cpu", weights_only=True)["model"]
    model = FENFormer()
    model.load_state_dict(state)
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
    parser.add_argument("--weights", type=Path, help="default: released weights")
    parser.add_argument("--u2netp", type=Path, default=Path("weights/u2netp.onnx"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    weights = args.weights or download_weights(Path("weights/fenformer.safetensors"))
    model = load_model(weights, args.device)
    u2netp = download_u2netp(args.u2netp)
    for path in args.images:
        pixels = frame_board(load_image(path), weights=u2netp)
        labels = decode(logits(model, pixels[None]))[0]
        print(f"{labels_to_fen(labels.tolist())}\t{path}")


if __name__ == "__main__":
    main()
