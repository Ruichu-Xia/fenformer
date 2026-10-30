"""Board-centered framing shared by training and inference.

Pipeline: letterbox to 644 px -> U2NETP foreground box (+5% margin) -> crop -> letterbox
to 644 px. Training reads precomputed boxes; inference computes them with the same code.
"""

import hashlib
import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

SIZE = 644
U2NETP_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
U2NETP_SHA256 = "309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8"
MASK_SIZE = 320
MARGIN = 0.05


def load_image(path: str | Path) -> Image.Image:
    """Open an image as RGB, applying any EXIF orientation (common in phone photos)."""
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def letterbox(image: Image.Image, size: int = SIZE) -> Image.Image:
    """Resize to fit a black size x size canvas, preserving aspect ratio."""
    image = image.convert("RGB")
    if image.size == (size, size):
        return image
    fitted = ImageOps.contain(image, (size, size), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size))
    canvas.paste(fitted, ((size - fitted.width) // 2, (size - fitted.height) // 2))
    return canvas


def to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert an RGB image to uint8 [3, H, W]."""
    return torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1)


def download_u2netp(path: Path) -> Path:
    """Fetch the public U2NETP ONNX release once and verify its digest."""
    import urllib.request

    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(U2NETP_URL, path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != U2NETP_SHA256:
        raise ValueError(f"U2NETP digest differs from the pinned release: {path}")
    return path


@lru_cache(maxsize=1)
def _session(path: str, pid: int, cuda: bool):
    """Create one ONNX session per process; never reuse a parent's session after fork."""
    import onnxruntime as ort

    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != U2NETP_SHA256:
        raise ValueError("U2NETP weights differ from the pinned release")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    providers = ["CUDAExecutionProvider"] if cuda else []
    return ort.InferenceSession(path, options, providers=[*providers, "CPUExecutionProvider"])


def board_box(image: Image.Image, weights: str | Path, cuda: bool = False) -> tuple[int, ...]:
    """Return the (left, top, right, bottom) foreground box of a 644 px letterboxed image."""
    session = _session(str(Path(weights).resolve()), os.getpid(), cuda)
    rgb = np.array(image.resize((MASK_SIZE, MASK_SIZE), Image.Resampling.LANCZOS))
    values = rgb / max(rgb.max(), 1e-6)
    values = (values - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
    batch = values.transpose(2, 0, 1)[None].astype(np.float32)
    mask = session.run(None, {session.get_inputs()[0].name: batch})[0][0, 0]
    if mask.shape != (MASK_SIZE, MASK_SIZE) or not np.isfinite(mask).all():
        raise ValueError("U2NETP returned an invalid mask")
    low, high = float(mask.min()), float(mask.max())
    if high <= low:  # no foreground signal: keep the whole frame
        return (0, 0, image.width, image.height)
    mask = Image.fromarray(((mask - low) / (high - low) * 255).astype(np.uint8))
    mask = mask.resize(image.size, Image.Resampling.LANCZOS)
    bounds = mask.point(lambda v: 255 if v >= 128 else 0).getbbox()
    if bounds is None:
        return (0, 0, image.width, image.height)
    left, top, right, bottom = bounds
    margin = MARGIN * max(right - left, bottom - top)
    return (
        max(0, math.floor(left - margin)),
        max(0, math.floor(top - margin)),
        min(image.width, math.ceil(right + margin)),
        min(image.height, math.ceil(bottom + margin)),
    )


def frame_board(
    image: Image.Image, *, weights: str | Path | None = None, box=None, cuda: bool = False
) -> torch.Tensor:
    """Return the model input, uint8 [3, 644, 644], from a cached box or live U2NETP."""
    if (weights is None) == (box is None):
        raise ValueError("Pass exactly one of weights (live) or box (cached)")
    image = letterbox(image)
    if box is None:
        box = board_box(image, weights, cuda)
    left, top, right, bottom = box
    if not 0 <= left < right <= SIZE or not 0 <= top < bottom <= SIZE:
        raise ValueError(f"Invalid board box {box}")
    return to_tensor(letterbox(image.crop(tuple(box))))
