"""Mild label-preserving training augmentation, applied after board framing.

Scene-level diversity (cameras, materials, lighting, piece designs) comes from the data
itself; these operations only simulate small framing, exposure and sensor differences.
Every operation draws from Python's `random`, which the loader seeds per worker.
"""

import io
import random

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from .localize import to_tensor


def augment(pixels: torch.Tensor) -> torch.Tensor:
    """Return an augmented copy of uint8 [3, H, W]; shape and labels are unchanged."""
    image = Image.fromarray(pixels.permute(1, 2, 0).numpy())
    if random.random() < 0.5:
        image = jitter_geometry(image, degrees=8, min_scale=0.95, max_translate=0.03)
    if random.random() < 0.5:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
    if random.random() < 0.5:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.1))
    if random.random() < 0.5:
        gamma = random.uniform(0.9, 1.1)
        image = image.point([round(255 * (i / 255) ** gamma) for i in range(256)] * 3)
    if random.random() < 0.5:
        image = ImageEnhance.Color(image).enhance(random.uniform(0.9, 1.1))
    if random.random() < 0.15:
        image = image.filter(ImageFilter.GaussianBlur(random.uniform(0, 0.5)))
    if random.random() < 0.15:
        image = downsample(image, random.uniform(0.75, 1))
    if random.random() < 0.15:
        image = add_noise(image, sigma=random.uniform(0, 2))
    if random.random() < 0.15:
        image = jpeg(image, quality=random.randint(85, 100))
    return to_tensor(image)


def jitter_geometry(image, *, degrees, min_scale, max_translate):
    """Rotate, shrink and shift inside black padding without clipping any content."""
    width, height = image.size
    rotated = image.rotate(random.uniform(-degrees, degrees), Image.Resampling.BILINEAR, True)
    scale = random.uniform(min_scale, 1)
    fitted = ImageOps.contain(
        rotated, (max(1, int(width * scale)), max(1, int(height * scale))), Image.BILINEAR
    )
    offsets = []
    for extent, content in ((width, fitted.width), (height, fitted.height)):
        center = (extent - content) // 2
        shift = min(center, round(max_translate * extent))
        offsets.append(center + random.randint(-shift, shift))
    canvas = Image.new("RGB", (width, height))
    canvas.paste(fitted, tuple(offsets))
    return canvas


def downsample(image, factor):
    """Lose detail by resizing down and back up."""
    small = (max(1, round(image.width * factor)), max(1, round(image.height * factor)))
    return image.resize(small, Image.BILINEAR).resize(image.size, Image.BILINEAR)


def add_noise(image, sigma):
    """Add zero-mean Gaussian pixel noise."""
    rng = np.random.default_rng(random.getrandbits(64))
    values = np.asarray(image, dtype=np.float32) + rng.normal(
        0, sigma, (image.height, image.width, 3)
    )
    return Image.fromarray(np.rint(values).clip(0, 255).astype(np.uint8))


def jpeg(image, quality):
    """Round-trip through JPEG compression."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")
