import os
from pathlib import Path

import pytest
import torch
from PIL import Image

from fenformer.localize import SIZE, frame_board, letterbox

U2NETP = Path(os.environ.get("FENFORMER_U2NETP", "weights/u2netp.onnx"))


def test_letterbox_preserves_aspect_and_centers():
    """A wide image fills the width and is padded with black above and below."""
    image = Image.new("RGB", (1288, 644), (255, 0, 0))
    boxed = letterbox(image)
    assert boxed.size == (SIZE, SIZE)
    assert boxed.getpixel((SIZE // 2, 0)) == (0, 0, 0)
    assert boxed.getpixel((SIZE // 2, SIZE // 2)) == (255, 0, 0)
    assert boxed.getbbox() == (0, SIZE // 4, SIZE, 3 * SIZE // 4)


def test_cached_box_crops_and_rescales():
    """Cropping a box around a square region fills the whole model input."""
    image = Image.new("RGB", (SIZE, SIZE))
    image.paste((0, 255, 0), (100, 100, 300, 300))
    pixels = frame_board(image, box=(100, 100, 300, 300))
    assert pixels.shape == (3, SIZE, SIZE) and pixels.dtype == torch.uint8
    assert (pixels[1] == 255).all() and (pixels[0] == 0).all()


@pytest.mark.parametrize("box", [(0, 0, 0, 10), (10, 10, 5, 20), (0, 0, SIZE + 1, SIZE)])
def test_invalid_box_raises(box):
    """Degenerate or out-of-frame boxes are rejected rather than silently padded."""
    with pytest.raises(ValueError):
        frame_board(Image.new("RGB", (SIZE, SIZE)), box=box)


def test_requires_exactly_one_framing_source():
    """Callers must choose live localization or a cached box."""
    with pytest.raises(ValueError):
        frame_board(Image.new("RGB", (SIZE, SIZE)))


@pytest.mark.skipif(not U2NETP.exists(), reason="U2NETP weights not downloaded")
def test_live_localization_finds_salient_object():
    """U2NETP frames a bright object on a dark background, with a small margin."""
    image = Image.new("RGB", (SIZE, SIZE), (20, 20, 20))
    image.paste((230, 230, 230), (200, 220, 420, 440))
    from fenformer.localize import board_box

    left, top, right, bottom = board_box(image, U2NETP)
    assert left <= 200 <= left + 40 and right - 40 <= 420 <= right
    assert top <= 220 <= top + 40 and bottom - 40 <= 440 <= bottom
