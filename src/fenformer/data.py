"""Datasets on disk.

A training mixture is a directory with
    images/<id>.png   644 x 644 letterboxed, uncropped renders or photos
    split.json        {"samples": [{"id", "fen", "source", "sha256"}], "train": [...], "val": [...]}
    boxes.json        {"<id>": [left, top, right, bottom]}, precomputed U2NETP board boxes

A benchmark is a directory with images in any layout and
    labels.json       {"<image path relative to the directory>": "<fen>"}
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .augment import augment
from .board import fen_to_labels
from .localize import frame_board, load_image


def load_split(root: Path) -> dict:
    """Load and validate a mixture's split: disjoint, complete, uniquely identified."""
    split = json.loads((Path(root) / "split.json").read_text())
    ids = [s["id"] for s in split["samples"]]
    train, val = set(split["train"]), set(split["val"])
    if len(set(ids)) != len(ids) or train & val or train | val != set(ids):
        raise ValueError("Split must partition the samples into disjoint train and val sets")
    return split


class MixtureImages(Dataset):
    """Yield (uint8 [3, 644, 644], int64 [64] labels, sample index) from a mixture."""

    def __init__(self, root: Path, samples: list[dict], boxes: dict, *, train: bool):
        """Keep sample records in memory; images are decoded lazily per item."""
        self.root = Path(root)
        self.samples = samples
        self.boxes = boxes
        self.train = train
        self.targets = torch.tensor([fen_to_labels(s["fen"]) for s in samples])

    def __len__(self):
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(self, index):
        """Frame the board with its cached box, then augment if training."""
        sample = self.samples[index]
        image = load_image(self.root / "images" / f"{sample['id']}.png")
        pixels = frame_board(image, box=self.boxes[sample["id"]])
        if self.train:
            pixels = augment(pixels)
        return pixels, self.targets[index], index


def mixture_datasets(root: Path, limit: int | None = None) -> tuple[MixtureImages, MixtureImages]:
    """Build the train and val datasets of a mixture, optionally truncated for smoke tests."""
    split = load_split(root)
    boxes = json.loads((Path(root) / "boxes.json").read_text())
    by_id = {s["id"]: s for s in split["samples"]}
    train = [by_id[i] for i in split["train"]][:limit]
    val = [by_id[i] for i in split["val"]][:limit]
    return (
        MixtureImages(root, train, boxes, train=True),
        MixtureImages(root, val, boxes, train=False),
    )


def load_benchmark(root: Path) -> list[dict]:
    """Return [{"path", "fen"}] for a benchmark directory, sorted for determinism."""
    root = Path(root)
    labels = json.loads((root / "labels.json").read_text())
    samples = []
    for name, fen in sorted(labels.items()):
        fen_to_labels(fen)
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
        samples.append({"path": root / name, "fen": fen})
    return samples
