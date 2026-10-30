import json

import pytest
import torch
from PIL import Image

from fenformer.evaluate import board_metrics, evaluate, wilson_interval
from fenformer.model import FENFormer


def test_wilson_interval_known_value():
    """8/10 successes: the 95% Wilson interval is about [0.490, 0.943]."""
    low, high = wilson_interval(8, 10)
    assert low == pytest.approx(0.4902, abs=1e-3) and high == pytest.approx(0.9433, abs=1e-3)


def test_board_metrics_counts_exact_boards_and_squares():
    """One perfect board and one board with two wrong squares."""
    targets = torch.zeros(2, 64, dtype=torch.long)
    predictions = targets.clone()
    predictions[1, :2] = 3
    metrics = board_metrics(predictions, targets)
    assert metrics["board_accuracy"] == 0.5
    assert metrics["square_accuracy"] == pytest.approx(1 - 2 / 128)
    assert metrics["mean_wrong_squares"] == 1.0


def test_evaluate_benchmark_directory(tmp_path):
    """A benchmark directory yields one record per labeled image (framing disabled)."""
    (tmp_path / "images").mkdir()
    labels = {}
    for i, fen in enumerate(["8/8/8/4k3/8/8/8/K7", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"]):
        Image.new("RGB", (900, 600), (30 * i, 90, 60)).save(tmp_path / "images" / f"{i}.jpg")
        labels[f"images/{i}.jpg"] = fen
    (tmp_path / "labels.json").write_text(json.dumps(labels))
    metrics, records = evaluate(FENFormer().eval(), tmp_path, None, batch_size=2, workers=0)
    assert [r["image"] for r in records] == ["images/0.jpg", "images/1.jpg"]
    assert metrics["constrained"]["images"] == 2
    assert 0 <= metrics["argmax"]["square_accuracy"] <= 1
