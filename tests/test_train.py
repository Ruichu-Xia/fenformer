import json

import pytest
import torch
from PIL import Image

from fenformer.board import fen_to_labels
from fenformer.model import FENFormer
from fenformer.train import Config, train

FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
    "8/8/8/4k3/8/8/8/K7",
    "r3k2r/8/8/8/8/8/8/R3K2R",
    "8/5k2/8/8/2Q5/8/8/6K1",
]


@pytest.fixture
def mixture(tmp_path):
    """A four-image mixture with two sources, written in the on-disk training format."""
    (tmp_path / "images").mkdir()
    samples = []
    for i, fen in enumerate(FENS):
        sample_id = f"img{i}"
        Image.new("RGB", (644, 644), (40 * i, 80, 120)).save(
            tmp_path / "images" / f"{sample_id}.png"
        )
        samples.append({"id": sample_id, "fen": fen, "source": "a" if i % 2 else "b", "sha256": ""})
    split = {"samples": samples, "train": ["img0", "img1"], "val": ["img2", "img3"]}
    (tmp_path / "split.json").write_text(json.dumps(split))
    (tmp_path / "boxes.json").write_text(json.dumps({s["id"]: [0, 0, 644, 644] for s in samples}))
    return tmp_path


def test_model_output_shape():
    """Logits cover 64 squares x 13 classes for each image."""
    model = FENFormer().eval()
    with torch.no_grad():
        assert model(torch.rand(1, 3, 644, 644)).shape == (1, 64, 13)


class Interrupt(Exception):
    """Simulates a preempted container."""


def test_interrupted_run_resumes_at_next_epoch(mixture, tmp_path):
    """A run stopped after epoch 1 resumes from last.pt and finishes epoch 2."""
    config = Config(
        data=str(mixture),
        output=str(tmp_path / "run"),
        epochs=2,
        batch_size=2,
        warmup_epochs=1,
        device="cpu",
        precision="fp32",
        workers=0,
        pretrained=False,
    )

    def stop():
        """Abort after the first epoch's checkpoint is written."""
        raise Interrupt

    with pytest.raises(Interrupt):
        train(config, on_epoch_end=stop)
    history = train(config, resume=True)
    assert [h["epoch"] for h in history] == [1, 2]
    assert set(history[-1]["val"]["per_source"]) == {"a", "b"}
    assert history[-1]["val"]["images"] == 2

    best = torch.load(tmp_path / "run" / "best.pt", weights_only=False)
    assert set(best["model"]) == set(FENFormer().state_dict())


def test_resume_rejects_changed_config(mixture, tmp_path):
    """The LR schedule depends on the full config, so resuming with a different one fails."""
    config = Config(
        data=str(mixture),
        output=str(tmp_path / "run"),
        epochs=1,
        batch_size=2,
        warmup_epochs=0,
        device="cpu",
        precision="fp32",
        workers=0,
        pretrained=False,
    )
    train(config)
    with pytest.raises(ValueError):
        train(Config(**{**config.__dict__, "epochs": 3}), resume=True)


def test_targets_follow_fen_order(mixture):
    """Dataset targets are the FEN labels of each sample, in a8..h1 order."""
    from fenformer.data import mixture_datasets

    train_set, val_set = mixture_datasets(mixture)
    pixels, target, index = val_set[0]
    assert pixels.shape == (3, 644, 644)
    assert target.tolist() == fen_to_labels(val_set.samples[index]["fen"])
