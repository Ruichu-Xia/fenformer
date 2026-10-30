"""Train FENFormer on a mixture directory; resumable and tracked with W&B when enabled."""

import argparse
import json
import logging
import math
import os
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .board import CLASSES
from .data import mixture_datasets
from .model import FENFormer

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Every setting that defines a run; saved with each checkpoint."""

    data: str = "data/mixture"
    output: str = "outputs/run"
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-4  # head
    encoder_lr: float = 1e-5  # pretrained DINOv2
    min_lr_ratio: float = 1e-3  # cosine floor as a fraction of each group's peak
    warmup_epochs: int = 2
    weight_decay: float = 0.05
    patience: int = 8  # epochs without validation-loss improvement; 0 disables
    min_delta: float = 1e-4
    seed: int = 42
    device: str = "cuda"
    precision: str = "bf16"
    workers: int = 8
    limit: int | None = None  # truncate train and val for smoke tests
    pretrained: bool = True
    wandb_project: str = ""  # empty disables tracking


def square_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-image mean of 13-way cross-entropy over the 64 squares, shape [B]."""
    return (
        F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none")
        .view(targets.shape)
        .mean(1)
    )


def warmup_cosine(optimizer, config: Config, steps_per_epoch: int):
    """Linear warmup to each group's peak LR, then cosine decay to its floor, per update."""
    warmup = config.warmup_epochs * steps_per_epoch
    total = config.epochs * steps_per_epoch

    def factor(step):
        """LR multiplier for optimizer update `step` (0-based)."""
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup - 1))
        floor = config.min_lr_ratio
        return floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def make_optimizer(model: FENFormer, config: Config):
    """AdamW with a lower learning rate for the pretrained encoder than for the new head."""
    encoder = list(model.encoder.parameters())
    encoder_ids = {id(p) for p in encoder}
    head = [p for p in model.parameters() if id(p) not in encoder_ids]
    groups = [{"params": encoder, "lr": config.encoder_lr}, {"params": head, "lr": config.lr}]
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay)


def run_epoch(model, loader, config: Config, optimizer=None, scheduler=None) -> dict:
    """Train (with an optimizer) or evaluate one pass; return pooled and per-source metrics."""
    training = optimizer is not None
    model.train(training)
    sources = [s["source"] for s in loader.dataset.samples]
    totals = {}  # source -> [images, loss sum, exact boards, correct squares]
    with torch.set_grad_enabled(training):
        for pixels, targets, indices in loader:
            pixels = pixels.to(config.device, non_blocking=True).float().div_(255)
            targets = targets.to(config.device, non_blocking=True)
            with torch.autocast(
                torch.device(config.device).type,
                dtype=torch.bfloat16,
                enabled=config.precision == "bf16",
            ):
                logits = model(pixels)
                losses = square_loss(logits.float(), targets)
            loss = losses.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
            correct = (logits.argmax(-1) == targets).cpu()
            for i, index in enumerate(indices.tolist()):
                row = totals.setdefault(sources[index], [0, 0.0, 0, 0])
                row[0] += 1
                row[1] += losses[i].item()
                row[2] += int(correct[i].all())
                row[3] += int(correct[i].sum())
    return summarize(totals)


def summarize(totals: dict) -> dict:
    """Turn per-source counters into loss, exact-board accuracy and square accuracy."""

    def rates(images, loss, boards, squares):
        """Normalize one set of counters."""
        return {
            "loss": loss / images,
            "board_accuracy": boards / images,
            "square_accuracy": squares / (64 * images),
            "images": images,
        }

    pooled = [sum(column) for column in zip(*totals.values(), strict=True)]
    return {**rates(*pooled), "per_source": {k: rates(*v) for k, v in sorted(totals.items())}}


def seed_everything(seed: int):
    """Seed Python, NumPy and PyTorch, and require deterministic kernels."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def seed_worker(_worker_id):
    """Derive each loader worker's Python and NumPy seeds from its PyTorch seed."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def save(path: Path, state: dict):
    """Write a checkpoint atomically so an interrupted save never corrupts the last one."""
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def train(config: Config, *, resume: bool = False, on_epoch_end: Callable | None = None):
    """Run training; checkpoint `last.pt` every epoch and `best.pt` on new best val boards."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if config.precision not in ("fp32", "bf16"):
        raise ValueError("precision must be fp32 or bf16")
    seed_everything(config.seed)
    output = Path(config.output)
    train_set, val_set = mixture_datasets(Path(config.data), config.limit)
    generator = torch.Generator()
    loader_options = dict(
        batch_size=config.batch_size,
        num_workers=config.workers,
        worker_init_fn=seed_worker,
        pin_memory=config.device == "cuda",
        persistent_workers=config.workers > 0,
    )
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **loader_options)
    val_loader = DataLoader(val_set, shuffle=False, **loader_options)

    model = FENFormer(pretrained=config.pretrained and not resume).to(config.device)
    optimizer = make_optimizer(model, config)
    scheduler = warmup_cosine(optimizer, config, len(train_loader))
    state = {"epoch": 0, "history": [], "best": (-1.0, -math.inf), "best_loss": math.inf, "bad": 0}
    if resume:
        checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        if checkpoint["config"] != asdict(config):
            raise ValueError("Resume requires the exact configuration of the original run")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        state = checkpoint["state"]
        random.setstate(checkpoint["rng"]["python"])
        np.random.set_state(checkpoint["rng"]["numpy"])
        torch.set_rng_state(checkpoint["rng"]["torch"])
        if config.device == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["rng"]["cuda"])
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")

    tracking = None
    if config.wandb_project:
        import wandb

        tracking = wandb.init(
            project=config.wandb_project, name=output.name, config=asdict(config), dir=output
        )
    logger.info(
        "train=%d val=%d images; epochs %d..%d",
        len(train_set),
        len(val_set),
        state["epoch"] + 1,
        config.epochs,
    )
    try:
        for epoch in range(state["epoch"] + 1, config.epochs + 1):
            started = time.monotonic()
            generator.manual_seed(config.seed + epoch)  # reproducible order per epoch
            lr = [group["lr"] for group in optimizer.param_groups]
            train_metrics = run_epoch(model, train_loader, config, optimizer, scheduler)
            val_metrics = run_epoch(model, val_loader, config)

            # Early stopping watches val loss; checkpoint selection uses exact boards.
            if val_metrics["loss"] < state["best_loss"] - config.min_delta:
                state["best_loss"], state["bad"] = val_metrics["loss"], 0
            else:
                state["bad"] += 1
            score = (val_metrics["board_accuracy"], -val_metrics["loss"])
            is_best = score > tuple(state["best"])
            if is_best:
                state["best"] = score
                save(
                    output / "best.pt",
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "classes": CLASSES,
                        "val": val_metrics,
                    },
                )
            state["epoch"] = epoch
            state["history"].append(
                {"epoch": epoch, "lr": lr, "train": train_metrics, "val": val_metrics}
            )
            save(
                output / "last.pt",
                {
                    "config": asdict(config),
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "state": state,
                    "rng": {
                        "python": random.getstate(),
                        "numpy": np.random.get_state(),
                        "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if config.device == "cuda" else [],
                    },
                },
            )
            (output / "metrics.json").write_text(json.dumps(state["history"], indent=2) + "\n")
            if on_epoch_end:
                on_epoch_end()
            logger.info(
                "epoch %d %.0fs | train loss %.4f | val loss %.4f boards %.2f%% squares %.3f%%%s",
                epoch,
                time.monotonic() - started,
                train_metrics["loss"],
                val_metrics["loss"],
                100 * val_metrics["board_accuracy"],
                100 * val_metrics["square_accuracy"],
                " | best" if is_best else "",
            )
            if tracking:
                tracking.log(
                    flatten(
                        {
                            "train": train_metrics,
                            "val": val_metrics,
                            "lr/encoder": lr[0],
                            "lr/head": lr[1],
                        }
                    ),
                    step=epoch,
                )
            if config.patience and state["bad"] >= config.patience:
                logger.info("Early stop: no val-loss improvement for %d epochs", config.patience)
                break
    finally:
        if tracking:
            tracking.finish()
    return state["history"]


def flatten(metrics: dict, prefix: str = "") -> dict:
    """Flatten nested metric dicts into slash-separated keys for logging."""
    flat = {}
    for key, value in metrics.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, name + "/"))
        else:
            flat[name] = value
    return flat


def main():
    """Command-line entry point; every Config field is a flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    for key, value in asdict(Config()).items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=value)
        else:
            parser.add_argument(flag, type=type(value) if value is not None else int, default=value)
    parser.add_argument("--resume", action="store_true")
    args = vars(parser.parse_args())
    resume = args.pop("resume")
    train(Config(**args), resume=resume)


if __name__ == "__main__":
    main()
