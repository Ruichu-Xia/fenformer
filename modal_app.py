"""Train and evaluate on Modal.

    modal run modal_app.py --run fenformer-v1                        # train on one A100
    modal run modal_app.py --run fenformer-v1 --action evaluate      # score all benchmarks

Volumes: `fenformer-data` holds `mixture-v1.tar` and `benchmarks/<name>.tar`;
`fenformer-runs` receives checkpoints, metrics and evaluation results.
"""

import json
import os
import tarfile
import time
from dataclasses import asdict
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
# Name of the Modal Secret that holds WANDB_API_KEY. Baked into the image env so the
# container resolves the same name as the launching client.
WANDB_SECRET = os.environ.get("FENFORMER_WANDB_SECRET", "wandb")
BENCHMARKS = ("chessred-test", "chesscog-test", "chesscog-transfer", "samryan-500")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_sync(
        uv_project_dir=str(ROOT),
        uv_version="0.6.17",
        frozen=False,
        extra_options="--locked --no-dev --default-index https://pypi.org/simple",
    )
    # Parallelism comes from loader processes; avoid nested BLAS thread pools in each.
    .env(
        {
            "PYTHONPATH": "/workspace/src",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "FENFORMER_WANDB_SECRET": WANDB_SECRET,
        }
    )
    .add_local_dir(ROOT / "src", "/workspace/src", ignore=["**/__pycache__/**"])
)
app = modal.App("fenformer")
data = modal.Volume.from_name("fenformer-data")
runs = modal.Volume.from_name("fenformer-runs", create_if_missing=True)


def stage(archive: Path) -> Path:
    """Extract a dataset archive to container-local disk; return its top-level directory."""
    started = time.monotonic()
    destination = Path("/tmp/data")
    with tarfile.open(archive, "r|", bufsize=1 << 20) as tar:
        tar.extractall(destination, filter="data")
    (top,) = [p for p in destination.iterdir() if p.name == archive.stem]
    print(f"Staged {archive.name} in {time.monotonic() - started:.0f}s", flush=True)
    return top


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=16,
    memory=32768,
    timeout=int(7.5 * 3600),
    retries=0,
    volumes={"/data": data.read_only(), "/runs": runs},
    secrets=[modal.Secret.from_name(WANDB_SECRET, required_keys=["WANDB_API_KEY"])],
)
def train_remote(config: dict, resume: bool):
    """Stage the mixture locally, then train with checkpoints committed every epoch."""
    from fenformer.train import Config, train

    config = Config(**{**config, "data": str(stage(Path("/data/mixture-v1.tar")))})
    try:
        return train(config, resume=resume, on_epoch_end=runs.commit)[-1]
    finally:
        runs.commit()


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8,
    memory=16384,
    timeout=3600,
    volumes={"/data": data.read_only(), "/runs": runs},
)
def evaluate_remote(run: str, benchmarks: list[str], framing: bool = True) -> dict:
    """Score `best.pt` of a run on each benchmark and store results next to the run.

    `framing=False` feeds the whole letterboxed photo, skipping U2NETP (an ablation).
    """
    from fenformer.evaluate import evaluate
    from fenformer.localize import download_u2netp
    from fenformer.predict import load_model

    model = load_model(Path("/runs") / run / "best.pt", "cuda")
    u2netp = download_u2netp(Path("/tmp/u2netp.onnx"))
    output = Path("/runs") / run / ("evaluation" if framing else "evaluation-no-framing")
    output.mkdir(exist_ok=True)
    summary = {}
    for name in benchmarks:
        root = Path("/tmp/data") / name
        if not root.exists():
            root = stage(Path(f"/data/benchmarks/{name}.tar"))
        metrics, records = evaluate(model, root, u2netp if framing else None, batch_size=32)
        result = {"metrics": metrics, "predictions": records}
        (output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
        summary[name] = metrics
        print(name, json.dumps(metrics["constrained"]), flush=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    runs.commit()
    return summary


@app.local_entrypoint()
def main(
    run: str,
    action: str = "train",
    epochs: int = 20,
    limit: int = 0,
    resume: bool = False,
    wandb_project: str = "fenformer",
    no_framing: bool = False,
):
    """Launch training (or a short `--limit` probe) or evaluation for a named run."""
    if action == "train":
        from fenformer.train import Config

        config = Config(
            output=f"/runs/{run}",
            epochs=epochs,
            limit=limit or None,
            workers=16,
            wandb_project=wandb_project,
        )
        print(json.dumps(train_remote.remote(asdict(config), resume), indent=2))
    elif action == "evaluate":
        summary = evaluate_remote.remote(run, list(BENCHMARKS), framing=not no_framing)
        print(json.dumps(summary, indent=2))
    else:
        raise ValueError("action must be train or evaluate")
