"""Build the two zero-shot real-photo benchmarks from their public sources.

    uv run python scripts/prepare_benchmarks.py --cache downloads --output data/benchmarks

samryan-500        500 top-down photos of a vinyl tournament set
                   (github.com/samryan18/chess-dataset, MIT). The FEN is the filename
                   with "-" for "/"; a "_2" suffix marks a second photo of a position.
chesscog-transfer  29 photos of a wooden set from the ChessCog transfer-learning sample
                   (github.com/georg-wolflein/chesscog). Labels come from per-image JSON.

Neither physical set appears in training. Each benchmark is written as
<output>/<name>/{images/..., labels.json} and packed as <output>/<name>.tar.
"""

import argparse
import json
import re
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

from fenformer.board import fen_to_labels

SAMRYAN_REPO = "https://github.com/samryan18/chess-dataset.git"
CHESSCOG_TRANSFER = (
    "https://drive.usercontent.google.com/download"
    "?id=1Z9fTXRb7FlqzgTTXoywgiQP-Z1cH1v3W&export=download&confirm=t"
)


def build_samryan(clone: Path, output: Path) -> dict:
    """Copy the raw photos and derive labels from their filenames."""
    labels = {}
    for path in sorted((clone / "labeled_originals").glob("*.JPG")):
        fen = re.sub(r"_\d+$", "", path.stem).replace("-", "/")
        fen_to_labels(fen)
        labels[f"images/{path.name}"] = fen
        shutil.copy2(path, output / "images" / path.name)
    return labels


def build_chesscog_transfer(archive: Path, output: Path) -> dict:
    """Extract the train and test photos (none are used for training here)."""
    labels = {}
    with zipfile.ZipFile(archive) as source:
        for name in sorted(source.namelist()):
            if name.startswith("__MACOSX") or not name.endswith(".png"):
                continue
            target = name.replace("/", "-")  # train/01.png -> train-01.png
            (output / "images" / target).write_bytes(source.read(name))
            fen = json.loads(source.read(name[: -len(".png")] + ".json"))["fen"].split()[0]
            fen_to_labels(fen)
            labels[f"images/{target}"] = fen
    return labels


def main():
    """Download sources into the cache if missing, then write both benchmarks."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--cache", type=Path, default=Path("downloads"))
    parser.add_argument("--output", type=Path, default=Path("data/benchmarks"))
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    clone = args.cache / "chess-dataset"
    if not clone.exists():
        subprocess.run(["git", "clone", "--depth", "1", SAMRYAN_REPO, str(clone)], check=True)
    archive = args.cache / "chesscog-transfer.zip"
    if not archive.exists():
        subprocess.run(["curl", "-fsSL", "-o", str(archive), CHESSCOG_TRANSFER], check=True)

    builders = {
        "samryan-500": (build_samryan, clone),
        "chesscog-transfer": (build_chesscog_transfer, archive),
    }
    for name, (build, source) in builders.items():
        output = args.output / name
        (output / "images").mkdir(parents=True, exist_ok=False)
        labels = build(source, output)
        (output / "labels.json").write_text(json.dumps(labels, indent=1, sort_keys=True) + "\n")
        with tarfile.open(args.output / f"{name}.tar", "w") as tar:
            tar.add(output, arcname=name)
        print(f"{name}: {len(labels)} images")


if __name__ == "__main__":
    main()
