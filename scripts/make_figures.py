"""Render the README figures from saved run outputs.

uv run python scripts/make_figures.py data   --gallery <dir of training PNGs>
uv run python scripts/make_figures.py curves --metrics runs/fenformer-v1/metrics.json
uv run python scripts/make_figures.py results --ours <eval dir> --baseline <eval dir>
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from PIL import Image

from fenformer.board import CLASSES, fen_to_labels
from fenformer.decode import decode
from fenformer.localize import board_box, download_u2netp, frame_board, letterbox, load_image
from fenformer.predict import load_model

ASSETS = Path(__file__).resolve().parents[1] / "assets"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
RED = "#e34948"
LIGHT_SQUARE, DARK_SQUARE = "#eeeed2", "#b5c18e"
GLYPHS = dict(zip(CLASSES[1:], "♙♘♗♖♕♔♟♞♝♜♛♚", strict=True))
FAMILIES = {
    "legacy": "procedural",
    "chesssight": "ChessSight",
    "geometric": "geometric",
    "authored": "A Beautiful Game",
    "handcarved": "hand-carved",
    "stylized": "stylized",
    "lowpoly": "low-poly",
    "polyhaven": "Poly Haven",
    "staunton": "Staunton",
    "mountaineer": "Mountaineer",
}
# Training split of the mixture (see docs/data.md).
MIXTURE = [
    ("Synthetic renders", 34000, BLUE),
    ("Synthetic, low-angle", 17000, ORANGE),
    ("ChessReD photos", 6479, AQUA),
    ("ChessCog renders", 4400, YELLOW),
]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": INK,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)


def content(path: Path) -> Image.Image:
    """Drop the black letterbox bars around a training image."""
    image = Image.open(path).convert("RGB")
    return image.crop(image.getbbox())


def data_figure(gallery: list[Path], output: Path):
    """Twelve training renders (6 x 2) above the composition of the training mixture."""
    fig = plt.figure(figsize=(12, 5.2))
    grid = fig.add_gridspec(3, 6, height_ratios=[1, 1, 0.8], hspace=0.22, wspace=0.04)
    for i, path in enumerate(gallery[:12]):
        ax = fig.add_subplot(grid[i // 6, i % 6])
        ax.imshow(content(path))
        ax.set_axis_off()
        ax.set_title(FAMILIES[path.stem.split("-")[-1]], fontsize=9, color=MUTED, pad=3)
        if path.stem.startswith("targeted-"):
            ax.text(
                0.03,
                0.05,
                "low angle",
                transform=ax.transAxes,
                fontsize=8,
                color="white",
                bbox=dict(boxstyle="round,pad=0.25", fc=ORANGE, ec="none"),
            )

    ax = fig.add_subplot(grid[2, :])
    total = sum(n for _, n, _ in MIXTURE)
    left = 0
    for i, (name, count, color) in enumerate(MIXTURE):
        ax.barh(0, count - 120, left=left, height=0.42, color=color)  # surface gap
        label = f"{name} · {count:,}"
        if count > total / 4:  # wide segments carry their label inside
            ax.text(
                left + count / 2,
                0,
                label,
                ha="center",
                va="center",
                fontsize=10,
                color="white",
                fontweight="bold",
            )
        elif i == len(MIXTURE) - 1:  # narrow segments: one label above, one below
            ax.text(total, -0.3, label, ha="right", va="top", fontsize=10)
        else:
            ax.text(left + count / 2, 0.3, label, ha="center", va="bottom", fontsize=10)
        left += count
    ax.set_xlim(0, total)
    ax.set_ylim(-0.75, 0.75)
    ax.set_axis_off()
    ax.set_title(
        f"{total:,} training images, 90% rendered, 10% real photos",
        fontsize=11,
        color=INK,
        loc="left",
        pad=2,
    )
    fig.savefig(output, dpi=150, bbox_inches="tight")


def draw_board(ax, labels, target=None):
    """Draw a 2D diagram, White at the bottom; outline wrong squares, correct piece in red."""
    for index, label in enumerate(labels):
        rank, file = 7 - index // 8, index % 8
        color = LIGHT_SQUARE if (rank + file) % 2 else DARK_SQUARE
        ax.add_patch(Rectangle((file, rank), 1, 1, color=color, lw=0))
        if label:
            ax.text(
                file + 0.5,
                rank + 0.45,
                GLYPHS[CLASSES[label]],
                ha="center",
                va="center",
                fontsize=15,
                color=INK,
            )
        if target is not None and label != target[index]:
            ax.add_patch(
                Rectangle((file + 0.06, rank + 0.06), 0.88, 0.88, fill=False, ec=RED, lw=2.2)
            )
            if target[index]:  # the correct piece, small in the corner
                ax.text(
                    file + 0.78,
                    rank + 0.76,
                    GLYPHS[CLASSES[target[index]]],
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=RED,
                    bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none"),
                )
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.set_axis_off()


def square_index(name: str) -> int:
    """FEN-order index of an algebraic square such as 'e1'."""
    return (8 - int(name[1])) * 8 + "abcdefgh".index(name[0])


def cross_attention(model, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return logits [1, 64, 13] and head-averaged cross-attention [layers, 64, 46, 46]."""
    maps = []
    for layer in model.decoder.layers:
        forward = layer.multihead_attn.forward

        def capture(*args, _forward=forward, **kwargs):
            """Request the attention weights that the decoder layer normally discards."""
            kwargs.update(need_weights=True, average_attn_weights=True)
            output, weights = _forward(*args, **kwargs)
            maps.append(weights[0])
            return output, weights

        layer.multihead_attn.forward = capture
    with torch.inference_mode():
        logits = model(pixels.float().div(255)[None])
    for layer in model.decoder.layers:
        del layer.multihead_attn.forward  # restore the class method
    side = model.image_size // 14
    return logits, torch.stack(maps).view(len(maps), 64, side, side)


def explain_figure(weights: Path, image_path: Path, fen: str, squares: list[str], output: Path):
    """Photo -> framed input -> where three square queries look -> predicted board."""
    u2netp = download_u2netp(Path("weights/u2netp.onnx"))
    model = load_model(weights, "cpu")
    photo = load_image(image_path)
    pixels = frame_board(photo, weights=u2netp)
    logits, maps = cross_attention(model, pixels)
    labels = decode(logits)[0].tolist()
    target = fen_to_labels(fen)

    # Map the board box from the 644 px letterbox back to the photo for display.
    box = board_box(letterbox(photo), u2netp)
    scale = max(photo.size) / 644
    pad_x, pad_y = (644 - photo.width / scale) / 2, (644 - photo.height / scale) / 2
    left, top, right, bottom = box
    framed = pixels.permute(1, 2, 0).numpy()
    attention = maps.mean(0)  # average over decoder layers

    fig, axes = plt.subplots(
        1, 3 + len(squares), figsize=(3.1 * (3 + len(squares)), 3.4), gridspec_kw={"wspace": 0.08}
    )
    axes[0].imshow(photo)
    axes[0].add_patch(
        Rectangle(
            ((left - pad_x) * scale, (top - pad_y) * scale),
            (right - left) * scale,
            (bottom - top) * scale,
            fill=False,
            ec=YELLOW,
            lw=2,
        )
    )
    axes[0].set_title("1  photo + U²-Net-p box", fontsize=10, loc="left")
    axes[1].imshow(framed)
    axes[1].set_title("2  model input, 644 px", fontsize=10, loc="left")
    for ax, square in zip(axes[2 : 2 + len(squares)], squares, strict=True):
        heat = attention[square_index(square)].numpy()
        heat = np.kron(heat / heat.max(), np.ones((14, 14)))
        ax.imshow(framed)
        ax.imshow(heat, cmap="magma", alpha=0.6, vmin=0, vmax=1)
        piece = CLASSES[labels[square_index(square)]]
        name = {".": "empty"}.get(piece, GLYPHS.get(piece, piece))
        ax.set_title(f"query {square} → {name}", fontsize=10, loc="left")
    axes[-1].set_title(
        "3  prediction" + (" ✓ exact" if labels == target else ""), fontsize=10, loc="left"
    )
    draw_board(axes[-1], labels, target)
    for ax in axes[:-1]:
        ax.set_axis_off()
    fig.savefig(output, dpi=150, bbox_inches="tight")


def query_peaks(attention_map: torch.Tensor) -> np.ndarray:
    """Pixel location [64, 2] of each query's attention peak in one layer's map [64, 46, 46]."""
    attention = attention_map.flatten(1) ** 4  # sharpened centroid: the peak, not diffuse context
    attention = attention / attention.sum(1, keepdim=True)
    side = attention_map.shape[-1]
    ys, xs = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    centers = torch.stack([xs.flatten(), ys.flatten()], 1).float() * 14 + 7
    return (attention @ centers).numpy()


def grid_figure(weights: Path, images: list[Path], output: Path):
    """Mark where each square query looks in the first decoder layer.

    Only the first layer localizes squares (peaks fit a perspective grid to about a fifth of a
    square); later layers attend widely for context, so they are not shown.
    """
    u2netp = download_u2netp(Path("weights/u2netp.onnx"))
    model = load_model(weights, "cpu")
    fig, axes = plt.subplots(
        1, len(images), figsize=(4.2 * len(images), 4.2), gridspec_kw={"wspace": 0.04}
    )
    for ax, path in zip(np.atleast_1d(axes), images, strict=True):
        pixels = frame_board(load_image(path), weights=u2netp)
        _, maps = cross_attention(model, pixels)
        peaks = query_peaks(maps[0]).reshape(8, 8, 2)  # [rank 8..1, file a..h, xy]
        ax.imshow(pixels.permute(1, 2, 0).numpy())
        ax.scatter(*peaks.reshape(-1, 2).T, s=22, color=YELLOW, edgecolor=INK, lw=0.6, zorder=3)
        for name, (r, f) in {"a1": (7, 0), "h1": (7, 7), "a8": (0, 0)}.items():
            ax.annotate(
                name,
                peaks[r, f],
                xytext=(5, 5),
                textcoords="offset points",
                color="white",
                fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc=INK, ec="none", alpha=0.7),
            )
        ax.set_axis_off()
    fig.savefig(output, dpi=150, bbox_inches="tight")


BENCHMARKS = {
    "chessred-test": "ChessReD test\n2,129 phone photos · seen set",
    "chesscog-test": "ChessCog test\n342 renders · seen set",
    "chesscog-transfer": "ChessCog transfer\n29 photos · unseen set",
    "samryan-500": "samryan-500\n500 top-down photos · unseen set",
}


def load_metrics(directory: Path, decoding: str) -> dict:
    """Read `<benchmark>.json` files written by evaluate or the baseline script."""
    return {
        name: json.loads((directory / f"{name}.json").read_text())["metrics"][decoding]
        for name in BENCHMARKS
        if (directory / f"{name}.json").exists()
    }


def results_figure(ours: Path, baseline: Path | None, output: Path):
    """Exact-board accuracy per benchmark with 95% Wilson intervals."""
    systems = [("fenformer (ours, ViT-B, 92 M)", load_metrics(ours, "constrained"), BLUE)]
    if baseline:
        systems.append(
            (
                "ChessQueries (ViT-L, 372 M), released weights",
                load_metrics(baseline, "constrained"),
                ORANGE,
            )
        )
    names = [n for n in BENCHMARKS if all(n in m for _, m, _ in systems)]
    fig, ax = plt.subplots(figsize=(9, 1.0 + 0.95 * len(names)))
    height = 0.8 / len(systems)
    for k, (label, metrics, color) in enumerate(systems):
        for i, name in enumerate(names):
            m = metrics[name]
            y = i + (k - (len(systems) - 1) / 2) * height
            low, high = m["board_accuracy_ci95"]
            value = m["board_accuracy"]
            ax.barh(y, value, height=height - 0.06, color=color, label=label if i == 0 else None)
            ax.plot([low, high], [y, y], color=INK, lw=1, alpha=0.6)
            ax.text(max(high, value) + 0.012, y, f"{value:.1%}", va="center", fontsize=10)
    ax.set_yticks(range(len(names)), [BENCHMARKS[n] for n in names], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("exact-board accuracy (all 64 squares correct) · whiskers: 95% CI")
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, frameon=False, fontsize=10)
    fig.savefig(output, dpi=150, bbox_inches="tight")


SOURCES = {  # display name and color, same slots as the data figure
    "render": ("synthetic", BLUE),
    "render_low_angle": ("synthetic, low-angle", ORANGE),
    "chessred": ("ChessReD photos", AQUA),
    "chesscog": ("ChessCog", YELLOW),
}


def curves_figure(metrics: Path, output: Path):
    """Loss (left) and per-source validation exact-board accuracy (right) per epoch."""
    history = json.loads(metrics.read_text())
    epochs = [h["epoch"] for h in history]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"wspace": 0.28})
    for split, color, style in (("train", MUTED, "--"), ("val", INK, "-")):
        left.plot(epochs, [h[split]["loss"] for h in history], style, color=color, lw=2)
        left.annotate(
            split,
            (epochs[-1], history[-1][split]["loss"]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=color,
            fontsize=10,
        )
    left.set_yscale("log")
    left.set_title("per-square cross-entropy", loc="left", fontsize=11)
    finals = {}
    for source, (name, color) in SOURCES.items():
        values = [h["val"]["per_source"][source]["board_accuracy"] for h in history]
        right.plot(epochs, values, color=color, lw=2, marker="o", ms=4)
        finals[f"{name} {values[-1]:.1%}"] = values[-1]
    # Direct labels, pushed apart from the top down so close final values never overlap.
    floor = float("inf")
    for label, value in sorted(finals.items(), key=lambda item: -item[1]):
        floor = min(value, floor - 0.055)
        right.annotate(label, (epochs[-1] + 0.4, floor), va="center", fontsize=9, color=INK)
    right.set_ylim(0, 1)
    right.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    right.set_title("validation exact-board accuracy, by source", loc="left", fontsize=11)
    for ax in (left, right):
        ax.set_xlabel("epoch")
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlim(0.5, epochs[-1] + 0.5)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    fig.savefig(output, dpi=150, bbox_inches="tight")


def examples_figure(results: list[Path], roots: list[Path], output: Path, wrong: bool):
    """Photos beside predicted boards: successes, or failures with wrong squares outlined.

    Takes the first two qualifying images (successes) or the two with the fewest errors
    (failures, the informative near-misses) from each benchmark result file.
    """
    rows = []
    for result, root in zip(results, roots, strict=True):
        records = json.loads(result.read_text())["predictions"]
        chosen = [r for r in records if (r["wrong_squares"] > 0) == wrong]
        chosen.sort(key=lambda r: (r["wrong_squares"], r["image"]))
        rows += [(root / r["image"], r) for r in chosen[:2]]
    # Two examples per row, each as a photo panel followed by a board panel.
    fig, axes = plt.subplots(
        len(rows) // 2,
        4,
        figsize=(12, 3.0 * len(rows) / 2),
        gridspec_kw={"wspace": 0.04, "hspace": 0.14},
        squeeze=False,
    )
    for k, (path, record) in enumerate(rows):
        photo_ax, board_ax = axes[k // 2, 2 * (k % 2)], axes[k // 2, 2 * (k % 2) + 1]
        photo = load_image(path)
        photo_ax.imshow(photo)
        photo_ax.set_axis_off()
        photo_ax.set_title(path.parents[1].name, fontsize=9, color=MUTED, loc="left")
        predicted, target = fen_to_labels(record["prediction"]), fen_to_labels(record["target"])
        draw_board(board_ax, predicted, target)
        verdict = "exact" if predicted == target else f"{record['wrong_squares']} wrong"
        board_ax.set_title(f"prediction · {verdict}", fontsize=9, color=MUTED, loc="left")
    fig.savefig(output, dpi=150, bbox_inches="tight")


def main():
    """Dispatch to one figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "figure", choices=["data", "explain", "grid", "results", "curves", "examples", "failures"]
    )
    parser.add_argument("--gallery", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--image", type=Path, nargs="+")
    parser.add_argument("--fen")
    parser.add_argument("--squares", default="e1,d8,e4")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ours", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--benchmarks", type=Path, nargs="+", help="benchmark directories")
    args = parser.parse_args()
    ASSETS.mkdir(exist_ok=True)
    if args.figure == "data":
        names = [
            "0012960-lightweight-legacy",
            "0033897-lightweight-stylized",
            "0036888-lightweight-staunton",
            "0007975-lightweight-authored",
            "0027915-lightweight-chesssight",
            "0009969-lightweight-handcarved",
            "0034894-lightweight-lowpoly",
            "0001993-lightweight-polyhaven",
            "targeted-0004984-lightweight-polyhaven",
            "targeted-0012960-lightweight-authored",
            "targeted-0016948-lightweight-staunton",
            "targeted-0013957-lightweight-mountaineer",
        ]
        data_figure([args.gallery / f"{n}.png" for n in names], ASSETS / "training_data.png")
    elif args.figure == "explain":
        output = args.output or ASSETS / "how_it_works.png"
        explain_figure(args.weights, args.image[0], args.fen, args.squares.split(","), output)
    elif args.figure == "grid":
        grid_figure(args.weights, args.image, args.output or ASSETS / "learned_grid.png")
    elif args.figure == "results":
        results_figure(args.ours, args.baseline, args.output or ASSETS / "results.png")
    elif args.figure == "curves":
        curves_figure(args.metrics, args.output or ASSETS / "training_curves.png")
    elif args.figure in ("examples", "failures"):
        results = [args.ours / f"{root.name}.json" for root in args.benchmarks]
        output = args.output or ASSETS / f"{args.figure}.png"
        examples_figure(results, args.benchmarks, output, wrong=args.figure == "failures")


if __name__ == "__main__":
    main()
