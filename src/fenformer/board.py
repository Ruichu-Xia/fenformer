"""Board encoding: 64 squares in FEN order (a8, b8, ..., h1), 13 classes per square."""

CLASSES = ".PNBRQKpnbrqk"  # index 0 is the empty square


def fen_to_labels(fen: str) -> list[int]:
    """Decode the piece-placement field of a FEN into 64 class indices."""
    ranks = fen.strip().split(" ")[0].split("/")
    if len(ranks) != 8:
        raise ValueError(f"FEN must contain eight ranks: {fen!r}")
    labels = []
    for rank in ranks:
        row = []
        for char in rank:
            if char in "12345678":
                row.extend([0] * int(char))
            elif char in CLASSES[1:]:
                row.append(CLASSES.index(char))
            else:
                raise ValueError(f"Invalid FEN character {char!r} in {fen!r}")
        if len(row) != 8:
            raise ValueError(f"FEN rank must expand to eight squares: {rank!r}")
        labels.extend(row)
    return labels


def labels_to_fen(labels: list[int]) -> str:
    """Encode 64 class indices as a FEN piece-placement field."""
    if len(labels) != 64 or any(not 0 <= int(c) < len(CLASSES) for c in labels):
        raise ValueError("Expected 64 class indices in [0, 12]")
    ranks = []
    for start in range(0, 64, 8):
        rank, empty = "", 0
        for label in labels[start : start + 8]:
            if label == 0:
                empty += 1
            else:
                rank += (str(empty) if empty else "") + CLASSES[label]
                empty = 0
        ranks.append(rank + (str(empty) if empty else ""))
    return "/".join(ranks)
