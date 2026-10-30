"""Constrained decoding: the highest-scoring board that satisfies three placement rules.

Rules: exactly one king per side, the kings are not adjacent, and no pawn stands on the
first or last rank. These are necessary conditions for a legal position, not sufficient.
"""

import torch

from .board import CLASSES

WHITE_KING, BLACK_KING = CLASSES.index("K"), CLASSES.index("k")
PAWNS = [CLASSES.index("P"), CLASSES.index("p")]
BACK_RANKS = torch.tensor([*range(8), *range(56, 64)])
_squares = torch.arange(64)
# Same square or king-adjacent (Chebyshev distance <= 1).
ADJACENT = ((_squares[:, None] // 8 - _squares[None, :] // 8).abs() <= 1) & (
    (_squares[:, None] % 8 - _squares[None, :] % 8).abs() <= 1
)


def decode(logits: torch.Tensor) -> torch.Tensor:
    """Map logits [B, 64, 13] to labels [B, 64] that maximize the summed square scores.

    With both king squares fixed, every other square is independent, so the exact optimum
    is a search over 64 x 64 king placements. Boards whose argmax already satisfies the
    rules are returned unchanged (this also keeps exact ties stable).
    """
    if logits.ndim != 3 or logits.shape[1:] != (64, len(CLASSES)):
        raise ValueError("Expected logits [B, 64, 13]")
    scores = logits.detach().to("cpu", torch.float64).clone()  # never mutate the caller's
    argmax = scores.argmax(-1)

    scores[:, BACK_RANKS[:, None], PAWNS] = -torch.inf
    non_king = scores.clone()
    non_king[:, :, [WHITE_KING, BLACK_KING]] = -torch.inf
    best_other, labels = non_king.max(-1)
    # Gain of placing a king on a square instead of that square's best other class.
    gain_white = scores[:, :, WHITE_KING] - best_other
    gain_black = scores[:, :, BLACK_KING] - best_other
    pair = (gain_white[:, :, None] + gain_black[:, None, :]).masked_fill(ADJACENT, -torch.inf)
    best = pair.flatten(1).argmax(-1)
    rows = torch.arange(len(scores))
    labels[rows, best // 64] = WHITE_KING
    labels[rows, best % 64] = BLACK_KING

    return torch.where(satisfies_rules(argmax)[:, None], argmax, labels).to(logits.device)


def satisfies_rules(labels: torch.Tensor) -> torch.Tensor:
    """Return a [B] mask of boards that already satisfy all three placement rules."""
    one_each = ((labels == WHITE_KING).sum(1) == 1) & ((labels == BLACK_KING).sum(1) == 1)
    back = labels[:, BACK_RANKS]
    no_back_pawns = ~((back == PAWNS[0]) | (back == PAWNS[1])).any(1)
    white = (labels == WHITE_KING).long().argmax(1)
    black = (labels == BLACK_KING).long().argmax(1)
    apart = ~ADJACENT[white, black]
    return one_each & no_back_pawns & apart
