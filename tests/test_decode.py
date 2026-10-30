import itertools

import torch

from fenformer.board import CLASSES, fen_to_labels
from fenformer.decode import ADJACENT, BACK_RANKS, PAWNS, decode, satisfies_rules


def confident_logits(fen: str, margin: float = 5.0) -> torch.Tensor:
    """Logits [1, 64, 13] whose argmax is the given board."""
    logits = torch.zeros(1, 64, 13)
    logits[0, torch.arange(64), torch.tensor(fen_to_labels(fen))] = margin
    return logits


def total_score(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Sum of the chosen class scores over all squares."""
    return logits[torch.arange(64), labels].sum().item()


def test_valid_argmax_is_unchanged():
    """A board that already satisfies the rules passes through untouched."""
    logits = confident_logits("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
    assert torch.equal(decode(logits), logits.argmax(-1))


def test_second_white_king_is_replaced_by_runner_up():
    """With two white kings, the less confident one takes its next-best class."""
    logits = confident_logits("4k3/8/8/8/8/8/8/K6K")
    h1 = 63
    logits[0, h1, CLASSES.index("K")] = 3.0
    logits[0, h1, CLASSES.index("R")] = 2.0
    labels = decode(logits)[0]
    assert CLASSES[labels[56]] == "K" and CLASSES[labels[h1]] == "R"


def test_matches_brute_force_on_random_logits():
    """Decoding equals an exhaustive search over king squares (all other squares independent)."""
    torch.manual_seed(0)
    logits = torch.randn(8, 64, 13)
    labels = decode(logits)
    assert satisfies_rules(labels).all()
    allowed = logits.clone()
    allowed[:, BACK_RANKS[:, None], PAWNS] = -torch.inf
    allowed[:, :, [CLASSES.index("K"), CLASSES.index("k")]] = -torch.inf
    base = allowed.max(-1).values
    for b in range(len(logits)):
        if satisfies_rules(logits[b : b + 1].argmax(-1))[0]:
            continue
        best = max(
            base[b].sum()
            - base[b, w]
            - base[b, k]
            + logits[b, w, CLASSES.index("K")]
            + logits[b, k, CLASSES.index("k")]
            for w, k in itertools.product(range(64), repeat=2)
            if not ADJACENT[w, k]
        )
        assert abs(total_score(logits[b], labels[b]) - best.item()) < 1e-4


def test_does_not_mutate_input():
    """The caller's logits stay intact."""
    logits = torch.randn(2, 64, 13, dtype=torch.float64)
    before = logits.clone()
    decode(logits)
    assert torch.equal(logits, before)
