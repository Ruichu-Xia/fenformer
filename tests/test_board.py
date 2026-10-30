import pytest

from fenformer.board import CLASSES, fen_to_labels, labels_to_fen

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def test_start_position_layout():
    """a8 is the first square and holds a black rook; h1 is the last and holds a white rook."""
    labels = fen_to_labels(START)
    assert len(labels) == 64
    assert CLASSES[labels[0]] == "r" and CLASSES[labels[63]] == "R"
    assert labels[16:48] == [0] * 32


@pytest.mark.parametrize("fen", [START, "8/8/8/4k3/8/8/8/K7", "r3k2r/8/8/8/8/8/8/R3K2R"])
def test_round_trip(fen):
    """Encoding then decoding returns the same placement."""
    assert labels_to_fen(fen_to_labels(fen)) == fen


def test_full_fen_uses_placement_field_only():
    """Side to move, castling and clocks are ignored."""
    assert fen_to_labels(START + " w KQkq - 0 1") == fen_to_labels(START)


@pytest.mark.parametrize("fen", ["8/8/8/8/8/8/8", "9/8/8/8/8/8/8/8", "x7/8/8/8/8/8/8/8"])
def test_invalid_fen_raises(fen):
    """Malformed placements fail loudly instead of producing a wrong board."""
    with pytest.raises(ValueError):
        fen_to_labels(fen)
