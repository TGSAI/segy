"""Test the inference utilities that are not covered via SegyFile."""

from __future__ import annotations

import numpy as np

from segy.ebcdic import ASCII_TO_EBCDIC
from segy.inference import infer_text_header_encoding
from segy.schema import TextHeaderEncoding

ROWS = 40
COLS = 80


def make_text(rows: int = ROWS) -> str:
    """Build an unwrapped, SEG-Y sized textual header with card numbers."""
    cards = [
        f"C{row:02d} SAMPLE TEXTUAL HEADER".ljust(COLS) for row in range(1, rows + 1)
    ]
    return "".join(cards)


def to_ebcdic(text: str) -> bytes:
    """Encode ASCII string to EBCDIC bytes."""
    buffer = np.frombuffer(text.encode("ascii"), dtype="uint8")
    return ASCII_TO_EBCDIC[buffer].tobytes()  # type: ignore[no-any-return]


class TestInferTextHeaderEncoding:
    """Test textual file header encoding inference."""

    def test_infer_ascii(self) -> None:
        """ASCII header must be detected."""
        buffer = make_text().encode("ascii")

        assert infer_text_header_encoding(buffer) == TextHeaderEncoding.ASCII

    def test_infer_ebcdic(self) -> None:
        """EBCDIC header must be detected."""
        buffer = to_ebcdic(make_text())

        assert infer_text_header_encoding(buffer) == TextHeaderEncoding.EBCDIC

    def test_ambiguous_header_falls_back_to_ebcdic(self) -> None:
        """A header valid in both encodings must keep the EBCDIC write default."""
        buffer = b"\x40" * (ROWS * COLS)  # EBCDIC space, ASCII "@"

        assert infer_text_header_encoding(buffer) == TextHeaderEncoding.EBCDIC

    def test_invalid_header_falls_back_to_ebcdic(self) -> None:
        """A header valid in neither encoding must keep the EBCDIC write default."""
        buffer = b"\x01" * (ROWS * COLS)

        assert infer_text_header_encoding(buffer) == TextHeaderEncoding.EBCDIC

    def test_infer_ascii_with_nul_padding(self) -> None:
        """ASCII header whose unused cards are NUL filled must still be ASCII."""
        text = make_text(rows=ROWS - 1)
        buffer = text.encode("ascii") + b"\x00" * COLS

        assert infer_text_header_encoding(buffer) == TextHeaderEncoding.ASCII
