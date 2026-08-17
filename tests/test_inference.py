"""Test the inference utilities that are not covered via SegyFile."""

from __future__ import annotations

import numpy as np
import pytest

from segy.ebcdic import ASCII_TO_EBCDIC
from segy.inference import infer_text_header_encoding
from segy.schema import TextHeaderEncoding
from segy.schema import TextHeaderSpec

ROWS = 40
COLS = 80


def make_text() -> str:
    """Build an unwrapped, SEG-Y sized textual header with card numbers."""
    cards = [
        f"C{row:02d} SAMPLE TEXTUAL HEADER".ljust(COLS) for row in range(1, ROWS + 1)
    ]
    return "".join(cards)


def to_ebcdic(text: str) -> bytes:
    """Encode ASCII string to EBCDIC bytes."""
    buffer = np.frombuffer(text.encode("ascii"), dtype="uint8")
    return ASCII_TO_EBCDIC[buffer].tobytes()  # type: ignore[no-any-return]


class TestInferTextHeaderEncoding:
    """Test textual file header encoding inference."""

    @pytest.mark.parametrize(
        "spec_encoding",
        [TextHeaderEncoding.EBCDIC, TextHeaderEncoding.ASCII],
    )
    def test_infer_ascii(self, spec_encoding: TextHeaderEncoding) -> None:
        """ASCII header must be detected, no matter what the spec falls back to."""
        spec = TextHeaderSpec(rows=ROWS, cols=COLS, encoding=spec_encoding)
        buffer = make_text().encode("ascii")

        assert infer_text_header_encoding(buffer, spec) == TextHeaderEncoding.ASCII

    @pytest.mark.parametrize(
        "spec_encoding",
        [TextHeaderEncoding.EBCDIC, TextHeaderEncoding.ASCII],
    )
    def test_infer_ebcdic(self, spec_encoding: TextHeaderEncoding) -> None:
        """EBCDIC header must be detected, no matter what the spec falls back to."""
        spec = TextHeaderSpec(rows=ROWS, cols=COLS, encoding=spec_encoding)
        buffer = to_ebcdic(make_text())

        assert infer_text_header_encoding(buffer, spec) == TextHeaderEncoding.EBCDIC

    @pytest.mark.parametrize(
        "spec_encoding",
        [TextHeaderEncoding.EBCDIC, TextHeaderEncoding.ASCII],
    )
    def test_ambiguous_header_keeps_spec_encoding(
        self, spec_encoding: TextHeaderEncoding
    ) -> None:
        """A header that is valid in both encodings must keep the spec encoding."""
        spec = TextHeaderSpec(rows=ROWS, cols=COLS, encoding=spec_encoding)
        buffer = b"\x40" * (ROWS * COLS)  # EBCDIC space, ASCII "@"

        assert infer_text_header_encoding(buffer, spec) == spec_encoding

    @pytest.mark.parametrize(
        "spec_encoding",
        [TextHeaderEncoding.EBCDIC, TextHeaderEncoding.ASCII],
    )
    def test_invalid_header_keeps_spec_encoding(
        self, spec_encoding: TextHeaderEncoding
    ) -> None:
        """A header that is valid in neither encoding must keep the spec encoding."""
        spec = TextHeaderSpec(rows=ROWS, cols=COLS, encoding=spec_encoding)
        buffer = bytes(ROWS * COLS)

        assert infer_text_header_encoding(buffer, spec) == spec_encoding
