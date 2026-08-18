"""Test the inference utilities that are not covered via SegyFile."""

from __future__ import annotations

import numpy as np
import pytest

from segy.ebcdic import ASCII_TO_EBCDIC
from segy.inference import _is_valid_text_header
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


def make_wrapped_text() -> str:
    """Build a newline-wrapped, SEG-Y sized textual header."""
    return "\n".join(
        f"C{row:02d} SAMPLE TEXTUAL HEADER".ljust(COLS) for row in range(1, ROWS + 1)
    )


def to_ebcdic(text: str) -> bytes:
    """Encode ASCII string to EBCDIC bytes."""
    buffer = np.frombuffer(text.encode("ascii"), dtype="uint8")
    return ASCII_TO_EBCDIC[buffer].tobytes()  # type: ignore[no-any-return]


class TestIsValidTextHeader:
    """Cover layout / printable predicates used by encoding inference."""

    def test_valid_header(self) -> None:
        """A correctly wrapped printable header is accepted."""
        assert _is_valid_text_header(make_wrapped_text(), ROWS, COLS) is True

    def test_wrong_row_count(self) -> None:
        """Too few rows must fail the layout check."""
        text = "\n".join([" " * COLS] * (ROWS - 1))
        assert _is_valid_text_header(text, ROWS, COLS) is False

    def test_wrong_column_count(self) -> None:
        """A short card must fail the layout check."""
        lines = [" " * COLS] * ROWS
        lines[0] = " " * (COLS - 1)
        assert _is_valid_text_header("\n".join(lines), ROWS, COLS) is False

    def test_non_printable(self) -> None:
        """NUL is 7-bit but not printable."""
        lines = [" " * COLS] * ROWS
        lines[0] = "\x00" + " " * (COLS - 1)
        assert _is_valid_text_header("\n".join(lines), ROWS, COLS) is False

    def test_non_ascii(self) -> None:
        """A char above 127 must fail even if Python marks it printable."""
        lines = [" " * COLS] * ROWS
        lines[0] = "\u00e9" + " " * (COLS - 1)
        assert _is_valid_text_header("\n".join(lines), ROWS, COLS) is False


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
        buffer = b"\x01" * (ROWS * COLS)

        assert infer_text_header_encoding(buffer, spec) == spec_encoding

    def test_infer_ascii_with_nul_padding(self) -> None:
        """ASCII header with NUL padding must still be detected as ASCII."""
        spec = TextHeaderSpec(rows=ROWS, cols=COLS)
        text = make_text()
        raw = bytearray(text.encode("ascii"))
        raw[292] = 0
        raw[397] = 0
        raw[553] = 0
        raw[798] = 0

        buffer = bytes(raw)

        assert infer_text_header_encoding(buffer, spec) == TextHeaderEncoding.ASCII
        ascii_spec = TextHeaderSpec(
            rows=ROWS, cols=COLS, encoding=TextHeaderEncoding.ASCII
        )
        decoded = ascii_spec.processor.decode(buffer)
        assert "\x00" not in decoded
        assert decoded[292] == " "
