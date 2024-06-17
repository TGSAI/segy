"""Tests for the Text Headers, Binary Headers, and Trace Headers for different SEGY revisions."""

from __future__ import annotations

import numpy as np
import pytest

from segy.schema import TextHeaderEncoding
from segy.schema import TextHeaderSpec
from segy.schema.text_header import ExtendedTextHeaderSpec


class TestTextHeaderSpec:
    """Tests for the text header spec initialization and encoding/decoding."""

    @pytest.mark.parametrize(
        ("rows", "cols", "encoding", "text"),
        [
            (2, 5, TextHeaderEncoding.ASCII, "hello\nworld"),
            (3, 6, TextHeaderEncoding.EBCDIC, "hello,\n world\nebcdic"),
        ],
    )
    def test_encode_decode(
        self, rows: int, cols: int, encoding: TextHeaderEncoding, text: str
    ) -> None:
        """Test reading ASCII or EBCDIC text headers."""
        text_spec = TextHeaderSpec(rows=rows, cols=cols, encoding=encoding)

        text_bytes = text_spec.encode(text)
        text_roundtrip = text_spec.decode(text_bytes)

        num_char = rows * cols
        assert text_spec.dtype == np.dtype(("uint8", (num_char,)))
        assert text_roundtrip == text


class TestExtTextHeaderSpec:
    """Tests for the text header spec initialization and encoding/decoding."""

    @pytest.mark.parametrize(
        ("rows", "cols", "encoding", "text"),
        [
            (2, 7, TextHeaderEncoding.ASCII, ["hello, \nworld! "]),
            (2, 4, TextHeaderEncoding.EBCDIC, ["foo,\nbar.", "fizz\nbuzz"]),
        ],
    )
    def test_encode_decode(
        self, rows: int, cols: int, encoding: TextHeaderEncoding, text: list[str]
    ) -> None:
        """Test reading ASCII or EBCDIC text headers."""
        count = len(text)

        text_spec = TextHeaderSpec(rows=rows, cols=cols, encoding=encoding)
        ext_text_spec = ExtendedTextHeaderSpec(spec=text_spec, count=count)

        text_bytes = ext_text_spec.encode(text)
        text_roundtrip = ext_text_spec.decode(text_bytes)

        num_char = rows * cols * count
        assert ext_text_spec.dtype == np.dtype(("uint8", (num_char,)))
        assert text_roundtrip == text
