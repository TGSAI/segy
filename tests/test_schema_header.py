"""Tests for the Text Headers, Binary Headers, and Trace Headers for different SEGY revisions."""

from __future__ import annotations

import numpy as np
import pytest

from segy.schema import ScalarType
from segy.schema import TextHeaderEncoding
from segy.schema import TextHeaderSpec


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
        text_spec = TextHeaderSpec(
            rows=rows, cols=cols, encoding=encoding, format=ScalarType.UINT8
        )

        text_unwrapped = text_spec._unwrap(text)
        text_bytes = text_spec._encode(text_unwrapped)
        text_decoded = text_spec._decode(text_bytes)
        text_wrapped = text_spec._wrap(text_decoded)

        num_char = rows * cols
        assert text_spec.dtype == np.dtype(("uint8", (num_char,)))
        assert text_decoded == text.replace("\n", "")
        assert text_wrapped == text

    def test_encode_invalid_shape_exception(self) -> None:
        """Test exception is raised when shape is invalid in encode function."""
        test_spec = TextHeaderSpec(
            rows=2, cols=5, encoding=TextHeaderEncoding.EBCDIC, format=ScalarType.UINT8
        )

        with pytest.raises(ValueError, match="length must be equal to rows x cols."):
            test_spec._encode("obviously not 10 characters")

    def test_wrap_invalid_shape_exception(self) -> None:
        """Test exception is raised when shape is invalid in wrap function."""
        text_spec = TextHeaderSpec(
            rows=3, cols=4, encoding=TextHeaderEncoding.EBCDIC, format=ScalarType.UINT8
        )
        with pytest.raises(ValueError, match="rows x cols must be equal wrapped text"):
            text_spec._wrap("obviously not 12 characters")
