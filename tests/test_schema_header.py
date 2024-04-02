"""Tests for the Text Headers, Binary Headers, and Trace Headers for different SEGY revisions."""

from __future__ import annotations

import numpy as np
import pytest

from segy.schema import ScalarType
from segy.schema.header import TextHeaderDescriptor
from segy.schema.header import TextHeaderEncoding


class TestTextHeaderDescriptor:
    """Tests for the TextHeaderDescriptor class initialization and encoding/decoding."""

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
        text_descriptor = TextHeaderDescriptor(
            rows=rows, cols=cols, encoding=encoding, format=ScalarType.UINT8
        )

        text_unwrapped = text_descriptor._unwrap(text)
        text_bytes = text_descriptor._encode(text_unwrapped)
        text_decoded = text_descriptor._decode(text_bytes)
        text_wrapped = text_descriptor._wrap(text_decoded)

        num_char = rows * cols
        assert text_descriptor.dtype == np.dtype(("uint8", (num_char,)))
        assert text_decoded == text.replace("\n", "")
        assert text_wrapped == text

    def test_encode_invalid_shape_exception(self) -> None:
        """Test exception is raised when shape is invalid in encode function."""
        text_descriptor = TextHeaderDescriptor(
            rows=2, cols=5, encoding=TextHeaderEncoding.EBCDIC, format=ScalarType.UINT8
        )

        with pytest.raises(ValueError, match="length must be equal to rows x cols."):
            text_descriptor._encode("obviously not 10 characters")

    def test_wrap_invalid_shape_exception(self) -> None:
        """Test exception is raised when shape is invalid in wrap function."""
        text_descriptor = TextHeaderDescriptor(
            rows=3, cols=4, encoding=TextHeaderEncoding.EBCDIC, format=ScalarType.UINT8
        )
        with pytest.raises(ValueError, match="rows x cols must be equal wrapped text"):
            text_descriptor._wrap("obviously not 12 characters")
