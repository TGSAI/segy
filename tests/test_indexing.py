"""Tests for indexing.

The indexer classes are tested via the `SegyFile` tests so no
need to test it here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from segy.indexing import bounds_check
from segy.indexing import merge_cat_file

if TYPE_CHECKING:
    from fsspec.implementations.memory import MemoryFileSystem


class TestBoundsCheckIndices:
    """Tests for bounds_check function."""

    @pytest.mark.parametrize(
        ("indices", "size"),
        [
            ([1, 2, 3], 5),  # within bounds
            ([4], 5),  # max index (upper edge case)
            ([0], 5),  # min index (lower edge case)
            ([], 5),  # empty index list
        ],
    )
    def test_in_bounds(self, indices: list[int], size: int) -> None:
        """Test the case where indices are in bounds."""
        bounds_check(np.asarray(indices), size, "trace")

    @pytest.mark.parametrize(
        ("indices", "size"),
        [
            ([0, 16], 5),  # out of bounds item (beyond upper limit)
            ([-1], 5),  # negative index (unsupported)
        ],
    )
    def test_out_of_bounds(self, indices: list[int], size: int) -> None:
        """Test the case where indices out of bounds or negative."""
        with pytest.raises(IndexError, match="out of bounds"):
            bounds_check(np.asarray(indices), size, "")


class TestMergeCatFile:
    """Tests for merge_cat_file function."""

    def test_merge_cat_file(self, mock_filesystem: MemoryFileSystem) -> None:
        """Test reading the correct ranges from fake binary data."""
        test_data = b"This is a test file containing test data."
        uri = "/text_merge_cat_file.bin"

        file = mock_filesystem.open(uri, mode="wb")
        file.write(test_data)

        starts = [0, 10, 24]
        ends = [9, 23, 33]

        result = merge_cat_file(mock_filesystem, uri, starts, ends)

        expected_result = b"".join(
            test_data[s:e] for s, e in zip(starts, ends, strict=True)
        )
        assert result == expected_result
