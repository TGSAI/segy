"""Tests for HeaderField and HeaderSpec helpers."""

from __future__ import annotations

from typing import TypeAlias

import pytest

from segy.schema import ScalarType
from segy.schema.header import HeaderField
from segy.schema.header import _validate_non_overlapping_fields
from segy.schema.header import ranges_overlap

FieldRange: TypeAlias = tuple[int, int]


@pytest.mark.parametrize(
    ("range1", "range2", "expected"),
    [
        ((1, 5), (5, 9), False),  # Adjacent, no overlap
        ((1, 5), (3, 7), True),  # Overlap
        ((1, 5), (6, 10), False),  # No overlap
    ],
)
def test_ranges_overlap(range1: FieldRange, range2: FieldRange, expected: bool) -> None:
    """Test function that checks if ranges overlap."""
    assert ranges_overlap(range1, range2) == expected


def test_validate_non_overlapping_headers() -> None:
    """Test validation of non-overlapping headers."""
    valid_fields = [
        HeaderField(name="a", byte=1, format=ScalarType.INT32),  # [1,5)
        HeaderField(name="b", byte=5, format=ScalarType.INT32),  # [5,9)
    ]
    _validate_non_overlapping_fields(valid_fields)  # No error

    duplicate_names = [HeaderField(name="a", byte=1, format=ScalarType.INT32)] * 2
    with pytest.raises(ValueError, match="Duplicate header field names detected"):
        _validate_non_overlapping_fields(duplicate_names)

    overlapping = [
        HeaderField(name="a", byte=1, format=ScalarType.INT32),  # [1,5)
        HeaderField(name="b", byte=3, format=ScalarType.INT32),  # [3,7)
    ]
    with pytest.raises(ValueError, match="Header fields overlap"):
        _validate_non_overlapping_fields(overlapping)
