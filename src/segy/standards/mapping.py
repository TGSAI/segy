"""Mappings from internal schemas to SEG-Y definitions."""

from __future__ import annotations

from bidict import bidict

from segy.schema import ScalarType

SEGY_FORMAT_MAP = bidict(
    {
        ScalarType.IBM32: 1,
        ScalarType.INT32: 2,
        ScalarType.INT16: 3,
        ScalarType.FLOAT32: 5,
        ScalarType.FLOAT64: 6,
        ScalarType.INT8: 8,
        ScalarType.INT64: 9,
        ScalarType.UINT32: 10,
        ScalarType.UINT16: 11,
        ScalarType.UINT64: 12,
        ScalarType.UINT8: 16,
    }
)
