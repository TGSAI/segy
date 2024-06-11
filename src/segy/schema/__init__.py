"""Data models for manipulating all kinds of SEG-Y files."""

from segy.schema.base import Endianness
from segy.schema.format import ScalarType
from segy.schema.format import TextHeaderEncoding
from segy.schema.header import HeaderField
from segy.schema.header import HeaderSpec
from segy.schema.segy import SegySpec
from segy.schema.segy import SegyStandard
from segy.schema.text_header import ExtendedTextHeaderSpec
from segy.schema.text_header import TextHeaderSpec
from segy.schema.trace import TraceDataSpec
from segy.schema.trace import TraceSpec

__all__ = [
    "Endianness",
    "ExtendedTextHeaderSpec",
    "ScalarType",
    "HeaderField",
    "HeaderSpec",
    "SegySpec",
    "SegyStandard",
    "TextHeaderEncoding",
    "TextHeaderSpec",
    "TraceDataSpec",
    "TraceSpec",
]
