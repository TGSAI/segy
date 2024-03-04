"""Data models for manipulating all kinds of SEG-Y files."""


from segy.schema.data_type import Endianness
from segy.schema.data_type import ScalarType
from segy.schema.header import TextHeaderDescriptor
from segy.schema.header import TextHeaderEncoding
from segy.schema.segy import SegyDescriptor
from segy.schema.segy import SegyStandard
from segy.schema.trace import TraceDataDescriptor
from segy.schema.trace import TraceDescriptor

__all__ = [
    "Endianness",
    "ScalarType",
    "TextHeaderDescriptor",
    "TextHeaderEncoding",
    "TraceDataDescriptor",
    "TraceDescriptor",
    "SegyDescriptor",
    "SegyStandard",
]
