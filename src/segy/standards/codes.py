"""Metadata and attribute encoding for headers in the SEG-Y standard."""

from enum import IntEnum


class SegyEndianCode(IntEnum):
    """Enumeration for endianness indicator from raw SEG-Y bytes.

    SEG-Y Revision 2 defines byte range 3297-3300 with an integer
    constant (32-bit) for unambiguous detection of file endianness.
    There are three cases. If it reads the value:
    - 0: Unknown pre-rev2, assume big-endian.
    - 1690906010: Don't swap bytes
    - 6730598510: Reverse bytes
    - 3362099510: Swap consecutive bytes
    """

    LEGACY = 0x00_00_00_00
    NATIVE = 0x01_02_03_04
    REVERSE = 0x04_03_02_01
    PAIRWISE_SWAP = 0x02_01_04_03


class DataSampleFormatCode(IntEnum):
    """Trace data type (format) codes for SEG-Y."""

    IBM32 = 1
    INT32 = 2
    INT16 = 3
    FLOAT32 = 5
    FLOAT64 = 6
    INT8 = 8
    INT64 = 9
    UINT32 = 10
    UINT16 = 11
    UINT64 = 12
    UINT8 = 16


class TraceSortingCode(IntEnum):
    """Trace sorting codes for SEG-Y trace headers."""

    OTHER = -1
    UNKNOWN = 0
    AS_RECORDED = 1
    CDP_ENSEMBLE = 2
    SINGLE_FOLD = 3
    HORIZONTALLY_STACKED = 4
    COMMON_SOURCE_POINT = 5
    COMMON_RECEIVER_POINT = 6
    COMMON_OFFSET_POINT = 7
    COMMOND_MID_POINT = 8
    COMMON_CONVERSION_POINT = 9


class SweepTypeCode(IntEnum):
    """Enumeration for types of sweep codes."""

    LINEAR = 1
    PARABOLIC = 2
    EXPONENTIAL = 3
    OTHER = 4


class SweepTaperType(IntEnum):
    """Enumeration for types of sweep taper codes."""

    LINEAR = 1
    COSINE_SQUARED = 2
    OTHER = 3


class CorrelatedTraces(IntEnum):
    """Enumeration for correlated traces."""

    NO = 1
    YES = 2


class BinaryGainRecovered(IntEnum):
    """Enumeration for binary gain recovery."""

    YES = 1
    NO = 2


class AmplitudeRecoverMethod(IntEnum):
    """Enumeration for amplitude recovery method."""

    NONE = 1
    SPHERICAL_DIVERGENCE = 2
    AGC = 3
    OTHER = 4


class MeasurementSystem(IntEnum):
    """Enumeration for units used in measurements."""

    METERS = 1
    FEET = 2
