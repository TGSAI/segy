"""SEG-Y file specification with minimal, mandatory fields."""

from segy.schema import Endianness
from segy.schema import HeaderField
from segy.schema import HeaderSpec
from segy.schema import ScalarType
from segy.schema import SegySpec
from segy.schema import TextHeaderEncoding
from segy.schema import TextHeaderSpec
from segy.schema import TraceDataSpec
from segy.schema import TraceSpec

BINARY_FILE_HEADER_FIELDS = [
    HeaderField(
        name="sample_interval",
        byte=17,
        format=ScalarType.INT16,
        description="Sample Interval",
    ),
    HeaderField(
        name="sample_interval_orig",
        byte=19,
        format=ScalarType.INT16,
        description="Sample Interval of Original Field Recording",
    ),
    HeaderField(
        name="samples_per_trace",
        byte=21,
        format=ScalarType.INT16,
        description="Number of Samples per Data Trace",
    ),
    HeaderField(
        name="samples_per_trace_orig",
        byte=13,
        format=ScalarType.INT16,
        description="Number of Samples per Data Trace for Original Field Recording",
    ),
    HeaderField(
        name="data_sample_format",
        byte=25,
        format=ScalarType.INT16,
        description="Data Sample Format Code",
    ),
    HeaderField(
        name="seg_y_revision",
        byte=301,
        format=ScalarType.INT16,
        description="SEG Y Format Revision Number",
    ),
    HeaderField(
        name="fixed_length_trace_flag",
        byte=303,
        format=ScalarType.INT16,
        description="Fixed Length Trace Flag",
    ),
    HeaderField(
        name="ext_textual_headers",
        byte=305,
        format=ScalarType.INT16,
        description="Number of 3200-byte, Extended Textual File Header Records Following the Binary Header",  # noqa: E501
    ),
]

textual_file_header = TextHeaderSpec(
    rows=40,
    cols=80,
    offset=0,
    encoding=TextHeaderEncoding.EBCDIC,
    format=ScalarType.UINT8,
)

binary_file_header = HeaderSpec(
    fields=BINARY_FILE_HEADER_FIELDS, item_size=400, offset=3200
)

trace_header = HeaderSpec(fields=[], item_size=240)
trace_data = TraceDataSpec(format=ScalarType.IBM32)
trace = TraceSpec(header_spec=trace_header, data_spec=trace_data)

minimal_segy = SegySpec(
    segy_standard=None,
    text_file_header=textual_file_header,
    binary_file_header=binary_file_header,
    trace=trace,
    endianness=Endianness.BIG,
)
