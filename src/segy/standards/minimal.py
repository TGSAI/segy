"""SEG-Y file specification with minimal, mandatory fields."""

from segy.schema import ScalarType
from segy.schema import SegyDescriptor
from segy.schema import TextHeaderDescriptor
from segy.schema import TextHeaderEncoding
from segy.schema import TraceDescriptor
from segy.schema import TraceSampleDescriptor
from segy.schema.data_type import Endianness
from segy.schema.data_type import StructuredDataTypeDescriptor
from segy.schema.data_type import StructuredFieldDescriptor

BINARY_FILE_HEADER_FIELDS = [
    StructuredFieldDescriptor(
        name="sample_interval",
        offset=16,
        format=ScalarType.INT16,
        description="Sample Interval",
    ),
    StructuredFieldDescriptor(
        name="sample_interval_orig",
        offset=18,
        format=ScalarType.INT16,
        description="Sample Interval of Original Field Recording",
    ),
    StructuredFieldDescriptor(
        name="samples_per_trace",
        offset=20,
        format=ScalarType.INT16,
        description="Number of Samples per Data Trace",
    ),
    StructuredFieldDescriptor(
        name="samples_per_trace_orig",
        offset=22,
        format=ScalarType.INT16,
        description="Number of Samples per Data Trace for Original Field Recording",
    ),
    StructuredFieldDescriptor(
        name="data_sample_format",
        offset=24,
        format=ScalarType.INT16,
        description="Data Sample Format Code",
    ),
    StructuredFieldDescriptor(
        name="seg_y_revision",
        offset=300,
        format=ScalarType.INT16,
        description="SEG Y Format Revision Number",
    ),
    StructuredFieldDescriptor(
        name="fixed_length_trace_flag",
        offset=302,
        format=ScalarType.INT16,
        description="Fixed Length Trace Flag",
    ),
    StructuredFieldDescriptor(
        name="extended_textual_headers",
        offset=304,
        format=ScalarType.INT16,
        description="Number of 3200-byte, Extended Textual File Header Records Following the Binary Header",  # noqa: E501
    ),
]

textual_file_header = TextHeaderDescriptor(
    rows=40,
    cols=80,
    offset=0,
    encoding=TextHeaderEncoding.EBCDIC,
    format=ScalarType.UINT8,
)

binary_file_header = StructuredDataTypeDescriptor(
    fields=BINARY_FILE_HEADER_FIELDS, item_size=400, offset=3200
)

trace_header = StructuredDataTypeDescriptor(fields=[], item_size=240)
trace_data = TraceSampleDescriptor(format=ScalarType.IBM32)
trace = TraceDescriptor(header_descriptor=trace_header, sample_descriptor=trace_data)

minimal_segy = SegyDescriptor(
    segy_standard=None,
    text_file_header=textual_file_header,
    binary_file_header=binary_file_header,
    trace=trace,
    endianness=Endianness.BIG,
)
