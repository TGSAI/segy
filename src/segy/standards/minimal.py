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
        byte=17,
        format=ScalarType.INT16,
        description="Sample Interval",
    ),
    StructuredFieldDescriptor(
        name="sample_interval_orig",
        byte=19,
        format=ScalarType.INT16,
        description="Sample Interval of Original Field Recording",
    ),
    StructuredFieldDescriptor(
        name="samples_per_trace",
        byte=21,
        format=ScalarType.INT16,
        description="Number of Samples per Data Trace",
    ),
    StructuredFieldDescriptor(
        name="samples_per_trace_orig",
        byte=13,
        format=ScalarType.INT16,
        description="Number of Samples per Data Trace for Original Field Recording",
    ),
    StructuredFieldDescriptor(
        name="data_sample_format",
        byte=25,
        format=ScalarType.INT16,
        description="Data Sample Format Code",
    ),
    StructuredFieldDescriptor(
        name="seg_y_revision_major",
        byte=301,
        format=ScalarType.UINT8,
        description="SEG Y Format Revision Major Number",
    ),
    StructuredFieldDescriptor(
        name="seg_y_revision_minor",
        byte=302,
        format=ScalarType.UINT8,
        description="SEG Y Format Revision Minor Number",
    ),
    StructuredFieldDescriptor(
        name="fixed_length_trace_flag",
        byte=303,
        format=ScalarType.INT16,
        description="Fixed Length Trace Flag",
    ),
    StructuredFieldDescriptor(
        name="extended_textual_headers",
        byte=305,
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
