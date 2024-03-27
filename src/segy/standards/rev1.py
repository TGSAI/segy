"""SEG-Y Revision 1 Specification."""

from segy.schema import Endianness
from segy.schema import ScalarType
from segy.schema import SegyDescriptor
from segy.schema import SegyStandard
from segy.schema import TextHeaderDescriptor
from segy.schema import TextHeaderEncoding
from segy.schema import TraceDescriptor
from segy.schema import TraceSampleDescriptor
from segy.schema.data_type import StructuredDataTypeDescriptor
from segy.schema.data_type import StructuredFieldDescriptor
from segy.standards.rev0 import BINARY_FILE_HEADER_FIELDS_REV0
from segy.standards.rev0 import TRACE_HEADER_FIELDS_REV0

BINARY_FILE_HEADER_FIELDS_REV1 = BINARY_FILE_HEADER_FIELDS_REV0 + [
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
    StructuredFieldDescriptor(
        name="additional_trace_headers",
        offset=306,
        format=ScalarType.INT16,
        description="Maximum Number of Additional Trace Headers",
    ),
]


TRACE_HEADER_FIELDS_REV1 = TRACE_HEADER_FIELDS_REV0 + [
    StructuredFieldDescriptor(
        name="x_coordinate",
        offset=180,
        format=ScalarType.INT32,
        description="X coordinate of ensemble (CDP) position",
    ),
    StructuredFieldDescriptor(
        name="y_coordinate",
        offset=184,
        format=ScalarType.INT32,
        description="Y coordinate of ensemble (CDP) position",
    ),
    StructuredFieldDescriptor(
        name="inline_no",
        offset=188,
        format=ScalarType.INT32,
        description="Inline number",
    ),
    StructuredFieldDescriptor(
        name="crossline_no",
        offset=192,
        format=ScalarType.INT32,
        description="Crossline number",
    ),
    StructuredFieldDescriptor(
        name="shotpoint_no",
        offset=196,
        format=ScalarType.INT32,
        description="Shotpoint number",
    ),
    StructuredFieldDescriptor(
        name="scalar_apply_shotpoint",
        offset=200,
        format=ScalarType.INT16,
        description="Scalar to be applied to the shotpoint number",
    ),
    StructuredFieldDescriptor(
        name="trace_value_measurement_unit",
        offset=202,
        format=ScalarType.INT16,
        description="Trace value measurement unit",
    ),
    StructuredFieldDescriptor(
        name="transduction_constant_mantissa",
        offset=204,
        format=ScalarType.INT32,
        description="Transduction Constant Mantissa",
    ),
    StructuredFieldDescriptor(
        name="transduction_constant_exponent",
        offset=208,
        format=ScalarType.INT16,
        description="Transduction Constant Exponent",
    ),
    StructuredFieldDescriptor(
        name="transduction_units",
        offset=210,
        format=ScalarType.INT16,
        description="Transduction Units",
    ),
    StructuredFieldDescriptor(
        name="device_trace_id",
        offset=212,
        format=ScalarType.INT16,
        description="Device/Trace Identifier",
    ),
    StructuredFieldDescriptor(
        name="times_scalar",
        offset=214,
        format=ScalarType.INT16,
        description="Scalar to be applied to times",
    ),
    StructuredFieldDescriptor(
        name="source_type_orientation",
        offset=216,
        format=ScalarType.INT16,
        description="Source Type/Orientation",
    ),
    StructuredFieldDescriptor(
        name="source_energy_direction_mantissa",
        offset=218,
        format=ScalarType.INT32,
        description="Source Energy Direction with respect to vertical [Mantissa]",
    ),
    StructuredFieldDescriptor(
        name="source_energy_direction_exponent",
        offset=222,
        format=ScalarType.INT16,
        description="Source Energy Direction with respect to vertical [Exponent]",
    ),
    StructuredFieldDescriptor(
        name="source_measurement_mantissa",
        offset=224,
        format=ScalarType.INT32,
        description="Source Measurement Mantissa",
    ),
    StructuredFieldDescriptor(
        name="source_measurement_exponent",
        offset=228,
        format=ScalarType.INT16,
        description="Source Measurement Exponent",
    ),
    StructuredFieldDescriptor(
        name="source_measurement_unit",
        offset=230,
        format=ScalarType.INT16,
        description="Source Measurement Unit",
    ),
]


rev1_textual_file_header = TextHeaderDescriptor(
    rows=40,
    cols=80,
    offset=0,
    encoding=TextHeaderEncoding.EBCDIC,
    format=ScalarType.UINT8,  # noqa: A003
)


rev1_binary_file_header = StructuredDataTypeDescriptor(
    fields=BINARY_FILE_HEADER_FIELDS_REV1,
    item_size=400,
    offset=3200,
)


rev1_extended_text_header = TextHeaderDescriptor(
    rows=40,
    cols=80,
    offset=3600,
    encoding=TextHeaderEncoding.EBCDIC,
    format=ScalarType.UINT8,  # noqa: A003
)


rev1_trace_header = StructuredDataTypeDescriptor(
    fields=TRACE_HEADER_FIELDS_REV1,
    item_size=240,
)


rev1_trace_data = TraceSampleDescriptor(
    format=ScalarType.IBM32,  # noqa: A003
)


rev1_trace = TraceDescriptor(
    header_descriptor=rev1_trace_header,
    sample_descriptor=rev1_trace_data,
)


rev1_segy = SegyDescriptor(
    segy_standard=SegyStandard.REV1,
    text_file_header=rev1_textual_file_header,
    binary_file_header=rev1_binary_file_header,
    extended_text_header=rev1_extended_text_header,
    trace=rev1_trace,
    endianness=Endianness.BIG,
)
