"""SEG-Y Revision 1 Specification."""

from segy.schema import Endianness
from segy.schema import HeaderField
from segy.schema import HeaderSpec
from segy.schema import ScalarType
from segy.schema import SegySpec
from segy.schema import SegyStandard
from segy.schema import TextHeaderEncoding
from segy.schema import TextHeaderSpec
from segy.schema import TraceDataSpec
from segy.schema import TraceSpec
from segy.standards.rev0 import BINARY_FILE_HEADER_FIELDS_REV0
from segy.standards.rev0 import TRACE_HEADER_FIELDS_REV0

BINARY_FILE_HEADER_FIELDS_REV1 = BINARY_FILE_HEADER_FIELDS_REV0 + [
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


TRACE_HEADER_FIELDS_REV1 = TRACE_HEADER_FIELDS_REV0 + [
    HeaderField(
        name="cdp_x",
        byte=181,
        format=ScalarType.INT32,
        description="X coordinate of ensemble (CDP) position",
    ),
    HeaderField(
        name="cdp_y",
        byte=185,
        format=ScalarType.INT32,
        description="Y coordinate of ensemble (CDP) position",
    ),
    HeaderField(
        name="inline",
        byte=189,
        format=ScalarType.INT32,
        description="Inline number",
    ),
    HeaderField(
        name="crossline",
        byte=193,
        format=ScalarType.INT32,
        description="Crossline number",
    ),
    HeaderField(
        name="shot_point",
        byte=197,
        format=ScalarType.INT32,
        description="Shotpoint number",
    ),
    HeaderField(
        name="shot_point_scalar",
        byte=201,
        format=ScalarType.INT16,
        description="Scalar to be applied to the shotpoint number",
    ),
    HeaderField(
        name="trace_value_measurement_unit",
        byte=203,
        format=ScalarType.INT16,
        description="Trace value measurement unit",
    ),
    HeaderField(
        name="transduction_constant_mantissa",
        byte=205,
        format=ScalarType.INT32,
        description="Transduction Constant Mantissa",
    ),
    HeaderField(
        name="transduction_constant_exponent",
        byte=209,
        format=ScalarType.INT16,
        description="Transduction Constant Exponent",
    ),
    HeaderField(
        name="transduction_units",
        byte=211,
        format=ScalarType.INT16,
        description="Transduction Units",
    ),
    HeaderField(
        name="device_trace_id",
        byte=213,
        format=ScalarType.INT16,
        description="Device/Trace Identifier",
    ),
    HeaderField(
        name="times_scalar",
        byte=215,
        format=ScalarType.INT16,
        description="Scalar to be applied to times",
    ),
    HeaderField(
        name="source_type_orientation",
        byte=217,
        format=ScalarType.INT16,
        description="Source Type/Orientation",
    ),
    HeaderField(
        name="source_energy_direction_mantissa",
        byte=219,
        format=ScalarType.INT32,
        description="Source Energy Direction with respect to vertical [Mantissa]",
    ),
    HeaderField(
        name="source_energy_direction_exponent",
        byte=223,
        format=ScalarType.INT16,
        description="Source Energy Direction with respect to vertical [Exponent]",
    ),
    HeaderField(
        name="source_measurement_mantissa",
        byte=225,
        format=ScalarType.INT32,
        description="Source Measurement Mantissa",
    ),
    HeaderField(
        name="source_measurement_exponent",
        byte=229,
        format=ScalarType.INT16,
        description="Source Measurement Exponent",
    ),
    HeaderField(
        name="source_measurement_unit",
        byte=231,
        format=ScalarType.INT16,
        description="Source Measurement Unit",
    ),
]


rev1_textual_file_header = TextHeaderSpec(
    rows=40,
    cols=80,
    offset=0,
    encoding=TextHeaderEncoding.EBCDIC,
    format=ScalarType.UINT8,  # noqa: A003
)


rev1_binary_file_header = HeaderSpec(
    fields=BINARY_FILE_HEADER_FIELDS_REV1,
    item_size=400,
    offset=3200,
)


rev1_ext_text_header = TextHeaderSpec(
    rows=40,
    cols=80,
    offset=3600,
    encoding=TextHeaderEncoding.EBCDIC,
    format=ScalarType.UINT8,  # noqa: A003
)


rev1_trace_header = HeaderSpec(
    fields=TRACE_HEADER_FIELDS_REV1,
    item_size=240,
)


rev1_trace_data = TraceDataSpec(
    format=ScalarType.IBM32,  # noqa: A003
)


rev1_trace = TraceSpec(header_spec=rev1_trace_header, data_spec=rev1_trace_data)


rev1_segy = SegySpec(
    segy_standard=SegyStandard.REV1,
    text_file_header=rev1_textual_file_header,
    binary_file_header=rev1_binary_file_header,
    ext_text_header=rev1_ext_text_header,
    trace=rev1_trace,
    endianness=Endianness.BIG,
)
