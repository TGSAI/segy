"""SEG-Y Revision 1 Specification."""

from segy.schema import BinaryHeaderDescriptor
from segy.schema import Endianness
from segy.schema import HeaderFieldDescriptor
from segy.schema import ScalarType
from segy.schema import SegyDescriptor
from segy.schema import SegyStandard
from segy.schema import TextHeaderDescriptor
from segy.schema import TextHeaderEncoding
from segy.schema import TraceDataDescriptor
from segy.schema import TraceDescriptor
from segy.schema import TraceHeaderDescriptor
from segy.standards.rev0 import BINARY_FILE_HEADER_FIELDS_REV0
from segy.standards.rev0 import TRACE_HEADER_FIELDS_REV0

BINARY_FILE_HEADER_FIELDS_REV1 = BINARY_FILE_HEADER_FIELDS_REV0 + [
    HeaderFieldDescriptor(
        name="seg_y_revision",
        offset=300,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="SEG Y Format Revision Number",
    ),
    HeaderFieldDescriptor(
        name="fixed_length_trace_flag",
        offset=302,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Fixed Length Trace Flag",
    ),
    HeaderFieldDescriptor(
        name="extended_textual_headers",
        offset=304,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Number of 3200-byte, Extended Textual File Header Records Following the Binary Header",  # noqa: E501
    ),
    HeaderFieldDescriptor(
        name="additional_trace_headers",
        offset=306,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Maximum Number of Additional Trace Headers",
    ),
]


TRACE_HEADER_FIELDS_REV1 = TRACE_HEADER_FIELDS_REV0 + [
    HeaderFieldDescriptor(
        name="x_coordinate",
        offset=180,
        format=ScalarType.INT32,
        endianness=Endianness.BIG,
        description="X coordinate of ensemble (CDP) position",
    ),
    HeaderFieldDescriptor(
        name="y_coordinate",
        offset=184,
        format=ScalarType.INT32,
        endianness=Endianness.BIG,
        description="Y coordinate of ensemble (CDP) position",
    ),
    HeaderFieldDescriptor(
        name="inline_no",
        offset=188,
        format=ScalarType.INT32,
        endianness=Endianness.BIG,
        description="Inline number",
    ),
    HeaderFieldDescriptor(
        name="crossline_no",
        offset=192,
        format=ScalarType.INT32,
        endianness=Endianness.BIG,
        description="Crossline number",
    ),
    HeaderFieldDescriptor(
        name="shotpoint_no",
        offset=196,
        format=ScalarType.INT32,
        endianness=Endianness.BIG,
        description="Shotpoint number",
    ),
    HeaderFieldDescriptor(
        name="scalar_apply_shotpoint",
        offset=200,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Scalar to be applied to the shotpoint number",
    ),
    HeaderFieldDescriptor(
        name="trace_value_measurement_unit",
        offset=202,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Trace value measurement unit",
    ),
    HeaderFieldDescriptor(
        name="transduction_constant_mantissa",
        offset=204,
        format=ScalarType.INT32,
        endianness=Endianness.BIG,
        description="Transduction Constant Mantissa",
    ),
    HeaderFieldDescriptor(
        name="transduction_constant_exponent",
        offset=208,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Transduction Constant Exponent",
    ),
    HeaderFieldDescriptor(
        name="transduction_units",
        offset=210,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Transduction Units",
    ),
    HeaderFieldDescriptor(
        name="device_trace_id",
        offset=212,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Device/Trace Identifier",
    ),
    HeaderFieldDescriptor(
        name="times_scalar",
        offset=214,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Scalar to be applied to times",
    ),
    HeaderFieldDescriptor(
        name="source_type_orientation",
        offset=216,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Source Type/Orientation",
    ),
    HeaderFieldDescriptor(
        name="source_energy_direction_mantissa",
        offset=218,
        format=ScalarType.INT32,
        endianness=Endianness.BIG,
        description="Source Energy Direction with respect to vertical [Mantissa]",
    ),
    HeaderFieldDescriptor(
        name="source_energy_direction_exponent",
        offset=222,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Source Energy Direction with respect to vertical [Exponent]",
    ),
    HeaderFieldDescriptor(
        name="source_measurement_mantissa",
        offset=224,
        format=ScalarType.INT32,
        endianness=Endianness.BIG,
        description="Source Measurement Mantissa",
    ),
    HeaderFieldDescriptor(
        name="source_measurement_exponent",
        offset=228,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Source Measurement Exponent",
    ),
    HeaderFieldDescriptor(
        name="source_measurement_unit",
        offset=230,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Source Measurement Unit",
    ),
]


class TextualFileHeaderDescriptorRev1(TextHeaderDescriptor):
    """Textual file header spec with SEG-Y Rev1 defaults."""

    description: str = "3200-byte textual file header with 40 lines of text."
    rows: int = 40
    cols: int = 80
    offset: int = 0
    encoding: TextHeaderEncoding = TextHeaderEncoding.EBCDIC
    format: ScalarType = ScalarType.UINT8  # noqa: A003


class ExtendedTextualHeaderDescriptorRev1(TextHeaderDescriptor):
    """Extended text header spec with SEG-Y Rev1 defaults."""

    description: str = "3200-byte extended textual header with 40 lines of text."
    rows: int = 40
    cols: int = 80
    offset: int = 3600
    encoding: TextHeaderEncoding = TextHeaderEncoding.EBCDIC
    format: ScalarType = ScalarType.UINT8  # noqa: A003


class BinaryHeaderDescriptorRev1(BinaryHeaderDescriptor):
    """Binary file header spec with SEG-Y Rev1 defaults."""

    description: str = "400-byte binary file header with structured fields."
    fields: list[HeaderFieldDescriptor] = BINARY_FILE_HEADER_FIELDS_REV1
    item_size: int = 400
    offset: int = 3200


class TraceHeaderDescriptorRev1(TraceHeaderDescriptor):
    """Trace header spec with SEG-Y Rev1 defaults."""

    description: str = "240-byte trace header with structured fields."
    fields: list[HeaderFieldDescriptor] = TRACE_HEADER_FIELDS_REV1
    item_size: int = 240


class TraceDataDescriptorRev1(TraceDataDescriptor):
    """Trace data spec with SEG-Y Rev1 defaults."""

    description: str = "Trace data with given format and sample count."
    format: ScalarType = ScalarType.IBM32  # noqa: A003
    endianness: Endianness = Endianness.BIG


class TraceDescriptorRev1(TraceDescriptor):
    """Trace spec with SEG-Y Rev1 defaults."""

    description: str = "Trace spec with header and data information."
    header_descriptor: TraceHeaderDescriptor = TraceHeaderDescriptorRev1()
    data_descriptor: TraceDataDescriptor = TraceDataDescriptorRev1()


class SegyDescriptorRev1(SegyDescriptor):
    """SEG-Y file spec with SEG-Y Rev1 defaults."""

    segy_standard: SegyStandard = SegyStandard.REV1
    text_file_header: TextHeaderDescriptor = TextualFileHeaderDescriptorRev1()
    binary_file_header: BinaryHeaderDescriptor = BinaryHeaderDescriptorRev1()
    extended_text_header: TextHeaderDescriptor = ExtendedTextualHeaderDescriptorRev1()
    trace: TraceDescriptor = TraceDescriptorRev1()
