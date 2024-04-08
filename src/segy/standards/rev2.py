"""SEG-Y Revision 2 Specification."""

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
from segy.standards.rev1 import BINARY_FILE_HEADER_FIELDS_REV1
from segy.standards.rev1 import TRACE_HEADER_FIELDS_REV1

BINARY_FILE_HEADER_FIELDS_REV2 = BINARY_FILE_HEADER_FIELDS_REV1 + [
    StructuredFieldDescriptor(
        name="extended_data_traces_ensemble",
        offset=60,
        format=ScalarType.INT32,
        description="Extended number of data traces per ensemble.",
    ),
    StructuredFieldDescriptor(
        name="extended_aux_traces_ensemble",
        offset=64,
        format=ScalarType.INT32,
        description="Extended number of auxiliary traces per ensemble.",
    ),
    StructuredFieldDescriptor(
        name="extended_samples_per_trace",
        offset=68,
        format=ScalarType.INT32,
        description="Extended number of samples per data trace.",
    ),
    StructuredFieldDescriptor(
        name="extended_sample_interval",
        offset=72,
        format=ScalarType.FLOAT64,
        description="Extended sample interval, IEEE double precision.",
    ),
    StructuredFieldDescriptor(
        name="extended_sample_interval_orig",
        offset=80,
        format=ScalarType.FLOAT64,
        description="Extended sample interval of original field recording.",
    ),
    StructuredFieldDescriptor(
        name="extended_samples_per_trace_orig",
        offset=88,
        format=ScalarType.INT32,
        description="Extended number of samples per data trace in original recording.",
    ),
    StructuredFieldDescriptor(
        name="extended_ensemble_fold",
        offset=92,
        format=ScalarType.INT32,
        description="Extended ensemble fold.",
    ),
    StructuredFieldDescriptor(
        name="endian_indicator",
        offset=96,
        format=ScalarType.INT32,
        description="The integer constants for endianness. Indicates byte ordering.",
    ),
    StructuredFieldDescriptor(
        name="max_extended_trace_headers",
        offset=306,
        format=ScalarType.INT32,
        description="Maximum number of additional 240 byte trace headers.",
    ),
    StructuredFieldDescriptor(
        name="time_basis_code",
        offset=310,
        format=ScalarType.INT16,
        description="Time Basis Code",
    ),
    StructuredFieldDescriptor(
        name="no_traces",
        offset=312,
        format=ScalarType.UINT64,
        description="Time Basis Code",
    ),
    StructuredFieldDescriptor(
        name="first_trace_offset",
        offset=320,
        format=ScalarType.UINT64,
        description="Byte offset of first trace relative to start of file or stream.",
    ),
    StructuredFieldDescriptor(
        name="no_data_trailer_records",
        offset=328,
        format=ScalarType.INT32,
        description="Number of 3200-byte data trailer stanza records following last trace.",
    ),
]


TRACE_HEADER_FIELDS_REV2 = TRACE_HEADER_FIELDS_REV1 + [
    StructuredFieldDescriptor(
        name="trace_header_name",
        offset=232,
        format=ScalarType.STRING8,
        description="Either zero or trace header name 'SEG00000' in ASCII or EBCDIC.",
    ),
]

EXTENDED_TRACE_HEADER_FIELDS_REV2 = [
    StructuredFieldDescriptor(
        name="extended_trace_seq_line",
        offset=0,
        format=ScalarType.UINT64,
        description="Extended trace sequence number within line",
    ),
    StructuredFieldDescriptor(
        name="extended_trace_seq_file",
        offset=8,
        format=ScalarType.UINT64,
        description="Extended trace sequence number within SEG-Y file",
    ),
    StructuredFieldDescriptor(
        name="extended_field_rec_no",
        offset=16,
        format=ScalarType.INT64,
        description="Extended original field record number",
    ),
    StructuredFieldDescriptor(
        name="extended_ensemble_no",
        offset=24,
        format=ScalarType.INT64,
        description="Extended ensemble number",
    ),
    StructuredFieldDescriptor(
        name="extended_rec_elev",
        offset=32,
        format=ScalarType.FLOAT64,
        description="Extended elevation of receiver group",
    ),
    StructuredFieldDescriptor(
        name="extended_rec_depth",
        offset=40,
        format=ScalarType.FLOAT64,
        description="Receiver group depth below surface",
    ),
    StructuredFieldDescriptor(
        name="extended_src_elev",
        offset=48,
        format=ScalarType.FLOAT64,
        description="Extended surface elevation at source location",
    ),
    StructuredFieldDescriptor(
        name="extended_src_depth",
        offset=56,
        format=ScalarType.FLOAT64,
        description="Extended source depth below surface",
    ),
    StructuredFieldDescriptor(
        name="extended_datum_elev_rec",
        offset=64,
        format=ScalarType.FLOAT64,
        description="Extended Seismic Datum elevation at receiver group",
    ),
    StructuredFieldDescriptor(
        name="extended_datum_elev_src",
        offset=72,
        format=ScalarType.FLOAT64,
        description="Extended Seismic Datum elevation at source",
    ),
    StructuredFieldDescriptor(
        name="extended_water_depth_src",
        offset=80,
        format=ScalarType.FLOAT64,
        description="Extended water column height at source location",
    ),
    StructuredFieldDescriptor(
        name="extended_water_depth_rec",
        offset=88,
        format=ScalarType.FLOAT64,
        description="Extended water column height at receiver group location",
    ),
    StructuredFieldDescriptor(
        name="extended_src_x",
        offset=96,
        format=ScalarType.FLOAT64,
        description="Extended source coordinate - X",
    ),
    StructuredFieldDescriptor(
        name="extended_src_y",
        offset=104,
        format=ScalarType.FLOAT64,
        description="Extended source coordinate - Y",
    ),
    StructuredFieldDescriptor(
        name="extended_rec_x",
        offset=112,
        format=ScalarType.FLOAT64,
        description="Extended group coordinate - X",
    ),
    StructuredFieldDescriptor(
        name="extended_rec_y",
        offset=120,
        format=ScalarType.FLOAT64,
        description="Extended group coordinate - Y",
    ),
    StructuredFieldDescriptor(
        name="extended_dist_src_to_rec",
        offset=128,
        format=ScalarType.FLOAT64,
        description="Distance from center of the source to the center of the receiver",
    ),
    StructuredFieldDescriptor(
        name="extended_samples_per_trace",
        offset=136,
        format=ScalarType.UINT32,
        description="Extended number of samples in this trace",
    ),
    StructuredFieldDescriptor(
        name="nanoseconds_to_add",
        offset=140,
        format=ScalarType.INT32,
        description="Nanoseconds to add to Second of minute",
    ),
    StructuredFieldDescriptor(
        name="extended_sample_interval",
        offset=144,
        format=ScalarType.FLOAT64,
        description="Extended, microsecond sample interval",
    ),
    StructuredFieldDescriptor(
        name="cable_or_rec_id",
        offset=152,
        format=ScalarType.INT32,
        description="Cable number for multi-cable acquisition or Recording Device/Sensor ID number",
    ),
    StructuredFieldDescriptor(
        name="num_additional_extended_headers",
        offset=156,
        format=ScalarType.UINT16,
        description="Number of additional trace header extension blocks including this one",
    ),
    StructuredFieldDescriptor(
        name="last_trace_flag",
        offset=158,
        format=ScalarType.INT16,
        description="Last trace flag",
    ),
    StructuredFieldDescriptor(
        name="extended_x_coord_ensemble",
        offset=160,
        format=ScalarType.FLOAT64,
        description="Extended X coordinate of ensemble (CDP) position of this trace",
    ),
    StructuredFieldDescriptor(
        name="extended_y_coord_ensemble",
        offset=168,
        format=ScalarType.FLOAT64,
        description="Extended Y coordinate of ensemble (CDP) position of this trace",
    ),
    StructuredFieldDescriptor(
        name="trace_header_name",
        offset=232,
        format=ScalarType.STRING8,
        description="Eight character trace header name",
    ),
]


rev2_textual_file_header = TextHeaderDescriptor(
    rows=40,
    cols=80,
    offset=0,
    encoding=TextHeaderEncoding.EBCDIC,
    format=ScalarType.UINT8,  # noqa: A003
)


rev2_binary_file_header = StructuredDataTypeDescriptor(
    fields=BINARY_FILE_HEADER_FIELDS_REV2,
    item_size=400,
    offset=3200,
)


rev2_extended_text_header = TextHeaderDescriptor(
    rows=40,
    cols=80,
    offset=3600,
    encoding=TextHeaderEncoding.EBCDIC,
    format=ScalarType.UINT8,  # noqa: A003
)


rev2_trace_header = StructuredDataTypeDescriptor(
    fields=TRACE_HEADER_FIELDS_REV2,
    item_size=240,
)

rev2_extended_trace_header = StructuredDataTypeDescriptor(
    fields=EXTENDED_TRACE_HEADER_FIELDS_REV2,
    item_size=240,
)

rev2_trace_data = TraceSampleDescriptor(
    format=ScalarType.IBM32,  # noqa: A003
)


rev2_trace = TraceDescriptor(
    header_descriptor=rev2_trace_header,
    extended_header_descriptor=rev2_extended_trace_header,
    sample_descriptor=rev2_trace_data,
)


rev2_segy = SegyDescriptor(
    segy_standard=SegyStandard.REV2,
    text_file_header=rev2_textual_file_header,
    binary_file_header=rev2_binary_file_header,
    extended_text_header=rev2_extended_text_header,
    trace=rev2_trace,
    endianness=Endianness.BIG,
)
