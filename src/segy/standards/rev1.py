"""SEG-Y Revision 1 Specification."""

from segy.schema import BinaryHeaderDescriptor
from segy.schema import HeaderFieldDescriptor
from segy.schema import ScalarType
from segy.schema import SegyDescriptor
from segy.schema import SegyStandard
from segy.schema import TextHeaderDescriptor
from segy.schema import TextHeaderEncoding
from segy.schema import TraceDataDescriptor
from segy.schema import TraceDescriptor
from segy.schema import TraceHeaderDescriptor
from segy.schema.data_type import Endianness
from segy.schema.data_type import StructuredFieldDescriptor

BINARY_FILE_HEADER_FIELDS_REV1 = [
    HeaderFieldDescriptor(
        name="job_id",
        offset=0,
        format=ScalarType.INT32,
        endianness=Endianness.BIG,
        description="Job Identification Number",
    ),
    HeaderFieldDescriptor(
        name="line_no",
        offset=4,
        format=ScalarType.INT32,
        endianness=Endianness.BIG,
        description="Line Number",
    ),
    HeaderFieldDescriptor(
        name="reel_no",
        offset=8,
        format=ScalarType.INT32,
        endianness=Endianness.BIG,
        description="Reel Number",
    ),
    HeaderFieldDescriptor(
        name="data_traces_ensemble",
        offset=12,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Number of Data Traces per Ensemble",
    ),
    HeaderFieldDescriptor(
        name="aux_traces_ensemble",
        offset=14,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Number of Auxiliary Traces per Ensemble",
    ),
    HeaderFieldDescriptor(
        name="sample_interval",
        offset=16,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Sample Interval",
    ),
    HeaderFieldDescriptor(
        name="sample_interval_orig",
        offset=18,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Sample Interval of Original Field Recording",
    ),
    HeaderFieldDescriptor(
        name="samples_per_trace",
        offset=20,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Number of Samples per Data Trace",
    ),
    HeaderFieldDescriptor(
        name="samples_per_trace_orig",
        offset=22,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Number of Samples per Data Trace for Original Field Recording",
    ),
    HeaderFieldDescriptor(
        name="data_sample_format",
        offset=24,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Data Sample Format Code",
    ),
    HeaderFieldDescriptor(
        name="ensemble_fold",
        offset=26,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Ensemble Fold",
    ),
    HeaderFieldDescriptor(
        name="trace_sorting",
        offset=28,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Trace Sorting Code",
    ),
    HeaderFieldDescriptor(
        name="vertical_sum",
        offset=30,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Vertical Sum Code",
    ),
    HeaderFieldDescriptor(
        name="sweep_freq_start",
        offset=32,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Sweep Frequency at Start",
    ),
    HeaderFieldDescriptor(
        name="sweep_freq_end",
        offset=34,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Sweep Frequency at End",
    ),
    HeaderFieldDescriptor(
        name="sweep_length",
        offset=36,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Sweep Length",
    ),
    HeaderFieldDescriptor(
        name="sweep_type",
        offset=38,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Sweep Type Code",
    ),
    HeaderFieldDescriptor(
        name="sweep_trace_no",
        offset=40,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Trace Number of Sweep Channel",
    ),
    HeaderFieldDescriptor(
        name="sweep_taper_start",
        offset=42,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Sweep Trace Taper Length at Start",
    ),
    HeaderFieldDescriptor(
        name="sweep_taper_end",
        offset=44,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Sweep Trace Taper Length at End",
    ),
    HeaderFieldDescriptor(
        name="taper_type",
        offset=46,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Taper Type",
    ),
    HeaderFieldDescriptor(
        name="correlated_traces",
        offset=48,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Correlated Data Traces",
    ),
    HeaderFieldDescriptor(
        name="binary_gain",
        offset=50,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Binary Gain Recovered",
    ),
    HeaderFieldDescriptor(
        name="amp_recovery_method",
        offset=52,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Amplitude Recovery Method",
    ),
    HeaderFieldDescriptor(
        name="measurement_system",
        offset=54,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Measurement System",
    ),
    HeaderFieldDescriptor(
        name="impulse_signal_polarity",
        offset=56,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Impulse Signal Polarity",
    ),
    HeaderFieldDescriptor(
        name="vibratory_polarity",
        offset=58,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Vibratory Polarity Code",
    ),
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


TRACE_HEADER_FIELDS_REV1 = [
    HeaderFieldDescriptor(
        name="trace_seq_line",
        offset=0,
        format=ScalarType.INT32,
        endianness=Endianness.BIG,
        description="Trace Sequence Number within Line",
    ),
    HeaderFieldDescriptor(
        name="trace_seq_file",
        offset=4,
        format=ScalarType.INT32,
        endianness=Endianness.BIG,
        description="Trace Sequence Number within File",
    ),
    HeaderFieldDescriptor(
        name="field_rec_no",
        offset=8,
        format=ScalarType.INT32,
        endianness=Endianness.BIG,
        description="Original Field Record Number",
    ),
    HeaderFieldDescriptor(
        name="trace_no_field_rec",
        offset=12,
        format=ScalarType.INT32,
        endianness=Endianness.BIG,
        description="Trace Number within the Field Record",
    ),
    HeaderFieldDescriptor(
        name="energy_src_pt",
        offset=16,
        format=ScalarType.INT32,
        endianness=Endianness.BIG,
        description="Energy Source Point Number",
    ),
    HeaderFieldDescriptor(
        name="cdp_ens_no",
        offset=20,
        format=ScalarType.INT32,
        endianness=Endianness.BIG,
        description="Ensemble Number (CDP, CMP, etc.)",
    ),
    HeaderFieldDescriptor(
        name="trace_no_ens",
        offset=24,
        format=ScalarType.INT32,
        endianness=Endianness.BIG,
        description="Trace Number within the Ensemble",
    ),
    HeaderFieldDescriptor(
        name="trace_id",
        offset=28,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Trace Identification Code",
    ),
    HeaderFieldDescriptor(
        name="vert_sum",
        offset=30,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Number of Vertically Stacked Traces",
    ),
    HeaderFieldDescriptor(
        name="horiz_stack",
        offset=32,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Number of Horizontally Stacked Traces",
    ),
    HeaderFieldDescriptor(
        name="data_use",
        offset=34,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Data Use",
    ),
    HeaderFieldDescriptor(
        name="dist_src_to_rec",
        offset=36,
        format=ScalarType.INT32,
        endianness=Endianness.BIG,
        description="Distance from Source Point to Receiver Group",
    ),
    HeaderFieldDescriptor(
        name="rec_elev",
        offset=40,
        format=ScalarType.INT32,
        endianness=Endianness.BIG,
        description="Receiver Group Elevation",
    ),
    HeaderFieldDescriptor(
        name="src_elev",
        offset=44,
        format=ScalarType.INT32,
        endianness=Endianness.BIG,
        description="Source Elevation",
    ),
    HeaderFieldDescriptor(
        name="src_depth",
        offset=48,
        format=ScalarType.INT32,
        endianness=Endianness.BIG,
        description="Source Depth",
    ),
    HeaderFieldDescriptor(
        name="datum_elev_rec",
        offset=52,
        format=ScalarType.INT32,
        endianness=Endianness.BIG,
        description="Datum Elevation at Receiver Group",
    ),
    HeaderFieldDescriptor(
        name="datum_elev_src",
        offset=56,
        format=ScalarType.INT32,
        endianness=Endianness.BIG,
        description="Datum Elevation at Source",
    ),
    HeaderFieldDescriptor(
        name="water_depth_src",
        offset=60,
        format=ScalarType.INT32,
        endianness=Endianness.BIG,
        description="Water Depth at Source",
    ),
    HeaderFieldDescriptor(
        name="water_depth_rec",
        offset=64,
        format=ScalarType.INT32,
        endianness=Endianness.BIG,
        description="Water Depth at Receiver Group",
    ),
    HeaderFieldDescriptor(
        name="scalar_apply_elev",
        offset=68,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Scalar to be applied to all elevations and depths",
    ),
    HeaderFieldDescriptor(
        name="scalar_apply_coords",
        offset=70,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Scalar to be applied to all coordinates",
    ),
    HeaderFieldDescriptor(
        name="src_x",
        offset=72,
        format=ScalarType.INT32,
        endianness=Endianness.BIG,
        description="Source X coordinate",
    ),
    HeaderFieldDescriptor(
        name="src_y",
        offset=76,
        format=ScalarType.INT32,
        endianness=Endianness.BIG,
        description="Source Y coordinate",
    ),
    HeaderFieldDescriptor(
        name="rec_x",
        offset=80,
        format=ScalarType.INT32,
        endianness=Endianness.BIG,
        description="Receiver X coordinate",
    ),
    HeaderFieldDescriptor(
        name="rec_y",
        offset=84,
        format=ScalarType.INT32,
        endianness=Endianness.BIG,
        description="Receiver Y coordinate",
    ),
    HeaderFieldDescriptor(
        name="coord_units",
        offset=88,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Coordinate units",
    ),
    HeaderFieldDescriptor(
        name="weathering_vel",
        offset=90,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Weathering Velocity",
    ),
    HeaderFieldDescriptor(
        name="subweathering_vel",
        offset=92,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Subweathering Velocity",
    ),
    HeaderFieldDescriptor(
        name="uphole_time_src",
        offset=94,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Uphole Time at Source",
    ),
    HeaderFieldDescriptor(
        name="uphole_time_rec",
        offset=96,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Uphole Time at Receiver",
    ),
    HeaderFieldDescriptor(
        name="src_static_corr",
        offset=98,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Source Static Correction",
    ),
    HeaderFieldDescriptor(
        name="rec_static_corr",
        offset=100,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Receiver Static Correction",
    ),
    HeaderFieldDescriptor(
        name="total_static",
        offset=102,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Total Static Applied",
    ),
    HeaderFieldDescriptor(
        name="lag_time_a",
        offset=104,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Lag Time A",
    ),
    HeaderFieldDescriptor(
        name="lag_time_b",
        offset=106,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Lag Time B",
    ),
    HeaderFieldDescriptor(
        name="delay_rec_time",
        offset=108,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Delay Recording Time",
    ),
    HeaderFieldDescriptor(
        name="mute_start",
        offset=110,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Start Time of Mute",
    ),
    HeaderFieldDescriptor(
        name="mute_end",
        offset=112,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="End Time of Mute",
    ),
    HeaderFieldDescriptor(
        name="nsamples",
        offset=114,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Number of Samples in this Trace",
    ),
    HeaderFieldDescriptor(
        name="sample_interval",
        offset=116,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Sample Interval for this Trace",
    ),
    HeaderFieldDescriptor(
        name="gain_type",
        offset=118,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Gain Type of Field Instruments",
    ),
    HeaderFieldDescriptor(
        name="instrument_gain",
        offset=120,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Instrument Gain Constant",
    ),
    HeaderFieldDescriptor(
        name="instrument_early_gain",
        offset=122,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Instrument Early Gain",
    ),
    HeaderFieldDescriptor(
        name="correlated",
        offset=124,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Correlated",
    ),
    HeaderFieldDescriptor(
        name="sweep_freq_start",
        offset=126,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Sweep Frequency at Start",
    ),
    HeaderFieldDescriptor(
        name="sweep_freq_end",
        offset=128,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Sweep Frequency at End",
    ),
    HeaderFieldDescriptor(
        name="sweep_length",
        offset=130,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Sweep Length",
    ),
    HeaderFieldDescriptor(
        name="sweep_type",
        offset=132,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Sweep Type",
    ),
    HeaderFieldDescriptor(
        name="sweep_trace_taper_start",
        offset=134,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Sweep Trace Taper Length at Start",
    ),
    HeaderFieldDescriptor(
        name="sweep_trace_taper_end",
        offset=136,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Sweep Trace Taper Length at End",
    ),
    HeaderFieldDescriptor(
        name="taper_type",
        offset=138,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Taper Type",
    ),
    HeaderFieldDescriptor(
        name="alias_filter_freq",
        offset=140,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Alias Filter Frequency",
    ),
    HeaderFieldDescriptor(
        name="alias_filter_slope",
        offset=142,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Alias Filter Slope",
    ),
    HeaderFieldDescriptor(
        name="notch_filter_freq",
        offset=144,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Notch Filter Frequency",
    ),
    HeaderFieldDescriptor(
        name="notch_filter_slope",
        offset=146,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Notch Filter Slope",
    ),
    HeaderFieldDescriptor(
        name="low_cut_freq",
        offset=148,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Low Cut Frequency",
    ),
    HeaderFieldDescriptor(
        name="high_cut_freq",
        offset=150,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="High Cut Frequency",
    ),
    HeaderFieldDescriptor(
        name="low_cut_slope",
        offset=152,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Low Cut Slope",
    ),
    HeaderFieldDescriptor(
        name="high_cut_slope",
        offset=154,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="High Cut Slope",
    ),
    HeaderFieldDescriptor(
        name="year",
        offset=156,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Year Data Recorded",
    ),
    HeaderFieldDescriptor(
        name="day",
        offset=158,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Day of Year",
    ),
    HeaderFieldDescriptor(
        name="hour",
        offset=160,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Hour of Day",
    ),
    HeaderFieldDescriptor(
        name="minute",
        offset=162,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Minute of Hour",
    ),
    HeaderFieldDescriptor(
        name="second",
        offset=164,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Second of Minute",
    ),
    HeaderFieldDescriptor(
        name="time_basis_code",
        offset=166,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Time Basis Code",
    ),
    HeaderFieldDescriptor(
        name="trace_weighting_factor",
        offset=168,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Trace Weighting Factor",
    ),
    HeaderFieldDescriptor(
        name="geophone_group_no_roll1",
        offset=170,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Geophone Group Number of Roll Switch Position One",
    ),
    HeaderFieldDescriptor(
        name="geophone_group_no_first_trace",
        offset=172,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Geophone Group Number of Trace Number One within Original Field Record",  # noqa: E501
    ),
    HeaderFieldDescriptor(
        name="geophone_group_no_last_trace",
        offset=174,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Geophone Group Number of Last Trace within Original Field Record",  # noqa: E501
    ),
    HeaderFieldDescriptor(
        name="gap_size",
        offset=176,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Gap Size (total number of groups dropped)",
    ),
    HeaderFieldDescriptor(
        name="over_travel",
        offset=178,
        format=ScalarType.INT16,
        endianness=Endianness.BIG,
        description="Over Travel Associated with Taper",
    ),
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

    rows: int = 40
    cols: int = 80
    offset: int = 0
    encoding: TextHeaderEncoding = TextHeaderEncoding.EBCDIC
    format: ScalarType = ScalarType.UINT8  # noqa: A003


class ExtendedTextualHeaderDescriptorRev1(TextHeaderDescriptor):
    """Extended text header spec with SEG-Y Rev1 defaults."""

    rows: int = 40
    cols: int = 80
    offset: int = 3600
    encoding: TextHeaderEncoding = TextHeaderEncoding.EBCDIC
    format: ScalarType = ScalarType.UINT8  # noqa: A003


class BinaryHeaderDescriptorRev1(BinaryHeaderDescriptor):
    """Binary file header spec with SEG-Y Rev1 defaults."""

    fields: list[StructuredFieldDescriptor] = BINARY_FILE_HEADER_FIELDS_REV1
    item_size: int = 400
    offset: int = 3200


class TraceHeaderDescriptorRev1(TraceHeaderDescriptor):
    """Trace header spec with SEG-Y Rev1 defaults."""

    fields: list[StructuredFieldDescriptor] = TRACE_HEADER_FIELDS_REV1
    item_size: int = 240


class TraceDataDescriptorRev1(TraceDataDescriptor):
    """Trace data spec with SEG-Y Rev1 defaults."""

    format: ScalarType = ScalarType.IBM32  # noqa: A003
    endianness: Endianness = Endianness.BIG


class TraceDescriptorRev1(TraceDescriptor):
    """Trace spec with SEG-Y Rev1 defaults."""

    header_descriptor: TraceHeaderDescriptor = TraceHeaderDescriptorRev1()
    data_descriptor: TraceDataDescriptor = TraceDataDescriptorRev1()


class SegyDescriptorRev1(SegyDescriptor):
    """SEG-Y file spec with SEG-Y Rev1 defaults."""

    segy_standard: SegyStandard = SegyStandard.REV1
    text_file_header: TextHeaderDescriptor = TextualFileHeaderDescriptorRev1()
    binary_file_header: BinaryHeaderDescriptor = BinaryHeaderDescriptorRev1()
    extended_text_header: TextHeaderDescriptor = ExtendedTextualHeaderDescriptorRev1()
    trace: TraceDescriptor = TraceDescriptorRev1()
