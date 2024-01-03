"""SEG-Y Revision 0 Specification."""

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

BINARY_FILE_HEADER_FIELDS_REV0 = [
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
]

TRACE_HEADER_FIELDS_REV0 = [
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
]


class TextualFileHeaderDescriptorRev0(TextHeaderDescriptor):
    """Textual file header spec with SEG-Y Rev0 defaults."""

    description: str = "3200-byte textual file header with 40 lines of text."
    rows: int = 40
    cols: int = 80
    offset: int = 0
    encoding: TextHeaderEncoding = TextHeaderEncoding.EBCDIC
    format: ScalarType = ScalarType.UINT8  # noqa: A003


class BinaryHeaderDescriptorRev0(BinaryHeaderDescriptor):
    """Binary file header spec with SEG-Y Rev0 defaults."""

    description: str = "400-byte binary file header with structured fields."
    fields: list[HeaderFieldDescriptor] = BINARY_FILE_HEADER_FIELDS_REV0
    item_size: int = 400
    offset: int = 3200


class TraceHeaderDescriptorRev0(TraceHeaderDescriptor):
    """Trace header spec with SEG-Y Rev0 defaults."""

    description: str = "240-byte trace header with structured fields."
    fields: list[HeaderFieldDescriptor] = TRACE_HEADER_FIELDS_REV0
    item_size: int = 240


class TraceDataDescriptorRev0(TraceDataDescriptor):
    """Trace data spec with SEG-Y Rev0 defaults."""

    description: str = "Trace data with given format and sample count."
    format: ScalarType = ScalarType.IBM32  # noqa: A003
    endianness: Endianness = Endianness.BIG


class TraceDescriptorRev0(TraceDescriptor):
    """Trace spec with SEG-Y Rev1 defaults."""

    description: str = "Trace spec with header and data information."
    header_descriptor: TraceHeaderDescriptor = TraceHeaderDescriptorRev0()
    data_descriptor: TraceDataDescriptor = TraceDataDescriptorRev0()


class SegyDescriptorRev0(SegyDescriptor):
    """SEG-Y file spec with SEG-Y Rev1 defaults."""

    segy_standard: SegyStandard = SegyStandard.REV0
    text_file_header: TextHeaderDescriptor = TextualFileHeaderDescriptorRev0()
    binary_file_header: BinaryHeaderDescriptor = BinaryHeaderDescriptorRev0()
    trace: TraceDescriptor = TraceDescriptorRev0()
