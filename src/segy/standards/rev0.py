"""SEG-Y Revision 0 Specification."""

from segy.schema import ScalarType
from segy.schema import SegyDescriptor
from segy.schema import SegyStandard
from segy.schema import TextHeaderDescriptor
from segy.schema import TextHeaderEncoding
from segy.schema import TraceDescriptor
from segy.schema import TraceSampleDescriptor
from segy.schema.data_type import Endianness
from segy.schema.data_type import StructuredDataTypeDescriptor
from segy.schema.data_type import StructuredFieldDescriptor

BINARY_FILE_HEADER_FIELDS_REV0 = [
    StructuredFieldDescriptor(
        name="job_id",
        byte=1,
        format=ScalarType.INT32,
        description="Job Identification Number",
    ),
    StructuredFieldDescriptor(
        name="line_no",
        byte=5,
        format=ScalarType.INT32,
        description="Line Number",
    ),
    StructuredFieldDescriptor(
        name="reel_no",
        byte=9,
        format=ScalarType.INT32,
        description="Reel Number",
    ),
    StructuredFieldDescriptor(
        name="data_traces_ensemble",
        byte=13,
        format=ScalarType.INT16,
        description="Number of Data Traces per Ensemble",
    ),
    StructuredFieldDescriptor(
        name="aux_traces_ensemble",
        byte=15,
        format=ScalarType.INT16,
        description="Number of Auxiliary Traces per Ensemble",
    ),
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
        byte=23,
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
        name="ensemble_fold",
        byte=27,
        format=ScalarType.INT16,
        description="Ensemble Fold",
    ),
    StructuredFieldDescriptor(
        name="trace_sorting",
        byte=29,
        format=ScalarType.INT16,
        description="Trace Sorting Code",
    ),
    StructuredFieldDescriptor(
        name="vertical_sum",
        byte=31,
        format=ScalarType.INT16,
        description="Vertical Sum Code",
    ),
    StructuredFieldDescriptor(
        name="sweep_freq_start",
        byte=33,
        format=ScalarType.INT16,
        description="Sweep Frequency at Start",
    ),
    StructuredFieldDescriptor(
        name="sweep_freq_end",
        byte=35,
        format=ScalarType.INT16,
        description="Sweep Frequency at End",
    ),
    StructuredFieldDescriptor(
        name="sweep_length",
        byte=37,
        format=ScalarType.INT16,
        description="Sweep Length",
    ),
    StructuredFieldDescriptor(
        name="sweep_type",
        byte=39,
        format=ScalarType.INT16,
        description="Sweep Type Code",
    ),
    StructuredFieldDescriptor(
        name="sweep_trace_no",
        byte=41,
        format=ScalarType.INT16,
        description="Trace Number of Sweep Channel",
    ),
    StructuredFieldDescriptor(
        name="sweep_taper_start",
        byte=43,
        format=ScalarType.INT16,
        description="Sweep Trace Taper Length at Start",
    ),
    StructuredFieldDescriptor(
        name="sweep_taper_end",
        byte=45,
        format=ScalarType.INT16,
        description="Sweep Trace Taper Length at End",
    ),
    StructuredFieldDescriptor(
        name="taper_type",
        byte=47,
        format=ScalarType.INT16,
        description="Taper Type",
    ),
    StructuredFieldDescriptor(
        name="correlated_traces",
        byte=49,
        format=ScalarType.INT16,
        description="Correlated Data Traces",
    ),
    StructuredFieldDescriptor(
        name="binary_gain",
        byte=51,
        format=ScalarType.INT16,
        description="Binary Gain Recovered",
    ),
    StructuredFieldDescriptor(
        name="amp_recovery_method",
        byte=53,
        format=ScalarType.INT16,
        description="Amplitude Recovery Method",
    ),
    StructuredFieldDescriptor(
        name="measurement_system",
        byte=55,
        format=ScalarType.INT16,
        description="Measurement System",
    ),
    StructuredFieldDescriptor(
        name="impulse_signal_polarity",
        byte=57,
        format=ScalarType.INT16,
        description="Impulse Signal Polarity",
    ),
    StructuredFieldDescriptor(
        name="vibratory_polarity",
        byte=59,
        format=ScalarType.INT16,
        description="Vibratory Polarity Code",
    ),
]


TRACE_HEADER_FIELDS_REV0 = [
    StructuredFieldDescriptor(
        name="trace_seq_line",
        byte=1,
        format=ScalarType.INT32,
        description="Trace Sequence Number within Line",
    ),
    StructuredFieldDescriptor(
        name="trace_seq_file",
        byte=5,
        format=ScalarType.INT32,
        description="Trace Sequence Number within File",
    ),
    StructuredFieldDescriptor(
        name="field_rec_no",
        byte=9,
        format=ScalarType.INT32,
        description="Original Field Record Number",
    ),
    StructuredFieldDescriptor(
        name="trace_no_field_rec",
        byte=13,
        format=ScalarType.INT32,
        description="Trace Number within the Field Record",
    ),
    StructuredFieldDescriptor(
        name="energy_src_pt",
        byte=17,
        format=ScalarType.INT32,
        description="Energy Source Point Number",
    ),
    StructuredFieldDescriptor(
        name="ensemble_no",
        byte=21,
        format=ScalarType.INT32,
        description="Ensemble Number (CDP, CMP, etc.)",
    ),
    StructuredFieldDescriptor(
        name="trace_no_ens",
        byte=25,
        format=ScalarType.INT32,
        description="Trace Number within the Ensemble",
    ),
    StructuredFieldDescriptor(
        name="trace_id",
        byte=29,
        format=ScalarType.INT16,
        description="Trace Identification Code",
    ),
    StructuredFieldDescriptor(
        name="vertical_sum",
        byte=31,
        format=ScalarType.INT16,
        description="Number of Vertically Stacked Traces",
    ),
    StructuredFieldDescriptor(
        name="horiz_stack",
        byte=33,
        format=ScalarType.INT16,
        description="Number of Horizontally Stacked Traces",
    ),
    StructuredFieldDescriptor(
        name="data_use",
        byte=35,
        format=ScalarType.INT16,
        description="Data Use",
    ),
    StructuredFieldDescriptor(
        name="dist_src_to_rec",
        byte=37,
        format=ScalarType.INT32,
        description="Distance from Source Point to Receiver Group",
    ),
    StructuredFieldDescriptor(
        name="rec_elev",
        byte=41,
        format=ScalarType.INT32,
        description="Receiver Group Elevation",
    ),
    StructuredFieldDescriptor(
        name="src_elev",
        byte=45,
        format=ScalarType.INT32,
        description="Source Elevation",
    ),
    StructuredFieldDescriptor(
        name="src_depth",
        byte=49,
        format=ScalarType.INT32,
        description="Source Depth",
    ),
    StructuredFieldDescriptor(
        name="datum_elev_rec",
        byte=53,
        format=ScalarType.INT32,
        description="Datum Elevation at Receiver Group",
    ),
    StructuredFieldDescriptor(
        name="datum_elev_src",
        byte=57,
        format=ScalarType.INT32,
        description="Datum Elevation at Source",
    ),
    StructuredFieldDescriptor(
        name="water_depth_src",
        byte=61,
        format=ScalarType.INT32,
        description="Water Depth at Source",
    ),
    StructuredFieldDescriptor(
        name="water_depth_rec",
        byte=65,
        format=ScalarType.INT32,
        description="Water Depth at Receiver Group",
    ),
    StructuredFieldDescriptor(
        name="scalar_apply_elev",
        byte=69,
        format=ScalarType.INT16,
        description="Scalar to be applied to all elevations and depths",
    ),
    StructuredFieldDescriptor(
        name="scalar_apply_coords",
        byte=71,
        format=ScalarType.INT16,
        description="Scalar to be applied to all coordinates",
    ),
    StructuredFieldDescriptor(
        name="src_x",
        byte=73,
        format=ScalarType.INT32,
        description="Source X coordinate",
    ),
    StructuredFieldDescriptor(
        name="src_y",
        byte=77,
        format=ScalarType.INT32,
        description="Source Y coordinate",
    ),
    StructuredFieldDescriptor(
        name="rec_x",
        byte=81,
        format=ScalarType.INT32,
        description="Receiver X coordinate",
    ),
    StructuredFieldDescriptor(
        name="rec_y",
        byte=85,
        format=ScalarType.INT32,
        description="Receiver Y coordinate",
    ),
    StructuredFieldDescriptor(
        name="coord_units",
        byte=89,
        format=ScalarType.INT16,
        description="Coordinate units",
    ),
    StructuredFieldDescriptor(
        name="weathering_vel",
        byte=91,
        format=ScalarType.INT16,
        description="Weathering Velocity",
    ),
    StructuredFieldDescriptor(
        name="subweathering_vel",
        byte=93,
        format=ScalarType.INT16,
        description="Subweathering Velocity",
    ),
    StructuredFieldDescriptor(
        name="uphole_time_src",
        byte=95,
        format=ScalarType.INT16,
        description="Uphole Time at Source",
    ),
    StructuredFieldDescriptor(
        name="uphole_time_rec",
        byte=97,
        format=ScalarType.INT16,
        description="Uphole Time at Receiver",
    ),
    StructuredFieldDescriptor(
        name="src_static_corr",
        byte=99,
        format=ScalarType.INT16,
        description="Source Static Correction",
    ),
    StructuredFieldDescriptor(
        name="rec_static_corr",
        byte=101,
        format=ScalarType.INT16,
        description="Receiver Static Correction",
    ),
    StructuredFieldDescriptor(
        name="total_static",
        byte=103,
        format=ScalarType.INT16,
        description="Total Static Applied",
    ),
    StructuredFieldDescriptor(
        name="lag_time_a",
        byte=105,
        format=ScalarType.INT16,
        description="Lag Time A",
    ),
    StructuredFieldDescriptor(
        name="lag_time_b",
        byte=107,
        format=ScalarType.INT16,
        description="Lag Time B",
    ),
    StructuredFieldDescriptor(
        name="delay_rec_time",
        byte=109,
        format=ScalarType.INT16,
        description="Delay Recording Time",
    ),
    StructuredFieldDescriptor(
        name="mute_start",
        byte=111,
        format=ScalarType.INT16,
        description="Start Time of Mute",
    ),
    StructuredFieldDescriptor(
        name="mute_end",
        byte=113,
        format=ScalarType.INT16,
        description="End Time of Mute",
    ),
    StructuredFieldDescriptor(
        name="samples_per_trace",
        byte=115,
        format=ScalarType.INT16,
        description="Number of Samples in this Trace",
    ),
    StructuredFieldDescriptor(
        name="sample_interval",
        byte=117,
        format=ScalarType.INT16,
        description="Sample Interval for this Trace",
    ),
    StructuredFieldDescriptor(
        name="gain_type",
        byte=119,
        format=ScalarType.INT16,
        description="Gain Type of Field Instruments",
    ),
    StructuredFieldDescriptor(
        name="instrument_gain",
        byte=121,
        format=ScalarType.INT16,
        description="Instrument Gain Constant",
    ),
    StructuredFieldDescriptor(
        name="instrument_early_gain",
        byte=123,
        format=ScalarType.INT16,
        description="Instrument Early Gain",
    ),
    StructuredFieldDescriptor(
        name="correlated",
        byte=125,
        format=ScalarType.INT16,
        description="Correlated",
    ),
    StructuredFieldDescriptor(
        name="sweep_freq_start",
        byte=127,
        format=ScalarType.INT16,
        description="Sweep Frequency at Start",
    ),
    StructuredFieldDescriptor(
        name="sweep_freq_end",
        byte=129,
        format=ScalarType.INT16,
        description="Sweep Frequency at End",
    ),
    StructuredFieldDescriptor(
        name="sweep_length",
        byte=131,
        format=ScalarType.INT16,
        description="Sweep Length",
    ),
    StructuredFieldDescriptor(
        name="sweep_type",
        byte=133,
        format=ScalarType.INT16,
        description="Sweep Type",
    ),
    StructuredFieldDescriptor(
        name="sweep_trace_taper_start",
        byte=135,
        format=ScalarType.INT16,
        description="Sweep Trace Taper Length at Start",
    ),
    StructuredFieldDescriptor(
        name="sweep_trace_taper_end",
        byte=137,
        format=ScalarType.INT16,
        description="Sweep Trace Taper Length at End",
    ),
    StructuredFieldDescriptor(
        name="taper_type",
        byte=139,
        format=ScalarType.INT16,
        description="Taper Type",
    ),
    StructuredFieldDescriptor(
        name="alias_filter_freq",
        byte=141,
        format=ScalarType.INT16,
        description="Alias Filter Frequency",
    ),
    StructuredFieldDescriptor(
        name="alias_filter_slope",
        byte=143,
        format=ScalarType.INT16,
        description="Alias Filter Slope",
    ),
    StructuredFieldDescriptor(
        name="notch_filter_freq",
        byte=145,
        format=ScalarType.INT16,
        description="Notch Filter Frequency",
    ),
    StructuredFieldDescriptor(
        name="notch_filter_slope",
        byte=147,
        format=ScalarType.INT16,
        description="Notch Filter Slope",
    ),
    StructuredFieldDescriptor(
        name="low_cut_freq",
        byte=149,
        format=ScalarType.INT16,
        description="Low Cut Frequency",
    ),
    StructuredFieldDescriptor(
        name="high_cut_freq",
        byte=151,
        format=ScalarType.INT16,
        description="High Cut Frequency",
    ),
    StructuredFieldDescriptor(
        name="low_cut_slope",
        byte=153,
        format=ScalarType.INT16,
        description="Low Cut Slope",
    ),
    StructuredFieldDescriptor(
        name="high_cut_slope",
        byte=155,
        format=ScalarType.INT16,
        description="High Cut Slope",
    ),
    StructuredFieldDescriptor(
        name="year",
        byte=157,
        format=ScalarType.INT16,
        description="Year Data Recorded",
    ),
    StructuredFieldDescriptor(
        name="day",
        byte=159,
        format=ScalarType.INT16,
        description="Day of Year",
    ),
    StructuredFieldDescriptor(
        name="hour",
        byte=161,
        format=ScalarType.INT16,
        description="Hour of Day",
    ),
    StructuredFieldDescriptor(
        name="minute",
        byte=163,
        format=ScalarType.INT16,
        description="Minute of Hour",
    ),
    StructuredFieldDescriptor(
        name="second",
        byte=165,
        format=ScalarType.INT16,
        description="Second of Minute",
    ),
    StructuredFieldDescriptor(
        name="time_basis_code",
        byte=167,
        format=ScalarType.INT16,
        description="Time Basis Code",
    ),
    StructuredFieldDescriptor(
        name="trace_weighting_factor",
        byte=169,
        format=ScalarType.INT16,
        description="Trace Weighting Factor",
    ),
    StructuredFieldDescriptor(
        name="geophone_group_no_roll1",
        byte=171,
        format=ScalarType.INT16,
        description="Geophone Group Number of Roll Switch Position One",
    ),
    StructuredFieldDescriptor(
        name="geophone_group_no_first_trace",
        byte=173,
        format=ScalarType.INT16,
        description="Geophone Group Number of Trace Number One within Original Field Record",  # noqa: E501
    ),
    StructuredFieldDescriptor(
        name="geophone_group_no_last_trace",
        byte=175,
        format=ScalarType.INT16,
        description="Geophone Group Number of Last Trace within Original Field Record",  # noqa: E501
    ),
    StructuredFieldDescriptor(
        name="gap_size",
        byte=177,
        format=ScalarType.INT16,
        description="Gap Size (total number of groups dropped)",
    ),
    StructuredFieldDescriptor(
        name="over_travel",
        byte=179,
        format=ScalarType.INT16,
        description="Over Travel Associated with Taper",
    ),
]


rev0_textual_file_header = TextHeaderDescriptor(
    rows=40,
    cols=80,
    offset=0,
    encoding=TextHeaderEncoding.EBCDIC,
    format=ScalarType.UINT8,  # noqa: A003
)


rev0_binary_file_header = StructuredDataTypeDescriptor(
    fields=BINARY_FILE_HEADER_FIELDS_REV0,
    item_size=400,
    offset=3200,
)


rev0_trace_header = StructuredDataTypeDescriptor(
    fields=TRACE_HEADER_FIELDS_REV0,
    item_size=240,
)


rev0_trace_data = TraceSampleDescriptor(
    format=ScalarType.IBM32,  # noqa: A003
)


rev0_trace = TraceDescriptor(
    header_descriptor=rev0_trace_header,
    sample_descriptor=rev0_trace_data,
)


rev0_segy = SegyDescriptor(
    segy_standard=SegyStandard.REV0,
    text_file_header=rev0_textual_file_header,
    binary_file_header=rev0_binary_file_header,
    trace=rev0_trace,
    endianness=Endianness.BIG,
)
