"""SEG-Y Revision 0 Specification."""

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

BINARY_FILE_HEADER_FIELDS_REV0 = [
    HeaderField(
        name="job_id",
        byte=1,
        format=ScalarType.INT32,
        description="Job Identification Number",
    ),
    HeaderField(
        name="line_no",
        byte=5,
        format=ScalarType.INT32,
        description="Line Number",
    ),
    HeaderField(
        name="reel_no",
        byte=9,
        format=ScalarType.INT32,
        description="Reel Number",
    ),
    HeaderField(
        name="data_traces_ensemble",
        byte=13,
        format=ScalarType.INT16,
        description="Number of Data Traces per Ensemble",
    ),
    HeaderField(
        name="aux_traces_ensemble",
        byte=15,
        format=ScalarType.INT16,
        description="Number of Auxiliary Traces per Ensemble",
    ),
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
        byte=23,
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
        name="ensemble_fold",
        byte=27,
        format=ScalarType.INT16,
        description="Ensemble Fold",
    ),
    HeaderField(
        name="trace_sorting",
        byte=29,
        format=ScalarType.INT16,
        description="Trace Sorting Code",
    ),
    HeaderField(
        name="vertical_sum",
        byte=31,
        format=ScalarType.INT16,
        description="Vertical Sum Code",
    ),
    HeaderField(
        name="sweep_freq_start",
        byte=33,
        format=ScalarType.INT16,
        description="Sweep Frequency at Start",
    ),
    HeaderField(
        name="sweep_freq_end",
        byte=35,
        format=ScalarType.INT16,
        description="Sweep Frequency at End",
    ),
    HeaderField(
        name="sweep_length",
        byte=37,
        format=ScalarType.INT16,
        description="Sweep Length",
    ),
    HeaderField(
        name="sweep_type",
        byte=39,
        format=ScalarType.INT16,
        description="Sweep Type Code",
    ),
    HeaderField(
        name="sweep_trace_no",
        byte=41,
        format=ScalarType.INT16,
        description="Trace Number of Sweep Channel",
    ),
    HeaderField(
        name="sweep_taper_start",
        byte=43,
        format=ScalarType.INT16,
        description="Sweep Trace Taper Length at Start",
    ),
    HeaderField(
        name="sweep_taper_end",
        byte=45,
        format=ScalarType.INT16,
        description="Sweep Trace Taper Length at End",
    ),
    HeaderField(
        name="taper_type",
        byte=47,
        format=ScalarType.INT16,
        description="Taper Type",
    ),
    HeaderField(
        name="correlated_traces",
        byte=49,
        format=ScalarType.INT16,
        description="Correlated Data Traces",
    ),
    HeaderField(
        name="binary_gain",
        byte=51,
        format=ScalarType.INT16,
        description="Binary Gain Recovered",
    ),
    HeaderField(
        name="amp_recovery_method",
        byte=53,
        format=ScalarType.INT16,
        description="Amplitude Recovery Method",
    ),
    HeaderField(
        name="measurement_system",
        byte=55,
        format=ScalarType.INT16,
        description="Measurement System",
    ),
    HeaderField(
        name="impulse_signal_polarity",
        byte=57,
        format=ScalarType.INT16,
        description="Impulse Signal Polarity",
    ),
    HeaderField(
        name="vibratory_polarity",
        byte=59,
        format=ScalarType.INT16,
        description="Vibratory Polarity Code",
    ),
]


TRACE_HEADER_FIELDS_REV0 = [
    HeaderField(
        name="trace_seq_line",
        byte=1,
        format=ScalarType.INT32,
        description="Trace Sequence Number within Line",
    ),
    HeaderField(
        name="trace_seq_file",
        byte=5,
        format=ScalarType.INT32,
        description="Trace Sequence Number within File",
    ),
    HeaderField(
        name="field_rec_no",
        byte=9,
        format=ScalarType.INT32,
        description="Original Field Record Number",
    ),
    HeaderField(
        name="trace_no_field_rec",
        byte=13,
        format=ScalarType.INT32,
        description="Trace Number within the Field Record",
    ),
    HeaderField(
        name="energy_src_pt",
        byte=17,
        format=ScalarType.INT32,
        description="Energy Source Point Number",
    ),
    HeaderField(
        name="ensemble_no",
        byte=21,
        format=ScalarType.INT32,
        description="Ensemble Number (CDP, CMP, etc.)",
    ),
    HeaderField(
        name="trace_no_ens",
        byte=25,
        format=ScalarType.INT32,
        description="Trace Number within the Ensemble",
    ),
    HeaderField(
        name="trace_id",
        byte=29,
        format=ScalarType.INT16,
        description="Trace Identification Code",
    ),
    HeaderField(
        name="vertical_sum",
        byte=31,
        format=ScalarType.INT16,
        description="Number of Vertically Stacked Traces",
    ),
    HeaderField(
        name="horiz_stack",
        byte=33,
        format=ScalarType.INT16,
        description="Number of Horizontally Stacked Traces",
    ),
    HeaderField(
        name="data_use",
        byte=35,
        format=ScalarType.INT16,
        description="Data Use",
    ),
    HeaderField(
        name="dist_src_to_rec",
        byte=37,
        format=ScalarType.INT32,
        description="Distance from Source Point to Receiver Group",
    ),
    HeaderField(
        name="rec_elev",
        byte=41,
        format=ScalarType.INT32,
        description="Receiver Group Elevation",
    ),
    HeaderField(
        name="src_elev",
        byte=45,
        format=ScalarType.INT32,
        description="Source Elevation",
    ),
    HeaderField(
        name="src_depth",
        byte=49,
        format=ScalarType.INT32,
        description="Source Depth",
    ),
    HeaderField(
        name="datum_elev_rec",
        byte=53,
        format=ScalarType.INT32,
        description="Datum Elevation at Receiver Group",
    ),
    HeaderField(
        name="datum_elev_src",
        byte=57,
        format=ScalarType.INT32,
        description="Datum Elevation at Source",
    ),
    HeaderField(
        name="water_depth_src",
        byte=61,
        format=ScalarType.INT32,
        description="Water Depth at Source",
    ),
    HeaderField(
        name="water_depth_rec",
        byte=65,
        format=ScalarType.INT32,
        description="Water Depth at Receiver Group",
    ),
    HeaderField(
        name="scalar_apply_elev",
        byte=69,
        format=ScalarType.INT16,
        description="Scalar to be applied to all elevations and depths",
    ),
    HeaderField(
        name="scalar_apply_coords",
        byte=71,
        format=ScalarType.INT16,
        description="Scalar to be applied to all coordinates",
    ),
    HeaderField(
        name="src_x",
        byte=73,
        format=ScalarType.INT32,
        description="Source X coordinate",
    ),
    HeaderField(
        name="src_y",
        byte=77,
        format=ScalarType.INT32,
        description="Source Y coordinate",
    ),
    HeaderField(
        name="rec_x",
        byte=81,
        format=ScalarType.INT32,
        description="Receiver X coordinate",
    ),
    HeaderField(
        name="rec_y",
        byte=85,
        format=ScalarType.INT32,
        description="Receiver Y coordinate",
    ),
    HeaderField(
        name="coord_units",
        byte=89,
        format=ScalarType.INT16,
        description="Coordinate units",
    ),
    HeaderField(
        name="weathering_vel",
        byte=91,
        format=ScalarType.INT16,
        description="Weathering Velocity",
    ),
    HeaderField(
        name="subweathering_vel",
        byte=93,
        format=ScalarType.INT16,
        description="Subweathering Velocity",
    ),
    HeaderField(
        name="uphole_time_src",
        byte=95,
        format=ScalarType.INT16,
        description="Uphole Time at Source",
    ),
    HeaderField(
        name="uphole_time_rec",
        byte=97,
        format=ScalarType.INT16,
        description="Uphole Time at Receiver",
    ),
    HeaderField(
        name="src_static_corr",
        byte=99,
        format=ScalarType.INT16,
        description="Source Static Correction",
    ),
    HeaderField(
        name="rec_static_corr",
        byte=101,
        format=ScalarType.INT16,
        description="Receiver Static Correction",
    ),
    HeaderField(
        name="total_static",
        byte=103,
        format=ScalarType.INT16,
        description="Total Static Applied",
    ),
    HeaderField(
        name="lag_time_a",
        byte=105,
        format=ScalarType.INT16,
        description="Lag Time A",
    ),
    HeaderField(
        name="lag_time_b",
        byte=107,
        format=ScalarType.INT16,
        description="Lag Time B",
    ),
    HeaderField(
        name="delay_rec_time",
        byte=109,
        format=ScalarType.INT16,
        description="Delay Recording Time",
    ),
    HeaderField(
        name="mute_start",
        byte=111,
        format=ScalarType.INT16,
        description="Start Time of Mute",
    ),
    HeaderField(
        name="mute_end",
        byte=113,
        format=ScalarType.INT16,
        description="End Time of Mute",
    ),
    HeaderField(
        name="samples_per_trace",
        byte=115,
        format=ScalarType.INT16,
        description="Number of Samples in this Trace",
    ),
    HeaderField(
        name="sample_interval",
        byte=117,
        format=ScalarType.INT16,
        description="Sample Interval for this Trace",
    ),
    HeaderField(
        name="gain_type",
        byte=119,
        format=ScalarType.INT16,
        description="Gain Type of Field Instruments",
    ),
    HeaderField(
        name="instrument_gain",
        byte=121,
        format=ScalarType.INT16,
        description="Instrument Gain Constant",
    ),
    HeaderField(
        name="instrument_early_gain",
        byte=123,
        format=ScalarType.INT16,
        description="Instrument Early Gain",
    ),
    HeaderField(
        name="correlated",
        byte=125,
        format=ScalarType.INT16,
        description="Correlated",
    ),
    HeaderField(
        name="sweep_freq_start",
        byte=127,
        format=ScalarType.INT16,
        description="Sweep Frequency at Start",
    ),
    HeaderField(
        name="sweep_freq_end",
        byte=129,
        format=ScalarType.INT16,
        description="Sweep Frequency at End",
    ),
    HeaderField(
        name="sweep_length",
        byte=131,
        format=ScalarType.INT16,
        description="Sweep Length",
    ),
    HeaderField(
        name="sweep_type",
        byte=133,
        format=ScalarType.INT16,
        description="Sweep Type",
    ),
    HeaderField(
        name="sweep_trace_taper_start",
        byte=135,
        format=ScalarType.INT16,
        description="Sweep Trace Taper Length at Start",
    ),
    HeaderField(
        name="sweep_trace_taper_end",
        byte=137,
        format=ScalarType.INT16,
        description="Sweep Trace Taper Length at End",
    ),
    HeaderField(
        name="taper_type",
        byte=139,
        format=ScalarType.INT16,
        description="Taper Type",
    ),
    HeaderField(
        name="alias_filter_freq",
        byte=141,
        format=ScalarType.INT16,
        description="Alias Filter Frequency",
    ),
    HeaderField(
        name="alias_filter_slope",
        byte=143,
        format=ScalarType.INT16,
        description="Alias Filter Slope",
    ),
    HeaderField(
        name="notch_filter_freq",
        byte=145,
        format=ScalarType.INT16,
        description="Notch Filter Frequency",
    ),
    HeaderField(
        name="notch_filter_slope",
        byte=147,
        format=ScalarType.INT16,
        description="Notch Filter Slope",
    ),
    HeaderField(
        name="low_cut_freq",
        byte=149,
        format=ScalarType.INT16,
        description="Low Cut Frequency",
    ),
    HeaderField(
        name="high_cut_freq",
        byte=151,
        format=ScalarType.INT16,
        description="High Cut Frequency",
    ),
    HeaderField(
        name="low_cut_slope",
        byte=153,
        format=ScalarType.INT16,
        description="Low Cut Slope",
    ),
    HeaderField(
        name="high_cut_slope",
        byte=155,
        format=ScalarType.INT16,
        description="High Cut Slope",
    ),
    HeaderField(
        name="year",
        byte=157,
        format=ScalarType.INT16,
        description="Year Data Recorded",
    ),
    HeaderField(
        name="day",
        byte=159,
        format=ScalarType.INT16,
        description="Day of Year",
    ),
    HeaderField(
        name="hour",
        byte=161,
        format=ScalarType.INT16,
        description="Hour of Day",
    ),
    HeaderField(
        name="minute",
        byte=163,
        format=ScalarType.INT16,
        description="Minute of Hour",
    ),
    HeaderField(
        name="second",
        byte=165,
        format=ScalarType.INT16,
        description="Second of Minute",
    ),
    HeaderField(
        name="time_basis_code",
        byte=167,
        format=ScalarType.INT16,
        description="Time Basis Code",
    ),
    HeaderField(
        name="trace_weighting_factor",
        byte=169,
        format=ScalarType.INT16,
        description="Trace Weighting Factor",
    ),
    HeaderField(
        name="geophone_group_no_roll1",
        byte=171,
        format=ScalarType.INT16,
        description="Geophone Group Number of Roll Switch Position One",
    ),
    HeaderField(
        name="geophone_group_no_first_trace",
        byte=173,
        format=ScalarType.INT16,
        description="Geophone Group Number of Trace Number One within Original Field Record",  # noqa: E501
    ),
    HeaderField(
        name="geophone_group_no_last_trace",
        byte=175,
        format=ScalarType.INT16,
        description="Geophone Group Number of Last Trace within Original Field Record",  # noqa: E501
    ),
    HeaderField(
        name="gap_size",
        byte=177,
        format=ScalarType.INT16,
        description="Gap Size (total number of groups dropped)",
    ),
    HeaderField(
        name="over_travel",
        byte=179,
        format=ScalarType.INT16,
        description="Over Travel Associated with Taper",
    ),
]


rev0_textual_file_header = TextHeaderSpec(
    rows=40,
    cols=80,
    offset=0,
    encoding=TextHeaderEncoding.EBCDIC,
    format=ScalarType.UINT8,  # noqa: A003
)


rev0_binary_file_header = HeaderSpec(
    fields=BINARY_FILE_HEADER_FIELDS_REV0,
    item_size=400,
    offset=3200,
)


rev0_trace_header = HeaderSpec(
    fields=TRACE_HEADER_FIELDS_REV0,
    item_size=240,
)


rev0_trace_data = TraceDataSpec(
    format=ScalarType.IBM32,  # noqa: A003
)


rev0_trace = TraceSpec(header_spec=rev0_trace_header, data_spec=rev0_trace_data)


rev0_segy = SegySpec(
    segy_standard=SegyStandard.REV0,
    text_file_header=rev0_textual_file_header,
    binary_file_header=rev0_binary_file_header,
    trace=rev0_trace,
    endianness=Endianness.BIG,
)
