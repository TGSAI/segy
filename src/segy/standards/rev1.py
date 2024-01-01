"""SEG-Y Revision 1 Specification."""


from typing import Optional

import fsspec
import numpy as np

from segy.schema.header import BinaryHeaderDescriptor
from segy.schema.header import HeaderFieldDescriptor
from segy.schema.header import TextHeaderDescriptor
from segy.schema.header import TraceHeaderDescriptor
from segy.schema.segy import SegyDescriptor
from segy.schema.segy import SegyStandard
from segy.schema.trace import TraceDataDescriptor
from segy.schema.trace import TraceDescriptor

REV1_TEXTUAL_FILE_HEADER = TextHeaderDescriptor(
    offset=0,  # depends on how many extended headers
    rows=40,
    cols=80,
    encoding="ebcdic",  # can be ascii too, but this is a good default
    format="uint8",  # ebcdic/ascii are 8-bit
)

REV1_EXTENDED_TEXTUAL_FILE_HEADER = TextHeaderDescriptor(
    offset=None,  # depends on how many extended headers
    rows=40,
    cols=80,
    encoding="ebcdic",  # can be ascii too, but this is a good default
    format="uint8",  # ebcdic/ascii are 8-bit
)

REV1_BINARY_FILE_HEADER = BinaryHeaderDescriptor(
    offset=3200,
    item_size=400,
    fields=[
        HeaderFieldDescriptor(
            name="job_id",
            offset=0,
            format="int32",
            endianness="big",
            description="Job Identification Number",
        ),
        HeaderFieldDescriptor(
            name="line_no",
            offset=4,
            format="int32",
            endianness="big",
            description="Line Number",
        ),
        HeaderFieldDescriptor(
            name="reel_no",
            offset=8,
            format="int32",
            endianness="big",
            description="Reel Number",
        ),
        HeaderFieldDescriptor(
            name="data_traces_ensemble",
            offset=12,
            format="int16",
            endianness="big",
            description="Number of Data Traces per Ensemble",
        ),
        HeaderFieldDescriptor(
            name="aux_traces_ensemble",
            offset=14,
            format="int16",
            endianness="big",
            description="Number of Auxiliary Traces per Ensemble",
        ),
        HeaderFieldDescriptor(
            name="sample_interval",
            offset=16,
            format="int16",
            endianness="big",
            description="Sample Interval",
        ),
        HeaderFieldDescriptor(
            name="sample_interval_orig",
            offset=18,
            format="int16",
            endianness="big",
            description="Sample Interval of Original Field Recording",
        ),
        HeaderFieldDescriptor(
            name="samples_per_trace",
            offset=20,
            format="int16",
            endianness="big",
            description="Number of Samples per Data Trace",
        ),
        HeaderFieldDescriptor(
            name="samples_per_trace_orig",
            offset=22,
            format="int16",
            endianness="big",
            description="Number of Samples per Data Trace for Original Field Recording",
        ),
        HeaderFieldDescriptor(
            name="data_sample_format",
            offset=24,
            format="int16",
            endianness="big",
            description="Data Sample Format Code",
        ),
        HeaderFieldDescriptor(
            name="ensemble_fold",
            offset=26,
            format="int16",
            endianness="big",
            description="Ensemble Fold",
        ),
        HeaderFieldDescriptor(
            name="trace_sorting",
            offset=28,
            format="int16",
            endianness="big",
            description="Trace Sorting Code",
        ),
        HeaderFieldDescriptor(
            name="vertical_sum",
            offset=30,
            format="int16",
            endianness="big",
            description="Vertical Sum Code",
        ),
        HeaderFieldDescriptor(
            name="sweep_freq_start",
            offset=32,
            format="int16",
            endianness="big",
            description="Sweep Frequency at Start",
        ),
        HeaderFieldDescriptor(
            name="sweep_freq_end",
            offset=34,
            format="int16",
            endianness="big",
            description="Sweep Frequency at End",
        ),
        HeaderFieldDescriptor(
            name="sweep_length",
            offset=36,
            format="int16",
            endianness="big",
            description="Sweep Length",
        ),
        HeaderFieldDescriptor(
            name="sweep_type",
            offset=38,
            format="int16",
            endianness="big",
            description="Sweep Type Code",
        ),
        HeaderFieldDescriptor(
            name="sweep_trace_no",
            offset=40,
            format="int16",
            endianness="big",
            description="Trace Number of Sweep Channel",
        ),
        HeaderFieldDescriptor(
            name="sweep_taper_start",
            offset=42,
            format="int16",
            endianness="big",
            description="Sweep Trace Taper Length at Start",
        ),
        HeaderFieldDescriptor(
            name="sweep_taper_end",
            offset=44,
            format="int16",
            endianness="big",
            description="Sweep Trace Taper Length at End",
        ),
        HeaderFieldDescriptor(
            name="taper_type",
            offset=46,
            format="int16",
            endianness="big",
            description="Taper Type",
        ),
        HeaderFieldDescriptor(
            name="correlated_traces",
            offset=48,
            format="int16",
            endianness="big",
            description="Correlated Data Traces",
        ),
        HeaderFieldDescriptor(
            name="binary_gain",
            offset=50,
            format="int16",
            endianness="big",
            description="Binary Gain Recovered",
        ),
        HeaderFieldDescriptor(
            name="amp_recovery_method",
            offset=52,
            format="int16",
            endianness="big",
            description="Amplitude Recovery Method",
        ),
        HeaderFieldDescriptor(
            name="measurement_system",
            offset=54,
            format="int16",
            endianness="big",
            description="Measurement System",
        ),
        HeaderFieldDescriptor(
            name="impulse_signal_polarity",
            offset=56,
            format="int16",
            endianness="big",
            description="Impulse Signal Polarity",
        ),
        HeaderFieldDescriptor(
            name="vibratory_polarity",
            offset=58,
            format="int16",
            endianness="big",
            description="Vibratory Polarity Code",
        ),
        HeaderFieldDescriptor(
            name="seg_y_revision",
            offset=300,
            format="int16",
            endianness="big",
            description="SEG Y Format Revision Number",
        ),
        HeaderFieldDescriptor(
            name="fixed_length_trace_flag",
            offset=302,
            format="int16",
            endianness="big",
            description="Fixed Length Trace Flag",
        ),
        HeaderFieldDescriptor(
            name="extended_textual_headers",
            offset=304,
            format="int16",
            endianness="big",
            description="Number of 3200-byte, Extended Textual File Header Records Following the Binary Header",  # noqa: E501
        ),
        HeaderFieldDescriptor(
            name="additional_trace_headers",
            offset=306,
            format="int16",
            endianness="big",
            description="Maximum Number of Additional Trace Headers",
        ),
    ],
)


REV1_TRACE_HEADER = TraceHeaderDescriptor(
    offset=None,  # depends on extended text headers. must set later.
    item_size=240,
    fields=[
        HeaderFieldDescriptor(
            name="trace_seq_line",
            offset=0,
            format="int32",
            endianness="big",
            description="Trace Sequence Number within Line",
        ),
        HeaderFieldDescriptor(
            name="trace_seq_file",
            offset=4,
            format="int32",
            endianness="big",
            description="Trace Sequence Number within File",
        ),
        HeaderFieldDescriptor(
            name="field_rec_no",
            offset=8,
            format="int32",
            endianness="big",
            description="Original Field Record Number",
        ),
        HeaderFieldDescriptor(
            name="trace_no_field_rec",
            offset=12,
            format="int32",
            endianness="big",
            description="Trace Number within the Field Record",
        ),
        HeaderFieldDescriptor(
            name="energy_src_pt",
            offset=16,
            format="int32",
            endianness="big",
            description="Energy Source Point Number",
        ),
        HeaderFieldDescriptor(
            name="cdp_ens_no",
            offset=20,
            format="int32",
            endianness="big",
            description="Ensemble Number (CDP, CMP, etc.)",
        ),
        HeaderFieldDescriptor(
            name="trace_no_ens",
            offset=24,
            format="int32",
            endianness="big",
            description="Trace Number within the Ensemble",
        ),
        HeaderFieldDescriptor(
            name="trace_id",
            offset=28,
            format="int16",
            endianness="big",
            description="Trace Identification Code",
        ),
        HeaderFieldDescriptor(
            name="vert_sum",
            offset=30,
            format="int16",
            endianness="big",
            description="Number of Vertically Stacked Traces",
        ),
        HeaderFieldDescriptor(
            name="horiz_stack",
            offset=32,
            format="int16",
            endianness="big",
            description="Number of Horizontally Stacked Traces",
        ),
        HeaderFieldDescriptor(
            name="data_use",
            offset=34,
            format="int16",
            endianness="big",
            description="Data Use",
        ),
        HeaderFieldDescriptor(
            name="dist_src_to_rec",
            offset=36,
            format="int32",
            endianness="big",
            description="Distance from Source Point to Receiver Group",
        ),
        HeaderFieldDescriptor(
            name="rec_elev",
            offset=40,
            format="int32",
            endianness="big",
            description="Receiver Group Elevation",
        ),
        HeaderFieldDescriptor(
            name="src_elev",
            offset=44,
            format="int32",
            endianness="big",
            description="Source Elevation",
        ),
        HeaderFieldDescriptor(
            name="src_depth",
            offset=48,
            format="int32",
            endianness="big",
            description="Source Depth",
        ),
        HeaderFieldDescriptor(
            name="datum_elev_rec",
            offset=52,
            format="int32",
            endianness="big",
            description="Datum Elevation at Receiver Group",
        ),
        HeaderFieldDescriptor(
            name="datum_elev_src",
            offset=56,
            format="int32",
            endianness="big",
            description="Datum Elevation at Source",
        ),
        HeaderFieldDescriptor(
            name="water_depth_src",
            offset=60,
            format="int32",
            endianness="big",
            description="Water Depth at Source",
        ),
        HeaderFieldDescriptor(
            name="water_depth_rec",
            offset=64,
            format="int32",
            endianness="big",
            description="Water Depth at Receiver Group",
        ),
        HeaderFieldDescriptor(
            name="scalar_apply_elev",
            offset=68,
            format="int16",
            endianness="big",
            description="Scalar to be applied to all elevations and depths",
        ),
        HeaderFieldDescriptor(
            name="scalar_apply_coords",
            offset=70,
            format="int16",
            endianness="big",
            description="Scalar to be applied to all coordinates",
        ),
        HeaderFieldDescriptor(
            name="src_x",
            offset=72,
            format="int32",
            endianness="big",
            description="Source X coordinate",
        ),
        HeaderFieldDescriptor(
            name="src_y",
            offset=76,
            format="int32",
            endianness="big",
            description="Source Y coordinate",
        ),
        HeaderFieldDescriptor(
            name="rec_x",
            offset=80,
            format="int32",
            endianness="big",
            description="Receiver X coordinate",
        ),
        HeaderFieldDescriptor(
            name="rec_y",
            offset=84,
            format="int32",
            endianness="big",
            description="Receiver Y coordinate",
        ),
        HeaderFieldDescriptor(
            name="coord_units",
            offset=88,
            format="int16",
            endianness="big",
            description="Coordinate units",
        ),
        HeaderFieldDescriptor(
            name="weathering_vel",
            offset=90,
            format="int16",
            endianness="big",
            description="Weathering Velocity",
        ),
        HeaderFieldDescriptor(
            name="subweathering_vel",
            offset=92,
            format="int16",
            endianness="big",
            description="Subweathering Velocity",
        ),
        HeaderFieldDescriptor(
            name="uphole_time_src",
            offset=94,
            format="int16",
            endianness="big",
            description="Uphole Time at Source",
        ),
        HeaderFieldDescriptor(
            name="uphole_time_rec",
            offset=96,
            format="int16",
            endianness="big",
            description="Uphole Time at Receiver",
        ),
        HeaderFieldDescriptor(
            name="src_static_corr",
            offset=98,
            format="int16",
            endianness="big",
            description="Source Static Correction",
        ),
        HeaderFieldDescriptor(
            name="rec_static_corr",
            offset=100,
            format="int16",
            endianness="big",
            description="Receiver Static Correction",
        ),
        HeaderFieldDescriptor(
            name="total_static",
            offset=102,
            format="int16",
            endianness="big",
            description="Total Static Applied",
        ),
        HeaderFieldDescriptor(
            name="lag_time_a",
            offset=104,
            format="int16",
            endianness="big",
            description="Lag Time A",
        ),
        HeaderFieldDescriptor(
            name="lag_time_b",
            offset=106,
            format="int16",
            endianness="big",
            description="Lag Time B",
        ),
        HeaderFieldDescriptor(
            name="delay_rec_time",
            offset=108,
            format="int16",
            endianness="big",
            description="Delay Recording Time",
        ),
        HeaderFieldDescriptor(
            name="mute_start",
            offset=110,
            format="int16",
            endianness="big",
            description="Start Time of Mute",
        ),
        HeaderFieldDescriptor(
            name="mute_end",
            offset=112,
            format="int16",
            endianness="big",
            description="End Time of Mute",
        ),
        HeaderFieldDescriptor(
            name="nsamples",
            offset=114,
            format="int16",
            endianness="big",
            description="Number of Samples in this Trace",
        ),
        HeaderFieldDescriptor(
            name="sample_interval",
            offset=116,
            format="int16",
            endianness="big",
            description="Sample Interval for this Trace",
        ),
        HeaderFieldDescriptor(
            name="gain_type",
            offset=118,
            format="int16",
            endianness="big",
            description="Gain Type of Field Instruments",
        ),
        HeaderFieldDescriptor(
            name="instrument_gain",
            offset=120,
            format="int16",
            endianness="big",
            description="Instrument Gain Constant",
        ),
        HeaderFieldDescriptor(
            name="instrument_early_gain",
            offset=122,
            format="int16",
            endianness="big",
            description="Instrument Early Gain",
        ),
        HeaderFieldDescriptor(
            name="correlated",
            offset=124,
            format="int16",
            endianness="big",
            description="Correlated",
        ),
        HeaderFieldDescriptor(
            name="sweep_freq_start",
            offset=126,
            format="int16",
            endianness="big",
            description="Sweep Frequency at Start",
        ),
        HeaderFieldDescriptor(
            name="sweep_freq_end",
            offset=128,
            format="int16",
            endianness="big",
            description="Sweep Frequency at End",
        ),
        HeaderFieldDescriptor(
            name="sweep_length",
            offset=130,
            format="int16",
            endianness="big",
            description="Sweep Length",
        ),
        HeaderFieldDescriptor(
            name="sweep_type",
            offset=132,
            format="int16",
            endianness="big",
            description="Sweep Type",
        ),
        HeaderFieldDescriptor(
            name="sweep_trace_taper_start",
            offset=134,
            format="int16",
            endianness="big",
            description="Sweep Trace Taper Length at Start",
        ),
        HeaderFieldDescriptor(
            name="sweep_trace_taper_end",
            offset=136,
            format="int16",
            endianness="big",
            description="Sweep Trace Taper Length at End",
        ),
        HeaderFieldDescriptor(
            name="taper_type",
            offset=138,
            format="int16",
            endianness="big",
            description="Taper Type",
        ),
        HeaderFieldDescriptor(
            name="alias_filter_freq",
            offset=140,
            format="int16",
            endianness="big",
            description="Alias Filter Frequency",
        ),
        HeaderFieldDescriptor(
            name="alias_filter_slope",
            offset=142,
            format="int16",
            endianness="big",
            description="Alias Filter Slope",
        ),
        HeaderFieldDescriptor(
            name="notch_filter_freq",
            offset=144,
            format="int16",
            endianness="big",
            description="Notch Filter Frequency",
        ),
        HeaderFieldDescriptor(
            name="notch_filter_slope",
            offset=146,
            format="int16",
            endianness="big",
            description="Notch Filter Slope",
        ),
        HeaderFieldDescriptor(
            name="low_cut_freq",
            offset=148,
            format="int16",
            endianness="big",
            description="Low Cut Frequency",
        ),
        HeaderFieldDescriptor(
            name="high_cut_freq",
            offset=150,
            format="int16",
            endianness="big",
            description="High Cut Frequency",
        ),
        HeaderFieldDescriptor(
            name="low_cut_slope",
            offset=152,
            format="int16",
            endianness="big",
            description="Low Cut Slope",
        ),
        HeaderFieldDescriptor(
            name="high_cut_slope",
            offset=154,
            format="int16",
            endianness="big",
            description="High Cut Slope",
        ),
        HeaderFieldDescriptor(
            name="year",
            offset=156,
            format="int16",
            endianness="big",
            description="Year Data Recorded",
        ),
        HeaderFieldDescriptor(
            name="day",
            offset=158,
            format="int16",
            endianness="big",
            description="Day of Year",
        ),
        HeaderFieldDescriptor(
            name="hour",
            offset=160,
            format="int16",
            endianness="big",
            description="Hour of Day",
        ),
        HeaderFieldDescriptor(
            name="minute",
            offset=162,
            format="int16",
            endianness="big",
            description="Minute of Hour",
        ),
        HeaderFieldDescriptor(
            name="second",
            offset=164,
            format="int16",
            endianness="big",
            description="Second of Minute",
        ),
        HeaderFieldDescriptor(
            name="time_basis_code",
            offset=166,
            format="int16",
            endianness="big",
            description="Time Basis Code",
        ),
        HeaderFieldDescriptor(
            name="trace_weighting_factor",
            offset=168,
            format="int16",
            endianness="big",
            description="Trace Weighting Factor",
        ),
        HeaderFieldDescriptor(
            name="geophone_group_no_roll1",
            offset=170,
            format="int16",
            endianness="big",
            description="Geophone Group Number of Roll Switch Position One",
        ),
        HeaderFieldDescriptor(
            name="geophone_group_no_first_trace",
            offset=172,
            format="int16",
            endianness="big",
            description="Geophone Group Number of Trace Number One within Original Field Record",  # noqa: E501
        ),
        HeaderFieldDescriptor(
            name="geophone_group_no_last_trace",
            offset=174,
            format="int16",
            endianness="big",
            description="Geophone Group Number of Last Trace within Original Field Record",  # noqa: E501
        ),
        HeaderFieldDescriptor(
            name="gap_size",
            offset=176,
            format="int16",
            endianness="big",
            description="Gap Size (total number of groups dropped)",
        ),
        HeaderFieldDescriptor(
            name="over_travel",
            offset=178,
            format="int16",
            endianness="big",
            description="Over Travel Associated with Taper",
        ),
        HeaderFieldDescriptor(
            name="x_coordinate",
            offset=180,
            format="int32",
            endianness="big",
            description="X coordinate of ensemble (CDP) position",
        ),
        HeaderFieldDescriptor(
            name="y_coordinate",
            offset=184,
            format="int32",
            endianness="big",
            description="Y coordinate of ensemble (CDP) position",
        ),
        HeaderFieldDescriptor(
            name="inline_no",
            offset=188,
            format="int32",
            endianness="big",
            description="Inline number",
        ),
        HeaderFieldDescriptor(
            name="crossline_no",
            offset=192,
            format="int32",
            endianness="big",
            description="Crossline number",
        ),
        HeaderFieldDescriptor(
            name="shotpoint_no",
            offset=196,
            format="int32",
            endianness="big",
            description="Shotpoint number",
        ),
        HeaderFieldDescriptor(
            name="scalar_apply_shotpoint",
            offset=200,
            format="int16",
            endianness="big",
            description="Scalar to be applied to the shotpoint number",
        ),
        HeaderFieldDescriptor(
            name="trace_value_measurement_unit",
            offset=202,
            format="int16",
            endianness="big",
            description="Trace value measurement unit",
        ),
        HeaderFieldDescriptor(
            name="transduction_constant_mantissa",
            offset=204,
            format="int32",
            endianness="big",
            description="Transduction Constant Mantissa",
        ),
        HeaderFieldDescriptor(
            name="transduction_constant_exponent",
            offset=208,
            format="int16",
            endianness="big",
            description="Transduction Constant Exponent",
        ),
        HeaderFieldDescriptor(
            name="transduction_units",
            offset=210,
            format="int16",
            endianness="big",
            description="Transduction Units",
        ),
        HeaderFieldDescriptor(
            name="device_trace_id",
            offset=212,
            format="int16",
            endianness="big",
            description="Device/Trace Identifier",
        ),
        HeaderFieldDescriptor(
            name="times_scalar",
            offset=214,
            format="int16",
            endianness="big",
            description="Scalar to be applied to times",
        ),
        HeaderFieldDescriptor(
            name="source_type_orientation",
            offset=216,
            format="int16",
            endianness="big",
            description="Source Type/Orientation",
        ),
        HeaderFieldDescriptor(
            name="source_energy_direction_mantissa",
            offset=218,
            format="int32",
            endianness="big",
            description="Source Energy Direction with respect to vertical [Mantissa]",
        ),
        HeaderFieldDescriptor(
            name="source_energy_direction_exponent",
            offset=222,
            format="int16",
            endianness="big",
            description="Source Energy Direction with respect to vertical [Exponent]",
        ),
        HeaderFieldDescriptor(
            name="source_measurement_mantissa",
            offset=224,
            format="int32",
            endianness="big",
            description="Source Measurement Mantissa",
        ),
        HeaderFieldDescriptor(
            name="source_measurement_exponent",
            offset=228,
            format="int16",
            endianness="big",
            description="Source Measurement Exponent",
        ),
        HeaderFieldDescriptor(
            name="source_measurement_unit",
            offset=230,
            format="int16",
            endianness="big",
            description="Source Measurement Unit",
        ),
    ],
)

REV1_TRACE_DATA = TraceDataDescriptor(
    format="ibm32",
    endianness="big",
    samples=None,  # Can be variable.
)

REV1_TRACE = TraceDescriptor(
    header_descriptor=REV1_TRACE_HEADER,
    data_descriptor=REV1_TRACE_DATA,
)


def open_rev1(
    path: str,
    text_file_header: Optional[TextHeaderDescriptor] = None,
    binary_file_header: Optional[BinaryHeaderDescriptor] = None,
    trace_header_descriptor: Optional[TraceHeaderDescriptor] = None,
    trace_data_descriptor: Optional[TraceDataDescriptor] = None,
    samples_per_trace_key: str = "samples_per_trace",
    extended_textual_headers_key: str = "extended_textual_headers",
) -> SegyDescriptor:
    segy_standard = SegyStandard.REV1

    if text_file_header is None:
        text_file_header = REV1_TEXTUAL_FILE_HEADER
    else:
        segy_standard = SegyStandard.CUSTOM

    if binary_file_header is None:
        binary_file_header = REV1_BINARY_FILE_HEADER
    else:
        segy_standard = SegyStandard.CUSTOM

    if trace_header_descriptor is not None:
        REV1_TRACE.header_descriptor = trace_header_descriptor
        segy_standard = SegyStandard.CUSTOM

    if trace_data_descriptor is not None:
        REV1_TRACE.data_descriptor = trace_data_descriptor
        segy_standard = SegyStandard.CUSTOM

    rev1_desc = SegyDescriptor(
        segy_standard=segy_standard,
        text_file_header=text_file_header,
        binary_file_header=binary_file_header,
        traces=REV1_TRACE,
    )

    bin_dtype = rev1_desc.binary_file_header.dtype
    bin_offset = rev1_desc.binary_file_header.offset

    with fsspec.open(path, mode="rb") as fp:
        fp.seek(bin_offset)
        buffer = bytearray(bin_dtype.itemsize)
        fp.readinto(buffer)
        bin_hdr = np.frombuffer(buffer, dtype=bin_dtype).squeeze()

    # Update number of samples
    rev1_desc.traces.data_descriptor.samples = bin_hdr[samples_per_trace_key]

    traces_offset = (
        rev1_desc.text_file_header.itemsize + rev1_desc.binary_file_header.itemsize
    )

    if extended_textual_headers_key in bin_hdr.dtype.names:
        extended_header_count = bin_hdr[extended_textual_headers_key]
        if extended_header_count > 0:
            traces_offset += (
                extended_header_count * REV1_EXTENDED_TEXTUAL_FILE_HEADER.itemsize
            )

    rev1_desc.traces.offset = traces_offset

    return rev1_desc
