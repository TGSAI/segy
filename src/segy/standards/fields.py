"""SEG-Y header fields and Enums."""

from __future__ import annotations

from enum import Enum

from segy.compat import StrEnum
from segy.schema import HeaderField
from segy.schema import ScalarType


class HeaderEnum(str, Enum):
    """A subclass for Enum for convenience and reduce repetition.

    We don't want to define the same things as Enum, HeaderField, etc.
    Using this we can conveniently have Enums for users; and also
    use it to build the HeaderField lists to generate templates for
    SEG-Y standards.

    Args:
        field: HeaderField instance

    Attributes:
        byte: Start byte location.
        format: Data format.
        description: Long description of the field.
    """

    def __init__(self, field: HeaderField) -> None:
        self.byte = field.byte
        self.format = field.format  # type: ignore[assignment, method-assign]
        self.description = field.description
        self._value_ = field

    def __repr__(self) -> str:
        """Nice representation to users."""
        return f"{self.__class__.__name__}({self._value_})"

    def __new__(cls, field: HeaderField) -> HeaderEnum:
        """Create a string member but set value to HeaderField."""
        return str.__new__(cls, field.name)


# fmt: off
class BinFieldRev0(HeaderEnum):
    """Header fields for SEG-Y binary headers revision 0."""
    JOB_ID                   = HeaderField(name="job_id", byte=1, format=ScalarType.INT32, description="Job identification number.")
    LINE_NUM                 = HeaderField(name="line_num", byte=5, format=ScalarType.INT32, description="Line number.")
    REEL_NUM                 = HeaderField(name="reel_num", byte=9, format=ScalarType.INT32, description="Reel number.")
    DATA_TRACES_PER_ENSEMBLE = HeaderField(name="data_traces_per_ensemble", byte=13, format=ScalarType.INT16, description="Number of data traces per ensemble.")
    AUX_TRACES_PER_ENSEMBLE  = HeaderField(name="aux_traces_per_ensemble", byte=15, format=ScalarType.INT16, description="Number of auxiliary traces per ensemble.")
    SAMPLE_INTERVAL          = HeaderField(name="sample_interval", byte=17, format=ScalarType.INT16, description="Sample interval (microseconds).")
    ORIG_SAMPLE_INTERVAL     = HeaderField(name="orig_sample_interval", byte=19, format=ScalarType.INT16, description="Sample interval of original field recording (microseconds).")
    SAMPLES_PER_TRACE        = HeaderField(name="samples_per_trace", byte=21, format=ScalarType.INT16, description="Number of samples per data trace.")
    ORIG_SAMPLES_PER_TRACE   = HeaderField(name="orig_samples_per_trace", byte=23, format=ScalarType.INT16, description="Number of samples per data trace for original field recording.")
    DATA_SAMPLE_FORMAT       = HeaderField(name="data_sample_format", byte=25, format=ScalarType.INT16, description="Data sample format code.")
    ENSEMBLE_FOLD            = HeaderField(name="ensemble_fold", byte=27, format=ScalarType.INT16, description="Ensemble fold.")
    TRACE_SORTING_CODE       = HeaderField(name="trace_sorting_code", byte=29, format=ScalarType.INT16, description="Trace sorting code.")
    VERTICAL_SUM_CODE        = HeaderField(name="vertical_sum_code", byte=31, format=ScalarType.INT16, description="Vertical sum code.")
    SWEEP_FREQ_START         = HeaderField(name="sweep_freq_start", byte=33, format=ScalarType.INT16, description="Sweep frequency at start.")
    SWEEP_FREQ_END           = HeaderField(name="sweep_freq_end", byte=35, format=ScalarType.INT16, description="Sweep frequency at end.")
    SWEEP_LENGTH             = HeaderField(name="sweep_length", byte=37, format=ScalarType.INT16, description="Sweep length (ms).")
    SWEEP_TYPE_CODE          = HeaderField(name="sweep_type_code", byte=39, format=ScalarType.INT16, description="Sweep type code.")
    SWEEP_TRACE_NUM          = HeaderField(name="sweep_trace_num", byte=41, format=ScalarType.INT16, description="Trace number of sweep channel.")
    SWEEP_TAPER_START        = HeaderField(name="sweep_taper_start", byte=43, format=ScalarType.INT16, description="Sweep trace taper length at start (ms).")
    SWEEP_TAPER_END          = HeaderField(name="sweep_taper_end", byte=45, format=ScalarType.INT16, description="Sweep trace taper length at end (ms).")
    TAPER_TYPE_CODE          = HeaderField(name="taper_type_code", byte=47, format=ScalarType.INT16, description="Taper type.")
    CORRELATED_DATA_CODE     = HeaderField(name="correlated_data_code", byte=49, format=ScalarType.INT16, description="Correlated data traces.")
    BINARY_GAIN_CODE         = HeaderField(name="binary_gain_code", byte=51, format=ScalarType.INT16, description="Binary gain recovered.")
    AMP_RECOVERY_CODE        = HeaderField(name="amp_recovery_code", byte=53, format=ScalarType.INT16, description="Amplitude recovery method.")
    MEASUREMENT_SYSTEM_CODE  = HeaderField(name="measurement_system_code", byte=55, format=ScalarType.INT16, description="Measurement system.")
    IMPULSE_POLARITY_CODE    = HeaderField(name="impulse_polarity_code", byte=57, format=ScalarType.INT16, description="Impulse signal polarity.")
    VIBRATORY_POLARITY_CODE  = HeaderField(name="vibratory_polarity_code", byte=59, format=ScalarType.INT16, description="Vibratory polarity code.")


class BinFieldRev1(HeaderEnum):
    """Extra header fields for SEG-Y binary headers revision 1."""
    SEGY_REVISION             = HeaderField(name="segy_revision", byte=301, format=ScalarType.INT16, description="SEG-Y format revision number.")
    FIXED_LENGTH_TRACE_FLAG   = HeaderField(name="fixed_length_trace_flag", byte=303, format=ScalarType.INT16, description="Fixed length trace flag.")
    NUM_EXTENDED_TEXT_HEADERS = HeaderField(name="num_extended_text_headers", byte=305, format=ScalarType.INT16, description="Number of 3200-byte extended textual file header records.")


class BinFieldRev2(HeaderEnum):
    """Extra header fields for SEG-Y binary headers revision 2."""
    EXTENDED_DATA_TRACES_PER_ENSEMBLE = HeaderField(name="extended_data_traces_per_ensemble", byte=61, format=ScalarType.INT32, description="Extended number of data traces per ensemble.")
    EXTENDED_AUX_TRACES_PER_ENSEMBLE  = HeaderField(name="extended_aux_traces_per_ensemble", byte=65, format=ScalarType.INT32, description="Extended number of auxiliary traces per ensemble.")
    EXTENDED_SAMPLES_PER_TRACE        = HeaderField(name="extended_samples_per_trace", byte=69, format=ScalarType.INT32, description="Extended number of samples per data trace.")
    EXTENDED_SAMPLE_INTERVAL          = HeaderField(name="extended_sample_interval", byte=73, format=ScalarType.FLOAT64, description="Extended sample interval.")
    EXTENDED_ORIG_SAMPLE_INTERVAL     = HeaderField(name="extended_orig_sample_interval", byte=81, format=ScalarType.FLOAT64, description="Extended sample interval of original field recording.")
    EXTENDED_ORIG_SAMPLES_PER_TRACE   = HeaderField(name="extended_orig_samples_per_trace", byte=89, format=ScalarType.INT32, description="Extended number of samples per data trace in original recording.")
    EXTENDED_ENSEMBLE_FOLD            = HeaderField(name="extended_ensemble_fold", byte=93, format=ScalarType.INT32, description="Extended ensemble fold. Overrides bytes 3227–3228 if nonzero.")
    BYTE_ORDER                        = HeaderField(name="byte_order", byte=97, format=ScalarType.INT32, description="Integer constant for byte order detection.")
    SEGY_REVISION_MAJOR               = HeaderField(name="segy_revision_major", byte=301, format=ScalarType.UINT8, description="Major SEG-Y Format Revision Number.")
    SEGY_REVISION_MINOR               = HeaderField(name="segy_revision_minor", byte=302, format=ScalarType.UINT8, description="Major SEG-Y Format Revision Number.")
    MAX_EXTENDED_TRACE_HEADERS        = HeaderField(name="max_extended_trace_headers", byte=307, format=ScalarType.INT16, description="Maximum number of additional 240-byte trace headers. Zero indicates none.")
    SURVEY_TYPE                       = HeaderField(name="survey_type", byte=309, format=ScalarType.INT16, description="Survey type: sum of options from each group.")
    TIME_BASIS_CODE                   = HeaderField(name="time_basis_code", byte=311, format=ScalarType.INT16, description="Time basis code: 1 = Local, 2 = GMT, 3 = Other, 4 = UTC, 5 = GPS.")
    NUM_TRACES                        = HeaderField(name="num_traces", byte=313, format=ScalarType.UINT64, description="Number of traces in this file or stream.")
    BYTE_OFFSET_FIRST_TRACE           = HeaderField(name="byte_offset_first_trace", byte=321, format=ScalarType.UINT64, description="Byte offset of first trace relative to start of file or stream.")
    NUM_DATA_TRAILER_STANZAS          = HeaderField(name="num_data_trailer_stanzas", byte=329, format=ScalarType.INT32, description="Number of 3200-byte data trailer stanza records. Zero indicates none.")
# fmt: on


BIN_HDR_FIELDS_REV0 = [field.value for field in BinFieldRev0]
BIN_HDR_FIELDS_REV1 = BIN_HDR_FIELDS_REV0 + [field.value for field in BinFieldRev1]
BIN_HDR_FIELDS_REV2 = BIN_HDR_FIELDS_REV1 + [field.value for field in BinFieldRev2]
BIN_HDR_FIELDS_REV2.remove(BinFieldRev1.SEGY_REVISION.value)
BIN_HDR_FIELDS_REV2 = sorted(BIN_HDR_FIELDS_REV2, key=lambda f: f.byte)


TRC_HDR_FIELDS_REV0 = [
    HeaderField(
        name="trace_seq_num_line",
        byte=1,
        format=ScalarType.INT32,
        description="Trace sequence number within line.",
    ),
    HeaderField(
        name="trace_seq_num_reel",
        byte=5,
        format=ScalarType.INT32,
        description="Trace sequence number within reel.",
    ),
    HeaderField(
        name="orig_field_record_num",
        byte=9,
        format=ScalarType.INT32,
        description="Original field record number.",
    ),
    HeaderField(
        name="trace_num_orig_record",
        byte=13,
        format=ScalarType.INT32,
        description="Trace number within the original field record.",
    ),
    HeaderField(
        name="energy_source_point_num",
        byte=17,
        format=ScalarType.INT32,
        description="Energy source point number.",
    ),
    HeaderField(
        name="ensemble_num",
        byte=21,
        format=ScalarType.INT32,
        description="Ensemble number (CDP, CMP, ...).",
    ),
    HeaderField(
        name="trace_num_ensemble",
        byte=25,
        format=ScalarType.INT32,
        description="Trace number within the ensemble.",
    ),
    HeaderField(
        name="trace_id_code",
        byte=29,
        format=ScalarType.INT16,
        description="Trace identification code.",
    ),
    HeaderField(
        name="vertically_summed_traces",
        byte=31,
        format=ScalarType.INT16,
        description="Number of vertically summed traces.",
    ),
    HeaderField(
        name="horizontally_stacked_traces",
        byte=33,
        format=ScalarType.INT16,
        description="Number of horizontally stacked traces.",
    ),
    HeaderField(
        name="data_use",
        byte=35,
        format=ScalarType.INT16,
        description="Data use (production or test).",
    ),
    HeaderField(
        name="source_to_receiver_distance",
        byte=37,
        format=ScalarType.INT32,
        description="Distance from center of source to the center of the receiver.",
    ),
    HeaderField(
        name="receiver_group_elevation",
        byte=41,
        format=ScalarType.INT32,
        description="Elevation at the receiver group.",
    ),
    HeaderField(
        name="source_surface_elevation",
        byte=45,
        format=ScalarType.INT32,
        description="Surface elevation at the source.",
    ),
    HeaderField(
        name="source_depth_below_surface",
        byte=49,
        format=ScalarType.INT32,
        description="Source depth below surface.",
    ),
    HeaderField(
        name="receiver_datum_elevation",
        byte=53,
        format=ScalarType.INT32,
        description="Datum elevation at the receiver group.",
    ),
    HeaderField(
        name="source_datum_elevation",
        byte=57,
        format=ScalarType.INT32,
        description="Datum elevation at the source.",
    ),
    HeaderField(
        name="source_water_depth",
        byte=61,
        format=ScalarType.INT32,
        description="Water depth at the source.",
    ),
    HeaderField(
        name="receiver_water_depth",
        byte=65,
        format=ScalarType.INT32,
        description="Water depth at the receiver group.",
    ),
    HeaderField(
        name="elevation_depth_scalar",
        byte=69,
        format=ScalarType.INT16,
        description="Scalar to be applied to all elevations and depths.",
    ),
    HeaderField(
        name="coordinate_scalar",
        byte=71,
        format=ScalarType.INT16,
        description="Scalar to be applied to coordinates.",
    ),
    HeaderField(
        name="source_coord_x",
        byte=73,
        format=ScalarType.INT32,
        description="Source coordinate - X.",
    ),
    HeaderField(
        name="source_coord_y",
        byte=77,
        format=ScalarType.INT32,
        description="Source coordinate - Y.",
    ),
    HeaderField(
        name="group_coord_x",
        byte=81,
        format=ScalarType.INT32,
        description="Receiver coordinate - X.",
    ),
    HeaderField(
        name="group_coord_y",
        byte=85,
        format=ScalarType.INT32,
        description="Receiver coordinate - Y.",
    ),
    HeaderField(
        name="coordinate_units",
        byte=89,
        format=ScalarType.INT16,
        description="Coordinate units.",
    ),
    HeaderField(
        name="weathering_velocity",
        byte=91,
        format=ScalarType.INT16,
        description="Weathering velocity.",
    ),
    HeaderField(
        name="subweathering_velocity",
        byte=93,
        format=ScalarType.INT16,
        description="Subweathering velocity.",
    ),
    HeaderField(
        name="uphole_time_source",
        byte=95,
        format=ScalarType.INT16,
        description="Uphole time at the source.",
    ),
    HeaderField(
        name="uphole_time_group",
        byte=97,
        format=ScalarType.INT16,
        description="Uphole time at the receiver group.",
    ),
    HeaderField(
        name="source_static_correction",
        byte=99,
        format=ScalarType.INT16,
        description="Source static correction.",
    ),
    HeaderField(
        name="receiver_static_correction",
        byte=101,
        format=ScalarType.INT16,
        description="Receiver static correction.",
    ),
    HeaderField(
        name="total_static_applied",
        byte=103,
        format=ScalarType.INT16,
        description="Total static applied.",
    ),
    HeaderField(
        name="lag_time_a",
        byte=105,
        format=ScalarType.INT16,
        description="Lag time A",
    ),
    HeaderField(
        name="lag_time_b",
        byte=107,
        format=ScalarType.INT16,
        description="Lag time B",
    ),
    HeaderField(
        name="delay_recording_time",
        byte=109,
        format=ScalarType.INT16,
        description="Delay recording time.",
    ),
    HeaderField(
        name="mute_time_start",
        byte=111,
        format=ScalarType.INT16,
        description="Mute time - start.",
    ),
    HeaderField(
        name="mute_time_end",
        byte=113,
        format=ScalarType.INT16,
        description="Mute time - end.",
    ),
    HeaderField(
        name="samples_per_trace",
        byte=115,
        format=ScalarType.INT16,
        description="Number of samples in this trace.",
    ),
    HeaderField(
        name="sample_interval",
        byte=117,
        format=ScalarType.INT16,
        description="Sample interval in microseconds for this trace.",
    ),
    HeaderField(
        name="instrument_gain_type",
        byte=119,
        format=ScalarType.INT16,
        description="Gain type of field instruments.",
    ),
    HeaderField(
        name="instrument_gain_const",
        byte=121,
        format=ScalarType.INT16,
        description="Instrument gain constant.",
    ),
    HeaderField(
        name="instrument_gain_initial",
        byte=123,
        format=ScalarType.INT16,
        description="Instrument early or initial gain.",
    ),
    HeaderField(
        name="correlated_data",
        byte=125,
        format=ScalarType.INT16,
        description="Correlated.",
    ),
    HeaderField(
        name="sweep_freq_start",
        byte=127,
        format=ScalarType.INT16,
        description="Sweep frequency at start.",
    ),
    HeaderField(
        name="sweep_freq_end",
        byte=129,
        format=ScalarType.INT16,
        description="Sweep frequency at end.",
    ),
    HeaderField(
        name="sweep_length",
        byte=131,
        format=ScalarType.INT16,
        description="Sweep length in ms.",
    ),
    HeaderField(
        name="sweep_type",
        byte=133,
        format=ScalarType.INT16,
        description="Sweep type code.",
    ),
    HeaderField(
        name="sweep_taper_start",
        byte=135,
        format=ScalarType.INT16,
        description="Sweep trace taper length at start in ms.",
    ),
    HeaderField(
        name="sweep_taper_end",
        byte=137,
        format=ScalarType.INT16,
        description="Sweep trace taper length at end in ms.",
    ),
    HeaderField(
        name="taper_type",
        byte=139,
        format=ScalarType.INT16,
        description="Taper type.",
    ),
    HeaderField(
        name="alias_filter_freq",
        byte=141,
        format=ScalarType.INT16,
        description="Alias filter frequency, if used.",
    ),
    HeaderField(
        name="alias_filter_slope",
        byte=143,
        format=ScalarType.INT16,
        description="Alias filter slope.",
    ),
    HeaderField(
        name="notch_filter_freq",
        byte=145,
        format=ScalarType.INT16,
        description="Notch filter frequency, if used.",
    ),
    HeaderField(
        name="notch_filter_slope",
        byte=147,
        format=ScalarType.INT16,
        description="Notch filter slope.",
    ),
    HeaderField(
        name="low_cut_freq",
        byte=149,
        format=ScalarType.INT16,
        description="Low cut frequency, if used.",
    ),
    HeaderField(
        name="high_cut_freq",
        byte=151,
        format=ScalarType.INT16,
        description="High cut frequency, if used.",
    ),
    HeaderField(
        name="low_cut_slope",
        byte=153,
        format=ScalarType.INT16,
        description="Low cut slope.",
    ),
    HeaderField(
        name="high_cut_slope",
        byte=155,
        format=ScalarType.INT16,
        description="High cut slope.",
    ),
    HeaderField(
        name="year_recorded",
        byte=157,
        format=ScalarType.INT16,
        description="Year data recorded.",
    ),
    HeaderField(
        name="day_of_year",
        byte=159,
        format=ScalarType.INT16,
        description="Day of year.",
    ),
    HeaderField(
        name="hour_of_day",
        byte=161,
        format=ScalarType.INT16,
        description="Hour of day (24-hour clock).",
    ),
    HeaderField(
        name="minute_of_hour",
        byte=163,
        format=ScalarType.INT16,
        description="Minute of hour.",
    ),
    HeaderField(
        name="second_of_minute",
        byte=165,
        format=ScalarType.INT16,
        description="Second of minute.",
    ),
    HeaderField(
        name="time_basis_code",
        byte=167,
        format=ScalarType.INT16,
        description="Time basis code.",
    ),
    HeaderField(
        name="trace_weighting_factor",
        byte=169,
        format=ScalarType.INT16,
        description="Trace weighting factor.",
    ),
    HeaderField(
        name="group_num_roll_switch",
        byte=171,
        format=ScalarType.INT16,
        description="Geophone group number of roll switch position one.",
    ),
    HeaderField(
        name="group_num_first_trace",
        byte=173,
        format=ScalarType.INT16,
        description="Geophone group number of trace number one, original field record.",
    ),
    HeaderField(
        name="group_num_last_trace",
        byte=175,
        format=ScalarType.INT16,
        description="Geophone group number of last trace, original field record.",
    ),
    HeaderField(
        name="gap_size",
        byte=177,
        format=ScalarType.INT16,
        description="Gap size (total number of groups dropped).",
    ),
    HeaderField(
        name="taper_overtravel",
        byte=179,
        format=ScalarType.INT16,
        description="Overtravel associated with taper.",
    ),
]


TRC_HDR_FIELDS_REV1 = TRC_HDR_FIELDS_REV0 + [
    HeaderField(
        name="cdp_x",
        byte=181,
        format=ScalarType.INT32,
        description="X coordinate of ensemble (CDP) position of this trace.",
    ),
    HeaderField(
        name="cdp_y",
        byte=185,
        format=ScalarType.INT32,
        description="Y coordinate of ensemble (CDP) position of this trace.",
    ),
    HeaderField(
        name="inline",
        byte=189,
        format=ScalarType.INT32,
        description="For 3D poststack data, in-line number.",
    ),
    HeaderField(
        name="crossline",
        byte=193,
        format=ScalarType.INT32,
        description="For 3D poststack data, cross-line number.",
    ),
    HeaderField(
        name="shot_point",
        byte=197,
        format=ScalarType.INT32,
        description="Shot-point number.",
    ),
    HeaderField(
        name="shot_point_scalar",
        byte=201,
        format=ScalarType.INT16,
        description="Scalar to be applied to shotpoint number.",
    ),
    HeaderField(
        name="trace_value_units",
        byte=203,
        format=ScalarType.INT16,
        description="Trace value measurement unit.",
    ),
    HeaderField(
        name="transduction_const_mantissa",
        byte=205,
        format=ScalarType.INT32,
        description="Transduction constant mantissa.",
    ),
    HeaderField(
        name="transduction_const_exponent",
        byte=209,
        format=ScalarType.INT16,
        description="Transduction constant exponent.",
    ),
    HeaderField(
        name="transduction_units",
        byte=211,
        format=ScalarType.INT16,
        description="Transduction units.",
    ),
    HeaderField(
        name="device_trace_id",
        byte=213,
        format=ScalarType.INT16,
        description="Device/Trace identifier.",
    ),
    HeaderField(
        name="times_scalar",
        byte=215,
        format=ScalarType.INT16,
        description="Scalar to be applied to times specified in bytes 109-114.",
    ),
    HeaderField(
        name="source_type_orientation",
        byte=217,
        format=ScalarType.INT16,
        description="Source type/orientation.",
    ),
    HeaderField(
        name="source_energy_dir_mantissa",
        byte=219,
        format=ScalarType.INT32,
        description="Source Energy Direction with respect to vertical, mantissa.",
    ),
    HeaderField(
        name="source_energy_dir_exponent",
        byte=223,
        format=ScalarType.INT16,
        description="Source Energy Direction with respect to vertical, exponent.",
    ),
    HeaderField(
        name="source_measurement_mantissa",
        byte=225,
        format=ScalarType.INT32,
        description="Source measurement mantissa.",
    ),
    HeaderField(
        name="source_measurement_exponent",
        byte=229,
        format=ScalarType.INT16,
        description="Source measurement exponent.",
    ),
    HeaderField(
        name="source_measurement_units",
        byte=231,
        format=ScalarType.INT16,
        description="Source measurement unit.",
    ),
]

TRC_HDR_FIELDS_REV2 = TRC_HDR_FIELDS_REV1 + [
    HeaderField(
        name="trace_header_name",
        byte=233,
        format=ScalarType.STRING8,
        description='Either binary zeros or the eight-character trace header name "SEG00000".',
    ),
]


TraceFieldRev0 = StrEnum(  # type: ignore[misc]
    "TraceHeaderFieldRev0",
    {field.name.upper(): field.name for field in TRC_HDR_FIELDS_REV0},
)
TraceFieldRev1 = StrEnum(  # type: ignore[misc]
    "TraceHeaderFieldRev1",
    {field.name.upper(): field.name for field in TRC_HDR_FIELDS_REV1},
)
TraceFieldRev2 = StrEnum(  # type: ignore[misc]
    "TraceHeaderFieldRev2",
    {field.name.upper(): field.name for field in TRC_HDR_FIELDS_REV2},
)
