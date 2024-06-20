"""Seismic Unix trace header mappings.

We added aliases for SEG-Y Rev0 equivalent fields.

Aliases for extended SU headers (after Rev0 before Rev1) are not
included because the types of them don't match SEG-Y Rev1, and we
don't want to point to corrupt data if user expects SU type data.

Source:
https://github-wiki-see.page/m/JohnWStockwellJr/SeisUnix/wiki/Seismic-Unix-data-format
"""

from segy.standards.fields import trace

# fmt: off
SEIS_UNIX_TRACE_FIELD_MAP = {
    "tracl": trace.Rev0.TRACE_SEQ_NUM_LINE,
    "tracr": trace.Rev0.TRACE_SEQ_NUM_REEL,
    "fldr": trace.Rev0.ORIG_FIELD_RECORD_NUM,
    "tracf": trace.Rev0.TRACE_NUM_ORIG_RECORD,
    "ep": trace.Rev0.ENERGY_SOURCE_POINT_NUM,
    "cdp": trace.Rev0.ENSEMBLE_NUM,
    "cdpt": trace.Rev0.TRACE_NUM_ENSEMBLE,
    "trid": trace.Rev0.TRACE_ID_CODE,
    "nvs": trace.Rev0.VERTICALLY_SUMMED_TRACES,
    "nhs": trace.Rev0.HORIZONTALLY_STACKED_TRACES,
    "duse": trace.Rev0.DATA_USE,
    "offset": trace.Rev0.SOURCE_TO_RECEIVER_DISTANCE,
    "gelev": trace.Rev0.RECEIVER_GROUP_ELEVATION,
    "selev": trace.Rev0.SOURCE_SURFACE_ELEVATION,
    "sdepth": trace.Rev0.SOURCE_DEPTH_BELOW_SURFACE,
    "gdel": trace.Rev0.RECEIVER_DATUM_ELEVATION,
    "sdel": trace.Rev0.SOURCE_DATUM_ELEVATION,
    "swdep": trace.Rev0.SOURCE_WATER_DEPTH,
    "gwdep": trace.Rev0.RECEIVER_WATER_DEPTH,
    "scalel": trace.Rev0.ELEVATION_DEPTH_SCALAR,
    "scalco": trace.Rev0.COORDINATE_SCALAR,
    "sx": trace.Rev0.SOURCE_COORD_X,
    "sy": trace.Rev0.SOURCE_COORD_Y,
    "gx": trace.Rev0.GROUP_COORD_X,
    "gy": trace.Rev0.GROUP_COORD_Y,
    "counit": trace.Rev0.COORDINATE_UNIT,
    "wevel": trace.Rev0.WEATHERING_VELOCITY,
    "swevel": trace.Rev0.SUBWEATHERING_VELOCITY,
    "sut": trace.Rev0.SOURCE_UPHOLE_TIME,
    "gut": trace.Rev0.GROUP_UPHOLE_TIME,
    "sstat": trace.Rev0.SOURCE_STATIC_CORRECTION,
    "gstat": trace.Rev0.RECEIVER_STATIC_CORRECTION,
    "tstat": trace.Rev0.TOTAL_STATIC_APPLIED,
    "laga": trace.Rev0.LAG_TIME_A,
    "lagb": trace.Rev0.LAG_TIME_B,
    "delrt": trace.Rev0.DELAY_RECORDING_TIME,
    "muts": trace.Rev0.MUTE_TIME_START,
    "mute": trace.Rev0.MUTE_TIME_END,
    "ns": trace.Rev0.SAMPLES_PER_TRACE,
    "dt": trace.Rev0.SAMPLE_INTERVAL,
    "gain": trace.Rev0.INSTRUMENT_GAIN_TYPE,
    "igc": trace.Rev0.INSTRUMENT_GAIN_CONST,
    "igi": trace.Rev0.INSTRUMENT_GAIN_INITIAL,
    "corr": trace.Rev0.CORRELATED_DATA,
    "sfs": trace.Rev0.SWEEP_FREQ_START,
    "sfe": trace.Rev0.SWEEP_FREQ_END,
    "slen": trace.Rev0.SWEEP_LENGTH,
    "styp": trace.Rev0.SWEEP_TYPE,
    "stas": trace.Rev0.SWEEP_TAPER_START,
    "stae": trace.Rev0.SWEEP_TAPER_END,
    "tatyp": trace.Rev0.TAPER_TYPE,
    "afilf": trace.Rev0.ALIAS_FILTER_FREQ,
    "afils": trace.Rev0.ALIAS_FILTER_SLOPE,
    "nofilf": trace.Rev0.NOTCH_FILTER_FREQ,
    "nofils": trace.Rev0.NOTCH_FILTER_SLOPE,
    "lcf": trace.Rev0.LOW_CUT_FREQ,
    "hcf": trace.Rev0.HIGH_CUT_FREQ,
    "lcs": trace.Rev0.LOW_CUT_SLOPE,
    "hcs": trace.Rev0.HIGH_CUT_SLOPE,
    "year": trace.Rev0.YEAR_RECORDED,
    "day": trace.Rev0.DAY_OF_YEAR,
    "hour": trace.Rev0.HOUR_OF_DAY,
    "minute": trace.Rev0.MINUTE_OF_HOUR,
    "sec": trace.Rev0.SECOND_OF_MINUTE,
    "timbas": trace.Rev0.TIME_BASIS_CODE,
    "trwf": trace.Rev0.TRACE_WEIGHTING_FACTOR,
    "grnors": trace.Rev0.GROUP_NUM_ROLL_SWITCH,
    "grnofr": trace.Rev0.GROUP_NUM_FIRST_TRACE,
    "grnlof": trace.Rev0.GROUP_NUM_LAST_TRACE,
    "gaps": trace.Rev0.GAP_SIZE,
    "otrav": trace.Rev0.TAPER_OVERTRAVEL,
}
# fmt: on
