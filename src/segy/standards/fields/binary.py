"""SEG-Y binary header definitions."""

from segy.standards.fields.base import SegStandardEnum


# fmt: off
class Rev0(SegStandardEnum):
    """Definition of SEG-Y Rev0 binary headers."""

    JOB_ID                            = (1, "int32")
    LINE_NUM                          = (5, "int32")
    REEL_NUM                          = (9, "int32")
    DATA_TRACES_PER_ENSEMBLE          = (13, "int16")
    AUX_TRACES_PER_ENSEMBLE           = (15, "int16")
    SAMPLE_INTERVAL                   = (17, "int16")
    ORIG_SAMPLE_INTERVAL              = (19, "int16")
    SAMPLES_PER_TRACE                 = (21, "int16")
    ORIG_SAMPLES_PER_TRACE            = (23, "int16")
    DATA_SAMPLE_FORMAT                = (25, "int16")
    ENSEMBLE_FOLD                     = (27, "int16")
    TRACE_SORTING_CODE                = (29, "int16")
    VERTICAL_SUM_CODE                 = (31, "int16")
    SWEEP_FREQ_START                  = (33, "int16")
    SWEEP_FREQ_END                    = (35, "int16")
    SWEEP_LENGTH                      = (37, "int16")
    SWEEP_TYPE_CODE                   = (39, "int16")
    SWEEP_TRACE_NUM                   = (41, "int16")
    SWEEP_TAPER_START                 = (43, "int16")
    SWEEP_TAPER_END                   = (45, "int16")
    TAPER_TYPE_CODE                   = (47, "int16")
    CORRELATED_DATA_CODE              = (49, "int16")
    BINARY_GAIN_CODE                  = (51, "int16")
    AMP_RECOVERY_CODE                 = (53, "int16")
    MEASUREMENT_SYSTEM_CODE           = (55, "int16")
    IMPULSE_POLARITY_CODE             = (57, "int16")
    VIBRATORY_POLARITY_CODE           = (59, "int16")


class Rev1(SegStandardEnum):
    """Definition of SEG-Y Rev1 binary headers."""

    SEGY_REVISION                     = (301, "int16")
    FIXED_LENGTH_TRACE_FLAG           = (303, "int16")
    NUM_EXTENDED_TEXT_HEADERS         = (305, "int16")


class Rev2(SegStandardEnum):
    """Definition of SEG-Y Rev2 binary headers."""

    EXTENDED_DATA_TRACES_PER_ENSEMBLE = (61, "int32")
    EXTENDED_AUX_TRACES_PER_ENSEMBLE  = (65, "int32")
    EXTENDED_SAMPLES_PER_TRACE        = (69, "int32")
    EXTENDED_SAMPLE_INTERVAL          = (73, "float64")
    EXTENDED_ORIG_SAMPLE_INTERVAL     = (81, "float64")
    EXTENDED_ORIG_SAMPLES_PER_TRACE   = (89, "int32")
    EXTENDED_ENSEMBLE_FOLD            = (93, "int32")
    BYTE_ORDER                        = (97, "int32")
    SEGY_REVISION_MAJOR               = (301, "uint8")
    SEGY_REVISION_MINOR               = (302, "uint8")
    MAX_EXTENDED_TRACE_HEADERS        = (307, "int16")
    SURVEY_TYPE                       = (309, "int16")
    TIME_BASIS_CODE                   = (311, "int16")
    NUM_TRACES                        = (313, "uint64")
    BYTE_OFFSET_FIRST_TRACE           = (321, "uint64")
    NUM_DATA_TRAILER_STANZAS          = (329, "int32")
# fmt:on
