"""Segyio binary and trace header mappings.

We added segyio header name mappings to canonical header names we define
for SEG-Y headers.

segyio implements Rev0, Rev1, and some Rev2 header fields in their standard.
We have mapped everything to the forms we defined.

Sources:
https://segyio.readthedocs.io/en/latest/segyio.html#trace-header-keys
https://segyio.readthedocs.io/en/latest/segyio.html#binary-header-keys
"""

from segy.standards.fields import binary
from segy.standards.fields import trace

# fmt: off
SEGYIO_BIN_FIELD_MAP = {
    "JobID":                 binary.Rev0.JOB_ID,
    "LineNumber":            binary.Rev0.LINE_NUM,
    "ReelNumber":            binary.Rev0.REEL_NUM,
    "Traces":                binary.Rev0.DATA_TRACES_PER_ENSEMBLE,
    "AuxTraces":             binary.Rev0.AUX_TRACES_PER_ENSEMBLE,
    "Interval":              binary.Rev0.SAMPLE_INTERVAL,
    "IntervalOriginal":      binary.Rev0.ORIG_SAMPLE_INTERVAL,
    "Samples":               binary.Rev0.SAMPLES_PER_TRACE,
    "SamplesOriginal":       binary.Rev0.ORIG_SAMPLES_PER_TRACE,
    "Format":                binary.Rev0.DATA_SAMPLE_FORMAT,
    "EnsembleFold":          binary.Rev0.ENSEMBLE_FOLD,
    "SortingCode":           binary.Rev0.TRACE_SORTING_CODE,
    "VerticalSum":           binary.Rev0.VERTICAL_SUM_CODE,
    "SweepFrequencyStart":   binary.Rev0.SWEEP_FREQ_START,
    "SweepFrequencyEnd":     binary.Rev0.SWEEP_FREQ_END,
    "SweepLength":           binary.Rev0.SWEEP_LENGTH,
    "Sweep":                 binary.Rev0.SWEEP_TYPE_CODE,
    "SweepChannel":          binary.Rev0.SWEEP_TRACE_NUM,
    "SweepTaperStart":       binary.Rev0.SWEEP_TAPER_START,
    "SweepTaperEnd":         binary.Rev0.SWEEP_TAPER_END,
    "Taper":                 binary.Rev0.TAPER_TYPE_CODE,
    "CorrelatedTraces":      binary.Rev0.CORRELATED_DATA_CODE,
    "BinaryGainRecovery":    binary.Rev0.BINARY_GAIN_CODE,
    "AmplitudeRecovery":     binary.Rev0.AMP_RECOVERY_CODE,
    "MeasurementSystem":     binary.Rev0.MEASUREMENT_SYSTEM_CODE,
    "ImpulseSignalPolarity": binary.Rev0.IMPULSE_POLARITY_CODE,
    "VibratoryPolarity":     binary.Rev0.VIBRATORY_POLARITY_CODE,
    "ExtTraces":             binary.Rev2.EXTENDED_DATA_TRACES_PER_ENSEMBLE,
    "ExtAuxTraces":          binary.Rev2.EXTENDED_AUX_TRACES_PER_ENSEMBLE,
    "ExtSamples":            binary.Rev2.EXTENDED_SAMPLES_PER_TRACE,
    "ExtSamplesOriginal":    binary.Rev2.EXTENDED_ORIG_SAMPLES_PER_TRACE,
    "ExtEnsembleFold":       binary.Rev2.EXTENDED_ENSEMBLE_FOLD,
    "SEGYRevision":          binary.Rev2.SEGY_REVISION_MAJOR,
    "SEGYRevisionMinor":     binary.Rev2.SEGY_REVISION_MINOR,
    "TraceFlag":             binary.Rev1.FIXED_LENGTH_TRACE_FLAG,
    "ExtendedHeaders":       binary.Rev1.NUM_EXTENDED_TEXT_HEADERS,
}

SEGYIO_TRACE_FIELD_MAP = {
    "TRACE_SEQUENCE_LINE":                    trace.Rev0.TRACE_SEQ_NUM_LINE,
    "TRACE_SEQUENCE_FILE":                    trace.Rev0.TRACE_SEQ_NUM_REEL,
    "FieldRecord":                            trace.Rev0.ORIG_FIELD_RECORD_NUM,
    "TraceNumber":                            trace.Rev0.TRACE_NUM_ORIG_RECORD,
    "EnergySourcePoint":                      trace.Rev0.ENERGY_SOURCE_POINT_NUM,
    "CDP":                                    trace.Rev0.ENSEMBLE_NUM,
    "CDP_TRACE":                              trace.Rev0.TRACE_NUM_ENSEMBLE,
    "TraceIdentificationCode":                trace.Rev0.TRACE_ID_CODE,
    "NSummedTraces":                          trace.Rev0.VERTICALLY_SUMMED_TRACES,
    "NStackedTraces":                         trace.Rev0.HORIZONTALLY_STACKED_TRACES,
    "DataUse":                                trace.Rev0.DATA_USE,
    "offset":                                 trace.Rev0.SOURCE_TO_RECEIVER_DISTANCE,
    "ReceiverGroupElevation":                 trace.Rev0.RECEIVER_GROUP_ELEVATION,
    "SourceSurfaceElevation":                 trace.Rev0.SOURCE_SURFACE_ELEVATION,
    "SourceDepth":                            trace.Rev0.SOURCE_DEPTH_BELOW_SURFACE,
    "ReceiverDatumElevation":                 trace.Rev0.RECEIVER_DATUM_ELEVATION,
    "SourceDatumElevation":                   trace.Rev0.SOURCE_DATUM_ELEVATION,
    "SourceWaterDepth":                       trace.Rev0.SOURCE_WATER_DEPTH,
    "GroupWaterDepth":                        trace.Rev0.RECEIVER_WATER_DEPTH,
    "ElevationScalar":                        trace.Rev0.ELEVATION_DEPTH_SCALAR,
    "SourceGroupScalar":                      trace.Rev0.COORDINATE_SCALAR,
    "SourceX":                                trace.Rev0.SOURCE_COORD_X,
    "SourceY":                                trace.Rev0.SOURCE_COORD_Y,
    "GroupX":                                 trace.Rev0.GROUP_COORD_X,
    "GroupY":                                 trace.Rev0.GROUP_COORD_Y,
    "CoordinateUnits":                        trace.Rev0.COORDINATE_UNIT,
    "WeatheringVelocity":                     trace.Rev0.WEATHERING_VELOCITY,
    "SubWeatheringVelocity":                  trace.Rev0.SUBWEATHERING_VELOCITY,
    "SourceUpholeTime":                       trace.Rev0.SOURCE_UPHOLE_TIME,
    "GroupUpholeTime":                        trace.Rev0.GROUP_UPHOLE_TIME,
    "SourceStaticCorrection":                 trace.Rev0.SOURCE_STATIC_CORRECTION,
    "GroupStaticCorrection":                  trace.Rev0.RECEIVER_STATIC_CORRECTION,
    "TotalStaticApplied":                     trace.Rev0.TOTAL_STATIC_APPLIED,
    "LagTimeA":                               trace.Rev0.LAG_TIME_A,
    "LagTimeB":                               trace.Rev0.LAG_TIME_B,
    "DelayRecordingTime":                     trace.Rev0.DELAY_RECORDING_TIME,
    "MuteTimeStart":                          trace.Rev0.MUTE_TIME_START,
    "MuteTimeEND":                            trace.Rev0.MUTE_TIME_END,
    "TRACE_SAMPLE_COUNT":                     trace.Rev0.SAMPLES_PER_TRACE,
    "TRACE_SAMPLE_INTERVAL":                  trace.Rev0.SAMPLE_INTERVAL,
    "GainType":                               trace.Rev0.INSTRUMENT_GAIN_TYPE,
    "InstrumentGainConstant":                 trace.Rev0.INSTRUMENT_GAIN_CONST,
    "InstrumentInitialGain":                  trace.Rev0.INSTRUMENT_GAIN_INITIAL,
    "Correlated":                             trace.Rev0.CORRELATED_DATA,
    "SweepFrequencyStart":                    trace.Rev0.SWEEP_FREQ_START,
    "SweepFrequencyEnd":                      trace.Rev0.SWEEP_FREQ_END,
    "SweepLength":                            trace.Rev0.SWEEP_LENGTH,
    "SweepType":                              trace.Rev0.SWEEP_TYPE,
    "SweepTraceTaperLengthStart":             trace.Rev0.SWEEP_TAPER_START,
    "SweepTraceTaperLengthEnd":               trace.Rev0.SWEEP_TAPER_END,
    "TaperType":                              trace.Rev0.TAPER_TYPE,
    "AliasFilterFrequency":                   trace.Rev0.ALIAS_FILTER_FREQ,
    "AliasFilterSlope":                       trace.Rev0.ALIAS_FILTER_SLOPE,
    "NotchFilterFrequency":                   trace.Rev0.NOTCH_FILTER_FREQ,
    "NotchFilterSlope":                       trace.Rev0.NOTCH_FILTER_SLOPE,
    "LowCutFrequency":                        trace.Rev0.LOW_CUT_FREQ,
    "HighCutFrequency":                       trace.Rev0.HIGH_CUT_FREQ,
    "LowCutSlope":                            trace.Rev0.LOW_CUT_SLOPE,
    "HighCutSlope":                           trace.Rev0.HIGH_CUT_SLOPE,
    "YearDataRecorded":                       trace.Rev0.YEAR_RECORDED,
    "DayOfYear":                              trace.Rev0.DAY_OF_YEAR,
    "HourOfDay":                              trace.Rev0.HOUR_OF_DAY,
    "MinuteOfHour":                           trace.Rev0.MINUTE_OF_HOUR,
    "SecondOfMinute":                         trace.Rev0.SECOND_OF_MINUTE,
    "TimeBaseCode":                           trace.Rev0.TIME_BASIS_CODE,
    "TraceWeightingFactor":                   trace.Rev0.TRACE_WEIGHTING_FACTOR,
    "GeophoneGroupNumberRoll1":               trace.Rev0.GROUP_NUM_ROLL_SWITCH,
    "GeophoneGroupNumberFirstTraceOrigField": trace.Rev0.GROUP_NUM_FIRST_TRACE,
    "GeophoneGroupNumberLastTraceOrigField":  trace.Rev0.GROUP_NUM_LAST_TRACE,
    "GapSize":                                trace.Rev0.GAP_SIZE,
    "OverTravel":                             trace.Rev0.TAPER_OVERTRAVEL,
    "CDP_X":                                  trace.Rev1.CDP_X,
    "CDP_Y":                                  trace.Rev1.CDP_Y,
    "INLINE_3D":                              trace.Rev1.INLINE,
    "CROSSLINE_3D":                           trace.Rev1.CROSSLINE,
    "ShotPoint":                              trace.Rev1.SHOT_POINT,
    "ShotPointScalar":                        trace.Rev1.SHOT_POINT_SCALAR,
    "TraceValueMeasurementUnit":              trace.Rev1.TRACE_VALUE_UNIT,
    "TransductionConstantMantissa":           trace.Rev1.TRANSDUCTION_CONST_MANTISSA,
    "TransductionConstantPower":              trace.Rev1.TRANSDUCTION_CONST_EXPONENT,
    "TransductionUnit":                       trace.Rev1.TRANSDUCTION_UNIT,
    "TraceIdentifier":                        trace.Rev1.DEVICE_TRACE_ID,
    "ScalarTraceHeader":                      trace.Rev1.TIMES_SCALAR,
    "SourceType":                             trace.Rev1.SOURCE_TYPE_ORIENTATION,
    "SourceEnergyDirectionMantissa":          trace.Rev1.SOURCE_ENERGY_DIR_MANTISSA,
    "SourceEnergyDirectionExponent":          trace.Rev1.SOURCE_ENERGY_DIR_EXPONENT,
    "SourceMeasurementMantissa":              trace.Rev1.SOURCE_MEASUREMENT_MANTISSA,
    "SourceMeasurementExponent":              trace.Rev1.SOURCE_MEASUREMENT_EXPONENT,
    "SourceMeasurementUnit":                  trace.Rev1.SOURCE_MEASUREMENT_UNIT,
}
# fmt: on
