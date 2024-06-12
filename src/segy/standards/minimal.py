"""SEG-Y file specification with minimal, mandatory fields."""

from segy.schema.base import Endianness
from segy.schema.format import ScalarType
from segy.schema.format import TextHeaderEncoding
from segy.schema.header import HeaderSpec
from segy.schema.segy import SegySpec
from segy.schema.text_header import TextHeaderSpec
from segy.schema.trace import TraceDataSpec
from segy.schema.trace import TraceSpec
from segy.standards.fields import binary

BIN_HDR_FIELDS = [
    binary.Rev0.SAMPLE_INTERVAL,
    binary.Rev0.ORIG_SAMPLE_INTERVAL,
    binary.Rev0.SAMPLES_PER_TRACE,
    binary.Rev0.ORIG_SAMPLES_PER_TRACE,
    binary.Rev0.DATA_SAMPLE_FORMAT,
    binary.Rev1.SEGY_REVISION,
    binary.Rev1.FIXED_LENGTH_TRACE_FLAG,
    binary.Rev1.NUM_EXTENDED_TEXT_HEADERS,
]
BIN_HDR_FIELD_MODELS = [field.model for field in BIN_HDR_FIELDS]

textual_file_header = TextHeaderSpec(
    rows=40,
    cols=80,
    offset=0,
    encoding=TextHeaderEncoding.EBCDIC,
)

minimal_segy = SegySpec(
    segy_standard=None,
    text_header=textual_file_header,
    binary_header=HeaderSpec(fields=BIN_HDR_FIELD_MODELS, item_size=400, offset=3200),
    trace=TraceSpec(
        header=HeaderSpec(fields=[], item_size=240),
        data=TraceDataSpec(format=ScalarType.IBM32),
    ),
    endianness=Endianness.BIG,
)
