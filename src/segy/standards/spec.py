"""SEG-Y Revision 0 Specification."""

from segy.schema.base import Endianness
from segy.schema.format import ScalarType
from segy.schema.format import TextHeaderEncoding
from segy.schema.header import HeaderSpec
from segy.schema.segy import SegySpec
from segy.schema.segy import SegyStandard
from segy.schema.text_header import ExtendedTextHeaderSpec
from segy.schema.text_header import TextHeaderSpec
from segy.schema.trace import TraceDataSpec
from segy.schema.trace import TraceSpec
from segy.standards.fields import binary
from segy.standards.fields import trace

BIN_HDR_FIELDS_REV0 = [field.model for field in binary.Rev0]
TRC_HDR_FIELDS_REV0 = [field.model for field in trace.Rev0]

BIN_HDR_FIELDS_REV1 = BIN_HDR_FIELDS_REV0 + [field.model for field in binary.Rev1]
TRC_HDR_FIELDS_REV1 = TRC_HDR_FIELDS_REV0 + [field.model for field in trace.Rev1]

# Rev2 removes segy_revision and splits it into major/minor revision fields
BIN_HDR_FIELDS_REV2 = BIN_HDR_FIELDS_REV1 + [field.model for field in binary.Rev2]
BIN_HDR_FIELDS_REV2.remove(binary.Rev1.SEGY_REVISION.model)
BIN_HDR_FIELDS_REV2 = sorted(BIN_HDR_FIELDS_REV2, key=lambda f: f.byte)
TRC_HDR_FIELDS_REV2 = TRC_HDR_FIELDS_REV1 + [field.model for field in trace.Rev2]

# Rev2.1 splits maximum extended trace headers into two 16-bit.
# First reduces it from 32-bit to 16-bit
# Second gets new "survey type" field
BIN_HDR_FIELDS_REV21 = BIN_HDR_FIELDS_REV2 + [field.model for field in binary.Rev21]
BIN_HDR_FIELDS_REV21.remove(binary.Rev2.MAX_EXTENDED_TRACE_HEADERS.model)
BIN_HDR_FIELDS_REV21 = sorted(BIN_HDR_FIELDS_REV21, key=lambda f: f.byte)
TRC_HDR_FIELDS_REV21 = TRC_HDR_FIELDS_REV2  # no change


text_header_ebcdic = TextHeaderSpec(
    rows=40,
    cols=80,
    encoding=TextHeaderEncoding.EBCDIC,
)

ext_text_header_ebcdic_3200 = ExtendedTextHeaderSpec(spec=text_header_ebcdic)

REV0 = SegySpec(
    segy_standard=SegyStandard.REV0,
    endianness=Endianness.BIG,
    text_header=text_header_ebcdic,
    binary_header=HeaderSpec(fields=BIN_HDR_FIELDS_REV0, item_size=400, offset=3200),
    trace=TraceSpec(
        data=TraceDataSpec(format=ScalarType.IBM32),
        header=HeaderSpec(fields=TRC_HDR_FIELDS_REV0, item_size=240),
    ),
)

REV1 = SegySpec(
    segy_standard=SegyStandard.REV1,
    endianness=Endianness.BIG,
    text_header=text_header_ebcdic,
    binary_header=HeaderSpec(fields=BIN_HDR_FIELDS_REV1, item_size=400, offset=3200),
    ext_text_header=ext_text_header_ebcdic_3200,
    trace=TraceSpec(
        data=TraceDataSpec(format=ScalarType.IBM32),
        header=HeaderSpec(fields=TRC_HDR_FIELDS_REV1, item_size=240),
    ),
)

REV2 = SegySpec(
    segy_standard=SegyStandard.REV2,
    endianness=Endianness.BIG,
    text_header=text_header_ebcdic,
    binary_header=HeaderSpec(fields=BIN_HDR_FIELDS_REV2, item_size=400, offset=3200),
    ext_text_header=ext_text_header_ebcdic_3200,
    trace=TraceSpec(
        data=TraceDataSpec(format=ScalarType.IBM32),
        header=HeaderSpec(fields=TRC_HDR_FIELDS_REV2, item_size=240),
    ),
)

REV21 = SegySpec(
    segy_standard=SegyStandard.REV21,
    endianness=Endianness.BIG,
    text_header=text_header_ebcdic,
    binary_header=HeaderSpec(fields=BIN_HDR_FIELDS_REV21, item_size=400, offset=3200),
    ext_text_header=ext_text_header_ebcdic_3200,
    trace=TraceSpec(
        data=TraceDataSpec(format=ScalarType.IBM32),
        header=HeaderSpec(fields=TRC_HDR_FIELDS_REV21, item_size=240),
    ),
)
