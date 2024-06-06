"""SEG-Y Revision 0 Specification."""

from segy.schema import Endianness
from segy.schema import HeaderSpec
from segy.schema import ScalarType
from segy.schema import SegySpec
from segy.schema import SegyStandard
from segy.schema import TextHeaderEncoding
from segy.schema import TextHeaderSpec
from segy.schema import TraceDataSpec
from segy.schema import TraceSpec
from segy.standards.fields import BIN_HDR_FIELDS_REV0
from segy.standards.fields import BIN_HDR_FIELDS_REV1
from segy.standards.fields import BIN_HDR_FIELDS_REV2
from segy.standards.fields import TRC_HDR_FIELDS_REV0
from segy.standards.fields import TRC_HDR_FIELDS_REV1
from segy.standards.fields import TRC_HDR_FIELDS_REV2

text_header_ebcdic_3200 = TextHeaderSpec(
    rows=40,
    cols=80,
    offset=0,
    encoding=TextHeaderEncoding.EBCDIC,
    format=ScalarType.UINT8,  # noqa: A003
)

ext_text_header_ebcdic_3200 = TextHeaderSpec(
    rows=40,
    cols=80,
    offset=3600,
    encoding=TextHeaderEncoding.EBCDIC,
    format=ScalarType.UINT8,  # noqa: A003
)

REV0 = SegySpec(
    segy_standard=SegyStandard.REV0,
    endianness=Endianness.BIG,
    text_header=text_header_ebcdic_3200,
    binary_header=HeaderSpec(fields=BIN_HDR_FIELDS_REV0, item_size=400, offset=3200),
    trace=TraceSpec(
        data_spec=TraceDataSpec(format=ScalarType.IBM32),
        header_spec=HeaderSpec(fields=TRC_HDR_FIELDS_REV0, item_size=240),
    ),
)

REV1 = SegySpec(
    segy_standard=SegyStandard.REV1,
    endianness=Endianness.BIG,
    text_header=text_header_ebcdic_3200,
    binary_header=HeaderSpec(fields=BIN_HDR_FIELDS_REV1, item_size=400, offset=3200),
    ext_text_header=ext_text_header_ebcdic_3200,
    trace=TraceSpec(
        data_spec=TraceDataSpec(format=ScalarType.IBM32),
        header_spec=HeaderSpec(fields=TRC_HDR_FIELDS_REV1, item_size=240),
    ),
)

REV2 = SegySpec(
    segy_standard=SegyStandard.REV2,
    endianness=Endianness.BIG,
    text_header=text_header_ebcdic_3200,
    binary_header=HeaderSpec(fields=BIN_HDR_FIELDS_REV2, item_size=400, offset=3200),
    ext_text_header=ext_text_header_ebcdic_3200,
    trace=TraceSpec(
        data_spec=TraceDataSpec(format=ScalarType.IBM32),
        header_spec=HeaderSpec(fields=TRC_HDR_FIELDS_REV2, item_size=240),
    ),
)
