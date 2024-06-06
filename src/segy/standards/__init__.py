"""Implementation of SEG-Y standards."""

from segy.schema import SegyStandard
from segy.standards.registry import get_segy_standard
from segy.standards.registry import register_segy_standard
from segy.standards.spec import REV0
from segy.standards.spec import REV1
from segy.standards.spec import REV2

register_segy_standard(SegyStandard.REV0, REV0)
register_segy_standard(SegyStandard.REV1, REV1)
register_segy_standard(SegyStandard.REV2, REV2)


__all__ = ["get_segy_standard", "SegyStandard"]
