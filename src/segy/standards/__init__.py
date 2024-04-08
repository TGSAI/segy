"""Implementation of SEG-Y standards."""

from segy.schema import SegyStandard
from segy.standards.registry import get_segy_standard
from segy.standards.registry import register_segy_standard
from segy.standards.rev0 import rev0_segy
from segy.standards.rev1 import rev1_segy
from segy.standards.rev2 import rev2_segy

register_segy_standard(SegyStandard.REV0, rev0_segy)
register_segy_standard(SegyStandard.REV1, rev1_segy)
register_segy_standard(SegyStandard.REV2, rev2_segy)


__all__ = ["get_segy_standard", "SegyStandard"]
