"""Implementation of SEG-Y standards."""

from segy.schema import SegyStandard
from segy.standards.minimal import minimal_segy
from segy.standards.registry import register_spec
from segy.standards.rev0 import rev0_segy
from segy.standards.rev1 import rev1_segy

register_spec(SegyStandard.MINIMAL, minimal_segy)
register_spec(SegyStandard.REV0, rev0_segy)
register_spec(SegyStandard.REV1, rev1_segy)


__all__ = ["minimal_segy", "rev0_segy", "rev1_segy", "SegyStandard"]
