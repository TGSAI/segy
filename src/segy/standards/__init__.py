"""Implementation of SEG-Y standards."""


from segy.schema import SegyStandard
from segy.standards.registry import register_spec
from segy.standards.rev0 import SegyDescriptorRev0
from segy.standards.rev1 import SegyDescriptorRev1

register_spec(SegyStandard.REV0, SegyDescriptorRev0)
register_spec(SegyStandard.REV1, SegyDescriptorRev1)


__all__ = ["SegyDescriptorRev0", "SegyDescriptorRev1"]
