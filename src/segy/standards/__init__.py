"""Implementation of SEG-Y standards."""

from segy.schema.segy import SegyStandard
from segy.standards.registry import get_segy_standard
from segy.standards.registry import register_segy_standard
from segy.standards.spec import REV0
from segy.standards.spec import REV1
from segy.standards.spec import REV2

register_segy_standard(version_or_name=0.0, spec=REV0)
register_segy_standard(version_or_name=1.0, spec=REV1)
register_segy_standard(version_or_name=2.0, spec=REV2)


__all__ = ["get_segy_standard", "SegyStandard"]
