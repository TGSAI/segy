"""Tests for functions in segy.schema.registry."""

import pytest

from segy.schema import SegyDescriptor
from segy.standards import SegyStandard
from segy.standards import get_segy_standard
from segy.standards.rev0 import rev0_segy
from segy.standards.rev1 import rev1_segy


@pytest.mark.parametrize(
    ("standard_enum", "standard_spec"),
    [(SegyStandard.REV0, rev0_segy), (SegyStandard.REV1, rev1_segy)],
)
def test_get_standard(
    standard_enum: SegyStandard, standard_spec: SegyDescriptor
) -> None:
    """Test retrieving SegyStandard from registry."""
    spec_copy = get_segy_standard(standard_enum)
    assert spec_copy == standard_spec
    assert id(spec_copy) != id(standard_spec)


def test_get_nonexistent_spec_error() -> None:
    """Test missing / non-existent SegyStandard from registry."""
    with pytest.raises(NotImplementedError):
        get_segy_standard("non_existent")  # type: ignore
