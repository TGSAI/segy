"""Tests for functions in segy.schema.registry."""

import pytest

from segy.schema import SegySpec
from segy.standards import SegyStandard
from segy.standards import get_segy_standard
from segy.standards.spec import REV0
from segy.standards.spec import REV1


@pytest.mark.parametrize(
    ("standard_enum", "standard_spec"),
    [(SegyStandard.REV0, REV0), (SegyStandard.REV1, REV1)],
)
def test_get_standard(standard_enum: SegyStandard, standard_spec: SegySpec) -> None:
    """Test retrieving SegyStandard from registry."""
    spec_copy = get_segy_standard(standard_enum)
    assert spec_copy == standard_spec
    assert id(spec_copy) != id(standard_spec)


def test_get_nonexistent_spec_error() -> None:
    """Test missing / non-existent SegyStandard from registry."""
    with pytest.raises(NotImplementedError):
        get_segy_standard("non_existent")
