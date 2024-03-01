"""tests for functions in segy.schema.registry."""
import pytest

from segy.schema import SegyDescriptor
from segy.standards import SegyStandard
from segy.standards import rev0_segy
from segy.standards import rev1_segy
from segy.standards.registry import get_spec
from segy.standards.registry import register_spec


@pytest.mark.parametrize(
    ("standard_enum", "base_spec"),
    [(SegyStandard.REV0, rev0_segy), (SegyStandard.REV1, rev1_segy)],
)
def test_get_spec(standard_enum: SegyStandard, base_spec: SegyDescriptor) -> None:
    """Test retrieving SegyStandard from registry."""
    spec_copy = get_spec(SegyStandard(standard_enum))
    assert spec_copy == base_spec
    assert id(spec_copy) != id(base_spec)


def test_get_nonexistent_spec_error() -> None:
    """Test missing / non-existent SegyStandard from registry."""
    with pytest.raises(NotImplementedError):
        get_spec("non_existent")  # type: ignore


def test_register_custom_descriptor() -> None:
    """Test registering a custom descriptor."""
    spec = rev1_segy.customize(extended_text_spec=rev1_segy.text_file_header)
    register_spec(SegyStandard.CUSTOM, spec)
    assert get_spec(SegyStandard.CUSTOM) == spec


def test_register_nondescriptor_error() -> None:
    """Test if not providing a descriptor to registration."""
    msg = "spec_cls must be a subclass of SegyDescriptor."
    with pytest.raises(ValueError, match=msg):
        register_spec(SegyStandard.CUSTOM, "not_a_descriptor")  # type: ignore
