"""Implements a registry for various SEG-Y standards."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from segy.schema import SegySpec
    from segy.schema import SegyStandard

segy_standard_registry = {}


def register_segy_standard(spec_type: SegyStandard, spec: SegySpec) -> None:
    """Register a SEG-Y standard with its spec."""
    segy_standard_registry[spec_type] = spec


def get_segy_standard(spec_type: SegyStandard) -> SegySpec:
    """Get a registered SEG-Y standard's spec."""
    spec = segy_standard_registry.get(spec_type)

    if spec is None:
        msg = (
            f"Unknown or unsupported SEG-Y spec: {spec_type}. If you "
            f"would like to use {spec_type}, please register it with "
            f"the `SegySpecFactory` using its `register_spec` method."
        )
        raise NotImplementedError(msg)

    return spec.model_copy(deep=True)
