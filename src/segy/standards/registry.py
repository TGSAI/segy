"""Implements a registry for various SEG-Y standards."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from segy.schema import SegySpec


logger = logging.getLogger(__name__)


segy_standard_registry = {}


def register_segy_standard(version_or_name: float | str, spec: SegySpec) -> None:
    """Register a SEG-Y standard with its spec."""
    logger.info("Registered new SEG-Y spec: %s", version_or_name)
    segy_standard_registry[version_or_name] = spec


def get_segy_standard(version_or_name: float | str) -> SegySpec:
    """Get a registered SEG-Y standard's spec."""
    spec = segy_standard_registry.get(version_or_name)

    if spec is None:
        msg = (
            f"Unknown or unsupported SEG-Y spec: {version_or_name}. If you "
            f"would like to use {version_or_name}, please register it with "
            f"the `SegySpecFactory` using its `register_spec` method."
        )
        logger.error(msg)
        raise NotImplementedError(msg)

    return spec.model_copy(deep=True)
