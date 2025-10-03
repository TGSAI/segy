"""SEG-Y parser configuration."""

from __future__ import annotations

from collections.abc import Mapping  # noqa: TCH003
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

from segy.schema import Endianness  # noqa: TCH001


class SegyBaseSettings(BaseSettings):
    """Base class for settings."""

    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)


class SegyHeaderOverrides(SegyBaseSettings):
    """SEG-Y header parsing overrides.

    Any value that is set to an integer will override what is parsed from
    the binary header in the actual file. If you override with a float
    please ensure that the field is defined as a float. If not, the float
    will get down-cast to an integer.
    """

    binary_header: Mapping[str, int | float] = Field(
        default_factory=dict,
        description="Header fields to override in binary header during read.",
    )
    trace_header: Mapping[str, int | float] = Field(
        default_factory=dict,
        description="Header fields to override in trace headers during read.",
    )
    model_config = SettingsConfigDict(env_prefix="SEGY_OVERRIDE_")


class SegyFileSettings(SegyBaseSettings):
    """SEG-Y file parsing settings."""

    endianness: Endianness | None = Field(
        default=None,
        description="Override the inferred endianness of the file.",
    )
    storage_options: dict[str, Any] = Field(
        default_factory=dict,
        description="Storage options to pass to the storage backend.",
    )
    model_config = SettingsConfigDict(env_prefix="SEGY_")
