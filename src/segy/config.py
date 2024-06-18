"""SEG-Y parser configuration."""

from __future__ import annotations

from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

from segy.schema import Endianness


class SegyBaseSettings(BaseSettings):
    """Base class for settings."""

    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)


class BinaryHeaderSettings(SegyBaseSettings):
    """SEG-Y binary header parsing overrides.

    Any value that is set to an integer will override what is parsed from
    the binary header in the actual file.
    """

    samples_per_trace: int | None = Field(
        default=None, description="Override samples per trace."
    )
    sample_interval: int | None = Field(
        default=None, description="Override sample interval."
    )
    ext_text_header: int | None = Field(
        default=None, description="Override extended text headers."
    )
    revision: int | float | None = Field(
        default=None, description="SEG-Y revision of the file."
    )


class SegySettings(SegyBaseSettings):
    """SEG-Y file parsing settings."""

    binary: BinaryHeaderSettings = Field(
        default_factory=BinaryHeaderSettings,
        description="Overrides for binary file header settings.",
    )
    endianness: Endianness = Field(
        default=Endianness.BIG,
        description="Override the inferred endianness of the file.",
    )
    storage_options: dict[str, Any] = Field(
        default_factory=dict,
        description="Storage options to pass to the storage backend.",
    )

    model_config = SettingsConfigDict(
        env_prefix="SEGY__",
        env_nested_delimiter="__",
    )
