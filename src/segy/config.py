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


class SegyFieldSetting(SegyBaseSettings):
    """Setting class to configure a field (key or override)."""

    key: str = Field(...)
    value: int | None = Field(...)


class SamplesPerTraceSetting(SegyFieldSetting):
    """Configuration for samples per trace parsing."""

    key: str = "samples_per_trace"
    value: int | None = None


class SampleIntervalSetting(SegyFieldSetting):
    """Configuration for samples interval parsing."""

    key: str = "sample_interval"
    value: int | None = None


class ExtendedTextHeaderSetting(SegyFieldSetting):
    """Configuration for extended textual headers parsing."""

    key: str = "extended_textual_headers"
    value: int | None = None


class SegyBinaryHeaderSettings(SegyBaseSettings):
    """SEG-Y binary header parsing settings."""

    samples_per_trace: SamplesPerTraceSetting = SamplesPerTraceSetting()
    sample_interval: SampleIntervalSetting = SampleIntervalSetting()
    extended_text_header: ExtendedTextHeaderSetting = ExtendedTextHeaderSetting()


class SegyFileSettings(SegyBaseSettings):
    """SEG-Y file parsing settings."""

    binary: SegyBinaryHeaderSettings = Field(default_factory=SegyBinaryHeaderSettings)
    endianness: Endianness = Field(default=Endianness.BIG)
    revision: int | float | None = Field(default=None)

    storage_options: dict[str, Any] = Field(default_factory=dict)
    apply_transforms: bool = Field(default=True)

    model_config = SettingsConfigDict(
        env_prefix="SEGY__",
        env_nested_delimiter="__",
    )
