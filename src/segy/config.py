"""SEG-Y parser configuration."""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

from segy.schema import Endianness


class SegyBaseSettings(BaseSettings):
    """Base class for settings."""

    model_config = SettingsConfigDict(extra="ignore", case_sensitive=True)


class SegyFieldSetting(SegyBaseSettings):
    """Setting class to configure a field (key or override)."""

    KEY: str = Field(...)
    VALUE: int | None = Field(...)


class SamplesPerTraceSetting(SegyFieldSetting):
    """Configuration for samples per trace parsing."""

    KEY: str = "samples_per_trace"
    VALUE: int | None = None


class SampleIntervalSetting(SegyFieldSetting):
    """Configuration for samples interval parsing."""

    KEY: str = "sample_interval"
    VALUE: int | None = None


class ExtendedTextHeaderSetting(SegyFieldSetting):
    """Configuration for extended textual headers parsing."""

    KEY: str = "extended_textual_headers"
    VALUE: int | None = None


class SegyBinaryHeaderSettings(SegyBaseSettings):
    """SEG-Y binary header parsing settings."""

    SAMPLES_PER_TRACE: SamplesPerTraceSetting = SamplesPerTraceSetting()
    SAMPLE_INTERVAL: SampleIntervalSetting = SampleIntervalSetting()
    EXTENDED_TEXT_HEADER: ExtendedTextHeaderSetting = ExtendedTextHeaderSetting()


class SegyFileSettings(SegyBaseSettings):
    """SEG-Y file parsing settings."""

    BINARY: SegyBinaryHeaderSettings = SegyBinaryHeaderSettings()
    ENDIAN: Endianness | None = Field(default=Endianness.BIG)
    REVISION: int | None = Field(default=None)

    USE_PANDAS: bool = Field(default=True)

    model_config = SettingsConfigDict(
        env_prefix="SEGY__",
        env_nested_delimiter="__",
    )
