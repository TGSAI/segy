"""SEG-Y file in-memory constructors."""

from __future__ import annotations

import struct
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from fsspec.core import url_to_fs

from segy.accessors import TraceAccessor
from segy.arrays import HeaderArray
from segy.config import SegyFileSettings
from segy.indexing import HeaderIndexer
from segy.indexing import SampleIndexer
from segy.indexing import TraceIndexer
from segy.schema import Endianness
from segy.schema import ScalarType
from segy.schema import SegyStandard
from segy.standards import get_segy_standard
from segy.standards.mapping import SEGY_FORMAT_MAP
from segy.standards.rev1 import rev1_binary_file_header
from segy.standards.rev1 import rev1_extended_text_header
from segy.transforms import TransformFactory
from segy.transforms import TransformPipeline

if TYPE_CHECKING:
    from fsspec import AbstractFileSystem
    from numpy.typing import NDArray

    from segy.indexing import AbstractIndexer
    from segy.schema import SegyDescriptor


def create_spec(
    standard: SegyStandard,
    endian: Endianness,
    sample_format: ScalarType | None = None,
) -> SegyDescriptor:
    """Create SegyDescriptor from SegyStandard, Endianness, and ScalarType."""
    spec = get_segy_standard(standard)
    spec.endianness = endian

    if sample_format is not None:
        spec.trace.sample_descriptor.format = sample_format

    return spec


def get_spec_from_settings(settings: SegyFileSettings) -> SegyDescriptor:
    """Get the SEG-Y spec from the settings instance."""
    standard = SegyStandard(settings.revision)
    return create_spec(standard, settings.endianness)


def read_default_binary_file_header_buffer(
    fs: AbstractFileSystem, url: str
) -> bytearray:
    """Read a binary file header from a URL."""
    buffer = fs.read_block(
        fn=url,
        offset=rev1_binary_file_header.offset,
        length=rev1_binary_file_header.itemsize,
    )

    return bytearray(buffer)


def unpack_binary_header(
    buffer: bytearray, endianness: Endianness
) -> tuple[int, float, int]:
    """Unpack binary header sample rate and revision."""
    format_ = f"{endianness.symbol}h"
    sample_increment = struct.unpack_from(format_, buffer, offset=16)[0]

    # Get sample format
    sample_format = struct.unpack_from(format_, buffer, offset=24)[0]

    # # Get revision. Per SEG-Y standard, there is a Q-point between the
    # bytes. Dividing by 2^8 to get the floating-point value of the revision.
    revision = struct.unpack_from(format_, buffer, offset=300)[0] / 256.0

    return sample_increment, revision, sample_format


def infer_spec_from_binary_header(buffer: bytearray) -> SegyDescriptor:
    """Try to infer SEG-Y file revision and endianness to build a SegyDescriptor."""
    for endianness in [Endianness.BIG, Endianness.LITTLE]:
        unpacked = unpack_binary_header(buffer, endianness)
        sample_increment, revision, sample_format_int = unpacked

        # Validate the inferred values.
        in_spec = revision in {0.0, 1.0}
        increment_is_positive = sample_increment > 0
        format_is_valid = sample_format_int in SEGY_FORMAT_MAP.values()

        if in_spec and increment_is_positive and format_is_valid:
            standard = SegyStandard(revision)
            sample_format: ScalarType = SEGY_FORMAT_MAP.inverse[sample_format_int]
            return create_spec(standard, endianness, sample_format)

    # If both fail, raise error.
    msg = "Could not infer SEG-Y standard. Please provide spec."
    raise ValueError(msg)


class SegyFile:
    """A SEG-Y file class that has various accessors.

    Args:
        url: Path to SEG-Y file on disk or remote store.
        spec: The schema / spec describing the SEG-Y file. This
            is optional and by default it will try to infer the
            SEG-Y standard from the binary header.
        settings: A settings instance to configure / override
            the SEG-Y parsing logic. Optional.
    """

    fs: AbstractFileSystem
    url: str

    def __init__(
        self,
        url: str,
        spec: SegyDescriptor | None = None,
        settings: SegyFileSettings | None = None,
    ):
        self.settings = SegyFileSettings() if settings is None else settings

        self.fs, self.url = url_to_fs(url, **self.settings.storage_options)

        self.spec = self._infer_spec() if spec is None else spec

        self._info = self.fs.info(self.url)
        self._parse_binary_header()
        self.accessors = TraceAccessor(self.spec.trace)

    @property
    def file_size(self) -> int:
        """Return file size in bytes."""
        size: int = self._info["size"]
        return size

    @property
    def num_ext_text(self) -> int:
        """Return number of extended text headers."""
        if self.spec.segy_standard == SegyStandard.REV0:
            return 0

        # Overriding from settings
        if self.settings.binary.extended_text_header.value is not None:
            return self.settings.binary.extended_text_header.value

        header_key = self.settings.binary.extended_text_header.key
        return int(self.binary_header[header_key][0])

    @property
    def samples_per_trace(self) -> int:
        """Return samples per trace in file based on spec."""
        return int(self.binary_header["samples_per_trace"].item())

    @property
    def sample_interval(self) -> int:
        """Return samples interval in file based on spec."""
        return int(self.binary_header["sample_interval"].item())

    @property
    def sample_labels(self) -> NDArray[np.int32]:
        """Return sample axis labels."""
        max_samp = self.sample_interval * self.samples_per_trace
        return np.arange(0, max_samp, self.sample_interval, dtype="int32")

    @property
    def num_traces(self) -> int:
        """Return number of traces in file based on size and spec."""
        file_textual_hdr_size = self.spec.text_file_header.itemsize
        file_bin_hdr_size = self.spec.binary_file_header.itemsize
        trace_size = self.spec.trace.itemsize

        file_metadata_size = file_textual_hdr_size + file_bin_hdr_size

        if self.num_ext_text > 0:
            self.spec.extended_text_header = rev1_extended_text_header

            ext_text_size = self.spec.extended_text_header.itemsize * self.num_ext_text
            file_metadata_size = file_metadata_size + ext_text_size

        return (self.file_size - file_metadata_size) // trace_size

    @cached_property
    def text_header(self) -> str:
        """Return textual file header."""
        text_hdr_desc = self.spec.text_file_header

        buffer = self.fs.read_block(
            fn=self.url,
            offset=text_hdr_desc.offset,
            length=text_hdr_desc.itemsize,
        )

        text_header = text_hdr_desc._decode(buffer)
        return text_hdr_desc._wrap(text_header)

    def _infer_spec(self) -> SegyDescriptor:
        """Infer the SEG-Y specification for the file."""
        if self.settings.revision is not None:
            return get_spec_from_settings(self.settings)

        binary_header_buffer = read_default_binary_file_header_buffer(self.fs, self.url)
        return infer_spec_from_binary_header(binary_header_buffer)

    @cached_property
    def binary_header(self) -> HeaderArray:
        """Read binary header from store, based on spec."""
        buffer_bytes = self.fs.read_block(
            fn=self.url,
            offset=self.spec.binary_file_header.offset,
            length=self.spec.binary_file_header.itemsize,
        )
        buffer = bytearray(buffer_bytes)

        bin_hdr = np.frombuffer(buffer, dtype=self.spec.binary_file_header.dtype)

        transforms = TransformPipeline()

        if self.spec.endianness == Endianness.BIG:
            little_endian = TransformFactory.create("byte_swap", Endianness.LITTLE)
            transforms.add_transform(little_endian)

        return HeaderArray(transforms.apply(bin_hdr))

    def _parse_binary_header(self) -> None:
        """Parse the binary header and apply some rules."""
        # Calculate sizes for dynamic file metadata
        text_hdr_size = self.spec.text_file_header.itemsize
        bin_hdr_size = self.spec.binary_file_header.itemsize

        # Update trace start offset and sample length
        self.spec.trace.sample_descriptor.samples = self.samples_per_trace
        self.spec.trace.offset = text_hdr_size + bin_hdr_size

        if self.num_ext_text > 0:
            self.spec.extended_text_header = rev1_extended_text_header

            ext_text_size = self.spec.extended_text_header.itemsize * self.num_ext_text
            self.spec.trace.offset = self.spec.trace.offset + ext_text_size

    @property
    def sample(self) -> AbstractIndexer:
        """Way to access the file to fetch trace data only."""
        return SampleIndexer(
            self.fs,
            self.url,
            self.spec.trace,
            self.num_traces,
            kind="sample",
            settings=self.settings,
            transforms=self.accessors.sample_decode_transforms,
        )

    @property
    def header(self) -> HeaderIndexer:
        """Way to access the file to fetch trace headers only."""
        return HeaderIndexer(
            self.fs,
            self.url,
            self.spec.trace,
            self.num_traces,
            kind="header",
            settings=self.settings,
            transforms=self.accessors.header_decode_transforms,
        )

    @property
    def trace(self) -> TraceIndexer:
        """Way to access the file to fetch full traces (headers + data)."""
        return TraceIndexer(
            self.fs,
            self.url,
            self.spec.trace,
            self.num_traces,
            kind="trace",
            settings=self.settings,
            transforms=self.accessors.trace_decode_transforms,
        )
