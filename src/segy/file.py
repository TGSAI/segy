"""SEG-Y file in-memory constructors."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from fsspec.core import url_to_fs

from segy.arrays import HeaderArray
from segy.config import SegyFileSettings
from segy.indexing import HeaderIndexer
from segy.indexing import SampleIndexer
from segy.indexing import TraceIndexer
from segy.schema import Endianness
from segy.schema import ScalarType
from segy.schema import SegyStandard
from segy.standards.registry import get_spec
from segy.standards.rev1 import rev1_binary_file_header
from segy.standards.rev1 import rev1_extended_text_header
from segy.transforms import TransformFactory
from segy.transforms import TransformPipeline

if TYPE_CHECKING:
    from fsspec import AbstractFileSystem
    from numpy.typing import NDArray

    from segy.indexing import AbstractIndexer
    from segy.schema import SegyDescriptor


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

        self.spec = self._infer_standard() if spec is None else spec

        self._info = self.fs.info(self.url)
        self._update_spec_endianness()
        self._parse_binary_header()

    def _update_spec_endianness(self) -> None:
        """If spec has no endianness, get it from settings."""
        if self.spec.endianness is None:
            self.spec.endianness = self.settings.endianness

    @property
    def file_size(self) -> int:
        """Return file size in bytes."""
        size: int = self._info["size"]
        return size

    @property
    def num_ext_text(self) -> int:
        """Return number of extended text headers."""
        if self.spec.segy_standard in {SegyStandard.REV0, SegyStandard.CUSTOM}:
            return 0

        # Overriding from settings
        if self.settings.binary.extended_text_header.value is not None:
            return self.settings.binary.extended_text_header.value

        header_key = self.settings.binary.extended_text_header.key
        return int(self.binary_header[header_key][0])

    @property
    def samples_per_trace(self) -> int:
        """Return samples per trace in file based on spec."""
        # Overriding from settings
        if self.settings.binary.samples_per_trace.value is not None:
            return self.settings.binary.samples_per_trace.value

        header_key = self.settings.binary.samples_per_trace.key
        return int(self.binary_header[header_key].item())

    @property
    def sample_interval(self) -> int:
        """Return samples interval in file based on spec."""
        # Overriding from settings
        if self.settings.binary.sample_interval.value is not None:
            return self.settings.binary.sample_interval.value

        header_key = self.settings.binary.sample_interval.key
        return int(self.binary_header[header_key].item())

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

    def _infer_standard(self) -> SegyDescriptor:
        if self.settings.revision is None:
            buffer = self.fs.read_block(
                fn=self.url,
                offset=rev1_binary_file_header.offset,
                length=rev1_binary_file_header.item_size,
            )
            revision = np.frombuffer(buffer, offset=300, dtype=">i2", count=1).item()

            # Per SEG-Y standard, there is a Q-point between the bytes. Dividing
            # by 2^8 to get the floating-point value of the revision.
            revision = revision / 256.0

        else:
            revision = self.settings.revision

        standard = SegyStandard(revision)
        return get_spec(standard)

    @cached_property
    def binary_header(self) -> HeaderArray:
        """Read binary header from store, based on spec."""
        buffer = self.fs.read_block(
            fn=self.url,
            offset=self.spec.binary_file_header.offset,
            length=self.spec.binary_file_header.itemsize,
        )

        bin_hdr = np.frombuffer(buffer, dtype=self.spec.binary_file_header.dtype)

        transforms = TransformPipeline()
        transforms.add_transform(
            TransformFactory.create("byte_swap", Endianness.LITTLE),
        )

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
        transforms = [
            TransformFactory.create("byte_swap", Endianness.LITTLE),
        ]

        if self.spec.trace.sample_descriptor.format == ScalarType.IBM32:
            transforms.append(TransformFactory.create("ibm_float", "to_ieee"))

        return SampleIndexer(
            self.fs,
            self.url,
            self.spec.trace,
            self.num_traces,
            kind="sample",
            settings=self.settings,
            transforms=transforms,
        )

    @property
    def header(self) -> HeaderIndexer:
        """Way to access the file to fetch trace headers only."""
        transforms = [
            TransformFactory.create("byte_swap", Endianness.LITTLE),
        ]

        return HeaderIndexer(
            self.fs,
            self.url,
            self.spec.trace,
            self.num_traces,
            kind="header",
            settings=self.settings,
            transforms=transforms,
        )

    @property
    def trace(self) -> TraceIndexer:
        """Way to access the file to fetch full traces (headers + data)."""
        transforms = [
            TransformFactory.create("byte_swap", Endianness.LITTLE),
        ]

        if self.spec.trace.sample_descriptor.format == ScalarType.IBM32:
            transforms.append(
                TransformFactory.create("ibm_float", "to_ieee", keys=["sample"])
            )

        return TraceIndexer(
            self.fs,
            self.url,
            self.spec.trace,
            self.num_traces,
            kind="trace",
            settings=self.settings,
            transforms=transforms,
        )
