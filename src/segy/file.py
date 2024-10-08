"""SEG-Y file in-memory constructors."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING
from typing import cast

import numpy as np
from fsspec.core import url_to_fs

from segy.accessors import TraceAccessor
from segy.arrays import HeaderArray
from segy.config import SegySettings
from segy.constants import REV1_BASE16
from segy.exceptions import EndiannessInferenceError
from segy.indexing import DataIndexer
from segy.indexing import HeaderIndexer
from segy.indexing import TraceIndexer
from segy.schema import Endianness
from segy.schema import ScalarType
from segy.standards import get_segy_standard
from segy.standards.mapping import SEGY_FORMAT_MAP
from segy.transforms import TransformFactory
from segy.transforms import TransformPipeline

if TYPE_CHECKING:
    from fsspec import AbstractFileSystem
    from numpy.typing import NDArray

    from segy.indexing import AbstractIndexer
    from segy.schema import SegySpec


@dataclass
class SegyScanResult:
    """A scan result of a SEG-Y file.

    Attributes:
        endianness: Endianness of the file.
        revision: SEG-Y revision as float.
        sample_format: SEG-Y sample format.
    """

    __slots__ = ("endianness", "revision", "sample_format")

    endianness: Endianness
    revision: float
    sample_format: ScalarType


def infer_endianness(
    fs: AbstractFileSystem,
    url: str,
    spec: SegySpec,
) -> SegyScanResult:
    """Infer endianness of a binary header buffer given header spec.

    The buffer length and binary header spec itemsize must be the same.

    Args:
         fs: FSSpec filesystem instance.
         url: Path to the SEG-Y file.
         spec: SEG-Y spec containing how to parse binary header.

    Returns:
        A SegyScanResult instance filled with inferred endianness, revision, and format.

    Raises:
        EndiannessInferenceError: When inference fails.
    """
    bin_spec = spec.binary_header.model_copy(deep=True)  # we will mutate, so copy
    buffer = fs.read_block(url, offset=bin_spec.offset, length=bin_spec.itemsize)

    for endianness in [Endianness.BIG, Endianness.LITTLE]:
        bin_spec.endianness = endianness
        bin_hdr = np.frombuffer(buffer, dtype=bin_spec.dtype)

        revision = bin_hdr["segy_revision"].item() / REV1_BASE16
        sample_increment = bin_hdr["sample_interval"].item()
        sample_format_int = bin_hdr["data_sample_format"].item()

        # Validate the inferred values.
        in_spec = revision in {0.0, 1.0, 2.0}
        increment_is_positive = sample_increment > 0
        format_is_valid = sample_format_int in SEGY_FORMAT_MAP.values()

        if in_spec and increment_is_positive and format_is_valid:
            sample_format = SEGY_FORMAT_MAP.inverse[sample_format_int]
            return SegyScanResult(endianness, revision, sample_format)

    msg = (
        f"Can't infer file endianness, please specify manually. "
        f"Detected {revision=}, {sample_increment=}, and {sample_format_int=}."
    )
    raise EndiannessInferenceError(msg)


def infer_spec(fs: AbstractFileSystem, url: str) -> SegySpec:
    """Try to infer SEG-Y file revision and endianness to build a SegySpec."""
    spec = get_segy_standard(1.0)
    scan_result = infer_endianness(fs, url, spec)
    new_spec = get_segy_standard(scan_result.revision)
    new_spec.trace.data.format = scan_result.sample_format
    new_spec.endianness = scan_result.endianness
    return new_spec


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
        spec: SegySpec | None = None,
        settings: SegySettings | None = None,
    ):
        self.settings = settings if settings is not None else SegySettings()

        self.fs, self.url = url_to_fs(url, **self.settings.storage_options)
        self._info = self.fs.info(self.url)

        # Spec setting overrides.
        # The control flow here is terrible; needs a refactor.
        if self.settings.binary.revision is not None:
            self.spec = get_segy_standard(self.settings.binary.revision)

            # Override/Infer endianness
            if self.settings.endianness is None:
                scan_result = infer_endianness(self.fs, self.url, self.spec)
                self.spec.endianness = scan_result.endianness
            else:
                self.spec.endianness = self.settings.endianness

        # Default, infer if no spec provided.
        elif spec is None:
            self.spec = infer_spec(self.fs, self.url)

        # If spec is provided set to it and update endianness if its None.
        else:
            self.spec = spec

            # Override/Infer endianness
            if self.spec.endianness is None:
                if self.settings.endianness is None:
                    scan_result = infer_endianness(self.fs, self.url, self.spec)
                    self.spec.endianness = scan_result.endianness
                else:
                    self.spec.endianness = self.settings.endianness

        self._update_spec()
        self.accessors = TraceAccessor(self.spec.trace)

    @property
    def file_size(self) -> int:
        """Return file size in bytes."""
        size: int = self._info["size"]
        return size

    @property
    def samples_per_trace(self) -> int:
        """Return samples per trace in file based on spec."""
        return cast(int, self.spec.trace.data.samples)  # we know for sure its int

    @property
    def sample_interval(self) -> int:
        """Return samples interval in file based on spec."""
        return cast(int, self.spec.trace.data.interval)  # we know for sure its int

    @property
    def sample_labels(self) -> NDArray[np.int32]:
        """Return sample axis labels."""
        max_samp = self.sample_interval * self.samples_per_trace
        return np.arange(0, max_samp, self.sample_interval, dtype="int32")

    @property
    def num_ext_text(self) -> int:
        """Return number of extended text headers."""
        if self.spec.ext_text_header is None:
            return 0

        return self.spec.ext_text_header.count

    @property
    def num_traces(self) -> int:
        """Return number of traces in file based on size and spec."""
        return cast(int, self.spec.trace.count)  # we know for sure its int

    @cached_property
    def text_header(self) -> str:
        """Return textual file header."""
        text_hdr_spec = self.spec.text_header

        buffer = self.fs.read_block(
            fn=self.url,
            offset=text_hdr_spec.offset,
            length=text_hdr_spec.itemsize,
        )

        return text_hdr_spec.decode(buffer)

    @cached_property
    def ext_text_header(self) -> list[str]:
        """Return textual file header."""
        ext_text_hdr_spec = self.spec.ext_text_header

        if ext_text_hdr_spec is None:
            return []

        buffer = self.fs.read_block(
            fn=self.url,
            offset=ext_text_hdr_spec.offset,
            length=ext_text_hdr_spec.itemsize,
        )

        return ext_text_hdr_spec.decode(buffer)

    @cached_property
    def binary_header(self) -> HeaderArray:
        """Read binary header from store, based on spec."""
        buffer_bytes = self.fs.read_block(
            fn=self.url,
            offset=self.spec.binary_header.offset,
            length=self.spec.binary_header.itemsize,
        )
        buffer = bytearray(buffer_bytes)

        bin_hdr = np.frombuffer(buffer, dtype=self.spec.binary_header.dtype)

        transforms = TransformPipeline()

        if self.spec.endianness == Endianness.BIG:
            little_endian = TransformFactory.create("byte_swap", Endianness.LITTLE)
            transforms.add_transform(little_endian)

        interpret_revision = TransformFactory.create("segy_revision")
        transforms.add_transform(interpret_revision)

        return HeaderArray(transforms.apply(bin_hdr))

    def _update_spec(self) -> None:
        """Parse the binary header and apply some rules."""
        if self.spec.ext_text_header is not None:
            num_ext_text = self.binary_header["num_extended_text_headers"].item()
            self.spec.ext_text_header.count = num_ext_text

            if self.settings.binary.ext_text_header is not None:
                settings_num_ext_text = self.settings.binary.ext_text_header
                self.spec.ext_text_header.count = settings_num_ext_text

        self.spec.trace.data.samples = self.binary_header["samples_per_trace"].item()
        self.spec.trace.data.interval = self.binary_header["sample_interval"].item()

        self.spec.update_offsets()

        trace_offset = cast(int, self.spec.trace.offset)  # we know for sure not None
        trace_itemsize = self.spec.trace.itemsize
        self.spec.trace.count = (self.file_size - trace_offset) // trace_itemsize

    @property
    def sample(self) -> AbstractIndexer:
        """Way to access the file to fetch trace data only."""
        return DataIndexer(
            self.fs,
            self.url,
            self.spec.trace,
            self.num_traces,
            transform_pipeline=self.accessors.sample_decode_pipeline,
        )

    @property
    def header(self) -> HeaderIndexer:
        """Way to access the file to fetch trace headers only."""
        return HeaderIndexer(
            self.fs,
            self.url,
            self.spec.trace,
            self.num_traces,
            transform_pipeline=self.accessors.header_decode_pipeline,
        )

    @property
    def trace(self) -> TraceIndexer:
        """Way to access the file to fetch full traces (headers + data)."""
        return TraceIndexer(
            self.fs,
            self.url,
            self.spec.trace,
            self.num_traces,
            transform_pipeline=self.accessors.trace_decode_pipeline,
        )
