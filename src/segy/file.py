"""SEG-Y file in-memory constructors."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING
from typing import cast

import numpy as np
from fsspec.core import url_to_fs

from segy.accessors import TraceAccessor
from segy.arrays import HeaderArray
from segy.config import SegySettings
from segy.exceptions import EndiannessInferenceError
from segy.indexing import DataIndexer
from segy.indexing import HeaderIndexer
from segy.indexing import TraceIndexer
from segy.schema import Endianness
from segy.schema import ScalarType
from segy.schema import SegyStandard
from segy.standards import get_segy_standard
from segy.standards.codes import DataSampleFormatCode
from segy.standards.codes import SegyEndianCode
from segy.transforms import TransformFactory
from segy.transforms import TransformPipeline

if TYPE_CHECKING:
    from fsspec import AbstractFileSystem
    from numpy.typing import NDArray

    from segy.indexing import AbstractIndexer
    from segy.schema import SegySpec


@dataclass
class SegyInferResult:
    """A scan result of a SEG-Y file.

    Attributes:
        endianness: Endianness of the file.
        revision: SEG-Y revision as float.
    """

    __slots__ = ("endianness", "revision", "sample_format")

    endianness: Endianness
    revision: float


class EndiannessAction(Enum):
    """Descriptive flag enum for endianness reversal."""

    REVERSE = True
    KEEP = False


def infer_endianness(
    buffer: bytes,
    settings: SegySettings,
) -> EndiannessAction:
    """Infer if we need to reverse the endianness of the seismic data.

    Args:
        buffer: Bytes representing the binary header.
        settings: Settings instance to configure / override.

    Returns:
        A boolean indicating if the endianness need to be reversed.

    Raises:
        EndiannessInferenceError: When inference fails.
    """
    # Use endianness from settings if provided
    if settings.endianness is not None:
        return EndiannessAction(settings.endianness != sys.byteorder)

    # Define offsets and data types
    endian_offset = 96
    format_offset = 24
    endian_dtype = np.dtype("uint32")
    format_dtype = np.dtype("uint16")
    supported_formats = set(DataSampleFormatCode._value2member_map_.keys())

    # Attempt to read explicit endianness code (SEGY Rev2+)
    endian_code = np.frombuffer(buffer, endian_dtype, offset=endian_offset, count=1)[0]

    if endian_code == SegyEndianCode.NATIVE:
        return EndiannessAction.KEEP
    if endian_code == SegyEndianCode.REVERSE:
        return EndiannessAction.REVERSE
    if endian_code == SegyEndianCode.PAIRWISE_SWAP:
        msg = "File bytes are pairwise swapped. Currently not supported."
        raise NotImplementedError(msg)
    if endian_code != 0:
        msg = (
            f"Explicit endianness code has ambiguous value: {endian_code}. "
            "Expected one of {{16909060, 67305985, 33620995}} at byte 97. "
            "Provide endianness using SegyFileSettings or SegySpec."
        )
        raise EndiannessInferenceError(msg)

    # Legacy method for SEGY Rev <2.0
    def check_format_code(dtype: np.dtype) -> bool:
        format_value = np.frombuffer(buffer, dtype, offset=format_offset, count=1)[0]
        return format_value in supported_formats

    # Check with native machine endianness
    if check_format_code(format_dtype):
        return EndiannessAction.KEEP

    # Check with reverse machine endianness
    if check_format_code(format_dtype.newbyteorder()):
        return EndiannessAction.REVERSE

    # Inference failed
    msg = (
        "Cannot automatically infer file endianness using explicit or legacy "
        "methods. Please provide it using SegyFileSettings."
    )
    raise EndiannessInferenceError(msg)


def infer_revision(
    buffer: bytes,
    endianness_action: EndiannessAction,
    settings: SegySettings,
) -> float:
    """Infer the revision number from the binary header of a SEG-Y file.

    Args:
        buffer: The binary header buffer.
        endianness_action: The action to take for endianness.
        settings: The SegySettings, which may override the revision.

    Returns:
        The revision number as a float (e.g., 1.0, 1.2, 2.0).
    """
    # Handle override from settings
    if settings.binary.revision is not None:
        return settings.binary.revision

    # Rev2+ defines major and minor version as 1-byte fields. Read major first.
    revision_dtype = np.dtype("uint8")
    revision_major = np.frombuffer(buffer, revision_dtype, offset=300, count=1)[0]

    # If major is 2, read remaining (minor)
    if revision_major >= SegyStandard.REV2:
        revision_minor = np.frombuffer(buffer, revision_dtype, offset=301, count=1)[0]

    # Legacy (Revision <2.0) fallback
    # Read revision from 2-byte field with correct endian
    else:
        revision_dtype = np.dtype("uint16")
        if endianness_action == EndiannessAction.REVERSE:
            revision_dtype = revision_dtype.newbyteorder()

        revision = np.frombuffer(buffer, revision_dtype, offset=300, count=1)[0]
        revision_major = revision >> 8
        revision_minor = revision & 0xFF

    return revision_major + revision_minor / 10


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
        self.spec = spec
        self.settings = settings if settings is not None else SegySettings()

        self.fs, self.url = url_to_fs(url, **self.settings.storage_options)
        self._info = self.fs.info(self.url)

        if self.spec is None:
            scan_result = self._infer_spec()
            self.spec = get_segy_standard(scan_result.revision)
            self.spec.endianness = scan_result.endianness

        if self.spec.endianness is None:
            scan_result = self._infer_spec()
            self.spec.endianness = scan_result.endianness

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

    def _infer_spec(self) -> SegyInferResult:
        bin_header_buffer = self.fs.read_block(fn=self.url, offset=3200, length=400)
        endianness_action = infer_endianness(bin_header_buffer, self.settings)
        revision = infer_revision(bin_header_buffer, endianness_action, self.settings)

        if endianness_action == EndiannessAction.REVERSE:
            byte_order = "big" if sys.byteorder == "little" else "little"
        else:
            byte_order = sys.byteorder

        return SegyInferResult(
            endianness=Endianness(byte_order),
            revision=revision,
        )

    def _update_spec(self) -> None:
        """Parse the binary header and apply some rules."""
        if self.spec.ext_text_header is not None:
            num_ext_text = self.binary_header["num_extended_text_headers"].item()
            self.spec.ext_text_header.count = num_ext_text

            if self.settings.binary.ext_text_header is not None:
                settings_num_ext_text = self.settings.binary.ext_text_header
                self.spec.ext_text_header.count = settings_num_ext_text

        sample_format_value = self.binary_header["data_sample_format"]
        sample_format_code = DataSampleFormatCode(sample_format_value)
        sample_format = ScalarType[sample_format_code.name]
        self.spec.trace.data.format = sample_format
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
