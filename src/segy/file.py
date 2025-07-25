"""SEG-Y file in-memory constructors."""

from __future__ import annotations

import logging
import sys
from functools import cached_property
from typing import TYPE_CHECKING
from typing import cast

import numpy as np
from fsspec.core import url_to_fs

from segy.accessors import TraceAccessor
from segy.arrays import HeaderArray
from segy.config import SegySettings
from segy.exceptions import SegyFileSpecMismatchError
from segy.indexing import DataIndexer
from segy.indexing import HeaderIndexer
from segy.indexing import TraceIndexer
from segy.inference import EndiannessAction
from segy.inference import SegyInferResult
from segy.inference import infer_endianness
from segy.inference import interpret_revision
from segy.schema import Endianness
from segy.schema import ScalarType
from segy.standards import get_segy_standard
from segy.standards.codes import DataSampleFormatCode
from segy.transforms import TransformFactory
from segy.transforms import TransformPipeline

if TYPE_CHECKING:
    from fsspec import AbstractFileSystem
    from numpy.typing import NDArray

    from segy.indexing import AbstractIndexer
    from segy.schema import SegySpec

logger = logging.getLogger(__name__)


class SegyFile:
    """A SEG-Y file class that has various accessors.

    Args:
        url: Path to SEG-Y file on disk or remote store.
        spec: The schema / spec describing the SEG-Y file. This
            is optional and by default it will try to infer the
            SEG-Y standard from the binary header.
        settings: A settings instance to configure / override
            the SEG-Y parsing logic. Optional.

    Attributes:
        fs: The filesystem instance used to access the file.
        url: The URL of the SEG-Y file on the filesystem.
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

        self.spec = self._initialize_spec(spec)
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
        return cast("int", self.spec.trace.data.samples)  # we know for sure its int

    @property
    def sample_interval(self) -> int:
        """Return samples interval in file based on spec."""
        return cast("int", self.spec.trace.data.interval)  # we know for sure its int

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
        return cast("int", self.spec.trace.count)  # we know for sure its int

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

    def _initialize_spec(self, spec: SegySpec | None) -> SegySpec:
        """Initialize the spec based on the settings and/or file contents."""
        if spec is None:
            logger.info("No spec provided, inferring standard from binary header.")
            scan_result = self._infer_spec()
            inferred_spec = get_segy_standard(scan_result.revision)
            inferred_spec.endianness = scan_result.endianness
            spec = inferred_spec
        if spec.endianness is None:
            logger.info("No endianness provided, inferring from binary header.")
            spec.endianness = self._infer_spec().endianness
        return spec

    def _infer_spec(self) -> SegyInferResult:
        """Scan the file and infer the endianness and SEG-Y standard."""
        logger.info("Scanning binary header to infer SEG-Y standard and endianness.")
        bin_header_buffer = self.fs.read_block(fn=self.url, offset=3200, length=400)
        endianness_action = infer_endianness(bin_header_buffer, self.settings)
        revision = interpret_revision(
            bin_header_buffer, endianness_action, self.settings
        )

        if endianness_action == EndiannessAction.REVERSE:
            logger.debug("File not machine endianness, will reverse endian.")
            byte_order = "big" if sys.byteorder == "little" else "little"
        else:
            logger.debug("File is same as machine endianness, will read as-is.")
            byte_order = sys.byteorder

        return SegyInferResult(
            endianness=Endianness(byte_order),
            revision=revision,
        )

    def _update_spec(self) -> None:
        """Parse the binary header and apply some rules."""
        ext_text_header_spec = self.spec.ext_text_header

        if ext_text_header_spec is not None:
            num_ext_text = self.binary_header["num_extended_text_headers"].item()
            logger.info("Ext text headers, count from binary header: %s.", num_ext_text)
            ext_text_header_spec.count = num_ext_text

            settings_num_ext_text = self.settings.binary.ext_text_header
            if settings_num_ext_text is not None:
                logger.info(
                    "Using provided ext text header count from settings: %s",
                    settings_num_ext_text,
                )
                ext_text_header_spec.count = settings_num_ext_text

        sample_format_value = self.binary_header["data_sample_format"].item()
        data_sample_format_code = DataSampleFormatCode(sample_format_value)
        sample_format = ScalarType[data_sample_format_code.name]

        trace_data_spec = self.spec.trace.data
        trace_data_spec.format = sample_format
        trace_data_spec.samples = self.binary_header["samples_per_trace"].item()
        trace_data_spec.interval = self.binary_header["sample_interval"].item()

        logger.debug("Parsed sample format: %s", trace_data_spec.format)
        logger.debug("Parsed samples per trace: %s", trace_data_spec.samples)
        logger.debug("Parsed sample interval: %s", trace_data_spec.interval)

        self.spec.update_offsets()

        trace_offset = cast("int", self.spec.trace.offset)  # we know for sure not None
        trace_itemsize = self.spec.trace.itemsize
        trace_count = (self.file_size - trace_offset) // trace_itemsize

        # Ensure trace count and all other offsets match file size
        logger.debug("Calculated trace count: %s", trace_count)
        inferred_size = trace_offset + trace_count * trace_itemsize
        actual_size = self.file_size
        if inferred_size != actual_size:
            msg = f"{actual_size=} doesn't match parsed spec: {inferred_size} ."
            logger.error(msg)
            raise SegyFileSpecMismatchError(msg)

        self.spec.trace.count = trace_count

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
