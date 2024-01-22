"""SEG-Y file in-memory constructors."""
from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
from fsspec.core import url_to_fs
from pandas import DataFrame

from segy.config import SegyFileSettings
from segy.indexing import AbstractIndexer
from segy.indexing import DataIndexer
from segy.indexing import HeaderIndexer
from segy.indexing import TraceIndexer
from segy.schema import Endianness
from segy.schema import SegyDescriptor
from segy.schema import SegyStandard
from segy.standards.registry import get_spec
from segy.standards.rev1 import rev1_binary_file_header

if TYPE_CHECKING:
    from fsspec import AbstractFileSystem
    from numpy.typing import NDArray


class SegyFile:
    """A SEG-Y file opener with fancy features."""

    fs: AbstractFileSystem
    url: str

    def __init__(
        self,
        url: str,
        spec: SegyDescriptor | None = None,
        settings: SegyFileSettings | None = None,
    ):
        """Parse some metadata and get file ready for manipulating."""
        if settings is None:
            self.settings = SegyFileSettings()

        self.fs, self.url = url_to_fs(url)

        self.spec = self._infer_standard() if spec is None else spec

        self._info = self.fs.info(self.url)
        self._parse_binary_header()

    @classmethod
    def from_spec(
        cls: type[SegyFile],
        url: str,
        spec: SegyDescriptor,
        **kwargs: dict[str, Any],
    ) -> SegyFile:
        """Open a SEG-Y file based on custom spec."""
        return cls(url=url, spec=spec, **kwargs)

    @property
    def file_size(self) -> int:
        """Return file size in bytes."""
        return self._info["size"]

    @property
    def num_traces(self) -> int:
        """Return number of traces in file based on size and spec."""
        file_textual_hdr_size = self.spec.text_file_header.itemsize
        file_bin_hdr_size = self.spec.text_file_header.itemsize
        trace_size = self.spec.trace.itemsize

        rev0_file = self.spec.segy_standard == SegyStandard.REV0
        if self.settings.BINARY.EXTENDED_TEXT_HEADER.VALUE is None and not rev0_file:
            header_key = self.settings.BINARY.EXTENDED_TEXT_HEADER.KEY

            num_ext_text = 0
            if header_key in self.binary_header:
                num_ext_text = self.binary_header[header_key]
        else:
            num_ext_text = self.settings.BINARY.EXTENDED_TEXT_HEADER.VALUE

        file_metadata_size = file_textual_hdr_size + file_bin_hdr_size

        if num_ext_text is not None:
            file_metadata_size += num_ext_text * self.spec.extended_text_header.itemsize

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
        if self.settings.REVISION is None:
            buffer = self.fs.read_block(
                fn=self.url,
                offset=rev1_binary_file_header.offset,
                length=rev1_binary_file_header.item_size,
            )
            revision = np.frombuffer(buffer, offset=300, dtype=">i2", count=1).item()

        else:
            revision = self.settings.REVISION

        standard = SegyStandard(revision)
        return get_spec(standard)

    @cached_property
    def binary_header(self) -> NDArray[Any] | DataFrame:
        """Read binary header from store, based on spec."""
        buffer = bytearray(
            self.fs.read_block(
                fn=self.url,
                offset=self.spec.binary_file_header.offset,
                length=self.spec.binary_file_header.itemsize,
            )
        )

        bin_hdr = np.frombuffer(buffer, dtype=self.spec.binary_file_header.dtype)

        if self.settings.ENDIAN == Endianness.BIG:
            bin_hdr = bin_hdr.byteswap(inplace=True).newbyteorder()

        if self.settings.USE_PANDAS:
            return DataFrame(bin_hdr.reshape(1))

        return bin_hdr.squeeze()

    def _parse_binary_header(self) -> None:
        """Parse the binary header and apply some rules."""
        # Extract number of samples and extended text headers.
        if self.settings.BINARY.SAMPLES_PER_TRACE.VALUE is None:
            header_key = self.settings.BINARY.SAMPLES_PER_TRACE.KEY
            samples_per_trace = self.binary_header[header_key]
        else:
            samples_per_trace = self.settings.BINARY.SAMPLES_PER_TRACE.VALUE

        # Calculate sizes for dynamic file metadata
        text_hdr_size = self.spec.text_file_header.itemsize
        bin_hdr_size = self.spec.binary_file_header.itemsize

        # Update trace start offset and sample length
        self.spec.trace.data_descriptor.samples = int(samples_per_trace)
        self.spec.trace.offset = text_hdr_size + bin_hdr_size

        rev0_file = self.spec.segy_standard == SegyStandard.REV0
        if self.settings.BINARY.EXTENDED_TEXT_HEADER.VALUE is None and not rev0_file:
            header_key = self.settings.BINARY.EXTENDED_TEXT_HEADER.KEY

            num_ext_text = 0
            if header_key in self.binary_header:
                num_ext_text = self.binary_header[header_key]
        else:
            num_ext_text = self.settings.BINARY.EXTENDED_TEXT_HEADER.VALUE

        if num_ext_text is not None:
            ext_text_size = self.spec.extended_text_header.itemsize * int(num_ext_text)
            self.spec.trace.offset = self.spec.trace.offset + ext_text_size

    @property
    def data(self) -> AbstractIndexer:
        """Way to access the file to fetch trace data only."""
        return DataIndexer(
            self.fs,
            self.url,
            self.spec.trace,
            self.num_traces,
            kind="data",
        )

    @property
    def header(self) -> AbstractIndexer:
        """Way to access the file to fetch trace headers only."""
        return HeaderIndexer(
            self.fs,
            self.url,
            self.spec.trace,
            self.num_traces,
            kind="header",
            settings=self.settings,
        )

    @property
    def trace(self) -> AbstractIndexer:
        """Way to access the file to fetch full traces (headers + data)."""
        return TraceIndexer(
            self.fs,
            self.url,
            self.spec.trace,
            self.num_traces,
            kind="trace",
            settings=self.settings,
        )
