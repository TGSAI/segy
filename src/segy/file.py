"""SEG-Y file in-memory constructors."""


from typing import Any

import numpy as np
from fsspec import AbstractFileSystem
from fsspec.core import url_to_fs
from numpy.typing import NDArray
from pandas import DataFrame

from segy.indexing import AbstractIndexer
from segy.indexing import DataIndexer
from segy.indexing import HeaderIndexer
from segy.indexing import TraceIndexer
from segy.schema.segy import SegyDescriptor
from segy.schema.segy import SegyStandard
from segy.standards.rev1 import REV1_SEGY

STANDARD_MAPPING = {
    SegyStandard.REV1: REV1_SEGY,
}


class SegyFile:
    """A SEG-Y file opener with fancy features."""

    fs: AbstractFileSystem
    url: str

    def __init__(
        self,
        url: str,
        revision: SegyStandard = SegyStandard.REV1,
        pandas_headers: bool = True,
    ):
        """Parse some metadata and get file ready for manipulating."""
        self.fs, self.url = url_to_fs(url)

        self.spec = STANDARD_MAPPING[revision]
        self._postprocess_kwargs = {"pandas_headers": pandas_headers}

        self._info = self.fs.info(self.url)
        self._binary_header = None
        self._read_binary_header()
        self._parse_binary_header()

    @classmethod
    def from_spec(cls, url: str, spec: SegyDescriptor, **kwargs: dict[str, Any]):
        STANDARD_MAPPING[SegyStandard.CUSTOM] = spec
        return cls(url=url, revision=SegyStandard.CUSTOM, **kwargs)

    @property
    def file_size(self) -> int:
        """Return file size in bytes."""
        return self._info["size"]

    @property
    def num_traces(self) -> int:
        """Return number of traces in file based on size and spec."""
        # TODO(Altay): Not counting extended headers for now
        trace_size = self.spec.traces.itemsize
        file_metadata_size = (
            self.spec.text_file_header.itemsize + self.spec.binary_file_header.item_size
        )

        return (self.file_size - file_metadata_size) // trace_size

    @property
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

    @property
    def binary_header(self) -> NDArray[Any] | DataFrame:
        """Return binary header as struct or pandas DataFrame."""
        using_pandas = self._postprocess_kwargs.get("pandas_headers", False)
        if using_pandas:
            return DataFrame(self._binary_header.reshape(1))

        return self._binary_header

    def _read_binary_header(self) -> None:
        """Read binary header from store, based on spec."""
        buffer = bytearray(
            self.fs.read_block(
                fn=self.url,
                offset=self.spec.binary_file_header.offset,
                length=self.spec.binary_file_header.itemsize,
            )
        )

        self._binary_header = (
            np.frombuffer(buffer, dtype=self.spec.binary_file_header.dtype)
            .squeeze()
            .byteswap(inplace=True)
            .newbyteorder()
        )

    def _parse_binary_header(
        self,
        samples_per_trace_key: str = "samples_per_trace",
        extended_textual_headers_key: str = "extended_textual_headers",
    ) -> None:
        """Parse the binary header and apply some rules.

        It currently supports:
        - Get number of samples and update spec.

        Args:
            samples_per_trace_key: The key to access the number of samples per trace
                in the binary header. Default is "samples_per_trace".
            extended_textual_headers_key: The key to get the number of extended
                textual headers in the binary header. Default is
                "extended_textual_headers".
        """
        # Handle trace sample size
        samples_per_trace = self.binary_header[samples_per_trace_key]
        self.spec.traces.data_descriptor.samples = int(samples_per_trace)

        # TODO(Altay): Handle extended text headers (i.e. traces start offset)
        if extended_textual_headers_key:
            pass

        text_hdr_size = self.spec.text_file_header.itemsize
        bin_hdr_size = self.spec.binary_file_header.itemsize
        traces_offset = text_hdr_size + bin_hdr_size
        self.spec.traces.offset = int(traces_offset)

    @property
    def data(self) -> AbstractIndexer:
        """Way to access the file to fetch trace data only."""
        return DataIndexer(
            self.fs,
            self.url,
            self.spec.traces,
            self.num_traces,
            kind="data",
        )

    @property
    def header(self) -> AbstractIndexer:
        """Way to access the file to fetch trace headers only."""
        return HeaderIndexer(
            self.fs,
            self.url,
            self.spec.traces,
            self.num_traces,
            kind="header",
            postprocess_kwargs=self._postprocess_kwargs,
        )

    @property
    def trace(self) -> AbstractIndexer:
        """Way to access the file to fetch full traces (headers + data)."""
        return TraceIndexer(
            self.fs,
            self.url,
            self.spec.traces,
            self.num_traces,
            kind="trace",
            postprocess_kwargs=self._postprocess_kwargs,
        )
