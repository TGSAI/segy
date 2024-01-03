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
from segy.schema import SegyDescriptor
from segy.schema import SegyStandard
from segy.standards.rev1 import SegyDescriptorRev1


class SegySpecFactory:
    """A SEG-Y spec factory with spec registry."""

    _specs = {}

    @classmethod
    def register_spec(
        cls: type["SegySpecFactory"],
        spec_type: SegyStandard,
        spec_cls: type[SegyDescriptor],
    ) -> None:
        """Register a new SEG-Y spec."""
        if not issubclass(spec_cls, SegyDescriptor):
            msg = "spec_cls must be a subclass of SegyDescriptor."
            raise ValueError(msg)
        cls._specs[spec_type] = spec_cls

    @classmethod
    def create_spec(
        cls: type["SegySpecFactory"], spec_type: SegyStandard
    ) -> SegyDescriptor:
        """Create an instance of spec from known registry."""
        spec_cls = cls._specs.get(spec_type)

        if not spec_cls:
            msg = (
                f"Unknown or unsupported SEG-Y spec: {spec_type}. If you "
                f"would like to use {spec_type}, please register it with "
                f"the `SegySpecFactory` using its `register_spec` method."
            )
            raise NotImplementedError(msg)

        return spec_cls()


SegySpecFactory.register_spec(SegyStandard.REV1, SegyDescriptorRev1)


class SegyFile:
    """A SEG-Y file opener with fancy features."""

    fs: AbstractFileSystem
    url: str

    def __init__(
        self,
        url: str,
        segy_standard: SegyStandard = SegyStandard.REV1,
        pandas_headers: bool = True,
    ):
        """Parse some metadata and get file ready for manipulating."""
        self.fs, self.url = url_to_fs(url)

        # Validate standard
        standard = SegyStandard(segy_standard)

        self.spec = SegySpecFactory.create_spec(standard)
        self._postprocess_kwargs = {"pandas_headers": pandas_headers}

        self._info = self.fs.info(self.url)
        self._binary_header = None
        self._read_binary_header()
        self._parse_binary_header()

    @classmethod
    def from_spec(
        cls: type["SegyFile"],
        url: str,
        spec: type[SegyDescriptor],
        **kwargs: dict[str, Any],
    ) -> "SegyFile":
        """Open a SEG-Y file based on custom spec."""
        SegySpecFactory.register_spec(SegyStandard.CUSTOM, spec)
        return cls(url=url, segy_standard=SegyStandard.CUSTOM, **kwargs)

    @property
    def file_size(self) -> int:
        """Return file size in bytes."""
        return self._info["size"]

    @property
    def num_traces(self) -> int:
        """Return number of traces in file based on size and spec."""
        # TODO(Altay): Not counting extended headers for now
        trace_size = self.spec.trace.itemsize
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
        self.spec.trace.data_descriptor.samples = int(samples_per_trace)

        # TODO(Altay): Handle extended text headers (i.e. traces start offset)
        if extended_textual_headers_key:
            pass

        text_hdr_size = self.spec.text_file_header.itemsize
        bin_hdr_size = self.spec.binary_file_header.itemsize
        traces_offset = text_hdr_size + bin_hdr_size
        self.spec.trace.offset = int(traces_offset)

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
            postprocess_kwargs=self._postprocess_kwargs,
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
            postprocess_kwargs=self._postprocess_kwargs,
        )
