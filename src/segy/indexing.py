"""Indexers for SEG-Y files."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from fsspec.utils import merge_offset_ranges
from pandas import DataFrame

from segy.config import SegyFileSettings
from segy.ibm import ibm2ieee
from segy.schema import Endianness
from segy.schema import ScalarType

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from fsspec import AbstractFileSystem
    from numpy.typing import NDArray

    from segy.schema import TraceDescriptor
    from segy.schema.base import BaseTypeDescriptor


def trace_ibm2ieee_inplace(trace: NDArray[Any]) -> NDArray[Any]:
    """Convert data of a trace (headers + data) from IBM32 to float32 in place.

    Args:
        trace: A numpy array of type <trace-dtype> containing the trace data.

    Returns:
        A numpy array of type <new-trace-dtype> converted from the input trace data,
        preserving the original header information.

    Note:
        This method converts IBM format trace data to IEEE format inplace, without
        creating a copy of the trace array.
    """
    header_dtype = trace.dtype["header"]
    data_dtype = trace.dtype["data"]

    num_samp = data_dtype.shape
    data_dtype_f32 = np.dtype(("float32", num_samp))

    trace_dtype_fp32 = np.dtype([("header", header_dtype), ("data", data_dtype_f32)])
    trace["data"] = ibm2ieee(trace["data"]).view("uint32")

    return trace.view(trace_dtype_fp32)


def merge_cat_file(
    fs: AbstractFileSystem,
    url: str | Path,
    starts: list[int],
    ends: list[int],
    block_size: int = 8_388_608,
) -> bytearray:
    """Merge sequential byte start/ends and fetch from store.

    Args:
        fs: fsspec FileSystem instance.
        url: Path/URL to file.
        starts: List of start byte locations.
        ends: List of end byte locations.
        block_size: Optional, block size for concurrent downloads.
            Default is 8MiB.

    Returns:
        Bytearray containing all the requested data.
    """
    paths = [url] * len(starts)

    paths, starts, ends = merge_offset_ranges(
        paths,
        starts,
        ends,
        max_block=block_size,
        sort=True,
    )

    buffer_bytes = fs.cat_ranges(
        paths=paths,
        starts=starts,
        ends=ends,
    )

    return bytearray(b"".join(buffer_bytes))


def bounds_check(indices: list[int], max_: int, type_: str) -> None:
    """Check if indices are out of bounds (negative, or more than max).

    Wrapping negative indices is not supported yet. The `type_` argument
    will be used in exceptions to be descriptive.

    Args:
        indices: A list of integer indices.
        max_: The maximum value of the index range.
        type_: The type of indices being checked.

    Raises:
        IndexError: If any of the indices are negative or exceed the maximum value.
    """
    negative_indices = [index for index in indices if index < 0]
    out_of_range_indices = [index for index in indices if index >= max_]

    outliers = negative_indices + out_of_range_indices

    if outliers:
        msg = (
            f"Requested {type_} indices {outliers} are out of bounds. SEG-Y "
            f"file has {max_} traces. Valid indices are "
            f"[0, {max_ - 1})."
        )
        raise IndexError(msg)


class AbstractIndexer(ABC):
    """Abstract class for indexing and fetching structured data from a remote file.

    We calculate byte ranges from indexing of SEG-Y components and use them
    to fetch the data and decode it.

    Args:
        fs: An instance of `fsspec` file-system.
        url: A string representing the URL of the file.
        spec: An instance of BaseTypeDescriptor.
        max_value: An integer representing the maximum value of the index.
        kind: A string representing the kind of index.
        settings: Optional parsing settings.
    """

    def __init__(  # noqa: PLR0913
        self,
        fs: AbstractFileSystem,
        url: str,
        spec: BaseTypeDescriptor,
        max_value: int,
        kind: str,
        settings: SegyFileSettings | None = None,
    ):
        self.fs = fs
        self.url = url
        self.spec = spec
        self.max_value = max_value
        self.kind = kind
        self.settings = SegyFileSettings() if settings is None else settings

    @abstractmethod
    def indices_to_byte_ranges(self, indices: list[int]) -> tuple[list[int], list[int]]:
        """Logic to calculate start/end bytes."""

    @abstractmethod
    def decode(self, buffer: bytearray) -> NDArray[Any]:
        """How to decode the bytes after reading."""

    def __getitem__(self, item: int | list[int] | slice) -> Any:  # noqa: ANN401
        """Operator for integers, lists, and slices with bounds checking."""
        indices = None

        if isinstance(item, int):
            indices = [item]
            bounds_check(indices, self.max_value, self.kind)

        elif isinstance(item, list):
            indices = item
            bounds_check(indices, self.max_value, self.kind)

        elif isinstance(item, slice):
            if item.step == 0:
                msg = "Step of 0 is invalid for slicing."
                raise ValueError(msg)

            start = item.start or 0
            stop = item.stop or self.max_value

            bounds_check([start, stop - 1], self.max_value, self.kind)
            indices = list(range(*item.indices(self.max_value)))

        if len(indices) == 0:
            msg = "Couldn't parse request. Please ensure it is a valid index."
            raise IndexError(msg)

        data = self.fetch(indices)
        return self.post_process(data)

    def post_process(self, data: NDArray[Any]) -> Any:  # noqa: ANN401
        """Optional post-processing. Override in subclass if needed."""
        return data

    def fetch(self, indices: list[int]) -> NDArray[Any]:
        """Fetches and decodes binary data from the given indices.

        Args:
            indices: A list of integers representing the indices.

        Returns:
            An NDArray of any type representing the fetched data.

        Note:
            - This method internally converts the indices to byte ranges using
                the 'indices_to_byte_ranges' method.
            - The byte ranges are used to fetch the corresponding data from the
                file specified by the 'url' parameter.
            - The fetched data is then decoded and squeezed before being returned.
        """
        starts, ends = self.indices_to_byte_ranges(indices)
        buffer = merge_cat_file(self.fs, self.url, starts, ends)
        return self.decode(buffer).squeeze()


class TraceIndexer(AbstractIndexer):
    """Indexer for reading traces (headers + data).

    Inherits from AbstractIndexer. Implements decoding based on trace
    descriptor. It will optionally return the headers as a Pandas
    DataFrame.
    """

    spec: TraceDescriptor

    def indices_to_byte_ranges(self, indices: list[int]) -> tuple[list[int], list[int]]:
        """Convert trace indices to byte ranges."""
        if self.spec.offset is None:
            msg = "Descriptor offset must be specified."
            raise ValueError(msg)

        start_offset = self.spec.offset
        trace_itemsize = self.spec.dtype.itemsize

        starts = [start_offset + i * trace_itemsize for i in indices]
        ends = [start + trace_itemsize for start in starts]

        return starts, ends

    def decode(self, buffer: bytearray) -> NDArray[Any]:
        """Decode whole traces (header + data)."""
        data = np.frombuffer(buffer, dtype=self.spec.dtype)

        if self.settings.ENDIAN == Endianness.BIG:
            data = data.byteswap(inplace=True).newbyteorder()

        if self.spec.data_descriptor.format == ScalarType.IBM32:
            data = trace_ibm2ieee_inplace(data)

        return data

    def post_process(
        self, data: NDArray[Any]
    ) -> NDArray[Any] | dict[str, NDArray[Any] | DataFrame]:
        """Either return struct array or (Header) DataFrame + (Data) Array."""
        if self.settings.USE_PANDAS:
            return {"header": DataFrame(data["header"]), "data": data["data"]}

        return data


class HeaderIndexer(AbstractIndexer):
    """Indexer for reading trace headers only.

    Inherits from AbstractIndexer. Implements decoding based on trace
    descriptor. It will optionally return the headers as a Pandas
    DataFrame.
    """

    spec: TraceDescriptor

    def indices_to_byte_ranges(self, indices: list[int]) -> tuple[list[int], list[int]]:
        """Convert header indices to byte ranges (without trace data)."""
        trace_itemsize = self.spec.dtype.itemsize
        header_itemsize = self.spec.header_descriptor.itemsize

        if self.spec.offset is None:
            msg = "Descriptor offset must be specified."
            raise ValueError(msg)

        start_offset = self.spec.offset

        starts = [start_offset + i * trace_itemsize for i in indices]
        ends = [start + header_itemsize for start in starts]

        return starts, ends

    def decode(self, buffer: bytearray) -> NDArray[Any]:
        """Decode headers only."""
        data = np.frombuffer(buffer, dtype=self.spec.header_descriptor.dtype)

        # TODO(Altay): Handle float/ibm32 etc headers.
        # https://github.com/TGSAI/segy/issues/5
        if self.settings.ENDIAN == Endianness.BIG:
            data = data.byteswap(inplace=True).newbyteorder()

        return data  # noqa: RET504

    def post_process(self, data: NDArray[Any]) -> NDArray[Any] | DataFrame:
        """Either return header as struct array or DataFrame."""
        if self.settings.USE_PANDAS:
            return DataFrame(data)

        # The numpy array breaks downstream logic so for now
        # turning it off and raising a not implemented error.
        msg = "Not using pandas for headers not implemented yet."
        raise NotImplementedError(msg)
        # return bin_hdr.squeeze()


class DataIndexer(AbstractIndexer):
    """Indexer for reading trace data only.

    Inherits from AbstractIndexer. Implements decoding based on trace
    descriptor.
    """

    spec: TraceDescriptor

    def indices_to_byte_ranges(self, indices: list[int]) -> tuple[list[int], list[int]]:
        """Convert data indices to byte ranges (without trace headers)."""
        trace_itemsize = self.spec.dtype.itemsize
        data_itemsize = self.spec.data_descriptor.dtype.itemsize
        header_itemsize = self.spec.header_descriptor.dtype.itemsize

        if self.spec.offset is None:
            msg = "Descriptor offset must be specified."
            raise ValueError(msg)

        start_offset = self.spec.offset + header_itemsize

        starts = [start_offset + i * trace_itemsize for i in indices]
        ends = [start + data_itemsize for start in starts]

        return starts, ends

    def decode(self, buffer: bytearray) -> NDArray[Any]:
        """Decode trace data only."""
        data = np.frombuffer(buffer, dtype=self.spec.data_descriptor.dtype)

        if self.settings.ENDIAN == Endianness.BIG:
            data = data.byteswap(inplace=True).newbyteorder()

        if self.spec.data_descriptor.format == ScalarType.IBM32:
            data = ibm2ieee(data).view("float32")

        return data
