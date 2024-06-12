"""Indexers for SEG-Y files."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
from fsspec.utils import merge_offset_ranges

from segy.arrays import HeaderArray
from segy.arrays import TraceArray
from segy.config import SegySettings
from segy.transforms import TransformPipeline

if TYPE_CHECKING:
    from pathlib import Path

    from fsspec import AbstractFileSystem
    from numpy.typing import NDArray

    from segy.schema import TraceSpec
    from segy.schema.base import BaseDataType


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
        spec: An instance of BaseDataType.
        max_value: An integer representing the maximum value of the index.
        settings: Optional parsing settings.
        transform_pipeline: The transforms pipeline to apply for decoding.
    """

    kind: str = "Abstract"

    def __init__(  # noqa: PLR0913
        self,
        fs: AbstractFileSystem,
        url: str,
        spec: BaseDataType,
        max_value: int,
        settings: SegySettings | None = None,
        transform_pipeline: TransformPipeline | None = None,
    ):
        self.fs = fs
        self.url = url
        self.spec = spec
        self.max_value = max_value
        self.settings = SegySettings() if settings is None else settings

        self.transform_pipeline = (
            TransformPipeline() if transform_pipeline is None else transform_pipeline
        )

    @abstractmethod
    def indices_to_byte_ranges(self, indices: list[int]) -> tuple[list[int], list[int]]:
        """Logic to calculate start/end bytes."""

    @abstractmethod
    def decode(self, buffer: bytearray) -> NDArray[Any]:
        """How to decode the bytes after reading."""

    def post_process(self, data: NDArray[Any]) -> NDArray[Any]:
        """Apply transforms to the data after decoding."""
        return self.transform_pipeline.apply(data)

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
    spec. It will optionally return the headers as a Pandas DataFrame.
    """

    spec: TraceSpec
    kind: str = "trace"

    def indices_to_byte_ranges(self, indices: list[int]) -> tuple[list[int], list[int]]:
        """Convert trace indices to byte ranges."""
        if self.spec.offset is None:
            msg = "Trace starting offset must be specified."
            raise ValueError(msg)

        start_offset = self.spec.offset
        trace_itemsize = self.spec.dtype.itemsize

        starts = [start_offset + i * trace_itemsize for i in indices]
        ends = [start + trace_itemsize for start in starts]

        return starts, ends

    def decode(self, buffer: bytearray) -> TraceArray:
        """Decode whole traces (header + data)."""
        data = np.frombuffer(buffer, dtype=self.spec.dtype)
        return TraceArray(data)


class HeaderIndexer(AbstractIndexer):
    """Indexer for reading trace headers only.

    Inherits from AbstractIndexer. Implements decoding based on trace
    spec. It will optionally return the headers as a Pandas DataFrame.
    """

    spec: TraceSpec
    kind: str = "header"

    def indices_to_byte_ranges(self, indices: list[int]) -> tuple[list[int], list[int]]:
        """Convert header indices to byte ranges (without trace data)."""
        trace_itemsize = self.spec.dtype.itemsize
        header_itemsize = self.spec.header.itemsize

        if self.spec.offset is None:
            msg = "Trace starting offset must be specified."
            raise ValueError(msg)

        start_offset = self.spec.offset

        starts = [start_offset + i * trace_itemsize for i in indices]
        ends = [start + header_itemsize for start in starts]

        return starts, ends

    def decode(self, buffer: bytearray) -> HeaderArray:
        """Decode headers only."""
        data = np.frombuffer(buffer, dtype=self.spec.dtype["header"])
        return HeaderArray(data)


class DataIndexer(AbstractIndexer):
    """Indexer for reading trace data samples only.

    Inherits from AbstractIndexer. Implements decoding based on trace spec.
    """

    spec: TraceSpec
    kind: str = "data"

    def indices_to_byte_ranges(self, indices: list[int]) -> tuple[list[int], list[int]]:
        """Convert data indices to byte ranges (without trace headers)."""
        trace_itemsize = self.spec.itemsize
        data_itemsize = self.spec.data.itemsize
        header_itemsize = self.spec.header.itemsize

        if self.spec.offset is None:
            msg = "Trace starting offset must be specified."
            raise ValueError(msg)

        start_offset = self.spec.offset + header_itemsize

        starts = [start_offset + i * trace_itemsize for i in indices]
        ends = [start + data_itemsize for start in starts]

        return starts, ends

    def decode(self, buffer: bytearray) -> NDArray[Any]:
        """Decode trace samples only."""
        return np.frombuffer(buffer, dtype=self.spec.dtype["data"])
