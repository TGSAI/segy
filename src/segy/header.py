"""Header interface.

We use facade structural design pattern for HeaderAccessor and strategy
patterns to make header interactions more user-friendly and configurable.
"""

from __future__ import annotations

from json import dumps as json_dumps
from typing import TYPE_CHECKING

from pandas import DataFrame

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import DTypeLike
    from numpy.typing import NDArray

    from segy.schema import Endianness


class HeaderTransformationStrategy:
    """Base class for header transformation strategies."""

    def transform(self, data: NDArray[Any]) -> NDArray[Any]:
        """Forward transformation implementation."""
        raise NotImplementedError

    def inverse_transform(self, data: NDArray[Any]) -> NDArray[Any]:
        """Inverse transformation implementation."""
        raise NotImplementedError


class ScalingStrategy(HeaderTransformationStrategy):
    """Transform to scale user defined keys in a structured dtype.

    Args:
        scale_factor: Scalar to use in transformations.
        keys: List of keys to apply the transformation on.
    """

    def __init__(self, scale_factor: int | float, keys: list[str]):
        self.scale_factor = scale_factor
        self.keys = keys

    def transform(self, data: NDArray[Any]) -> NDArray[Any]:
        """Forward transformation implementation."""
        if data.dtype.names is None:
            msg = f"{self.__class__.__name__} can only work on structured arrays."
            raise ValueError(msg)

        for key in self.keys:
            if key in data.dtype.names:
                data[key] = [vals * self.scale_factor for vals in data[key]]
        return data

    def inverse_transform(self, data: NDArray[Any]) -> NDArray[Any]:
        """Inverse transformation implementation."""
        if data.dtype.names is None:
            msg = f"{self.__class__.__name__} can only work on structured arrays."
            raise ValueError(msg)

        for key in self.keys:
            if key in data.dtype.names:
                data[key] = [vals / self.scale_factor for vals in data[key]]
        return data


class ByteSwapStrategy(HeaderTransformationStrategy):
    """Transform to swap bytes in a structured dtype.

    We need to store both source and target, so its reversible.

    Args:
        source_byteorder: Source byte order.
        target_byteorder: Desired byte order.
    """

    def __init__(self, source_byteorder: Endianness, target_byteorder: Endianness):
        self.source_byteorder = source_byteorder
        self.target_byteorder = target_byteorder

    def transform(self, data: NDArray[Any]) -> NDArray[Any]:
        """Swap bytes if target != source. Else it is a no-op."""
        if self.source_byteorder is not self.target_byteorder:
            data = data.newbyteorder(self.target_byteorder.symbol).byteswap()
        return data

    def inverse_transform(self, data: NDArray[Any]) -> NDArray[Any]:
        """Swap bytes if source != target. Else it is a no-op."""
        if self.target_byteorder is not self.source_byteorder:
            data = data.newbyteorder(self.source_byteorder.symbol).byteswap()
        return data


class TransformationPipeline:
    """Pipeline to chain transforms forward and backward."""

    def __init__(self) -> None:
        self.transformations: list[HeaderTransformationStrategy] = []

    def add_transformation(self, transformation: HeaderTransformationStrategy) -> None:
        """Add transformation to pipeline."""
        self.transformations.append(transformation)

    def transform(self, data: NDArray[Any]) -> NDArray[Any]:
        """Apply all transformations in sequence."""
        for transformation in self.transformations:
            data = transformation.transform(data)
        return data

    def inverse_transform(self, data: NDArray[Any]) -> NDArray[Any]:
        """Apply all inverse transformations in reverse sequence."""
        for transformation in reversed(self.transformations):
            data = transformation.inverse_transform(data)
        return data


class TransformStrategyFactory:
    """Factory class to generate transformation strategies."""

    @staticmethod
    def create_strategy(
        transform_type: str, parameters: dict[str, Any]
    ) -> HeaderTransformationStrategy:
        """Create an instance of transformation and return it."""
        if transform_type == "byte_swap":
            return ByteSwapStrategy(**parameters)

        if transform_type == "scale":
            return ScalingStrategy(**parameters)

        msg = f"Unsupported transformation type: {transform_type}"
        raise ValueError(msg)


class HeaderAccessor:
    """Acts as a primary interface for users to interact with SEG-Y headers."""

    def __init__(self, data: NDArray[Any]) -> None:
        self._data = data
        self._transform_pipeline = TransformationPipeline()

    def queue_transform(self, transform_type: str, parameters: dict[str, Any]) -> None:
        """Add a transform to the accessor, by name and parameter dict."""
        strategy = TransformStrategyFactory.create_strategy(transform_type, parameters)
        self._transform_pipeline.add_transformation(strategy)

    def _apply_transform(self) -> NDArray[Any]:
        """Apply transforms to raw data."""
        return self._transform_pipeline.transform(self._data.copy())

    def _apply_inverse_transform(self, new_data: NDArray[Any]) -> NDArray[Any]:
        """Apply inverse transform to new data."""
        return self._transform_pipeline.inverse_transform(new_data)

    @property
    def dtype(self) -> dtype[Any]:
        """Data type of the transformed array."""
        return self._apply_transform().dtype

    def view(self, dtype: DTypeLike) -> NDArray[Any]:
        """Data type of the transformed array."""
        return self._apply_transform().view(dtype)

    def __getitem__(self, key: str) -> NDArray[Any]:
        """Return a copy of the transformed data."""
        return self._apply_transform()[key]

    def __setitem__(self, key: str, value: NDArray[Any] | int | float) -> None:
        """Replace raw data with inverse transformed new data."""
        new_data = self._data.copy()
        new_data[key] = value
        self._data = self._apply_inverse_transform(new_data)

    def to_dict(self) -> dict[str, Any]:
        """Convert header to dict."""
        result_dict = {}
        data = self._apply_transform()

        if self.dtype.names is None:
            msg = f"{self.__class__.__name__} can only work on structured arrays."
            raise ValueError(msg)

        for field in self.dtype.names:
            field_values = data[field]
            result_dict[field] = field_values.squeeze().tolist()

        return result_dict

    def to_json(self, indent: int = 2) -> str:
        """Convert header to JSON."""
        return json_dumps(self.to_dict(), indent=indent)

    def to_dataframe(self) -> DataFrame:
        """Convert structured data to pandas DataFrame."""
        return DataFrame.from_records(self._apply_transform())
