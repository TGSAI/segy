"""Transforms to apply to arrays and structured arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray

    from segy.schema import Endianness


class TransformStrategy:
    """Base class for header transformation strategies."""

    def transform(self, data: NDArray[Any]) -> NDArray[Any]:
        """Forward transformation implementation."""
        raise NotImplementedError

    def inverse_transform(self, data: NDArray[Any]) -> NDArray[Any]:
        """Inverse transformation implementation."""
        raise NotImplementedError


class ScaleStrategy(TransformStrategy):
    """Transform to scale a numpy array.

    Args:
        scale_factor: Scalar to use in transformations.
    """

    def __init__(self, scale_factor: int | float):
        self.scale_factor = scale_factor

    def transform(self, data: NDArray[Any]) -> NDArray[Any]:
        """Forward transformation implementation."""
        return data * self.scale_factor

    def inverse_transform(self, data: NDArray[Any]) -> NDArray[Any]:
        """Inverse transformation implementation."""
        return data / self.scale_factor


class ScaleFieldStrategy(TransformStrategy):
    """Transform to scale user defined keys in a structured dtype.

    Args:
        scale_factor: Scalar to use in transformations.
        keys: List of fields to apply the transformation on.
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


class ByteSwapStrategy(TransformStrategy):
    """Transform to swap bytes in a structured dtype.

    We need to store both source and target, so its reversible.

    Args:
        source_order: Source byte order.
        target_order: Desired byte order.
    """

    def __init__(self, source_order: Endianness, target_order: Endianness):
        self.source_order = source_order
        self.target_order = target_order

    def transform(self, data: NDArray[Any]) -> NDArray[Any]:
        """Swap bytes if target != source. Else it is a no-op."""
        if self.source_order is not self.target_order:
            data = data.newbyteorder(self.target_order.symbol).byteswap()
        return data

    def inverse_transform(self, data: NDArray[Any]) -> NDArray[Any]:
        """Swap bytes if source != target. Else it is a no-op."""
        if self.target_order is not self.source_order:
            data = data.newbyteorder(self.source_order.symbol).byteswap()
        return data


class TransformPipeline:
    """Pipeline to chain transforms forward and backward."""

    def __init__(self) -> None:
        self.transformations: list[TransformStrategy] = []

    def add_transformation(self, transformation: TransformStrategy) -> None:
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
    ) -> TransformStrategy:
        """Create an instance of transformation and return it."""
        if transform_type == "byte_swap":
            return ByteSwapStrategy(**parameters)

        if transform_type == "scale":
            return ScaleStrategy(**parameters)

        if transform_type == "scale_field":
            return ScaleFieldStrategy(**parameters)

        msg = f"Unsupported transformation type: {transform_type}"
        raise KeyError(msg)
