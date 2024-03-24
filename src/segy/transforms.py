"""Transforms to apply to arrays and structured arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from numpy import dtype
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import KeysView
    from collections.abc import ValuesView
    from typing import Any
    from typing import Protocol
    from typing import Union

    from numpy.typing import NDArray

    from segy.schema import Endianness

    class DTypeFields(Protocol):
        """Typed protocol for dtype.fields mapping methods."""

        def keys(self) -> KeysView[str]:
            """Keys view protocol."""
            ...

        def values(self) -> ValuesView[Any]:
            """Values view protocol."""
            ...

        def items(self) -> Iterable[tuple[str, Any]]:
            """Mapping items protocol."""
            ...

    class SupportsFields(Protocol):
        """Typed protocol for dtype methods."""

        fields: DTypeFields | None
        names: tuple[str, ...] | None
        char: str
        itemsize: int
        base: dtype[Any] | None

    StructDtype = Union[dtype[Any], SupportsFields]


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
                data[key] = data[key] * self.scale_factor
        return data

    def inverse_transform(self, data: NDArray[Any]) -> NDArray[Any]:
        """Inverse transformation implementation."""
        if data.dtype.names is None:
            msg = f"{self.__class__.__name__} can only work on structured arrays."
            raise ValueError(msg)

        for key in self.keys:
            if key in data.dtype.names:
                data[key] = data[key] / self.scale_factor
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


class FloatConversionStrategy(TransformStrategy):
    """Transform to convert float types in a structured dtype.

    The logic of this strategy is to handle converting subfields
    of an array that match 'source_dtype' to a different 'target_dtype'.
    If an array has no fields, it will check if the 'source_dtype'
    matches the type of the array, and will convert the whole array
    to 'target_dtype'.

    Args:
        source_dtype: The dtype to match for converting.
        target_dtype: The dtype to convert to.
        forward_convert_func: Function to convert source_dtype to target_dtype
        reverse_convert_func: Function to convert target_dtype to source_dtype
    """

    def __init__(
        self,
        source_dtype: dtype[Any],
        target_dtype: dtype[Any],
        forward_convert_func: Callable[..., NDArray[Any]],
        reverse_convert_func: Callable[..., NDArray[Any]],
    ):
        self.source_dtype = source_dtype
        self.target_dtype = target_dtype
        self.forward_convert_func = forward_convert_func
        self.reverse_convert_func = reverse_convert_func

    def _find_type_fields(self, struct_dtype: StructDtype, type_char: str) -> list[str]:
        if struct_dtype.fields is not None:
            # Find the names of the fields that match the dtype to convert
            return [
                name
                for (name, ndtype) in struct_dtype.fields.items()
                if ndtype[0].char == type_char
            ]
        return []

    def _replace_type_fields(
        self,
        struct_dtype: dtype[Any],
        field_names_to_replace: list[str],
        replacement_type: dtype[Any],
    ) -> dtype[Any]:
        if struct_dtype.fields is not None:
            # Check all the named fields in dtype, replace the ones to convert
            # otherwise keep the field the same
            return dtype(
                [
                    (
                        name,
                        (
                            cur_dtype[0]
                            if name not in field_names_to_replace
                            else replacement_type
                        ),
                    )
                    for name, cur_dtype in struct_dtype.fields.items()
                ]
            )
        return struct_dtype

    def _convert_fields(
        self,
        data: NDArray[Any],
        convert_func: Callable[..., NDArray[Any]],
        from_type: dtype[Any],
        to_type: dtype[Any],
    ) -> NDArray[Any]:
        # Check if any fields of data match the type to convert
        if convert_fields := self._find_type_fields(data.dtype, from_type.char):
            # Create a new dtype with replaced types for fields to convert
            converted_dtype = self._replace_type_fields(
                data.dtype, convert_fields, to_type
            )
            data_view = data.view(converted_dtype)
            for field in convert_fields:
                data_view[field] = convert_func(data[field])
            return data_view
        # If no fields in data, check if the type to convert matches the whole array
        if data.dtype == from_type:
            data_view = data.view(to_type)
            data_view[:] = convert_func(data)
            return data_view
        return data

    def transform(self, data: NDArray[Any]) -> NDArray[Any]:
        """Forward transformation implementation.

        Args:
            data: Input data to transform.

        Returns:
            Transformed data.
        """
        return self._convert_fields(
            data, self.forward_convert_func, self.source_dtype, self.target_dtype
        )

    def inverse_transform(self, data: NDArray[Any]) -> NDArray[Any]:
        """Inverse transformation implementation.

        Args:
            data: Input data to transform.

        Returns:
            Transformed data.
        """
        return self._convert_fields(
            data, self.reverse_convert_func, self.target_dtype, self.source_dtype
        )


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

        if transform_type == "float_convert":
            return FloatConversionStrategy(**parameters)

        msg = f"Unsupported transformation type: {transform_type}"
        raise KeyError(msg)
