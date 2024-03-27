"""Transforms to apply to arrays and structured arrays."""

from __future__ import annotations

import sys
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from segy.ibm import ibm2ieee
from segy.ibm import ieee2ibm
from segy.schema import Endianness

if TYPE_CHECKING:
    from typing import Any

    from numpy._typing._dtype_like import _DTypeDict
    from numpy.typing import DTypeLike
    from numpy.typing import NDArray


def get_endianness(data: NDArray[Any]) -> Endianness:
    """Map the numpy byte order character to Endianness."""
    if data.dtype.isnative:
        return Endianness[sys.byteorder.upper()]

    return Endianness.LITTLE if data.dtype.byteorder == "<" else Endianness.BIG


def _extract_nested_dtype(dtype: np.dtype[Any]) -> DTypeLike:
    """Extract nested dtype information from struct dtype."""
    if dtype.names is None or dtype.fields is None:
        return dtype

    dtype_info: _DTypeDict = {
        "names": dtype.names,
        "formats": [
            _extract_nested_dtype(dtype.fields[name][0]) for name in dtype.names
        ],
        "offsets": [dtype.fields[name][1] for name in dtype.names],
        "itemsize": dtype.itemsize,
    }

    return dtype_info


def _modify_dtype_field(
    old_dtype: np.dtype[np.void],
    key: str,
    new_type: np.dtype[Any],
) -> np.dtype[np.void]:
    """Constructs a new dtype for the structured array after a type change in one of its fields."""
    if old_dtype.names is None:  # pragma: no cover
        msg = "Cannot modify dtype for non-structured array."
        raise ValueError(msg)

    dtype_info: _DTypeDict = _extract_nested_dtype(old_dtype)  # type: ignore[assignment]

    key_index = dtype_info["names"].index(key)
    key_to_modify = dtype_info["formats"][key_index]

    if key_to_modify.kind == "V":
        new_type = np.dtype((new_type.str,) + key_to_modify.subdtype[1:])

    dtype_info["formats"][key_index] = new_type  # type: ignore[index]

    # Check if field size changed
    old_size = key_to_modify.itemsize
    new_size = new_type.itemsize
    diff_size = new_size - old_size

    if diff_size != 0:
        offsets = dtype_info["offsets"]
        itemsize = dtype_info["itemsize"]
        dtype_info["offsets"] = [
            x + (diff_size if i > key_index else 0) for i, x in enumerate(offsets)
        ]
        dtype_info["itemsize"] = itemsize + diff_size

    return np.dtype(dtype_info)


def _modify_structured_field(
    data: NDArray[np.void],
    key: str,
    transformed: NDArray[Any],
) -> NDArray[np.void]:
    """Handle structured array assignment with dtype change or promotion."""
    if data.dtype.names is None:  # pragma: no cover
        msg = "Cannot modify dtype for non-structured array."
        raise ValueError(msg)

    # Assign and return early if basic types are the same. Numpy handles endianness.
    if transformed.dtype == data[key].dtype:
        data[key] = transformed
        return data

    # If transform changed field type, we need to define the new array dtype.
    old_field_dtype = data[key].dtype
    new_field_dtype = transformed.dtype
    new_struct_dtype = _modify_dtype_field(data.dtype, key, new_field_dtype)

    # If the field size hasn't changed we can do it in-place with view magic, no copy.
    # Example: uint32 scaled by a negative integer, result fits into output: int32.
    if new_struct_dtype.itemsize == data.dtype.itemsize:
        data[key] = transformed.view(old_field_dtype)
        return data.view(new_struct_dtype)

    # Worst case scenario, field size changes. We need to initialize a new
    # array and copy all the old data and the new data to appropriate fields.
    # Example: int32 field scaled by float32 scalar, resulting in float64 output.
    new_data = np.empty_like(data, dtype=new_struct_dtype)
    for orig_key in data.dtype.names:
        if orig_key == key:  # skip transformed key
            continue
        new_data[orig_key] = data[orig_key]

    new_data[key] = transformed

    return new_data


class Transform:
    """Base class for header transformation strategies."""

    def __init__(self, keys: list[str] | None = None) -> None:
        self.keys = keys

    def apply(self, data: NDArray[Any]) -> NDArray[Any]:
        """Applies transformation based on ndarray or struct array."""
        if self.keys is None:
            return self._transform(data)

        return self._transform_struct(data)

    @abstractmethod
    def _transform(self, data: NDArray[Any]) -> NDArray[Any]:
        """Abstract transformation method on numpy arrays."""

    def _transform_struct(self, data: NDArray[Any]) -> NDArray[Any]:
        """Generalized transform for structured arrays."""
        if self.keys is None:  # pragma: no cover
            msg = "Trying to modify structured array fields with no keys provided."
            raise ValueError(msg)

        if data.dtype.names is None:
            msg = "The array to transform doesn't contain any fields."
            raise ValueError(msg)

        missing_fields = set(self.keys) - set(data.dtype.names)
        if missing_fields:
            msg = f"Field(s) {missing_fields} not in the array."
            raise ValueError(msg)

        for key in self.keys:
            transformed = self._transform(data[key])
            data = _modify_structured_field(data, key, transformed)

        return data


class ScaleTransform(Transform):
    """Scales numeric data by a specified factor.

    Args:
        scalar: Scalar to apply to the data.
        keys: Optional list of keys to apply the transform.
    """

    def __init__(self, scalar: float, keys: list[str] | None = None) -> None:
        super().__init__(keys)
        self.scalar = scalar

    def _transform(self, data: NDArray[Any]) -> NDArray[Any]:
        orig_endian = get_endianness(data)
        data = data * self.scalar
        new_endian = get_endianness(data)
        if new_endian is not orig_endian:
            data = data.byteswap().newbyteorder()
        return data


class ByteSwapTransform(Transform):
    """Byte swaps numeric data by based on target order.

    Args:
        target_order: Desired byte order.
    """

    def __init__(self, target_order: Endianness) -> None:
        super().__init__()
        self.target_order = target_order

    def _transform(self, data: NDArray[Any]) -> NDArray[Any]:
        source_order = get_endianness(data)

        if source_order != self.target_order:
            data = data.newbyteorder(self.target_order.symbol).byteswap()

        return data


class CastTypeTransform(Transform):
    """Casts numeric data based on target type.

    Args:
        target_type: Desired data type to cast.
        keys: Optional list of keys to apply the transform.
    """

    def __init__(self, target_type: DTypeLike, keys: list[str] | None = None) -> None:
        super().__init__(keys=keys)
        self.target_type = np.dtype(target_type)

    def _transform(self, data: NDArray[Any]) -> NDArray[Any]:
        return data.astype(self.target_type)


class IbmFloatTransform(Transform):
    """IBM float convert an array.

    Args:
        direction: IBM Float conversion direction.
        keys: Optional list of keys to apply the transform.
    """

    ibm_func_map = {
        "to_ibm": lambda x: ieee2ibm(x.astype("float32")),
        "to_ieee": lambda x: ibm2ieee(x.view("uint32")),
    }

    def __init__(self, direction: str, keys: list[str] | None = None) -> None:
        super().__init__(keys)
        self.direction = direction

    def _transform(self, data: NDArray[Any]) -> NDArray[Any]:
        return self.ibm_func_map[self.direction](data)  # type: ignore


class TransformFactory:
    """Factory class to generate transformation strategies."""

    transform_map: dict[str, type[Transform]] = {
        "scale": ScaleTransform,
        "byte_swap": ByteSwapTransform,
        "ibm_float": IbmFloatTransform,
        "cast": CastTypeTransform,
    }

    @classmethod
    def create(
        cls: type[TransformFactory],
        transform_type: str,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> Transform:
        """Create an instance of transformation and return it."""
        if transform_type not in cls.transform_map:
            msg = f"Unsupported transformation type: {transform_type}"
            raise KeyError(msg)

        return cls.transform_map[transform_type](*args, **kwargs)


class TransformPipeline:
    """Executes a sequence of transformations on data."""

    def __init__(self) -> None:
        self.transforms: list[Transform] = []

    def add_transform(self, transform: Transform) -> None:
        """Adds a transformation to the pipeline."""
        self.transforms.append(transform)

    def apply(self, data: NDArray[Any]) -> NDArray[Any]:
        """Applies all transformations in sequence."""
        for transform in self.transforms:
            data = transform.apply(data)
        return data
