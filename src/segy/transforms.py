"""Transforms to apply to arrays and structured arrays."""

from __future__ import annotations

import logging
import sys
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from segy.constants import REV1_BASE16
from segy.schema import SegyStandard
from segy.schema.base import Endianness

if TYPE_CHECKING:
    from typing import Any

    from numpy._typing._dtype_like import _DTypeDict
    from numpy.typing import DTypeLike
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)


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

    # Handle the case where field to modify is a struct
    # For example; a Trace struct with "header" amd "data" fields.
    if isinstance(key_to_modify, np.dtype) and key_to_modify.kind == "V":
        new_type = np.dtype((new_type.str,) + key_to_modify.subdtype[1:])  # type: ignore

    dtype_info["formats"][key_index] = new_type  # type: ignore[index]

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
    # Example: ibm32 to float32 still fits in the same 32-bits.
    data[key] = transformed.view(old_field_dtype)
    return data.view(new_struct_dtype)


class Transform:
    """Base class for header transformation strategies."""

    def __init__(self, keys: list[str] | None = None, copy: bool = False) -> None:
        self.keys = keys
        self.copy = copy

        logger.debug("Initialized %s.", self.__class__.__name__)

    def apply(self, data: NDArray[Any]) -> NDArray[Any]:
        """Applies transformation based on ndarray or struct array.

        For memory efficiency, by default, the transforms are applied to input array.
        This could cause mutation of input array depending on the transform. Set the
        `copy` option to `True` if you want to ensure input data isn't modified.

        Args:
            data: The array to be transformed.

        Returns:
            Modified array or copy of array.
        """
        if self.copy:
            data = data.copy()

        if self.keys is None:
            return self.transform(data)

        return self.transform_struct(data)

    @abstractmethod
    def transform(self, data: NDArray[Any]) -> NDArray[Any]:
        """Abstract transformation method on numpy arrays."""

    def transform_struct(self, data: NDArray[Any]) -> NDArray[Any]:
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
            transformed = self.transform(data[key])
            data = _modify_structured_field(data, key, transformed)

        return data


class ByteSwapTransform(Transform):
    """Byte swaps numeric data by based on target order.

    Args:
        target_order: Desired byte order.
    """

    def __init__(self, target_order: Endianness) -> None:
        super().__init__()
        self.target_order = target_order
        logger.debug("Converting to %s endian.", self.target_order.value)

    def transform(self, data: NDArray[Any]) -> NDArray[Any]:
        """Byte swap numpy array given target order."""
        source_order = get_endianness(data)

        if source_order != self.target_order:
            data = data.byteswap(inplace=True)
            swapped_dtype = data.dtype.newbyteorder(self.target_order.symbol)
            data = data.view(swapped_dtype)

        return data


class IbmFloatTransform(Transform):
    """IBM float convert an array.

    Args:
        direction: IBM Float conversion direction.
        keys: Optional list of keys to apply the transform.
    """

    # To map user parameter to compiled function and its expected type.
    ibm_func_map = {
        "to_ibm": ("ieee2ibm", "float32"),
        "to_ieee": ("ibm2ieee", "uint32"),
    }

    def __init__(self, direction: str, keys: list[str] | None = None) -> None:
        super().__init__(keys)
        self.direction = direction
        logger.debug("Converting %s.", self.direction)

    def transform(self, data: NDArray[Any]) -> NDArray[Any]:
        """Convert floats between IEEE and IBM."""
        from segy import ibm

        func_name, cast_dtype = self.ibm_func_map[self.direction]
        func = getattr(ibm, func_name)

        return func(data.astype(cast_dtype))  # type: ignore


class SegyRevisionTransform(Transform):
    """Interpret the SEG-Y revision field in binary header."""

    def __init__(self) -> None:
        super().__init__()

    def transform(self, data: NDArray[Any]) -> NDArray[Any]:
        """Parse SEG-Y standard from binary header."""
        if data.dtype.names is not None and "segy_revision" not in data.dtype.names:
            return data  # rev0, no-op

        # Rev1 needs special treatment.
        # Rev1 is 16-bit with Q-point between the bytes. That means
        # SEG-Y 1.0 is written as 00000001 00000000 in binary, 256 in base-2.
        if data["segy_revision"] == REV1_BASE16:
            data["segy_revision"] = SegyStandard.REV1

        # Rev2 doesn't need special treatment because it splits into
        # two 8-bit integers for major and minor versions.
        # SEG-Y Rev2.0 is 00000010 00000000 in binary, (2, 0) in base-2
        # SEG-Y Rev2.1 is 00000010 00000001 in binary, (2, 1) in base-2

        return data


class TraceTransform(Transform):
    """Composite transform to apply header and data pipeline to trace.

    This transform is a workaround for a design flaw in applying transforms.
    Please refactor this at some point! The problem is: if we have a trace
    struct with "headers" and "data", and we want to apply another transform
    to a header field within "headers", we can't. This is a problem with the
    current Transform and TransformPipeline.

    Args:
        header_pipeline: Transform pipeline to apply to struct "header" field.
        data_pipeline: Transform pipeline to apply to struct "header" field.
    """

    def __init__(
        self, header_pipeline: TransformPipeline, data_pipeline: TransformPipeline
    ) -> None:
        super().__init__()
        self.header_transform = Transform(["header"])  # type: ignore[abstract]
        self.header_transform.transform = header_pipeline.apply  # type: ignore[method-assign]

        self.data_transform = Transform(["data"])  # type: ignore[abstract]
        self.data_transform.transform = data_pipeline.apply  # type: ignore[method-assign]

    def transform(self, data: NDArray[Any]) -> NDArray[Any]:
        """Applies independent transform pipelines to trace struct."""
        data = self.header_transform.apply(data)
        data = self.data_transform.apply(data)

        return data  # noqa: RET504


class TransformFactory:
    """Factory class to generate transformation strategies."""

    transform_map: dict[str, type[Transform]] = {
        "byte_swap": ByteSwapTransform,
        "ibm_float": IbmFloatTransform,
        "segy_revision": SegyRevisionTransform,
        "trace": TraceTransform,
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
            logger.error(msg)
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
