"""Custom array interface.

We subclass NumPy ndarray with some methods to enrich it for
better use experience. Like dictionary or JSON dumps for
structured arrays.

See here for details:
https://numpy.org/doc/stable/user/basics.subclassing.html
"""

from __future__ import annotations

from copy import copy
from json import dumps as json_dumps
from typing import TYPE_CHECKING
from typing import Any

import numpy as np

from segy.alias.core import normalize_key
from segy.alias.core import validate_key

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from typing import Callable
    from typing import Literal

    from numpy.typing import NDArray
    from pandas import DataFrame

    OrderKACF = Literal[None, "K", "A", "C", "F"]


class SegyArray(np.ndarray):  # type: ignore[type-arg]
    """Base class for array interface. Like ndarray but extensible."""

    def __new__(cls, input_array: NDArray[Any]) -> SegyArray:
        """Numpy subclass logic."""
        return np.asarray(input_array).view(cls)

    def __array_finalize__(self, obj: NDArray[Any] | None) -> None:
        """Numpy subclass logic."""
        if obj is None:
            return

    def __array_function__(
        self,
        func: Callable[..., Any],
        types: Iterable[type],
        args: Iterable[Any],
        kwargs: Mapping[str, Any],
    ) -> Any:  # noqa: ANN401
        """Ensure return type is still SegyArray and subtypes if we run numpy funcs.

        Functions like `np.concatenate` come here, and to ensure we keep the type
        as `SegyArray` or its subclasses when its run.
        """
        if not all(issubclass(t, SegyArray) for t in types):
            return NotImplemented

        result = super().__array_function__(func, types, args, kwargs)
        if isinstance(result, np.ndarray):
            return result.view(type(self))

        return result

    def copy(self, order: OrderKACF = "K") -> SegyArray:
        """Copy structured array preserving the padded bytes as is.

        This method ensures that the copy includes raw binary data and any padding
        bytes, preserving the entire memory layout of the array. This is necessary
        for working with SEG-Y data where not all fields are parsed, but raw binary
        data preservation is crucial.
        """
        void_view = self.view("V")
        void_copy = np.copy(void_view, order=order, subok=True)
        return void_copy.view(self.dtype)  # type: ignore[no-any-return]


class HeaderArray(SegyArray):
    """Header ndarray with convenience features."""

    def to_dict(self) -> dict[str, Any]:
        """Convert header to dict."""
        result_dict = {}

        if self.dtype.names is None:
            msg = f"{self.__class__.__name__} can only work on structured arrays."
            raise ValueError(msg)

        for field in self.dtype.names:
            field_values = self[field]
            result_dict[field] = field_values.squeeze().tolist()

        return result_dict

    def to_json(self, indent: int = 2) -> str:
        """Convert header to JSON."""
        return json_dumps(self.to_dict(), indent=indent)

    def to_dataframe(self) -> DataFrame:
        """Convert structured data to pandas DataFrame."""
        from pandas import DataFrame

        return DataFrame.from_records(self)

    def _normalize_and_validate_keys(self, key: str | list[str]) -> str | list[str]:
        if isinstance(key, str):
            original_key = copy(key)
            key = normalize_key(key)
            validate_key(key, original_key, self.dtype.names)
        elif isinstance(key, list):
            original_keys = copy(key)
            key = [normalize_key(k) for k in key]
            for k, orig_k in zip(key, original_keys):
                validate_key(k, orig_k, self.dtype.names)

        return key

    def __getitem__(self, item: Any) -> Any:  # noqa: ANN401
        """Special getitem where we normalize header keys. Pass along to numpy."""
        if isinstance(item, str) and item in self.dtype.names:
            return super().__getitem__(item)

        if isinstance(item, str):
            item = self._normalize_and_validate_keys(item)
        elif isinstance(item, list) and all(isinstance(i, str) for i in item):
            if all(key in self.dtype.names for key in item):
                return super().__getitem__(item)

            item = self._normalize_and_validate_keys(item)

        return super().__getitem__(item)

    def __setitem__(self, key: Any, value: Any) -> None:  # noqa: ANN401
        """Special getitem where we normalize header keys. Pass along to numpy."""
        if isinstance(key, str) and key in self.dtype.names:
            super().__setitem__(key, value)  # type: ignore[no-untyped-call]
            return

        if isinstance(key, str):
            key = self._normalize_and_validate_keys(key)

        elif isinstance(key, list) and all(isinstance(i, str) for i in key):
            if all(k in self.dtype.names for k in key):
                super().__setitem__(key, value)  # type: ignore[no-untyped-call]
                return

            key = self._normalize_and_validate_keys(key)

        super().__setitem__(key, value)  # type: ignore[no-untyped-call]


class TraceArray(SegyArray):
    """Trace ndarray with convenience features."""

    @property
    def header(self) -> HeaderArray:
        """Access headers of the trace(s)."""
        return HeaderArray(self["header"])

    @property
    def sample(self) -> NDArray[Any]:
        """Access data of the trace(s)."""
        return self["data"]
