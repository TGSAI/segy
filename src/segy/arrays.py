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

    def __getitem__(self, item: Any) -> HeaderArray:  # noqa: ANN401
        """Special getitem where we normalize header keys. Pass along to numpy."""
        if isinstance(item, list) and all(isinstance(key, str) for key in item):
            item_orig = copy(item)
            item = [normalize_key(key) for key in item]
            [validate_key(k1, k2, self.dtype.names) for k1, k2 in zip(item, item_orig)]

        if isinstance(item, str):
            item_orig = copy(item)
            item = normalize_key(item)
            validate_key(item, item_orig, self.dtype.names)

        return super().__getitem__(item)  # type: ignore[no-any-return]


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
