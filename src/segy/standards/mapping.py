"""Mappings from internal schemas to SEG-Y definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import TypeVar

from segy.schema import ScalarType

if TYPE_CHECKING:
    from collections.abc import Iterable

K = TypeVar("K")
V = TypeVar("V")


class BiDict(dict):
    """Bidirectional dictionary."""

    def __init__(self, *args: Iterable[tuple[K, V]], **kwargs: V) -> None:
        super().__init__(*args, **kwargs)
        self.reverse = {v: k for k, v in self.items()}

    def __setitem__(self, key: K, value: V) -> None:
        """Set a new key in bidirectional dictionary."""
        if value in self.reverse:
            msg = "Value already exists with a different key"
            raise ValueError(msg)
        super().__setitem__(key, value)
        self.reverse[value] = key

    def __delitem__(self, key: K) -> None:
        """Delete key from dictionary."""
        value = self[key]
        super().__delitem__(key)
        del self.reverse[value]

    def __contains__(self, item: K | V) -> bool:
        """Return True if item is in bidirectional dictionary."""
        return super().__contains__(item) or item in self.reverse

    def update(self, *args: Iterable[tuple[K, V]], **kwargs: V) -> None:
        """Update the bidirectional dictionary."""
        for k, v in dict(*args, **kwargs).items():
            self.__setitem__(k, v)

    def inverse(self, value: V) -> K:
        """Get key by value."""
        return self.reverse[value]


SEGY_FORMAT_MAP = BiDict(
    {
        ScalarType.IBM32: 1,
        ScalarType.INT32: 2,
        ScalarType.INT16: 3,
        ScalarType.FLOAT32: 5,
        ScalarType.FLOAT64: 6,
        ScalarType.INT8: 8,
        ScalarType.INT64: 9,
        ScalarType.UINT32: 10,
        ScalarType.UINT16: 11,
        ScalarType.UINT64: 12,
        ScalarType.UINT8: 16,
    }
)
