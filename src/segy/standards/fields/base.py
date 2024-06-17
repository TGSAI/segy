"""Base tuple and enum definitions for SEG-Y standard definitions."""

from __future__ import annotations

from collections import namedtuple
from enum import Enum

from segy.schema import HeaderField

FieldTuple = namedtuple("FieldTuple", ["byte", "format"])


class SegStandardEnum(FieldTuple, Enum):
    """A special enum for defining standards with data model generation attribute."""

    def __repr__(self) -> str:
        """Custom concise repr."""
        cls_name = self.__class__.__name__
        val_name = self._name_
        val_repr = f"byte={self.byte}, format={self.format}"
        return f"<{cls_name}.{val_name}: {val_repr}>"

    def __new__(cls, byte: int, format: str) -> SegStandardEnum:  # noqa: A002
        """This will ensure any tuple is converted to FieldTuple."""
        return FieldTuple.__new__(cls, byte, format)

    @property
    def model(self) -> HeaderField:
        """Generate proper data model for header field with validation."""
        return HeaderField(name=self.name.lower(), byte=self.byte, format=self.format)
