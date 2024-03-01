"""Compatibility layer for different Python versions."""

import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    # Fallback for older Python versions
    from enum import Enum

    class StrEnum(str, Enum):
        """Fallback StrEnum implementation for Python <= 3.10."""

        def __new__(cls, value: str) -> str:  # type: ignore
            """Ensures that enum members are instances of str."""
            member = str.__new__(cls, value)
            member._value_ = value
            return member


__all__ = ["StrEnum"]
