"""Core functionality for SEG-Y ninja templates."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Literal
from typing import cast

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic.alias_generators import to_camel

from segy.compat import StrEnum

if TYPE_CHECKING:
    from typing import Any

    import numpy as np


class CamelCaseModel(BaseModel):
    """Base model with camel case aliases. Extends BaseModel.

    Attributes:
        model_config: The configuration dictionary for the model.
            - alias_generator: The function used to generate aliases.
            - populate_by_name: Flag indicating whether to populate by name.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        validate_assignment=True,
    )

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        """Dump the model into a dictionary by alias."""
        return super().model_dump(*args, **kwargs, by_alias=True)

    def model_dump_json(self, *args: Any, **kwargs: Any) -> str:  # noqa: ANN401
        """Dump the model into a JSON string by alias."""
        return super().model_dump_json(*args, **kwargs, by_alias=True)

    def _repr_json_(self) -> dict[str, Any]:
        """Return JSON-able dictionary form of model to render in notebooks."""
        return self.model_dump(mode="json")


class BaseDataType(CamelCaseModel):
    """A base model for all SEG-Y Ninja types."""

    @property
    @abstractmethod
    def dtype(self) -> np.dtype[Any]:
        """Abstract property for subclasses to return the numpy dtype."""

    @property
    def itemsize(self) -> int:
        """Number of bytes for the data type."""
        return self.dtype.itemsize


class Endianness(StrEnum):
    """Enumeration class with three possible endianness values.

    Examples:
        >>> endian = Endianness.BIG
        >>> print(endian.symbol)
        >
    """

    BIG = "big"
    LITTLE = "little"
    NATIVE = "native"

    def __init__(self, _: str) -> None:
        self._symbol_map = {
            "big": ">",
            "little": "<",
            "native": "=",
        }

    @property
    def symbol(self) -> Literal["<", ">", "="]:
        """Get the numpy symbol for the endianness from mapping."""
        return cast(Literal["<", ">", "="], self._symbol_map[self.value])
