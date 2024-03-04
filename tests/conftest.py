"""Helper functions that are used throughout tests.

This is very janky, we need to find a better way to do this.
"""


from __future__ import annotations

import string
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import pytest

from segy.schema import Endianness
from segy.schema import ScalarType
from segy.schema import TraceDataDescriptor
from segy.schema import TraceDescriptor
from segy.schema.data_type import StructuredDataTypeDescriptor
from segy.schema.data_type import StructuredFieldDescriptor

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture(scope="session")
def format_str_to_text_header() -> Callable[[str], str]:
    """Fixture wrapper around helper function to format text headers."""

    def _format_str_to_text_header(text: str) -> str:
        """Helper function to create fixed size text headers from a given string."""
        return "{0:{fill}{align}{n}}".format(
            text.replace("\n", ""), fill="", align="<", n=3200
        )

    return _format_str_to_text_header


@pytest.fixture(scope="module")
def make_header_field_descriptor() -> (
    Callable[
        [str, list[str] | None, list[int] | None, Endianness],
        dict[str, list[StructuredFieldDescriptor] | int],
    ]
):
    """Fixture wrapper around helper function to generate params for descriptors."""

    def _make_header_field_descriptor(
        dt_string: str = "i2",
        names: list[str] | None = None,
        offsets: list[int] | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> dict[str, list[StructuredFieldDescriptor] | int]:
        """Convenience function for creating parameters needed for descriptors.

        Args:
            dt_string: numpy dtype string. Defaults to "i2".
            names: list of field names. Defaults to None.
            offsets: list of field offsets. Defaults to None.
            endianness: flag for field endianness. Defaults to "big".

        Returns:
            dict: parameters for creating other descriptors
        """
        names = (
            names
            if names is not None
            else generate_unique_names(dt_string.count(",") + 1)
        )
        temp_dt = np.dtype(dt_string)
        item_size = temp_dt.itemsize
        # becuase mypy doesn't like MappingProxy
        temp_dt_field_values: list[tuple[Any, ...]] = list(temp_dt.fields.values())  # type: ignore[union-attr]
        dt_offsets = (
            offsets
            if offsets is not None
            else [field[-1] for field in temp_dt_field_values]
        )
        header_fields = [
            StructuredFieldDescriptor(
                name=n,
                format=ScalarType(np.dtype(dstr).name),
                offset=offs,
                endianness=endianness,
            )
            for n, dstr, offs in zip(names, dt_string.split(","), dt_offsets)
        ]
        return {"fields": header_fields, "item_size": item_size, "offset": 0}

    return _make_header_field_descriptor


@pytest.fixture(scope="module")
def make_trace_header_descriptor(
    make_header_field_descriptor: Callable[
        [str, list[str] | None, list[int] | None, str | Endianness],
        dict[str, list[StructuredFieldDescriptor] | int],
    ],
) -> Callable[..., StructuredDataTypeDescriptor]:
    """Fixture wrapper for helper function to create trace header descriptors."""

    def _make_trace_header_descriptor(
        dt_string: str = "i2",
        names: list[str] | None = None,
        offsets: list[int] | None = None,
        endianness: str | Endianness = Endianness.BIG,
    ) -> StructuredDataTypeDescriptor:
        """Convenience function for creating trace header descriptors.

        Args:
            dt_string: numpy dtype string. Defaults to "i2".
            names: list of field names. Defaults to None.
            offsets: list of field offsets. Defaults to None.
            endianness: flag for field endianness. Defaults to "big".

        Returns:
            Descriptor object for trace headers.
        """
        head_field_desc: dict[str, Any] = make_header_field_descriptor(
            dt_string, names, offsets, endianness
        )

        return StructuredDataTypeDescriptor(
            fields=head_field_desc["fields"],
            item_size=head_field_desc["item_size"],
            offset=head_field_desc["offset"],
        )

    return _make_trace_header_descriptor


@pytest.fixture(scope="module")
def make_trace_data_descriptor() -> Callable[..., TraceDataDescriptor]:
    """Fixture wrapper for helper function to create TraceDataDescriptor."""

    def _make_trace_data_descriptor(
        format: ScalarType = ScalarType.IBM32,  # noqa: A002
        endianness: Endianness = Endianness.BIG,
        description: str | None = None,
        samples: int = 10,
    ) -> TraceDataDescriptor:
        """Convenience function for creating TraceDataDescriptors.

        Args:
            format: ScalarType of data. Defaults to ScalarType.IBM32.
            endianness: flag for field byteorder. Defaults to Endianness.BIG.
            description: descriptive text attached to field. Defaults to None.
            samples: integer defining the shape of the dtype. Defaults to 10.

        Returns:
            TraceDataDescriptor: Descriptor object for TraceDataDescriptor
        """
        return TraceDataDescriptor(
            format=format,
            endianness=endianness,
            description=description,
            samples=samples,
        )

    return _make_trace_data_descriptor


@pytest.fixture(scope="module")
def make_trace_descriptor(
    make_trace_header_descriptor: Callable[..., StructuredDataTypeDescriptor],
    make_trace_data_descriptor: Callable[..., TraceDataDescriptor],
) -> Callable[..., TraceDescriptor]:
    """Fixture wrapper for helper function to create trace descriptors."""

    def _make_trace_descriptor(
        head_params: dict[str, str | list[str] | Endianness],
        data_params: dict[str, str | int | Endianness],
    ) -> TraceDescriptor:
        """Convenience function for creating TraceDescriptor object.

        Args:
            head_params: dictionary containing params for trace header descriptor
            data_params: dictionary containing params for trace data descriptor


        Returns:
            TraceDescriptor: Descriptor object for TraceDescriptor
        """
        return TraceDescriptor(
            header_descriptor=make_trace_header_descriptor(**head_params),
            data_descriptor=make_trace_data_descriptor(**data_params),
        )

    return _make_trace_descriptor


@pytest.fixture(scope="module")
def make_binary_header_descriptor(
    make_header_field_descriptor: Callable[
        [str, list[str] | None, list[int] | None, Endianness | str],
        dict[str, list[StructuredFieldDescriptor] | int],
    ],
) -> Callable[..., StructuredDataTypeDescriptor]:
    """Fixture wrapper around helper function for creating binary header."""

    def _make_binary_header_descriptor(
        dt_string: str = "i2",
        names: list[str] | None = None,
        offsets: list[int] | None = None,
        endianness: str = Endianness.BIG,
    ) -> StructuredDataTypeDescriptor:
        """Helper function for creating binary header descriptor objects.

        Args:
            dt_string: numpy dtype string. Defaults to "i2".
            names: list of field names. Defaults to None.
            offsets: list of field offsets. Defaults to None.
            endianness: flag for field endianness. Defaults to "big".

        Returns:
            Structured data type descriptor object for binary header
        """
        head_field_desc: dict[str, Any] = make_header_field_descriptor(
            dt_string, names, offsets, endianness
        )
        return StructuredDataTypeDescriptor(
            fields=head_field_desc["fields"],
            item_size=head_field_desc["item_size"],
            offset=head_field_desc["offset"],
        )

    return _make_binary_header_descriptor


def generate_unique_names(count: int) -> list[str]:
    """Helper function to create random unique names as placeholders during testing."""
    names: set[str] = set()
    rng = np.random.default_rng()
    while len(names) < count:
        name_length = rng.integers(5, 10)  # noqa: S311
        letters = rng.choice(list(string.ascii_uppercase), size=name_length)  # noqa: S311
        name = "".join(letters)
        names.add(name)
    return list(names)
