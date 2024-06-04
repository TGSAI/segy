"""Tests for components that define fields and data types of a SEGY Schema."""

from __future__ import annotations

import numpy as np
import pytest

from segy.schema.data_type import DataTypeDescriptor
from segy.schema.data_type import Endianness
from segy.schema.data_type import ScalarType
from segy.schema.data_type import StructuredDataTypeDescriptor
from segy.schema.data_type import StructuredFieldDescriptor


class TestBaseDataTypes:
    """Tests for basic and structured data types and their dtype method."""

    @pytest.mark.parametrize(
        ("format_", "expected"),
        [
            (ScalarType.IBM32, "uint32"),
            (ScalarType.INT64, "int64"),
            (ScalarType.UINT16, "uint16"),
            (ScalarType.UINT8, "uint8"),
            (ScalarType.FLOAT32, "float32"),
        ],
    )
    def test_data_type_descriptor(self, format_: ScalarType, expected: str) -> None:
        """Test creating data type descriptors for `ScalarType`s."""
        dtype_descr = DataTypeDescriptor(format=format_)
        assert dtype_descr.dtype == np.dtype(expected)

    @pytest.mark.parametrize("endianness", [Endianness.BIG, Endianness.LITTLE])
    @pytest.mark.parametrize(
        ("names", "bytes_", "formats", "itemsize"),
        [
            (["f1"], [1], [ScalarType.INT32], None),
            (["f1", "f2"], [1, 9], [ScalarType.UINT8, ScalarType.INT32], None),
            (["f1", "f2"], [1, 13], [ScalarType.UINT8, ScalarType.INT16], 20),
        ],
    )
    def test_structured_data_type_descriptor(  # noqa: PLR0913
        self,
        endianness: Endianness,
        names: list[str],
        bytes_: list[int],
        formats: list[ScalarType],
        itemsize: int,
    ) -> None:
        """Test StructuredDataTypeDescriptor and its dtype attribute."""
        fields = [
            StructuredFieldDescriptor(name=name, byte=byte, format=format_)
            for name, byte, format_ in zip(names, bytes_, formats)
        ]

        struct_descriptor = StructuredDataTypeDescriptor(
            fields=fields, item_size=itemsize, endianness=endianness
        )

        assert struct_descriptor.dtype.names == tuple(names)
        if endianness == Endianness.LITTLE:
            assert struct_descriptor.dtype.isnative
        if endianness == Endianness.BIG:
            assert not struct_descriptor.dtype.isnative
        if itemsize is not None:
            assert struct_descriptor.dtype.itemsize == itemsize


def test_structured_data_type_descriptor_json_validate() -> None:
    """Test for validating recreating a StrucrutedDataTypeDescriptor from a JSON string."""
    struct_json = """
    {
      "description": "dummy description",
      "fields": [
        {"description": "field1", "format": "int32", "name": "f1", "byte": 1},
        {"description": "field2", "format": "ibm32", "name": "f2", "byte": 5}
      ],
      "itemSize": 12,
      "offset": 200,
      "endianness": "big"
    }
    """
    actual_model = StructuredDataTypeDescriptor.model_validate_json(struct_json)
    assert actual_model.description == "dummy description"
    assert len(actual_model.fields) == 2  # noqa: PLR2004
    assert actual_model.dtype.names == ("f1", "f2")
    assert actual_model.dtype.itemsize == 12  # noqa: PLR2004
    assert not actual_model.dtype.isnative
