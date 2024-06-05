"""Tests for components that define fields and data types of a SEGY Schema."""

from __future__ import annotations

import numpy as np
import pytest

from segy.schema.data_type import DataTypeDescriptor
from segy.schema.data_type import Endianness
from segy.schema.data_type import ScalarType
from segy.schema.data_type import StructuredDataTypeDescriptor
from segy.schema.data_type import StructuredFieldDescriptor


class TestScalarTypeDescriptor:
    """Tests for scalar data type descriptor."""

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


class TestStructuredDataTypeDescriptor:
    """Tests for structured field and structured type descriptors."""

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

    def test_add_structured_field(self) -> None:
        """Test adding fields to descriptor with and without overwrite."""
        struct_descriptor = StructuredDataTypeDescriptor(fields=[])

        field1 = StructuredFieldDescriptor(name="f1", byte=3, format=ScalarType.INT16)
        field2 = StructuredFieldDescriptor(name="f2", byte=7, format=ScalarType.INT16)

        struct_descriptor.add_field(field1)
        struct_descriptor.add_field(field2)

        actual_fields = dict(struct_descriptor.dtype.fields)  # type: ignore
        assert struct_descriptor.dtype.itemsize == field2.offset + field2.dtype.itemsize
        assert struct_descriptor.dtype.names == ("f1", "f2")
        assert actual_fields["f1"] == (field1.dtype, field1.offset)
        assert actual_fields["f2"] == (field2.dtype, field2.offset)

        # Test with overwrite.
        field2 = StructuredFieldDescriptor(name="f2", byte=11, format=ScalarType.INT32)
        struct_descriptor.add_field(field2, overwrite=True)

        actual_fields = dict(struct_descriptor.dtype.fields)  # type: ignore
        assert struct_descriptor.dtype.itemsize == field2.offset + field2.dtype.itemsize
        assert actual_fields["f1"] == (field1.dtype, field1.offset)
        assert actual_fields["f2"] == (field2.dtype, field2.offset)

    def test_remove_structured_field(self) -> None:
        """Test field removal from structured data type."""
        struct_descriptor = StructuredDataTypeDescriptor(
            fields=[
                StructuredFieldDescriptor(name="f1", byte=11, format=ScalarType.INT8),
                StructuredFieldDescriptor(name="f2", byte=21, format=ScalarType.INT16),
            ]
        )

        struct_descriptor.remove_field("f2")

        remaining_field = struct_descriptor.fields[0]
        expected_itemsize = remaining_field.offset + remaining_field.dtype.itemsize
        assert struct_descriptor.itemsize == expected_itemsize
        assert struct_descriptor.dtype.names == ("f1",)


class TestStructuredDataTypeDescriptorExceptions:
    """Test exceptions in structured descriptors."""

    def test_duplicate_field_exception(self) -> None:
        """Test expected failure when multiple keys are provided more than once."""
        fields_with_duplicate = [
            StructuredFieldDescriptor(name="f1", byte=3, format=ScalarType.INT16),
            StructuredFieldDescriptor(name="f1", byte=9, format=ScalarType.INT16),
        ]
        with pytest.raises(ValueError, match="Duplicate header fields detected"):
            StructuredDataTypeDescriptor(fields=fields_with_duplicate)

    def test_add_field_exception(self) -> None:
        """Test adding fields that already exists without overwrite flag."""
        struct_descriptor = StructuredDataTypeDescriptor(
            fields=[
                StructuredFieldDescriptor(name="f1", byte=3, format=ScalarType.INT16),
                StructuredFieldDescriptor(name="f2", byte=11, format=ScalarType.INT16),
            ]
        )

        field = StructuredFieldDescriptor(name="f1", byte=7, format=ScalarType.UINT8)
        with pytest.raises(KeyError, match="f1 already exists."):
            struct_descriptor.add_field(field)

    def test_remove_field_exception(self) -> None:
        """Test field removal from structured data type."""
        struct_descriptor = StructuredDataTypeDescriptor(
            fields=[
                StructuredFieldDescriptor(name="f1", byte=11, format=ScalarType.INT8),
                StructuredFieldDescriptor(name="f2", byte=21, format=ScalarType.INT16),
            ]
        )

        with pytest.raises(KeyError, match="f3 does not exist."):
            struct_descriptor.remove_field("f3")

    @pytest.mark.parametrize(
        ("names", "bytes_", "formats", "itemsize"),
        [
            (["f1"], [1], [ScalarType.UINT64], 4),
            (["f1", "f2"], [1, 24], [ScalarType.UINT8, ScalarType.INT16], 16),
        ],
    )
    def test_struct_size_overflow(
        self,
        names: list[str],
        bytes_: list[int],
        formats: list[ScalarType],
        itemsize: int,
    ) -> None:
        """Test validation where the max field size exceeds the item size."""
        fields = [
            StructuredFieldDescriptor(name=name, byte=byte, format=format_)
            for name, byte, format_ in zip(names, bytes_, formats)
        ]
        with pytest.raises(ValueError, match="Offsets exceed allowed header size."):
            StructuredDataTypeDescriptor(fields=fields, item_size=itemsize)


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
