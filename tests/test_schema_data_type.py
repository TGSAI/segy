"""Tests for components that define fields and data types of a SEGY Schema."""

from __future__ import annotations

import pytest

from segy.schema import Endianness
from segy.schema import HeaderField
from segy.schema import HeaderSpec
from segy.schema import ScalarType


class TestHeaderSpec:
    """Tests for header field spec and header spec."""

    @pytest.mark.parametrize("endianness", [Endianness.BIG, Endianness.LITTLE])
    @pytest.mark.parametrize(
        ("names", "bytes_", "formats", "itemsize"),
        [
            (["f1"], [1], [ScalarType.INT32], None),
            (["f1", "f2"], [1, 9], [ScalarType.UINT8, ScalarType.INT32], None),
            (["f1", "f2"], [1, 13], [ScalarType.UINT8, ScalarType.INT16], 20),
        ],
    )
    def test_header_spec(  # noqa: PLR0913
        self,
        endianness: Endianness,
        names: list[str],
        bytes_: list[int],
        formats: list[ScalarType],
        itemsize: int,
    ) -> None:
        """Test HeaderSpec and its methods."""
        fields = [
            HeaderField(name=name, byte=byte, format=format_)
            for name, byte, format_ in zip(names, bytes_, formats, strict=True)
        ]

        header_spec = HeaderSpec(
            fields=fields, item_size=itemsize, endianness=endianness
        )

        assert header_spec.dtype.names == tuple(names)
        if endianness == Endianness.LITTLE:
            assert header_spec.dtype.isnative
        if endianness == Endianness.BIG:
            assert not header_spec.dtype.isnative
        if itemsize is not None:
            assert header_spec.dtype.itemsize == itemsize

    def test_add_field(self) -> None:
        """Test adding fields to header spec with and without overwrite."""
        header_spec = HeaderSpec(fields=[])

        field1 = HeaderField(name="f1", byte=3, format=ScalarType.INT16)
        field2 = HeaderField(name="f2", byte=7, format=ScalarType.INT16)

        header_spec.add_field(field1)
        header_spec.add_field(field2)

        actual_fields = dict(header_spec.dtype.fields)  # type: ignore
        assert header_spec.dtype.itemsize == field2.offset + field2.dtype.itemsize
        assert header_spec.dtype.names == ("f1", "f2")
        assert actual_fields["f1"] == (field1.dtype, field1.offset)
        assert actual_fields["f2"] == (field2.dtype, field2.offset)

        # Test with overwrite.
        field2 = HeaderField(name="f2", byte=11, format=ScalarType.INT32)
        header_spec.add_field(field2, overwrite=True)

        actual_fields = dict(header_spec.dtype.fields)  # type: ignore
        assert header_spec.dtype.itemsize == field2.offset + field2.dtype.itemsize
        assert actual_fields["f1"] == (field1.dtype, field1.offset)
        assert actual_fields["f2"] == (field2.dtype, field2.offset)

    def test_remove_field(self) -> None:
        """Test field removal from header spec."""
        header_spec = HeaderSpec(
            fields=[
                HeaderField(name="f1", byte=11, format=ScalarType.INT8),
                HeaderField(name="f2", byte=21, format=ScalarType.INT16),
            ]
        )

        header_spec.remove_field("f2")

        remaining_field = header_spec.fields[0]
        expected_itemsize = remaining_field.offset + remaining_field.dtype.itemsize
        assert header_spec.itemsize == expected_itemsize
        assert header_spec.dtype.names == ("f1",)

    def test_header_spec_deserialize(self) -> None:
        """Test for validating recreating a header spec from a JSON string."""
        struct_json = """
        {
          "fields": [
            {"description": "field1", "format": "int32", "name": "f1", "byte": 1},
            {"description": "field2", "format": "ibm32", "name": "f2", "byte": 5}
          ],
          "itemSize": 12,
          "offset": 200,
          "endianness": "big"
        }
        """
        actual_spec = HeaderSpec.model_validate_json(struct_json)
        assert len(actual_spec.fields) == 2  # noqa: PLR2004
        assert actual_spec.dtype.names == ("f1", "f2")
        assert actual_spec.dtype.itemsize == 12  # noqa: PLR2004
        assert not actual_spec.dtype.isnative


class TestHeaderSpecExceptions:
    """Test exceptions in header spec."""

    def test_validation_duplicate_field_exception(self) -> None:
        """Test expected failure when multiple keys are provided more than once."""
        fields_with_duplicate = [
            HeaderField(name="f1", byte=3, format=ScalarType.INT16),
            HeaderField(name="f1", byte=9, format=ScalarType.INT16),
        ]
        with pytest.raises(ValueError, match="Duplicate header fields detected"):
            HeaderSpec(fields=fields_with_duplicate)

    def test_add_field_exists_exception(self) -> None:
        """Test adding fields that already exists without overwrite flag."""
        header_spec = HeaderSpec(
            fields=[
                HeaderField(name="f1", byte=3, format=ScalarType.INT16),
                HeaderField(name="f2", byte=11, format=ScalarType.INT16),
            ]
        )

        field = HeaderField(name="f1", byte=7, format=ScalarType.UINT8)
        with pytest.raises(KeyError, match="f1 already exists."):
            header_spec.add_field(field)

    def test_remove_field_missing_exception(self) -> None:
        """Test field removal from structured data type."""
        header_spec = HeaderSpec(
            fields=[
                HeaderField(name="f1", byte=11, format=ScalarType.INT8),
                HeaderField(name="f2", byte=21, format=ScalarType.INT16),
            ]
        )

        with pytest.raises(KeyError, match="f3 does not exist."):
            header_spec.remove_field("f3")

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
            HeaderField(name=name, byte=byte, format=format_)
            for name, byte, format_ in zip(names, bytes_, formats, strict=True)
        ]
        with pytest.raises(ValueError, match="Offsets exceed allowed header size."):
            HeaderSpec(fields=fields, item_size=itemsize)
