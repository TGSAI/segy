"""Tests for HeaderField and HeaderSpec."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from segy.schema import Endianness
from segy.schema import HeaderField
from segy.schema import HeaderSpec
from segy.schema import ScalarType


class TestHeaderField:
    """Tests for HeaderField."""

    def test_header_field_initialization(self) -> None:
        """Test successful initialization of HeaderField."""
        field = HeaderField(name="test", format=ScalarType.FLOAT32, byte=9)
        assert field.name == "test"
        assert field.byte == 9
        assert field.offset == 8
        assert field.dtype == np.dtype("float32")
        assert field.range == (9, 13)  # float32 is 4 bytes

    @pytest.mark.parametrize(
        "invalid_byte",
        [0, -1, "not_an_int"],
        ids=["zero", "negative", "non_integer"],
    )
    def test_header_field_invalid_byte(self, invalid_byte: int | str) -> None:
        """Test that invalid byte values raise validation errors."""
        if isinstance(invalid_byte, int):
            expected_regex = "Input should be greater than or equal to 1"
        else:
            expected_regex = "Input should be a valid integer"

        with pytest.raises(ValidationError, match=expected_regex):
            HeaderField(
                name="test_field",
                byte=invalid_byte,
                format=ScalarType.INT32,
            )

    @pytest.mark.parametrize(
        ("name", "format_", "byte"),
        [
            # Different types
            ("field", ScalarType.UINT8, 100),
            ("field", ScalarType.INT16, 100),
            ("field", ScalarType.UINT32, 100),
            ("field", ScalarType.FLOAT64, 100),
            # Different byte positions
            ("test", ScalarType.INT16, 50),
            ("test", ScalarType.INT16, 200),
        ],
    )
    def test_header_field_properties(
        self, name: str, format_: ScalarType, byte: int
    ) -> None:
        """Test range property with different field configurations."""
        field = HeaderField(name=name, format=format_, byte=byte)

        expected_offset = byte - 1
        assert field.offset == expected_offset, "start offset doesn't match"

        expected_dtype = np.dtype(format_)
        assert field.dtype == expected_dtype, "field dtype doesn't match"

        start, stop = field.range
        assert start == byte
        assert stop == byte + format_.dtype.itemsize, "field byte range doesn't match"


class TestHeaderSpec:
    """Tests for HeaderSpec."""

    def test_header_spec_initialization(self, basic_fields: list[HeaderField]) -> None:
        """Test successful initialization of HeaderSpec."""
        header = HeaderSpec(
            fields=basic_fields, item_size=30, endianness=Endianness.LITTLE
        )

        expected_formats = [np.dtype("int32"), np.dtype("int16"), np.dtype("int32")]
        assert len(header.fields) == 3
        assert header.names == ["foo", "bar", "fizz"]
        assert header.offsets == [0, 4, 16]
        assert header.formats == expected_formats
        assert header.item_size == 30
        assert header.endianness == Endianness.LITTLE

        # Check dtype with gaps and padding
        expected_dtype = np.dtype(
            {
                "names": ["foo", "bar", "fizz"],
                "formats": ["<i4", "<i2", "<i4"],
                "offsets": [0, 4, 16],
                "itemsize": 30,
            }
        )
        assert header.dtype == expected_dtype

    def test_header_spec_no_fields(self) -> None:
        """Test empty fields attribute."""
        header = HeaderSpec(fields=[])
        assert header.dtype == np.dtype([])  # Empty dtype

    def test_header_spec_duplicate_names(self, basic_fields: list[HeaderField]) -> None:
        """Test exception when duplicate names exist."""
        duplicate_field = HeaderField(name="foo", format=ScalarType.INT32, byte=21)
        fields_with_duplicate = basic_fields + [duplicate_field]
        with pytest.raises(ValueError, match="Duplicate header fields detected: foo"):
            HeaderSpec(fields=fields_with_duplicate)

    def test_header_spec_exceed_size(self, basic_fields: list[HeaderField]) -> None:
        """Test exception when HeaderSpec size is smaller than requested offset.."""
        with pytest.raises(ValueError, match="Offsets exceed allowed header size"):
            HeaderSpec(fields=basic_fields, item_size=10)  # fields need 20

    @pytest.mark.parametrize("endianness", [Endianness.LITTLE, Endianness.BIG])
    def test_header_spec_endianness(
        self, basic_fields: list[HeaderField], endianness: Endianness
    ) -> None:
        """Test HeaderSpec endianness configuration."""
        header = HeaderSpec(fields=basic_fields, endianness=endianness)
        assert header.dtype.descr[0][1].startswith(endianness.symbol)

    def test_add_field(self, basic_header_spec: HeaderSpec) -> None:
        """Test adding field with its options."""
        new_field = HeaderField(name="buzz", format=ScalarType.INT8, byte=21)
        basic_header_spec.add_field(new_field)
        assert "buzz" in basic_header_spec.names
        assert basic_header_spec.offsets[-1] == 20

        # Test overwrite
        overwrite_field = HeaderField(name="foo", format=ScalarType.INT64, byte=1)
        basic_header_spec.add_field(overwrite_field, overwrite=True)
        assert basic_header_spec.formats[0] == np.dtype("int64")

        with pytest.raises(KeyError, match="Field named foo already exists"):
            basic_header_spec.add_field(overwrite_field)  # No overwrite

    def test_remove_field(self, basic_header_spec: HeaderSpec) -> None:
        """Test remove field from existing HeaderSpec."""
        basic_header_spec.remove_field("bar")
        assert "bar" not in basic_header_spec.names
        assert len(basic_header_spec.fields) == 2

        with pytest.raises(KeyError, match="Field named missing does not exist"):
            basic_header_spec.remove_field("missing")

    def test_customize(self, basic_header_spec: HeaderSpec) -> None:
        """Test HeaderSpec customization."""
        new_fields = [
            HeaderField(name="new1", format=ScalarType.INT32, byte=6),  # Overlaps bar
            HeaderField(name="foo", format=ScalarType.FLOAT32, byte=1),  # Replaces foo
        ]
        basic_header_spec.customize(new_fields)

        assert basic_header_spec.names == ["foo", "fizz", "new1"]  # no more bar
        assert basic_header_spec.formats[0] == np.dtype("float32")  # foo overwritten

        # Test invalid customize (duplicates in new_fields)
        duplicate_new = [HeaderField(name="dup", byte=25, format=ScalarType.INT32)] * 2
        with pytest.raises(ValueError, match="Duplicate header field names detected"):
            basic_header_spec.customize(duplicate_new)

        # Test overlapping in new_fields, should be invalid
        overlapping_new = [
            HeaderField(name="overlap1", byte=25, format=ScalarType.INT32),
            HeaderField(name="overlap2", byte=27, format=ScalarType.INT32),
        ]
        with pytest.raises(ValueError, match="Header fields overlap"):
            basic_header_spec.customize(overlapping_new)
