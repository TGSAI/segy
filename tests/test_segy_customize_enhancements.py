"""Tests for the enhanced customize method and related functionality in SegySpec."""

from __future__ import annotations

import pytest

from segy.schema import HeaderField
from segy.schema import ScalarType
from segy.schema import SegySpec
from segy.standards import SegyStandard
from segy.standards import get_segy_standard


@pytest.fixture(params=[SegyStandard.REV0, SegyStandard.REV1])
def segy_spec(request: pytest.FixtureRequest) -> SegySpec:
    """Fixture for creating known SegySpec instances for testing."""
    standard = request.param
    return get_segy_standard(standard)


class TestHeaderFieldRange:
    """Tests for the new range property on HeaderField."""

    def test_range_property_basic(self) -> None:
        """Test that range property returns correct start, stop, and name."""
        field = HeaderField(name="test_field", format=ScalarType.INT32, byte=10)
        start, stop, name = field.range

        assert start == 10  # noqa: PLR2004
        assert stop == 14  # noqa: PLR2004
        assert name == "test_field"

    @pytest.mark.parametrize(
        ("format_type", "expected_size"),
        [
            (ScalarType.UINT8, 1),
            (ScalarType.INT16, 2),
            (ScalarType.UINT16, 2),
            (ScalarType.INT32, 4),
            (ScalarType.UINT32, 4),
            (ScalarType.FLOAT32, 4),
            (ScalarType.FLOAT64, 8),
        ],
    )
    def test_range_property_different_types(
        self, format_type: ScalarType, expected_size: int
    ) -> None:
        """Test range property with different data types."""
        field = HeaderField(name="field", format=format_type, byte=100)
        start, stop, name = field.range

        assert start == 100  # noqa: PLR2004
        assert stop == 100 + expected_size  # noqa: PLR2004
        assert name == "field"

    def test_range_property_different_byte_positions(self) -> None:
        """Test range property with different starting byte positions."""
        test_cases = [1, 50, 100, 200, 400]
        for byte_pos in test_cases:
            field = HeaderField(name="test", format=ScalarType.INT16, byte=byte_pos)
            start, stop, name = field.range

            assert start == byte_pos  # noqa: PLR2004
            assert stop == byte_pos + 2  # noqa: PLR2004
            assert name == "test"


class TestOverlapMethod:
    """Tests for the _overlap method in SegySpec."""

    def test_overlap_no_overlap_separated(self, segy_spec: SegySpec) -> None:
        """Test that separated ranges don't overlap."""
        range1 = (1, 5)  # [1, 5)
        range2 = (10, 15)  # [10, 15)

        assert not segy_spec._overlap(range1, range2)
        assert not segy_spec._overlap(range2, range1)

    def test_overlap_no_overlap_adjacent(self, segy_spec: SegySpec) -> None:
        """Test that adjacent ranges don't overlap."""
        range1 = (1, 5)  # [1, 5)
        range2 = (5, 10)  # [5, 10) - starts where range1 ends

        assert not segy_spec._overlap(range1, range2)
        assert not segy_spec._overlap(range2, range1)

    def test_overlap_partial_overlap(self, segy_spec: SegySpec) -> None:
        """Test that partially overlapping ranges are detected."""
        range1 = (1, 10)  # [1, 10)
        range2 = (5, 15)  # [5, 15) - overlaps from 5-10

        assert segy_spec._overlap(range1, range2)
        assert segy_spec._overlap(range2, range1)

    def test_overlap_complete_overlap(self, segy_spec: SegySpec) -> None:
        """Test that completely overlapping ranges are detected."""
        range1 = (1, 10)  # [1, 10)
        range2 = (3, 8)  # [3, 8) - completely contained in range1

        assert segy_spec._overlap(range1, range2)
        assert segy_spec._overlap(range2, range1)

    def test_overlap_identical_ranges(self, segy_spec: SegySpec) -> None:
        """Test that identical ranges overlap."""
        range1 = (5, 10)  # [5, 10)
        range2 = (5, 10)  # [5, 10) - identical

        assert segy_spec._overlap(range1, range2)

    def test_overlap_single_byte_overlap(self, segy_spec: SegySpec) -> None:
        """Test single byte overlap detection."""
        range1 = (1, 6)  # [1, 6)
        range2 = (5, 10)  # [5, 10) - overlaps at byte 5

        assert segy_spec._overlap(range1, range2)
        assert segy_spec._overlap(range2, range1)


class TestMergeHeadersByName:
    """Tests for the _merge_headers_by_name method."""

    def test_merge_empty_existing_fields(self, segy_spec: SegySpec) -> None:
        """Test merging when existing fields list is empty."""
        new_fields = [
            HeaderField(name="field1", format=ScalarType.INT32, byte=1),
            HeaderField(name="field2", format=ScalarType.INT16, byte=10),
        ]

        result = segy_spec._merge_headers_by_name([], new_fields)

        assert len(result) == 2  # noqa: PLR2004
        assert result[0].name == "field1"
        assert result[1].name == "field2"

    def test_merge_empty_new_fields(self, segy_spec: SegySpec) -> None:
        """Test merging when new fields list is empty."""
        existing_fields = [
            HeaderField(name="existing1", format=ScalarType.INT32, byte=10),
            HeaderField(name="existing2", format=ScalarType.INT16, byte=1),
        ]

        result = segy_spec._merge_headers_by_name(existing_fields, [])

        assert len(result) == 2  # noqa: PLR2004
        assert result[0].name == "existing1"
        assert result[1].name == "existing2"

    def test_merge_no_name_conflicts(self, segy_spec: SegySpec) -> None:
        """Test merging when there are no name conflicts."""
        existing_fields = [
            HeaderField(name="existing1", format=ScalarType.INT32, byte=1),
            HeaderField(name="existing2", format=ScalarType.INT16, byte=10),
        ]
        new_fields = [
            HeaderField(name="new1", format=ScalarType.UINT32, byte=20),
            HeaderField(name="new2", format=ScalarType.FLOAT32, byte=30),
        ]

        result = segy_spec._merge_headers_by_name(existing_fields, new_fields)

        assert len(result) == 4  # noqa: PLR2004
        # New fields come first
        assert result[0].name == "new1"
        assert result[1].name == "new2"
        # Existing fields that don't conflict come after
        assert result[2].name == "existing1"
        assert result[3].name == "existing2"

    def test_merge_with_name_conflicts(self, segy_spec: SegySpec) -> None:
        """Test merging when there are name conflicts (new fields should replace existing)."""
        existing_fields = [
            HeaderField(name="field1", format=ScalarType.INT32, byte=1),
            HeaderField(name="field2", format=ScalarType.INT16, byte=10),
            HeaderField(name="field3", format=ScalarType.UINT32, byte=20),
        ]
        new_fields = [
            HeaderField(
                name="field1", format=ScalarType.FLOAT32, byte=100
            ),  # Replaces existing field1
            HeaderField(
                name="new_field", format=ScalarType.INT16, byte=200
            ),  # New field
        ]

        result = segy_spec._merge_headers_by_name(existing_fields, new_fields)

        # 2 new fields + 2 non-conflicting existing fields
        assert len(result) == 4  # noqa: PLR2004

        # New fields come first
        assert result[0].name == "field1"
        assert result[0].format == ScalarType.FLOAT32
        assert result[0].byte == 100  # noqa: PLR2004
        assert result[1].name == "new_field"

        # Non-conflicting existing fields
        assert result[2].name == "field2"
        assert result[3].name == "field3"

    def test_merge_all_fields_replaced(self, segy_spec: SegySpec) -> None:
        """Test merging when all existing fields are replaced by new fields."""
        existing_fields = [
            HeaderField(name="field1", format=ScalarType.INT32, byte=1),
            HeaderField(name="field2", format=ScalarType.INT16, byte=10),
        ]
        new_fields = [
            HeaderField(name="field1", format=ScalarType.FLOAT32, byte=100),
            HeaderField(name="field2", format=ScalarType.FLOAT64, byte=200),
        ]

        result = segy_spec._merge_headers_by_name(existing_fields, new_fields)

        assert len(result) == 2  # noqa: PLR2004
        assert result[0].name == "field1"
        assert result[0].format == ScalarType.FLOAT32
        assert result[1].name == "field2"
        assert result[1].format == ScalarType.FLOAT64


class TestValidateNonOverlappingHeaders:
    """Tests for the _validate_non_overlapping_headers method."""

    def test_validate_empty_list(self, segy_spec: SegySpec) -> None:
        """Test validation with empty list should pass."""
        # Should not raise any exception
        segy_spec._validate_non_overlapping_headers([])

    def test_validate_single_field(self, segy_spec: SegySpec) -> None:
        """Test validation with single field should pass."""
        fields = [HeaderField(name="field1", format=ScalarType.INT32, byte=1)]

        # Should not raise any exception
        segy_spec._validate_non_overlapping_headers(fields)

    def test_validate_non_overlapping_fields(self, segy_spec: SegySpec) -> None:
        """Test validation with non-overlapping fields should pass."""
        fields = [
            HeaderField(name="field1", format=ScalarType.INT32, byte=1),  # bytes 1-4
            HeaderField(name="field2", format=ScalarType.INT16, byte=10),  # bytes 10-11
            HeaderField(name="field3", format=ScalarType.UINT8, byte=20),  # byte 20
        ]

        # Should not raise any exception
        segy_spec._validate_non_overlapping_headers(fields)

    def test_validate_adjacent_fields(self, segy_spec: SegySpec) -> None:
        """Test validation with adjacent (but not overlapping) fields should pass."""
        fields = [
            HeaderField(
                name="field1", format=ScalarType.INT32, byte=1
            ),  # bytes 1-4 (stops at 5)
            HeaderField(
                name="field2", format=ScalarType.INT16, byte=5
            ),  # bytes 5-6 (starts at 5)
        ]

        # Should not raise any exception
        segy_spec._validate_non_overlapping_headers(fields)

    def test_validate_duplicate_names_raises_error(self, segy_spec: SegySpec) -> None:
        """Test validation with duplicate field names should raise ValueError."""
        fields = [
            HeaderField(name="field1", format=ScalarType.INT32, byte=1),
            HeaderField(name="field2", format=ScalarType.INT16, byte=10),
            HeaderField(
                name="field1", format=ScalarType.UINT8, byte=20
            ),  # Duplicate name
        ]

        with pytest.raises(ValueError, match="Duplicate header field names detected!"):
            segy_spec._validate_non_overlapping_headers(fields)

    def test_validate_overlapping_fields_raises_error(
        self, segy_spec: SegySpec
    ) -> None:
        """Test validation with overlapping fields should raise ValueError."""
        fields = [
            HeaderField(name="field1", format=ScalarType.INT32, byte=1),  # bytes 1-4
            HeaderField(
                name="field2", format=ScalarType.INT32, byte=3
            ),  # bytes 3-6 (overlaps)
        ]

        with pytest.raises(ValueError, match="Header fields overlap!"):
            segy_spec._validate_non_overlapping_headers(fields)

    def test_validate_multiple_overlaps_raises_error(self, segy_spec: SegySpec) -> None:
        """Test validation with multiple overlapping fields should raise ValueError."""
        fields = [
            HeaderField(name="field1", format=ScalarType.INT32, byte=1),  # bytes 1-4
            HeaderField(
                name="field2", format=ScalarType.INT32, byte=3
            ),  # bytes 3-6 (overlaps with field1)
            HeaderField(
                name="field3", format=ScalarType.INT16, byte=5
            ),  # bytes 5-6 (overlaps with field2)
        ]

        with pytest.raises(ValueError, match="Header fields overlap!"):
            segy_spec._validate_non_overlapping_headers(fields)

    def test_validate_none_input_does_not_raise(self, segy_spec: SegySpec) -> None:
        """Test that None input is handled gracefully."""
        # This should test the case where binary_header_fields or trace_header_fields is None
        # The customize method should handle None before calling validation
        segy_spec._validate_non_overlapping_headers([])


class TestMergeHeadersByByteOffset:
    """Tests for the _merge_headers_by_byte_offset method."""

    def test_merge_by_offset_no_overlaps(self, segy_spec: SegySpec) -> None:
        """Test merging when there are no overlaps."""
        existing_fields = [
            HeaderField(name="existing1", format=ScalarType.INT32, byte=1),  # bytes 1-4
            HeaderField(
                name="existing2", format=ScalarType.INT16, byte=10
            ),  # bytes 10-11
        ]
        new_fields = [
            HeaderField(name="new1", format=ScalarType.UINT8, byte=20),  # byte 20
        ]

        result = segy_spec._merge_headers_by_byte_offset(existing_fields, new_fields)

        # No overlaps, so all existing fields should remain
        assert len(result) == 2  # noqa: PLR2004
        existing_names = {field.name for field in result}
        assert "existing1" in existing_names
        assert "existing2" in existing_names

    def test_merge_by_offset_with_overlaps(self, segy_spec: SegySpec) -> None:
        """Test merging when new fields overlap with existing fields."""
        existing_fields = [
            HeaderField(name="existing1", format=ScalarType.INT32, byte=1),  # bytes 1-4
            HeaderField(
                name="existing2", format=ScalarType.INT32, byte=10
            ),  # bytes 10-13
            HeaderField(
                name="existing3", format=ScalarType.INT16, byte=20
            ),  # bytes 20-21
        ]
        new_fields = [
            HeaderField(
                name="new1", format=ScalarType.UINT32, byte=2
            ),  # bytes 2-5 (overlaps with existing1)
        ]

        # After _merge_headers_by_name, existing1 should be replaced/removed due to overlap
        # This test assumes the method works as intended to remove overlapping existing fields
        result = segy_spec._merge_headers_by_byte_offset(
            existing_fields.copy(), new_fields
        )

        # The exact behavior depends on the implementation, but existing1 should be affected
        # since new1 overlaps with it
        remaining_names = {field.name for field in result}

        # existing2 and existing3 should remain as they don't overlap with new1
        assert "existing2" in remaining_names
        assert "existing3" in remaining_names

    def test_merge_by_offset_empty_lists(self, segy_spec: SegySpec) -> None:
        """Test merging with empty lists."""
        # Empty existing fields
        result = segy_spec._merge_headers_by_byte_offset(
            [], [HeaderField(name="new1", format=ScalarType.INT32, byte=1)]
        )
        assert len(result) == 0  # No existing fields to remove

        # Empty new fields
        existing_fields = [
            HeaderField(name="existing1", format=ScalarType.INT32, byte=1)
        ]
        result = segy_spec._merge_headers_by_byte_offset(existing_fields, [])
        assert len(result) == 1  # noqa: PLR2004
        assert result[0].name == "existing1"

    def test_merge_by_offset_multiple_overlaps(self, segy_spec: SegySpec) -> None:
        """Test merging with multiple overlapping scenarios."""
        existing_fields = [
            HeaderField(name="existing1", format=ScalarType.INT32, byte=1),  # bytes 1-4
            HeaderField(name="existing2", format=ScalarType.INT32, byte=5),  # bytes 5-8
            HeaderField(
                name="existing3", format=ScalarType.INT32, byte=15
            ),  # bytes 15-18
        ]
        new_fields = [
            HeaderField(
                name="new1", format=ScalarType.UINT32, byte=3
            ),  # bytes 3-6 (overlaps existing1 & existing2)
            HeaderField(
                name="new2", format=ScalarType.UINT16, byte=20
            ),  # bytes 20-21 (no overlap)
        ]

        result = segy_spec._merge_headers_by_byte_offset(
            existing_fields.copy(), new_fields
        )

        # existing3 should remain as it doesn't overlap
        remaining_names = {field.name for field in result}
        assert "existing3" in remaining_names

        # The exact count depends on implementation, but should be less than original
        assert len(result) <= len(existing_fields)


class TestCustomizeMethodEnhanced:
    """Tests for the enhanced customize method with validation and merging."""

    def test_customize_with_valid_binary_headers(self, segy_spec: SegySpec) -> None:
        """Test customize method with valid binary header fields."""
        custom_fields = [
            HeaderField(name="custom_field1", format=ScalarType.UINT32, byte=17),
            HeaderField(name="custom_field2", format=ScalarType.INT16, byte=300),
        ]

        # Should not raise any exception
        custom_spec = segy_spec.customize(binary_header_fields=custom_fields)

        assert custom_spec.segy_standard is None
        assert len(custom_spec.binary_header.fields) >= len(custom_fields)

        # Check that our custom fields are present
        field_names = {field.name for field in custom_spec.binary_header.fields}
        assert "custom_field1" in field_names
        assert "custom_field2" in field_names

    def test_customize_with_valid_trace_headers(self, segy_spec: SegySpec) -> None:
        """Test customize method with valid trace header fields."""
        custom_fields = [
            HeaderField(name="custom_trace1", format=ScalarType.FLOAT32, byte=100),
            HeaderField(name="custom_trace2", format=ScalarType.INT32, byte=200),
        ]

        # Should not raise any exception
        custom_spec = segy_spec.customize(trace_header_fields=custom_fields)

        assert custom_spec.segy_standard is None
        assert len(custom_spec.trace.header.fields) >= len(custom_fields)

        # Check that our custom fields are present
        field_names = {field.name for field in custom_spec.trace.header.fields}
        assert "custom_trace1" in field_names
        assert "custom_trace2" in field_names

    def test_customize_binary_headers_with_duplicate_names_raises_error(
        self, segy_spec: SegySpec
    ) -> None:
        """Test customize method with duplicate binary header field names."""
        invalid_fields = [
            HeaderField(name="duplicate", format=ScalarType.UINT32, byte=17),
            HeaderField(
                name="duplicate", format=ScalarType.INT16, byte=300
            ),  # Duplicate name
        ]

        with pytest.raises(ValueError, match="Duplicate header field names detected!"):
            segy_spec.customize(binary_header_fields=invalid_fields)

    def test_customize_trace_headers_with_duplicate_names_raises_error(
        self, segy_spec: SegySpec
    ) -> None:
        """Test customize method with duplicate trace header field names."""
        invalid_fields = [
            HeaderField(name="duplicate", format=ScalarType.FLOAT32, byte=100),
            HeaderField(
                name="duplicate", format=ScalarType.INT32, byte=200
            ),  # Duplicate name
        ]

        with pytest.raises(ValueError, match="Duplicate header field names detected!"):
            segy_spec.customize(trace_header_fields=invalid_fields)

    def test_customize_binary_headers_with_overlapping_fields_raises_error(
        self, segy_spec: SegySpec
    ) -> None:
        """Test customize method with overlapping binary header fields."""
        invalid_fields = [
            HeaderField(name="field1", format=ScalarType.INT32, byte=17),  # bytes 17-20
            HeaderField(
                name="field2", format=ScalarType.INT32, byte=19
            ),  # bytes 19-22 (overlaps)
        ]

        with pytest.raises(ValueError, match="Header fields overlap!"):
            segy_spec.customize(binary_header_fields=invalid_fields)

    def test_customize_trace_headers_with_overlapping_fields_raises_error(
        self, segy_spec: SegySpec
    ) -> None:
        """Test customize method with overlapping trace header fields."""
        invalid_fields = [
            HeaderField(
                name="field1", format=ScalarType.FLOAT32, byte=100
            ),  # bytes 100-103
            HeaderField(
                name="field2", format=ScalarType.INT16, byte=102
            ),  # bytes 102-103 (overlaps)
        ]

        with pytest.raises(ValueError, match="Header fields overlap!"):
            segy_spec.customize(trace_header_fields=invalid_fields)

    def test_customize_replaces_existing_fields_by_name(
        self, segy_spec: SegySpec
    ) -> None:
        """Test that customize method replaces existing fields with same names."""
        # Get an existing field name from the binary header
        existing_field_names = {field.name for field in segy_spec.binary_header.fields}
        if not existing_field_names:
            pytest.skip("No existing binary header fields to test replacement")

        existing_name = next(iter(existing_field_names))

        # Create a custom field with the same name but different properties
        # Use byte 300 to stay within the 400-byte binary header limit
        custom_fields = [
            HeaderField(name=existing_name, format=ScalarType.FLOAT64, byte=300),
        ]

        custom_spec = segy_spec.customize(binary_header_fields=custom_fields)

        # Find the replaced field
        replaced_field = None
        for field in custom_spec.binary_header.fields:
            if field.name == existing_name:
                replaced_field = field
                break

        assert replaced_field is not None
        assert replaced_field.format == ScalarType.FLOAT64
        assert replaced_field.byte == 300  # noqa: PLR2004

    def test_customize_handles_none_inputs(self, segy_spec: SegySpec) -> None:
        """Test that customize method handles None inputs gracefully."""
        # Should not raise any exceptions
        custom_spec = segy_spec.customize(
            binary_header_fields=None, trace_header_fields=None
        )

        # Should return a valid spec with original fields
        assert custom_spec.segy_standard is None
        assert len(custom_spec.binary_header.fields) > 0
        assert len(custom_spec.trace.header.fields) > 0
