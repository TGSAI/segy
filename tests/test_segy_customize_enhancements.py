"""Tests for the enhanced customize method and related functionality in SegySpec."""

from __future__ import annotations

import pytest

from segy.schema import HeaderField
from segy.schema import ScalarType
from segy.schema import SegySpec
from segy.standards import SegyStandard
from segy.standards import get_segy_standard


def assert_fields_match(actual_fields: list[HeaderField], expected_fields: list[HeaderField], description: str) -> None:
    """Helper function to compare actual and expected HeaderField lists."""
    assert len(actual_fields) == len(expected_fields), (
        f"Failed for {description}: expected {len(expected_fields)} fields, got {len(actual_fields)}"
    )
    
    for i, (actual, expected) in enumerate(zip(actual_fields, expected_fields)):
        assert actual.name == expected.name, (
            f"Failed for {description}: field {i} name mismatch - expected '{expected.name}', got '{actual.name}'"
        )
        assert actual.format == expected.format, (
            f"Failed for {description}: field {i} format mismatch - expected {expected.format}, got {actual.format}"
        )
        assert actual.byte == expected.byte, (
            f"Failed for {description}: field {i} byte mismatch - expected {expected.byte}, got {actual.byte}"
        )


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

    @pytest.mark.parametrize("byte_pos", [1, 50, 100, 200, 400])
    def test_range_property_different_byte_positions(self, byte_pos: int) -> None:
        """Test range property with different starting byte positions."""
        field = HeaderField(name="test", format=ScalarType.INT16, byte=byte_pos)
        start, stop, name = field.range

        assert start == byte_pos  # noqa: PLR2004
        assert stop == byte_pos + 2  # noqa: PLR2004
        assert name == "test"


class TestOverlapMethod:
    """Tests for the _overlap method in SegySpec."""

    # Test data for overlap scenarios
    OVERLAP_TEST_CASES = [
        # (range1, range2, expected_overlap, description)
        ((1, 5), (10, 15), False, "separated ranges"),
        ((1, 5), (5, 10), False, "adjacent ranges"),
        ((1, 10), (5, 15), True, "partial overlap"),
        ((1, 10), (3, 8), True, "complete overlap"),
        ((5, 10), (5, 10), True, "identical ranges"),
        ((1, 6), (5, 10), True, "single byte overlap"),
        ((0, 4), (2, 6), True, "middle overlap"),
        ((10, 20), (15, 25), True, "end overlap"),
        ((5, 15), (0, 10), True, "start overlap"),
        ((100, 200), (150, 250), True, "large number overlap"),
        ((5, 6), (6, 7), False, "single byte adjacent"),
        # Edge cases
        ((5, 5), (5, 5), False, "zero width ranges"),
    ]

    @pytest.mark.parametrize(
        "range1,range2,expected_overlap,description", OVERLAP_TEST_CASES
    )
    def test_overlap_scenarios(
        self,
        segy_spec: SegySpec,
        range1: tuple[int, int],
        range2: tuple[int, int],
        expected_overlap: bool,
        description: str,
    ) -> None:
        """Test overlap detection for various range scenarios."""
        # Test both directions (range1 vs range2 and range2 vs range1)
        assert segy_spec._overlap(range1, range2) == expected_overlap, (
            f"Failed for {description}: {range1} vs {range2}"
        )
        assert segy_spec._overlap(range2, range1) == expected_overlap, (
            f"Failed for {description}: {range2} vs {range1}"
        )


class TestMergeHeadersByName:
    """Tests for the _merge_headers_by_name method."""

    # Test data for merge scenarios - covering the core logic without redundancy
    MERGE_TEST_CASES = [
        # (existing_fields, new_fields, expected_fields, description)
        (
            [],  # existing_fields
            [
                HeaderField(name="field1", format=ScalarType.INT32, byte=1),
                HeaderField(name="field2", format=ScalarType.INT16, byte=10),
            ],  # new_fields
            [
                HeaderField(name="field1", format=ScalarType.INT32, byte=1),
                HeaderField(name="field2", format=ScalarType.INT16, byte=10),
            ],  # expected_fields
            "empty existing fields"
        ),
        (
            [
                HeaderField(name="existing1", format=ScalarType.INT32, byte=10),
                HeaderField(name="existing2", format=ScalarType.INT16, byte=1),
            ],  # existing_fields
            [],  # new_fields
            [
                HeaderField(name="existing1", format=ScalarType.INT32, byte=10),
                HeaderField(name="existing2", format=ScalarType.INT16, byte=1),
            ],  # expected_fields
            "empty new fields"
        ),
        (
            [
                HeaderField(name="existing1", format=ScalarType.INT32, byte=1),
                HeaderField(name="existing2", format=ScalarType.INT16, byte=10),
            ],  # existing_fields
            [
                HeaderField(name="new1", format=ScalarType.UINT32, byte=20),
                HeaderField(name="new2", format=ScalarType.FLOAT32, byte=30),
            ],  # new_fields
            [
                HeaderField(name="new1", format=ScalarType.UINT32, byte=20),
                HeaderField(name="new2", format=ScalarType.FLOAT32, byte=30),
                HeaderField(name="existing1", format=ScalarType.INT32, byte=1),
                HeaderField(name="existing2", format=ScalarType.INT16, byte=10),
            ],  # expected_fields
            "new fields added without conflicts"
        ),
        (
            [
                HeaderField(name="field1", format=ScalarType.INT32, byte=1),
                HeaderField(name="field2", format=ScalarType.INT16, byte=10),
            ],  # existing_fields
            [
                HeaderField(name="field1", format=ScalarType.FLOAT32, byte=100),
                HeaderField(name="new_field", format=ScalarType.INT16, byte=200),
            ],  # new_fields
            [
                HeaderField(name="field1", format=ScalarType.FLOAT32, byte=100),
                HeaderField(name="new_field", format=ScalarType.INT16, byte=200),
                HeaderField(name="field2", format=ScalarType.INT16, byte=10),
            ],  # expected_fields
            "field replacement and addition"
        ),
    ]



    @pytest.mark.parametrize(
        "existing_fields,new_fields,expected_fields,description",
        MERGE_TEST_CASES
    )
    def test_merge_scenarios(
        self,
        segy_spec: SegySpec,
        existing_fields: list[HeaderField],
        new_fields: list[HeaderField],
        expected_fields: list[HeaderField],
        description: str,
    ) -> None:
        """Test merge scenarios with various field configurations."""
        result = segy_spec._merge_headers_by_name(existing_fields, new_fields)
        
        # Use helper function to compare all field properties
        assert_fields_match(result, expected_fields, description)


class TestValidateNonOverlappingHeaders:
    """Tests for the _validate_non_overlapping_headers method."""

    # Test data for validation scenarios
    VALIDATION_TEST_CASES = [
        # (fields, should_raise, expected_error)
        ([], False, None),
        ([HeaderField(name="field1", format=ScalarType.INT32, byte=1)], False, None),
        (
            [
                HeaderField(name="field1", format=ScalarType.INT32, byte=1),  # bytes 1-4
                HeaderField(name="field2", format=ScalarType.INT16, byte=10),  # bytes 10-11
                HeaderField(name="field3", format=ScalarType.UINT8, byte=20),  # byte 20
            ],
            False,
            None,
        ),
        (
            [
                HeaderField(name="field1", format=ScalarType.INT32, byte=1),  # bytes 1-4 (stops at 5)
                HeaderField(name="field2", format=ScalarType.INT16, byte=5),  # bytes 5-6 (starts at 5)
            ],
            False,
            None,
        ),
        (
            [
                HeaderField(name="field1", format=ScalarType.INT32, byte=1),
                HeaderField(name="field2", format=ScalarType.INT16, byte=10),
                HeaderField(name="field1", format=ScalarType.UINT8, byte=20),  # Duplicate name
            ],
            True,
            "Duplicate header field names detected",
        ),
        (
            [
                HeaderField(name="field1", format=ScalarType.INT32, byte=1),  # bytes 1-4
                HeaderField(name="field2", format=ScalarType.INT32, byte=3),  # bytes 3-6 (overlaps)
            ],
            True,
            "Header fields overlap",
        ),
    ]

    @pytest.mark.parametrize(
        "fields,should_raise,expected_error",
        VALIDATION_TEST_CASES
    )
    def test_validation_scenarios(
        self,
        segy_spec: SegySpec,
        fields: list[HeaderField],
        should_raise: bool,
        expected_error: str | None,
    ) -> None:
        """Test validation scenarios with various field configurations."""
        if should_raise:
            with pytest.raises(ValueError, match=expected_error):
                segy_spec._validate_non_overlapping_headers(fields)
        else:
            # Should not raise any exception
            segy_spec._validate_non_overlapping_headers(fields)


class TestMergeHeadersByByteOffset:
    """Tests for the _merge_headers_by_byte_offset method.
    
    This method is called AFTER _merge_headers_by_name and removes existing fields
    that overlap with the newly added fields. It simulates the real usage where
    existing_fields contains the merged state and new_fields are the original input.
    """

    # Test data for realistic byte offset merge scenarios
    BYTE_OFFSET_TEST_CASES = [
        # (merged_fields_after_name_merge, original_new_fields, expected_result_fields, description)
        (
            [
                # State after name merge: new field + non-conflicting existing fields
                HeaderField(name="new_field", format=ScalarType.UINT32, byte=1),  # bytes 1-4
                HeaderField(name="existing1", format=ScalarType.INT16, byte=10),  # bytes 10-11
                HeaderField(name="existing2", format=ScalarType.INT32, byte=12),  # bytes 12-15
            ],  # merged_fields_after_name_merge
            [
                HeaderField(name="new_field", format=ScalarType.UINT32, byte=1),  # bytes 1-4
            ],  # original_new_fields
            [
                HeaderField(name="new_field", format=ScalarType.UINT32, byte=1),
                HeaderField(name="existing1", format=ScalarType.INT16, byte=10),
                HeaderField(name="existing2", format=ScalarType.INT32, byte=12),
            ],  # expected_result_fields
            "no overlaps after name merge"
        ),
        (
            [
                # State after name merge: new field overlaps with next existing field
                HeaderField(name="new_field", format=ScalarType.UINT32, byte=1),  # bytes 1-4
                HeaderField(name="existing1", format=ScalarType.INT16, byte=3),  # bytes 3-4 (overlaps!)
                HeaderField(name="existing2", format=ScalarType.INT32, byte=10),  # bytes 10-13
            ],  # merged_fields_after_name_merge
            [
                HeaderField(name="new_field", format=ScalarType.UINT32, byte=1),
            ],  # original_new_fields
            [
                HeaderField(name="new_field", format=ScalarType.UINT32, byte=1),
                HeaderField(name="existing2", format=ScalarType.INT32, byte=10),
            ],  # expected_result_fields (existing1 removed due to overlap)
            "new field overlaps with existing field"
        ),
        (
            [
                # State after name merge: multiple overlaps
                HeaderField(name="new_field1", format=ScalarType.UINT32, byte=1),  # bytes 1-4
                HeaderField(name="existing1", format=ScalarType.INT16, byte=3),  # bytes 3-4 (overlaps with new_field1)
                HeaderField(name="new_field2", format=ScalarType.INT32, byte=5),  # bytes 5-8
                HeaderField(name="existing2", format=ScalarType.INT16, byte=7),  # bytes 7-8 (overlaps with new_field2)
                HeaderField(name="existing3", format=ScalarType.INT32, byte=20),  # bytes 20-23 (no overlap)
            ],  # merged_fields_after_name_merge
            [
                HeaderField(name="new_field1", format=ScalarType.UINT32, byte=1),
                HeaderField(name="new_field2", format=ScalarType.INT32, byte=5),
            ],  # original_new_fields
            [
                HeaderField(name="new_field1", format=ScalarType.UINT32, byte=1),
                HeaderField(name="new_field2", format=ScalarType.INT32, byte=5),
                HeaderField(name="existing3", format=ScalarType.INT32, byte=20),
            ],  # expected_result_fields (existing1 and existing2 removed)
            "multiple new fields with overlaps"
        ),
        (
            [
                HeaderField(name="existing1", format=ScalarType.INT32, byte=1),
                HeaderField(name="existing2", format=ScalarType.INT16, byte=10),
            ],  # merged_fields_after_name_merge
            [],  # original_new_fields (empty)
            [
                HeaderField(name="existing1", format=ScalarType.INT32, byte=1),
                HeaderField(name="existing2", format=ScalarType.INT16, byte=10),
            ],  # expected_result_fields (no changes)
            "no new fields"
        ),
    ]



    @pytest.mark.parametrize(
        "merged_fields_after_name_merge,original_new_fields,expected_result_fields,description",
        BYTE_OFFSET_TEST_CASES
    )
    def test_merge_by_byte_offset_scenarios(
        self,
        segy_spec: SegySpec,
        merged_fields_after_name_merge: list[HeaderField],
        original_new_fields: list[HeaderField],
        expected_result_fields: list[HeaderField],
        description: str,
    ) -> None:
        """Test merge by byte offset scenarios that reflect real usage patterns."""
        result = segy_spec._merge_headers_by_byte_offset(
            merged_fields_after_name_merge.copy(), original_new_fields
        )
        
        # Use helper function to compare all field properties
        assert_fields_match(result, expected_result_fields, description)


class TestCustomizeMethodEnhanced:
    """Tests for the enhanced customize method with validation and merging."""

    # Test data for customize method scenarios
    CUSTOMIZE_TEST_CASES = [
        # (header_type, custom_fields, should_raise, expected_error, description)
        (
            "binary",
            [
                HeaderField(name="custom_field1", format=ScalarType.UINT32, byte=17),
                HeaderField(name="custom_field2", format=ScalarType.INT16, byte=300),
            ],
            False,
            None,
            "valid binary header fields"
        ),
        (
            "trace",
            [
                HeaderField(name="custom_trace1", format=ScalarType.FLOAT32, byte=100),
                HeaderField(name="custom_trace2", format=ScalarType.INT32, byte=200),
            ],
            False,
            None,
            "valid trace header fields"
        ),
        (
            "binary",
            [
                HeaderField(name="duplicate", format=ScalarType.UINT32, byte=17),
                HeaderField(name="duplicate", format=ScalarType.INT16, byte=300),  # Duplicate name
            ],
            True,
            "Duplicate header field names detected",
            "binary headers with duplicate names"
        ),
        (
            "trace",
            [
                HeaderField(name="duplicate", format=ScalarType.FLOAT32, byte=100),
                HeaderField(name="duplicate", format=ScalarType.INT32, byte=200),  # Duplicate name
            ],
            True,
            "Duplicate header field names detected",
            "trace headers with duplicate names"
        ),
        (
            "binary",
            [
                HeaderField(name="field1", format=ScalarType.INT32, byte=17),  # bytes 17-20
                HeaderField(name="field2", format=ScalarType.INT32, byte=19),  # bytes 19-22 (overlaps)
            ],
            True,
            "Header fields overlap",
            "binary headers with overlapping fields"
        ),
        (
            "trace",
            [
                HeaderField(name="field1", format=ScalarType.FLOAT32, byte=100),  # bytes 100-103
                HeaderField(name="field2", format=ScalarType.INT16, byte=102),  # bytes 102-103 (overlaps)
            ],
            True,
            "Header fields overlap",
            "trace headers with overlapping fields"
        ),
    ]

    @pytest.mark.parametrize(
        "header_type,custom_fields,should_raise,expected_error,description",
        CUSTOMIZE_TEST_CASES
    )
    def test_customize_scenarios(
        self,
        segy_spec: SegySpec,
        header_type: str,
        custom_fields: list[HeaderField],
        should_raise: bool,
        expected_error: str | None,
        description: str,
    ) -> None:
        """Test customize method scenarios for both binary and trace headers."""
        # Prepare the kwargs based on header type
        kwargs = {}
        if header_type == "binary":
            kwargs["binary_header_fields"] = custom_fields
        elif header_type == "trace":
            kwargs["trace_header_fields"] = custom_fields

        if should_raise:
            with pytest.raises(ValueError, match=expected_error):
                segy_spec.customize(**kwargs)
        else:
            # Should not raise any exception
            custom_spec = segy_spec.customize(**kwargs)

            assert custom_spec.segy_standard is None
            
            # Check that custom fields are present
            if header_type == "binary":
                assert len(custom_spec.binary_header.fields) >= len(custom_fields)
                field_names = {field.name for field in custom_spec.binary_header.fields}
            elif header_type == "trace":
                assert len(custom_spec.trace.header.fields) >= len(custom_fields)
                field_names = {field.name for field in custom_spec.trace.header.fields}
            
            for field in custom_fields:
                assert field.name in field_names, f"Custom field '{field.name}' not found in {header_type} header"

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
