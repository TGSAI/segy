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

    @pytest.mark.parametrize(
        "field_name,format_type,byte_pos,expected_size",
        [
            # Basic test case
            ("test_field", ScalarType.INT32, 10, 4),
            # Different types
            ("field", ScalarType.UINT8, 100, 1),
            ("field", ScalarType.INT16, 100, 2),
            ("field", ScalarType.UINT16, 100, 2),
            ("field", ScalarType.INT32, 100, 4),
            ("field", ScalarType.UINT32, 100, 4),
            ("field", ScalarType.FLOAT32, 100, 4),
            ("field", ScalarType.FLOAT64, 100, 8),
            # Different byte positions
            ("test", ScalarType.INT16, 1, 2),
            ("test", ScalarType.INT16, 50, 2),
            ("test", ScalarType.INT16, 200, 2),
            ("test", ScalarType.INT16, 400, 2),
        ],
    )
    def test_range_property(
        self, 
        field_name: str, 
        format_type: ScalarType, 
        byte_pos: int, 
        expected_size: int
    ) -> None:
        """Test range property with different field configurations."""
        field = HeaderField(name=field_name, format=format_type, byte=byte_pos)
        start, stop, name = field.range

        assert start == byte_pos
        assert stop == byte_pos + expected_size
        assert name == field_name


class TestHeaderFieldOperations:
    """Consolidated tests for header field operations (merge, validation, overlap)."""

    # Combined test data for various field operations
    FIELD_OPERATION_TEST_CASES = [
        # (operation_type, test_data, expected_result, should_raise, expected_error, description)
        
        # Merge tests
        ("merge", ([], [HeaderField(name="field1", format=ScalarType.INT32, byte=1)]), 
         [HeaderField(name="field1", format=ScalarType.INT32, byte=1)], False, None, "merge with empty existing fields"),
        
        ("merge", ([HeaderField(name="existing1", format=ScalarType.INT32, byte=10)], []), 
         [HeaderField(name="existing1", format=ScalarType.INT32, byte=10)], False, None, "merge with empty new fields"),
        
        ("merge", (
            [HeaderField(name="existing1", format=ScalarType.INT32, byte=1)],
            [HeaderField(name="existing1", format=ScalarType.FLOAT32, byte=100)]
         ), [HeaderField(name="existing1", format=ScalarType.FLOAT32, byte=100)], False, None, "merge with field replacement"),
        
        # Validation tests
        ("validate", [], None, False, None, "validate empty list"),
        ("validate", [HeaderField(name="field1", format=ScalarType.INT32, byte=1)], None, False, None, "validate single field"),
        ("validate", [
            HeaderField(name="field1", format=ScalarType.INT32, byte=1),
            HeaderField(name="field2", format=ScalarType.INT16, byte=10),
         ], None, False, None, "validate non-overlapping fields"),
        ("validate", [
            HeaderField(name="field1", format=ScalarType.INT32, byte=1),
            HeaderField(name="field1", format=ScalarType.UINT8, byte=20),  # Duplicate name
         ], None, True, "Duplicate header field names detected", "validate duplicate names"),
        ("validate", [
            HeaderField(name="field1", format=ScalarType.INT32, byte=1),  # bytes 1-4
            HeaderField(name="field2", format=ScalarType.INT32, byte=3),  # bytes 3-6 (overlaps)
         ], None, True, "Header fields overlap", "validate overlapping fields"),
        
        # Overlap tests
        ("overlap", ((1, 5), (10, 15)), False, False, None, "non-overlapping ranges"),
        ("overlap", ((1, 5), (5, 10)), False, False, None, "adjacent ranges"),
        ("overlap", ((1, 10), (5, 15)), True, False, None, "overlapping ranges"),
        ("overlap", ((5, 10), (5, 10)), True, False, None, "identical ranges"),
    ]

    @pytest.mark.parametrize(
        "operation_type,test_data,expected_result,should_raise,expected_error,description",
        FIELD_OPERATION_TEST_CASES
    )
    def test_field_operations(
        self,
        segy_spec: SegySpec,
        operation_type: str,
        test_data: tuple | list,
        expected_result: list[HeaderField] | bool | None,
        should_raise: bool,
        expected_error: str | None,
        description: str,
    ) -> None:
        """Test various header field operations with unified logic."""
        if operation_type == "merge":
            existing_fields, new_fields = test_data
            if should_raise:
                with pytest.raises(ValueError, match=expected_error):
                    segy_spec._merge_headers_by_name(existing_fields, new_fields)
            else:
                result = segy_spec._merge_headers_by_name(existing_fields, new_fields)
                assert_fields_match(result, expected_result, description)
        
        elif operation_type == "validate":
            fields = test_data
            if should_raise:
                with pytest.raises(ValueError, match=expected_error):
                    segy_spec._validate_non_overlapping_headers(fields)
            else:
                # Should not raise any exception
                segy_spec._validate_non_overlapping_headers(fields)

        elif operation_type == "overlap":
            range1, range2 = test_data
            result = segy_spec._overlap(range1, range2)
            assert result == expected_result, f"Failed for {description}: {range1} vs {range2}"
            # Test commutative property
            assert segy_spec._overlap(range2, range1) == expected_result, f"Failed for {description}: {range2} vs {range1}"



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
