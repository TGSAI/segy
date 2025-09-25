"""Shared fixtures for tests."""

from __future__ import annotations

import pytest
from fsspec.implementations.memory import MemoryFileSystem

from segy.factory import get_default_text
from segy.schema import HeaderField
from segy.schema import HeaderSpec
from segy.schema import ScalarType
from segy.standards import get_segy_standard


@pytest.fixture(scope="session")
def mock_filesystem() -> MemoryFileSystem:
    """Return a mocked filesystem implementation in memory."""
    return MemoryFileSystem()


@pytest.fixture(scope="session")
def default_text() -> str:
    """Get the default text header to be used in comparisons downstream."""
    spec = get_segy_standard(1.0)
    spec.trace.data.samples = 101
    spec.trace.data.interval = 4000

    return get_default_text(spec)


@pytest.fixture
def basic_fields() -> list[HeaderField]:
    """Simple header fields to be used in HeaderSpec and HeaderField tests."""
    return [
        HeaderField(name="foo", format=ScalarType.INT32, byte=1),
        HeaderField(name="bar", format=ScalarType.INT16, byte=5),
        HeaderField(name="fizz", format=ScalarType.INT32, byte=17),
    ]


@pytest.fixture
def basic_header_spec(basic_fields: list[HeaderField]) -> HeaderSpec:
    """Mock HeaderSpec with basic fields used in HeaderSpec and HeaderField tests."""
    return HeaderSpec(fields=basic_fields)
