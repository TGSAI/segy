"""Shared fixtures for tests."""

from __future__ import annotations

import pytest
from fsspec.implementations.memory import MemoryFileSystem

from segy.factory import get_default_text
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
