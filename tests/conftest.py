"""Shared fixtures for tests."""

from __future__ import annotations

import pytest
from fsspec.implementations.memory import MemoryFileSystem


@pytest.fixture()
def mock_filesystem() -> MemoryFileSystem:
    """Return a mocked filesystem implementation in memory."""
    return MemoryFileSystem()
