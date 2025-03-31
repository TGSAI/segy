"""Tests for the CLI."""

from __future__ import annotations

import os

import pytest
from typer.testing import CliRunner

from segy.cli.segy import app

runner = CliRunner()


@pytest.fixture
def s3_path() -> str:
    """Fixture for Stratton dataset on S3 (SEG Wiki)."""
    return "s3://open.source.geoscience/open_data/stratton/segy/navmerged/swath_1_geometry.sgy"


class TestDump:
    """Test class for CLI's dump options."""

    @classmethod
    def setup_class(cls: type[TestDump]) -> None:
        """Set environment variable for anon access to S3."""
        os.environ["SEGY__STORAGE_OPTIONS"] = '{"anon": true}'

    def test_info_dump(self, s3_path: str) -> None:
        """Test generic info dump."""
        result = runner.invoke(app, ["dump", "info", s3_path])
        assert result.exit_code == 0
        assert "numTraces" in result.stdout
        assert "fileSize" in result.stdout

    def test_text_dump(self, s3_path: str) -> None:
        """Test text header dump."""
        result = runner.invoke(app, ["dump", "text-header", s3_path])
        assert result.exit_code == 0
        assert "CLIENT: BUREAU OF ECONOMIC GEOLOGY" in result.stdout

    def test_binary_header_dump(self, s3_path: str) -> None:
        """Test binary header dump."""
        result = runner.invoke(app, ["dump", "binary-header", s3_path])
        assert result.exit_code == 0
        assert "sample_interval" in result.stdout
        assert "samples_per_trace" in result.stdout

    def test_trace_header_dump(self, s3_path: str) -> None:
        """Test trace header dump."""
        args = ["dump", "trace-header", s3_path]
        args += ["--index", "100", "--index", "101"]
        args += ["--field", "source_coord_x"]
        args += ["--field", "coordinate_scalar"]

        result = runner.invoke(app, args)
        assert result.exit_code == 0
        assert "source_coord_x" in result.stdout
        assert "coordinate_scalar" in result.stdout
        assert "101" in result.stdout
        assert "70628086" in result.stdout
        assert "-100" in result.stdout

    def test_trace_data_dump(self, s3_path: str) -> None:
        """Test trace data dump."""
        args = ["dump", "trace-data", s3_path]
        args += ["--index", "501", "--index", "1000"]

        result = runner.invoke(app, args)
        assert result.exit_code == 0
        assert "-5.3372304e-08" in result.stdout
        assert "4.2979627e-07" in result.stdout
