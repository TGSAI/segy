"""Tests for the CLI."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from segy.cli.segy import app

runner = CliRunner()


@pytest.fixture
def s3_path() -> str:
    """Fixture for the Soda Lake field record (public HTTPS)."""
    return (
        "https://gdr-data-lake.s3.us-west-2.amazonaws.com/"
        "soda_lake/raw_seismic/2010/v1.0.0/F7733R1.SGY"
    )


class TestDump:
    """Test class for CLI's dump options."""

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
        assert "END EBCDIC" in result.stdout

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
        assert "7735193" in result.stdout
        assert "-1" in result.stdout

    def test_trace_data_dump(self, s3_path: str) -> None:
        """Test trace data dump."""
        args = ["dump", "trace-data", s3_path]
        args += ["--index", "1", "--index", "100"]

        result = runner.invoke(app, args)
        assert result.exit_code == 0
        assert "5.4378182e-02" in result.stdout
        assert "6.1868603e-05" in result.stdout
