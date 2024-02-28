"""conftest for main tests."""
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from segy import SegyFile
from segy import ibm as convert_floats
from segy.standards import SegyStandard
from segy.standards import registry

text_header_default_content = "This is a sample text header. "


def create_mock_segy_rev0(
    tmp_file: Path,
    num_samples: int,
    num_traces: int,
    format_str_to_text_header: Callable,
) -> SegyFile:
    """Create a temporary file that mocks a segy Rev0 file structure."""
    rev0_spec = registry.get_spec(SegyStandard.REV0)
    rev0_spec.trace.data_descriptor.samples = num_samples

    text_header_content = text_header_default_content
    sample_text_header = format_str_to_text_header(text_header_content)
    binary_header_vals = np.zeros((), dtype=rev0_spec.binary_file_header.dtype)
    binary_header_vals["samples_per_trace"] = num_samples

    trace_header_vals = np.zeros((), dtype=rev0_spec.trace.header_descriptor.dtype)
    trace_header_vals["trace_seq_line"] = 1
    trace_data_vals = np.zeros((), dtype=rev0_spec.trace.data_descriptor.dtype)
    trace_data_vals[:] = convert_floats.ieee2ibm(np.ones(num_samples, dtype="float32"))

    buffer_out = b""
    buffer_out += rev0_spec.text_file_header._encode(sample_text_header)
    buffer_out += binary_header_vals.tobytes()

    for _ in range(num_traces):
        buffer_out += trace_header_vals.copy(order="K").tobytes()
        buffer_out += trace_data_vals.copy(order="K").tobytes()
        trace_header_vals["trace_seq_line"] += 1

    with tmp_file.open(mode="wb") as fh:
        fh.write(buffer_out)

    return SegyFile(str(tmp_file), spec=rev0_spec)


@pytest.fixture(scope="session")
def mock_segy_rev0(
    request: list[int], tmp_path: Path, format_str_to_text_header: Callable
) -> SegyFile:
    """Returns a temp file that for rev0 SegyFile object."""
    req_params = getattr(request, "param", [10, 10])
    num_samples, num_traces = req_params[0], req_params[1]

    temp_rev0 = tmp_path / "rev0_test.segy"

    return create_mock_segy_rev0(
        temp_rev0, num_samples, num_traces, format_str_to_text_header
    )


def create_mock_segy_rev1(
    tmp_file: Path,
    num_samples: int,
    num_traces: int,
    num_extended_headers: int,
    format_str_to_text_header: Callable,
) -> SegyFile:
    """Create a temporary file that mocks a segy Rev1 file structure."""
    rev1_spec = registry.get_spec(SegyStandard.REV1)
    rev1_spec.trace.data_descriptor.samples = num_samples

    text_header_content = text_header_default_content
    sample_text_header = format_str_to_text_header(text_header_content)
    extended_text_headers = [
        format_str_to_text_header(f"This is extended text header number {i}")
        for i in range(1, num_extended_headers + 1)
    ]

    binary_header_vals = np.zeros((), dtype=rev1_spec.binary_file_header.dtype)
    binary_header_vals["samples_per_trace"] = num_samples
    binary_header_vals["extended_textual_headers"] = num_extended_headers
    binary_header_vals["seg_y_revision"] = 1

    trace_header_vals = np.zeros((), dtype=rev1_spec.trace.header_descriptor.dtype)
    trace_header_vals["trace_seq_line"] = 1
    trace_data_vals = np.zeros((), dtype=rev1_spec.trace.data_descriptor.dtype)
    trace_data_vals[:] = convert_floats.ieee2ibm(np.ones(num_samples, dtype="float32"))

    buffer_out = b""
    buffer_out += rev1_spec.text_file_header._encode(sample_text_header)
    buffer_out += binary_header_vals.tobytes()

    for extended_txt in extended_text_headers:
        buffer_out += rev1_spec.text_file_header._encode(extended_txt)

    for _ in range(num_traces):
        buffer_out += trace_header_vals.copy(order="K").tobytes()
        buffer_out += trace_data_vals.copy(order="K").tobytes()
        trace_header_vals["trace_seq_line"] += 1

    with tmp_file.open(mode="wb") as fh:
        fh.write(buffer_out)

    return SegyFile(str(tmp_file), spec=rev1_spec)


@pytest.fixture(scope="session")
def mock_segy_rev1(
    request: list[int], tmp_path: Path, format_str_to_text_header: Callable
) -> SegyFile:
    """Returns a temp file that for rev1 SegyFile object."""
    req_params = getattr(request, "param", [10, 10, 2])
    num_samples, num_traces, num_extended_headers = (
        req_params[0],
        req_params[1],
        req_params[2],
    )

    temp_rev1 = tmp_path / "rev1_test.segy"

    return create_mock_segy_rev1(
        temp_rev1,
        num_samples,
        num_traces,
        num_extended_headers,
        format_str_to_text_header,
    )
