"""Test the usage of SegyFile class."""

import pytest
from numpy.testing import assert_array_equal

from segy import SegyFile


@pytest.mark.parametrize(
    ("mock_segy_rev0", "num_samples", "num_traces", "sample_interval"),
    [((5, 5, 2000), 5, 5, 2000), ((10, 5, 4000), 10, 5, 4000)],
    indirect=["mock_segy_rev0"],
)
def test_segy_rev0(
    mock_segy_rev0: SegyFile,
    num_samples: int,
    num_traces: int,
    sample_interval: int,
) -> None:
    """Tests various attributes and methods of a SegyFile with Rev 0 specs."""
    sample_labels_expected = range(0, num_samples * sample_interval, sample_interval)
    assert_array_equal(mock_segy_rev0.sample_labels, sample_labels_expected)

    assert "This is a sample text header" in mock_segy_rev0.text_header
    assert mock_segy_rev0.num_traces == num_traces
    assert mock_segy_rev0.samples_per_trace == num_samples
    assert mock_segy_rev0.num_ext_text == 0

    assert mock_segy_rev0.spec.trace.data_descriptor.samples == num_samples
    assert len(mock_segy_rev0.data[:]) == num_traces
    assert (
        mock_segy_rev0.header[:]["trace_seq_line"] == list(range(1, num_traces + 1))
    ).all()

    expected_value = 1.0
    assert_array_equal(mock_segy_rev0.data[:], expected_value)
    assert_array_equal(mock_segy_rev0.trace[:].header, mock_segy_rev0.header[:])
    assert_array_equal(mock_segy_rev0.trace[:].data, mock_segy_rev0.data[:])


@pytest.mark.parametrize(
    ("mock_segy_rev1", "num_samples", "num_traces", "n_ext_headers", "sample_interval"),
    [((5, 5, 2, 2000), 5, 5, 2, 2000), ((10, 5, 3, 4000), 10, 5, 3, 4000)],
    indirect=["mock_segy_rev1"],
)
def test_segy_rev1(
    mock_segy_rev1: SegyFile,
    num_samples: int,
    num_traces: int,
    n_ext_headers: int,
    sample_interval: int,
) -> None:
    """Tests various attributes and methods of a SegyFile with Rev 1 specs."""
    sample_labels_expected = range(0, num_samples * sample_interval, sample_interval)
    assert_array_equal(mock_segy_rev1.sample_labels, sample_labels_expected)

    assert "This is a sample text header" in mock_segy_rev1.text_header
    assert mock_segy_rev1.num_traces == num_traces
    assert mock_segy_rev1.samples_per_trace == num_samples
    assert mock_segy_rev1.num_ext_text == n_ext_headers

    assert mock_segy_rev1.spec.trace.data_descriptor.samples == num_samples
    assert len(mock_segy_rev1.data[:]) == num_traces
    assert (
        mock_segy_rev1.header[:]["trace_seq_line"] == list(range(1, num_traces + 1))
    ).all()

    expected_value = 1.0
    assert_array_equal(mock_segy_rev1.data[:], expected_value)
    assert_array_equal(mock_segy_rev1.trace[:].header, mock_segy_rev1.header[:])
    assert_array_equal(mock_segy_rev1.trace[:].data, mock_segy_rev1.data[:])
