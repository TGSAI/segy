"""Test the usage of SegyFile class."""


import pytest

from segy import SegyFile


@pytest.mark.parametrize(
    ("mock_segy_rev0", "num_samples", "num_traces"),
    [((5, 5), 5, 5), ((10, 5), 10, 5)],
    indirect=["mock_segy_rev0"],
)
def test_segy_rev0(mock_segy_rev0: SegyFile, num_samples: int, num_traces: int) -> None:
    """Tests various attributes and methods of a SegyFile with Rev 0 specs."""
    assert "This is a sample text header" in mock_segy_rev0.text_header
    assert mock_segy_rev0.num_traces == num_traces
    assert mock_segy_rev0.spec.trace.data_descriptor.samples == num_samples
    assert len(mock_segy_rev0.data[:]) == num_traces
    expected_value = 1.0
    assert (mock_segy_rev0.data[:] == expected_value).all()
    assert (
        mock_segy_rev0.header[:]["trace_seq_line"].values
        == list(range(1, num_traces + 1))
    ).all()
    assert (mock_segy_rev0.trace[:]["header"] == mock_segy_rev0.header[:]).all().all()
    assert (mock_segy_rev0.trace[:]["data"] == mock_segy_rev0.data[:]).all()


@pytest.mark.parametrize(
    ("mock_segy_rev1", "num_samples", "num_traces"),
    [((5, 5, 2), 5, 5), ((10, 5, 3), 10, 5)],
    indirect=["mock_segy_rev1"],
)
def test_segy_rev1(mock_segy_rev1: SegyFile, num_samples: int, num_traces: int) -> None:
    """Tests various attributes and methods of a SegyFile with Rev 1 specs."""
    assert "This is a sample text header" in mock_segy_rev1.text_header
    assert mock_segy_rev1.num_traces == num_traces
    assert mock_segy_rev1.spec.trace.data_descriptor.samples == num_samples
    assert len(mock_segy_rev1.data[:]) == num_traces
    expected_value = 1.0
    assert (mock_segy_rev1.data[:] == expected_value).all()
    assert (
        mock_segy_rev1.header[:]["trace_seq_line"].values
        == list(range(1, num_traces + 1))
    ).all()
    assert (mock_segy_rev1.trace[:]["header"] == mock_segy_rev1.header[:]).all().all()
    assert (mock_segy_rev1.trace[:]["data"] == mock_segy_rev1.data[:]).all()
