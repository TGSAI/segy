"""Test Numpy array subclasses."""

import numpy as np

from segy.arrays import SegyArray


def test_segy_array_copy() -> None:
    """Test copying a segy array with exact underlying buffer."""
    buffer_expected = np.asarray([0, 1, 2, 3, 4], dtype="uint16").tobytes()

    dtype = np.dtype(
        {
            "names": ["f1", "f2"],
            "offsets": [0, 4],
            "formats": ["uint16", "uint16"],
            "itemsize": 10,
        }
    )
    segy_array = SegyArray(np.frombuffer(buffer_expected, dtype=dtype))
    segy_array_copy = segy_array.copy()

    assert segy_array_copy.tobytes() == buffer_expected
    assert np.may_share_memory(segy_array_copy, segy_array) is False
