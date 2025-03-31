"""Test Numpy array subclasses."""

import numpy as np
import pytest

from segy.arrays import HeaderArray
from segy.arrays import SegyArray
from segy.exceptions import InvalidFieldError
from segy.exceptions import NonSpecFieldError


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


@pytest.mark.parametrize(
    ("keys", "aliases", "values"),
    [
        (["samples_per_trace", "sample_interval"], ["ns", "interval"], (5, 2)),
        (["samples_per_trace", "sample_interval"], ["Samples", "dt"], (1, 6)),
    ],
)
class TestHeaderAlias:
    """Test header alias get/set item methods and exceptions."""

    def test_getitem_alias(
        self, keys: list[str], aliases: list[str], values: list[int]
    ) -> None:
        """Test getting values by SU and segyio aliases."""
        array = np.asarray(values, dtype="int16")
        buffer = bytearray(array.tobytes())
        dtype = np.dtype({"names": keys, "formats": ["int16", "int16"]})
        struct = np.frombuffer(buffer, dtype=dtype)
        header_array = HeaderArray(struct)

        # str branch tests
        assert header_array[aliases[0]].item() == values[0]
        assert header_array[aliases[1]].item() == values[1]
        with pytest.raises(InvalidFieldError, match="Invalid key"):
            _ = header_array["foo"]
        with pytest.raises(NonSpecFieldError, match="spec does not define this field"):
            _ = header_array["SEGYRevision"]

        # list[str branch] tests
        assert header_array[aliases].item() == values
        with pytest.raises(NonSpecFieldError, match="spec does not define this field"):
            _ = header_array[["SEGYRevision", "ExtSamples"]]

        # test shortcuts (i.e. not using alias)
        assert header_array[keys[0]].item() == values[0]
        assert header_array[keys].item() == values

    def test_setitem_alias(
        self, keys: list[str], aliases: list[str], values: list[int]
    ) -> None:
        """Test setting values by SU and segyio aliases."""
        array = np.zeros(len(values), dtype="int16")
        buffer = bytearray(array.tobytes())
        dtype = np.dtype({"names": keys, "formats": ["int16", "int16"]})
        struct = np.frombuffer(buffer, dtype=dtype)
        header_array = HeaderArray(struct)

        # str branch tests
        header_array[aliases[0]] = values[0]
        header_array[aliases[1]] = values[1]
        assert header_array[aliases].item() == values

        # list[str branch] tests
        values_x2 = tuple(val * 2 for val in values)
        header_array[aliases] = values_x2
        assert header_array[aliases].item() == values_x2

        # test shortcuts (i.e. not using alias)
        values_x3 = tuple(val * 3 for val in values)
        header_array[keys] = values_x3
        assert header_array[aliases].item() == values_x3
