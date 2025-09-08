"""Tests for IBM and IEEE floating point conversions.

Some references for test values
https://en.wikipedia.org/wiki/IBM_hexadecimal_floating-point
https://www.crewes.org/Documents/ResearchReports/2017/CRR201725.pdf
"""

import numpy as np
import pytest

from segy.ibm import ibm2ieee
from segy.ibm import ibm2ieee_single
from segy.ibm import ieee2ibm
from segy.ibm import ieee2ibm_single


@pytest.mark.parametrize(
    ("ieee", "ibm"),
    [
        (0.0, 0x00000000),
        (-0.0, 0x00000000),
        (0.1, 0x40199999),
        (-1, 0xC1100000),
        (3.141593, 0x413243F7),
        (-0.15625, 0xC0280000),
        (118.625, 0x4276A000),
        (-8521603, 0xC6820783),
        (3.4028235e38, 0x60FFFFFF),
        (-3.4028235e38, 0xE0FFFFFF),
        ([-0.0, 0.1], [0x00000000, 0x40199999]),
        ([0.0, 0.1, 3.141593], [0x00000000, 0x40199999, 0x413243F7]),
        ([[0.0], [0.1], [3.141593]], [[0x00000000], [0x40199999], [0x413243F7]]),
    ],
)
class TestIbmIeee:
    """Test conversions between values as Ibm floats to Ieee floats."""

    def test_ieee_to_ibm(self, ieee: float, ibm: int) -> None:
        """Test converting values from IEEE to IBM."""
        ieee_float32 = np.float32(ieee)
        actual_ibm = ieee2ibm(ieee_float32)
        expected_ibm = np.uint32(ibm)
        np.testing.assert_array_equal(actual_ibm, expected_ibm)

    def test_ibm_to_ieee(self, ieee: float, ibm: int) -> None:
        """Test converting values from IBM to IEEE."""
        expected_ieee = np.float32(ieee)
        actual_ibm = np.uint32(ibm)
        actual_ieee = ibm2ieee(actual_ibm)
        np.testing.assert_array_almost_equal(actual_ieee, expected_ieee)


@pytest.mark.parametrize(
    ("ieee", "ibm"),
    [
        (0.0, 0x00000000),
        (-0.0, 0x00000000),
        (0.1, 0x40199999),
        (-1, 0xC1100000),
        (3.141593, 0x413243F7),
        (-0.15625, 0xC0280000),
        (118.625, 0x4276A000),
        (-8521603, 0xC6820783),
        (3.4028235e38, 0x60FFFFFF),
        (-3.4028235e38, 0xE0FFFFFF),
    ],
)
class TestSingleIbmIeee:
    """Test conversions for single values from Ibm floats to Ieee floats."""

    def test_single_ieee_to_ibm(self, ieee: float, ibm: int) -> None:
        """Test converting single values from IEEE to IBM."""
        ieee_float32 = np.float32(ieee)
        actual_ibm = ieee2ibm_single(ieee_float32)
        expected_ibm = np.uint32(ibm)
        np.testing.assert_array_almost_equal(actual_ibm, expected_ibm)

    def test_single_ibm_to_ieee(self, ieee: float, ibm: int) -> None:
        """Test converting values from IBM to IEEE."""
        expected_ieee = np.float32(ieee)
        actual_ibm = np.uint32(ibm)
        acutal_ieee = ibm2ieee_single(actual_ibm)
        np.testing.assert_array_almost_equal(acutal_ieee, expected_ieee)


@pytest.mark.parametrize("shape", [(1,), (10,), (20, 20), (150, 150)])
def test_ieee_to_ibm_roundtrip(shape: tuple[int, ...]) -> None:
    """Convert values from IEEE to IBM and back to IEEE."""
    rng = np.random.default_rng()
    expected_ieee = rng.normal(size=shape).astype("float32")
    actual_ibm = ieee2ibm(expected_ieee)
    actual_ieee = ibm2ieee(actual_ibm)

    np.testing.assert_array_almost_equal(actual_ieee, expected_ieee)
