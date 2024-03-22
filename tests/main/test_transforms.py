"""Testing the header accessor and transforms."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import pytest

from segy.schema import Endianness
from segy.transforms import TransformPipeline
from segy.transforms import TransformStrategyFactory

if TYPE_CHECKING:
    from numpy.typing import NDArray


@pytest.fixture()
def mock_header() -> NDArray[Any]:
    """Generate a mock structured array to test transforms."""
    names = ["field1", "field2", "field3"]
    formats = [">u4", ">i4", ">f4"]

    dtype = np.dtype({"names": names, "formats": formats})
    arr = np.empty(shape=1, dtype=dtype)
    arr[:] = (42, 1, 3.1415)

    return arr


@pytest.fixture()
def mock_data() -> NDArray[Any]:
    """Generate a mock structured array to test transforms."""
    return np.asarray([[-1, 0, 1], [2, 3, 4]])


class TestTransforms:
    """Test all forward and reverse transformations."""

    def test_byteswap(self, mock_header: NDArray[Any]) -> None:
        """Test byte swap transform."""
        expected = (42, 1, 3.1415)
        strategy = TransformStrategyFactory.create_strategy(
            transform_type="byte_swap",
            parameters={
                "source_order": Endianness.BIG,
                "target_order": Endianness.LITTLE,
            },
        )

        transformed_header = strategy.transform(mock_header)
        np.testing.assert_allclose(transformed_header.item(), expected)
        assert transformed_header.dtype == mock_header.dtype.newbyteorder()

        roundtrip_header = strategy.inverse_transform(transformed_header)
        np.testing.assert_allclose(roundtrip_header.item(), expected)
        assert roundtrip_header.dtype == mock_header.dtype

    def test_scale(self, mock_data: NDArray[Any]) -> None:
        """Test array scaling."""
        scale_factor = 5
        strategy = TransformStrategyFactory.create_strategy(
            transform_type="scale",
            parameters={
                "scale_factor": scale_factor,
            },
        )

        transformed_data = strategy.transform(mock_data)

        expected = mock_data * scale_factor
        np.testing.assert_array_equal(transformed_data, expected)

        rountrip_data = strategy.inverse_transform(transformed_data)
        np.testing.assert_array_equal(rountrip_data, mock_data)

    def test_scale_field(self, mock_header: NDArray[Any]) -> None:
        """Test structured array field(s) scaling."""
        expected = (420, 10, 3.1415)
        expected_roundtrip = (42, 1, 3.1415)
        strategy = TransformStrategyFactory.create_strategy(
            transform_type="scale_field",
            parameters={"scale_factor": 10, "keys": ["field1", "field2"]},
        )

        transformed_header = strategy.transform(mock_header)
        np.testing.assert_allclose(transformed_header.item(), expected)

        roundtrip_header = strategy.inverse_transform(transformed_header)
        np.testing.assert_allclose(roundtrip_header.item(), expected_roundtrip)

    def test_transform_pipeline(self, mock_data: NDArray[Any]) -> None:
        """Test transformation pipeline."""
        pipeline = TransformPipeline()

        scale_factor = 3
        pipeline.add_transformation(
            TransformStrategyFactory.create_strategy(
                transform_type="scale",
                parameters={"scale_factor": scale_factor},
            )
        )

        pipeline.add_transformation(
            TransformStrategyFactory.create_strategy(
                transform_type="byte_swap",
                parameters={
                    "source_order": Endianness.LITTLE,
                    "target_order": Endianness.BIG,
                },
            )
        )

        transformed_data = pipeline.transform(mock_data)
        np.testing.assert_allclose(transformed_data, mock_data * scale_factor)

        roundtrip_data = pipeline.inverse_transform(transformed_data)
        np.testing.assert_allclose(roundtrip_data, mock_data)
