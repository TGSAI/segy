"""Testing the header accessor and transforms."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import pytest

from segy.header import TransformationPipeline
from segy.header import TransformStrategyFactory
from segy.schema import Endianness

if TYPE_CHECKING:
    from numpy.typing import NDArray


@pytest.fixture()
def mock_header() -> NDArray[Any]:
    """Generate a mock structured array to test transforms."""
    names = ["field1", "field2", "field3"]
    formats = [">u4", ">i4", ">f8"]

    dtype = np.dtype({"names": names, "formats": formats})
    arr = np.empty(shape=1, dtype=dtype)
    arr[:] = (42, 1, 3.1415)

    return arr


def test_transform_pipeline(mock_header: NDArray[Any]) -> None:
    """Test transformation pipeline and existing transforms."""
    expected = (420, 10, 3.1415)
    pipeline = TransformationPipeline()

    pipeline.add_transformation(
        TransformStrategyFactory.create_strategy(
            transform_type="scale",
            parameters={"scale_factor": 10, "keys": ["field1", "field2"]},
        )
    )

    pipeline.add_transformation(
        TransformStrategyFactory.create_strategy(
            transform_type="byte_swap",
            parameters={
                "source_byteorder": Endianness.BIG,
                "target_byteorder": Endianness.LITTLE,
            },
        )
    )

    transformed_header = pipeline.transform(mock_header)

    assert transformed_header.item() == expected
    assert transformed_header.dtype == mock_header.dtype.newbyteorder()
