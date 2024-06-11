"""Testing the header accessor and transforms."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import pytest

from segy.schema.base import Endianness
from segy.transforms import TransformFactory
from segy.transforms import TransformPipeline

if TYPE_CHECKING:
    from numpy.typing import NDArray


@pytest.fixture()
def mock_header_little() -> NDArray[Any]:
    """Generate a mock structured array to test transforms with little endian."""
    names = ["field1", "field2", "field3"]
    formats = ["<u4", "<i4", "<f4"]

    dtype = np.dtype({"names": names, "formats": formats})
    arr = np.empty(shape=1, dtype=dtype)
    arr[:] = (42, 1, 3.1415)

    return arr


@pytest.fixture()
def mock_header_big(mock_header_little: NDArray[Any]) -> NDArray[Any]:
    """Generate a mock structured array to test transforms with big endian."""
    return mock_header_little.byteswap().newbyteorder()


@pytest.fixture()
def mock_header_ibm() -> NDArray[Any]:
    """Generate a mock structured array to test IBM float field transform."""
    names = ["u4_field", "ibm_field", "u2_field", "ibm_field2"]
    formats = ["<u4", "<u4", "<u2", "<u4"]

    dtype = np.dtype({"names": names, "formats": formats})
    arr = np.empty(shape=1, dtype=dtype)
    arr[:] = (256, 0x4276A000, 8, 0x413243F7)

    return arr


@pytest.fixture()
def mock_data_little() -> NDArray[Any]:
    """Generate a mock little endian structured array to test transforms."""
    return np.asarray([[3.14, 42, 0], [-2, 3, -4]]).astype(np.float32)


@pytest.fixture()
def mock_data_big(mock_data_little: NDArray[Any]) -> NDArray[Any]:
    """Generate a mock big endian structured array to test transforms."""
    return mock_data_little.byteswap().newbyteorder()


class TestByteSwap:
    """Test byte swap transformations."""

    @pytest.mark.parametrize(
        ("input_endian", "expected_order"),
        [
            ("little", Endianness.BIG),
            ("big", Endianness.LITTLE),
            ("big", Endianness.BIG),  # test_noop
        ],
    )
    def test_byteswap(
        self,
        request: pytest.FixtureRequest,
        input_endian: str,
        expected_order: Endianness,
    ) -> None:
        """Test byte swap transform."""
        mock_data = request.getfixturevalue(f"mock_data_{input_endian}")
        expected_dtype = mock_data.dtype.newbyteorder(expected_order.symbol)

        # Transform but make copy before because transform is in place.
        transform = TransformFactory.create("byte_swap", expected_order)
        swapped_data = transform.apply(mock_data.copy())

        np.testing.assert_allclose(swapped_data, mock_data)
        assert swapped_data.dtype == expected_dtype


class TestIbmFloat:
    """Test IBM Float transformations."""

    def test_ibm_float(self) -> None:
        """Test array scaling."""
        ieee_value = np.asarray(3.141593)
        expected_ibm_uint32 = 0x413243F7

        transform = TransformFactory.create("ibm_float", "to_ibm")
        ibm_uint32 = transform.apply(ieee_value)

        assert ibm_uint32 == expected_ibm_uint32

        ibm_value = np.asarray([0x4276A000], dtype="uint32")
        expected_ieee = np.asarray([118.625])

        transform = TransformFactory.create("ibm_float", "to_ieee")
        ibm_value = transform.apply(ibm_value)

        np.testing.assert_array_equal(ibm_value, expected_ieee)

    def test_ibm_float_field(self, mock_header_ibm: NDArray[Any]) -> None:
        """Test array scaling."""
        expected_dtype = [
            ("", "uint32"),
            ("", "float32"),
            ("", "uint16"),
            ("", "float32"),
        ]
        expected = np.asarray((256, 118.625, 8, 3.141593), dtype=expected_dtype)

        transform = TransformFactory.create(
            "ibm_float", "to_ieee", keys=["ibm_field", "ibm_field2"]
        )
        transformed_header = transform.apply(mock_header_ibm)

        assert transformed_header[0].item() == expected.item()


class TestTransformPipeline:
    """Tests for transform pipeline and transform integration."""

    def test_transform_pipeline(self, mock_data_little: NDArray[Any]) -> None:
        """Test transformation pipeline and transform integration."""
        expected_data = mock_data_little.copy()

        pipeline = TransformPipeline()
        pipeline.add_transform(TransformFactory.create("byte_swap", Endianness.BIG))
        pipeline.add_transform(TransformFactory.create("byte_swap", Endianness.LITTLE))

        transformed_data = pipeline.apply(mock_data_little)
        np.testing.assert_allclose(transformed_data, expected_data)
        assert transformed_data.dtype == expected_data.dtype

    def test_transform_pipeline_field(self, mock_header_ibm: NDArray[Any]) -> None:
        """Test transformation pipeline and transform integration for struct fields."""
        expected_dtype = [
            ("u4_field", "uint32"),
            ("ibm_field", "float32"),
            ("u2_field", "uint16"),
            ("ibm_field2", "float32"),
        ]
        expected_data = np.asarray((256, 118.625, 8, 3.141593), dtype=expected_dtype)

        keys = ["ibm_field", "ibm_field2"]
        pipeline = TransformPipeline()
        pipeline.add_transform(TransformFactory.create("byte_swap", Endianness.BIG))
        pipeline.add_transform(TransformFactory.create("byte_swap", Endianness.LITTLE))
        pipeline.add_transform(TransformFactory.create("ibm_float", "to_ieee", keys))

        transformed_data = pipeline.apply(mock_header_ibm.copy())

        assert transformed_data == expected_data
        assert transformed_data.dtype == expected_dtype

    def test_transform_exceptions(
        self,
        mock_data_little: NDArray[Any],
        mock_header_big: NDArray[Any],
    ) -> None:
        """Test transformation field errors."""
        scale_transform = TransformFactory.create("ibm_float", "to_ibm", ["field1"])
        with pytest.raises(ValueError, match="doesn't contain any fields"):
            scale_transform.apply(mock_data_little)

        scale_transform = TransformFactory.create("ibm_float", "to_ibm", ["unknown"])
        with pytest.raises(ValueError, match="{'unknown'} not in the array"):
            scale_transform.apply(mock_header_big)

    def test_transform_factory_exception(self) -> None:
        """Test unknown transformation type."""
        factory = TransformFactory()

        with pytest.raises(KeyError, match="Unsupported transformation"):
            factory.create("non_existent_transform")
