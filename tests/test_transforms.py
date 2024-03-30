"""Testing the header accessor and transforms."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import pytest

from segy.schema import Endianness
from segy.transforms import TransformFactory
from segy.transforms import TransformPipeline
from segy.transforms import get_endianness

if TYPE_CHECKING:
    from numpy.typing import DTypeLike
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

        transform = TransformFactory.create("byte_swap", expected_order)
        swapped_data = transform.apply(mock_data)

        np.testing.assert_allclose(swapped_data, mock_data)
        assert swapped_data.dtype == expected_dtype


@pytest.mark.parametrize("endian", [Endianness.LITTLE, Endianness.BIG])
class TestScaling:
    """Test scaling swap transformations."""

    @pytest.mark.parametrize("fields", [None, ["field1"]])
    def test_scale(
        self,
        request: pytest.FixtureRequest,
        endian: Endianness,
        fields: list[str] | None,
    ) -> None:
        """Test array scaling."""
        scale_factor = 5
        is_structured = fields is not None

        mock_fixture = f"mock_data_{endian.value}"

        if is_structured:
            mock_fixture = mock_fixture.replace("data", "header")

        mock_data = request.getfixturevalue(mock_fixture)

        if is_structured and fields is not None:
            expected = mock_data.copy()
            for field in fields:
                expected[field] = mock_data[field] * scale_factor
        else:
            expected = mock_data * scale_factor

        transform = TransformFactory.create("scale", scale_factor, keys=fields)
        scaled_data = transform.apply(mock_data)

        np.testing.assert_array_equal(scaled_data, expected)
        assert get_endianness(scaled_data) == endian

    def test_scale_field_casting(
        self,
        request: pytest.FixtureRequest,
        endian: Endianness,
    ) -> None:
        """Test casted structured field scaling."""
        mock_data = request.getfixturevalue(f"mock_header_{endian.value}")

        scale_factor = 0.1
        keys = ["field1"]
        expected = (4.2, 1, 3.1415)

        transform = TransformFactory.create("scale", scale_factor, keys=keys)
        scaled_data = transform.apply(mock_data)

        np.testing.assert_array_almost_equal(scaled_data.item(), expected)
        assert get_endianness(scaled_data) == endian


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


@pytest.mark.parametrize(
    ("endian", "cast_to"),
    [
        (Endianness.LITTLE, "<i4"),
        (Endianness.BIG, ">i4"),
    ],
)
class TestCastType:
    """Test dtype casting transforms."""

    def test_cast_dtype(
        self,
        request: pytest.FixtureRequest,
        endian: Endianness,
        cast_to: DTypeLike,
    ) -> None:
        """Test array casting with little endian."""
        mock_data = request.getfixturevalue(f"mock_data_{endian.value}")
        expected = mock_data.astype(cast_to)

        transform = TransformFactory.create("cast", cast_to)
        cast_data = transform.apply(mock_data)

        np.testing.assert_array_equal(cast_data, expected)
        assert get_endianness(cast_data) == endian

    def test_cast_dtype_field(
        self,
        request: pytest.FixtureRequest,
        endian: Endianness,
        cast_to: DTypeLike,
    ) -> None:
        """Test structured array field casting with little endian."""
        mock_data = request.getfixturevalue(f"mock_header_{endian.value}")
        expected = (42, 1, 3)

        transform = TransformFactory.create("cast", cast_to, keys=["field3"])
        cast_data = transform.apply(mock_data)

        np.testing.assert_array_equal(cast_data.item(), expected)
        assert get_endianness(cast_data) == endian


class TestTransformPipeline:
    """Tests for transform pipeline and transform integration."""

    @staticmethod
    def build_transform_pipeline(
        scale_factor: int,
        fields: list[str] | None,
    ) -> TransformPipeline:
        """Build common transform pipeline for tests."""
        cast_type = "float16"

        scale_transform = TransformFactory.create("scale", scale_factor, keys=fields)
        endian_transform = TransformFactory.create("byte_swap", Endianness.LITTLE)
        cast_transform = TransformFactory.create("cast", cast_type, keys=fields)

        pipeline = TransformPipeline()
        pipeline.add_transform(scale_transform)
        pipeline.add_transform(endian_transform)
        pipeline.add_transform(cast_transform)

        return pipeline

    def test_transform_pipeline(self, mock_data_big: NDArray[Any]) -> None:
        """Test transformation pipeline and transform integration."""
        scale_factor = 3
        expected_data = mock_data_big.byteswap().newbyteorder()
        expected_data = (expected_data * scale_factor).astype("float16")

        expected_dtype = expected_data.dtype

        pipeline = self.build_transform_pipeline(scale_factor=scale_factor, fields=None)

        transformed_data = pipeline.apply(mock_data_big)
        np.testing.assert_allclose(transformed_data, expected_data)
        assert transformed_data.dtype == expected_dtype

    def test_transform_pipeline_field(self, mock_header_big: NDArray[Any]) -> None:
        """Test transformation pipeline and transform integration for struct fields."""
        scale_factor = 3
        fields = ["field2", "field3"]

        names = ["field1", "field2", "field3"]
        formats = ["<u4", "<f2", "<f2"]
        expected_dtype = np.dtype({"names": names, "formats": formats})

        expected_data = mock_header_big.copy()
        for field in fields:
            expected_data[field] = expected_data[field] * scale_factor

        expected_data = expected_data.byteswap().newbyteorder()
        expected_data = expected_data.astype(expected_dtype)

        pipeline = self.build_transform_pipeline(
            scale_factor=scale_factor, fields=fields
        )

        transformed_data = pipeline.apply(mock_header_big)
        assert transformed_data == expected_data
        assert transformed_data.dtype == expected_dtype

    def test_transform_exceptions(
        self,
        mock_data_little: NDArray[Any],
        mock_header_big: NDArray[Any],
    ) -> None:
        """Test transformation field errors."""
        scale_transform = TransformFactory.create("scale", 1, ["field1"])
        with pytest.raises(ValueError, match="doesn't contain any fields"):
            scale_transform.apply(mock_data_little)

        scale_transform = TransformFactory.create("scale", 1, ["non_existent"])
        with pytest.raises(ValueError, match="{'non_existent'} not in the array"):
            scale_transform.apply(mock_header_big)

    def test_transform_factory_exception(self) -> None:
        """Test unknown transformation type."""
        factory = TransformFactory()

        with pytest.raises(KeyError, match="Unsupported transformation"):
            factory.create("non_existent_transform")
