"""Low-level floating point conversion operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numba as nb
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    NDArrayUint32 = NDArray[np.uint32]
    NDArrayFloat32 = NDArray[np.float32]


# IEEE to IBM MASKS ETC
IEEE32_SIGN = np.uint32(0x80000000)
IEEE32_EXPONENT = np.int32(0x7F800000)
IEEE32_FRACTION = np.uint32(0x7FFFFF)

# IBM to IEEE MASKS ETC
BASE2POW24 = np.uint32(0x1000000)
IBM32_EXPONENT = np.uint32(0x7F000000)
IBM32_FRACTION = np.uint32(0xFFFFFF)


@nb.njit(  # type: ignore
    "uint32(float32)",
    nogil=True,
    cache=True,
    locals={
        "sign": nb.uint32,
        "exponent": nb.int32,
        "exp_remainder": nb.int8,
        "downshift": nb.int8,
        "ibm_mantissa": nb.int32,
    },
)
def ieee2ibm_single(ieee):  # noqa: ANN201,ANN001,DOC106,DOC107
    """IEEE Float to IBM Float conversion.

    Modified from here:
    https://mail.python.org/pipermail/scipy-user/2011-June/029661.html

    Had to do some CPU and memory optimizations + Numba JIT compilation

    Assuming `ieee_array` is little endian and float32. Will convert to float32 if not.
    Returns `ibm_array` as little endian too.

    Byte swapping is up to user after this function.

    Args:
        ieee: Numpy IEEE 32-bit float array.

    Returns:
        IBM 32-bit float converted array with int32 view.
    """
    ieee = np.float32(ieee).view(np.uint32)

    if ieee in [0, 2147483648]:  # 0.0 or np.float32(-0.0).view('uint32')
        return np.uint32(0)

    # Get IEEE's sign and exponent
    sign = ieee & IEEE32_SIGN
    exponent = ((ieee & IEEE32_EXPONENT) >> 23) - 127
    # The IBM 7-bit exponent is to the base 16 and the mantissa is presumed to
    # be entirely to the right of the radix point. In contrast, the IEEE
    # exponent is to the base 2 and there is an assumed 1-bit to the left of
    # the radix point.
    # Note: reusing exponent variable, -> it is actually exp16

    # exp16, exp_remainder
    exponent, exp_remainder = divmod(exponent + 1, 4)
    exponent += exp_remainder != 0
    downshift = 4 - exp_remainder if exp_remainder else 0
    exponent = exponent + 64
    # From here down exponent -> ibm_exponent
    exponent = 0 if exponent < 0 else exponent
    exponent = 127 if exponent > 127 else exponent  # noqa: PLR2004
    exponent = exponent << 24
    exponent = exponent if ieee else 0

    # Add the implicit initial 1-bit to the 23-bit IEEE mantissa to get the
    # 24-bit IBM mantissa. Downshift it by the remainder from the exponent's
    # division by 4. It is allowed to have up to 3 leading 0s.
    ibm_mantissa = ((ieee & IEEE32_FRACTION) | 0x800000) >> downshift

    return sign | exponent | ibm_mantissa


@nb.njit(  # type: ignore
    "float32(uint32)",
    cache=True,
    nogil=True,
    locals={
        "sign_bit": nb.boolean,
        "sign": nb.int8,
        "exponent": nb.uint8,
        "mantissa": nb.float32,
        "ieee": nb.float32,
    },
)
def ibm2ieee_single(ibm):  # noqa: ANN201,ANN001,DOC106,DOC107
    """Converts a 32-bit IBM floating point number into 32-bit IEEE format.

    https://en.wikipedia.org/wiki/IBM_hexadecimal_floating-point
    https://en.wikipedia.org/wiki/IEEE_754

    FP Number = (IBM) -1**sign_bit × 0.significand × 16**(exponent−64)

    Args:
        ibm: Value in 32-bit IBM Float in Little-Endian Format.

    Returns:
        Value parsed to 32-bit IEEE Float in Little-Endian Format.
    """
    if ibm & IBM32_FRACTION == 0:
        return np.float32(0)

    sign_bit = ibm >> 31

    exponent = (ibm & IBM32_EXPONENT) >> 24
    mantissa = (ibm & IBM32_FRACTION) / BASE2POW24

    # (1 - 2 * sign_bit) is about 50x faster than (-1)**sign_bit
    sign = 1 - 2 * sign_bit

    # This 16.0 (instead of just 16) is super important.
    # If the base is not a float, it won't work for negative
    # exponents, and fail silently and return zero.
    return sign * mantissa * 16.0 ** (exponent - 64)


@nb.vectorize("uint32(float32)", cache=True)  # type: ignore
def ieee2ibm(ieee_array: NDArrayFloat32) -> NDArrayUint32:  # pragma: no cover
    """Wrapper for vectorizing IEEE to IBM conversion to arrays."""
    ibm: NDArrayUint32 = ieee2ibm_single(ieee_array)
    return ibm


@nb.vectorize("float32(uint32)", cache=True)  # type: ignore
def ibm2ieee(ibm_array: NDArrayUint32) -> NDArrayFloat32:  # pragma: no cover
    """Wrapper for vectorizing IBM to IEEE conversion to arrays."""
    ieee: NDArrayFloat32 = ibm2ieee_single(ibm_array)
    return ieee
