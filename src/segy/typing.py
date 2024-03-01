"""Useful aliases for typing."""


from typing import TypeAlias

from numpy import float32
from numpy import uint32
from numpy.typing import NDArray

NDArrayUint32: TypeAlias = NDArray[uint32]
NDArrayFloat32: TypeAlias = NDArray[float32]
