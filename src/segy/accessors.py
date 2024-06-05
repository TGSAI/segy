"""Accessors for accessing parts of TraceArrays."""

from __future__ import annotations

from typing import TYPE_CHECKING

from segy.schema import Endianness
from segy.schema import ScalarType
from segy.transforms import TransformFactory

if TYPE_CHECKING:
    from segy.schema import TraceSpec
    from segy.transforms import Transform


class TraceAccessor:
    """Accessor for applying required transforms for reading SegyArrays and subclasses.

    trace_spec: Trace specification as instance of TraceSpec dtype.
    """

    def __init__(self, trace_spec: TraceSpec) -> None:
        self.trace_spec = trace_spec
        self.header_ibm_keys = [
            field.name
            for field in self.trace_spec.header_spec.fields
            if field.format == ScalarType.IBM32
        ]
        self.header_decode_transforms: list[Transform] = []
        self.sample_decode_transforms: list[Transform] = []
        self.trace_decode_transforms: list[Transform] = []
        self._update_decoders()

    def _update_decoders(self) -> None:
        self.sample_decode_transforms.append(
            TransformFactory.create("byte_swap", Endianness.LITTLE)
        )
        self.header_decode_transforms.append(
            TransformFactory.create("byte_swap", Endianness.LITTLE)
        )
        self.trace_decode_transforms.append(
            TransformFactory.create("byte_swap", Endianness.LITTLE)
        )
        if self.trace_spec.data_spec.format == ScalarType.IBM32:
            self.sample_decode_transforms.append(
                TransformFactory.create("ibm_float", "to_ieee")
            )
            self.trace_decode_transforms.append(
                TransformFactory.create("ibm_float", "to_ieee", ["data"])
            )

        if len(self.header_ibm_keys) != 0:
            self.header_decode_transforms.append(
                TransformFactory.create(
                    "ibm_float", "to_ieee", keys=self.header_ibm_keys
                )
            )
