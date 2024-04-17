"""Accessors for accessing parts of TraceArrays."""

from __future__ import annotations

from typing import TYPE_CHECKING

from segy.schema import Endianness
from segy.schema import ScalarType
from segy.transforms import TransformFactory

if TYPE_CHECKING:
    from segy.schema import TraceDescriptor
    from segy.transforms import Transform


class TraceAccessor:
    """Accessor for applying required transforms for reading SegyArrays and subclasses.

    trace_spec: Descriptor of TraceArray dtype.
    """

    def __init__(self, trace_spec: TraceDescriptor) -> None:
        self.trace_spec = trace_spec
        self.header_ibm_keys = [
            field.name
            for field in self.trace_spec.header_descriptor.fields
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
        if self.trace_spec.sample_descriptor.format == ScalarType.IBM32:
            self.sample_decode_transforms.append(
                TransformFactory.create("ibm_float", "to_ieee")
            )
            self.trace_decode_transforms.append(
                TransformFactory.create("ibm_float", "to_ieee", ["sample"])
            )

        if len(self.header_ibm_keys) != 0:
            self.header_decode_transforms.append(
                TransformFactory.create(
                    "ibm_float", "to_ieee", keys=self.header_ibm_keys
                )
            )
