"""Accessors for accessing parts of TraceArrays."""

from __future__ import annotations

from typing import TYPE_CHECKING
from segy.schema import ScalarType
from segy.schema import Endianness
from segy.transforms import TransformFactory

if TYPE_CHECKING:
    from segy.schema import TraceDescriptor


class TraceAccessor:
    def __init__(self, trace_spec: TraceDescriptor) -> None:
        self.trace_spec = trace_spec
        self.header_ibm_keys = [
            field.name
            for field in self.trace_spec.header_descriptor.fields
            if field.dtype == ScalarType.IBM32
        ]
        self.header_decode_transforms = []
        self.sample_decode_transforms = []
        self.trace_decode_transforms = []
        self._update_decoders()

    def _update_decoders(self) -> None:
        self.sample_decode_transforms.append(
            TransformFactory.create("byte_swap", Endianness.LITTLE)
        )
        if self.trace_spec.sample_descriptor.format == ScalarType.IBM32:
            self.sample_decode_transforms.append(
                TransformFactory.create("ibm_float", "to_ieee")
            )

        self.header_decode_transforms.append(
            TransformFactory.create("byte_swap", Endianness.LITTLE)
        )
        self.trace_decode_transforms.append(
            TransformFactory.create("byte_swap", Endianness.LITTLE)
        )

        if self.header_ibm_keys:
            self.header_decode_transforms.append(
                TransformFactory.create("ibm_float", "to_ieee", self.header_ibm_keys)
            )
            self.trace_decode_transforms.append(
                TransformFactory.create(
                    "trace_ibm_float", "to_ieee", self.header_ibm_keys
                )
            )
