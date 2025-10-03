"""Accessors for accessing parts of TraceArrays."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from segy.config import SegyHeaderOverrides
from segy.schema import Endianness
from segy.schema import ScalarType
from segy.transforms import TransformFactory
from segy.transforms import TransformPipeline

if TYPE_CHECKING:
    from segy.schema import TraceSpec


logger = logging.getLogger(__name__)


class TraceAccessor:
    """Accessor for applying required transforms for reading SegyArrays and subclasses.

    trace_spec: Trace specification as instance of TraceSpec dtype.
    header_overrides: Overrides for replacing header field values with custom values.
    """

    def __init__(
        self,
        trace_spec: TraceSpec,
        header_overrides: SegyHeaderOverrides | None = None,
    ) -> None:
        logger.debug("Initializing %s", self.__class__.__name__)

        self.trace_spec = trace_spec
        self.header_overrides = header_overrides or SegyHeaderOverrides()

        self.header_ibm_keys = [
            field.name
            for field in self.trace_spec.header.fields
            if field.format == ScalarType.IBM32
        ]

        self.header_decode_pipeline = TransformPipeline()
        self.sample_decode_pipeline = TransformPipeline()
        self.trace_decode_pipeline = TransformPipeline()

        self._update_header_pipeline()
        self._update_sample_pipeline()
        self._update_trace_pipeline()

    def _update_header_pipeline(self) -> None:
        endian_transform = TransformFactory.create("byte_swap", Endianness.LITTLE)
        self.header_decode_pipeline.add_transform(endian_transform)

        if len(self.header_ibm_keys) != 0:
            ibm2ieee = TransformFactory.create(
                "ibm_float", "to_ieee", keys=self.header_ibm_keys
            )
            self.header_decode_pipeline.add_transform(ibm2ieee)

        if len(self.header_overrides.trace_header) > 0:
            for key, value in self.header_overrides.trace_header.items():
                self.header_decode_pipeline.add_transform(
                    TransformFactory.create("replace_header_value", key, value)
                )

    def _update_sample_pipeline(self) -> None:
        endian_transform = TransformFactory.create("byte_swap", Endianness.LITTLE)
        self.sample_decode_pipeline.add_transform(endian_transform)

        if self.trace_spec.data.format == ScalarType.IBM32:
            ibm2ieee = TransformFactory.create("ibm_float", "to_ieee")
            self.sample_decode_pipeline.add_transform(ibm2ieee)

    def _update_trace_pipeline(self) -> None:
        trace_transform = TransformFactory.create(
            "trace", self.header_decode_pipeline, self.sample_decode_pipeline
        )
        self.trace_decode_pipeline.add_transform(trace_transform)
