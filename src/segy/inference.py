"""Utilities to extract information from the SEG-Y files.

We have the following inference options:
1. Endianness inference
2. Revision interpretation
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from segy.config import SegyFileSettings
from segy.config import SegyHeaderOverrides
from segy.exceptions import EndiannessInferenceError
from segy.schema import Endianness
from segy.schema import SegyStandard
from segy.standards.codes import DataSampleFormatCode
from segy.standards.codes import SegyEndianCode

if TYPE_CHECKING:
    from numpy._typing import DTypeLike

logger = logging.getLogger(__name__)


class EndiannessAction(Enum):
    """Descriptive flag enum for endianness reversal."""

    REVERSE = True
    KEEP = False


@dataclass
class SegyInferResult:
    """A scan result of a SEG-Y file.

    Attributes:
        endianness: Endianness of the file.
        revision: SEG-Y revision as float.
    """

    __slots__ = ("endianness", "revision")

    endianness: Endianness
    revision: float


def infer_endianness(
    buffer: bytes,
    settings: SegyFileSettings,
) -> EndiannessAction:
    """Infer if we need to reverse the endianness of the seismic data.

    Args:
        buffer: Bytes representing the binary header.
        settings: Settings instance to configure / override.

    Returns:
        A boolean indicating if the endianness need to be reversed.

    Raises:
        EndiannessInferenceError: When all methods fail.
        NotImplementedError: When a pairwise swapped endianness is detected.
    """
    logger.debug("Starting endianness inference.")

    # Method 1: Use settings if available
    if settings.endianness is not None:
        logger.info("Using provided endianness from settings: %s", settings.endianness)
        return EndiannessAction(settings.endianness != sys.byteorder)

    # Method 2: Explicit endianness code (SEG-Y Rev2+)
    logger.debug("Trying explicit endianness code (SEGY Rev2+).")
    endian_code = np.frombuffer(buffer, "uint32", offset=96, count=1)[0]

    if endian_code == SegyEndianCode.NATIVE:
        logger.info("Detected native endianness.")
        return EndiannessAction.KEEP
    if endian_code == SegyEndianCode.REVERSE:
        logger.info("Detected reverse endianness.")
        return EndiannessAction.REVERSE
    if endian_code == SegyEndianCode.PAIRWISE_SWAP:
        msg = "Pairwise swapped endianness detected. Not supported."
        logger.error(msg)
        raise NotImplementedError(msg)
    if endian_code != 0:
        logger.warning("Ambiguous explicit endianness code: %s", endian_code)

    # Method 3: Legacy method using sample format for inference (SEG-Y <Rev2)
    logger.debug("Trying legacy method for SEGY Rev <2.0.")
    format_dtype = np.dtype("uint16")
    supported_formats = set(DataSampleFormatCode._value2member_map_.keys())

    def _is_supported_format(dtype: DTypeLike) -> bool:
        format_value = np.frombuffer(buffer, dtype, offset=24, count=1)[0]
        return format_value in supported_formats

    if _is_supported_format(format_dtype):
        logger.info("Detected native endianness using legacy method.")
        return EndiannessAction.KEEP

    if _is_supported_format(format_dtype.newbyteorder()):
        logger.info("Detected reverse endianness using legacy method.")
        return EndiannessAction.REVERSE

    # If all methods fail
    error_message = (
        "Endianness inference failed after attempting all methods. "
        "Ensure the file is valid or provide explicit settings."
    )
    logger.error(error_message)
    raise EndiannessInferenceError(error_message)


def interpret_revision(
    buffer: bytes,
    endianness_action: EndiannessAction | None = None,
    overrides: SegyHeaderOverrides | None = None,
) -> int | float:
    """Infer the revision number from the binary header of a SEG-Y file.

    Args:
        buffer: The binary header buffer.
        endianness_action: The action to take for endianness.
        overrides: The SegyHeaderOverrides, which may override the revision.

    Returns:
        The revision number as a float (e.g., 1.0, 1.2, 2.0).
    """
    if endianness_action is None:
        endianness_action = EndiannessAction.KEEP

    if overrides is None:
        overrides = SegyHeaderOverrides()

    logger.debug("Starting revision inference.")

    # Method 1: Use override if available
    user_revision = overrides.binary_header.get("segy_revision", None)
    if user_revision is not None:
        logger.info("Using provided revision from override: %s", user_revision)
        return user_revision

    # Method 2: Major/minor from single byte integers (SEG-Y Rev2+)
    logger.debug("Checking if file is SEG-Y Rev2+.")
    major_revision = np.frombuffer(buffer, "uint8", offset=300, count=1)[0]

    if major_revision >= SegyStandard.REV2:
        minor_revision = np.frombuffer(buffer, "uint8", offset=301, count=1)[0]
    else:
        # Method 3: Major/minor from 16-bit integer (SEG-Y <Rev2)
        logger.debug("File is SEG-Y <Rev2, reading revision from 16-bits.")
        dtype = np.dtype("uint16")
        if endianness_action == EndiannessAction.REVERSE:
            dtype = dtype.newbyteorder()

        revision = np.frombuffer(buffer, dtype, offset=300, count=1)[0]
        major_revision = revision >> 8
        minor_revision = revision & 0xFF

    revision_float = int(major_revision) + int(minor_revision) / 10
    logger.info("Detected revision from binary header as %s", revision_float)
    return revision_float
