"""Tests for the compontent classes in the schema directory."""

import itertools
import string
from collections.abc import Callable

import numpy as np
import pytest

from segy.schema import BinaryHeaderDescriptor
from segy.schema import ScalarType
from segy.schema import TraceDataDescriptor
from segy.schema import TraceHeaderDescriptor
from segy.schema.data_type import DataTypeDescriptor
from segy.schema.data_type import Endianness

# Constants defined for ScalarType
DTYPE_FORMATS = [s.value for s in ScalarType]

DTYPE_ENDIANNESS = [Endianness.LITTLE, Endianness.BIG]

# For cases where a description is supplied or not
DTYPE_DESCRIPTIONS = [None, "this is a data type description"]


# create different combinations of data types in binary header
BINARY_HEADER_TEST_DTYPE_STRINGS = [
    ",".join(["i4"] * 5),
    ",".join([*["i4"] * 2, *["i2"] * 3, *["i4"]]),
    ",".join([*["i4"] * 3, *["i2"] * 24]),
]

TRACE_HEADER_TEST_DTYPE_STRINGS = [
    ",".join([*["i4"] * 2, *["i2"] * 3, *["i4"]]),
    ",".join([*["i4"] * 3, *["i2"] * 4, *["i4"] * 5, *["i2"] * 6]),
]


@pytest.fixture(
    params=[
        (dt_string, None, dt_endian)
        for dt_string in BINARY_HEADER_TEST_DTYPE_STRINGS
        for dt_endian in DTYPE_ENDIANNESS
    ]
)
def binary_header_descriptors(
    request: pytest.FixtureRequest, make_binary_header_descriptor: Callable
) -> BinaryHeaderDescriptor:
    """Generates BinaryHeaderDescriptor objects from parameters.

    Args:
        request: params for creating BinaryHeaderDescriptor.
        make_binary_header_descriptor: helper function for creating object.

    Returns:
        BinaryHeaderDescriptor: Descriptor object of BinaryHeaderDescriptor
    """
    return make_binary_header_descriptor(
        dt_string=request.param[0], names=request.param[1], endianness=request.param[2]
    )


@pytest.fixture(
    params=[
        (dt_string, dt_endian)
        for dt_string in TRACE_HEADER_TEST_DTYPE_STRINGS
        for dt_endian in DTYPE_ENDIANNESS
    ]
)
def trace_header_descriptors(
    request: pytest.FixtureRequest, make_trace_header_descriptor: Callable
) -> TraceHeaderDescriptor:
    """Generates TraceHeaderDescriptor objects from parameters.

    Args:
        request: params for creating TraceHeaderDescriptor.
        make_trace_header_descriptor: helper function for creating object

    Returns:
        TraceHeaderDescriptor: Descriptor object of TraceHeaderDescriptor

    """
    return make_trace_header_descriptor(
        dt_string=request.param[0], endianness=request.param[1]
    )


def generate_unique_names(count: int) -> list[str]:
    """Helper function to create random unique names as placeholders during testing."""
    names: set[str] = set()
    rng = np.random.default_rng()
    while len(names) < count:
        name_length = rng.integers(5, 10)  # noqa: S311
        letters = rng.choice(list(string.ascii_uppercase), size=name_length)  # noqa: S311
        name = "".join(letters)
        names.add(name)
    return list(names)


@pytest.fixture(
    params=[
        (p1, p2, DTYPE_DESCRIPTIONS[1])
        for p1, p2 in zip(DTYPE_FORMATS, itertools.cycle(DTYPE_ENDIANNESS))
    ]
)
def data_types(request: pytest.FixtureRequest) -> DataTypeDescriptor:
    """Fixture to create all combinations of data types defined in ScalarType."""
    return DataTypeDescriptor(
        format=request.param[0],
        endianness=request.param[1],
        description=request.param[2],
    )


@pytest.fixture(
    params=[
        (p1, p2, DTYPE_DESCRIPTIONS[1], 100)
        for p1, p2 in zip(DTYPE_FORMATS, itertools.cycle(DTYPE_ENDIANNESS))
    ]
)
def trace_data_descriptors(
    request: pytest.FixtureRequest, make_trace_data_descriptor: Callable
) -> TraceDataDescriptor:
    """Fixture that creates TraceDataDescriptors of different data types and endianness."""
    return make_trace_data_descriptor(
        format=request.param[0],
        endianness=request.param[1],
        description=request.param[2],
        samples=request.param[3],
    )


sample_text = """
Here's some sample text. It should have numbers,
letters, punctuation and newlines,
~~ , 123, -456, 0.1234, 123145.01234.
Finally make sure these are in order: abcdeABCDE
"""
sample_real_header_text = """
C01 AREA        : OFFSHORE REGION A - OFFSHORE REGION A SEISMIC DATABASE
C02 DESCRIPTION : SEISMIC COVERAGE - PHASE MATCHED IN GEODETAIL
C03 ===========================================================================
C04 DATE     :1996       CLASS      :RAW MIGRATION /+90 DEGREE PHASE SHIFT
C05 OPERATOR :ABC        PROCESSING :XYZ COUNTRY
C06 ===========================================================================
C07 THE 3D DATA HAS BEEN DATUM AND PHASE SHIFTED. DATA HAS BEEN MERGED WITH
C08 CHECKED NAV AND EXPORTED FROM GEODETAIL 4.2 IN STANDARD SEGY.
C09 INLINES/SP RANGE :510-796 CDP INC       :1        SAMPLE INTERVAL :4000
C10 XLINES/CDP RANGE :58-792  SAMPLES/TRACE :1251     FINAL TIME :5000
C11 LINE INC      :1  TRACES/LINE   :VARIABLE IL/XL X/EAST Y/NORTH
C12 ===========================================================================
C13 THIS DATASET WAS PREPARED AND COMPILED BY THE SEISMIC RESEARCH INSTITUTE
C14 AND TECHNOLOGY SERVICES LIMITED (SRITS), 1 OCEAN VIEW TERRACE,
C15 METROPOLIS, PACIFIC OCEAN. FUNDING FOR THIS PROJECT WAS PROVIDED BY THE
C16 PACIFIC RESEARCH FOUNDATION: CO5X0302 AND CO5X0905.
C17
C18 THIS DATA IS PROVIDED ON A "AS IS" BASIS AND ALTHOUGH DATA HAS BEEN
C19 MODIFIED BY SRITS, NO WARRANTY, EXPRESSED OR IMPLIED, IS MADE BY
C20 SRITS AS TO THE ACCURACY OF THE DATA OR RELATED MATERIALS, ITS
C21 COMPLETENESS OR FITNESS FOR PURPOSE. IN NO EVENT WILL SRITS, ITS
C22 EMPLOYEES, AGENTS OR CONTRACTORS BE LIABLE FOR ANY LOSS COSTS OR DAMAGE
C23 ARISING FROM ANY PARTIES USE OR RELIANCE ON THE DATASET INCLUDING ANY
C24 CONSEQUENTIAL, SPECIAL, INDIRECT, INCIDENTAL, PUNITIVE OR EXEMPLARY
C25 DAMAGES, COSTS, EXPENSES OR LOSSES. SRITS WILL NOT ACCEPT ANY
C26 LIABILITY FOR THE CONSEQUENCES OF ANY PARTY ACTING ON THIS INFORMATION.
C27 ===========================================================================
C28 BYTE LOCATIONS :      SURVEY GEOMETRY    SURVEY/DATA PARAMETERS
C29 LINE      :BYTES 221  MIN Line :510      DATA TYPE  :SEGY
C30 CDP       :BYTES 21   MAX Line :796      MEDIA No   :E02337 - E02342
C31 SOURCE X  :BYTES 73   MIN CDP  :58       PROJECTION :NZTM
C32 SOURCE Y  :BYTES 77   MAX CDP  :792      DATUM      :NZGD2000
C33 ===========================================================================
C34 POINTS USED FOR        LINE 510  CDP 58        LINE 792  CDP 796
C35 SURVEY DEFINITION:     1703638   5571677       1689838   5608539
C36                        LINE 510  CDP 792
C37                        1704135   5608346
C38 DATE CREATED: 1 FEB 2010
C39 USER: A.NONYMOUS
C40
"""


@pytest.fixture(params=[sample_text, sample_real_header_text])
def text_header_samples(
    request: pytest.FixtureRequest, format_str_to_text_header: Callable
) -> str:
    """Fixture that generates fixed size text header test data from strings."""
    return format_str_to_text_header(request.param)
