"""Helper functions that are used throughout tests."""
import operator
import random
import string
from collections.abc import Callable

import numpy as np

from segy.schema.data_type import StructuredFieldDescriptor


class TestHelpers:
    """This class acts as a namespace for helper functions."""

    @staticmethod
    def void_buffer(buff_size: int) -> np.ndarray:
        """Creates a new buffer of requested number of bytes with void(number_bytes) datatype.
        Prefills with random bytes.
        """
        rng = np.random.default_rng()
        new_void_buffer = None
        if isinstance(buff_size, int):
            new_void_buffer = np.frombuffer(
                rng.bytes(buff_size), dtype=np.void(buff_size)
            )
        return new_void_buffer

    @staticmethod
    def get_dt_info(
        dt: np.dtype,
        atrnames: list[str] | None = None,
    ) -> dict:
        """Helper function to get info about a numpy dtype."""
        if atrnames is None:
            atrnames = [
                "descr",
                "str",
                "fields",
                "itemsize",
                "byteorder",
                "shape",
                "names",
            ]
        dt_info = dict(zip(atrnames, operator.attrgetter(*atrnames)(dt), strict=False))
        dt_info["offsets"] = [f[-1] for f in dt_info["fields"].values()]
        dt_info["combo_str"] = ",".join([f[1] for f in dt_info["descr"]])
        return dt_info

    @staticmethod
    def build_sfd_helper(
        format: str, endianness: str, name: str, offset: int, asdict: bool = False
    ) -> StructuredFieldDescriptor:
        """Convenience helper for creating the StrucuredFieldDescriptors."""
        params_dict = {
            "format": format,
            "endianness": endianness,
            "name": name,
            "offset": offset,
        }
        if asdict:
            return params_dict
        return StructuredFieldDescriptor(**params_dict)

    @staticmethod
    def build_sdt_fields(*params) -> list[StructuredFieldDescriptor]:
        """Convenience for creating a list of StructuredFieldDescriptors."""
        return [TestHelpers.build_sfd_helper(*p) for p in params]

    @staticmethod
    def custom_format_parser(
        formats: list[str],
        names: list[str] | None = None,
        titles: list[str] | None = None,
        aligned: bool = False,
        byteorder: str = "<",
    ) -> np.format_parser:
        """Takes a list of numpy compatible data type strings and builds a new np.dtype.

        Args:
            formats (list[str]): _description_
            names (list[str] | None, optional): _description_. Defaults to None.
            titles (list[str] | None, optional): _description_. Defaults to None.
            aligned (bool, optional): _description_. Defaults to False.
            byteorder (str, optional): _description_. Defaults to "<".

        Returns:
            np.format_parser: _description_
        """
        names = [] if names is None else names
        titles = [] if titles is None else titles
        return np.format_parser(
            formats, names, titles, aligned=aligned, byteorder=byteorder
        )

    @staticmethod
    def values_to_dtype_strings(
        values: list,
        delim: str = ",",
        dt_func: Callable = np.min_scalar_type,
        replace_types: list[tuple] | None = None,
    ) -> str:
        """Takes a list of arbitrary values and finds a numpy compatible data type string.

        Args:
            values (list): _description_
            delim (str, optional): _description_. Defaults to ",".
            dt_func (Callable, optional): _description_. Defaults to np.min_scalar_type.
            replace_types (list[tuple] | None, optional): _description_. Defaults to None.

        Returns:
            str: _description_
        """
        if replace_types is None:
            replace_types = [("f2", "f4")]
        dt_string = delim.join([dt_func(n).str for n in values])
        for rep_types in replace_types:
            dt_string = dt_string.replace(*rep_types)
        return dt_string

    @staticmethod
    def format_str_to_text_header(text: str) -> str:
        """Helper function to create fixed size text headers from a given string."""
        return "{0:{fill}{align}{n}}".format(
            text.replace("\n", ""), fill="", align="<", n=3200
        )

    @staticmethod
    def generate_unique_names(N: int) -> list[str]:
        """Helper function to create random unique names as placeholders during testing."""
        names = set()
        while len(names) < N:
            name_length = random.randint(5, 10)
            name = "".join(random.choices(string.ascii_uppercase, k=name_length))
            names.add(name)
        return list(names)
