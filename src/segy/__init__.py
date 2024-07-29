"""'The' SEG-Y implementation."""
from importlib import metadata

from segy.factory import SegyFactory
from segy.file import SegyFile

__all__ = ["SegyFile", "SegyFactory"]

try:
    __version__ = metadata.version("segy")
except metadata.PackageNotFoundError:
    __version__ = "unknown"
