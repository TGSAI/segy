"""Normalization logic for header field names."""

from rapidfuzz import process
from rapidfuzz.fuzz import WRatio

from segy.alias.segyio import SEGYIO_BIN_FIELD_MAP
from segy.alias.segyio import SEGYIO_TRACE_FIELD_MAP
from segy.alias.seis_unix import SEIS_UNIX_TRACE_FIELD_MAP
from segy.exceptions import InvalidFieldError
from segy.exceptions import NonSpecFieldError
from segy.standards.fields import binary
from segy.standards.fields import trace

FIELD_MAP = {}

FIELD_MAP.update({k.lower(): v.name.lower() for k, v in SEGYIO_BIN_FIELD_MAP.items()})
FIELD_MAP.update({k.lower(): v.name.lower() for k, v in SEGYIO_TRACE_FIELD_MAP.items()})
FIELD_MAP.update(
    {k.lower(): v.name.lower() for k, v in SEIS_UNIX_TRACE_FIELD_MAP.items()}
)

CANONICAL_KEYS = (
    set()
    # Binary header
    | {field.name for field in binary.Rev0}
    | {field.name for field in binary.Rev1}
    | {field.name for field in binary.Rev2}
    # Trace header
    | {field.name for field in trace.Rev0}
    | {field.name for field in trace.Rev1}
    | {field.name for field in trace.Rev2}
)
CANONICAL_KEYS = {name.lower() for name in CANONICAL_KEYS}


def process_str_for_fuzz(string: str) -> str:
    """Remove underscores and lowercase string."""
    return string.replace("_", "").lower()


def get_suggestion_keys(key: str) -> list[str]:
    """Get similar keys using weighted text similarity metrics."""
    suggestions = process.extract(
        key,
        CANONICAL_KEYS,
        limit=3,
        scorer=WRatio,
        processor=process_str_for_fuzz,
    )

    return [match[0] for match in suggestions]


def normalize_key(key: str) -> str:
    """Normalize a key to its canonical form."""
    key_lower = key.lower()
    if key_lower in CANONICAL_KEYS:
        return key_lower

    alias_key = FIELD_MAP.get(key_lower)

    if alias_key is None:
        suggestion_keys = get_suggestion_keys(key)
        raise InvalidFieldError(key, suggestion_keys)

    return alias_key


def validate_key(key: str, orig_key: str, defined_keys: tuple[str, ...]) -> None:
    """Validate a key against known fields and aliases."""
    if key not in defined_keys:
        msg = (
            f"The header field '{orig_key}' is found in field alias table as "
            f"'{key}'. However, the current SEG-Y spec does not define this "
            f"field in header fields."
        )
        raise NonSpecFieldError(msg)
