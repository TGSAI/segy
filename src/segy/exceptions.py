"""SEG-Y library exceptions."""


class SegyError(Exception):
    """Base class for all exceptions in this library."""


class InvalidFieldError(SegyError):
    """Raised when a header key does not match known values."""

    def __init__(self, key: str, suggestions: list[str]):
        self.key = key
        self.suggestions = suggestions
        msg = self._generate_message()
        super().__init__(msg)

    def _generate_message(self) -> str:
        """Generate custom user message."""
        first, second, third = self.suggestions
        suggestion_text = f"'{first}', '{second}', or '{third}'"
        return f"Invalid key '{self.key}'. Did you mean one of: {suggestion_text}?"


class NonSpecFieldError(SegyError):
    """Raised when a header field is missing from a HeaderArray."""
