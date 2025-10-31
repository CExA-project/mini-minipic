"""Exceptions for the project."""


class MiniPICError(Exception):
    """Base error class."""


class ValueMiniPICError(MiniPICError):
    """Invalid value."""


class ThresholdValueMiniPICError(ValueMiniPICError):
    """Value above threshold."""


class IncorrectValueMiniPICError(ValueMiniPICError):
    """Incorrect value."""


class FileMiniPICError(MiniPICError):
    """Invalid file."""


class MissingFileMiniPICError(FileMiniPICError):
    """File missing."""


class IncorrectFileMiniPICError(FileMiniPICError):
    """File with wrong content."""
