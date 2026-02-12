"""Custom exception hierarchy for the quant package.

All quant-specific exceptions inherit from QuantError so callers can catch
the entire family with a single except clause when appropriate.
"""


class QuantError(Exception):
    """Base exception for all quant module errors."""


class DataError(QuantError):
    """Raised when data validation fails.

    Examples: missing columns, negative prices, NaN-only series,
    insufficient history for a requested calculation.
    """


class OptimizationError(QuantError):
    """Raised when portfolio optimization fails to converge."""


class ConfigError(QuantError):
    """Raised for invalid configuration parameters.

    Examples: unknown risk measure name, negative lookback window,
    missing required config keys.
    """


class InsufficientDataError(DataError):
    """Raised when there are not enough observations for a calculation.

    Parameters
    ----------
    required : int
        Minimum number of observations needed.
    actual : int
        Number of observations available.
    """

    def __init__(self, required: int, actual: int, context: str = "") -> None:
        self.required = required
        self.actual = actual
        self.context = context
        msg = f"Need at least {required} observations, got {actual}"
        if context:
            msg += f" ({context})"
        super().__init__(msg)
