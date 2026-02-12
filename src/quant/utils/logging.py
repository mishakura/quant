"""Structured logging configuration for the quant package.

Usage
-----
At the top of any module::

    from quant.utils.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Downloading prices for %d tickers", len(tickers))

At script entry points::

    from quant.utils.logging import setup_logging

    setup_logging(level="INFO")
"""

import logging
import sys


_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging(
    level: str | int = "INFO",
    fmt: str = _LOG_FORMAT,
    date_fmt: str = _DATE_FORMAT,
) -> None:
    """Configure root logger with a stream handler to stderr.

    Safe to call multiple times — subsequent calls are no-ops.

    Parameters
    ----------
    level : str or int
        Log level (e.g. ``"DEBUG"``, ``"INFO"``, ``logging.WARNING``).
    fmt : str
        Log message format string.
    date_fmt : str
        Date format string for the ``%(asctime)s`` placeholder.
    """
    global _configured
    if _configured:
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger within the ``quant`` namespace.

    Parameters
    ----------
    name : str
        Logger name — typically ``__name__``.

    Returns
    -------
    logging.Logger
    """
    return logging.getLogger(name)
