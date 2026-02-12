"""I/O utilities for reading and writing common data formats.

Provides thin wrappers around pandas I/O with sensible defaults for
financial data (datetime index parsing, Parquet compression, etc.).
"""

from pathlib import Path

import pandas as pd
import yaml


# ── CSV ──────────────────────────────────────────────────────────────────────

def read_csv(
    path: str | Path,
    index_col: int | str = 0,
    parse_dates: bool = True,
    **kwargs: object,
) -> pd.DataFrame:
    """Read a CSV file into a DataFrame with datetime index by default.

    Parameters
    ----------
    path : str or Path
        File path.
    index_col : int or str
        Column to use as index (default: first column).
    parse_dates : bool
        Attempt to parse the index as dates (default: True).
    **kwargs
        Forwarded to ``pd.read_csv``.

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_csv(path, index_col=index_col, parse_dates=parse_dates, **kwargs)


def write_csv(df: pd.DataFrame, path: str | Path, **kwargs: object) -> Path:
    """Write a DataFrame to CSV, creating parent directories if needed.

    Returns
    -------
    Path
        The written file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, **kwargs)
    return path


# ── Parquet ──────────────────────────────────────────────────────────────────

def read_parquet(path: str | Path, **kwargs: object) -> pd.DataFrame:
    """Read a Parquet file into a DataFrame.

    Parameters
    ----------
    path : str or Path
        File path.
    **kwargs
        Forwarded to ``pd.read_parquet``.

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_parquet(path, **kwargs)


def write_parquet(
    df: pd.DataFrame,
    path: str | Path,
    compression: str = "snappy",
    **kwargs: object,
) -> Path:
    """Write a DataFrame to Parquet, creating parent directories if needed.

    Returns
    -------
    Path
        The written file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression=compression, **kwargs)
    return path


# ── Excel ────────────────────────────────────────────────────────────────────

def read_excel(
    path: str | Path,
    sheet_name: str | int = 0,
    index_col: int | str | None = 0,
    **kwargs: object,
) -> pd.DataFrame:
    """Read an Excel file into a DataFrame.

    Parameters
    ----------
    path : str or Path
        File path.
    sheet_name : str or int
        Sheet to read (default: first sheet).
    index_col : int, str, or None
        Column to use as index (default: first column).
    **kwargs
        Forwarded to ``pd.read_excel``.

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_excel(path, sheet_name=sheet_name, index_col=index_col, **kwargs)


def write_excel(
    df: pd.DataFrame,
    path: str | Path,
    sheet_name: str = "Sheet1",
    **kwargs: object,
) -> Path:
    """Write a DataFrame to Excel, creating parent directories if needed.

    Returns
    -------
    Path
        The written file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, sheet_name=sheet_name, **kwargs)
    return path


# ── YAML ─────────────────────────────────────────────────────────────────────

def read_yaml(path: str | Path) -> dict:
    """Read a YAML configuration file.

    Parameters
    ----------
    path : str or Path
        File path.

    Returns
    -------
    dict
        Parsed YAML contents.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
