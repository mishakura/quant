"""Centralized configuration — paths, directories, and project-level settings.

All path constants are derived from PROJECT_ROOT so the package works
regardless of where it is installed.
"""

from pathlib import Path

# ── Project root (the repo directory containing pyproject.toml) ──────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# ── Data directories ─────────────────────────────────────────────────────────
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
CACHE_DIR: Path = DATA_DIR / "cache"
SAMPLE_DATA_DIR: Path = DATA_DIR / "samples"

# ── Output directories ───────────────────────────────────────────────────────
OUTPUT_DIR: Path = PROJECT_ROOT / "output"
BACKTEST_OUTPUT_DIR: Path = OUTPUT_DIR / "backtests"
REPORT_OUTPUT_DIR: Path = OUTPUT_DIR / "reports"
OPTIMIZATION_OUTPUT_DIR: Path = OUTPUT_DIR / "optimizations"
SIMULATION_OUTPUT_DIR: Path = OUTPUT_DIR / "simulations"

# ── Configuration directory ──────────────────────────────────────────────────
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"

# ── Notebook directory ───────────────────────────────────────────────────────
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"


def ensure_directories() -> None:
    """Create all standard output directories if they don't exist."""
    for d in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        CACHE_DIR,
        SAMPLE_DATA_DIR,
        BACKTEST_OUTPUT_DIR,
        REPORT_OUTPUT_DIR,
        OPTIMIZATION_OUTPUT_DIR,
        SIMULATION_OUTPUT_DIR,
        CONFIGS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
