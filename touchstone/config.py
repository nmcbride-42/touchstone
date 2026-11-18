"""Environment-variable configuration. Every knob is documented in README.md
and .env.example; nothing here reads files outside the repo."""
from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    """Where cached datasets (light curves) live. Default: <repo>/data."""
    return Path(os.environ.get("TOUCHSTONE_DATA_DIR", _REPO_ROOT / "data"))


def output_dir() -> Path:
    """Where example runs write ledgers and artifacts. Default: <repo>/runs."""
    return Path(os.environ.get("TOUCHSTONE_OUTPUT_DIR", _REPO_ROOT / "runs"))


def env(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name)
    return val if val not in (None, "") else default
