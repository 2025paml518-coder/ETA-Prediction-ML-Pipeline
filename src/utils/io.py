"""Idempotent writes.

A batch stage that writes directly to its target is not safe to retry: a crash
midway leaves a truncated file that looks like a valid output to the next stage.
Everything here writes to a staging path first and then atomically replaces the
target, so a stage either produces a complete output or leaves the previous one
untouched (M2 section 2.4.1).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


def _staging_path(path: Path) -> Path:
    return path.with_name(path.name + ".staging")


def atomic_write_parquet(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = _staging_path(path)
    frame.to_parquet(staging, index=False)
    os.replace(staging, path)
    return path


def atomic_write_text(text: str, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = _staging_path(path)
    staging.write_text(text, encoding="utf-8")
    os.replace(staging, path)
    return path


def atomic_write_json(payload: object, path: str | Path, indent: int | None = 2) -> Path:
    return atomic_write_text(json.dumps(payload, indent=indent), path)
