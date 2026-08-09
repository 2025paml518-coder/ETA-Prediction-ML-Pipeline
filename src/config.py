"""Project paths and parameter loading.

Every stage reads its configuration from ``params.yaml`` so that DVC can detect
parameter changes and invalidate exactly the stages that depend on them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"

_CACHE: dict[Path, dict[str, Any]] = {}


def load_params(path: str | Path | None = None) -> dict[str, Any]:
    """Load ``params.yaml`` (cached per path)."""
    resolved = Path(path).resolve() if path is not None else PARAMS_PATH
    if resolved not in _CACHE:
        with resolved.open("r", encoding="utf-8") as handle:
            _CACHE[resolved] = yaml.safe_load(handle)
    return _CACHE[resolved]


def project_path(relative: str | Path) -> Path:
    """Resolve a repo-relative path against the project root."""
    candidate = Path(relative)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if missing and return it."""
    resolved = project_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_parent(path: str | Path) -> Path:
    """Create the parent directory of a file path and return the file path."""
    resolved = project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
