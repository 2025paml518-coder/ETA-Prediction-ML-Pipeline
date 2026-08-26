"""Project paths and parameter loading.

Every stage reads its configuration from ``params.yaml`` so that DVC can detect
parameter changes and invalidate exactly the stages that depend on them.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"

_CACHE: dict[Path, dict[str, Any]] = {}

# Configuration validation schema
VALIDATION_RULES = {
    'seed': lambda v: isinstance(v, int) and v > 0,
    'generate.n_trips': lambda v: isinstance(v, int) and v >= 10000,
    'generate.start_date': lambda v: _is_valid_date(v),
    'generate.end_date': lambda v: _is_valid_date(v),
    'validate.max_bad_row_fraction': lambda v: isinstance(v, (int, float)) and 0 <= v <= 1.0,
    'validate.min_rows': lambda v: isinstance(v, int) and v > 0,
    'split.val_fraction': lambda v: isinstance(v, (int, float)) and 0 < v < 1.0,
    'split.test_fraction': lambda v: isinstance(v, (int, float)) and 0 < v < 1.0,
    'features.n_zone_clusters': lambda v: isinstance(v, int) and v > 0,
}


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


def resolve_mlflow_tracking_uri(value: str | Path) -> str:
    """Resolve an MLflow tracking URI from params.

    Plain paths stay repo-relative; URI-style values such as ``sqlite:///mlflow.db``
    are passed through unchanged.
    """
    text = str(value)
    if "://" in text or text.startswith("file:"):
        return text
    return project_path(text).as_uri()


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


# ────────────────────────────────────────────────────────────────────────────
# Configuration Validation (Week 1 enhancement #3)
# ────────────────────────────────────────────────────────────────────────────

class ConfigValidationError(ValueError):
    """Raised when params.yaml contains invalid values."""
    pass


def _is_valid_date(value: Any) -> bool:
    """Check if value is a valid ISO date string."""
    if not isinstance(value, str):
        return False
    try:
        datetime.fromisoformat(value.replace('Z', '+00:00'))
        return True
    except (ValueError, TypeError):
        return False


def _get_nested(obj: dict, keys: str) -> Any:
    """Get nested dict value using dot notation (e.g., 'generate.n_trips')."""
    parts = keys.split('.')
    current = obj
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def validate_params(params: dict[str, Any] | None = None) -> None:
    """Validate parameters in params.yaml against VALIDATION_RULES.
    
    Raises ConfigValidationError if validation fails.
    """
    if params is None:
        params = load_params()
    
    errors = []
    
    for param_path, validator in VALIDATION_RULES.items():
        value = _get_nested(params, param_path)
        
        if value is None:
            errors.append(f"Missing parameter: {param_path}")
        elif not validator(value):
            errors.append(f"Invalid value for {param_path}: {value}")
    
    # Additional cross-field validations
    if params.get('generate'):
        start = params['generate'].get('start_date')
        end = params['generate'].get('end_date')
        if start and end and _is_valid_date(start) and _is_valid_date(end) and start >= end:
            errors.append(f"Invalid date range: start_date ({start}) >= end_date ({end})")
    
    # Check split fractions don't exceed 1.0
    if params.get('split'):
        val_frac = params['split'].get('val_fraction', 0)
        test_frac = params['split'].get('test_fraction', 0)
        train_frac = 1.0 - val_frac - test_frac
        
        if train_frac <= 0:
            errors.append(f"Invalid split fractions: train_frac={train_frac:.1%} must be > 0")
    
    if errors:
        raise ConfigValidationError(
            f"Configuration validation failed with {len(errors)} error(s):\n" +
            "\n".join(f"  • {e}" for e in errors)
        )


def safe_load_params(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate params.yaml.
    
    This is the safe entry point for param loading. Use this in all stages.
    
    Raises ConfigValidationError if validation fails.
    """
    params = load_params(path)
    validate_params(params)
    return params
