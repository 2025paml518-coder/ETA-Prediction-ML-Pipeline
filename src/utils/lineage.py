"""Run provenance.

M2 section 2.9 asks that lineage be stored in the experiment tracker alongside the
model. Without it, a logged metric is an orphan: you know a run scored 3.9 MAE but
not which commit or which dataset produced it, so you cannot reproduce it.

Everything here degrades gracefully - a missing git binary or an absent dvc.lock
yields "unknown" rather than failing a training run.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

from src.config import PROJECT_ROOT

UNKNOWN = "unknown"


def _git(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else UNKNOWN
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN


def git_commit() -> str:
    return _git("rev-parse", "HEAD")


def git_branch() -> str:
    return _git("rev-parse", "--abbrev-ref", "HEAD")


def git_is_dirty() -> bool:
    """True when tracked files differ from HEAD.

    A dirty tree means the commit hash alone does not identify the code that ran,
    so the run is only partially reproducible. Recording it is more useful than
    refusing to train.
    """
    status = _git("status", "--porcelain")
    return bool(status) and status != UNKNOWN


def dvc_output_hashes(paths: tuple[str, ...]) -> dict[str, str]:
    """Look up the recorded md5 of specific DVC outputs from dvc.lock."""
    lock_path = PROJECT_ROOT / "dvc.lock"
    if not lock_path.exists():
        return {}

    try:
        lock = yaml.safe_load(lock_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}

    wanted = set(paths)
    hashes: dict[str, str] = {}
    for stage in (lock.get("stages") or {}).values():
        for out in stage.get("outs", []) or []:
            if out.get("path") in wanted:
                hashes[out["path"]] = out.get("md5") or out.get("hash", UNKNOWN)
    return hashes


def collect(data_paths: tuple[str, ...] = ()) -> dict[str, str]:
    """Assemble the provenance tags attached to every MLflow run."""
    tags = {
        "git_commit": git_commit(),
        "git_branch": git_branch(),
        "git_dirty": str(git_is_dirty()).lower(),
    }
    for path, digest in dvc_output_hashes(data_paths).items():
        tags[f"data_md5.{Path(path).name}"] = digest
    return tags
