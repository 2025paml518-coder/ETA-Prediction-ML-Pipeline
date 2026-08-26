"""DVC stage: drift simulation and drift report (Week 4 / M5).

The brief asks us to "simulate drift (e.g., festival/rush-hour surge)" and produce
meaningful monitoring signals. Because the training data comes from a documented
generating process we *own*, a drift is a known intervention on that process rather
than a guess: setting ``monitoring.drift.scenario`` to ``festival_surge`` shifts
congestion and trip duration by a controlled amount, so the drift we detect is
ground truth we can check the detector against.

Two independent signals are produced, because the two things that break a deployed
model are different and a monitor that watches only one is blind to the other:

* **Feature (data) drift** — the *inputs* move. Reuses the Week 1 Level 3 machinery
  (KS / chi-squared / mean-shift) against the same training baseline profile, so the
  monitor and the ingestion gate speak the same language.
* **Performance drift** — the *input-to-output relationship* moves. The model is
  scored on a reference (baseline-scenario) window and on the drifted window; the
  ratio of the two MAEs is what a covariate-only detector cannot see (concept drift
  leaves the feature distributions looking normal while error climbs).

The stage is deterministic: same scenario, same seed, same report. That is what
makes the Week 4 artefact reproducible under ``dvc repro``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import mlflow
import pandas as pd

from src.config import ensure_dir, load_params, project_path
from src.data.generate_synthetic import SCENARIOS, generate_trips
from src.data.statistical_validation import compare_to_baseline
from src.data.validate import _derive_trip_metrics
from src.features.build_features import TARGET, FeaturePipeline
from src.models.evaluate import calculate_metrics
from src.utils.io import atomic_write_json, atomic_write_text
from src.utils.lineage import git_commit, git_is_dirty
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _load_model_and_pipeline(params: dict) -> tuple[Any, FeaturePipeline]:
    """Load the exported model and the feature pipeline the API also serves from.

    The standalone export is used rather than the MLflow registry so the stage runs
    in isolation (CI, a fresh clone) without a tracking server, and so it scores
    through byte-identical feature code to the service.
    """
    models_root = project_path(params["paths"]["models"])
    pipeline_dir = models_root / "feature_pipeline"
    model_dir = models_root / "trained" / "model"
    if not pipeline_dir.exists():
        raise FileNotFoundError(
            f"Feature pipeline missing at {pipeline_dir}; run `dvc repro features` first."
        )
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model export missing at {model_dir}; run `dvc repro train` first."
        )
    pipeline = FeaturePipeline.load(pipeline_dir)
    model = mlflow.sklearn.load_model(str(model_dir))
    return model, pipeline


def _simulate_window(scenario_name: str, cfg: dict) -> pd.DataFrame:
    """Generate one clean, drift-controlled window and derive its trip metrics."""
    scenario = SCENARIOS[scenario_name]
    frame = generate_trips(
        n_trips=cfg["n_trips"],
        start_date=cfg["start_date"],
        end_date=cfg["end_date"],
        seed=cfg["seed"],
        scenario=scenario,
    )
    return _derive_trip_metrics(frame)


def _score(model: Any, pipeline: FeaturePipeline, frame: pd.DataFrame) -> dict:
    """Model error on a window, using the same feature path as serving."""
    features = pipeline.transform(frame)
    predictions = model.predict(features)
    return calculate_metrics(frame[TARGET].to_numpy(dtype=float), predictions)


def _performance_signal(reference: dict, current: dict, thr: dict) -> dict:
    ref_mae = reference["mae"]
    cur_mae = current["mae"]
    ratio = round(cur_mae / ref_mae, 4) if ref_mae > 0 else float("inf")

    if ratio >= thr["perf_mae_fail_ratio"] or cur_mae >= thr["perf_mae_fail_abs_min"]:
        status = "fail"
    elif ratio >= thr["perf_mae_warn_ratio"] or cur_mae >= thr["perf_mae_warn_abs_min"]:
        status = "warn"
    else:
        status = "pass"

    return {
        "status": status,
        "reference_mae": ref_mae,
        "current_mae": cur_mae,
        "mae_ratio": ratio,
        "reference": reference,
        "current": current,
    }


def _feature_signal(comparison: dict, thr: dict) -> dict:
    flagged = comparison["failed_columns"] + comparison["warned_columns"]
    n_flagged = len(flagged)
    if n_flagged >= thr["feature_fail_columns"]:
        status = "fail"
    elif n_flagged >= thr["feature_warn_columns"]:
        status = "warn"
    else:
        status = "pass"
    return {
        "status": status,
        "level3_status": comparison["status"],
        "n_pass": comparison["n_pass"],
        "n_warn": comparison["n_warn"],
        "n_fail": comparison["n_fail"],
        "flagged_columns": flagged,
        "checks": comparison["checks"],
    }


def build_drift_report(params: dict) -> dict:
    """Run the full drift simulation and assemble the report payload."""
    cfg = params["monitoring"]["drift"]
    thr = params["monitoring"]["thresholds"]

    profile_path = project_path(
        f"{params['paths']['models']}/data_profile/{params['profile']['output_file']}"
    )
    if not profile_path.exists():
        raise FileNotFoundError(
            f"Baseline profile missing at {profile_path}; run `dvc repro profile` first."
        )
    profile = json.loads(profile_path.read_text(encoding="utf-8"))

    model, pipeline = _load_model_and_pipeline(params)

    reference = _simulate_window(cfg["reference_scenario"], cfg)
    current = _simulate_window(cfg["scenario"], cfg)
    logger.info(
        "Simulated windows | reference=%s current=%s | %s rows each",
        cfg["reference_scenario"],
        cfg["scenario"],
        f"{cfg['n_trips']:,}",
    )

    reference_metrics = _score(model, pipeline, reference)
    current_metrics = _score(model, pipeline, current)
    performance = _performance_signal(reference_metrics, current_metrics, thr)

    comparison = compare_to_baseline(current, profile, params["validate"]["statistical"])
    feature_drift = _feature_signal(comparison, thr)

    logger.info(
        "Performance drift: MAE %.2f -> %.2f (x%.2f) [%s] | feature drift: %s flagged [%s]",
        reference_metrics["mae"],
        current_metrics["mae"],
        performance["mae_ratio"],
        performance["status"],
        len(feature_drift["flagged_columns"]),
        feature_drift["status"],
    )

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "scenario": cfg["scenario"],
        "reference_scenario": cfg["reference_scenario"],
        "n_trips": cfg["n_trips"],
        "seed": cfg["seed"],
        "scenario_description": SCENARIOS[cfg["scenario"]].description,
        "performance": performance,
        "feature_drift": feature_drift,
        "lineage": {"git_commit": git_commit(), "git_dirty": git_is_dirty()},
    }


_STATUS_ICON = {"pass": "🟢 pass", "warn": "🟡 warn", "fail": "🔴 fail"}


def render_markdown(report: dict) -> str:
    perf = report["performance"]
    feat = report["feature_drift"]
    lines = [
        "# Drift Simulation Report",
        "",
        f"- **Generated:** {report['generated_at_utc']}",
        f"- **Scenario:** `{report['scenario']}` vs reference `{report['reference_scenario']}`",
        f"- **Window size:** {report['n_trips']:,} trips each (seed {report['seed']})",
        f"- _{report['scenario_description']}_",
        "",
        "## Performance drift (concept)",
        "",
        "| Window | MAE (min) | RMSE (min) | R² | p90 abs err |",
        "| --- | --- | --- | --- | --- |",
        f"| Reference ({report['reference_scenario']}) | {perf['reference']['mae']:.2f} | "
        f"{perf['reference']['rmse']:.2f} | {perf['reference']['r2']:.3f} | "
        f"{perf['reference']['p90_abs_error']:.2f} |",
        f"| Current ({report['scenario']}) | {perf['current']['mae']:.2f} | "
        f"{perf['current']['rmse']:.2f} | {perf['current']['r2']:.3f} | "
        f"{perf['current']['p90_abs_error']:.2f} |",
        "",
        f"MAE moved **x{perf['mae_ratio']}** under the drift → **{_STATUS_ICON[perf['status']]}**.",
        "",
        "## Feature drift (covariate)",
        "",
        f"Level 3 checks against the training baseline: "
        f"{feat['n_pass']} pass · {feat['n_warn']} warn · {feat['n_fail']} fail "
        f"→ **{_STATUS_ICON[feat['status']]}**.",
        "",
    ]
    if feat["flagged_columns"]:
        lines.append("Flagged columns: " + ", ".join(f"`{c}`" for c in feat["flagged_columns"]))
    else:
        lines.append("No columns flagged.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    params = load_params()
    report = build_drift_report(params)

    out_dir = ensure_dir(f"{params['paths']['reports']}/monitoring")
    atomic_write_json(report, out_dir / "drift_report.json")
    atomic_write_text(render_markdown(report), out_dir / "drift_report.md")
    logger.info("Wrote drift report -> %s", out_dir / "drift_report.json")


if __name__ == "__main__":
    main()
