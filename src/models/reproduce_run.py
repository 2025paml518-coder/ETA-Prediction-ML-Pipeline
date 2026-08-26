"""Reproduce a tracked run from its logged configuration.

The M3 rubric asks for the ability to reproduce a chosen run from what was logged.
That is a stronger claim than "the code is deterministic": it means the tracking
record is complete enough to stand alone. This script tests that claim by reading
*only* MLflow - parameters, tags and the recorded dataset hashes - refitting, and
comparing the resulting metrics against what was originally recorded.

It fails loudly on divergence. A silent near-match is exactly the kind of drift
that makes an experiment log untrustworthy months later.
"""

from __future__ import annotations

import argparse
import sys

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from src.config import load_params, project_path, resolve_mlflow_tracking_uri
from src.models.evaluate import calculate_metrics
from src.models.hyperparameters import SEARCH_SPACES
from src.models.train import PARTITIONS, TRACKED_DATA, build_estimator, split_xy
from src.utils.io import atomic_write_json
from src.utils.lineage import dvc_output_hashes
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Logged under log_params for bookkeeping rather than as estimator arguments.
NON_ESTIMATOR_PARAMS = {
    "model",
    "seed",
    "n_features",
    "n_train_rows",
    "cv_n_splits",
    "search_n_iter",
}


def _coerce(value: str):
    """MLflow stores every parameter as a string; recover the original type."""
    if value in {"None", "null", ""}:
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def recorded_config(run) -> tuple[str, int, dict]:
    params = run.data.params
    if "model" not in params:
        raise SystemExit(f"Run {run.info.run_id} has no 'model' parameter; cannot reproduce.")

    model_name = params["model"]
    seed = int(params.get("seed", 42))
    estimator_params = {
        key: _coerce(value)
        for key, value in params.items()
        if key not in NON_ESTIMATOR_PARAMS
    }
    return model_name, seed, estimator_params


def verify_data_lineage(run) -> list[str]:
    """Compare the dataset hashes on disk now against those recorded by the run."""
    current = dvc_output_hashes(TRACKED_DATA)
    problems = []
    for path, digest in current.items():
        tag = f"data_md5.{path.split('/')[-1]}"
        recorded = run.data.tags.get(tag)
        if recorded and recorded != digest:
            problems.append(f"{tag}: run recorded {recorded[:12]}, workspace has {digest[:12]}")
    return problems


def reproduce(run_id: str, tolerance: float, params: dict) -> dict:
    client = MlflowClient()
    run = client.get_run(run_id)

    model_name, seed, estimator_params = recorded_config(run)
    logger.info("Reproducing run %s | model=%s | seed=%s", run_id[:8], model_name, seed)

    lineage_problems = verify_data_lineage(run)
    for problem in lineage_problems:
        logger.warning("Dataset has changed since the run: %s", problem)
    if run.data.tags.get("git_dirty") == "true":
        logger.warning("Original run was recorded from a dirty working tree.")

    processed_dir = project_path(params["paths"]["processed"])
    data = {
        name: split_xy(pd.read_parquet(processed_dir / f"{name}_features.parquet"))
        for name in PARTITIONS
    }

    estimator = build_estimator(model_name, seed)
    if model_name in SEARCH_SPACES and estimator_params:
        # Refit directly on the winning hyperparameters: the search only existed to
        # find them, and re-running it would prove nothing beyond its own determinism.
        estimator.set_params(**estimator_params)
    estimator.fit(*data["train"])

    comparison = {}
    max_delta = 0.0
    for partition in PARTITIONS:
        X, y = data[partition]
        fresh = calculate_metrics(y, estimator.predict(X))
        for metric in ("mae", "rmse", "r2"):
            key = f"{partition}_{metric}"
            original = run.data.metrics.get(key)
            if original is None:
                continue
            delta = abs(fresh[metric] - original)
            max_delta = max(max_delta, delta)
            comparison[key] = {
                "original": round(original, 6),
                "reproduced": round(fresh[metric], 6),
                "delta": round(delta, 8),
            }

    reproduced = max_delta <= tolerance
    return {
        "run_id": run_id,
        "model": model_name,
        "params": estimator_params,
        "tolerance": tolerance,
        "max_abs_delta": round(max_delta, 8),
        "reproduced": reproduced,
        "git_commit_of_run": run.data.tags.get("git_commit", "unknown"),
        "run_recorded_from_dirty_tree": run.data.tags.get("git_dirty") == "true",
        "dataset_drift": lineage_problems,
        "metrics": comparison,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce an MLflow run from its log.")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--run-id", default=None, help="Defaults to the selected best run.")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    params = load_params(args.params)
    mlflow.set_tracking_uri(resolve_mlflow_tracking_uri(params["train"]["tracking_uri"]))

    run_id = args.run_id
    if run_id is None:
        metadata_path = project_path(f"{params['paths']['models']}/trained/best_model_metadata.json")
        if not metadata_path.exists():
            raise SystemExit("No --run-id given and no trained model metadata found.")
        import json

        run_id = json.loads(metadata_path.read_text(encoding="utf-8"))["best_run_id"]

    result = reproduce(run_id, args.tolerance, params)

    destination = project_path(
        args.output or f"{params['paths']['reports']}/training/reproducibility_check.json"
    )
    atomic_write_json(result, destination)

    for key, value in result["metrics"].items():
        logger.info(
            "%-14s original=%.6f reproduced=%.6f delta=%.2e",
            key,
            value["original"],
            value["reproduced"],
            value["delta"],
        )

    if result["reproduced"]:
        logger.info(
            "Run %s reproduced within %.1e (max delta %.2e)",
            run_id[:8],
            args.tolerance,
            result["max_abs_delta"],
        )
    else:
        logger.error(
            "Run %s did NOT reproduce: max delta %.2e exceeds tolerance %.1e",
            run_id[:8],
            result["max_abs_delta"],
            args.tolerance,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
