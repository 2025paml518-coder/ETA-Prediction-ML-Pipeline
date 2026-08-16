"""Model training and experiment tracking (M3).

Four candidates are trained and compared: a median baseline, Ridge, Random Forest
and LightGBM. The baseline exists to answer the question that a table of R² scores
cannot - whether the learned models are worth their operational cost at all.

Three decisions here are deliberate and load-bearing:

1. **Selection happens on validation, never on test.** The test partition is scored
   once, for the final report. A model chosen by its test score turns that score
   into a biased estimate of production error, which is precisely the number the
   report is claiming to give.

2. **Cross-validation is TimeSeriesSplit.** The split stage ordered rows in time on
   purpose. A shuffled KFold inside the search would train on later trips to
   predict earlier ones and quietly reintroduce the leakage that ordering avoided.

3. **Ridge is wrapped in a Pipeline with StandardScaler.** Its L2 penalty is
   scale-dependent, and this feature matrix mixes latitudes near 40.7 with sine
   terms near 1. Scaling inside the pipeline also means the scaler is fitted per
   CV fold and travels with the serialised model, so serving cannot drift from
   training (M2 2.6.2, 2.7.2).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # No display on CI or in a container.

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from mlflow.models import infer_signature
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import ensure_dir, load_params, project_path
from src.features.build_features import FEATURE_COLUMNS, TARGET
from src.models.evaluate import calculate_metrics, format_metrics
from src.models.hyperparameters import SEARCH_SPACES
from src.utils.io import atomic_write_json
from src.utils.lineage import collect as collect_lineage
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

PARTITIONS = ("train", "val", "test")

TRACKED_DATA = (
    "data/processed/train_features.parquet",
    "data/processed/val_features.parquet",
    "data/processed/test_features.parquet",
    "models/feature_pipeline",
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_partitions(processed_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_parquet(processed_dir / f"{name}_features.parquet")
        for name in PARTITIONS
    }


def split_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select features by the canonical contract, and the target from the same row.

    Taking X and y from one frame removes a whole class of alignment bug: reading
    the target from a separate file relies on both keeping identical row order
    forever, and would fail silently the day one of them did not.
    """
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Feature table is missing contract columns: {missing}")
    if TARGET not in frame.columns:
        raise ValueError(f"Feature table has no target column {TARGET!r}")
    return frame.loc[:, list(FEATURE_COLUMNS)], frame[TARGET].astype(float)


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------
def build_estimator(name: str, seed: int):
    if name == "baseline":
        return DummyRegressor(strategy="median")
    if name == "ridge":
        # The scaler belongs inside the pipeline so it is refitted per CV fold.
        return Pipeline([("scaler", StandardScaler()), ("model", Ridge(random_state=seed))])
    if name == "random_forest":
        return RandomForestRegressor(random_state=seed, n_jobs=-1)
    if name == "lightgbm":
        return LGBMRegressor(random_state=seed, n_jobs=-1, verbose=-1)
    raise ValueError(f"Unknown model {name!r}")


def fit_candidate(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict,
    seed: int,
) -> tuple[Any, dict, float | None]:
    """Fit one candidate, tuning it when it has a search space."""
    cfg = params["train"]
    estimator = build_estimator(name, seed)

    if name not in SEARCH_SPACES:
        estimator.fit(X_train, y_train)
        return estimator, {}, None

    splitter = TimeSeriesSplit(n_splits=cfg["cv"]["n_splits"])
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=SEARCH_SPACES[name],
        n_iter=cfg["search"]["n_iter"][name],
        cv=splitter,
        scoring=cfg["search"]["scoring"],
        random_state=cfg["search"]["random_state"],
        # The estimators already parallelise internally; letting the search fan out
        # as well oversubscribes the CPU and slows the whole thing down.
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, float(-search.best_score_)


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------
def _residual_plot(y_true: pd.Series, y_pred: np.ndarray, title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(y_pred, y_true - y_pred, s=4, alpha=0.25)
    axes[0].axhline(0, color="crimson", lw=1)
    axes[0].set_xlabel("Predicted duration (min)")
    axes[0].set_ylabel("Residual (min)")
    axes[0].set_title("Residuals vs prediction")

    axes[1].scatter(y_true, y_pred, s=4, alpha=0.25)
    limit = float(max(y_true.max(), y_pred.max()))
    axes[1].plot([0, limit], [0, limit], color="crimson", lw=1)
    axes[1].set_xlabel("Actual duration (min)")
    axes[1].set_ylabel("Predicted duration (min)")
    axes[1].set_title("Predicted vs actual")

    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _importance_plot(model: Any, title: str, path: Path) -> list[float] | None:
    estimator = model.named_steps["model"] if isinstance(model, Pipeline) else model
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_, dtype=float)).ravel()
    else:
        return None

    order = np.argsort(values)[::-1][:20]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([FEATURE_COLUMNS[i] for i in order][::-1], values[order][::-1])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return values.tolist()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_candidate(
    name: str,
    data: dict[str, tuple[pd.DataFrame, pd.Series]],
    params: dict,
    lineage: dict[str, str],
    artifact_root: Path,
) -> dict:
    """Train, evaluate and log one candidate as an MLflow run."""
    cfg = params["train"]
    seed = params["seed"]
    X_train, y_train = data["train"]

    logger.info("Training %s", name)
    started = time.perf_counter()
    model, best_params, cv_mae = fit_candidate(name, X_train, y_train, params, seed)
    fit_seconds = time.perf_counter() - started

    metrics: dict[str, dict] = {}
    predictions: dict[str, np.ndarray] = {}
    for partition, (X, y) in data.items():
        predict_started = time.perf_counter()
        y_pred = model.predict(X)
        predictions[partition] = y_pred
        metrics[partition] = calculate_metrics(y, y_pred)
        metrics[partition]["predict_ms_per_1k"] = round(
            (time.perf_counter() - predict_started) / max(len(X), 1) * 1e6, 4
        )

    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags(
            {
                **lineage,
                "model_family": name,
                "cv_strategy": cfg["cv"]["strategy"],
                "selection_partition": cfg["selection"]["partition"],
                "feature_contract_size": str(len(FEATURE_COLUMNS)),
            }
        )
        mlflow.log_params(
            {
                "model": name,
                "seed": seed,
                "n_features": len(FEATURE_COLUMNS),
                "n_train_rows": len(X_train),
                "cv_n_splits": cfg["cv"]["n_splits"],
                "search_n_iter": cfg["search"]["n_iter"].get(name, 0),
                **{k: v for k, v in best_params.items()},
            }
        )

        for partition, values in metrics.items():
            for metric_name, value in values.items():
                mlflow.log_metric(f"{partition}_{metric_name}", value)
        if cv_mae is not None:
            mlflow.log_metric("cv_mae", cv_mae)
        mlflow.log_metric("fit_seconds", round(fit_seconds, 3))

        run_artifacts = artifact_root / name
        _residual_plot(
            data["val"][1], predictions["val"], f"{name} - validation", run_artifacts / "residuals.png"
        )
        importances = _importance_plot(
            model, f"{name} - top 20 features", run_artifacts / "feature_importance.png"
        )
        mlflow.log_artifacts(str(run_artifacts), artifact_path="diagnostics")

        # Signature and input example are what let the serving layer validate its
        # payload against the model rather than guessing at column order.
        signature = infer_signature(X_train.head(100), model.predict(X_train.head(100)))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(5),
        )

        result = {
            "model_type": name,
            "run_id": run.info.run_id,
            "params": best_params,
            "cv_mae": cv_mae,
            "fit_seconds": round(fit_seconds, 3),
            "metrics": metrics,
            "feature_importance": importances,
        }

    logger.info(
        "%-14s val %s", name, format_metrics(metrics["val"])
    )
    return result


def select_best(results: list[dict], params: dict) -> dict:
    """Choose the winner on the validation partition."""
    cfg = params["train"]["selection"]
    partition, metric = cfg["partition"], cfg["metric"]
    if partition == "test":
        raise ValueError(
            "Refusing to select on the test partition: it would bias the reported "
            "generalisation estimate."
        )
    chooser = min if cfg["lower_is_better"] else max
    return chooser(results, key=lambda r: r["metrics"][partition][metric])


def run_training(params: dict | None = None) -> tuple[list[dict], dict]:
    params = params or load_params()
    cfg = params["train"]

    mlflow.set_tracking_uri((project_path(cfg["tracking_uri"])).as_uri())
    mlflow.set_experiment(cfg["experiment_name"])

    processed_dir = project_path(params["paths"]["processed"])
    frames = load_partitions(processed_dir)
    data = {name: split_xy(frame) for name, frame in frames.items()}
    logger.info(
        "Loaded %s train / %s val / %s test rows | %s features",
        f"{len(data['train'][0]):,}",
        f"{len(data['val'][0]):,}",
        f"{len(data['test'][0]):,}",
        len(FEATURE_COLUMNS),
    )

    lineage = collect_lineage(TRACKED_DATA)
    if lineage.get("git_dirty") == "true":
        logger.warning("Working tree is dirty; runs will not be fully reproducible.")

    artifact_root = ensure_dir(f"{params['paths']['reports']}/training")
    enabled = [name for name, on in cfg["models"].items() if on]

    results = [
        train_candidate(name, data, params, lineage, artifact_root) for name in enabled
    ]
    best = select_best(results, params)

    models_dir = ensure_dir(f"{params['paths']['models']}/trained")
    atomic_write_json(
        {
            "selected_by": f"{cfg['selection']['partition']}_{cfg['selection']['metric']}",
            "best_model": best["model_type"],
            "best_run_id": best["run_id"],
            "params": best["params"],
            "metrics": best["metrics"],
            "lineage": lineage,
            "candidates": [
                {
                    "model_type": r["model_type"],
                    "run_id": r["run_id"],
                    "val_mae": r["metrics"]["val"]["mae"],
                    "test_mae": r["metrics"]["test"]["mae"],
                }
                for r in results
            ],
        },
        models_dir / "best_model_metadata.json",
    )

    registered = mlflow.register_model(
        model_uri=f"runs:/{best['run_id']}/model",
        name=cfg["registered_model_name"],
    )
    logger.info(
        "Registered %s version %s from run %s",
        cfg["registered_model_name"],
        registered.version,
        best["run_id"],
    )

    atomic_write_json(
        [
            {k: v for k, v in r.items() if k != "feature_importance"}
            for r in results
        ],
        artifact_root / "run_results.json",
    )

    # Flat and shallow so `dvc metrics show` and `dvc metrics diff` can read it.
    atomic_write_json(
        {
            "best_model": best["model_type"],
            **{
                f"{partition}_{metric}": value
                for partition, values in best["metrics"].items()
                for metric, value in values.items()
            },
            **{
                f"{r['model_type']}_val_mae": r["metrics"]["val"]["mae"] for r in results
            },
        },
        project_path(f"{params['paths']['reports']}/metrics.json"),
    )
    return results, best


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare ETA models.")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    results, best = run_training(params)

    from src.models.compare import compare_models

    compare_models(results, best, params)


if __name__ == "__main__":
    main()
