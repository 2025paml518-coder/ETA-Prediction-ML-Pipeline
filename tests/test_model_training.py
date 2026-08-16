"""Model training and evaluation contracts (M3)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import load_params, project_path
from src.features.build_features import FEATURE_COLUMNS, TARGET
from src.models.evaluate import calculate_metrics, format_metrics
from src.models.train import build_estimator, select_best, split_xy

PROCESSED = project_path("data/processed")


@pytest.fixture(scope="module")
def train_frame() -> pd.DataFrame:
    path = PROCESSED / "train_features.parquet"
    if not path.exists():
        pytest.skip("Feature tables not built; run `dvc repro` first")
    return pd.read_parquet(path)


class TestMetrics:
    def test_calculate_metrics(self):
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([12, 18, 32, 38, 52])
        metrics = calculate_metrics(y_true, y_pred)

        assert set(metrics) == {"mae", "rmse", "r2", "mape", "p90_abs_error"}
        assert metrics["mae"] == pytest.approx(2.0)
        assert metrics["rmse"] == pytest.approx(2.0)
        assert metrics["r2"] > 0.9

    def test_perfect_prediction_scores_zero_error(self):
        y = np.array([5.0, 10.0, 15.0])
        metrics = calculate_metrics(y, y.copy())
        assert metrics["mae"] == 0
        assert metrics["mape"] == 0
        assert metrics["r2"] == pytest.approx(1.0)

    def test_mape_ignores_zero_targets(self):
        """An unguarded MAPE divides by zero and poisons every comparison."""
        y_true = np.array([0.0, 10.0, 20.0])
        y_pred = np.array([1.0, 11.0, 22.0])
        metrics = calculate_metrics(y_true, y_pred)
        assert np.isfinite(metrics["mape"])
        assert metrics["mape"] == pytest.approx(10.0)

    def test_p90_captures_the_error_tail(self):
        """A mean hides a bad tail; p90 is what the worst-served riders feel."""
        y_true = np.zeros(100)
        y_pred = np.concatenate([np.zeros(85), np.full(15, 50.0)])
        metrics = calculate_metrics(y_true, y_pred)
        assert metrics["mae"] == pytest.approx(7.5)
        assert metrics["p90_abs_error"] == pytest.approx(50.0)
        assert metrics["p90_abs_error"] > metrics["mae"]

    def test_format_metrics(self):
        formatted = format_metrics(
            {"mae": 12.34, "rmse": 15.67, "r2": 0.882, "mape": 5.21, "p90_abs_error": 20.0}
        )
        assert "12.3400" in formatted
        assert "0.8820" in formatted


class TestFeatureContract:
    def test_split_xy_returns_contract_columns_in_order(self, train_frame):
        X, y = split_xy(train_frame)
        assert list(X.columns) == list(FEATURE_COLUMNS)
        assert len(X) == len(y)
        assert y.name == TARGET

    def test_split_xy_takes_target_from_the_same_frame(self, train_frame):
        """X and y must come from one frame; positional joins across files drift."""
        X, y = split_xy(train_frame)
        pd.testing.assert_series_equal(
            y, train_frame[TARGET].astype(float), check_names=False
        )
        assert X.index.equals(y.index)

    def test_split_xy_rejects_a_frame_missing_contract_columns(self, train_frame):
        broken = train_frame.drop(columns=[FEATURE_COLUMNS[0]])
        with pytest.raises(ValueError, match="missing contract columns"):
            split_xy(broken)

    def test_split_xy_rejects_a_frame_without_the_target(self, train_frame):
        with pytest.raises(ValueError, match="no target column"):
            split_xy(train_frame.drop(columns=[TARGET]))

    def test_features_are_finite(self, train_frame):
        X, _ = split_xy(train_frame)
        assert X.notna().all().all()
        assert np.isfinite(X.to_numpy()).all()


class TestEstimators:
    def test_ridge_is_wrapped_in_a_scaler(self):
        """Ridge's L2 penalty is scale-dependent, and this matrix is not scaled."""
        estimator = build_estimator("ridge", seed=42)
        assert isinstance(estimator, Pipeline)
        assert isinstance(estimator.named_steps["scaler"], StandardScaler)
        assert isinstance(estimator.named_steps["model"], Ridge)

    def test_baseline_predicts_the_median(self):
        estimator = build_estimator("baseline", seed=42)
        assert isinstance(estimator, DummyRegressor)

        y = pd.Series([1.0, 2.0, 3.0, 100.0])
        X = pd.DataFrame({"a": range(4)})
        estimator.fit(X, y)
        assert estimator.predict(X)[0] == pytest.approx(2.5)

    def test_unknown_model_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown model"):
            build_estimator("magic", seed=42)


def _result(name: str, val_mae: float, test_mae: float) -> dict:
    return {
        "model_type": name,
        "run_id": f"run-{name}",
        "fit_seconds": 1.0,
        "metrics": {
            "train": {"mae": val_mae},
            "val": {"mae": val_mae},
            "test": {"mae": test_mae},
        },
    }


class TestSelection:
    def test_winner_is_chosen_on_validation_not_test(self):
        """The whole point: a model that only looks best on test must not win."""
        params = load_params()
        results = [
            _result("lightgbm", val_mae=3.0, test_mae=9.9),
            _result("ridge", val_mae=8.0, test_mae=1.0),
        ]
        best = select_best(results, params)
        assert best["model_type"] == "lightgbm"

    def test_selecting_on_test_is_refused(self):
        params = copy.deepcopy(load_params())
        params["train"]["selection"]["partition"] = "test"
        with pytest.raises(ValueError, match="Refusing to select on the test partition"):
            select_best([_result("ridge", 1.0, 1.0)], params)

    def test_configured_selection_uses_validation(self):
        selection = load_params()["train"]["selection"]
        assert selection["partition"] == "val"
        assert selection["lower_is_better"] is True


class TestTemporalDiscipline:
    def test_cross_validation_is_time_aware(self):
        """Shuffled folds would undo the temporal split made in Week 1."""
        assert load_params()["train"]["cv"]["strategy"] == "time_series"

    def test_target_distribution(self, train_frame):
        y = train_frame[TARGET]
        assert y.std() > 1.0
        assert y.min() >= 1
        assert y.max() <= 300


class TestTrainedArtifacts:
    def test_metadata_records_how_the_model_was_chosen(self):
        path = Path("models/trained/best_model_metadata.json")
        if not path.exists():
            pytest.skip("Model not trained yet")

        metadata = json.loads(path.read_text(encoding="utf-8"))
        assert metadata["selected_by"] == "val_mae"
        assert metadata["best_model"]
        assert metadata["best_run_id"]
        assert "lineage" in metadata
        assert metadata["lineage"]["git_commit"]

    def test_best_model_really_is_the_validation_winner(self):
        path = Path("models/trained/best_model_metadata.json")
        if not path.exists():
            pytest.skip("Model not trained yet")

        metadata = json.loads(path.read_text(encoding="utf-8"))
        winner = min(metadata["candidates"], key=lambda c: c["val_mae"])
        assert metadata["best_model"] == winner["model_type"]

    def test_learned_models_beat_the_median_baseline(self):
        path = Path("models/trained/best_model_metadata.json")
        if not path.exists():
            pytest.skip("Model not trained yet")

        metadata = json.loads(path.read_text(encoding="utf-8"))
        candidates = {c["model_type"]: c["val_mae"] for c in metadata["candidates"]}
        if "baseline" not in candidates:
            pytest.skip("Baseline not enabled")
        assert metadata["metrics"]["val"]["mae"] < candidates["baseline"]
