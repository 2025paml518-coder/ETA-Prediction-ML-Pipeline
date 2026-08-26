"""Week 4 (M5) monitoring: drift signals and the retraining trigger.

The trigger rule is the graded artefact ("a sound retraining trigger design"), so it
is tested as a pure function against hand-built signals rather than only end to end:
every branch of observe / retrain is pinned, independent of whether trained model
artefacts happen to be present.
"""

from __future__ import annotations

import copy
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.config import load_params
from src.monitoring import drift_simulation as ds
from src.monitoring import retraining_decision as rd
from src.monitoring.prediction_log import SCHEMA


@pytest.fixture(scope="module")
def cfg() -> dict:
    return copy.deepcopy(load_params())


def _metrics(mae: float) -> dict:
    return {"mae": mae, "rmse": mae, "r2": 0.9, "mape": 10.0, "p90_abs_error": mae * 2}


def _drift_report(perf_status="pass", feat_status="pass", flagged=None) -> dict:
    flagged = flagged or []
    return {
        "scenario": "festival_surge",
        "reference_scenario": "baseline",
        "performance": {
            "status": perf_status,
            "reference_mae": 4.0,
            "current_mae": 5.0,
            "mae_ratio": 1.25,
            "reference": _metrics(4.0),
            "current": _metrics(5.0),
        },
        "feature_drift": {
            "status": feat_status,
            "flagged_columns": flagged,
            "n_warn": len(flagged),
            "n_fail": 0,
        },
    }


class TestPerformanceSignal:
    def test_pass_when_stable(self, cfg):
        thr = cfg["monitoring"]["thresholds"]
        result = ds._performance_signal(_metrics(4.0), _metrics(4.2), thr)
        assert result["status"] == "pass"

    def test_warn_on_ratio(self, cfg):
        thr = cfg["monitoring"]["thresholds"]
        result = ds._performance_signal(_metrics(4.0), _metrics(4.8), thr)
        assert result["status"] == "warn"
        assert result["mae_ratio"] == pytest.approx(1.2)

    def test_fail_on_ratio(self, cfg):
        thr = cfg["monitoring"]["thresholds"]
        result = ds._performance_signal(_metrics(4.0), _metrics(5.4), thr)
        assert result["status"] == "fail"

    def test_fail_on_absolute_floor(self, cfg):
        # Ratio is tiny (1.025) but absolute MAE breaches the 8.0 min floor.
        thr = cfg["monitoring"]["thresholds"]
        result = ds._performance_signal(_metrics(8.0), _metrics(8.2), thr)
        assert result["status"] == "fail"


class TestFeatureSignal:
    def _comparison(self, warned, failed) -> dict:
        return {
            "status": "fail" if failed else ("warn" if warned else "pass"),
            "n_pass": 5,
            "n_warn": len(warned),
            "n_fail": len(failed),
            "warned_columns": warned,
            "failed_columns": failed,
            "checks": [],
        }

    def test_pass_when_none_flagged(self, cfg):
        thr = cfg["monitoring"]["thresholds"]
        result = ds._feature_signal(self._comparison([], []), thr)
        assert result["status"] == "pass"

    def test_warn_on_one_flag(self, cfg):
        thr = cfg["monitoring"]["thresholds"]
        result = ds._feature_signal(self._comparison(["traffic_index"], []), thr)
        assert result["status"] == "warn"
        assert result["flagged_columns"] == ["traffic_index"]

    def test_fail_on_two_flags(self, cfg):
        thr = cfg["monitoring"]["thresholds"]
        result = ds._feature_signal(self._comparison(["traffic_index"], ["trip_duration_min"]), thr)
        assert result["status"] == "fail"


class TestDecision:
    def test_observe_when_all_pass(self, cfg):
        decision = rd.decide(_drift_report(), {"status": "insufficient", "reason": "cold"}, cfg)
        assert decision["decision"] == "observe"
        assert decision["triggered"] is False
        assert decision["n_fail_signals"] == 0

    def test_retrain_on_single_failing_signal(self, cfg):
        report = _drift_report(perf_status="fail")
        decision = rd.decide(report, {"status": "insufficient", "reason": "cold"}, cfg)
        assert decision["decision"] == "retrain"
        assert decision["triggered"] is True
        assert decision["failing_signals"] == ["performance"]

    def test_warn_stays_observe_but_is_reported(self, cfg):
        report = _drift_report(feat_status="warn", flagged=["traffic_index"])
        decision = rd.decide(report, {"status": "pass", "live_mae": 3.0, "labelled": 200}, cfg)
        assert decision["decision"] == "observe"
        assert "feature_drift" in decision["warning_signals"]

    def test_live_failure_triggers_retrain(self, cfg):
        live = {"status": "fail", "live_mae": 9.5, "labelled": 300}
        decision = rd.decide(_drift_report(), live, cfg)
        assert decision["decision"] == "retrain"
        assert decision["failing_signals"] == ["live_error"]

    def test_insufficient_live_never_counts_as_fail(self, cfg):
        live = {"status": "insufficient", "reason": "only 3 labelled rows (< 50)", "labelled": 3}
        decision = rd.decide(_drift_report(), live, cfg)
        assert decision["decision"] == "observe"
        assert "live_error" not in decision["failing_signals"]

    def test_requires_two_signals_when_configured(self, cfg):
        cfg = copy.deepcopy(cfg)
        cfg["monitoring"]["retrain"]["fail_signals_to_trigger"] = 2
        report = _drift_report(perf_status="fail")  # only one failing signal
        decision = rd.decide(report, {"status": "insufficient"}, cfg)
        assert decision["decision"] == "observe"


class TestLiveErrorSignal:
    def _make_db(self, path: Path, rows: list[tuple[float, float]]) -> None:
        connection = sqlite3.connect(path)
        connection.executescript(SCHEMA)
        now = datetime.now(UTC).isoformat(timespec="seconds")
        connection.executemany(
            """
            INSERT INTO predictions (
                request_id, predicted_at_utc, pickup_datetime, model_name, model_version,
                predicted_duration_min, distance_km, features_json,
                actual_duration_min, absolute_error_min, feedback_at_utc
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            [
                (f"r{i}", now, now, "m", "v1", pred, 5.0, "{}", actual, abs(pred - actual), now)
                for i, (pred, actual) in enumerate(rows)
            ],
        )
        connection.commit()
        connection.close()

    def _params_pointing_at(self, cfg, tmp_path) -> dict:
        cfg = copy.deepcopy(cfg)
        cfg["paths"]["monitoring"] = str(tmp_path)
        return cfg

    def test_missing_store_abstains(self, cfg, tmp_path):
        params = self._params_pointing_at(cfg, tmp_path)
        signal = rd._live_error_signal(params)
        assert signal["status"] == "insufficient"
        assert signal["labelled"] == 0

    def test_disabled_abstains(self, cfg, tmp_path):
        params = self._params_pointing_at(cfg, tmp_path)
        params["monitoring"]["lookback"]["use_live_feedback"] = False
        signal = rd._live_error_signal(params)
        assert signal["status"] == "insufficient"
        assert "disabled" in signal["reason"]

    def test_too_few_labelled_abstains(self, cfg, tmp_path):
        params = self._params_pointing_at(cfg, tmp_path)
        params["monitoring"]["lookback"]["min_labelled"] = 50
        self._make_db(tmp_path / "predictions.db", [(10.0, 12.0)] * 5)
        signal = rd._live_error_signal(params)
        assert signal["status"] == "insufficient"
        assert signal["labelled"] == 5

    def test_high_error_fails(self, cfg, tmp_path):
        params = self._params_pointing_at(cfg, tmp_path)
        params["monitoring"]["lookback"]["min_labelled"] = 10
        # Every prediction is 9 minutes off, above the 8.0 fail floor.
        self._make_db(tmp_path / "predictions.db", [(10.0, 19.0)] * 60)
        signal = rd._live_error_signal(params)
        assert signal["status"] == "fail"
        assert signal["live_mae"] == pytest.approx(9.0)

    def test_low_error_passes(self, cfg, tmp_path):
        params = self._params_pointing_at(cfg, tmp_path)
        params["monitoring"]["lookback"]["min_labelled"] = 10
        self._make_db(tmp_path / "predictions.db", [(10.0, 11.0)] * 60)
        signal = rd._live_error_signal(params)
        assert signal["status"] == "pass"


class TestReportRendering:
    def test_drift_markdown_has_sections(self):
        report = {
            "generated_at_utc": "2026-08-26T00:00:00+00:00",
            "scenario": "festival_surge",
            "reference_scenario": "baseline",
            "n_trips": 20000,
            "seed": 202,
            "scenario_description": "City-wide event.",
            "performance": {
                "status": "fail",
                "mae_ratio": 1.31,
                "reference": _metrics(4.0),
                "current": _metrics(5.3),
            },
            "feature_drift": {
                "status": "warn",
                "n_pass": 10,
                "n_warn": 1,
                "n_fail": 0,
                "flagged_columns": ["traffic_index"],
            },
        }
        md = ds.render_markdown(report)
        assert "# Drift Simulation Report" in md
        assert "festival_surge" in md
        assert "Performance drift" in md
        assert "traffic_index" in md

    def test_decision_markdown_states_verdict(self, cfg):
        decision = rd.decide(_drift_report(perf_status="fail"), {"status": "insufficient"}, cfg)
        md = rd.render_markdown(decision)
        assert "RETRAIN" in md
        assert "## Signals" in md
        assert "## Why" in md


@pytest.mark.skipif(
    not (Path("models/trained/model").exists() and Path("models/feature_pipeline").exists()),
    reason="requires trained model + feature pipeline (run `dvc repro train`)",
)
class TestDriftSimulationIntegration:
    def test_festival_surge_degrades_performance(self, cfg):
        params = copy.deepcopy(cfg)
        params["monitoring"]["drift"]["n_trips"] = 1500
        params["monitoring"]["drift"]["scenario"] = "festival_surge"
        report = ds.build_drift_report(params)

        assert set(report) >= {"performance", "feature_drift", "scenario"}
        perf = report["performance"]
        # A model trained on the baseline regime under-predicts a 28% duration surge,
        # so error on the drifted window must exceed the reference window's.
        assert perf["current_mae"] >= perf["reference_mae"]
        assert perf["status"] in {"warn", "fail"}
