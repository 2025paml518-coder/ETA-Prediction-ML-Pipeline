"""DVC stage: retraining trigger decision (Week 4 / M5).

The brief asks for "a sound retraining trigger design", not a retrain button. This
stage reads the drift report and, when available, recent labelled traffic from the
serving log, turns each into an independent pass / warn / fail signal, and applies
one explicit rule to reach a verdict: **observe** or **retrain**.

Three signals, deliberately independent:

* **performance** — simulated concept drift (model MAE on the drifted window vs the
  reference window). Reproducible, always present.
* **feature_drift** — simulated covariate drift (Level 3 checks vs the training
  baseline). Reproducible, always present.
* **live_error** — the mean absolute error of recently *labelled* predictions from
  the real serving log. Present only once enough feedback has arrived; below the
  ``min_labelled`` floor it abstains rather than voting on thin data.

Keeping the two simulated signals as the reproducible core means ``dvc repro``
always produces a decision; the live signal, when traffic exists, is what lets the
same rule fire on real observed error rather than only on a rehearsal.

The rule is intentionally boring: retrain when at least ``fail_signals_to_trigger``
signals reach 'fail'. A boring, written-down rule is auditable; a clever one that
lives in someone's head is not.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from src.config import ensure_dir, load_params, project_path
from src.utils.io import atomic_write_json, atomic_write_text
from src.utils.lineage import git_commit, git_is_dirty
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _live_error_signal(params: dict) -> dict:
    """Mean absolute error over recently labelled predictions from the serving log.

    Never raises: a missing or empty store yields an abstaining signal, because a
    cold monitoring database is not a reason to fail the pipeline.
    """
    cfg = params["monitoring"]["lookback"]
    thr = params["monitoring"]["thresholds"]
    signal: dict = {"status": "insufficient", "enabled": bool(cfg["use_live_feedback"])}

    if not cfg["use_live_feedback"]:
        signal["reason"] = "live feedback disabled in params"
        return signal

    db_path = project_path(f"{params['paths']['monitoring']}/predictions.db")
    if not db_path.exists():
        signal["reason"] = "no serving log found"
        signal["labelled"] = 0
        return signal

    try:
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT COUNT(*) AS labelled, AVG(absolute_error_min) AS mae
              FROM (
                    SELECT absolute_error_min
                      FROM predictions
                     WHERE actual_duration_min IS NOT NULL
                  ORDER BY feedback_at_utc DESC
                     LIMIT ?
              )
            """,
            (cfg["max_rows"],),
        ).fetchone()
        connection.close()
    except sqlite3.Error as exc:
        signal["reason"] = f"serving log unreadable: {exc}"
        signal["labelled"] = 0
        return signal

    labelled = int(row["labelled"] or 0)
    signal["labelled"] = labelled
    if labelled < cfg["min_labelled"]:
        signal["reason"] = f"only {labelled} labelled rows (< {cfg['min_labelled']})"
        return signal

    mae = round(float(row["mae"]), 4)
    signal["live_mae"] = mae
    if mae >= thr["live_mae_fail_min"]:
        signal["status"] = "fail"
    elif mae >= thr["live_mae_warn_min"]:
        signal["status"] = "warn"
    else:
        signal["status"] = "pass"
    return signal


def decide(drift_report: dict, live_signal: dict, params: dict) -> dict:
    """Combine the signals into an observe / retrain decision."""
    trigger_at = params["monitoring"]["retrain"]["fail_signals_to_trigger"]

    signals = {
        "performance": drift_report["performance"]["status"],
        "feature_drift": drift_report["feature_drift"]["status"],
        "live_error": live_signal["status"],
    }
    fail_signals = [name for name, status in signals.items() if status == "fail"]
    warn_signals = [name for name, status in signals.items() if status == "warn"]

    triggered = len(fail_signals) >= trigger_at
    decision = "retrain" if triggered else "observe"

    reasons: list[str] = []
    perf = drift_report["performance"]
    feat = drift_report["feature_drift"]
    reasons.append(
        f"performance [{perf['status']}]: MAE {perf['reference_mae']:.2f} → "
        f"{perf['current_mae']:.2f} min (x{perf['mae_ratio']}) under "
        f"`{drift_report['scenario']}`."
    )
    reasons.append(
        f"feature_drift [{feat['status']}]: {len(feat['flagged_columns'])} column(s) "
        f"flagged by Level 3 checks"
        + (": " + ", ".join(feat["flagged_columns"]) if feat["flagged_columns"] else "")
        + "."
    )
    if live_signal["status"] in {"pass", "warn", "fail"}:
        reasons.append(
            f"live_error [{live_signal['status']}]: {live_signal['live_mae']:.2f} min MAE "
            f"over {live_signal['labelled']} labelled prediction(s)."
        )
    else:
        reasons.append(
            f"live_error [insufficient]: {live_signal.get('reason', 'no data')} — "
            "signal abstained."
        )

    if triggered:
        recommendation = (
            f"RETRAIN: {len(fail_signals)} signal(s) failed "
            f"({', '.join(fail_signals)}), meeting the trigger of {trigger_at}. "
            "Regenerate training data covering the new regime and rerun `dvc repro train`."
        )
    elif warn_signals:
        recommendation = (
            f"OBSERVE: no failing signal, but {', '.join(warn_signals)} at warn level. "
            "Increase feedback sampling and re-evaluate before the next window."
        )
    else:
        recommendation = "OBSERVE: all signals within tolerance; the model is serving well."

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "decision": decision,
        "triggered": triggered,
        "scenario": drift_report["scenario"],
        "fail_signals_to_trigger": trigger_at,
        "n_fail_signals": len(fail_signals),
        "failing_signals": fail_signals,
        "warning_signals": warn_signals,
        "signals": {
            "performance": drift_report["performance"],
            "feature_drift": {
                "status": feat["status"],
                "flagged_columns": feat["flagged_columns"],
                "n_warn": feat["n_warn"],
                "n_fail": feat["n_fail"],
            },
            "live_error": live_signal,
        },
        "reasons": reasons,
        "recommendation": recommendation,
        "lineage": {"git_commit": git_commit(), "git_dirty": git_is_dirty()},
    }


_DECISION_ICON = {"retrain": "🔴 RETRAIN", "observe": "🟢 OBSERVE"}


def render_markdown(decision: dict) -> str:
    lines = [
        "# Retraining Trigger Decision",
        "",
        f"## Decision: {_DECISION_ICON[decision['decision']]}",
        "",
        f"- **Generated:** {decision['generated_at_utc']}",
        f"- **Drift scenario:** `{decision['scenario']}`",
        f"- **Failing signals:** {decision['n_fail_signals']} "
        f"(trigger at {decision['fail_signals_to_trigger']})",
        "",
        f"> {decision['recommendation']}",
        "",
        "## Signals",
        "",
        "| Signal | Status |",
        "| --- | --- |",
        f"| performance (concept drift) | {decision['signals']['performance']['status']} |",
        f"| feature_drift (covariate) | {decision['signals']['feature_drift']['status']} |",
        f"| live_error (serving log) | {decision['signals']['live_error']['status']} |",
        "",
        "## Why",
        "",
    ]
    lines.extend(f"- {reason}" for reason in decision["reasons"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    params = load_params()
    report_dir = ensure_dir(f"{params['paths']['reports']}/monitoring")
    drift_path = report_dir / "drift_report.json"
    if not drift_path.exists():
        raise FileNotFoundError(
            f"Drift report missing at {drift_path}; run `dvc repro drift_simulation` first."
        )
    drift_report = json.loads(drift_path.read_text(encoding="utf-8"))

    live_signal = _live_error_signal(params)
    decision = decide(drift_report, live_signal, params)

    atomic_write_json(decision, report_dir / "retraining_decision.json")
    atomic_write_text(render_markdown(decision), report_dir / "retraining_decision.md")
    logger.info(
        "Decision: %s (%s failing signal(s)) -> %s",
        decision["decision"].upper(),
        decision["n_fail_signals"],
        report_dir / "retraining_decision.json",
    )


if __name__ == "__main__":
    main()
