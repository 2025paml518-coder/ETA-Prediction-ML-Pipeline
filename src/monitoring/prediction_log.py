"""Durable prediction log.

Every served prediction is recorded with the features that produced it, the model
version responsible, and latency. Actual durations arrive later through /feedback
and are joined onto the original row by request id.

SQLite rather than a log file because the Week 4 monitoring job needs to query by
time window and join predictions to outcomes; grepping JSON lines for that is a
worse version of a database. It is also the store the chapter's "log predictions
to a datastore" lab step asks for, and it needs no extra service in the container.

Writes must never fail a request: a monitoring outage is not a serving outage, so
errors here are logged and swallowed.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from src.config import load_params, project_path
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    request_id            TEXT PRIMARY KEY,
    predicted_at_utc      TEXT NOT NULL,
    pickup_datetime       TEXT NOT NULL,
    model_name            TEXT NOT NULL,
    model_version         TEXT NOT NULL,
    predicted_duration_min REAL NOT NULL,
    distance_km           REAL NOT NULL,
    traffic_index         REAL,
    weather_condition     TEXT,
    weather_imputed       INTEGER NOT NULL DEFAULT 0,
    latency_ms            REAL,
    features_json         TEXT NOT NULL,
    actual_duration_min   REAL,
    absolute_error_min    REAL,
    feedback_at_utc       TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_time ON predictions(predicted_at_utc);
CREATE INDEX IF NOT EXISTS idx_predictions_actual ON predictions(actual_duration_min);
"""


class PredictionLog:
    def __init__(self, path: str | Path | None = None) -> None:
        params = load_params()
        self.path = Path(path) if path else project_path(
            f"{params['paths'].get('monitoring', 'monitoring/data')}/predictions.db"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # One connection per thread, reused. Connecting per request cost more than the
        # model inference did, and re-running the pragmas each time made it worse.
        self._local = threading.local()
        self._initialise()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
            connection.row_factory = sqlite3.Row
            # WAL lets readers run alongside the writer, and NORMAL avoids an fsync per
            # commit. The worst case on power loss is losing the last few logged
            # predictions, which is an acceptable trade for not taxing every request.
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            self._local.connection = connection

        yield connection
        connection.commit()

    def close(self) -> None:
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            self._local.connection = None

    def _initialise(self) -> None:
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def record(self, rows: list[dict]) -> None:
        """Persist served predictions. Never raises into the request path."""
        if not rows:
            return
        now = datetime.now(UTC).isoformat(timespec="seconds")
        payload = [
            (
                row["request_id"],
                now,
                str(row["pickup_datetime"]),
                row["model_name"],
                row["model_version"],
                float(row["predicted_duration_min"]),
                float(row["distance_km"]),
                row.get("traffic_index"),
                row.get("weather_condition"),
                int(bool(row.get("weather_imputed"))),
                float(row.get("latency_ms", 0.0)),
                json.dumps(row.get("features", {})),
            )
            for row in rows
        ]
        try:
            with self._lock, self._connect() as connection:
                connection.executemany(
                    """
                    INSERT OR REPLACE INTO predictions (
                        request_id, predicted_at_utc, pickup_datetime, model_name,
                        model_version, predicted_duration_min, distance_km, traffic_index,
                        weather_condition, weather_imputed, latency_ms, features_json
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    payload,
                )
        except sqlite3.Error as exc:
            logger.error("Failed to record %s prediction(s): %s", len(rows), exc)

    def record_feedback(self, request_id: str, actual_duration_min: float) -> dict | None:
        """Attach an observed duration to a prediction, returning the joined row."""
        try:
            with self._lock, self._connect() as connection:
                row = connection.execute(
                    "SELECT predicted_duration_min FROM predictions WHERE request_id = ?",
                    (request_id,),
                ).fetchone()
                if row is None:
                    return None

                predicted = float(row["predicted_duration_min"])
                error = abs(predicted - actual_duration_min)
                connection.execute(
                    """
                    UPDATE predictions
                       SET actual_duration_min = ?, absolute_error_min = ?, feedback_at_utc = ?
                     WHERE request_id = ?
                    """,
                    (
                        actual_duration_min,
                        error,
                        datetime.now(UTC).isoformat(timespec="seconds"),
                        request_id,
                    ),
                )
                return {"predicted_duration_min": predicted, "absolute_error_min": error}
        except sqlite3.Error as exc:
            logger.error("Failed to record feedback for %s: %s", request_id, exc)
            return None

    def counts(self) -> dict[str, int]:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT COUNT(*) AS total,
                           SUM(CASE WHEN actual_duration_min IS NOT NULL THEN 1 ELSE 0 END) AS labelled
                      FROM predictions
                    """
                ).fetchone()
                return {"total": int(row["total"] or 0), "labelled": int(row["labelled"] or 0)}
        except sqlite3.Error:
            return {"total": 0, "labelled": 0}


_LOG: PredictionLog | None = None


def get_prediction_log() -> PredictionLog:
    global _LOG
    if _LOG is None:
        _LOG = PredictionLog()
    return _LOG


def reset_prediction_log() -> None:
    global _LOG
    _LOG = None
