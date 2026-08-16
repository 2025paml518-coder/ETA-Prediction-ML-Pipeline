"""Model loading and inference.

The service builds features with the *same* ``FeaturePipeline`` object the training
run fitted, loaded from the same artefact directory. Nothing about feature
construction is re-implemented here - that duplication is the mechanism behind
training-serving skew (M2 2.7), and `tests/test_skew.py` asserts the two paths agree
byte for byte.

The model is resolved from the MLflow Model Registry by name so that Week 4 can
promote a retrained version without redeploying the service. A local artefact is
used as a fallback when no registry is reachable, which keeps the container
runnable in isolation.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from src.config import load_params, project_path
from src.features.build_features import FEATURE_COLUMNS, FeaturePipeline
from src.utils.geo import haversine_km
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ModelNotLoadedError(RuntimeError):
    """Raised when inference is attempted before the model is available."""


@dataclass
class LoadedModel:
    model: Any
    name: str
    version: str
    source: str
    run_id: str | None
    metadata: dict


class Predictor:
    """Holds the feature pipeline and the model, and turns requests into ETAs."""

    def __init__(self, params: dict | None = None) -> None:
        self._params = params or load_params()
        self._lock = threading.Lock()
        self._feature_pipeline: FeaturePipeline | None = None
        self._loaded: LoadedModel | None = None
        self._load_error: str | None = None

    # ----------------------------------------------------------------- loading
    def load(self) -> None:
        """Load artefacts once, at startup, so no request pays the cost."""
        with self._lock:
            try:
                self._feature_pipeline = self._load_feature_pipeline()
                self._loaded = self._load_model()
                self._load_error = None
                logger.info(
                    "Loaded %s v%s from %s",
                    self._loaded.name,
                    self._loaded.version,
                    self._loaded.source,
                )
            except Exception as exc:  # noqa: BLE001 - surfaced through /ready
                self._load_error = str(exc)
                logger.error("Startup load failed: %s", exc)

    def _load_feature_pipeline(self) -> FeaturePipeline:
        directory = project_path(f"{self._params['paths']['models']}/feature_pipeline")
        if not directory.exists():
            raise FileNotFoundError(
                f"Feature pipeline artefact missing at {directory}; run `dvc repro features`"
            )
        return FeaturePipeline.load(directory)

    def _metadata(self) -> dict:
        path = project_path(
            f"{self._params['paths']['models']}/trained/best_model_metadata.json"
        )
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {}

    def _load_model(self) -> LoadedModel:
        cfg = self._params["train"]
        metadata = self._metadata()
        name = cfg["registered_model_name"]
        models_root = project_path(self._params["paths"]["models"])

        tracking_dir = project_path(cfg["tracking_uri"])
        if tracking_dir.exists():
            mlflow.set_tracking_uri(tracking_dir.as_uri())
            try:
                from mlflow.tracking import MlflowClient

                client = MlflowClient()
                versions = client.search_model_versions(f"name='{name}'")
                if versions:
                    latest = max(versions, key=lambda v: int(v.version))
                    # The native estimator rather than the pyfunc wrapper: pyfunc
                    # re-validates the frame against the signature on every call, which
                    # is redundant once Pydantic and the feature contract have both run,
                    # and it doubled per-request inference cost.
                    model = mlflow.sklearn.load_model(f"models:/{name}/{latest.version}")
                    return LoadedModel(
                        model=model,
                        name=name,
                        version=str(latest.version),
                        source="mlflow-registry",
                        run_id=latest.run_id,
                        metadata=metadata,
                    )
            except Exception as exc:  # noqa: BLE001 - fall through to the local artefact
                logger.warning("Registry lookup failed (%s); falling back to run artefact", exc)

        run_id = metadata.get("best_run_id")
        if run_id and tracking_dir.exists():
            mlflow.set_tracking_uri(tracking_dir.as_uri())
            model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
            return LoadedModel(
                model=model,
                name=name,
                version=f"run-{run_id[:8]}",
                source="mlflow-run",
                run_id=run_id,
                metadata=metadata,
            )

        # Standalone export. This is what the container ships: one model directory
        # rather than the entire tracking store.
        exported = models_root / "trained" / "model"
        if exported.exists():
            return LoadedModel(
                model=mlflow.sklearn.load_model(str(exported)),
                name=name,
                version=f"exported-{(run_id or 'unknown')[:8]}",
                source="local-export",
                run_id=run_id,
                metadata=metadata,
            )

        raise FileNotFoundError(
            "No model available: no registered version, run artefact or standalone "
            "export was found. Run `dvc repro train` first."
        )

    # ------------------------------------------------------------------ status
    @property
    def is_ready(self) -> bool:
        return self._feature_pipeline is not None and self._loaded is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def feature_pipeline_loaded(self) -> bool:
        return self._feature_pipeline is not None

    @property
    def model_loaded(self) -> bool:
        return self._loaded is not None

    def info(self) -> dict:
        if self._loaded is None:
            raise ModelNotLoadedError("Model has not been loaded")
        metadata = self._loaded.metadata
        return {
            "model_name": self._loaded.name,
            "model_version": self._loaded.version,
            "model_family": metadata.get("best_model"),
            "run_id": self._loaded.run_id,
            "trained_at": metadata.get("lineage", {}).get("git_commit"),
            "feature_count": len(FEATURE_COLUMNS),
            "feature_columns": list(FEATURE_COLUMNS),
            "selected_by": metadata.get("selected_by"),
            "metrics": metadata.get("metrics"),
        }

    # --------------------------------------------------------------- inference
    def predict(self, rows: list[dict]) -> tuple[np.ndarray, pd.DataFrame, float]:
        """Featurise and score a batch of validated request dictionaries."""
        if not self.is_ready:
            raise ModelNotLoadedError(self._load_error or "Model has not been loaded")

        frame = pd.DataFrame(rows)
        started = time.perf_counter()
        features = self._feature_pipeline.transform(frame)
        predictions = np.asarray(self._loaded.model.predict(features), dtype=float)
        # A negative ETA is never a useful answer, whatever the regressor says.
        predictions = np.maximum(predictions, 0.5)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return predictions, features, elapsed_ms

    @staticmethod
    def distances_km(rows: list[dict]) -> np.ndarray:
        frame = pd.DataFrame(rows)
        return haversine_km(
            frame["pickup_latitude"],
            frame["pickup_longitude"],
            frame["dropoff_latitude"],
            frame["dropoff_longitude"],
        )

    def warmup(self) -> None:
        """Force the first, slow prediction at startup rather than on a user request."""
        if not self.is_ready:
            return
        sample = {
            "pickup_datetime": pd.Timestamp("2024-07-15T08:30:00"),
            "pickup_latitude": 40.7549,
            "pickup_longitude": -73.9840,
            "dropoff_latitude": 40.6413,
            "dropoff_longitude": -73.7781,
            "passenger_count": None,
            "vendor_id": 1,
            "store_and_fwd_flag": "N",
            "weather_condition": None,
            "temperature_c": None,
            "precipitation_mm": None,
            "wind_kph": None,
            "traffic_index": 0.5,
        }
        try:
            self.predict([sample])
            logger.info("Warm-up prediction completed")
        except Exception as exc:  # noqa: BLE001 - warm-up must never block startup
            logger.warning("Warm-up failed: %s", exc)


_PREDICTOR: Predictor | None = None


def get_predictor() -> Predictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = Predictor()
    return _PREDICTOR


def reset_predictor() -> None:
    """Drop the cached singleton; used by tests."""
    global _PREDICTOR
    _PREDICTOR = None


def artifact_dir() -> Path:
    return project_path(f"{load_params()['paths']['models']}/feature_pipeline")
