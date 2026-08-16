"""FastAPI application for ETA prediction (M4).

Endpoints:

    GET  /health        liveness - the process is up
    GET  /ready         readiness - the model is loaded and can serve
    GET  /model/info    which model, which run, which features
    GET  /metrics       Prometheus exposition
    POST /predict       one trip
    POST /predict/batch many trips in a single call
    POST /feedback      report the actual duration for a served prediction

Liveness and readiness are separate on purpose. A container that is running but
whose model failed to load is alive and *not* ready; conflating the two makes an
orchestrator restart a healthy process or route traffic to a broken one.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from datetime import timedelta

from fastapi import APIRouter, FastAPI, Request, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src import __version__
from src.api.errors import register_exception_handlers
from src.api.middleware import (
    FEEDBACK_ERROR,
    MODEL_READY,
    PREDICTION_COUNT,
    PREDICTION_VALUE,
    register_middleware,
)
from src.api.predictor import get_predictor
from src.api.schemas import (
    BatchPredictionResponse,
    BatchTripRequest,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    ReadinessResponse,
    TripRequest,
)
from src.monitoring.prediction_log import get_prediction_log
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

SERVICE_NAME = "eta-prediction-api"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load artefacts once at startup so no request pays the cost."""
    predictor = get_predictor()
    predictor.load()
    predictor.warmup()
    MODEL_READY.set(1 if predictor.is_ready else 0)
    get_prediction_log()
    yield
    logger.info("Shutting down %s", SERVICE_NAME)


app = FastAPI(
    title="ETA Prediction API",
    description=(
        "Predicts delivery / ride duration from trip geometry, time of day, "
        "weather and traffic conditions."
    ),
    version=__version__,
    lifespan=lifespan,
)

register_middleware(app)
register_exception_handlers(app)

router = APIRouter()


def _to_rows(trips: list[TripRequest]) -> list[dict]:
    """Convert validated requests into the frame the feature pipeline expects."""
    return [
        {
            "pickup_datetime": trip.pickup_datetime,
            "pickup_latitude": trip.pickup_latitude,
            "pickup_longitude": trip.pickup_longitude,
            "dropoff_latitude": trip.dropoff_latitude,
            "dropoff_longitude": trip.dropoff_longitude,
            "passenger_count": trip.passenger_count,
            "vendor_id": trip.vendor_id,
            "store_and_fwd_flag": trip.store_and_fwd_flag,
            "weather_condition": trip.weather_condition,
            "temperature_c": trip.temperature_c,
            "precipitation_mm": trip.precipitation_mm,
            "wind_kph": trip.wind_kph,
            "traffic_index": trip.traffic_index,
        }
        for trip in trips
    ]


def _serve(trips: list[TripRequest], request: Request) -> tuple[list[PredictionResponse], float]:
    predictor = get_predictor()
    rows = _to_rows(trips)

    started = time.perf_counter()
    predictions, features, _ = predictor.predict(rows)
    distances = predictor.distances_km(rows)
    total_ms = (time.perf_counter() - started) * 1000.0

    info = predictor.info()
    base_id = getattr(request.state, "request_id", "unknown")
    per_item_ms = total_ms / max(len(trips), 1)

    responses: list[PredictionResponse] = []
    log_rows: list[dict] = []
    for index, trip in enumerate(trips):
        request_id = base_id if len(trips) == 1 else f"{base_id}-{index}"
        duration = float(predictions[index])
        weather_imputed = bool(features.iloc[index]["weather_imputed"])

        responses.append(
            PredictionResponse(
                predicted_duration_min=round(duration, 2),
                predicted_dropoff_time=trip.pickup_datetime + timedelta(minutes=duration),
                distance_km=round(float(distances[index]), 3),
                model_name=info["model_name"],
                model_version=info["model_version"],
                request_id=request_id,
                latency_ms=round(per_item_ms, 3),
                weather_imputed=weather_imputed,
            )
        )
        log_rows.append(
            {
                "request_id": request_id,
                "pickup_datetime": trip.pickup_datetime,
                "model_name": info["model_name"],
                "model_version": info["model_version"],
                "predicted_duration_min": duration,
                "distance_km": float(distances[index]),
                "traffic_index": trip.traffic_index,
                "weather_condition": trip.weather_condition,
                "weather_imputed": weather_imputed,
                "latency_ms": per_item_ms,
                "features": features.iloc[index].to_dict(),
            }
        )

        PREDICTION_COUNT.labels(info["model_version"]).inc()
        PREDICTION_VALUE.observe(duration)

    get_prediction_log().record(log_rows)
    return responses, total_ms


@router.get("/health", response_model=HealthResponse, tags=["operations"])
def health() -> HealthResponse:
    """Liveness. Deliberately does not touch the model."""
    return HealthResponse(status="ok", service=SERVICE_NAME, version=__version__)


@router.get("/ready", response_model=ReadinessResponse, tags=["operations"])
def ready(response: Response) -> ReadinessResponse:
    predictor = get_predictor()
    MODEL_READY.set(1 if predictor.is_ready else 0)
    if predictor.is_ready:
        return ReadinessResponse(status="ready", model_loaded=True, feature_pipeline_loaded=True)

    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(
        status="not_ready",
        model_loaded=predictor.model_loaded,
        feature_pipeline_loaded=predictor.feature_pipeline_loaded,
        detail=predictor.load_error,
    )


@router.get("/model/info", response_model=ModelInfoResponse, tags=["operations"])
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(**get_predictor().info())


@router.get("/metrics", tags=["operations"])
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(trip: TripRequest, request: Request) -> PredictionResponse:
    responses, _ = _serve([trip], request)
    return responses[0]


@router.post("/predict/batch", response_model=BatchPredictionResponse, tags=["inference"])
def predict_batch(payload: BatchTripRequest, request: Request) -> BatchPredictionResponse:
    """Batched inference.

    One featurisation and one vectorised model call for the whole payload, which is
    why per-trip latency falls sharply with batch size.
    """
    responses, total_ms = _serve(payload.trips, request)
    return BatchPredictionResponse(
        predictions=responses, count=len(responses), latency_ms=round(total_ms, 3)
    )


@router.post("/feedback", response_model=FeedbackResponse, tags=["monitoring"])
def feedback(payload: FeedbackRequest, response: Response) -> FeedbackResponse:
    """Report an observed duration so error can be measured, not just drift."""
    result = get_prediction_log().record_feedback(
        payload.request_id, payload.actual_duration_min
    )
    if result is None:
        response.status_code = status.HTTP_404_NOT_FOUND
        return FeedbackResponse(
            request_id=payload.request_id,
            actual_duration_min=payload.actual_duration_min,
            predicted_duration_min=None,
            absolute_error_min=None,
            recorded=False,
        )

    FEEDBACK_ERROR.observe(result["absolute_error_min"])
    return FeedbackResponse(
        request_id=payload.request_id,
        actual_duration_min=payload.actual_duration_min,
        predicted_duration_min=round(result["predicted_duration_min"], 2),
        absolute_error_min=round(result["absolute_error_min"], 2),
        recorded=True,
    )


app.include_router(router)
