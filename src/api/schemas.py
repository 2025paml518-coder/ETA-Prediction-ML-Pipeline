"""Request and response contracts for the prediction service (M4).

Validation happens at the edge, before anything reaches the feature pipeline. A
malformed request should be rejected with a 422 that names the offending field,
not silently coerced into a confident prediction - a model given nonsense returns
a number, not an error, which is the failure mode M2 opens with.

Bounds mirror ``validate.bounds`` in params.yaml so that the service rejects the
same values the training pipeline would have quarantined. Serving accepting rows
that training would have thrown away is a form of training-serving skew.

Weather and passenger count are optional. The feature pipeline holds imputation
values learned from the training partition, so a caller that cannot supply the
weather still gets a prediction, and the imputation is flagged as a feature
exactly as it was during training.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.config import load_params

_BOUNDS = load_params()["validate"]["bounds"]
_LAT_MIN, _LAT_MAX = _BOUNDS["latitude"]
_LON_MIN, _LON_MAX = _BOUNDS["longitude"]
_TEMP_MIN, _TEMP_MAX = _BOUNDS["temperature_c"]
_PRECIP_MIN, _PRECIP_MAX = _BOUNDS["precipitation_mm"]
_WIND_MIN, _WIND_MAX = _BOUNDS["wind_kph"]
_PAX_MIN, _PAX_MAX = _BOUNDS["passenger_count"]

WeatherCondition = Literal["Clear", "Cloudy", "Rain", "Snow", "Fog"]

MAX_BATCH_SIZE = 500


class TripRequest(BaseModel):
    """One trip awaiting an ETA."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    pickup_datetime: datetime = Field(
        ..., description="Trip start time, ISO 8601.", examples=["2024-07-15T08:30:00"]
    )
    pickup_latitude: float = Field(..., ge=_LAT_MIN, le=_LAT_MAX, examples=[40.7549])
    pickup_longitude: float = Field(..., ge=_LON_MIN, le=_LON_MAX, examples=[-73.9840])
    dropoff_latitude: float = Field(..., ge=_LAT_MIN, le=_LAT_MAX, examples=[40.6413])
    dropoff_longitude: float = Field(..., ge=_LON_MIN, le=_LON_MAX, examples=[-73.7781])

    traffic_index: float = Field(
        ..., ge=0.0, le=1.0, description="0 is free-flowing, 1 is gridlock.", examples=[0.62]
    )

    vendor_id: Literal[1, 2, 3] = Field(default=1)
    passenger_count: int | None = Field(default=None, ge=_PAX_MIN, le=_PAX_MAX, examples=[2])
    store_and_fwd_flag: Literal["Y", "N"] = Field(default="N")

    weather_condition: WeatherCondition | None = Field(default=None, examples=["Clear"])
    temperature_c: float | None = Field(default=None, ge=_TEMP_MIN, le=_TEMP_MAX, examples=[24.5])
    precipitation_mm: float | None = Field(
        default=None, ge=_PRECIP_MIN, le=_PRECIP_MAX, examples=[0.0]
    )
    wind_kph: float | None = Field(default=None, ge=_WIND_MIN, le=_WIND_MAX, examples=[12.0])

    @field_validator("pickup_datetime")
    @classmethod
    def _drop_timezone(cls, value: datetime) -> datetime:
        """The model was trained on naive local timestamps; normalise to match."""
        return value.replace(tzinfo=None) if value.tzinfo else value

    @model_validator(mode="after")
    def _weather_must_be_whole_or_absent(self) -> TripRequest:
        """Reject half-populated weather.

        A partial weather record is the serving-time equivalent of the
        BR_PARTIAL_WEATHER_RECORD rule: it means the caller's own weather lookup
        half-failed, and quietly imputing the gaps would hide that.
        """
        fields = {
            "weather_condition": self.weather_condition,
            "temperature_c": self.temperature_c,
            "precipitation_mm": self.precipitation_mm,
            "wind_kph": self.wind_kph,
        }
        supplied = {name for name, value in fields.items() if value is not None}
        if supplied and len(supplied) != len(fields):
            missing = sorted(set(fields) - supplied)
            raise ValueError(
                "weather must be supplied in full or omitted entirely; missing: "
                + ", ".join(missing)
            )
        return self

    @model_validator(mode="after")
    def _precipitation_implies_wet_weather(self) -> TripRequest:
        """The Level 4 business rule, enforced at the edge."""
        if (
            self.precipitation_mm is not None
            and self.precipitation_mm > 0
            and self.weather_condition not in {"Rain", "Snow"}
        ):
            raise ValueError(
                f"precipitation_mm={self.precipitation_mm} contradicts "
                f"weather_condition={self.weather_condition!r}"
            )
        return self


class BatchTripRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trips: list[TripRequest] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)


class PredictionResponse(BaseModel):
    predicted_duration_min: float = Field(..., examples=[18.42])
    predicted_dropoff_time: datetime
    distance_km: float = Field(..., description="Great-circle pickup-to-dropoff distance.")
    model_name: str
    model_version: str
    request_id: str
    latency_ms: float
    weather_imputed: bool = Field(
        ..., description="True when the caller omitted weather and training-time values were used."
    )


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int
    latency_ms: float


class FeedbackRequest(BaseModel):
    """Observed outcome for a previous prediction.

    Without this the service can only monitor its inputs. Actual durations are what
    turn Week 4 monitoring from drift detection into error measurement.
    """

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(..., min_length=1)
    actual_duration_min: float = Field(..., gt=0, le=1440)


class FeedbackResponse(BaseModel):
    request_id: str
    actual_duration_min: float
    predicted_duration_min: float | None
    absolute_error_min: float | None
    recorded: bool


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    model_loaded: bool
    feature_pipeline_loaded: bool
    detail: str | None = None


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    model_family: str | None
    run_id: str | None
    trained_at: str | None
    feature_count: int
    feature_columns: list[str]
    selected_by: str | None
    metrics: dict | None


class ErrorDetail(BaseModel):
    field: str | None
    message: str


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: str
    errors: list[ErrorDetail] | None = None
