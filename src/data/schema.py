"""Pandera contract for the *validated* trip table (M2 Level 1).

This schema describes the guarantee the ingestion stage makes to everything
downstream. It is asserted at the end of validation and re-asserted by the unit
tests, so a silent change in the upstream feed fails loudly rather than
propagating into features and, eventually, into the served model.

Fields that are *required* for a row to be usable are non-nullable. Fields that are
merely *imputable* stay nullable on purpose: filling them here would mean computing
fill values from the whole batch, whereas M2 2.6.4 requires them to be learned from
the training partition alone and persisted for serving. That happens in the feature
pipeline, downstream of the split.
"""

from __future__ import annotations

import pandera.pandas as pa

from src.config import load_params

REQUIRED_RAW_COLUMNS: tuple[str, ...] = (
    "trip_id",
    "vendor_id",
    "pickup_datetime",
    "dropoff_datetime",
    "passenger_count",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "weather_condition",
    "temperature_c",
    "precipitation_mm",
    "wind_kph",
    "traffic_index",
)


def build_validated_schema(params: dict | None = None) -> pa.DataFrameSchema:
    params = params or load_params()
    cfg = params["validate"]
    bounds = cfg["bounds"]

    def between(key: str) -> pa.Check:
        low, high = bounds[key]
        return pa.Check.in_range(low, high, include_min=True, include_max=True)

    return pa.DataFrameSchema(
        columns={
            "trip_id": pa.Column(str, nullable=False, unique=True),
            "vendor_id": pa.Column(
                int, pa.Check.isin(cfg["allowed_vendors"]), nullable=False, coerce=True
            ),
            "pickup_datetime": pa.Column("datetime64[ns]", nullable=False),
            "dropoff_datetime": pa.Column("datetime64[ns]", nullable=False),
            "passenger_count": pa.Column(
                float, between("passenger_count"), nullable=True
            ),
            "pickup_latitude": pa.Column(float, between("latitude"), nullable=False),
            "pickup_longitude": pa.Column(float, between("longitude"), nullable=False),
            "dropoff_latitude": pa.Column(float, between("latitude"), nullable=False),
            "dropoff_longitude": pa.Column(float, between("longitude"), nullable=False),
            "store_and_fwd_flag": pa.Column(str, pa.Check.isin(["Y", "N"]), nullable=False),
            "weather_condition": pa.Column(
                str, pa.Check.isin(cfg["allowed_weather"]), nullable=True
            ),
            "temperature_c": pa.Column(float, between("temperature_c"), nullable=True),
            "precipitation_mm": pa.Column(float, between("precipitation_mm"), nullable=True),
            "wind_kph": pa.Column(float, between("wind_kph"), nullable=True),
            "traffic_index": pa.Column(float, between("traffic_index"), nullable=False),
            "trip_duration_min": pa.Column(float, between("trip_duration_min"), nullable=False),
            "avg_speed_kmph": pa.Column(float, between("avg_speed_kmph"), nullable=False),
        },
        checks=[
            pa.Check(
                lambda df: (df["dropoff_datetime"] > df["pickup_datetime"]).all(),
                name="dropoff_after_pickup",
                error="dropoff_datetime must be strictly after pickup_datetime",
            )
        ],
        strict=False,
        ordered=False,
        name="validated_trips",
    )
