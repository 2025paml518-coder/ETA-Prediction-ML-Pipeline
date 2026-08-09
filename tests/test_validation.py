"""Validation stage behaviour."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src.data.schema import build_validated_schema
from src.data.validate import ValidationFailure, validate_frame


def test_clean_batch_passes_without_quarantine(params, clean_trips):
    validated, quarantined, report = validate_frame(clean_trips, params)
    assert len(quarantined) == 0
    assert report["rows_validated"] == len(clean_trips)
    assert report["schema_passed"] is True


def test_planted_defects_are_quarantined(params, defective_trips):
    frame, planted = defective_trips
    validated, quarantined, report = validate_frame(frame, params)

    coord_cols = [
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
    ]
    # Every planted fatal defect must be caught; a row that carries several defects
    # is reported under a single reason code, so counts are bounded, not equal.
    assert frame[coord_cols].isna().any(axis=1).sum() == planted["missing_gps"]
    assert validated[coord_cols].notna().all().all()
    assert not validated["trip_id"].duplicated().any()

    reasons = report["quarantine_reasons"]
    assert sum(reasons.values()) == len(quarantined)
    assert reasons["DUPLICATE_TRIP_ID"] == planted["duplicate_trip_id"]
    assert reasons["MISSING_GPS"] <= planted["missing_gps"]
    assert reasons["INVALID_TIMESTAMP_ORDER"] > 0
    assert reasons["GPS_OUT_OF_BOUNDS"] > 0

    # Repairable defects are imputed, never silently dropped.
    assert report["repairs"]["weather_imputed"] > 0
    assert report["repairs"]["passenger_count_imputed"] > 0
    assert len(validated) + len(quarantined) == len(frame)


def test_validated_output_satisfies_schema_contract(params, defective_trips):
    frame, _ = defective_trips
    validated, _, _ = validate_frame(frame, params)
    build_validated_schema(params).validate(validated, lazy=True)

    assert validated["trip_id"].is_unique
    assert (validated["dropoff_datetime"] > validated["pickup_datetime"]).all()
    assert validated[["pickup_latitude", "dropoff_longitude"]].notna().all().all()


def test_quarantine_rows_carry_a_reason_code(params, defective_trips):
    frame, _ = defective_trips
    _, quarantined, _ = validate_frame(frame, params)
    assert quarantined["quarantine_reason"].ne("").all()
    assert quarantined["quarantine_reason"].notna().all()


def test_pipeline_aborts_when_batch_is_mostly_broken(params, clean_trips):
    strict = copy.deepcopy(params)
    strict["validate"]["max_bad_row_fraction"] = 0.01

    broken = clean_trips.copy()
    corrupt = broken.index[: int(len(broken) * 0.2)]
    broken.loc[corrupt, "pickup_latitude"] = np.nan

    with pytest.raises(ValidationFailure, match="exceeds"):
        validate_frame(broken, strict)


def test_pipeline_aborts_on_undersized_batch(params, clean_trips):
    strict = copy.deepcopy(params)
    strict["validate"]["min_rows"] = 10_000_000
    with pytest.raises(ValidationFailure, match="minimum"):
        validate_frame(clean_trips, strict)


def test_missing_required_column_is_rejected(params, clean_trips):
    with pytest.raises(ValidationFailure, match="missing required columns"):
        validate_frame(clean_trips.drop(columns=["traffic_index"]), params)


def test_derived_duration_matches_timestamps(params, clean_trips):
    validated, _, _ = validate_frame(clean_trips, params)
    expected = (
        validated["dropoff_datetime"] - validated["pickup_datetime"]
    ).dt.total_seconds() / 60.0
    pd.testing.assert_series_equal(
        validated["trip_duration_min"], expected, check_names=False, rtol=1e-9
    )
