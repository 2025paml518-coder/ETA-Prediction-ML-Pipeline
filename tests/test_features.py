"""Feature pipeline contract."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.validate import validate_frame
from src.features.build_features import (
    FEATURE_COLUMNS,
    TARGET,
    FeaturePipeline,
    FeaturePipelineNotFitted,
)


@pytest.fixture(scope="module")
def fitted(params, clean_trips):
    validated, _, _ = validate_frame(clean_trips, params)
    pipeline = FeaturePipeline.from_params(params).fit(validated)
    return pipeline, validated


def test_transform_emits_canonical_columns_in_order(fitted):
    pipeline, frame = fitted
    features = pipeline.transform(frame)
    assert list(features.columns) == list(FEATURE_COLUMNS)
    assert len(features) == len(frame)


def test_features_are_finite_and_non_null(fitted):
    pipeline, frame = fitted
    features = pipeline.transform(frame)
    assert features.notna().all().all()
    assert np.isfinite(features.to_numpy()).all()


def test_transform_is_deterministic(fitted):
    pipeline, frame = fitted
    np.testing.assert_array_equal(
        pipeline.transform(frame).to_numpy(), pipeline.transform(frame).to_numpy()
    )


def test_transform_never_reads_the_target(fitted):
    """No feature may depend on dropoff time, which does not exist at inference."""
    pipeline, frame = fitted
    without_target = frame.drop(columns=[TARGET, "dropoff_datetime", "avg_speed_kmph"])
    np.testing.assert_array_equal(
        pipeline.transform(without_target).to_numpy(), pipeline.transform(frame).to_numpy()
    )


def test_unfitted_pipeline_refuses_to_transform(params, clean_trips):
    with pytest.raises(FeaturePipelineNotFitted):
        FeaturePipeline.from_params(params).transform(clean_trips)


def test_missing_input_column_is_rejected(fitted):
    pipeline, frame = fitted
    with pytest.raises(ValueError, match="missing required columns"):
        pipeline.transform(frame.drop(columns=["traffic_index"]))


def test_unseen_zone_falls_back_to_a_usable_speed_prior(fitted):
    """A trip outside the training footprint must still yield a finite prediction input."""
    pipeline, frame = fitted
    far_away = frame.head(5).copy()
    far_away["pickup_latitude"] = 40.5795  # Staten Island: absent from the fixture
    far_away["pickup_longitude"] = -74.1502

    features = pipeline.transform(far_away)
    assert features["zone_hour_speed_prior"].between(1.0, 120.0).all()
    assert features["route_speed_prior"].between(1.0, 120.0).all()


def test_distance_features_are_consistent(fitted):
    pipeline, frame = fitted
    features = pipeline.transform(frame)
    assert (features["haversine_km"] >= 0).all()
    assert (features["manhattan_km"] >= features["haversine_km"] - 1e-6).all()


def test_cyclical_encodings_stay_on_the_unit_circle(fitted):
    pipeline, frame = fitted
    features = pipeline.transform(frame)
    for prefix in ("hour", "dow", "month", "bearing"):
        radius = features[f"{prefix}_sin"] ** 2 + features[f"{prefix}_cos"] ** 2
        np.testing.assert_allclose(radius.to_numpy(), 1.0, atol=1e-9)


def test_nearest_centroid_matches_sklearn_predict(fitted):
    """The hand-rolled assignment must equal what KMeans.predict would have returned."""
    pipeline, frame = fitted
    sample = frame.head(500)
    coords = sample[["pickup_latitude", "pickup_longitude"]].to_numpy(dtype=float)
    np.testing.assert_array_equal(
        pipeline._assign_zone(sample, "pickup"), pipeline.kmeans.predict(coords)
    )


def test_saved_artifact_is_byte_stable_across_refits(params, clean_trips, tmp_path):
    """Refitting on identical data must produce identical artefact bytes.

    Pickled estimators failed this, which made every `dvc repro` dirty the lock file
    even when nothing had actually changed.
    """
    from src.data.validate import validate_frame

    validated, _, _ = validate_frame(clean_trips, params)
    first = FeaturePipeline.from_params(params).fit(validated).save(tmp_path / "a")
    second = FeaturePipeline.from_params(params).fit(validated).save(tmp_path / "b")

    for name in ("speed_priors.json", "feature_spec.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


def test_imputation_values_come_from_training_only(params, defective_trips):
    """M2 2.6.4: fill values must be learned from the training split, not the batch."""
    from src.data.split import temporal_split
    from src.data.validate import validate_frame

    validated, _, _ = validate_frame(defective_trips[0], params)
    parts = temporal_split(validated, 0.15, 0.15, "pickup_datetime")

    pipeline = FeaturePipeline.from_params(params).fit(parts["train"])
    expected = float(parts["train"]["passenger_count"].median())

    assert pipeline.imputation["passenger_count"]["global"] == expected
    assert pipeline.imputation["weather_condition"]["strategy"] == "month_mode"
    assert pipeline.imputation["numeric"]["strategy"] == "month_median"

    # Refitting on the test partition would produce different values, which is exactly
    # what must never happen at serving time.
    other = FeaturePipeline.from_params(params).fit(parts["test"])
    assert other.imputation["numeric"]["global"] != pipeline.imputation["numeric"]["global"]


def test_imputation_survives_a_save_load_round_trip(params, defective_trips, tmp_path):
    from src.data.validate import validate_frame

    validated, _, _ = validate_frame(defective_trips[0], params)
    pipeline = FeaturePipeline.from_params(params).fit(validated)
    pipeline.save(tmp_path / "fp")
    reloaded = FeaturePipeline.load(tmp_path / "fp")

    assert reloaded.imputation == pipeline.imputation

    gapped = validated[validated["weather_condition"].isna()].head(20)
    assert len(gapped) > 0
    features = reloaded.transform(gapped)
    assert features["weather_imputed"].eq(1.0).all()
    assert features.notna().all().all()


def test_missing_covariates_are_flagged_not_dropped(params, defective_trips):
    from src.data.validate import validate_frame

    validated, _, _ = validate_frame(defective_trips[0], params)
    pipeline = FeaturePipeline.from_params(params).fit(validated)
    features = pipeline.transform(validated)

    assert len(features) == len(validated)
    assert features["weather_imputed"].sum() == validated["weather_condition"].isna().sum()
    assert (
        features["passenger_count_imputed"].sum()
        == validated["passenger_count"].isna().sum()
    )
