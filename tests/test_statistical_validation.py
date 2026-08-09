"""Level 3 statistical validation (M2 section 2.5.3).

Schema and range rules pass on every batch below, by construction. The point of
these tests is that Level 3 catches what Levels 1, 2 and 4 cannot see.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from src.data.statistical_validation import build_profile, compare_to_baseline
from src.data.validate import ValidationFailure, validate_frame


@pytest.fixture(scope="module")
def baseline(params, clean_trips):
    validated, _, _ = validate_frame(clean_trips, params)
    half = len(validated) // 2
    reference = validated.iloc[:half]
    holdout = validated.iloc[half:]
    return build_profile(reference, sample_size=1000, seed=1), holdout


def test_profile_covers_continuous_and_categorical_features(baseline):
    profile, _ = baseline
    assert "trip_duration_min" in profile["continuous"]
    assert "weather_condition" in profile["categorical"]
    assert profile["continuous"]["trip_duration_min"]["std"] > 0
    assert len(profile["continuous"]["trip_duration_min"]["sample"]) == 1000


def test_unshifted_batch_passes(params, baseline):
    profile, holdout = baseline
    result = compare_to_baseline(holdout, profile, params["validate"]["statistical"])
    assert result["status"] in {"pass", "warn"}
    assert result["n_fail"] == 0


def test_moderate_shift_warns_rather_than_failing(params, baseline):
    """M2 2.5.3: beyond one baseline standard deviation warrants investigation."""
    profile, holdout = baseline
    shifted = holdout.copy()
    shifted["traffic_index"] = np.clip(shifted["traffic_index"] + 0.30, 0.0, 1.0)

    result = compare_to_baseline(shifted, profile, params["validate"]["statistical"])
    check = next(c for c in result["checks"] if c["column"] == "traffic_index")
    assert 1.0 <= check["mean_shift_sd"] < 2.0
    assert check["status"] == "warn"

    # The range check cannot see this at all: every value is still inside [0, 1].
    lo, hi = params["validate"]["bounds"]["traffic_index"]
    assert shifted["traffic_index"].between(lo, hi).all()


def test_large_shift_fails(params, baseline):
    """M2 2.5.3: beyond two standard deviations warrants stopping the pipeline."""
    profile, holdout = baseline
    strict = copy.deepcopy(params["validate"]["statistical"])
    strict["mean_shift_sd_fail"] = 1.0

    shifted = holdout.copy()
    shifted["traffic_index"] = np.clip(shifted["traffic_index"] + 0.30, 0.0, 1.0)

    result = compare_to_baseline(shifted, profile, strict)
    assert result["status"] == "fail"
    assert "traffic_index" in result["failed_columns"]


def test_categorical_frequency_shift_is_detected(params, baseline):
    """A monsoon season: the weather mix moves without any new category appearing."""
    profile, holdout = baseline
    shifted = holdout.copy()
    rng = np.random.default_rng(0)
    flip = rng.random(len(shifted)) < 0.6
    shifted.loc[shifted.index[flip], "weather_condition"] = "Rain"

    result = compare_to_baseline(shifted, profile, params["validate"]["statistical"])
    assert result["status"] == "fail"
    assert "weather_condition" in result["failed_columns"]


def test_rising_null_rate_is_detected(params, baseline):
    profile, holdout = baseline
    degraded = holdout.copy()
    rng = np.random.default_rng(1)
    blank = rng.random(len(degraded)) < 0.25
    degraded.loc[degraded.index[blank], "temperature_c"] = np.nan

    result = compare_to_baseline(degraded, profile, params["validate"]["statistical"])
    assert "temperature_c" in result["failed_columns"]


def test_validate_frame_aborts_on_statistical_failure(params, clean_trips, baseline):
    profile, _ = baseline
    strict = copy.deepcopy(params)
    strict["validate"]["statistical"]["mean_shift_sd_fail"] = 1.0

    shifted = clean_trips.copy()
    shifted["traffic_index"] = np.clip(shifted["traffic_index"] + 0.30, 0.0, 1.0)

    with pytest.raises(ValidationFailure, match="Level 3"):
        validate_frame(shifted, strict, baseline=profile)


def test_validate_frame_skips_level_3_without_a_baseline(params, clean_trips):
    _, _, report = validate_frame(clean_trips, params)
    assert report["levels"]["level_3_statistical"].startswith("skipped")
