"""Training-serving skew guard.

The training pipeline transforms a large batch; the API transforms one request at
a time from a reloaded artefact. These tests assert the two paths agree exactly.
Without them, skew is only discoverable in production, and by then it looks like a
model quality problem rather than a plumbing problem.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.validate import validate_frame
from src.features.build_features import (
    FEATURE_COLUMNS,
    REQUIRED_INPUT_COLUMNS,
    FeaturePipeline,
)


@pytest.fixture(scope="module")
def fitted(params, clean_trips):
    validated, _, _ = validate_frame(clean_trips, params)
    return FeaturePipeline.from_params(params).fit(validated), validated


def test_single_row_matches_batch_transform(fitted):
    pipeline, frame = fitted
    batch = pipeline.transform(frame.head(50))
    for position in range(50):
        row = frame.iloc[[position]]
        np.testing.assert_allclose(
            pipeline.transform(row).to_numpy()[0], batch.to_numpy()[position], rtol=0, atol=0
        )


def test_saved_and_reloaded_pipeline_is_bit_identical(fitted, tmp_path):
    pipeline, frame = fitted
    pipeline.save(tmp_path / "feature_pipeline")
    reloaded = FeaturePipeline.load(tmp_path / "feature_pipeline")

    np.testing.assert_array_equal(
        pipeline.transform(frame).to_numpy(), reloaded.transform(frame).to_numpy()
    )
    assert reloaded.feature_spec()["feature_columns"] == list(FEATURE_COLUMNS)


def test_reloaded_pipeline_serves_a_request_carrying_only_api_fields(fitted, tmp_path):
    """Mirrors the API path: reload from disk and transform a request that carries
    only the documented input contract - no target, no derived training columns."""
    pipeline, frame = fitted
    pipeline.save(tmp_path / "feature_pipeline")
    reloaded = FeaturePipeline.load(tmp_path / "feature_pipeline")

    request = frame.iloc[[0]][list(REQUIRED_INPUT_COLUMNS)]
    served = reloaded.transform(request)
    trained = pipeline.transform(frame.iloc[[0]])

    assert list(served.columns) == list(FEATURE_COLUMNS)
    np.testing.assert_array_equal(served.to_numpy(), trained.to_numpy())


def test_row_order_does_not_change_feature_values(fitted):
    pipeline, frame = fitted
    sample = frame.head(200)
    forward = pipeline.transform(sample)
    reversed_frame = sample.iloc[::-1]
    backward = pipeline.transform(reversed_frame)
    np.testing.assert_allclose(
        forward.to_numpy(), backward.to_numpy()[::-1], rtol=0, atol=0
    )
