"""Shared pytest fixtures."""

from __future__ import annotations

import copy

import pytest

from src.config import load_params
from src.data.generate_synthetic import SCENARIOS, generate_trips, inject_defects

SMALL_N = 4000


@pytest.fixture(scope="session")
def params() -> dict:
    """A mutable copy of params.yaml with thresholds relaxed for small fixtures."""
    cfg = copy.deepcopy(load_params())
    cfg["validate"]["min_rows"] = 100
    cfg["features"]["n_zone_clusters"] = 8
    return cfg


@pytest.fixture(scope="session")
def clean_trips(params):
    return generate_trips(
        n_trips=SMALL_N,
        start_date=params["generate"]["start_date"],
        end_date=params["generate"]["end_date"],
        seed=7,
        scenario=SCENARIOS["baseline"],
    )


@pytest.fixture(scope="session")
def defective_trips(params, clean_trips):
    frame, planted = inject_defects(clean_trips, params["generate"]["defect_rates"], seed=7)
    return frame, planted
