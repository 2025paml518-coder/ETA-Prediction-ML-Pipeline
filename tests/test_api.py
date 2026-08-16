"""Prediction service contract and error handling (M4).

The malformed-input cases matter more than the happy path here. A model handed
nonsense returns a confident number rather than an error, so the boundary where a
bad request is turned away is the only place that failure can be caught.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.predictor import get_predictor
from src.monitoring.prediction_log import PredictionLog, reset_prediction_log

VALID_TRIP = {
    "pickup_datetime": "2024-07-15T08:30:00",
    "pickup_latitude": 40.7549,
    "pickup_longitude": -73.9840,
    "dropoff_latitude": 40.6413,
    "dropoff_longitude": -73.7781,
    "traffic_index": 0.62,
    "vendor_id": 1,
    "passenger_count": 2,
    "store_and_fwd_flag": "N",
    "weather_condition": "Clear",
    "temperature_c": 24.5,
    "precipitation_mm": 0.0,
    "wind_kph": 12.0,
}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # Route the prediction log to a scratch database so tests never touch real data.
    import src.monitoring.prediction_log as log_module

    reset_prediction_log()
    scratch = tmp_path_factory.mktemp("monitoring") / "predictions.db"
    log_module._LOG = PredictionLog(scratch)

    from src.api.main import app

    with TestClient(app) as test_client:
        if not get_predictor().is_ready:
            pytest.skip("Model artefacts unavailable; run `dvc repro train` first")
        yield test_client

    reset_prediction_log()


class TestOperations:
    def test_health_does_not_depend_on_the_model(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_ready_reports_model_state(self, client):
        response = client.get("/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ready"
        assert body["model_loaded"] is True
        assert body["feature_pipeline_loaded"] is True

    def test_model_info_exposes_the_feature_contract(self, client):
        body = client.get("/model/info").json()
        assert body["feature_count"] == len(body["feature_columns"])
        assert body["selected_by"] == "val_mae"
        assert body["model_version"]

    def test_metrics_are_exposed_for_prometheus(self, client):
        client.post("/predict", json=VALID_TRIP)
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "eta_requests_total" in response.text
        assert "eta_request_latency_seconds" in response.text

    def test_every_response_carries_a_correlation_id(self, client):
        response = client.get("/health")
        assert response.headers["X-Request-ID"]
        assert float(response.headers["X-Response-Time-ms"]) >= 0


class TestPrediction:
    def test_valid_trip_returns_a_plausible_eta(self, client):
        response = client.post("/predict", json=VALID_TRIP)
        assert response.status_code == 200

        body = response.json()
        assert 1 < body["predicted_duration_min"] < 300
        assert body["distance_km"] > 0
        assert body["weather_imputed"] is False
        assert body["request_id"]

    def test_dropoff_time_is_consistent_with_the_prediction(self, client):
        from datetime import datetime

        body = client.post("/predict", json=VALID_TRIP).json()
        pickup = datetime.fromisoformat(VALID_TRIP["pickup_datetime"])
        dropoff = datetime.fromisoformat(body["predicted_dropoff_time"])
        elapsed = (dropoff - pickup).total_seconds() / 60
        assert elapsed == pytest.approx(body["predicted_duration_min"], abs=0.02)

    def test_omitted_weather_is_imputed_and_flagged(self, client):
        payload = {
            key: value
            for key, value in VALID_TRIP.items()
            if key not in {"weather_condition", "temperature_c", "precipitation_mm", "wind_kph"}
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert response.json()["weather_imputed"] is True

    def test_longer_trips_predict_longer_durations(self, client):
        short = dict(VALID_TRIP, dropoff_latitude=40.7600, dropoff_longitude=-73.9800)
        long_trip = dict(VALID_TRIP)

        short_eta = client.post("/predict", json=short).json()["predicted_duration_min"]
        long_eta = client.post("/predict", json=long_trip).json()["predicted_duration_min"]
        assert long_eta > short_eta

    def test_heavier_traffic_predicts_longer_durations(self, client):
        light = client.post("/predict", json=dict(VALID_TRIP, traffic_index=0.05)).json()
        heavy = client.post("/predict", json=dict(VALID_TRIP, traffic_index=0.95)).json()
        assert heavy["predicted_duration_min"] > light["predicted_duration_min"]

    def test_batch_matches_single_predictions(self, client):
        single = client.post("/predict", json=VALID_TRIP).json()["predicted_duration_min"]
        batch = client.post("/predict/batch", json={"trips": [VALID_TRIP, VALID_TRIP]}).json()

        assert batch["count"] == 2
        for prediction in batch["predictions"]:
            assert prediction["predicted_duration_min"] == pytest.approx(single, abs=1e-6)

    def test_batch_ids_are_unique(self, client):
        batch = client.post("/predict/batch", json={"trips": [VALID_TRIP] * 3}).json()
        ids = [p["request_id"] for p in batch["predictions"]]
        assert len(set(ids)) == 3


class TestInputValidation:
    def test_empty_body_is_rejected(self, client):
        response = client.post("/predict", json={})
        assert response.status_code == 422
        assert response.json()["error"] == "validation_error"

    def test_missing_required_field_names_the_field(self, client):
        payload = {k: v for k, v in VALID_TRIP.items() if k != "pickup_latitude"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        assert any(e["field"] == "pickup_latitude" for e in response.json()["errors"])

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("pickup_latitude", 91.0),
            ("pickup_longitude", -200.0),
            ("dropoff_latitude", 0.0),
            ("traffic_index", 1.5),
            ("traffic_index", -0.1),
            ("passenger_count", 0),
            ("passenger_count", 99),
            ("temperature_c", 500.0),
        ],
    )
    def test_out_of_range_values_are_rejected(self, client, field, value):
        response = client.post("/predict", json=dict(VALID_TRIP, **{field: value}))
        assert response.status_code == 422

    def test_wrong_type_is_rejected(self, client):
        response = client.post("/predict", json=dict(VALID_TRIP, traffic_index="heavy"))
        assert response.status_code == 422

    def test_malformed_datetime_is_rejected(self, client):
        response = client.post("/predict", json=dict(VALID_TRIP, pickup_datetime="not-a-date"))
        assert response.status_code == 422

    def test_unknown_weather_category_is_rejected(self, client):
        response = client.post("/predict", json=dict(VALID_TRIP, weather_condition="Hail"))
        assert response.status_code == 422

    def test_unexpected_field_is_rejected(self, client):
        """extra=forbid: a typo'd field must not be silently ignored."""
        response = client.post("/predict", json=dict(VALID_TRIP, trafic_index=0.5))
        assert response.status_code == 422

    def test_partial_weather_is_rejected(self, client):
        """The serving-side twin of BR_PARTIAL_WEATHER_RECORD."""
        payload = dict(VALID_TRIP)
        del payload["wind_kph"]
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        assert "weather must be supplied in full" in str(response.json()["errors"])

    def test_precipitation_contradicting_clear_weather_is_rejected(self, client):
        response = client.post(
            "/predict", json=dict(VALID_TRIP, weather_condition="Clear", precipitation_mm=8.0)
        )
        assert response.status_code == 422

    def test_empty_batch_is_rejected(self, client):
        assert client.post("/predict/batch", json={"trips": []}).status_code == 422

    def test_oversized_batch_is_rejected(self, client):
        response = client.post("/predict/batch", json={"trips": [VALID_TRIP] * 501})
        assert response.status_code == 422


class TestFeedback:
    def test_feedback_joins_to_the_original_prediction(self, client):
        prediction = client.post("/predict", json=VALID_TRIP).json()
        response = client.post(
            "/feedback",
            json={"request_id": prediction["request_id"], "actual_duration_min": 42.0},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["recorded"] is True
        assert body["predicted_duration_min"] == pytest.approx(
            prediction["predicted_duration_min"], abs=0.01
        )
        assert body["absolute_error_min"] == pytest.approx(
            abs(42.0 - prediction["predicted_duration_min"]), abs=0.01
        )

    def test_feedback_for_an_unknown_request_is_404(self, client):
        response = client.post(
            "/feedback", json={"request_id": "does-not-exist", "actual_duration_min": 10.0}
        )
        assert response.status_code == 404
        assert response.json()["recorded"] is False

    def test_impossible_actual_duration_is_rejected(self, client):
        response = client.post(
            "/feedback", json={"request_id": "any", "actual_duration_min": -5.0}
        )
        assert response.status_code == 422


class TestPredictionLogging:
    def test_predictions_are_persisted_for_monitoring(self, client):
        from src.monitoring.prediction_log import get_prediction_log

        before = get_prediction_log().counts()["total"]
        client.post("/predict/batch", json={"trips": [VALID_TRIP] * 4})
        after = get_prediction_log().counts()

        assert after["total"] == before + 4
        assert after["labelled"] >= 0
