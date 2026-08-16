# API Sample Calls

Start the service first:

```powershell
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Interactive docs are at `http://localhost:8000/docs`.

---

## Health and readiness

Liveness — answers even when the model failed to load:

```bash
curl -s http://localhost:8000/health
```

```json
{ "status": "ok", "service": "eta-prediction-api", "version": "0.1.0" }
```

Readiness — returns **503** until the model and feature pipeline are both loaded. This
is what the container healthcheck and any load balancer should poll:

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/ready
```

## Which model is serving

```bash
curl -s http://localhost:8000/model/info
```

Returns the registered name and version, the originating MLflow run, how the model was
selected (`val_mae`), its recorded metrics, and the full 44-column feature contract.

---

## Single prediction

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "wind_kph": 12.0
  }'
```

```json
{
  "predicted_duration_min": 38.91,
  "predicted_dropoff_time": "2024-07-15T09:08:54",
  "distance_km": 21.284,
  "model_name": "eta-predictor",
  "model_version": "3",
  "request_id": "9f2c...",
  "latency_ms": 6.42,
  "weather_imputed": false
}
```

Keep the `request_id`: it is what `/feedback` joins on.

### Weather omitted

Weather is optional. Leave all four fields out and the service imputes them with the
values learned from the training partition, flagging that it did so:

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pickup_datetime": "2024-11-02T18:15:00",
    "pickup_latitude": 40.7075,
    "pickup_longitude": -74.0113,
    "dropoff_latitude": 40.7736,
    "dropoff_longitude": -73.9566,
    "traffic_index": 0.85
  }'
```

The response carries `"weather_imputed": true`.

---

## Batch prediction

One featurisation and one vectorised model call for the whole payload, so per-trip cost
drops sharply with size:

```bash
curl -s -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "trips": [
      {
        "pickup_datetime": "2024-07-15T08:30:00",
        "pickup_latitude": 40.7549, "pickup_longitude": -73.9840,
        "dropoff_latitude": 40.6413, "dropoff_longitude": -73.7781,
        "traffic_index": 0.62
      },
      {
        "pickup_datetime": "2024-07-15T23:10:00",
        "pickup_latitude": 40.6928, "pickup_longitude": -73.9903,
        "dropoff_latitude": 40.8116, "dropoff_longitude": -73.9465,
        "traffic_index": 0.15
      }
    ]
  }'
```

Maximum batch size is 500; larger payloads are rejected with 422.

---

## Feedback

Report the duration that actually elapsed. This is what turns monitoring from drift
detection into error measurement:

```bash
curl -s -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{ "request_id": "PASTE_REQUEST_ID", "actual_duration_min": 41.5 }'
```

```json
{
  "request_id": "PASTE_REQUEST_ID",
  "actual_duration_min": 41.5,
  "predicted_duration_min": 38.91,
  "absolute_error_min": 2.59,
  "recorded": true
}
```

An unknown `request_id` returns 404 with `"recorded": false`.

---

## Error handling

Every rejection names the offending field and echoes a correlation id.

**Out-of-range coordinate:**

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{ "pickup_datetime": "2024-07-15T08:30:00",
        "pickup_latitude": 91.0, "pickup_longitude": -73.9840,
        "dropoff_latitude": 40.6413, "dropoff_longitude": -73.7781,
        "traffic_index": 0.62 }'
```

```json
{
  "error": "validation_error",
  "detail": "The request body failed validation.",
  "request_id": "3ab1...",
  "errors": [{ "field": "pickup_latitude", "message": "Input should be less than or equal to 40.95" }]
}
```

**Contradictory weather** — rain measured under a clear sky, the serving-side twin of the
Level 4 business rule from Week 1:

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{ "pickup_datetime": "2024-07-15T08:30:00",
        "pickup_latitude": 40.7549, "pickup_longitude": -73.9840,
        "dropoff_latitude": 40.6413, "dropoff_longitude": -73.7781,
        "traffic_index": 0.62,
        "weather_condition": "Clear", "temperature_c": 20.0,
        "precipitation_mm": 8.0, "wind_kph": 10.0 }'
```

**Half-supplied weather** — a partial record means the caller's own weather lookup
half-failed, so it is rejected rather than silently imputed:

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{ "pickup_datetime": "2024-07-15T08:30:00",
        "pickup_latitude": 40.7549, "pickup_longitude": -73.9840,
        "dropoff_latitude": 40.6413, "dropoff_longitude": -73.7781,
        "traffic_index": 0.62, "weather_condition": "Rain" }'
```

**Unknown field** — the schema forbids extras, so a typo is reported rather than ignored:

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{ "pickup_datetime": "2024-07-15T08:30:00",
        "pickup_latitude": 40.7549, "pickup_longitude": -73.9840,
        "dropoff_latitude": 40.6413, "dropoff_longitude": -73.7781,
        "trafic_index": 0.62 }'
```

| Status | Meaning |
| --- | --- |
| 200 | Prediction returned |
| 404 | `/feedback` for an unknown `request_id` |
| 422 | Request failed validation; `errors[]` names each field |
| 500 | Unexpected server error; quote the `request_id` |
| 503 | Model not loaded — the request was fine, the service is not |

---

## Metrics

Prometheus exposition, scraped in Week 4:

```bash
curl -s http://localhost:8000/metrics | grep eta_
```

| Metric | Meaning |
| --- | --- |
| `eta_requests_total` | Requests by method, endpoint and status |
| `eta_request_latency_seconds` | Latency histogram, for p95/p99 |
| `eta_predictions_total` | Trips scored, by model version |
| `eta_predicted_duration_minutes` | Prediction distribution, watched for drift |
| `eta_absolute_error_minutes` | Error once actuals arrive via `/feedback` |
| `eta_model_ready` | 1 when the model is loaded |

---

## Container

```powershell
podman build --format docker -t eta-api:latest -f Containerfile .

# Preferred: compose declares the healthcheck explicitly
podman-compose up --build

# Plain run - the healthcheck must be passed explicitly, see note below
podman run --rm -p 8000:8000 `
  --health-cmd "curl -fsS http://localhost:8000/ready || exit 1" `
  --health-interval 30s --health-start-period 40s --health-retries 3 `
  eta-api:latest
```

Two Podman quirks worth knowing:

- `--format docker` is required at build time. Podman defaults to the OCI image
  format, which has no HEALTHCHECK field, so the instruction is discarded with only a
  warning.
- Even with the healthcheck recorded in the image, `podman run` does not inherit it —
  the container reports `starting` forever. `podman-compose` declares it explicitly and
  works; a plain run needs `--health-cmd`. Verified: with it, the container reports
  `healthy` within 15s.

## Benchmark

```powershell
python -m scripts.loadtest --url http://localhost:8000 --requests 500 --concurrency 16
```

Writes [reports/api/latency_report.md](../../reports/api/latency_report.md).
