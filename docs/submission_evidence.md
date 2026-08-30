# ETA Prediction ML Pipeline — Submission Evidence

**Course:** Machine Learning Engineering (PCAM\* ZC412) — EC-1 Mini-Project  
**Flavor:** A — Delivery / Ride ETA Prediction  
**Team:**

| Name | ID |
| --- | --- |
| Santosh | 2025paml516 |
| Sharavanan | 2025paml517 |
| Chandra | 2025paml518 |

---

## Submission Checklist

| # | Deliverable | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Versioned dataset and pipeline code with commit history | ✅ Done | `dvc.lock`, `dvc.yaml`, GitHub repo |
| 2 | Experiment tracking logs and model comparison report | ✅ Done | `reports/model_comparison.md`, `mlruns/` |
| 3 | Deployed model with working API endpoint and sample test calls | ✅ Done | `src/api/`, `docs/api/curl_examples.md`, `docs/api/postman_collection.json` |
| 4 | Monitoring log, drift-simulation report, retraining trigger design | ✅ Done | `reports/monitoring/drift_report.md`, `reports/monitoring/retraining_decision.md` |
| 5 | README with architecture diagram, setup instructions, demo | ✅ Done | `README.md` |

---

## Week 1 / M2 — Data Engineering

### Command to reproduce
```powershell
dvc repro generate validate split features profile
```

### What this does
1. **Generate** — creates 150,450 synthetic NYC trip records with deliberate defects planted for validation testing
2. **Validate** — runs 4-level validation (schema → range → statistical → business rules); quarantines fatal rows, carries repairable nulls forward
3. **Split** — temporal train/val/test split (no shuffle); train < val < test in time order
4. **Features** — builds 44 features from 13 raw inputs; fitted on train partition only; persists artefact for serving
5. **Profile** — saves baseline statistical profile for Level 3 drift checks in Week 4

### Validation results — `reports/validation/validation_report.md`

| Metric | Value |
| --- | --- |
| Rows ingested | 150,450 |
| Rows validated | 144,959 |
| Rows quarantined | 5,491 |
| Quarantine rate | 3.65% (threshold: 15%) |
| Schema contract (Level 1) | PASSED |
| Range checks (Level 2) | 4,901 rows rejected |
| Business rules (Level 4) | 590 rows rejected |

**Quarantine reasons:**

| Reason | Rows | Type |
| --- | --- | --- |
| `MISSING_GPS` | 1,796 | Fatal |
| `DURATION_OUT_OF_RANGE` | 991 | Fatal |
| `INVALID_TIMESTAMP_ORDER` | 889 | Fatal |
| `GPS_OUT_OF_BOUNDS` | 593 | Fatal |
| `BR_PRECIPITATION_WITHOUT_WET_WEATHER` | 590 | Fatal — business rule |
| `DUPLICATE_TRIP_ID` | 450 | Fatal |
| `SPEED_OUT_OF_RANGE` | 182 | Fatal |

**Repairable nulls carried forward for imputation (not dropped):**

| Field | Null rows |
| --- | --- |
| `weather_condition` | 1,150 |
| `temperature_c` | 1,150 |
| `precipitation_mm` | 1,150 |
| `wind_kph` | 1,150 |
| `passenger_count` | 586 |

### Feature pipeline — `models/feature_pipeline/feature_spec.json`

| Property | Value |
| --- | --- |
| Feature pipeline version | 1.0.0 |
| Total features engineered | 44 |
| Raw inputs required | 13 |
| Zone clusters (NYC) | 20 |
| Global speed prior | 20.02 km/h |
| Fitted on rows | 101,471 (train only) |
| Imputation strategy — weather | month_mode |
| Imputation strategy — numeric | month_median |
| Imputation strategy — passenger count | median |

**Feature groups:**

| Group | Features |
| --- | --- |
| Geometry | haversine_km, manhattan_km, bearing_sin/cos, raw coordinates |
| Cyclical time | hour, day-of-week, month encoded as sin/cos pairs |
| Calendar flags | is_weekend, is_rush_hour, is_night, is_holiday |
| Conditions | traffic_index, temperature_c, precipitation_mm, wind_kph, weather_severity, weather one-hot |
| Trip metadata | passenger_count, vendor one-hot, store_and_fwd_flag, imputation flags |
| Learned zone signals | zone centroids, same_zone, zone_hour_speed_prior, route_speed_prior, expected_duration |

### Dataset split — `reports/features/feature_summary.json`

| Partition | Rows | Target mean | Target std |
| --- | --- | --- | --- |
| Train | 101,471 | 24.45 min | 16.68 min |
| Val | 21,744 | 24.23 min | 16.53 min |
| Test | 21,744 | 24.71 min | 16.81 min |

### Key design decisions
- **Fatal vs repairable defects are handled differently.** A missing GPS coordinate cannot be trusted as a training target — quarantined with a reason code. Missing weather is a repairable covariate — carried forward and imputed in the feature pipeline, never silently dropped.
- **Imputation values come from the training partition only.** Fill values are learned, persisted in `feature_spec.json`, and reloaded at serving — they are never recomputed from production data (M2 2.6.4).
- **The same `FeaturePipeline` class is shared by training and the API.** `tests/test_skew.py` asserts that a single-row API transform is bit-identical to the batched training transform.
- **DVC versions the dataset.** `dvc repro --force` twice produces byte-identical outputs. The seed lives in `params.yaml`.

---

## Week 2 / M3 — Experimentation

### Command to reproduce
```powershell
dvc repro train
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Model comparison — `reports/model_comparison.md`

| Model | Train MAE | Val MAE | Test MAE | Test R² | Test p90 AE | Fit time |
| --- | --- | --- | --- | --- | --- | --- |
| **lightgbm** ✅ selected | 3.408 | **3.445** | 3.540 | 0.899 | 8.16 min | 94.9s |
| random_forest | 2.602 | 3.492 | 3.575 | 0.897 | 8.26 min | 192.5s |
| ridge | 3.669 | 3.644 | 3.746 | 0.881 | 8.45 min | 11.2s |
| baseline (median) | 12.030 | 11.904 | 12.137 | -0.082 | 28.56 min | 0.1s |

### Best model — `reports/metrics.json`

| Metric | Value |
| --- | --- |
| Best model | LightGBM |
| Selected by | val_mae |
| Val MAE | 3.4445 min |
| Test MAE | 3.5404 min |
| Test R² | 0.8993 |
| Test MAPE | 14.82% |
| Test p90 absolute error | 8.16 min |
| Improvement over baseline | 70.8% reduction in test MAE |

### MLflow tracking
- All 4 candidates tracked in MLflow (run IDs in `models/trained/best_model_metadata.json`)
- Each run logs: parameters, metrics across all partitions, residual plots, feature importance, model artefact, dataset MD5 hashes, git commit
- Best run: `8cbc0d4715b14fabaec5e50fb85f44ed`
- MLflow experiment logs are available in `mlruns/`, and the comparison summary is exported in `reports/model_comparison.md`

### Reproducing the selected run
```powershell
python -m src.models.reproduce_run --run-id 8cbc0d4715b14fabaec5e50fb85f44ed
```

### Key design decisions
- **Selection happens on validation, never on test.** A model chosen by its test score turns that score into a biased estimate of production error. `select_best` raises an exception if configured to select on test.
- **Cross-validation uses `TimeSeriesSplit`.** Shuffled K-fold would train on later trips to predict earlier ones — the leakage the temporal split was designed to prevent.
- **Ridge is wrapped in `Pipeline([StandardScaler, Ridge])`.** The L2 penalty is scale-dependent; the scaler is refitted per CV fold and travels with the serialised model.
- **Log1p target transformation.** Trip duration is right-skewed; modelling `log1p(duration)` shrinks the long-trip error tail and lowers MAE without extra capacity.

---

## Week 3 / M4 — Packaging and Serving

### Commands
```powershell
# Local
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Container
podman build --format docker -t eta-api:latest -f Containerfile .
podman-compose up --build
```

### API endpoints

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | GET | Liveness — answers even when model failed to load |
| `/ready` | GET | Readiness — 503 until model AND feature pipeline are loaded |
| `/model/info` | GET | Model name, version, run ID, feature contract |
| `/metrics` | GET | Prometheus exposition |
| `/predict` | POST | Single trip ETA prediction |
| `/predict/batch` | POST | Up to 500 trips in one vectorised call |
| `/feedback` | POST | Report actual duration for a served prediction |

### Sample test calls

| Test call | File / command | What it demonstrates |
| --- | --- | --- |
| curl request examples | `docs/api/curl_examples.md` | Direct sample request/response calls for each endpoint |
| Postman collection | `docs/api/postman_collection.json` | Browser-friendly / importable API tests |
| Manual browser test | `http://127.0.0.1:8000/docs` | Swagger UI can execute `GET /health`, `GET /ready`, `POST /predict`, and `POST /feedback` |

### Sample request — `POST /predict`
```json
{
  "pickup_datetime": "2024-07-15T08:30:00",
  "pickup_latitude": 40.7549,
  "pickup_longitude": -73.9840,
  "dropoff_latitude": 40.6413,
  "dropoff_longitude": -73.7781,
  "passenger_count": 2,
  "vendor_id": 1,
  "store_and_fwd_flag": "N",
  "weather_condition": "Clear",
  "temperature_c": 24.5,
  "precipitation_mm": 0.0,
  "wind_kph": 12.0,
  "traffic_index": 0.62
}
```

### Sample response — `POST /predict`

```json
{
  "predicted_duration_min": 29.44,
  "predicted_dropoff_time": "2024-07-15T08:59:26.400000",
  "distance_km": 13.297,
  "model_name": "eta-predictor",
  "model_version": "run-8cbc0d47",
  "request_id": "example-request-id",
  "latency_ms": 12.5,
  "weather_imputed": false
}
```

### Latency and throughput — `reports/api/latency_report.md`

| Metric | Value |
| --- | --- |
| Throughput | 71.39 req/s (single worker) |
| p50 latency | 218.79 ms |
| p95 latency | 299.66 ms |
| p99 latency | 363.95 ms |
| Batch of 250 trips | 0.313 ms per trip (3,192 trips/s) |

### Edge case and validation handling

| Case | Behaviour |
| --- | --- |
| Empty body | 422 — all required fields listed |
| `traffic_index > 1.0` | 422 — out of range |
| Unknown `weather_condition` (e.g. "Hail") | 422 — not in allowed set |
| Partial weather fields | 422 — all 4 weather fields must be supplied together |
| Rain under clear sky | 422 — business rule cross-field check |
| Weather fields all omitted | 200 — imputed from training values; `weather_imputed: true` in response |
| Typo'd field name | 422 — extra fields forbidden (`extra=forbid`) |

### Key design decisions
- **Liveness and readiness are separate endpoints.** A container that is running but whose model failed to load is alive and *not* ready.
- **Validation reuses the training-time bounds.** Pydantic reads the same `validate.bounds` block used at ingestion — the API refuses values that training would have quarantined.
- **Every prediction is logged to SQLite** with features, model version and latency. `/feedback` joins the observed duration back by `request_id` for Week 4 monitoring.
- **Container ships only `requirements-serving.txt`** — excludes DVC, pytest, ruff, Streamlit. Image size: 1.01 GB vs 1.39 GB with full dependencies.

---

## Week 4 / M5 — Monitoring and Drift

### Commands
```powershell
dvc repro drift_simulation
dvc repro retraining_decision
podman-compose up --build   # Prometheus (:9090) + Grafana (:3000)
```

### Drift simulation — `reports/monitoring/drift_report.md`

**Scenario:** `festival_surge` — city-wide event, congestion spikes and trips take materially longer.

**Performance drift (concept drift):**

| Window | MAE (min) | RMSE (min) | R² | p90 abs err |
| --- | --- | --- | --- | --- |
| Reference (baseline) | 3.51 | 5.31 | 0.900 | 8.11 |
| Current (festival_surge) | 10.88 | 15.79 | 0.678 | 24.56 |
| **Change** | **×3.10** | **×2.97** | **−0.222** | **×3.03** |

Status: 🔴 **FAIL**

**Feature drift (covariate drift):**

| Column | Status | KS Statistic | Baseline mean | Observed mean |
| --- | --- | --- | --- | --- |
| `trip_duration_min` | warn | 0.283 | 24.45 min | 39.36 min |
| `avg_speed_kmph` | warn | 0.514 | 20.63 km/h | 13.17 km/h |
| `traffic_index` | warn | 0.406 | 0.471 | 0.701 |
| `temperature_c` | warn | 0.092 | 14.28°C | 13.28°C |
| `wind_kph` | warn | 0.059 | 13.11 | 12.80 |
| `weather_condition` | warn | chi²=346.7 | — | — |

Overall: 8 pass · 6 warn · 0 fail → 🔴 **FAIL** (≥2 flagged columns threshold)

### Retraining decision — `reports/monitoring/retraining_decision.md`

| Signal | Status | Detail |
| --- | --- | --- |
| performance | 🔴 fail | MAE ×3.10 under festival_surge |
| feature_drift | 🔴 fail | 6 columns flagged by Level 3 checks |
| live_error | ⚪ insufficient | No serving log yet (no predictions recorded) |

**Decision: 🔴 RETRAIN**  
2 signals failed (performance, feature_drift), meeting the trigger threshold of 1.

**Trigger rule:** retrain when at least `fail_signals_to_trigger` (=1) signals reach `fail`.  
All thresholds live in `params.yaml` under `monitoring` — changing a threshold invalidates only the two Week 4 stages.

### Monitoring infrastructure
- **Prometheus** scrapes `/metrics` endpoint (request rate, latency histograms, prediction count, feedback error)
- **Grafana** auto-loads the *ETA API Overview* dashboard from `monitoring/grafana/`
- **Prediction log** (SQLite) records every served prediction with features, model version, latency
- **`/feedback` endpoint** joins observed durations back to predictions by `request_id` for live MAE tracking

### Key design decisions
- **Two independent drift signals.** Performance drift catches concept drift (input→output relationship changed). Feature drift catches covariate drift (input distribution changed). A monitor watching only one is blind to the other failure mode.
- **Drift is a controlled intervention, not a guess.** The generating process is owned, so `festival_surge` shifts congestion by a known amount — the detected drift is ground truth the detector is checked against.
- **The trigger rule is intentionally boring.** "Retrain when ≥N signals fail" is auditable. A clever rule living in someone's head is not.
- **Live error signal abstains below `min_labelled`.** A cold monitoring database yields "insufficient" rather than a false vote on thin data.

---

## Design Decision Justification

### Model choice

| Decision | Justification |
| --- | --- |
| LightGBM selected as the best model | It achieved the lowest validation MAE while remaining strong on test data, so it generalises better than the baseline, ridge, and random forest candidates. |
| Validation used for selection | Test data is reserved for the final unbiased report; choosing on test would inflate the reported generalisation performance. |
| Log-transformed target during training | Trip duration is right-skewed, and `log1p(duration)` reduces the long-trip error tail without changing the serving contract. |

### Drift-detection approach

| Decision | Justification |
| --- | --- |
| Synthetic `festival_surge` drift | The generator is owned, so the drift is a controlled intervention rather than an approximation; this makes the monitoring evaluation ground-truthable. |
| Two drift signals | Performance drift catches concept drift, while feature drift catches covariate drift; monitoring only one would miss the other failure mode. |
| Training baseline for Level 3 checks | The baseline is fitted on the training partition only, so drift is judged against the original data distribution rather than the batch under test. |

### Retraining trigger

| Decision | Justification |
| --- | --- |
| Retrain when at least `fail_signals_to_trigger` signals fail | The rule is explicit, auditable, and easy to explain in a demo. |
| Live error signal abstains when labelled data is insufficient | A cold monitoring store should not create a false alarm; abstaining is safer than voting on thin data. |
| Thresholds stored in `params.yaml` | Changing the trigger invalidates only the Week 4 monitoring stages, keeping the pipeline reproducible and traceable. |

---

## Test Coverage

| Test file | What it covers | Status |
| --- | --- | --- |
| `tests/test_validation.py` | 4-level validation, quarantine, business rules | ✅ Pass |
| `tests/test_features.py` | Feature pipeline contract, imputation, zone assignment | ✅ Pass |
| `tests/test_skew.py` | Training-serving skew guard (bit-identical transforms) | ✅ Pass |
| `tests/test_model_training.py` | Metric calculation, estimator structure, selection discipline | ✅ Pass |
| `tests/test_week2_integration.py` | Week 1→2 handoff: feature matrix dimensions, nulls, temporal order | ✅ Pass |
| `tests/test_monitoring.py` | Drift signals, retraining trigger rule (branch-by-branch) | ✅ Pass |
| `tests/test_statistical_validation.py` | Level 3 KS/chi² checks | ✅ Pass |
| `tests/test_api.py` | API contract, input validation, prediction logging, feedback | ⚠️ Requires `httpx` in venv |

Run all non-API tests:
```powershell
pytest -q tests/test_validation.py tests/test_features.py tests/test_model_training.py tests/test_skew.py tests/test_monitoring.py tests/test_week2_integration.py
```

Run full suite (after installing `httpx`):
```powershell
pytest
```

---

## Reproduce Everything

```powershell
# 1. Setup
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Run full pipeline
dvc repro

# 3. View metrics
dvc metrics show

# 4. Run tests
pytest

# 5. Start API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
# Open http://127.0.0.1:8000/docs

# 6. Run monitoring
dvc repro drift_simulation retraining_decision

# 7. Browse experiments
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
