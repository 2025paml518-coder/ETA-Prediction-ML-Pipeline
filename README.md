# ETA-Prediction-ML-Pipeline

End-to-end machine learning pipeline for **delivery / ride ETA prediction**, built for
the BITS Pilani WILP course **Machine Learning Engineering (PCAM\* ZC412)** — EC-1
Mini-Project, Flavor A.

The system ingests trip data, validates it, engineers time-, weather- and
location-based features, trains and compares models to predict trip duration, serves
the best model behind a REST API, and monitors it for drift with an automated
retraining trigger.

## Team

| Name | ID |
| --- | --- |
| Santosh | 2025paml516 |
| Sharavanan | 2025paml517 |
| Chandra | 2025paml518 |

## Architecture

```mermaid
flowchart LR
    subgraph W1["Week 1 · M2 — Data Engineering"]
        GEN[Synthetic trip<br/>generator] --> RAW[(raw<br/>trips)]
        RAW --> VAL{{Schema +<br/>statistical<br/>validation}}
        VAL -->|fatal defects| QUAR[(quarantine<br/>+ reason codes)]
        VAL -->|repairable| CLEAN[(validated<br/>trips)]
        CLEAN --> SPLIT[Temporal split<br/>train / val / test]
        SPLIT --> FEAT[FeaturePipeline<br/>fit on train only]
        SPLIT --> PROF[Baseline profile<br/>Level 3 reference]
        PROF -.->|next batch| VAL
        FEAT --> PROC[(feature<br/>tables)]
        FEAT --> FART[/feature pipeline<br/>artefact/]
    end

    subgraph W2["Week 2 · M3 — Experimentation"]
        PROC --> TRAIN[Train and compare<br/>Ridge · RF · LightGBM]
        TRAIN --> MLF[(MLflow<br/>tracking)]
        MLF --> REG[/Model registry<br/>eta-predictor/]
    end

    subgraph W3["Week 3 · M4 — Serving"]
        REG --> API[FastAPI service<br/>/predict · /feedback]
        FART --> API
        API --> CONT[Container image<br/>Podman]
    end

    subgraph W4["Week 4 · M5 — Monitoring"]
        API --> PLOG[(prediction log)]
        PLOG --> DRIFT[Evidently<br/>drift reports]
        PLOG --> PROM[Prometheus<br/>+ Grafana]
        DRIFT --> TRIG{{Retraining<br/>trigger}}
        TRIG -->|fires| TRAIN
    end
```

DVC drives the Week 1 stages, and the same `dvc repro` invocation is what the Week 4
retraining trigger calls to rebuild the model.

## Repository layout

```
├── params.yaml               # single source of truth for every stage
├── dvc.yaml / dvc.lock       # pipeline definition + reproducible state
├── src/
│   ├── config.py             # path + parameter loading
│   ├── utils/                # geo maths, NYC calendar, logging
│   ├── data/                 # generation, validation, schema, splitting
│   └── features/             # the shared FeaturePipeline
├── tests/                    # validation, feature and skew guards
├── reports/                  # graded artefacts, kept in Git
└── data/                     # DVC-tracked, not in Git
```

## Setup

Requires **Python 3.11** and, from Week 3, **Podman**.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Exact resolved versions are pinned in `requirements.lock.txt`.

## Reproduce the pipeline

```powershell
dvc repro          # generate -> validate -> split -> features -> profile -> train
dvc push           # store data artefacts in the configured remote
dvc metrics show   # headline model metrics
pytest             # guards over validation, features, statistics, skew and models
ruff check .
```

`dvc repro` is deterministic: the seed lives in `params.yaml`, so the same commit
always rebuilds the same dataset. Changing a threshold in `params.yaml` invalidates
only the stages downstream of it.

Browse the experiments with `mlflow ui --backend-store-uri mlruns`.

## Week 1 (M2) — data engineering results

| Metric | Value |
| --- | --- |
| Rows generated | 150,450 |
| Rows validated | 145,549 |
| Rows quarantined | 4,901 (3.26%) |
| Schema contract | PASSED |
| Engineered features | 43 |
| Train / val / test | 101,885 / 21,832 / 21,832 |

Full breakdown: [reports/validation/validation_report.md](reports/validation/validation_report.md).

### Data quality handling

Validation is organised as the four levels of M2 section 2.5:

| Level | Check | Implementation |
| --- | --- | --- |
| 1 | Schema — columns, dtypes, required fields | pandera contract in [src/data/schema.py](src/data/schema.py) |
| 2 | Range and domain — bounds, allowed values | [src/data/validate.py](src/data/validate.py) |
| 3 | Statistical — KS, chi-squared and mean shift vs. a training baseline | [src/data/statistical_validation.py](src/data/statistical_validation.py) |
| 4 | Business rules — compound, cross-field constraints | [src/data/validate.py](src/data/validate.py) |

Level 3 needs a baseline built from the training partition, which is downstream of
validation, so it reports `skipped` on the first pass over a fresh dataset and runs on
every batch ingested afterwards.

Defects are then split into two classes, because treating them identically is how
pipelines quietly lose data:

| Class | Examples | Action |
| --- | --- | --- |
| **Fatal** | missing GPS, dropoff before pickup, impossible speed, duplicate id, cross-field contradiction | Quarantined with a reason code — never silently dropped |
| **Repairable** | missing weather, missing passenger count | Nulls carried forward and imputed by the feature pipeline, with the imputation recorded as a boolean feature |

The stage aborts the pipeline if the quarantine rate exceeds 15%: a spike means the
upstream feed changed shape, which is a different failure from ordinary noise.

All six data quality dimensions are measured per batch rather than collapsed into a
single rejection count — see
[reports/validation/data_quality_dimensions.md](reports/validation/data_quality_dimensions.md).
Accuracy is reported as *not directly measurable*, with implied-speed plausibility as a
proxy, because no rule can tell whether a recorded duration is the one that elapsed.

### Engineered features

| Group | Features |
| --- | --- |
| Geometry | haversine / manhattan distance, bearing (sin, cos), raw pickup + dropoff coordinates |
| Cyclical time | hour, day-of-week and month encoded as sin/cos pairs |
| Calendar | is_weekend, is_rush_hour, is_night, is_holiday |
| Conditions | traffic index, temperature, precipitation, wind, weather severity + one-hot |
| Trip metadata | passenger count, vendor one-hot, store-and-forward flag, imputation flags |
| Learned zone signals | zone centroids, same-zone flag, zone-hour and route speed priors, expected duration |

### Preventing training–serving skew

Both the training pipeline and the API import the same `FeaturePipeline` class and
load the same persisted artefact (`speed_priors.json`, `feature_spec.json`).
`tests/test_skew.py` asserts that a single-row transform through a reloaded pipeline is
bit-identical to the batched training transform, and `feature_spec.json` is checked on
load so a stale artefact fails fast instead of silently reordering columns.

The artefact is plain JSON, not a pickle. Zone assignment is nearest-centroid, so only
the centroids need to be stored — which keeps the serving path free of arbitrary-code
deserialisation and decouples the container from the scikit-learn version used to train.

Imputation values live in the same artefact. They are learned from the training
partition, persisted, and applied inside `transform`, so serving never recomputes a
fill value from production data.

### Reproducibility

Running `dvc repro --force` twice produces byte-identical output for every data and
model artefact:

| Artefact | Stable across rebuilds |
| --- | --- |
| `trips_raw.parquet`, `trips_validated.parquet`, `quarantined_trips.parquet` | yes |
| `train/val/test.parquet`, `*_features.parquet` | yes |
| `models/feature_pipeline/` | yes |
| `trips_raw.meta.json`, `reports/validation/` | no — they embed a generation timestamp as provenance |

## Week 2 (M3) — experimentation results

Four candidates, tuned with `RandomizedSearchCV` over `TimeSeriesSplit` folds and
tracked in MLflow:

| Model | Train MAE | Val MAE | Test MAE | Test R² | Test p90 AE | Fit time |
| --- | --- | --- | --- | --- | --- | --- |
| **lightgbm** | 3.392 | **3.474** | 3.565 | 0.899 | 8.22 | 145.5s |
| random_forest | 2.611 | 3.509 | 3.596 | 0.898 | 8.25 | 215.6s |
| ridge | 3.894 | 3.828 | 3.951 | 0.882 | 8.60 | 10.6s |
| baseline (median) | 12.030 | 11.904 | 12.137 | -0.082 | 28.56 | 0.1s |

LightGBM cuts test MAE from the median baseline's 12.14 min to **3.56 min**, a 70.6%
improvement — that margin is what justifies operating a model at all. Full analysis:
[reports/model_comparison.md](reports/model_comparison.md).

Note the train-to-validation spread: Random Forest reaches 2.611 train MAE but only
3.509 on validation, while LightGBM moves 3.392 → 3.474. The forest is memorising the
training period; LightGBM generalises, and that is why it wins despite a nearly
identical headline score.

### Selection discipline

The winner is chosen on **validation**, never on test. `select_best` raises if the
configured selection partition is `test`, because a model picked by its test score
turns that score into a biased estimate of production error — precisely the number
the report exists to give honestly. Test is scored once.

Cross-validation inside the search is `TimeSeriesSplit`, not shuffled K-fold, for the
same reason the train/val/test split is temporal: shuffled folds train on later trips
to predict earlier ones, inflating the score through a mechanism that does not exist
in production.

Ridge is wrapped in `Pipeline([StandardScaler, Ridge])`. Its L2 penalty is
scale-dependent and this matrix mixes latitudes near 40.7 with sine terms near 1, so
without scaling the penalty lands almost entirely on the small-scale features. Scaling
inside the pipeline also refits the scaler per fold and serialises it with the model.

### Reproducing a run

```powershell
python -m src.models.reproduce_run --run-id <run_id>
```

This reads *only* MLflow — parameters, tags and recorded dataset hashes — refits, and
compares against the originally logged metrics, exiting non-zero on divergence. The
selected LightGBM run reproduces to a maximum delta of **0.0** across all nine tracked
metrics. Every run records its git commit, working-tree cleanliness and the DVC md5 of
each dataset consumed, so a run identifies the exact code and data behind it.

## Week 3 (M4) — serving

```powershell
uvicorn src.api.main:app --host 0.0.0.0 --port 8000    # docs at /docs
podman build --format docker -t eta-api:latest -f Containerfile .
podman-compose up --build                              # serves on :8000
```

The image is **1.01 GB** and ships a 0.58 MB standalone model export. The first build
came out at 1.39 GB because the Containerfile copied `mlruns/` — 640 MB of every
candidate from every training run, to serve one model — and installed the full
`requirements.txt` including DVC, pytest, ruff, Streamlit and Plotly, none of which are
reachable from the request path. Verified: the containerised service returns
**exactly the same prediction** as the local one, confirming the export path is
equivalent to loading from the registry.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Liveness — answers even when the model failed to load |
| `GET /ready` | Readiness — 503 until model *and* feature pipeline are loaded |
| `GET /model/info` | Registered name, version, originating run, feature contract |
| `GET /metrics` | Prometheus exposition |
| `POST /predict` | One trip |
| `POST /predict/batch` | Up to 500 trips in one vectorised call |
| `POST /feedback` | Report the actual duration for a served prediction |

Sample calls: [docs/api/curl_examples.md](docs/api/curl_examples.md) ·
Postman: [docs/api/postman_collection.json](docs/api/postman_collection.json)

**Liveness and readiness are separate** because conflating them breaks both ways: a
single model-checking probe makes an orchestrator restart a healthy process over a
missing artefact, and a single process-checking probe routes traffic to a container
that cannot serve.

**Validation reuses the training-time bounds.** Pydantic reads the same
`validate.bounds` block the ingestion stage uses, so the service refuses values that
training would have quarantined — accepting them would be a form of skew. `extra` is
forbidden, so a typo like `trafic_index` is a 422 rather than a silently defaulted
feature and a confident, wrong ETA. The two Level 4 business rules from Week 1 are
re-enforced at the edge: rain under a clear sky, and half-supplied weather.

Weather is optional but all-or-nothing. Omit it entirely and the feature pipeline fills
it from the values learned in training, with `weather_imputed: true` in the response.
Supply *part* of it and the request is rejected, because a partial record means the
caller's own weather lookup malfunctioned.

Every served prediction is written to SQLite with its features, model version and
latency; `/feedback` joins the observed duration back onto it by request id. Monitoring
cannot be retrofitted onto traffic that was never recorded, so Week 4 starts with
history rather than an empty table.

### Latency and throughput

Measured against the running service with
`python -m scripts.loadtest --requests 500 --concurrency 16`
([full report](reports/api/latency_report.md)):

| Metric | Value |
| --- | --- |
| Throughput | 45.5 req/s (single worker) |
| p50 / p95 / p99 | 332 / 516 / 742 ms |
| Batch of 250 | 0.39 ms per trip — 2,551 trips/s |

The first run managed only 16.5 req/s at a p50 of 936 ms. Profiling — rather than
guessing — found that `transform` was assigning 49 columns onto a live DataFrame one at
a time, costing a pandas block-manager insert apiece, and that the prediction log was
opening a SQLite connection per request. Building the frame once and holding a
thread-local connection took the transform from 20.5 → 5.9 ms and logging from
5.8 → 0.13 ms. The optimised transform reproduces the committed feature tables
bit-for-bit across all 145,000 rows, so no retraining was needed.

The remaining p50 is queueing, not work: the server handles ~22 ms per request, and
FastAPI runs synchronous endpoints in a threadpool where the GIL serialises CPU-bound
featurisation. Real concurrency comes from more Uvicorn workers, each holding its own
copy of the model.

## Dataset

Synthetic, generated by `src/data/generate_synthetic.py` from a documented

Synthetic data was chosen over the Kaggle NYC Taxi Trip Duration dump because the
brief requires weather and traffic features that the public dataset does not contain,
and because Week 4 drift simulation needs a *controllable* process — owning the
generator makes a festival surge a ground-truth intervention rather than a guess.
Rationale in full: [docs/design_decisions.md](docs/design_decisions.md).

## Roadmap

- [x] **Week 1 · M2** — ingestion, validation, feature pipeline, DVC versioning
- [x] **Week 2 · M3** — MLflow experiments, model comparison, run reproduction
- [x] **Week 3 · M4** — FastAPI service, container image, latency benchmarks
- [ ] **Week 4 · M5** — drift simulation, Prometheus/Grafana monitoring, retraining trigger

## References

- T1: Robert Crowe et al., *Machine Learning Production Systems*, O'Reilly, 2024.
- T2: Andriy Burkov, *Machine Learning Engineering*, 2020.
- R1: Andrew P. McMahon, *Machine Learning Engineering with Python*, 2nd ed., Packt, 2023.

Libraries: pandas, NumPy, pandera, scikit-learn, DVC. NYC geography and monthly
climatology values are approximations of public NOAA / NYC OpenData figures.
