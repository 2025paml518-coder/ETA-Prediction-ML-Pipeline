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
dvc repro          # generate -> validate -> split -> features -> profile
dvc push           # store data artefacts in the configured remote
pytest             # guards over validation, features, statistics and skew
ruff check .
```

`dvc repro` is deterministic: the seed lives in `params.yaml`, so the same commit
always rebuilds the same dataset. Changing a threshold in `params.yaml` invalidates
only the stages downstream of it.

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

## Dataset

Synthetic, generated by `src/data/generate_synthetic.py` from a documented
data-generating process over an NYC-shaped geography (mixture-of-Gaussians hotspots
across Manhattan, Brooklyn, Queens, the Bronx, JFK and LaGuardia).

Synthetic data was chosen over the Kaggle NYC Taxi Trip Duration dump because the
brief requires weather and traffic features that the public dataset does not contain,
and because Week 4 drift simulation needs a *controllable* process — owning the
generator makes a festival surge a ground-truth intervention rather than a guess.
Rationale in full: [docs/design_decisions.md](docs/design_decisions.md).

## Roadmap

- [x] **Week 1 · M2** — ingestion, validation, feature pipeline, DVC versioning
- [ ] **Week 2 · M3** — MLflow experiments, model comparison, run reproduction
- [ ] **Week 3 · M4** — FastAPI service, container image, latency benchmarks
- [ ] **Week 4 · M5** — drift simulation, Prometheus/Grafana monitoring, retraining trigger

## References

- T1: Robert Crowe et al., *Machine Learning Production Systems*, O'Reilly, 2024.
- T2: Andriy Burkov, *Machine Learning Engineering*, 2020.
- R1: Andrew P. McMahon, *Machine Learning Engineering with Python*, 2nd ed., Packt, 2023.

Libraries: pandas, NumPy, pandera, scikit-learn, DVC. NYC geography and monthly
climatology values are approximations of public NOAA / NYC OpenData figures.
