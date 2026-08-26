# Retraining Trigger Decision

## Decision: 🔴 RETRAIN

- **Generated:** 2026-08-26T03:30:25+00:00
- **Drift scenario:** `festival_surge`
- **Failing signals:** 3 (trigger at 1)

> RETRAIN: 3 signal(s) failed (performance, feature_drift, live_error), meeting the trigger of 1. Regenerate training data covering the new regime and rerun `dvc repro train`.

## Signals

| Signal | Status |
| --- | --- |
| performance (concept drift) | fail |
| feature_drift (covariate) | fail |
| live_error (serving log) | fail |

## Why

- performance [fail]: MAE 3.51 → 10.88 min (x3.0983) under `festival_surge`.
- feature_drift [fail]: 6 column(s) flagged by Level 3 checks: trip_duration_min, avg_speed_kmph, traffic_index, temperature_c, wind_kph, weather_condition.
- live_error [fail]: 112.83 min MAE over 60 labelled prediction(s).
