# Retraining Trigger Decision

## Decision: 🔴 RETRAIN

- **Generated:** 2026-08-26T12:08:58+00:00
- **Drift scenario:** `festival_surge`
- **Failing signals:** 2 (trigger at 1)

> RETRAIN: 2 signal(s) failed (performance, feature_drift), meeting the trigger of 1. Regenerate training data covering the new regime and rerun `dvc repro train`.

## Signals

| Signal | Status |
| --- | --- |
| performance (concept drift) | fail |
| feature_drift (covariate) | fail |
| live_error (serving log) | insufficient |

## Why

- performance [fail]: MAE 3.51 → 10.88 min (x3.0983) under `festival_surge`.
- feature_drift [fail]: 6 column(s) flagged by Level 3 checks: trip_duration_min, avg_speed_kmph, traffic_index, temperature_c, wind_kph, weather_condition.
- live_error [insufficient]: no serving log found — signal abstained.
