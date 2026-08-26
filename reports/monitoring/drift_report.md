# Drift Simulation Report

- **Generated:** 2026-08-26T03:29:04+00:00
- **Scenario:** `festival_surge` vs reference `baseline`
- **Window size:** 20,000 trips each (seed 202)
- _City-wide event: congestion spikes and trips take materially longer._

## Performance drift (concept)

| Window | MAE (min) | RMSE (min) | R² | p90 abs err |
| --- | --- | --- | --- | --- |
| Reference (baseline) | 3.51 | 5.31 | 0.900 | 8.11 |
| Current (festival_surge) | 10.88 | 15.79 | 0.678 | 24.56 |

MAE moved **x3.0983** under the drift → **🔴 fail**.

## Feature drift (covariate)

Level 3 checks against the training baseline: 8 pass · 6 warn · 0 fail → **🔴 fail**.

Flagged columns: `trip_duration_min`, `avg_speed_kmph`, `traffic_index`, `temperature_c`, `wind_kph`, `weather_condition`
