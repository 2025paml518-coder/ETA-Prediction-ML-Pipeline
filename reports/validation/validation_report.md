# Data Validation Report

Generated: 2026-08-09T14:44:18+00:00

## Volume

| Metric | Value |
| --- | --- |
| Rows ingested | 150,450 |
| Rows validated | 145,549 |
| Rows quarantined | 4,901 |
| Quarantine rate | 3.26% |
| Threshold | 15.00% |
| Schema contract | PASSED |

## Quarantine reasons

| Reason code | Rows |
| --- | --- |
| `MISSING_GPS` | 1,796 |
| `DURATION_OUT_OF_RANGE` | 991 |
| `INVALID_TIMESTAMP_ORDER` | 889 |
| `GPS_OUT_OF_BOUNDS` | 593 |
| `DUPLICATE_TRIP_ID` | 450 |
| `SPEED_OUT_OF_RANGE` | 182 |

## Repairs applied

| Repair | Rows |
| --- | --- |
| Weather imputed (month mode/median) | 1,154 |
| Passenger count imputed (median) | 588 |

## Detection check

Defects planted by the generator versus rows caught by validation.

| Planted defect | Count |
| --- | --- |
| `missing_gps` | 1,800 |
| `missing_weather` | 1,200 |
| `invalid_timestamp` | 900 |
| `extreme_duration` | 750 |
| `out_of_bounds_gps` | 600 |
| `missing_passenger_count` | 600 |
| `impossible_speed` | 450 |
| `duplicate_trip_id` | 450 |

Total planted: **6,750** | Total quarantined: **4,901**

> Counts differ by design: a single row can carry several defects and is
> reported under one reason code, while repairable defects (missing weather,
> missing passenger count) are imputed instead of quarantined.
