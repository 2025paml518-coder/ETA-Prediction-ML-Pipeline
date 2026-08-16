# Data Validation Report

Generated: 2026-08-16T06:10:42+00:00

## Four-level validation outcome

| Level | Check | Result |
| --- | --- | --- |
| 1 | Schema contract | passed |
| 2 | Range and domain | 4,901 rows rejected |
| 3 | Statistical vs baseline | skipped (no baseline supplied) |
| 4 | Business rules | 590 rows rejected |

## Volume

| Metric | Value |
| --- | --- |
| Rows ingested | 150,450 |
| Rows validated | 144,959 |
| Rows quarantined | 5,491 |
| Quarantine rate | 3.65% |
| Threshold | 15.00% |

## Quarantine reasons

| Reason code | Rows |
| --- | --- |
| `MISSING_GPS` | 1,796 |
| `DURATION_OUT_OF_RANGE` | 991 |
| `INVALID_TIMESTAMP_ORDER` | 889 |
| `GPS_OUT_OF_BOUNDS` | 593 |
| `BR_PRECIPITATION_WITHOUT_WET_WEATHER` | 590 |
| `DUPLICATE_TRIP_ID` | 450 |
| `SPEED_OUT_OF_RANGE` | 182 |

## Nulls carried forward for imputation

Repairable gaps are left intact here and filled by the feature pipeline using
values learned from the training partition only, then persisted for serving.

| Field | Null rows |
| --- | --- |
| `weather_condition` | 1,150 |
| `temperature_c` | 1,150 |
| `precipitation_mm` | 1,150 |
| `wind_kph` | 1,150 |
| `passenger_count` | 586 |

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
| `inconsistent_weather` | 600 |
| `impossible_speed` | 450 |
| `duplicate_trip_id` | 450 |

Total planted: **7,350** | Total quarantined: **5,491**

> Counts differ by design: a row can carry several defects and is reported
> under one reason code, and repairable defects (missing weather, missing
> passenger count) are carried forward for imputation rather than quarantined.
