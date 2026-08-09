# Data Quality Dimensions

Measured on the ingested batch, one section per dimension of M2 Table 2.1.

| Dimension | Headline metric | Value |
| --- | --- | --- |
| Completeness | Rows with every required field | 98.8029% |
| Accuracy | Implausible implied speed (proxy) | 1.2635% |
| Consistency | Rows free of cross-field contradictions | 99.0169% |
| Timeliness | Batch span (days) | 365.989 |
| Validity | Rows within all range and domain rules | 98.8262% |
| Uniqueness | Distinct trip ids / rows | 0.997009 |

## Completeness — least populated fields

| Field | Non-null rate |
| --- | --- |
| `temperature_c` | 99.2011% |
| `wind_kph` | 99.2011% |
| `weather_condition` | 99.2037% |
| `precipitation_mm` | 99.2037% |
| `dropoff_latitude` | 99.3998% |

## Validity — violations by rule

| Rule | Rows |
| --- | --- |
| `DURATION_OUT_OF_RANGE` | 991 |
| `GPS_OUT_OF_BOUNDS` | 593 |
| `SPEED_OUT_OF_RANGE` | 182 |

## Consistency — violations by business rule

| Rule | Rows |
| --- | --- |
| `INVALID_TIMESTAMP_ORDER` | 889 |
| `BR_PRECIPITATION_WITHOUT_WET_WEATHER` | 590 |

## Uniqueness

- Rows: 150,450
- Distinct `trip_id`: 150,000
- Duplicate rows: 450

## Timeliness

- Oldest event: 2024-01-01T00:09:02
- Newest event: 2024-12-31T23:53:35
- Median record age relative to newest: 182.539 days

## Accuracy

Accuracy cannot be validated programmatically: no rule can tell whether a
recorded duration is the duration that actually elapsed. The proxy reported
above flags 1.2635% of rows as
physically implausible. Establishing true accuracy requires a sampling audit
against raw GPS traces, which is a process investment rather than a check.
