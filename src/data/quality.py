"""Measurement of the six data quality dimensions (M2 section 2.2).

Data quality is not binary. Each dimension fails differently and demands a
different engineering response, so each is measured separately and reported
rather than collapsed into one "rows rejected" number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Fields whose absence makes a record unusable for training or inference.
REQUIRED_FIELDS: tuple[str, ...] = (
    "trip_id",
    "vendor_id",
    "pickup_datetime",
    "dropoff_datetime",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "traffic_index",
)

# Fields that may legitimately arrive empty and are imputed downstream.
IMPUTABLE_FIELDS: tuple[str, ...] = (
    "passenger_count",
    "weather_condition",
    "temperature_c",
    "precipitation_mm",
    "wind_kph",
)


def completeness(frame: pd.DataFrame) -> dict:
    """Proportion of records in which each expected field is populated."""
    per_column = {
        column: round(float(frame[column].notna().mean()), 6)
        for column in (*REQUIRED_FIELDS, *IMPUTABLE_FIELDS)
        if column in frame.columns
    }
    required = [per_column[c] for c in REQUIRED_FIELDS if c in per_column]
    return {
        "per_column_non_null_rate": per_column,
        "required_field_completeness": round(float(np.mean(required)), 6) if required else None,
        "fully_populated_row_rate": round(
            float(frame[list(REQUIRED_FIELDS)].notna().all(axis=1).mean()), 6
        ),
    }


def uniqueness(frame: pd.DataFrame, key: str = "trip_id") -> dict:
    """Distinct primary keys versus total rows (M2 Table 2.1 detection method)."""
    total = int(len(frame))
    distinct = int(frame[key].nunique(dropna=False))
    return {
        "key": key,
        "rows": total,
        "distinct_keys": distinct,
        "duplicate_rows": total - distinct,
        "uniqueness_ratio": round(distinct / total, 6) if total else None,
    }


def validity(frame: pd.DataFrame, reasons: pd.Series) -> dict:
    """Share of rows failing a range, domain or enumeration rule."""
    validity_codes = {
        "GPS_OUT_OF_BOUNDS",
        "DURATION_OUT_OF_RANGE",
        "SPEED_OUT_OF_RANGE",
        "INVALID_PASSENGER_COUNT",
        "INVALID_VENDOR",
        "UNKNOWN_WEATHER_CATEGORY",
    }
    failing = reasons.isin(validity_codes)
    return {
        "invalid_rows": int(failing.sum()),
        "validity_rate": round(float(1.0 - failing.mean()), 6) if len(frame) else None,
        "violations_by_rule": reasons[failing].value_counts().to_dict(),
    }


def consistency(frame: pd.DataFrame, reasons: pd.Series) -> dict:
    """Cross-field agreement: rows whose fields contradict one another."""
    consistency_codes = {
        "BR_PRECIPITATION_WITHOUT_WET_WEATHER",
        "BR_SNOW_ABOVE_FREEZING",
        "BR_PARTIAL_WEATHER_RECORD",
        "INVALID_TIMESTAMP_ORDER",
    }
    failing = reasons.isin(consistency_codes)
    return {
        "inconsistent_rows": int(failing.sum()),
        "consistency_rate": round(float(1.0 - failing.mean()), 6) if len(frame) else None,
        "violations_by_rule": reasons[failing].value_counts().to_dict(),
    }


def timeliness(frame: pd.DataFrame, column: str = "pickup_datetime") -> dict:
    """Record age relative to the newest event in the batch.

    Freshness is expressed against the batch itself rather than wall-clock time so
    the measurement stays reproducible; the same batch always reports the same age.
    """
    timestamps = pd.to_datetime(frame[column], errors="coerce").dropna()
    if timestamps.empty:
        return {"measurable": False}
    newest = timestamps.max()
    age_days = (newest - timestamps).dt.total_seconds() / 86400.0
    return {
        "measurable": True,
        "newest_event": newest.isoformat(),
        "oldest_event": timestamps.min().isoformat(),
        "span_days": round(float(age_days.max()), 3),
        "median_age_days": round(float(age_days.median()), 3),
        "share_within_30_days_of_newest": round(float((age_days <= 30).mean()), 6),
    }


def accuracy(frame: pd.DataFrame) -> dict:
    """Accuracy has no direct programmatic check (M2 Table 2.1).

    What can be done is a plausibility audit on the label: implied average speed is
    a physical quantity, so the share of trips outside a plausible band is a proxy
    for how often the recorded duration misrepresents the real journey.
    """
    if "avg_speed_kmph" not in frame.columns:
        return {"directly_measurable": False}
    speed = frame["avg_speed_kmph"].replace([np.inf, -np.inf], np.nan).dropna()
    if speed.empty:
        return {"directly_measurable": False}
    return {
        "directly_measurable": False,
        "proxy": "implied average speed plausibility",
        "implausible_speed_rate": round(float((~speed.between(3.0, 90.0)).mean()), 6),
        "median_implied_speed_kmph": round(float(speed.median()), 3),
        "note": "True label accuracy requires a sampling audit against GPS traces.",
    }


def assess(frame: pd.DataFrame, reasons: pd.Series) -> dict:
    """Measure all six dimensions for one ingested batch."""
    return {
        "completeness": completeness(frame),
        "accuracy": accuracy(frame),
        "consistency": consistency(frame, reasons),
        "timeliness": timeliness(frame),
        "validity": validity(frame, reasons),
        "uniqueness": uniqueness(frame),
    }


def to_markdown(assessment: dict) -> str:
    completeness_block = assessment["completeness"]
    worst = sorted(completeness_block["per_column_non_null_rate"].items(), key=lambda kv: kv[1])[:5]

    lines = [
        "# Data Quality Dimensions",
        "",
        "Measured on the ingested batch, one section per dimension of M2 Table 2.1.",
        "",
        "| Dimension | Headline metric | Value |",
        "| --- | --- | --- |",
        f"| Completeness | Rows with every required field | {completeness_block['fully_populated_row_rate']:.4%} |",
        f"| Accuracy | Implausible implied speed (proxy) | {assessment['accuracy'].get('implausible_speed_rate', float('nan')):.4%} |",
        f"| Consistency | Rows free of cross-field contradictions | {assessment['consistency']['consistency_rate']:.4%} |",
        f"| Timeliness | Batch span (days) | {assessment['timeliness'].get('span_days', 'n/a')} |",
        f"| Validity | Rows within all range and domain rules | {assessment['validity']['validity_rate']:.4%} |",
        f"| Uniqueness | Distinct trip ids / rows | {assessment['uniqueness']['uniqueness_ratio']:.6f} |",
        "",
        "## Completeness — least populated fields",
        "",
        "| Field | Non-null rate |",
        "| --- | --- |",
    ]
    lines += [f"| `{name}` | {rate:.4%} |" for name, rate in worst]

    lines += [
        "",
        "## Validity — violations by rule",
        "",
        "| Rule | Rows |",
        "| --- | --- |",
    ]
    lines += [
        f"| `{rule}` | {count:,} |"
        for rule, count in sorted(
            assessment["validity"]["violations_by_rule"].items(), key=lambda kv: -kv[1]
        )
    ]

    lines += [
        "",
        "## Consistency — violations by business rule",
        "",
        "| Rule | Rows |",
        "| --- | --- |",
    ]
    violations = assessment["consistency"]["violations_by_rule"]
    lines += (
        [f"| `{rule}` | {count:,} |" for rule, count in sorted(violations.items(), key=lambda kv: -kv[1])]
        if violations
        else ["| _none_ | 0 |"]
    )

    lines += [
        "",
        "## Uniqueness",
        "",
        f"- Rows: {assessment['uniqueness']['rows']:,}",
        f"- Distinct `trip_id`: {assessment['uniqueness']['distinct_keys']:,}",
        f"- Duplicate rows: {assessment['uniqueness']['duplicate_rows']:,}",
        "",
        "## Timeliness",
        "",
        f"- Oldest event: {assessment['timeliness'].get('oldest_event', 'n/a')}",
        f"- Newest event: {assessment['timeliness'].get('newest_event', 'n/a')}",
        f"- Median record age relative to newest: "
        f"{assessment['timeliness'].get('median_age_days', 'n/a')} days",
        "",
        "## Accuracy",
        "",
        "Accuracy cannot be validated programmatically: no rule can tell whether a",
        "recorded duration is the duration that actually elapsed. The proxy reported",
        f"above flags {assessment['accuracy'].get('implausible_speed_rate', 0):.4%} of rows as",
        "physically implausible. Establishing true accuracy requires a sampling audit",
        "against raw GPS traces, which is a process investment rather than a check.",
        "",
    ]
    return "\n".join(lines)
