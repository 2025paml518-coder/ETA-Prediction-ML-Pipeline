"""Ingestion + four-level validation stage (M2 section 2.5).

Level 1  Schema      column presence, dtypes, required fields    -> src/data/schema.py
Level 2  Range       per-field bounds and allowed-value enumerations
Level 3  Statistical distribution comparison against a training baseline
                                               -> src/data/statistical_validation.py
Level 4  Business    compound rules spanning several fields

Level 3 only runs when a baseline profile is supplied. The baseline is built from the
training partition, which is downstream of this stage, so on the first pass through a
fresh dataset there is nothing to compare against; every batch ingested afterwards is
checked against it.

Two classes of defect are handled deliberately differently, because treating them the
same is how pipelines quietly lose data:

* **Fatal** - the row cannot be trusted as a training target. It is *quarantined* with
  a reason code rather than dropped, so the volume and mix of failures stays auditable.
* **Repairable** - only a covariate is missing, so the row is kept with its nulls
  intact. Imputation happens in the feature pipeline, where the fill values are learned
  from the training partition alone and persisted for reuse at serving (M2 2.6.4).

The stage fails the pipeline when the quarantine rate exceeds a threshold: a spike
means the upstream feed changed shape, which is a different problem from noisy data.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pandera.pandas as pa

from src.config import load_params, project_path
from src.data import quality, statistical_validation
from src.data.schema import REQUIRED_RAW_COLUMNS, build_validated_schema
from src.utils.geo import haversine_km
from src.utils.io import atomic_write_json, atomic_write_parquet, atomic_write_text
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

WEATHER_FIELDS = ["weather_condition", "temperature_c", "precipitation_mm", "wind_kph"]


class ValidationFailure(RuntimeError):
    """Raised when the batch is too damaged to be admitted into the pipeline."""


def _derive_trip_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["pickup_datetime"] = pd.to_datetime(frame["pickup_datetime"], errors="coerce")
    frame["dropoff_datetime"] = pd.to_datetime(frame["dropoff_datetime"], errors="coerce")
    frame["trip_duration_min"] = (
        frame["dropoff_datetime"] - frame["pickup_datetime"]
    ).dt.total_seconds() / 60.0
    straight_km = haversine_km(
        frame["pickup_latitude"],
        frame["pickup_longitude"],
        frame["dropoff_latitude"],
        frame["dropoff_longitude"],
    )
    frame["straight_line_km"] = straight_km
    with np.errstate(divide="ignore", invalid="ignore"):
        frame["avg_speed_kmph"] = np.where(
            frame["trip_duration_min"].to_numpy() > 0,
            straight_km / frame["trip_duration_min"].to_numpy() * 60.0,
            np.nan,
        )
    return frame


def _structural_rules(frame: pd.DataFrame, cfg: dict) -> list[tuple[str, pd.Series]]:
    """Levels 1 and 2: presence, type coherence, ranges and allowed values."""
    bounds = cfg["bounds"]
    lat_lo, lat_hi = bounds["latitude"]
    lon_lo, lon_hi = bounds["longitude"]
    dur_lo, dur_hi = bounds["trip_duration_min"]
    spd_lo, spd_hi = bounds["avg_speed_kmph"]
    pax_lo, pax_hi = bounds["passenger_count"]

    coord_cols = ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]

    return [
        ("DUPLICATE_TRIP_ID", frame["trip_id"].duplicated(keep="first")),
        ("MISSING_GPS", frame[coord_cols].isna().any(axis=1)),
        ("MISSING_TIMESTAMP", frame["pickup_datetime"].isna() | frame["dropoff_datetime"].isna()),
        ("INVALID_TIMESTAMP_ORDER", frame["trip_duration_min"] <= 0),
        (
            "GPS_OUT_OF_BOUNDS",
            ~(
                frame["pickup_latitude"].between(lat_lo, lat_hi)
                & frame["dropoff_latitude"].between(lat_lo, lat_hi)
                & frame["pickup_longitude"].between(lon_lo, lon_hi)
                & frame["dropoff_longitude"].between(lon_lo, lon_hi)
            ),
        ),
        ("DURATION_OUT_OF_RANGE", ~frame["trip_duration_min"].between(dur_lo, dur_hi)),
        ("SPEED_OUT_OF_RANGE", ~frame["avg_speed_kmph"].between(spd_lo, spd_hi)),
        (
            "INVALID_PASSENGER_COUNT",
            frame["passenger_count"].notna() & ~frame["passenger_count"].between(pax_lo, pax_hi),
        ),
        ("INVALID_VENDOR", ~frame["vendor_id"].isin(cfg["allowed_vendors"])),
        (
            "UNKNOWN_WEATHER_CATEGORY",
            frame["weather_condition"].notna()
            & ~frame["weather_condition"].isin(cfg["allowed_weather"]),
        ),
    ]


def _business_rules(frame: pd.DataFrame, cfg: dict) -> list[tuple[str, pd.Series]]:
    """Level 4: compound constraints that span fields.

    Every value involved is individually legal, so no schema or range check can see
    these. They are also the rules most likely to indicate a genuine upstream defect
    rather than ordinary noise.
    """
    rules_cfg = cfg["business_rules"]
    weather_present = frame[WEATHER_FIELDS].notna()

    return [
        # Rain measured while the sky is reported clear: the two fields disagree.
        (
            "BR_PRECIPITATION_WITHOUT_WET_WEATHER",
            frame["precipitation_mm"].notna()
            & (frame["precipitation_mm"] > 0)
            & frame["weather_condition"].notna()
            & ~frame["weather_condition"].isin(rules_cfg["wet_weather_conditions"]),
        ),
        (
            "BR_SNOW_ABOVE_FREEZING",
            frame["weather_condition"].eq("Snow")
            & frame["temperature_c"].notna()
            & (frame["temperature_c"] > rules_cfg["snow_max_temperature_c"]),
        ),
        # A weather record must arrive whole or not at all; a partial one means the
        # join against the weather feed silently half-failed.
        ("BR_PARTIAL_WEATHER_RECORD", weather_present.any(axis=1) & ~weather_present.all(axis=1)),
        # The vehicle never moved, yet the meter ran for an hour.
        (
            "BR_STATIONARY_LONG_TRIP",
            (frame["straight_line_km"] < rules_cfg["stationary_max_km"])
            & (frame["trip_duration_min"] > rules_cfg["stationary_max_minutes"]),
        ),
    ]


def _reason_codes(frame: pd.DataFrame, cfg: dict) -> pd.Series:
    """First failing rule per row, or an empty string when the row is clean."""
    reasons = pd.Series("", index=frame.index, dtype="object")
    for code, mask in (*_structural_rules(frame, cfg), *_business_rules(frame, cfg)):
        reasons = reasons.mask(reasons.eq("") & mask.fillna(True), code)
    return reasons


def validate_frame(
    frame: pd.DataFrame, params: dict, baseline: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Split a raw batch into (validated, quarantined, report)."""
    cfg = params["validate"]
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in frame.columns]
    if missing:
        raise ValidationFailure(f"Raw feed is missing required columns: {missing}")

    frame = _derive_trip_metrics(frame)
    structural_codes = [code for code, _ in _structural_rules(frame, cfg)]
    business_codes = [code for code, _ in _business_rules(frame, cfg)]
    reasons = _reason_codes(frame, cfg)
    fatal_mask = reasons.ne("")

    quarantined = frame.loc[fatal_mask].copy()
    quarantined["quarantine_reason"] = reasons.loc[fatal_mask]

    validated = frame.loc[~fatal_mask].drop(columns=["straight_line_km"]).reset_index(drop=True)
    validated["vendor_id"] = validated["vendor_id"].astype("int64")

    total = int(len(frame))
    bad = int(fatal_mask.sum())
    bad_fraction = bad / total if total else 1.0

    report = {
        "validated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "rows_in": total,
        "rows_validated": int(len(validated)),
        "rows_quarantined": bad,
        "quarantine_rate": round(bad_fraction, 6),
        "max_bad_row_fraction": cfg["max_bad_row_fraction"],
        "quarantine_reasons": reasons.loc[fatal_mask].value_counts().to_dict(),
        "levels": {
            "level_1_schema": None,
            "level_2_range": int(reasons.isin(structural_codes).sum()),
            "level_3_statistical": "skipped (no baseline supplied)",
            "level_4_business": int(reasons.isin(business_codes).sum()),
        },
        "nulls_left_for_imputation": {
            column: int(validated[column].isna().sum())
            for column in (*WEATHER_FIELDS, "passenger_count")
        },
        "target_summary": {
            key: round(float(value), 4)
            for key, value in validated["trip_duration_min"].describe().to_dict().items()
        },
    }

    if total < cfg["min_rows"]:
        raise ValidationFailure(f"Only {total} rows ingested; minimum is {cfg['min_rows']}")
    if bad_fraction > cfg["max_bad_row_fraction"]:
        raise ValidationFailure(
            f"Quarantine rate {bad_fraction:.2%} exceeds the "
            f"{cfg['max_bad_row_fraction']:.2%} threshold - upstream feed likely changed"
        )

    schema = build_validated_schema(params)
    try:
        schema.validate(validated, lazy=True)
        report["levels"]["level_1_schema"] = "passed"
    except pa.errors.SchemaErrors as exc:
        report["levels"]["level_1_schema"] = "failed"
        raise ValidationFailure(
            f"Validated frame violated its own schema contract:\n{exc.failure_cases.head(20)}"
        ) from exc

    report["quality_dimensions"] = quality.assess(frame, reasons)

    if baseline is not None:
        comparison = statistical_validation.compare_to_baseline(
            validated, baseline, cfg["statistical"]
        )
        report["levels"]["level_3_statistical"] = comparison["status"]
        report["statistical_validation"] = comparison
        if comparison["status"] == "fail" and cfg["statistical"]["action_on_failure"] == "fail":
            raise ValidationFailure(
                "Level 3 statistical validation failed for "
                f"{comparison['failed_columns']}; the batch distribution has moved away "
                "from the training baseline."
            )

    return validated, quarantined, report


def _markdown_summary(report: dict, planted: dict) -> str:
    levels = report["levels"]
    lines = [
        "# Data Validation Report",
        "",
        f"Generated: {report['validated_at_utc']}",
        "",
        "## Four-level validation outcome",
        "",
        "| Level | Check | Result |",
        "| --- | --- | --- |",
        f"| 1 | Schema contract | {levels['level_1_schema']} |",
        f"| 2 | Range and domain | {levels['level_2_range']:,} rows rejected |",
        f"| 3 | Statistical vs baseline | {levels['level_3_statistical']} |",
        f"| 4 | Business rules | {levels['level_4_business']:,} rows rejected |",
        "",
        "## Volume",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Rows ingested | {report['rows_in']:,} |",
        f"| Rows validated | {report['rows_validated']:,} |",
        f"| Rows quarantined | {report['rows_quarantined']:,} |",
        f"| Quarantine rate | {report['quarantine_rate']:.2%} |",
        f"| Threshold | {report['max_bad_row_fraction']:.2%} |",
        "",
        "## Quarantine reasons",
        "",
        "| Reason code | Rows |",
        "| --- | --- |",
    ]
    for reason, count in sorted(report["quarantine_reasons"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{reason}` | {count:,} |")

    lines += [
        "",
        "## Nulls carried forward for imputation",
        "",
        "Repairable gaps are left intact here and filled by the feature pipeline using",
        "values learned from the training partition only, then persisted for serving.",
        "",
        "| Field | Null rows |",
        "| --- | --- |",
    ]
    for field, count in report["nulls_left_for_imputation"].items():
        lines.append(f"| `{field}` | {count:,} |")

    if planted:
        lines += [
            "",
            "## Detection check",
            "",
            "Defects planted by the generator versus rows caught by validation.",
            "",
            "| Planted defect | Count |",
            "| --- | --- |",
        ]
        for name, count in sorted(planted.items(), key=lambda kv: -kv[1]):
            lines.append(f"| `{name}` | {count:,} |")
        lines += [
            "",
            f"Total planted: **{sum(planted.values()):,}** | "
            f"Total quarantined: **{report['rows_quarantined']:,}**",
            "",
            "> Counts differ by design: a row can carry several defects and is reported",
            "> under one reason code, and repairable defects (missing weather, missing",
            "> passenger count) are carried forward for imputation rather than quarantined.",
        ]

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a raw trip batch.")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--quarantine", default=None)
    parser.add_argument("--report-dir", default=None)
    parser.add_argument(
        "--baseline",
        default=None,
        help="Baseline profile JSON enabling Level 3 statistical validation.",
    )
    args = parser.parse_args()

    params = load_params(args.params)
    paths = params["paths"]

    raw_path = project_path(args.input or f"{paths['raw']}/{params['generate']['output_file']}")
    out_path = project_path(args.output or f"{paths['interim']}/trips_validated.parquet")
    quarantine_path = project_path(
        args.quarantine or f"{paths['quarantine']}/quarantined_trips.parquet"
    )
    report_dir = project_path(args.report_dir or f"{paths['reports']}/validation")

    baseline = None
    if args.baseline:
        baseline_path = project_path(args.baseline)
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            logger.info("Level 3 enabled using baseline %s", baseline_path)
        else:
            logger.warning("Baseline %s not found; skipping Level 3", baseline_path)

    logger.info("Reading raw batch from %s", raw_path)
    raw = pd.read_parquet(raw_path)

    validated, quarantined, report = validate_frame(raw, params, baseline)

    meta_path = raw_path.with_suffix(".meta.json")
    planted = {}
    if meta_path.exists():
        planted = json.loads(meta_path.read_text(encoding="utf-8")).get("planted_defects", {})
        report["planted_defects"] = planted

    atomic_write_parquet(validated, out_path)
    atomic_write_parquet(quarantined, quarantine_path)
    atomic_write_json(report, report_dir / "validation_report.json")
    atomic_write_text(_markdown_summary(report, planted), report_dir / "validation_report.md")
    atomic_write_text(
        quality.to_markdown(report["quality_dimensions"]),
        report_dir / "data_quality_dimensions.md",
    )
    if "statistical_validation" in report:
        atomic_write_text(
            statistical_validation.to_markdown(report["statistical_validation"]),
            report_dir / "statistical_validation.md",
        )

    logger.info(
        "Validated %s rows | quarantined %s (%.2f%%) | L2=%s L3=%s L4=%s",
        f"{report['rows_validated']:,}",
        f"{report['rows_quarantined']:,}",
        report["quarantine_rate"] * 100,
        report["levels"]["level_2_range"],
        report["levels"]["level_3_statistical"],
        report["levels"]["level_4_business"],
    )
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
