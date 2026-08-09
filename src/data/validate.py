"""Ingestion + validation stage.

Two classes of defect are handled deliberately differently, because treating them
the same is how pipelines quietly lose data:

* **Fatal** (missing GPS, dropoff before pickup, impossible speed, duplicate id) —
  the row cannot be trusted as a training target. It is *quarantined* with a reason
  code rather than dropped, so the volume and mix of failures stays auditable.
* **Repairable** (missing weather, missing passenger count) — the target is still
  valid, so the row is imputed and the imputation is recorded in a boolean flag.
  Missingness is often informative, so the flag is carried forward as a feature.

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

from src.config import ensure_parent, load_params, project_path
from src.data.schema import REQUIRED_RAW_COLUMNS, build_validated_schema
from src.utils.geo import haversine_km
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

WEATHER_NUMERIC = ["temperature_c", "precipitation_mm", "wind_kph"]


class ValidationFailure(RuntimeError):
    """Raised when the batch is too damaged to be admitted into the pipeline."""


def _read_raw(path) -> pd.DataFrame:
    return pd.read_parquet(path)


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


def _fatal_reason_codes(frame: pd.DataFrame, cfg: dict) -> pd.Series:
    """Return the first fatal reason per row, or an empty string when the row is clean."""
    bounds = cfg["bounds"]
    lat_lo, lat_hi = bounds["latitude"]
    lon_lo, lon_hi = bounds["longitude"]
    dur_lo, dur_hi = bounds["trip_duration_min"]
    spd_lo, spd_hi = bounds["avg_speed_kmph"]
    pax_lo, pax_hi = bounds["passenger_count"]

    coord_cols = [
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
    ]

    # Ordered so the most diagnostic reason wins when a row breaks several rules.
    rules: list[tuple[str, pd.Series]] = [
        ("DUPLICATE_TRIP_ID", frame["trip_id"].duplicated(keep="first")),
        ("MISSING_GPS", frame[coord_cols].isna().any(axis=1)),
        (
            "MISSING_TIMESTAMP",
            frame["pickup_datetime"].isna() | frame["dropoff_datetime"].isna(),
        ),
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
            frame["passenger_count"].notna()
            & ~frame["passenger_count"].between(pax_lo, pax_hi),
        ),
        ("INVALID_VENDOR", ~frame["vendor_id"].isin(cfg["allowed_vendors"])),
        (
            "UNKNOWN_WEATHER_CATEGORY",
            frame["weather_condition"].notna()
            & ~frame["weather_condition"].isin(cfg["allowed_weather"]),
        ),
    ]

    reasons = pd.Series("", index=frame.index, dtype="object")
    for code, mask in rules:
        reasons = reasons.mask(reasons.eq("") & mask.fillna(True), code)
    return reasons


def _repair(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Impute repairable gaps using month-level statistics from the valid rows."""
    frame = frame.copy()
    month = frame["pickup_datetime"].dt.month

    frame["weather_imputed"] = frame["weather_condition"].isna()
    frame["passenger_count_imputed"] = frame["passenger_count"].isna()

    if frame["weather_imputed"].any():
        modes = (
            frame.loc[~frame["weather_imputed"]]
            .groupby(month[~frame["weather_imputed"]])["weather_condition"]
            .agg(lambda s: s.mode().iat[0])
        )
        global_mode = frame["weather_condition"].mode().iat[0]
        frame["weather_condition"] = frame["weather_condition"].fillna(
            month.map(modes).fillna(global_mode)
        )
        for column in WEATHER_NUMERIC:
            medians = frame.groupby(month)[column].transform("median")
            frame[column] = frame[column].fillna(medians).fillna(frame[column].median())

    if frame["passenger_count_imputed"].any():
        frame["passenger_count"] = frame["passenger_count"].fillna(
            frame["passenger_count"].median()
        )

    frame["passenger_count"] = frame["passenger_count"].round().astype("int64")
    frame["vendor_id"] = frame["vendor_id"].astype("int64")
    return frame


def validate_frame(frame: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Split a raw batch into (validated, quarantined, report)."""
    cfg = params["validate"]
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in frame.columns]
    if missing:
        raise ValidationFailure(f"Raw feed is missing required columns: {missing}")
    frame = _derive_trip_metrics(frame)

    reasons = _fatal_reason_codes(frame, cfg)
    fatal_mask = reasons.ne("")

    quarantined = frame.loc[fatal_mask].copy()
    quarantined["quarantine_reason"] = reasons.loc[fatal_mask]
    quarantined["quarantined_at_utc"] = datetime.now(UTC).isoformat(timespec="seconds")

    validated = _repair(frame.loc[~fatal_mask], cfg)
    validated = validated.drop(columns=["straight_line_km"]).reset_index(drop=True)

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
        "repairs": {
            "weather_imputed": int(validated["weather_imputed"].sum()),
            "passenger_count_imputed": int(validated["passenger_count_imputed"].sum()),
        },
        "target_summary": {
            key: round(float(value), 4)
            for key, value in validated["trip_duration_min"].describe().to_dict().items()
        },
        "schema_passed": None,
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
        report["schema_passed"] = True
    except pa.errors.SchemaErrors as exc:
        report["schema_passed"] = False
        report["schema_failures"] = (
            exc.failure_cases.groupby(["column", "check"]).size().reset_index(name="count").to_dict("records")
        )
        raise ValidationFailure(
            f"Validated frame violated its own schema contract:\n{exc.failure_cases.head(20)}"
        ) from exc

    return validated, quarantined, report


def _write_markdown_summary(report: dict, planted: dict, path) -> None:
    lines = [
        "# Data Validation Report",
        "",
        f"Generated: {report['validated_at_utc']}",
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
        f"| Schema contract | {'PASSED' if report['schema_passed'] else 'FAILED'} |",
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
        "## Repairs applied",
        "",
        "| Repair | Rows |",
        "| --- | --- |",
        f"| Weather imputed (month mode/median) | {report['repairs']['weather_imputed']:,} |",
        f"| Passenger count imputed (median) | {report['repairs']['passenger_count_imputed']:,} |",
    ]

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
            "> Counts differ by design: a single row can carry several defects and is",
            "> reported under one reason code, while repairable defects (missing weather,",
            "> missing passenger count) are imputed instead of quarantined.",
        ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a raw trip batch.")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--quarantine", default=None)
    parser.add_argument("--report-dir", default=None)
    args = parser.parse_args()

    params = load_params(args.params)
    paths = params["paths"]

    raw_path = project_path(
        args.input or f"{paths['raw']}/{params['generate']['output_file']}"
    )
    out_path = project_path(args.output or f"{paths['interim']}/trips_validated.parquet")
    quarantine_path = project_path(
        args.quarantine or f"{paths['quarantine']}/quarantined_trips.parquet"
    )
    report_dir = project_path(args.report_dir or f"{paths['reports']}/validation")

    logger.info("Reading raw batch from %s", raw_path)
    raw = _read_raw(raw_path)

    validated, quarantined, report = validate_frame(raw, params)

    meta_path = raw_path.with_suffix(".meta.json")
    planted = {}
    if meta_path.exists():
        planted = json.loads(meta_path.read_text(encoding="utf-8")).get("planted_defects", {})
        report["planted_defects"] = planted

    ensure_parent(out_path)
    ensure_parent(quarantine_path)
    report_dir.mkdir(parents=True, exist_ok=True)

    validated.to_parquet(out_path, index=False)
    quarantined.to_parquet(quarantine_path, index=False)
    (report_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _write_markdown_summary(report, planted, report_dir / "validation_report.md")

    logger.info(
        "Validated %s rows | quarantined %s (%.2f%%) | reasons=%s",
        f"{report['rows_validated']:,}",
        f"{report['rows_quarantined']:,}",
        report["quarantine_rate"] * 100,
        report["quarantine_reasons"],
    )
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
