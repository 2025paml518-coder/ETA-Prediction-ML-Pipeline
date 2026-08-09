"""Level 3 statistical validation (M2 section 2.5.3).

Schema and range rules only catch what someone thought to write down. They cannot
see a mean drifting upward, a category share doubling, or a null rate climbing from
2% to 15% while every individual value stays legal. Level 3 compares an incoming
batch against a baseline computed from the training partition.

The baseline is built from *train only*, for the same reason imputation parameters
are: a baseline computed from the batch being judged cannot detect that the batch
has moved.

Tests used, as named in the chapter:

* Kolmogorov-Smirnov two-sample test for continuous features.
* Chi-squared test of independence for categorical features.
* Mean shift expressed in baseline standard deviations, which is the simpler signal
  the chapter recommends when a p-value is hard to act on. With ~100k baseline rows a
  KS test reaches significance on differences far too small to act on, so a p-value
  alone would flag almost every batch. Significance must therefore be accompanied by a
  minimum effect size, and the shift magnitude is what decides a failure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

CONTINUOUS_COLUMNS: tuple[str, ...] = (
    "trip_duration_min",
    "avg_speed_kmph",
    "traffic_index",
    "temperature_c",
    "precipitation_mm",
    "wind_kph",
    "passenger_count",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
)

CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "weather_condition",
    "vendor_id",
    "store_and_fwd_flag",
)

PROFILE_VERSION = "1.0.0"


def build_profile(frame: pd.DataFrame, sample_size: int, seed: int) -> dict:
    """Summarise the training partition into a comparable baseline.

    A deterministic subsample of each continuous column is retained so that a true
    two-sample KS test can be run later; storing summary statistics alone would only
    support a mean-shift check.
    """
    rng = np.random.default_rng(seed)
    profile: dict = {
        "version": PROFILE_VERSION,
        "rows": int(len(frame)),
        "sample_size": sample_size,
        "seed": seed,
        "continuous": {},
        "categorical": {},
    }

    for column in CONTINUOUS_COLUMNS:
        if column not in frame.columns:
            continue
        values = frame[column].replace([np.inf, -np.inf], np.nan)
        clean = values.dropna().to_numpy(dtype=float)
        if clean.size == 0:
            continue
        take = min(sample_size, clean.size)
        sample = rng.choice(clean, size=take, replace=False)
        profile["continuous"][column] = {
            "mean": round(float(clean.mean()), 6),
            "std": round(float(clean.std(ddof=1)) if clean.size > 1 else 0.0, 6),
            "min": round(float(clean.min()), 6),
            "max": round(float(clean.max()), 6),
            "null_rate": round(float(values.isna().mean()), 6),
            "quantiles": {
                str(q): round(float(np.quantile(clean, q)), 6)
                for q in (0.01, 0.25, 0.5, 0.75, 0.99)
            },
            "sample": [round(float(v), 6) for v in np.sort(sample)],
        }

    for column in CATEGORICAL_COLUMNS:
        if column not in frame.columns:
            continue
        series = frame[column]
        counts = series.value_counts(dropna=True)
        total = int(counts.sum())
        profile["categorical"][column] = {
            "null_rate": round(float(series.isna().mean()), 6),
            "counts": {str(k): int(v) for k, v in counts.items()},
            "frequencies": {str(k): round(float(v / total), 6) for k, v in counts.items()},
        }

    return profile


def _compare_continuous(column: str, values: pd.Series, baseline: dict, cfg: dict) -> dict:
    clean = values.replace([np.inf, -np.inf], np.nan)
    observed = clean.dropna().to_numpy(dtype=float)
    result: dict = {
        "column": column,
        "type": "continuous",
        "baseline_mean": baseline["mean"],
        "baseline_std": baseline["std"],
        "observed_rows": int(observed.size),
    }
    if observed.size == 0:
        result.update(status="fail", reason="no non-null observations")
        return result

    observed_mean = float(observed.mean())
    std = baseline["std"] or 1e-9
    shift_sd = abs(observed_mean - baseline["mean"]) / std

    ks = stats.ks_2samp(np.asarray(baseline["sample"], dtype=float), observed)
    null_rate = float(clean.isna().mean())
    null_increase = null_rate - baseline["null_rate"]

    result.update(
        observed_mean=round(observed_mean, 6),
        mean_shift_sd=round(float(shift_sd), 4),
        ks_statistic=round(float(ks.statistic), 6),
        ks_pvalue=float(ks.pvalue),
        observed_null_rate=round(null_rate, 6),
        null_rate_increase=round(null_increase, 6),
    )

    if shift_sd >= cfg["mean_shift_sd_fail"] or null_increase >= cfg["null_rate_increase_fail"]:
        result["status"] = "fail"
    elif shift_sd >= cfg["mean_shift_sd_warn"] or (
        ks.pvalue < cfg["ks_pvalue_threshold"] and ks.statistic >= cfg["ks_statistic_warn"]
    ):
        result["status"] = "warn"
    else:
        result["status"] = "pass"
    return result


def _compare_categorical(column: str, values: pd.Series, baseline: dict, cfg: dict) -> dict:
    observed_counts = values.value_counts(dropna=True)
    categories = sorted(set(baseline["counts"]) | {str(k) for k in observed_counts.index})

    expected = np.array([baseline["counts"].get(c, 0) for c in categories], dtype=float)
    observed = np.array(
        [float(observed_counts.get(c, observed_counts.get(_coerce(c), 0))) for c in categories]
    )

    result: dict = {
        "column": column,
        "type": "categorical",
        "observed_rows": int(observed.sum()),
        "unseen_categories": [
            c for c in categories if c not in baseline["counts"] and observed[categories.index(c)] > 0
        ],
    }

    if observed.sum() == 0:
        result.update(status="fail", reason="no non-null observations")
        return result

    table = np.vstack([expected, observed])
    keep = table.sum(axis=0) > 0
    try:
        chi2, pvalue, _, _ = stats.chi2_contingency(table[:, keep])
    except ValueError:
        chi2, pvalue = float("nan"), 1.0

    baseline_freq = np.array([baseline["frequencies"].get(c, 0.0) for c in categories])
    observed_freq = observed / observed.sum()
    max_shift = float(np.abs(observed_freq - baseline_freq).max())

    result.update(
        chi2_statistic=round(float(chi2), 6),
        chi2_pvalue=float(pvalue),
        max_frequency_shift=round(max_shift, 6),
        observed_null_rate=round(float(values.isna().mean()), 6),
    )

    if result["unseen_categories"] or max_shift >= cfg["category_shift_fail"]:
        result["status"] = "fail"
    elif pvalue < cfg["chi2_pvalue_threshold"] or max_shift >= cfg["category_shift_warn"]:
        result["status"] = "warn"
    else:
        result["status"] = "pass"
    return result


def _coerce(value: str):
    """Categorical keys survive a JSON round-trip as strings; recover ints."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def compare_to_baseline(frame: pd.DataFrame, profile: dict, cfg: dict) -> dict:
    """Run Level 3 checks for one batch against a training baseline."""
    checks: list[dict] = []
    for column, baseline in profile.get("continuous", {}).items():
        if column in frame.columns:
            checks.append(_compare_continuous(column, frame[column], baseline, cfg))
    for column, baseline in profile.get("categorical", {}).items():
        if column in frame.columns:
            checks.append(_compare_categorical(column, frame[column], baseline, cfg))

    failed = [c for c in checks if c["status"] == "fail"]
    warned = [c for c in checks if c["status"] == "warn"]
    return {
        "baseline_rows": profile.get("rows"),
        "batch_rows": int(len(frame)),
        "checks": checks,
        "n_pass": len(checks) - len(failed) - len(warned),
        "n_warn": len(warned),
        "n_fail": len(failed),
        "failed_columns": [c["column"] for c in failed],
        "warned_columns": [c["column"] for c in warned],
        "status": "fail" if failed else ("warn" if warned else "pass"),
    }


def to_markdown(comparison: dict) -> str:
    lines = [
        "# Level 3 Statistical Validation",
        "",
        f"Baseline rows: {comparison['baseline_rows']:,} | "
        f"Batch rows: {comparison['batch_rows']:,} | "
        f"Overall: **{comparison['status'].upper()}**",
        "",
        f"Pass: {comparison['n_pass']} · Warn: {comparison['n_warn']} · Fail: {comparison['n_fail']}",
        "",
        "| Column | Type | Status | Mean shift (sd) | Effect size | p-value | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for check in comparison["checks"]:
        shift = check.get("mean_shift_sd")
        pvalue = check.get("ks_pvalue", check.get("chi2_pvalue"))
        effect = check.get("ks_statistic", check.get("max_frequency_shift"))
        notes = []
        if check.get("unseen_categories"):
            notes.append("unseen: " + ", ".join(check["unseen_categories"]))
        if check.get("null_rate_increase"):
            notes.append(f"null rate +{check['null_rate_increase']:.4f}")
        lines.append(
            "| `{column}` | {type} | {status} | {shift} | {effect} | {pvalue} | {notes} |".format(
                column=check["column"],
                type=check["type"],
                status=check["status"].upper(),
                shift=f"{shift:.3f}" if shift is not None else "-",
                effect=f"{effect:.4f}" if effect is not None else "-",
                pvalue=f"{pvalue:.3g}" if pvalue is not None else "-",
                notes="; ".join(notes) or "-",
            )
        )
    lines.append("")
    return "\n".join(lines)
