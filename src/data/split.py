"""Temporal train / validation / test split.

A random split would let the model see trips from the *same* rush hour, weather
system and road-works period on both sides of the split, inflating offline metrics
relative to production, where the model always predicts forward in time.
Splitting on ``pickup_datetime`` reproduces that constraint.

The split runs *before* feature engineering on purpose: the zone clusters and the
zone-hour speed priors are fitted on the training partition only, so no statistic
derived from validation or test data can leak into a feature.
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.config import ensure_dir, load_params, project_path
from src.utils.io import atomic_write_json, atomic_write_parquet
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def temporal_split(
    frame: pd.DataFrame, val_fraction: float, test_fraction: float, time_column: str
) -> dict[str, pd.DataFrame]:
    ordered = frame.sort_values(time_column, kind="mergesort").reset_index(drop=True)
    n = len(ordered)
    n_test = int(round(n * test_fraction))
    n_val = int(round(n * val_fraction))
    n_train = n - n_val - n_test
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError("Split fractions leave an empty partition")
    return {
        "train": ordered.iloc[:n_train].reset_index(drop=True),
        "val": ordered.iloc[n_train : n_train + n_val].reset_index(drop=True),
        "test": ordered.iloc[n_train + n_val :].reset_index(drop=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Split validated trips temporally.")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    params = load_params(args.params)
    paths = params["paths"]
    cfg = params["split"]

    if cfg["strategy"] != "temporal":
        raise SystemExit(f"Unsupported split strategy {cfg['strategy']!r}")

    input_path = project_path(args.input or f"{paths['interim']}/trips_validated.parquet")
    output_dir = ensure_dir(args.output_dir or paths["interim"])

    frame = pd.read_parquet(input_path)
    parts = temporal_split(
        frame, cfg["val_fraction"], cfg["test_fraction"], time_column="pickup_datetime"
    )

    manifest = {"strategy": cfg["strategy"], "source": str(input_path.name), "partitions": {}}
    for name, part in parts.items():
        destination = output_dir / f"{name}.parquet"
        atomic_write_parquet(part, destination)
        manifest["partitions"][name] = {
            "rows": int(len(part)),
            "start": part["pickup_datetime"].min().isoformat(),
            "end": part["pickup_datetime"].max().isoformat(),
        }
        logger.info(
            "%-5s %8s rows | %s -> %s",
            name,
            f"{len(part):,}",
            manifest["partitions"][name]["start"],
            manifest["partitions"][name]["end"],
        )

    atomic_write_json(manifest, output_dir / "split_manifest.json")


if __name__ == "__main__":
    main()
