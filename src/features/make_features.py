"""DVC stage: fit the feature pipeline on train and materialise every partition."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src.config import ensure_dir, load_params, project_path
from src.features.build_features import TARGET, FeaturePipeline
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

PARTITIONS = ("train", "val", "test")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build model-ready feature tables.")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--artifact-dir", default=None)
    args = parser.parse_args()

    params = load_params(args.params)
    paths = params["paths"]

    input_dir = project_path(args.input_dir or paths["interim"])
    output_dir = ensure_dir(args.output_dir or paths["processed"])
    artifact_dir = ensure_dir(args.artifact_dir or f"{paths['models']}/feature_pipeline")
    report_dir = ensure_dir(f"{paths['reports']}/features")

    frames = {name: pd.read_parquet(input_dir / f"{name}.parquet") for name in PARTITIONS}

    pipeline = FeaturePipeline.from_params(params)
    pipeline.fit(frames["train"])
    logger.info(
        "Fitted feature pipeline on %s train rows | %s zones | global speed %.2f km/h",
        f"{len(frames['train']):,}",
        pipeline.n_zone_clusters,
        pipeline.global_speed,
    )

    summary = {"partitions": {}, "feature_spec": pipeline.feature_spec()}
    for name, frame in frames.items():
        features = pipeline.transform(frame)
        features[TARGET] = frame[TARGET].to_numpy(dtype=float)
        features["pickup_datetime"] = frame["pickup_datetime"].to_numpy()
        features["trip_id"] = frame["trip_id"].to_numpy()

        destination = output_dir / f"{name}_features.parquet"
        features.to_parquet(destination, index=False)
        summary["partitions"][name] = {
            "rows": int(len(features)),
            "target_mean": round(float(features[TARGET].mean()), 4),
            "target_std": round(float(features[TARGET].std()), 4),
        }
        logger.info("%-5s -> %s (%s rows)", name, destination.name, f"{len(features):,}")

    pipeline.save(artifact_dir)
    (report_dir / "feature_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved feature pipeline artefacts -> %s", artifact_dir)


if __name__ == "__main__":
    main()
