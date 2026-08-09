"""DVC stage: build the Level 3 statistical baseline from the training partition.

The baseline exists so that *later* batches can be judged against the distribution
the model was actually trained on. Building it from anything wider than the training
partition would defeat the purpose: a baseline that already contains the batch under
test cannot reveal that the batch has moved.
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.config import load_params, project_path
from src.data.statistical_validation import build_profile
from src.utils.io import atomic_write_json, atomic_write_text
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _markdown(profile: dict) -> str:
    lines = [
        "# Level 3 Baseline Profile",
        "",
        f"Built from {profile['rows']:,} training rows | "
        f"reference sample per column: {profile['sample_size']:,}",
        "",
        "## Continuous features",
        "",
        "| Feature | Mean | Std | p1 | p50 | p99 | Null rate |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, stats in profile["continuous"].items():
        q = stats["quantiles"]
        lines.append(
            f"| `{name}` | {stats['mean']:.4f} | {stats['std']:.4f} | "
            f"{q['0.01']:.4f} | {q['0.5']:.4f} | {q['0.99']:.4f} | {stats['null_rate']:.4%} |"
        )

    lines += ["", "## Categorical features", "", "| Feature | Category | Share |", "| --- | --- | --- |"]
    for name, stats in profile["categorical"].items():
        for category, share in sorted(stats["frequencies"].items(), key=lambda kv: -kv[1]):
            lines.append(f"| `{name}` | {category} | {share:.4%} |")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the statistical baseline profile.")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--report-dir", default=None)
    args = parser.parse_args()

    params = load_params(args.params)
    paths = params["paths"]
    cfg = params["profile"]

    input_path = project_path(args.input or f"{paths['interim']}/train.parquet")
    output_path = project_path(
        args.output or f"{paths['models']}/data_profile/{cfg['output_file']}"
    )
    report_dir = project_path(args.report_dir or f"{paths['reports']}/profile")

    train = pd.read_parquet(input_path)
    profile = build_profile(train, cfg["reference_sample_size"], params["seed"])

    atomic_write_json(profile, output_path, indent=None)
    atomic_write_text(_markdown(profile), report_dir / "baseline_profile.md")

    logger.info(
        "Baseline profile from %s rows | %s continuous, %s categorical features -> %s",
        f"{len(train):,}",
        len(profile["continuous"]),
        len(profile["categorical"]),
        output_path,
    )


if __name__ == "__main__":
    main()
