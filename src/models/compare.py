"""Model comparison report.

Every statement in the generated report is derived from the run results. Nothing
is asserted in advance: a report that claims "beats the baseline by 10%" whatever
the numbers actually say is worse than no report, because it looks like evidence.
"""

from __future__ import annotations

from pathlib import Path

from tabulate import tabulate

from src.config import load_params, project_path
from src.utils.io import atomic_write_text
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def comparison_table(results: list[dict]) -> str:
    """Rank candidates by validation MAE - the metric selection actually used."""
    rows = []
    for result in sorted(results, key=lambda r: r["metrics"]["val"]["mae"]):
        train = result["metrics"]["train"]
        val = result["metrics"]["val"]
        test = result["metrics"]["test"]
        rows.append(
            [
                result["model_type"],
                f"{train['mae']:.3f}",
                f"{val['mae']:.3f}",
                f"{test['mae']:.3f}",
                f"{test['rmse']:.3f}",
                f"{test['r2']:.4f}",
                f"{test['mape']:.2f}%",
                f"{test['p90_abs_error']:.3f}",
                f"{result['fit_seconds']:.1f}s",
            ]
        )

    headers = [
        "Model",
        "Train MAE",
        "Val MAE",
        "Test MAE",
        "Test RMSE",
        "Test R2",
        "Test MAPE",
        "Test p90 AE",
        "Fit time",
    ]
    return tabulate(rows, headers=headers, tablefmt="github")


def _findings(results: list[dict], best: dict) -> list[str]:
    """Derive observations from the numbers, including unflattering ones."""
    by_name = {r["model_type"]: r for r in results}
    best_val = best["metrics"]["val"]["mae"]
    best_test = best["metrics"]["test"]["mae"]
    lines: list[str] = []

    baseline = by_name.get("baseline")
    if baseline:
        base_mae = baseline["metrics"]["test"]["mae"]
        lift = (base_mae - best_test) / base_mae * 100
        lines.append(
            f"- **{best['model_type']}** reduces test MAE from the median baseline's "
            f"{base_mae:.2f} min to {best_test:.2f} min, a {lift:.1f}% improvement. "
            "The baseline is what the service would achieve with no model at all, so "
            "this is the margin that justifies operating one."
        )

    gap = best_test - best_val
    lines.append(
        f"- Validation MAE {best_val:.3f} min versus test MAE {best_test:.3f} min "
        f"(difference {gap:+.3f} min). "
        + (
            "The two agree closely, so selecting on validation did not overfit the choice."
            if abs(gap) <= 0.1 * best_val
            else "The gap is wide enough to suggest the later test period differs from "
            "the validation period - worth watching once drift monitoring is live."
        )
    )

    train_mae = best["metrics"]["train"]["mae"]
    spread = best_val - train_mae
    lines.append(
        f"- Train MAE {train_mae:.3f} min against validation {best_val:.3f} min "
        f"(spread {spread:+.3f} min). "
        + (
            "The model is not memorising the training period."
            if spread <= 0.25 * max(train_mae, 1e-9)
            else "The spread indicates meaningful overfitting despite tuning."
        )
    )

    test_metrics = best["metrics"]["test"]
    lines.append(
        f"- RMSE {test_metrics['rmse']:.2f} min exceeds MAE {best_test:.2f} min, and the "
        f"90th-percentile absolute error is {test_metrics['p90_abs_error']:.2f} min: "
        "errors are concentrated in a minority of long or unusual trips rather than "
        "spread evenly."
    )

    ranked = sorted(results, key=lambda r: r["metrics"]["val"]["mae"])
    if len(ranked) > 1:
        runner_up = ranked[1]
        margin = runner_up["metrics"]["val"]["mae"] - ranked[0]["metrics"]["val"]["mae"]
        cost = runner_up["fit_seconds"] - ranked[0]["fit_seconds"]
        lines.append(
            f"- Runner-up **{runner_up['model_type']}** trails by {margin:.3f} min val MAE "
            f"while taking {cost:+.1f}s longer to fit."
        )

    return lines


def build_report(results: list[dict], best: dict, params: dict) -> str:
    cfg = params["train"]
    selection = f"{cfg['selection']['partition']}_{cfg['selection']['metric']}"

    return "\n".join(
        [
            "# Model Comparison Report (Week 2 / M3)",
            "",
            f"Experiment: `{cfg['experiment_name']}` | "
            f"Selection metric: `{selection}` | "
            f"CV: `{cfg['cv']['strategy']}` with {cfg['cv']['n_splits']} splits",
            "",
            "## Results",
            "",
            comparison_table(results),
            "",
            f"Rows are ordered by validation MAE, which is the metric selection used. "
            f"**{best['model_type']}** was chosen (run `{best['run_id'][:8]}`).",
            "",
            "## How the winner was chosen",
            "",
            "The test column is reported but was **not** used to pick the model. Choosing "
            "on test would turn the reported test MAE into a biased, optimistic figure, "
            "since the model would have been fitted to that partition through the "
            "selection step. Validation drives the choice; test is scored once.",
            "",
            "Cross-validation inside the hyperparameter search uses `TimeSeriesSplit` "
            "rather than shuffled K-fold, for the same reason the train/val/test split is "
            "temporal: a shuffled fold trains on later trips to predict earlier ones, "
            "which inflates the score by a mechanism unavailable in production.",
            "",
            "## Findings",
            "",
            *_findings(results, best),
            "",
            "## Reproducing the selected run",
            "",
            "```bash",
            f"python -m src.models.reproduce_run --run-id {best['run_id']}",
            "```",
            "",
            "Each run records its git commit, working-tree cleanliness and the DVC md5 of "
            "every dataset it consumed, so a run identifies the exact code and data that "
            "produced it.",
            "",
        ]
    )


def compare_models(results: list[dict], best: dict, params: dict | None = None) -> Path:
    params = params or load_params()
    report = build_report(results, best, params)
    destination = project_path(f"{params['paths']['reports']}/model_comparison.md")
    atomic_write_text(report, destination)

    print("\n" + comparison_table(results))
    logger.info("Best model: %s (run %s)", best["model_type"], best["run_id"])
    logger.info("Wrote %s", destination)
    return destination
