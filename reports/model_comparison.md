# Model Comparison Report (Week 2 / M3)

Experiment: `eta-prediction` | Selection metric: `val_mae` | CV: `time_series` with 3 splits

## Results

| Model         |   Train MAE |   Val MAE |   Test MAE |   Test RMSE |   Test R2 | Test MAPE   |   Test p90 AE | Fit time   |
|---------------|-------------|-----------|------------|-------------|-----------|-------------|---------------|------------|
| lightgbm      |       3.408 |     3.445 |      3.54  |       5.336 |    0.8993 | 14.82%      |         8.16  | 97.9s      |
| random_forest |       2.602 |     3.492 |      3.575 |       5.394 |    0.8971 | 14.96%      |         8.263 | 149.2s     |
| ridge         |       3.669 |     3.644 |      3.746 |       5.796 |    0.8812 | 15.73%      |         8.449 | 11.0s      |
| baseline      |      12.03  |    11.904 |     12.137 |      17.485 |   -0.0817 | 65.08%      |        28.559 | 0.1s       |

Rows are ordered by validation MAE, which is the metric selection used. **lightgbm** was chosen (run `67e5d4e1`).

## How the winner was chosen

The test column is reported but was **not** used to pick the model. Choosing on test would turn the reported test MAE into a biased, optimistic figure, since the model would have been fitted to that partition through the selection step. Validation drives the choice; test is scored once.

Cross-validation inside the hyperparameter search uses `TimeSeriesSplit` rather than shuffled K-fold, for the same reason the train/val/test split is temporal: a shuffled fold trains on later trips to predict earlier ones, which inflates the score by a mechanism unavailable in production.

## Findings

- **lightgbm** reduces test MAE from the median baseline's 12.14 min to 3.54 min, a 70.8% improvement. The baseline is what the service would achieve with no model at all, so this is the margin that justifies operating one.
- Validation MAE 3.445 min versus test MAE 3.540 min (difference +0.096 min). The two agree closely, so selecting on validation did not overfit the choice.
- Train MAE 3.408 min against validation 3.445 min (spread +0.036 min). The model is not memorising the training period.
- RMSE 5.34 min exceeds MAE 3.54 min, and the 90th-percentile absolute error is 8.16 min: errors are concentrated in a minority of long or unusual trips rather than spread evenly.
- Runner-up **random_forest** trails by 0.048 min val MAE while taking +51.3s longer to fit.

## Reproducing the selected run

```bash
python -m src.models.reproduce_run --run-id 67e5d4e1d8cc4a99abc75a68266e84fe
```

Each run records its git commit, working-tree cleanliness and the DVC md5 of every dataset it consumed, so a run identifies the exact code and data that produced it.
