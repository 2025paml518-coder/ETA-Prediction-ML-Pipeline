"""Hyperparameter search spaces.

Kept separate from the training code so a grid can be widened without touching
pipeline logic, and so the search space is reviewable on its own.

Ridge is tuned inside a Pipeline, hence the ``model__`` prefix: its penalty is
scale-dependent, so it must be preceded by a scaler fitted on the training folds
only. Tree models are scale-invariant and are tuned directly.

Search budget and CV strategy live in params.yaml, not here, so that changing them
invalidates the DVC training stage.
"""

# Ridge sits behind a StandardScaler, so parameters address the pipeline step.
RIDGE_PARAMS = {
    "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    "model__solver": ["auto", "svd", "cholesky", "lsqr", "sag", "saga"],
}

# Depth and estimator count are capped: an unbounded forest on ~100k rows costs
# minutes per fit and buys nothing measurable here.
RF_PARAMS = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.5],
}

LGBM_PARAMS = {
    "n_estimators": [100, 200, 300, 500],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7, 10, -1],
    "num_leaves": [31, 63, 127, 255],
    "min_child_samples": [5, 10, 20, 40],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
}

SEARCH_SPACES = {
    "ridge": RIDGE_PARAMS,
    "random_forest": RF_PARAMS,
    "lightgbm": LGBM_PARAMS,
}
