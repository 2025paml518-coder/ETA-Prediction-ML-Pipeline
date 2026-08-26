"""Hyperparameter search spaces.

Kept separate from the training code so a grid can be widened without touching
pipeline logic, and so the search space is reviewable on its own.

Every learned estimator is wrapped in a ``TransformedTargetRegressor`` (the target
is modelled on the log scale), so all keys carry a ``regressor__`` prefix that
addresses the wrapped estimator. Ridge is additionally tuned inside a Pipeline,
hence the further ``model__`` step: its penalty is scale-dependent, so it must be
preceded by a scaler fitted on the training folds only. Tree models are
scale-invariant and are tuned directly.

Search budget and CV strategy live in params.yaml, not here, so that changing them
invalidates the DVC training stage.
"""

# Ridge sits behind a StandardScaler inside a log-target regressor, so parameters
# address the wrapped pipeline step.
RIDGE_PARAMS = {
    "regressor__model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    "regressor__model__solver": ["auto", "svd", "cholesky", "lsqr", "sag", "saga"],
}

# Depth and estimator count are capped: an unbounded forest on ~100k rows costs
# minutes per fit and buys nothing measurable here.
RF_PARAMS = {
    "regressor__n_estimators": [100, 200, 300],
    "regressor__max_depth": [10, 15, 20],
    "regressor__min_samples_split": [2, 5, 10],
    "regressor__min_samples_leaf": [1, 2, 4],
    "regressor__max_features": ["sqrt", "log2", 0.5],
}

LGBM_PARAMS = {
    "regressor__n_estimators": [100, 200, 300, 500],
    "regressor__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "regressor__max_depth": [3, 5, 7, 10, -1],
    "regressor__num_leaves": [31, 63, 127, 255],
    "regressor__min_child_samples": [5, 10, 20, 40],
    "regressor__subsample": [0.7, 0.85, 1.0],
    "regressor__colsample_bytree": [0.7, 0.85, 1.0],
}

SEARCH_SPACES = {
    "ridge": RIDGE_PARAMS,
    "random_forest": RF_PARAMS,
    "lightgbm": LGBM_PARAMS,
}
