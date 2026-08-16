"""Hyperparameter grids for model tuning."""

# Ridge Regression hyperparameters
RIDGE_PARAMS = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag', 'saga'],
}

# Random Forest hyperparameters
RF_PARAMS = {
    'n_estimators': [50, 100, 200, 300, 500],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
}

# LightGBM hyperparameters
LGBM_PARAMS = {
    'n_estimators': [50, 100, 200, 300, 500],
    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10, 15],
    'num_leaves': [31, 63, 127, 255],
    'min_child_samples': [5, 10, 20],
}

# Tuning configuration
TUNING_CONFIG = {
    'n_iter': 20,           # 20 random combinations per model
    'cv': 5,                # 5-fold cross-validation
    'n_jobs': -1,           # Use all cores
    'random_state': 42,     # Reproducibility
    'verbose': 1,           # Show progress
}
