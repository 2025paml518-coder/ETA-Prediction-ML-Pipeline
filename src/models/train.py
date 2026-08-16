"""Train and compare ETA prediction models using RandomizedSearchCV."""

import json
import warnings
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV

from src.models.evaluate import calculate_metrics, format_metrics
from src.models.hyperparameters import LGBM_PARAMS, RF_PARAMS, RIDGE_PARAMS, TUNING_CONFIG

warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path('data/interim')
MODELS_DIR = Path('models/trained')
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load train/val/test data."""
    train = pd.read_parquet(DATA_DIR / 'train.parquet')
    val = pd.read_parquet(DATA_DIR / 'val.parquet')
    test = pd.read_parquet(DATA_DIR / 'test.parquet')
    
    train_features = pd.read_parquet('data/processed/train_features.parquet')
    val_features = pd.read_parquet('data/processed/val_features.parquet')
    test_features = pd.read_parquet('data/processed/test_features.parquet')
    
    return {
        'train': train,
        'val': val,
        'test': test,
        'train_features': train_features,
        'val_features': val_features,
        'test_features': test_features,
    }


def prepare_data(data):
    """Extract features and targets."""
    feature_cols = [col for col in data['train_features'].columns 
                   if col not in ['trip_id', 'pickup_datetime', 'trip_duration_min']]
    
    X_train = data['train_features'][feature_cols]
    y_train = data['train']['trip_duration_min']
    
    X_val = data['val_features'][feature_cols]
    y_val = data['val']['trip_duration_min']
    
    X_test = data['test_features'][feature_cols]
    y_test = data['test']['trip_duration_min']
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def train_ridge(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train Ridge Regression with RandomizedSearchCV."""
    print('\n' + '='*60)
    print('Training Ridge Regression...')
    print('='*60)
    
    model = Ridge()
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=RIDGE_PARAMS,
        **TUNING_CONFIG,
        scoring='neg_mean_absolute_error',
    )
    
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    
    # Predictions
    y_pred_train = best_model.predict(X_train)
    y_pred_val = best_model.predict(X_val)
    y_pred_test = best_model.predict(X_test)
    
    # Metrics
    metrics_train = calculate_metrics(y_train, y_pred_train)
    metrics_val = calculate_metrics(y_val, y_pred_val)
    metrics_test = calculate_metrics(y_test, y_pred_test)
    
    print(f"\nBest params: {search.best_params_}")
    print(f"Train: {format_metrics(metrics_train)}")
    print(f"Val:   {format_metrics(metrics_val)}")
    print(f"Test:  {format_metrics(metrics_test)}")
    
    return {
        'model': best_model,
        'model_type': 'Ridge',
        'params': search.best_params_,
        'metrics_train': metrics_train,
        'metrics_val': metrics_val,
        'metrics_test': metrics_test,
        'cv_score': search.best_score_,
    }


def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train Random Forest with RandomizedSearchCV."""
    print('\n' + '='*60)
    print('Training Random Forest...')
    print('='*60)
    
    model = RandomForestRegressor(random_state=42)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=RF_PARAMS,
        **TUNING_CONFIG,
        scoring='neg_mean_absolute_error',
    )
    
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    
    # Predictions
    y_pred_train = best_model.predict(X_train)
    y_pred_val = best_model.predict(X_val)
    y_pred_test = best_model.predict(X_test)
    
    # Metrics
    metrics_train = calculate_metrics(y_train, y_pred_train)
    metrics_val = calculate_metrics(y_val, y_pred_val)
    metrics_test = calculate_metrics(y_test, y_pred_test)
    
    print(f"\nBest params: {search.best_params_}")
    print(f"Train: {format_metrics(metrics_train)}")
    print(f"Val:   {format_metrics(metrics_val)}")
    print(f"Test:  {format_metrics(metrics_test)}")
    
    return {
        'model': best_model,
        'model_type': 'RandomForest',
        'params': search.best_params_,
        'metrics_train': metrics_train,
        'metrics_val': metrics_val,
        'metrics_test': metrics_test,
        'cv_score': search.best_score_,
    }


def train_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train LightGBM with RandomizedSearchCV."""
    print('\n' + '='*60)
    print('Training LightGBM...')
    print('='*60)
    
    model = LGBMRegressor(random_state=42, verbose=-1)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=LGBM_PARAMS,
        **TUNING_CONFIG,
        scoring='neg_mean_absolute_error',
    )
    
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    
    # Predictions
    y_pred_train = best_model.predict(X_train)
    y_pred_val = best_model.predict(X_val)
    y_pred_test = best_model.predict(X_test)
    
    # Metrics
    metrics_train = calculate_metrics(y_train, y_pred_train)
    metrics_val = calculate_metrics(y_val, y_pred_val)
    metrics_test = calculate_metrics(y_test, y_pred_test)
    
    print(f"\nBest params: {search.best_params_}")
    print(f"Train: {format_metrics(metrics_train)}")
    print(f"Val:   {format_metrics(metrics_val)}")
    print(f"Test:  {format_metrics(metrics_test)}")
    
    return {
        'model': best_model,
        'model_type': 'LightGBM',
        'params': search.best_params_,
        'metrics_train': metrics_train,
        'metrics_val': metrics_val,
        'metrics_test': metrics_test,
        'cv_score': search.best_score_,
    }


def log_to_mlflow(results):
    """Log all results to MLflow."""
    mlflow.set_experiment('eta_prediction_week2')
    
    for i, result in enumerate(results, 1):
        with mlflow.start_run(run_name=f"{result['model_type']}-run{i}"):
            # Log params
            for key, value in result['params'].items():
                mlflow.log_param(key, value)
            
            # Log metrics
            mlflow.log_metric('train_mae', result['metrics_train']['mae'])
            mlflow.log_metric('val_mae', result['metrics_val']['mae'])
            mlflow.log_metric('test_mae', result['metrics_test']['mae'])
            
            mlflow.log_metric('train_r2', result['metrics_train']['r2'])
            mlflow.log_metric('val_r2', result['metrics_val']['r2'])
            mlflow.log_metric('test_r2', result['metrics_test']['r2'])
            
            # Log model
            if result['model_type'] == 'LightGBM':
                mlflow.sklearn.log_model(
                    result['model'], 
                    f"{result['model_type']}-model",
                    skops_trusted_types=['collections.OrderedDict', 'lightgbm.basic.Booster', 'lightgbm.sklearn.LGBMRegressor']
                )
            else:
                mlflow.sklearn.log_model(result['model'], f"{result['model_type']}-model")
    
    print("\nAll results logged to MLflow!")


def save_best_model(results):
    """Save best model based on test MAE."""
    best = min(results, key=lambda x: x['metrics_test']['mae'])
    
    import joblib
    model_path = MODELS_DIR / f"{best['model_type'].lower()}-best.pkl"
    joblib.dump(best['model'], model_path)
    
    print(f"\nBest model ({best['model_type']}) saved to {model_path}")
    
    # Save metadata
    metadata = {
        'model_type': best['model_type'],
        'params': best['params'],
        'metrics_test': best['metrics_test'],
        'feature_importance': get_feature_importance(best['model'], best['model_type']),
    }
    
    metadata_path = MODELS_DIR / 'best_model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return best


def get_feature_importance(model, model_type):
    """Extract feature importance from model."""
    if model_type in ['RandomForest', 'LightGBM'] and hasattr(model, 'feature_importances_'):
        return model.feature_importances_.tolist()
    return []


def run_training(experiment_name='baseline'):
    """Run full training pipeline."""
    print("\nLoading data...")
    data = load_data()
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = prepare_data(data)
    
    print(f"Feature dimensions: {X_train.shape}")
    
    # Train all models
    results = []
    results.append(train_ridge(X_train, y_train, X_val, y_val, X_test, y_test))
    results.append(train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test))
    results.append(train_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test))
    
    # Log to MLflow
    log_to_mlflow(results)
    
    # Save best model
    best = save_best_model(results)
    
    return results, best


if __name__ == '__main__':
    results, best = run_training()
    print("\n" + "="*60)
    print("Week 2 Training Complete!")
    print(f"Best Model: {best['model_type']}")
    print(f"Test MAE: {best['metrics_test']['mae']} minutes")
    print("="*60)
