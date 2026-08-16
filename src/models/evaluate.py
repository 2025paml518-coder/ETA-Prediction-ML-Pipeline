"""Model evaluation metrics."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true, y_pred) -> dict:
    """Calculate MAE, RMSE, R² and MAPE."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'mae': round(mae, 4),
        'rmse': round(rmse, 4),
        'r2': round(r2, 4),
        'mape': round(mape, 4),
    }


def format_metrics(metrics: dict) -> str:
    """Format metrics for display."""
    return (
        f"MAE: {metrics['mae']:.4f} min | "
        f"RMSE: {metrics['rmse']:.4f} min | "
        f"R²: {metrics['r2']:.4f} | "
        f"MAPE: {metrics['mape']:.2f}%"
    )
