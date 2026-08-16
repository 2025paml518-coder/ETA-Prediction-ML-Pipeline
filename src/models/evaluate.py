"""Model evaluation metrics.

MAE is the headline metric: it is in minutes, so it is directly meaningful to a
rider being quoted an ETA, and it does not let a handful of very long trips
dominate the score the way RMSE does. RMSE is kept alongside precisely because
the gap between the two says how heavy the error tail is.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true, y_pred) -> dict:
    """MAE, RMSE, R2, MAPE and the 90th percentile absolute error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)

    # Guard the division: a zero-duration trip would otherwise make MAPE infinite
    # and silently poison every downstream comparison.
    nonzero = np.abs(y_true) > 1e-9
    mape = (
        float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
        if nonzero.any()
        else float("nan")
    )

    return {
        "mae": round(float(mae), 4),
        "rmse": round(rmse, 4),
        "r2": round(float(r2), 4),
        "mape": round(mape, 4),
        # What the worst-served tenth of riders experience, which an average hides.
        "p90_abs_error": round(float(np.percentile(np.abs(y_true - y_pred), 90)), 4),
    }


def format_metrics(metrics: dict) -> str:
    """Format metrics for display."""
    return (
        f"MAE: {metrics['mae']:.4f} min | "
        f"RMSE: {metrics['rmse']:.4f} min | "
        f"R²: {metrics['r2']:.4f} | "
        f"MAPE: {metrics['mape']:.2f}%"
    )
