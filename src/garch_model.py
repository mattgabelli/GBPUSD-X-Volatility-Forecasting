"""
garch_model.py
---------------
Provides functions to fit and forecast GARCH models for volatility estimation.
"""

import os
print(os.getcwd())
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from tqdm import tqdm
from data_loader import get_log_returns
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def one_step_ahead_forecast(returns, test_size=365):
    """
    Rolling one-step-ahead volatility forecast using GARCH(1,1).
    Uses Student-t residuals and disables internal rescaling to preserve units.
    """
    returns = returns.squeeze()  # ensure Series
    rolling_predictions = []
    index = returns.index[-test_size:]

    # start rolling forecasts
    print(f"\nStarting rolling GARCH(1,1) forecast for {test_size} days...")
    for i in tqdm(range(test_size)):
        train = returns.iloc[: -(test_size - i)]

        # Skip if training window too short
        if len(train) < 50:
            rolling_predictions.append(np.nan)
            continue

        # Fit GARCH(1,1) with Student-t distribution
        try:
            model = arch_model(train, p=1, q=1, dist="t", rescale=False)
            model_fit = model.fit(disp="off")
            pred_var = model_fit.forecast(horizon=1).variance.values[-1, 0]

            # Drop invalid forecasts
            if np.isnan(pred_var) or pred_var <= 0 or pred_var > 0.05:
                rolling_predictions.append(np.nan)
                continue

            rolling_predictions.append(np.sqrt(pred_var))

        except Exception as e:
            print(f"⚠️ Step {i}: fit failed ({e})")
            rolling_predictions.append(np.nan)
            continue

    forecast_vol = pd.Series(rolling_predictions, index=index)
    return forecast_vol


def evaluate_forecast(returns, forecast_vol):
    """
    Compare realized and forecasted volatility (both σ).
    Realized volatility = |returns|, both on the same scale.
    """
    returns = returns.squeeze()
    realized_vol = np.abs(returns[-len(forecast_vol):])

    realized_vol, forecast_vol = realized_vol.align(forecast_vol, join="inner")
    mask = forecast_vol.notna() & realized_vol.notna()

    print(f"\nRealized vol range: {realized_vol.min()} – {realized_vol.max()}")
    print(f"Forecast vol range: {forecast_vol[mask].min()} – {forecast_vol[mask].max()}")

    mae = mean_absolute_error(realized_vol[mask], forecast_vol[mask])
    rmse = root_mean_squared_error(realized_vol[mask], forecast_vol[mask])
    corr = realized_vol[mask].corr(forecast_vol[mask])

    plt.scatter(realized_vol, forecast_vol)
    plt.show()
    return {"MAE": mae, "RMSE": rmse, "Corr": corr}


if __name__ == "__main__":
    returns = get_log_returns()
    forecast = one_step_ahead_forecast(returns, test_size=365)
    metrics = evaluate_forecast(returns, forecast)

    print("\nFinal evaluation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")
