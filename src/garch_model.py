"""
garch_model.py
---------------
Provides functions to fit and forecast GARCH models for volatility estimation.
"""

import os
print(os.getcwd())
import numpy as np
import pandas as pd
from arch import arch_model
from data_loader import get_log_returns
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def fit_garch(returns, p=2, q=2, dist='normal'):
    # fit GARCH(1, 1) model to returns data
    model = arch_model(returns, p, q)
    model_fit = model.fit()
    return model_fit


def one_step_ahead_forecast(returns, test_size):
    """
    Perform a rolling one-step-ahead volatility forecast using GARCH(p, q).
    """
    returns *= 100 # convert to pct returns
    rolling_predictions = []
    index = returns.index[-test_size:]

    for i in range(test_size):
        train = returns[:-(test_size-i)]
        model = arch_model(train, p=1, q=1, dist='t', rescale=False) # student-t residuals capture fat-tailed shocks better
        model_fit = model.fit(disp='off')
        pred = model_fit.forecast(horizon=1)
        rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))

    forecast_vol = pd.Series(rolling_predictions, index=index)
    return forecast_vol

def evaluate_forecast(returns, forecast_vol):
    """
    Compute MAE, RMSE, and correlation between realized and forecasted volatilities.
    """
    forecast_vol **= 2
    realized_vol = pd.Series((returns[-len(forecast_vol):].iloc[:,0]) ** 2)
    df_eval = pd.concat([forecast_vol, realized_vol], axis=1)
    print(df_eval.head())

    mae = mean_absolute_error(realized_vol, forecast_vol)
    rmse = root_mean_squared_error(realized_vol, forecast_vol)
    realized_vol, forecast_vol = realized_vol.align(forecast_vol, join='inner')
    corr = forecast_vol.dropna().corr(realized_vol)
    return {"MAE": mae, "RMSE": rmse, "Corr": corr}

"""
-----------
"""
lr = get_log_returns()
print(lr.head())
fc = one_step_ahead_forecast(lr, 365)
print(evaluate_forecast(lr, fc))
#evaluate_forecast(get_log_returns(), one_step_ahead_forecast(get_log_returns(), 365))

"""returns = get_log_returns()
forecast = one_step_ahead_forecast(returns, 365)
metrics = evaluate_forecast(returns, forecast)
print(metrics)"""