from pmdarima.arima import ndiffs
from pmdarima.utils import diff
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np

def check_stationarity(ts, verbose=True):
    output = adfuller(ts, autolag="AIC")
    output = {
        "test_statistic": output[0],
        "pvalue": output[1],
        "n_lags": output[2],
        "n_obs": output[3],
        "critical": output[4],
    }

    if verbose:
        print(f"test_statistic: {output['test_statistic']}")
        print(f"pvalue: {output['pvalue']}")
        print(f"n_lags: {output['n_lags']}")
        print(f"n_obs: {output['n_obs']}")
        print(f"critical: {output['critical']}")

    p_value = output["pvalue"]

    if p_value <= 0.05:
        print("series is stationary")
    else:
        print("series is not stationary")


def estimate_differencing_term(ts):
    """estimates differencing term to make a series stationary"""
    kpss_diff = ndiffs(ts, alpha=0.05, test="kpss", max_d=12)
    adf_diff = ndiffs(ts, alpha=0.05, test="adf", max_d=12)
    n_diffs = max(adf_diff, kpss_diff)
    return n_diffs


def differencing(ts, n_diffs):
    """differences a series"""
    return diff(ts, lag=1, differences=n_diffs)


def seasonal_decomposition(ts, period=None, model="additive"):
    
    if isinstance(ts, pd.Series) or isinstance(ts, pd.DataFrame):
        period = None
    elif isinstance(ts, np.ndarray) and isinstance(period, int):
        period = period
    else:
        raise TypeError("series must be pd.Series, pd.DataFrame or np.array. \
        If array, period must be int")
    
    decomposition = seasonal_decompose(
        ts, period=period, model=model, extrapolate_trend="freq"
    )
    fig = decomposition.plot()

    return decomposition


def STL_decomposition(ts, period=None):
    
    if isinstance(ts, pd.Series) or isinstance(ts, pd.DataFrame):
        period = None
    elif isinstance(ts, np.ndarray) and isinstance(period, int):
        period = period
    else:
        raise TypeError("series must be pd.Series, pd.DataFrame or np.array. \
        If array, period must be int")

    stl = STL(ts, period=period, robust=True)
    res_robust = stl.fit()
    fig = res_robust.plot()

    return res_robust
