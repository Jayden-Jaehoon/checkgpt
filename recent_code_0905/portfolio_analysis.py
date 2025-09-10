from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.statistics import construct_representative_portfolio
from utils.data_loader import download_prices, get_risk_free_daily, get_sp500_index


# -----------------------------
# Weighting schemes
# -----------------------------

def calculate_inverse_volatility_weights(
    stocks: Sequence[str],
    historical_prices: pd.DataFrame,
    lookback_period: int = 63,
) -> pd.Series:
    """
    Compute inverse-volatility weights using lookback_period business days of historical prices.
    Returns weights as a pd.Series indexed by ticker, summing to 1.0.
    If a stock lacks enough data, it is excluded; if none remain, return equal weights.
    """
    prices = historical_prices[stocks].dropna(how="all", axis=1)
    # compute daily returns
    rets = prices.pct_change().dropna()
    if len(rets) == 0:
        # fallback equal weights
        w = pd.Series(1.0, index=list(prices.columns))
        return w / w.sum() if len(w) > 0 else pd.Series(dtype=float)
    if lookback_period is not None and lookback_period > 0 and len(rets) > lookback_period:
        rets_lb = rets.iloc[-lookback_period:]
    else:
        rets_lb = rets
    vol = rets_lb.std()
    vol = vol.replace(0, np.nan)
    inv_vol = 1.0 / vol
    inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).dropna()
    if inv_vol.empty:
        # fallback equal weights
        w = pd.Series(1.0, index=list(prices.columns))
        return w / w.sum() if len(w) > 0 else pd.Series(dtype=float)
    w = inv_vol / inv_vol.sum()
    return w


def calculate_equal_weights(stocks: Sequence[str]) -> pd.Series:
    if len(stocks) == 0:
        return pd.Series(dtype=float)
    w = pd.Series(1.0, index=list(stocks))
    return w / w.sum()


# -----------------------------
# Backtesting
# -----------------------------

@dataclass
class BacktestConfig:
    start: str
    end: str
    weighting: Literal["equal", "inverse_vol"] = "equal"
    lookback: int = 63  # for inverse vol


def backtest_portfolio(
    tickers: Sequence[str],
    cfg: BacktestConfig,
    rf_daily: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """
    Download prices for tickers, compute daily portfolio returns with chosen weighting.
    Returns a dictionary with metrics: cumulative_return, sharpe, sr_ci_low, sr_ci_high, ann_return, ann_vol.
    Uses time-varying rf_daily for Sharpe if provided; aligns and fills as needed.
    """
    if len(tickers) == 0:
        return {
            "cumulative_return": np.nan,
            "sharpe": np.nan,
            "sr_ci_low": np.nan,
            "sr_ci_high": np.nan,
            "ann_return": np.nan,
            "ann_vol": np.nan,
        }

    prices = download_prices(tickers, start=cfg.start, end=cfg.end)
    rets = prices.pct_change().dropna()

    # Determine weights
    if cfg.weighting == "equal":
        weights = calculate_equal_weights(rets.columns)
    elif cfg.weighting == "inverse_vol":
        weights = calculate_inverse_volatility_weights(rets.columns, prices, lookback_period=cfg.lookback)
        # ensure weights subset to available columns
        weights = weights.reindex(rets.columns).dropna()
        weights = weights / weights.sum() if weights.sum() > 0 else calculate_equal_weights(rets.columns)
    else:
        raise ValueError("Unsupported weighting scheme")

    # Align weights with returns
    rets = rets[weights.index]

    # Portfolio returns
    port_rets = (rets * weights.values).sum(axis=1)

    # Risk-free alignment and Sharpe
    if rf_daily is not None and not rf_daily.empty:
        rf = rf_daily.reindex(port_rets.index).fillna(method="ffill").fillna(0.0)
        excess = port_rets - rf
    else:
        excess = port_rets.copy()

    # Metrics
    cum_return = float((1.0 + port_rets).prod() - 1.0)
    ann_return = float((1.0 + port_rets).prod() ** (252.0 / max(1, len(port_rets))) - 1.0)
    ann_vol = float(port_rets.std() * np.sqrt(252.0)) if len(port_rets) > 1 else np.nan

    denom = excess.std()
    sr = float(excess.mean() / denom) if denom > 0 else np.nan

    sr_ci_low, sr_ci_high = calculate_sharpe_ci(sr, len(excess)) if np.isfinite(sr) else (np.nan, np.nan)

    return {
        "cumulative_return": cum_return,
        "sharpe": sr,
        "sr_ci_low": sr_ci_low,
        "sr_ci_high": sr_ci_high,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
    }


# -----------------------------
# Sharpe CI (Jobson-Korkie/Lo approximation)
# -----------------------------

def calculate_sharpe_ci(SR: float, T: int, confidence: float = 0.95) -> Tuple[float, float]:
    if T <= 1 or not np.isfinite(SR):
        return (np.nan, np.nan)
    # Determine z-score for two-sided CI
    z = 1.96
    try:
        from math import isclose
        if isclose(confidence, 0.95, rel_tol=1e-6):
            z = 1.96
        elif abs(confidence - 0.99) < 1e-6:
            z = 2.5758
        elif abs(confidence - 0.90) < 1e-6:
            z = 1.6449
        else:
            try:
                from scipy.stats import norm
                z = float(norm.ppf(0.5 + confidence / 2.0))
            except Exception:
                z = 1.96
    except Exception:
        z = 1.96
    SE = np.sqrt((1.0 / T) * (1.0 + 0.5 * (SR ** 2)))
    return SR - z * SE, SR + z * SE


# -----------------------------
# Regime analysis (Bear/Correction periods)
# -----------------------------

def identify_drawdown_periods(index_series: pd.Series, threshold: float = 0.10) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Identify periods where drawdown from rolling peak exceeds threshold (e.g., 0.10 for 10%).
    Returns list of (start_date, end_date) where end_date is when drawdown recovers to new high or series end.
    """
    s = index_series.dropna().copy()
    peaks = s.cummax()
    dd = (peaks - s) / peaks
    in_dd = False
    periods: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    start: Optional[pd.Timestamp] = None

    for date, d in dd.items():
        if not in_dd and d >= threshold:
            in_dd = True
            start = date
        elif in_dd and d < threshold:
            # end drawdown when it goes back above threshold (not necessarily new high)
            periods.append((start, date))
            in_dd = False
            start = None
    if in_dd and start is not None:
        periods.append((start, s.index[-1]))
    return periods


def max_drawdown(series: pd.Series) -> float:
    s = (1.0 + series).cumprod()
    peak = s.cummax()
    dd = (s / peak) - 1.0
    return float(dd.min()) if not dd.empty else np.nan


def regime_performance(
    portfolios: Dict[str, Sequence[str]],
    start: str,
    end: str,
    rf_daily: Optional[pd.Series] = None,
    threshold: float = 0.10,
    weighting: Literal["equal", "inverse_vol"] = "equal",
    lookback: int = 63,
) -> pd.DataFrame:
    """
    For each representative portfolio (dict name -> tickers), compute performance within bear/correction periods
    defined by SPX drawdowns >= threshold.
    Returns DataFrame with columns: [portfolio, period_start, period_end, return, mdd].
    """
    spx = get_sp500_index(start, end)
    periods = identify_drawdown_periods(spx, threshold=threshold)

    rows = []
    for name, tickers in portfolios.items():
        prices = download_prices(tickers, start=start, end=end)
        rets = prices.pct_change().dropna()

        if weighting == "inverse_vol":
            w = calculate_inverse_volatility_weights(rets.columns, prices, lookback_period=lookback)
            w = w.reindex(rets.columns).dropna()
            if w.sum() == 0:
                w = calculate_equal_weights(rets.columns)
        else:
            w = calculate_equal_weights(rets.columns)

        port_rets = (rets[w.index] * w.values).sum(axis=1)
        if rf_daily is not None and not rf_daily.empty:
            rf = rf_daily.reindex(port_rets.index).fillna(method="ffill").fillna(0.0)
            port_rets = port_rets - rf

        for (ps, pe) in periods:
            sub = port_rets.loc[(port_rets.index >= ps) & (port_rets.index <= pe)]
            if sub.empty:
                continue
            total_return = float((1.0 + sub).prod() - 1.0)
            mdd = max_drawdown(sub)
            rows.append({
                "portfolio": name,
                "period_start": ps.date(),
                "period_end": pe.date(),
                "bear_return": total_return,
                "bear_mdd": mdd,
            })

    return pd.DataFrame(rows)
