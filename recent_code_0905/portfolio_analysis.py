from __future__ import annotations

# ============================================================
# 포트폴리오 분석 모듈 (Professor 요구사항 반영 상세 주석)
# ------------------------------------------------------------
# 이 파일은 다음 요구사항을 충실히 반영합니다.
# 1) 샤프 지수 및 신뢰구간: 교수님 지시대로 Daily Sharpe(SR)과 Daily SE를 먼저 계산한 뒤,
#    두 값을 모두 연율화(√252 곱)하여 최종 Annualized Sharpe 및 95% CI를 산출합니다.
#    - 함수: backtest_portfolio (SR/SE 계산 흐름), calculate_sharpe_se (JK/Lo 근사식)
# 2) 가중 방식 확장: equal, inverse_vol(역변동성), market_weight(시가총액 가중) 지원.
#    - market_weight는 외부에서 전달된 시가총액 딕셔너리(티커→시총)를 사용합니다.
# 3) 시변 무위험수익률 적용: 일별 T-bill(^IRX → 일별 rf) 시계열을 받아 초과수익률로 Sharpe 계산.
# 4) 시장 국면(Regime) 분석: S&P 500의 하락 구간(고점대비 10% 이상) 식별 후 해당 구간에서
#    각 대표 포트폴리오의 구간수익률 및 최대낙폭(MDD)을 계산합니다. 모든 가중 방식에 대해 실행될 수 있도록
#    weighting 파라미터로 제어 가능.
# 5) 로컬 사전계산 데이터 지원: 사전 계산된 일별 수익률(returns)과 시가총액을 입력받아
#    야후 다운로드 없이 분석 가능하도록 설계됨.
# ============================================================

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Mapping

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


def calculate_market_cap_weights(
    stocks: Sequence[str],
    market_caps: Optional[Mapping[str, float]] = None,
) -> pd.Series:
    """
    Compute market-cap weights for given stocks using a provided market_caps mapping (ticker -> market cap).
    If mapping is missing or invalid for all, fall back to equal weights.
    """
    idx = list(stocks)
    if market_caps is None:
        return calculate_equal_weights(idx)
    # Build series and filter to positive caps
    try:
        s = pd.Series({k: float(market_caps.get(k, np.nan)) for k in idx})
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        s = s[s > 0]
    except Exception:
        return calculate_equal_weights(idx)
    if s.empty:
        return calculate_equal_weights(idx)
    w = s / s.sum()
    # Ensure index order aligns to stocks
    return w.reindex(idx).fillna(0.0)


# -----------------------------
# Backtesting
# -----------------------------

@dataclass
class BacktestConfig:
    start: str
    end: str
    weighting: Literal["equal", "inverse_vol", "market_weight"] = "equal"
    lookback: int = 63  # for inverse vol


def backtest_portfolio(
    tickers: Sequence[str],
    cfg: BacktestConfig,
    rf_daily: Optional[pd.Series] = None,
    preloaded_returns: Optional[pd.DataFrame] = None,
    market_caps: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """
    Compute daily portfolio returns with chosen weighting.
    If preloaded_returns is provided (DataFrame of daily returns indexed by date, columns tickers),
    it will be used instead of downloading with yfinance. market_caps can be provided to compute market_weight.
    Returns metrics: cumulative_return, sharpe (annualized), sr_ci_low, sr_ci_high, ann_return, ann_vol.
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

    # Prepare returns
    if preloaded_returns is not None and not preloaded_returns.empty:
        rets_full = preloaded_returns.copy()
        # filter dates
        try:
            rets_full.index = pd.to_datetime(rets_full.index)
        except Exception:
            pass
        mask = (rets_full.index >= pd.to_datetime(cfg.start)) & (rets_full.index <= pd.to_datetime(cfg.end))
        rets = rets_full.loc[mask]
        # subset to available tickers
        cols = [t for t in tickers if t in rets.columns]
        if not cols:
            # no overlap
            return {
                "cumulative_return": np.nan,
                "sharpe": np.nan,
                "sr_ci_low": np.nan,
                "sr_ci_high": np.nan,
                "ann_return": np.nan,
                "ann_vol": np.nan,
            }
        rets = rets[cols].dropna(how="all")
        prices = None
    else:
        prices = download_prices(tickers, start=cfg.start, end=cfg.end)
        rets = prices.pct_change().dropna()

    # Determine weights
    if cfg.weighting == "equal":
        weights = calculate_equal_weights(rets.columns)
    elif cfg.weighting == "inverse_vol":
        if prices is not None:
            weights = calculate_inverse_volatility_weights(rets.columns, prices, lookback_period=cfg.lookback)
        else:
            # derive from returns directly
            rets_lb = rets.iloc[-cfg.lookback:] if cfg.lookback and len(rets) > cfg.lookback else rets
            vol = rets_lb.std().replace(0, np.nan)
            inv_vol = 1.0 / vol
            inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).dropna()
            weights = inv_vol / inv_vol.sum() if not inv_vol.empty else calculate_equal_weights(rets.columns)
        weights = weights.reindex(rets.columns).dropna()
        weights = weights / weights.sum() if weights.sum() > 0 else calculate_equal_weights(rets.columns)
    elif cfg.weighting == "market_weight":
        weights = calculate_market_cap_weights(rets.columns, market_caps=market_caps)
        if weights.sum() == 0:
            weights = calculate_equal_weights(rets.columns)
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

    # --- Sharpe and CI (correct JK/Lo procedure) ---
    denom = excess.std()
    T = len(excess)

    if denom > 0 and np.isfinite(denom) and T > 1:
        # 1) Daily Sharpe (non-annualized)
        sr_daily = float(excess.mean() / denom)
        # 2) Daily SE via JK/Lo
        se_daily = calculate_sharpe_se(sr_daily, T)
        # 3) Annualize both SR and SE
        sr_ann = sr_daily * np.sqrt(252.0)
        se_ann = se_daily * np.sqrt(252.0) if np.isfinite(se_daily) else np.nan
        # 4) Two-sided 95% CI using z=1.96
        z = 1.96
        sr_ci_low = sr_ann - z * se_ann if np.isfinite(se_ann) else np.nan
        sr_ci_high = sr_ann + z * se_ann if np.isfinite(se_ann) else np.nan
    else:
        sr_ann = np.nan
        sr_ci_low, sr_ci_high = np.nan, np.nan

    return {
        "cumulative_return": cum_return,
        "sharpe": sr_ann,
        "sr_ci_low": sr_ci_low,
        "sr_ci_high": sr_ci_high,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
    }


# -----------------------------
# Sharpe SE (Jobson-Korkie/Lo approximation)
# -----------------------------

def calculate_sharpe_se(SR: float, T: int) -> float:
    """
    Computes the Standard Error (SE) using Jobson-Korkie/Lo approximation.
    SR should be the non-annualized Sharpe Ratio (e.g., daily SR if T is days).
    SE(SR) = sqrt((1/T) * (1 + 0.5 * SR^2))
    """
    if T <= 1 or not np.isfinite(SR):
        return np.nan
    SE = np.sqrt((1.0 / T) * (1.0 + 0.5 * (SR ** 2)))
    return SE


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
    weighting: Literal["equal", "inverse_vol", "market_weight"] = "equal",
    lookback: int = 63,
    preloaded_returns: Optional[pd.DataFrame] = None,
    market_caps: Optional[Mapping[str, float]] = None,
) -> pd.DataFrame:
    """
    For each representative portfolio (dict name -> tickers), compute performance within bear/correction periods
    defined by SPX drawdowns >= threshold.
    Returns DataFrame with columns: [portfolio, period_start, period_end, bear_return, bear_mdd].
    """
    spx = get_sp500_index(start, end)
    periods = identify_drawdown_periods(spx, threshold=threshold)

    rows = []
    for name, tickers in portfolios.items():
        # Prepare returns
        if preloaded_returns is not None and not preloaded_returns.empty:
            rets_full = preloaded_returns.copy()
            try:
                rets_full.index = pd.to_datetime(rets_full.index)
            except Exception:
                pass
            mask = (rets_full.index >= pd.to_datetime(start)) & (rets_full.index <= pd.to_datetime(end))
            rets = rets_full.loc[mask]
            cols = [t for t in tickers if t in rets.columns]
            if not cols:
                continue
            rets = rets[cols].dropna(how="all")
            prices = None
        else:
            prices = download_prices(tickers, start=start, end=end)
            rets = prices.pct_change().dropna()

        # Determine weights
        if weighting == "inverse_vol":
            if prices is not None:
                w = calculate_inverse_volatility_weights(rets.columns, prices, lookback_period=lookback)
            else:
                rets_lb = rets.iloc[-lookback:] if lookback and len(rets) > lookback else rets
                vol = rets_lb.std().replace(0, np.nan)
                inv_vol = 1.0 / vol
                inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).dropna()
                w = inv_vol / inv_vol.sum() if not inv_vol.empty else calculate_equal_weights(rets.columns)
            w = w.reindex(rets.columns).dropna()
            if w.sum() == 0:
                w = calculate_equal_weights(rets.columns)
        elif weighting == "market_weight":
            w = calculate_market_cap_weights(rets.columns, market_caps=market_caps)
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
