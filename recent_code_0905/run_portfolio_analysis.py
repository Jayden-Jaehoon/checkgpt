import os
import csv
from typing import Dict, List

# ============================================================
# 포트폴리오 실행 스크립트 (ChatGPT/Claude + 시장가중 + 국면분석 확대)
# ------------------------------------------------------------
# 본 스크립트는 다음 요구사항을 반영합니다.
# - ChatGPT와 Claude 데이터셋 각각에 대해 대표 포트폴리오 구성 및 백테스트/국면분석 수행
# - weighting 스킴: equal, inverse_vol, market_weight(시가총액 가중) 모두 지원
# - 로컬 사전계산 데이터 활용: rivision/Performance_Analysis(_Claude) 폴더의
#   sp500_2year_returns.csv(일별 수익률), market_caps_*.csv(시총) 사용
# - 국면(Regime) 분석: 모든 가중 방식에 대해 반복 실행하여 통합 결과 저장
# 주석은 교수님 요청사항이 코드의 어느 부분에서 구현되는지 명시합니다.
# ============================================================

import numpy as np
import pandas as pd

# tqdm(진행바) 선택적 사용 준비
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, desc=None, total=None):
        return iterable if iterable is not None else []

from utils.data_loader import (
    load_prompts_repetition_json,
    get_risk_free_daily,
)
from utils.statistics import (
    construct_representative_portfolio,
)
from portfolio_analysis import (
    BacktestConfig,
    backtest_portfolio,
    regime_performance,
)

# ----------------------
# Configuration (Portfolio)
# ----------------------

# Input JSON for building representative portfolios from prompt repetitions (ChatGPT)
PROMPTS_FOR_PORTFOLIOS_JSON = os.path.join(
    "/Users/jaehoon/Alphatross/70_Research/checkgpt", "results", "Prompts", "prompts_repetition.json"
)
# Input JSON for building representative portfolios from prompt repetitions (Claude)
PROMPTS_FOR_PORTFOLIOS_JSON_CLAUDE = os.path.join(
    "/Users/jaehoon/Alphatross/70_Research/checkgpt", "results", "Prompts", "prompts_repetition_claude.json"
)

# Results path
RESULTS_DIR = os.path.join(
    "/Users/jaehoon/Alphatross/70_Research/checkgpt", "results"
)

# Portfolio params
# 전체 평가 기간(포함): 2023-01-02 ~ 2025-06-30
BACKTEST_START = "2023-01-02"
BACKTEST_END = "2025-06-30"
WEIGHTINGS = ["equal", "inverse_vol", "market_weight"]
LOOKBACK = 63

# Bear 서브샘플(포함): 2023-08-01 ~ 2024-06-18
BEAR_START = "2023-08-01"
BEAR_END = "2024-06-18"

# Risk-free settings
RF_START = BACKTEST_START
RF_END = BACKTEST_END

# Representative portfolio size
K = 30

# Local data sources
CHATGPT_DIR = "/Users/jaehoon/Alphatross/70_Research/checkgpt/rivision/Performance_Analysis"
CLAUDE_DIR = "/Users/jaehoon/Alphatross/70_Research/checkgpt/rivision/Performance_Analysis_Claude"

# ----------------------
# Helpers
# ----------------------

def ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def save_csv(path: str, rows: List[Dict[str, object]], header: List[str]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ----------------------
# Builders
# ----------------------

def build_representative_portfolios_from_prompts_json(file_path: str) -> Dict[str, List[str]]:
    names, groups = load_prompts_repetition_json(file_path)
    portfolios: Dict[str, List[str]] = {}
    for name, reps in zip(names, groups):
        if not reps:
            continue
        portfolios[name] = construct_representative_portfolio(reps, K=K)
    return portfolios


def build_representative_portfolios_from_subset_json(file_path: str) -> Dict[str, List[str]]:
    """
    Load a JSON structured as { investor_type: [ [tickers], [tickers], ... ] } and build top-K frequency portfolios.
    """
    if not os.path.isfile(file_path):
        return {}
    try:
        import json
        with open(file_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[Builder] Failed to load subset portfolios from {file_path}: {e}")
        return {}
    portfolios: Dict[str, List[str]] = {}
    for name, reps in data.items():
        if isinstance(reps, list) and reps:
            reps_clean = []
            for rep in reps:
                if isinstance(rep, list):
                    reps_clean.append([str(x) for x in rep])
            if reps_clean:
                portfolios[name] = construct_representative_portfolio(reps_clean, K=K)
    return portfolios


# ----------------------
# Runners
# ----------------------

def load_local_dataset(data_dir: str):
    """
    Load precomputed returns (sp500_2year_returns.csv) and market caps (market_caps_*.csv) from data_dir.
    Returns (returns_df or None, market_caps mapping or None).
    """
    returns_df = None
    market_caps = None
    try:
        ret_path = os.path.join(data_dir, "sp500_2year_returns.csv")
        if os.path.isfile(ret_path):
            returns_df = pd.read_csv(ret_path, index_col=0, parse_dates=True)
    except Exception as e:
        print(f"[Loader] Failed to load returns from {ret_path}: {e}")
    try:
        # find a market_caps_*.csv
        candidates = [f for f in os.listdir(data_dir) if f.startswith("market_caps_") and f.endswith(".csv")]
        if candidates:
            caps_path = os.path.join(data_dir, sorted(candidates)[0])
            caps_df = pd.read_csv(caps_path, index_col=0)
            if "marketCap" in caps_df.columns:
                market_caps = caps_df["marketCap"].to_dict()
            else:
                # fallback to first column
                market_caps = caps_df.iloc[:, 0].to_dict()
    except Exception as e:
        print(f"[Loader] Failed to load market caps in {data_dir}: {e}")
    return returns_df, market_caps


def run_portfolio_backtests_for_dataset(portfolios: Dict[str, List[str]], dataset: str, data_dir: str):
    if not portfolios:
        print(f"[Portfolio-{dataset}] No portfolios to backtest. Skipping.")
        return

    rf_daily = None
    try:
        rf_daily = get_risk_free_daily(RF_START, RF_END)
    except Exception as e:
        print(f"[Portfolio-{dataset}] Could not download risk-free rates: {e}")

    preloaded_returns, market_caps = load_local_dataset(data_dir)

    rows = []
    for name, tickers in tqdm(list(portfolios.items()), desc=f"Backtests ({dataset})"):
        for w in WEIGHTINGS:
            cfg = BacktestConfig(start=BACKTEST_START, end=BACKTEST_END, weighting=w, lookback=LOOKBACK)
            try:
                res = backtest_portfolio(
                    tickers,
                    cfg,
                    rf_daily=rf_daily,
                    preloaded_returns=preloaded_returns,
                    market_caps=market_caps,
                )
            except Exception as e:
                print(f"[Portfolio-{dataset}] Error backtesting {name} ({w}): {e}")
                res = {k: np.nan for k in [
                    "cumulative_return", "sharpe", "sr_ci_low", "sr_ci_high", "ann_return", "ann_vol"
                ]}
            rows.append({
                "dataset": dataset,
                "portfolio": name,
                "weighting": w,
                **res,
            })

    save_csv(
        os.path.join(RESULTS_DIR, f"portfolio_performance_{dataset}.csv"),
        rows,
        header=[
            "dataset", "portfolio", "weighting", "cumulative_return", "sharpe", "sr_ci_low", "sr_ci_high", "ann_return", "ann_vol"
        ],
    )
    print(f"[Portfolio-{dataset}] Saved results to {os.path.join(RESULTS_DIR, f'portfolio_performance_{dataset}.csv')}\n")


def run_regime_analysis_for_dataset(portfolios: Dict[str, List[str]], dataset: str, data_dir: str):
    if not portfolios:
        print(f"[Regime-{dataset}] No portfolios for regime analysis. Skipping.")
        return
    try:
        rf_daily = get_risk_free_daily(RF_START, RF_END)
    except Exception:
        rf_daily = None

    preloaded_returns, market_caps = load_local_dataset(data_dir)

    all_results = []
    for w_scheme in tqdm(WEIGHTINGS, desc=f"Regime weights ({dataset})"):
        try:
            print(f"[Regime-{dataset}] Running analysis for weighting: {w_scheme} (Fixed bear period)")
            df = regime_performance(
                portfolios,
                BACKTEST_START,
                BACKTEST_END,
                rf_daily=rf_daily,
                threshold=0.10,
                weighting=w_scheme,
                lookback=LOOKBACK,
                preloaded_returns=preloaded_returns,
                market_caps=market_caps,
                fixed_periods=[(BEAR_START, BEAR_END)],
            )
            if df is not None and not df.empty:
                df["dataset"] = dataset
                df["weighting"] = w_scheme
                all_results.append(df)
        except Exception as e:
            print(f"[Regime-{dataset}] Error during regime analysis (Weighting: {w_scheme}): {e}")

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        header = ["dataset", "portfolio", "weighting", "period_start", "period_end", "bear_return", "bear_mdd"]
        save_csv(
            os.path.join(RESULTS_DIR, f"regime_analysis_bear_fixed_{dataset}.csv"),
            final_df.to_dict(orient="records"),
            header=header,
        )
        print(f"[Regime-{dataset}] Saved results to {os.path.join(RESULTS_DIR, f'regime_analysis_bear_fixed_{dataset}.csv')}\n")
    else:
        print(f"[Regime-{dataset}] No results generated.")

def run_portfolio_backtests(portfolios: Dict[str, List[str]]):
    if not portfolios:
        print("[Portfolio] No portfolios to backtest. Skipping.")
        return

    rf_daily = None
    try:
        rf_daily = get_risk_free_daily(RF_START, RF_END)
    except Exception as e:
        print(f"[Portfolio] Could not download risk-free rates: {e}")

    rows = []
    for name, tickers in portfolios.items():
        for w in WEIGHTINGS:
            cfg = BacktestConfig(start=BACKTEST_START, end=BACKTEST_END, weighting=w, lookback=LOOKBACK)
            try:
                res = backtest_portfolio(tickers, cfg, rf_daily=rf_daily)
            except Exception as e:
                print(f"[Portfolio] Error backtesting {name} ({w}): {e}")
                res = {k: np.nan for k in [
                    "cumulative_return", "sharpe", "sr_ci_low", "sr_ci_high", "ann_return", "ann_vol"
                ]}
            rows.append({
                "portfolio": name,
                "weighting": w,
                **res,
            })

    save_csv(
        os.path.join(RESULTS_DIR, "portfolio_performance.csv"),
        rows,
        header=[
            "portfolio", "weighting", "cumulative_return", "sharpe", "sr_ci_low", "sr_ci_high", "ann_return", "ann_vol"
        ],
    )
    print(f"[Portfolio] Saved results to {os.path.join(RESULTS_DIR, 'portfolio_performance.csv')}")


def run_regime_analysis(portfolios: Dict[str, List[str]]):
    if not portfolios:
        print("[Regime] No portfolios for regime analysis. Skipping.")
        return
    try:
        # Load RF once
        rf_daily = get_risk_free_daily(RF_START, RF_END)
    except Exception:
        rf_daily = None

    all_results = []
    for w_scheme in WEIGHTINGS:
        try:
            print(f"[Regime] Running analysis for weighting: {w_scheme} (Fixed bear period)")
            df = regime_performance(
                portfolios,
                BACKTEST_START,
                BACKTEST_END,
                rf_daily=rf_daily,
                threshold=0.10,
                weighting=w_scheme,
                lookback=LOOKBACK,
                fixed_periods=[(BEAR_START, BEAR_END)],
            )
            if df is not None and not df.empty:
                df["weighting"] = w_scheme
                all_results.append(df)
        except Exception as e:
            print(f"[Regime] Error during regime analysis (Weighting: {w_scheme}): {e}")

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        header = ["portfolio", "weighting", "period_start", "period_end", "bear_return", "bear_mdd"]
        save_csv(
            os.path.join(RESULTS_DIR, "regime_analysis_bear_fixed.csv"),
            final_df.to_dict(orient="records"),
            header=header,
        )
        print(f"[Regime] Saved results to {os.path.join(RESULTS_DIR, 'regime_analysis_bear_fixed.csv')}")
    else:
        print("[Regime] No results generated.")


# ----------------------
# Main (Portfolio)
# ----------------------

def main():
    ensure_dir(RESULTS_DIR)

    portfolios = {}
    if os.path.isfile(PROMPTS_FOR_PORTFOLIOS_JSON):
        portfolios = build_representative_portfolios_from_prompts_json(PROMPTS_FOR_PORTFOLIOS_JSON)
    else:
        print(f"[Portfolio] JSON not found: {PROMPTS_FOR_PORTFOLIOS_JSON}")

    # 각 데이터셋별 대표 포트폴리오 기본값을 해당 LLM의 prompts_repetition JSON에서 구성
    chatgpt_base = build_representative_portfolios_from_prompts_json(PROMPTS_FOR_PORTFOLIOS_JSON) if os.path.isfile(PROMPTS_FOR_PORTFOLIOS_JSON) else {}
    claude_base = build_representative_portfolios_from_prompts_json(PROMPTS_FOR_PORTFOLIOS_JSON_CLAUDE) if os.path.isfile(PROMPTS_FOR_PORTFOLIOS_JSON_CLAUDE) else {}

    chatgpt_portfolios = chatgpt_base
    claude_portfolios =  claude_base

    # Run for ChatGPT local dataset
    run_portfolio_backtests_for_dataset(chatgpt_portfolios, dataset="chatgpt", data_dir=CHATGPT_DIR)
    run_regime_analysis_for_dataset(chatgpt_portfolios, dataset="chatgpt", data_dir=CHATGPT_DIR)

    # Run for Claude local dataset
    run_portfolio_backtests_for_dataset(claude_portfolios, dataset="claude", data_dir=CLAUDE_DIR)
    run_regime_analysis_for_dataset(claude_portfolios, dataset="claude", data_dir=CLAUDE_DIR)


if __name__ == "__main__":
    main()
