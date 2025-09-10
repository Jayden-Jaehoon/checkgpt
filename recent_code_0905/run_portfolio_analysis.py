import os
import csv
from typing import Dict, List

import numpy as np
import pandas as pd

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

# Input JSON for building representative portfolios from prompt repetitions
PROMPTS_FOR_PORTFOLIOS_JSON = os.path.join(
    "/Users/jaehoon/Alphatross/70_Research/checkgpt", "results", "Prompts", "prompts_repetition.json"
)

# Results path
RESULTS_DIR = os.path.join(
    "/Users/jaehoon/Alphatross/70_Research/checkgpt", "results"
)

# Portfolio params
BACKTEST_START = "2023-10-01"
BACKTEST_END = "2025-09-01"
WEIGHTINGS = ["equal", "inverse_vol"]
LOOKBACK = 63

# Risk-free settings
RF_START = BACKTEST_START
RF_END = BACKTEST_END

# Representative portfolio size
K = 30

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


# ----------------------
# Runners
# ----------------------

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
            print(f"[Regime] Running analysis for weighting: {w_scheme}")
            df = regime_performance(
                portfolios,
                BACKTEST_START,
                BACKTEST_END,
                rf_daily=rf_daily,
                threshold=0.10,
                weighting=w_scheme,
                lookback=LOOKBACK,
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
            os.path.join(RESULTS_DIR, "regime_analysis.csv"),
            final_df.to_dict(orient="records"),
            header=header,
        )
        print(f"[Regime] Saved results to {os.path.join(RESULTS_DIR, 'regime_analysis.csv')}")
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

    run_portfolio_backtests(portfolios)
    run_regime_analysis(portfolios)


if __name__ == "__main__":
    main()
