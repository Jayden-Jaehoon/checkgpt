import os
import math
import csv
from typing import Callable, Iterable, List, Sequence, Tuple, Dict

import numpy as np
import pandas as pd

from utils.data_loader import (
    load_prompts_repetition_json,
    load_rephrase_repetition_json,
    get_risk_free_daily,
)
from utils.statistics import (
    calculate_jaccard,
    calculate_mean_pairwise_metric,
    construct_representative_portfolio,
)
from variability_analysis import (
    subsampling_bootstrap,
    compute_ci_from_T_distribution,
    compute_pvalue_from_T_distribution,
    permutation_test,
)
from portfolio_analysis import (
    BacktestConfig,
    backtest_portfolio,
    regime_performance,
)

# ----------------------
# Configuration
# ----------------------

# Input JSON files discovered in results directory
REPETITION_JSON = os.path.join("/Users/jaehoon/Alphatross/70_Research/checkgpt", "results", "Prompts", "prompts_repetition.json")
REPHRASE_JSON = os.path.join("/Users/jaehoon/Alphatross/70_Research/checkgpt", "results", "Rephrase", "Rephrase_Repetition_Result_NVG.json")
PROMPTS_FOR_PORTFOLIOS_JSON = REPETITION_JSON  # use same prompts repetition to build representative portfolios

# Results path
RESULTS_DIR = os.path.join("/Users/jaehoon/Alphatross/70_Research/checkgpt", "results")

# Statistical params
R = 100
b = 50
B = 500  # quick-run; set to 5000 for full analysis
K = 30
ALPHA = 0.05

# Backtest params
BACKTEST_START = "2023-10-01"
BACKTEST_END = "2025-09-01"
WEIGHTINGS = ["equal", "inverse_vol"]  # extendable
LOOKBACK = 63

# Risk-free settings (downloaded via ^IRX)
RF_START = BACKTEST_START
RF_END = BACKTEST_END


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

def append_or_write_csv(path: str, rows: List[Dict[str, object]], header: List[str]):
    ensure_dir(os.path.dirname(path))
    mode = "a" if os.path.exists(path) else "w"
    with open(path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if mode == "w":
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ----------------------
# Part 1: Variability Analyses
# ----------------------

def run_repetition_variability(base_dir: str):
    # Wrapper to use JSON-based loader (ignores base_dir)
    run_repetition_variability_from_prompts_json(REPETITION_JSON)


def run_grouped_permutation_test(base_dir: str, label: str):
    # This wrapper dispatches to JSON-based loaders using known result files
    if label.lower().startswith("prompt"):
        run_prompt_permutation_from_prompts_json(REPETITION_JSON)
    else:
        run_rephrase_permutation_from_rephrase_json(REPHRASE_JSON)


# ----------------------
# Part 2: Portfolio Analysis
# ----------------------

def build_representative_portfolios(base_dir: str) -> Dict[str, List[str]]:
    # Wrapper to build from JSON (ignores base_dir)
    return build_representative_portfolios_from_prompts_json(PROMPTS_FOR_PORTFOLIOS_JSON)


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
                res = {k: np.nan for k in ["cumulative_return", "sharpe", "sr_ci_low", "sr_ci_high", "ann_return", "ann_vol"]}
            rows.append({
                "portfolio": name,
                "weighting": w,
                **res,
            })

    save_csv(os.path.join(RESULTS_DIR, "portfolio_performance.csv"), rows,
             header=["portfolio", "weighting", "cumulative_return", "sharpe", "sr_ci_low", "sr_ci_high", "ann_return", "ann_vol"]) 
    print(f"[Portfolio] Saved results to {os.path.join(RESULTS_DIR, 'portfolio_performance.csv')}")


def run_regime_analysis(portfolios: Dict[str, List[str]]):
    if not portfolios:
        print("[Regime] No portfolios for regime analysis. Skipping.")
        return
    try:
        rf_daily = get_risk_free_daily(RF_START, RF_END)
    except Exception:
        rf_daily = None
    try:
        df = regime_performance(portfolios, BACKTEST_START, BACKTEST_END, rf_daily=rf_daily, threshold=0.10, weighting=WEIGHTINGS[0])
        save_csv(os.path.join(RESULTS_DIR, "regime_analysis.csv"), df.to_dict(orient="records"),
                 header=["portfolio", "period_start", "period_end", "bear_return", "bear_mdd"]) 
        print(f"[Regime] Saved results to {os.path.join(RESULTS_DIR, 'regime_analysis.csv')}")
    except Exception as e:
        print(f"[Regime] Error during regime analysis: {e}")


# ----------------------
# Main
# ----------------------

def main():
    ensure_dir(RESULTS_DIR)

    # Part 1: Variability (JSON-based)
    if os.path.isfile(REPETITION_JSON):
        run_repetition_variability_from_prompts_json(REPETITION_JSON)
        run_prompt_permutation_from_prompts_json(REPETITION_JSON)
    else:
        print(f"[Repetition/Prompt] JSON not found: {REPETITION_JSON}")

    if os.path.isfile(REPHRASE_JSON):
        run_rephrase_permutation_from_rephrase_json(REPHRASE_JSON)
    else:
        print(f"[Rephrase] JSON not found: {REPHRASE_JSON}")

    # Part 2: Portfolio analysis using representative portfolios built from prompts JSON
    portfolios = {}
    if os.path.isfile(PROMPTS_FOR_PORTFOLIOS_JSON):
        portfolios = build_representative_portfolios_from_prompts_json(PROMPTS_FOR_PORTFOLIOS_JSON)

    run_portfolio_backtests(portfolios)
    run_regime_analysis(portfolios)


if __name__ == "__main__":
    main()


# ----------------------
# JSON-based runners
# ----------------------

def run_repetition_variability_from_prompts_json(file_path: str):
    names, groups = load_prompts_repetition_json(file_path)
    if not groups:
        print(f"[Repetition] No data found in {file_path}. Skipping.")
        return
    rows = []
    for name, reps in zip(names, groups):
        R_eff = min(R, len(reps))
        b_eff = min(b, len(reps))
        mean_J_R, T_b_J = subsampling_bootstrap(
            reps, R=R_eff, b=b_eff, B=B,
            metric_func=lambda a, b: calculate_jaccard(a, b, return_dissimilarity=False),
        )
        ci_low, ci_high = compute_ci_from_T_distribution(mean_J_R, T_b_J, R=R_eff, alpha=ALPHA)
        mean_D_R, T_b_D = subsampling_bootstrap(
            reps, R=R_eff, b=b_eff, B=B,
            metric_func=lambda a, b: calculate_jaccard(a, b, return_dissimilarity=True),
        )
        p_value = compute_pvalue_from_T_distribution(mean_D_R, T_b_D, R=R_eff)
        rows.append({
            "prompt": name,
            "mean_jaccard": mean_J_R,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_value_D_gt_0": p_value,
        })
    save_csv(os.path.join(RESULTS_DIR, "repetition_variability.csv"), rows,
             header=["prompt", "mean_jaccard", "ci_low", "ci_high", "p_value_D_gt_0"]) 
    print(f"[Repetition] Saved results to {os.path.join(RESULTS_DIR, 'repetition_variability.csv')}")


def run_prompt_permutation_from_prompts_json(file_path: str):
    names, groups = load_prompts_repetition_json(file_path)
    if not groups or len(groups) < 2:
        print(f"[Prompt] Not enough prompt groups in {file_path}. Skipping.")
        return
    Q = len(groups)
    R_eff = min(R, min(len(g) for g in groups))
    trimmed = [g[:R_eff] for g in groups]
    D_obs, p_value, _ = permutation_test(trimmed, Q=Q, R=R_eff, B=B, K=K)
    rows = [{
        "test": "Prompt",
        "groups": Q,
        "R_per_group": R_eff,
        "observed_D": D_obs,
        "p_value": p_value,
    }]
    append_or_write_csv(os.path.join(RESULTS_DIR, "rephrase_prompt_variability.csv"), rows,
                        header=["test", "groups", "R_per_group", "observed_D", "p_value"]) 
    print(f"[Prompt] Appended results to {os.path.join(RESULTS_DIR, 'rephrase_prompt_variability.csv')}")


def run_rephrase_permutation_from_rephrase_json(file_path: str):
    inv_map = load_rephrase_repetition_json(file_path)
    if not inv_map:
        print(f"[Rephrase] No data found in {file_path}. Skipping.")
        return
    for inv, (names, groups) in inv_map.items():
        if not groups or len(groups) < 2:
            print(f"[Rephrase] Investor {inv}: not enough groups. Skipping.")
            continue
        Q = len(groups)
        R_eff = min(R, min(len(g) for g in groups))
        trimmed = [g[:R_eff] for g in groups]
        D_obs, p_value, _ = permutation_test(trimmed, Q=Q, R=R_eff, B=B, K=K)
        rows = [{
            "test": f"Rephrase:{inv}",
            "groups": Q,
            "R_per_group": R_eff,
            "observed_D": D_obs,
            "p_value": p_value,
        }]
        append_or_write_csv(os.path.join(RESULTS_DIR, "rephrase_prompt_variability.csv"), rows,
                            header=["test", "groups", "R_per_group", "observed_D", "p_value"]) 
        print(f"[Rephrase] {inv} appended to rephrase_prompt_variability.csv")


def build_representative_portfolios_from_prompts_json(file_path: str) -> Dict[str, List[str]]:
    names, groups = load_prompts_repetition_json(file_path)
    portfolios: Dict[str, List[str]] = {}
    for name, reps in zip(names, groups):
        if not reps:
            continue
        portfolios[name] = construct_representative_portfolio(reps, K=K)
    return portfolios
