import os
import json
from itertools import combinations
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# Defaults (can be overridden via CLI)
RESULT_FILE = "/results/Prompts/prompts_repetition.json"
OUTPUT_DIR = "/recent_code_0905"
INVESTOR_TYPES = [
    "neutral_investor",
    "value_investor",
    "growth_investor",
    "momentum_investor",
    "speculative_trader",
    "index_mimicker",
    "thematic_investor",
    "sentiment_driven_investor",
    "non_financial_background_investor",
    "low_risk_aversion_investor",
    "high_risk_aversion_investor",
]


def jaccard_similarity(set1: set, set2: set) -> float:
    """Compute Jaccard similarity between two sets."""
    inter = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return inter / union if union else 0.0


def calculate_pairwise_jaccard(portfolios: List[List[str]]) -> List[float]:
    """Compute all pairwise Jaccard similarity values for a list of portfolios.

    Args:
        portfolios: List of portfolios (each is a list of tickers)
    Returns:
        List of Jaccard similarities for all unique pairs
    """
    sets = [set(p) for p in portfolios]
    jvals: List[float] = []
    for i, j in combinations(range(len(sets)), 2):
        jvals.append(jaccard_similarity(sets[i], sets[j]))
    return jvals


def compute_convergence_table(data: Dict[str, List[List[str]]], ns: Optional[List[int]] = None) -> pd.DataFrame:
    """Compute convergence table of mean Jaccard by increasing sample size.

    Args:
        data: {investor_type: [portfolios...]}
        ns: list of sample sizes; default [50,60,...,100]
    Returns:
        DataFrame: index=n, columns=investor types, values=mean jaccard
    """
    if ns is None:
        ns = list(range(50, 101, 10))
    ns = sorted({n for n in ns if n >= 2})

    rows = []
    for n in ns:
        row = {"n": n}
        for investor_type in INVESTOR_TYPES:
            if investor_type not in data:
                continue
            portfolios = data[investor_type]
            if len(portfolios) < 2:
                continue
            n_use = min(n, len(portfolios))
            jvals = calculate_pairwise_jaccard(portfolios[:n_use])
            row[investor_type] = float(np.mean(jvals)) if len(jvals) else np.nan
        rows.append(row)

    df = pd.DataFrame(rows).set_index("n").sort_index()

    # Save outputs for paper inclusion
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "convergence_table.csv")
    df.to_csv(csv_path)
    try:
        md_path = os.path.join(OUTPUT_DIR, "convergence_table.md")
        with open(md_path, "w") as f:
            f.write(df.round(3).to_markdown())
    except Exception:
        pass
    return df


def compute_convergence_summary(conv_df: pd.DataFrame) -> pd.DataFrame:
    """Compute stabilization metrics from the convergence table.

    Metrics per investor type:
      - mean_80, mean_90, mean_100 (if available)
      - delta_80_90, delta_90_100, abs_delta_80_100
      - stabilized_from_n_(<0.01): smallest n where |mean_n - mean_100| < 0.01
    """
    rows = []
    ns = list(conv_df.index)
    for inv in [c for c in conv_df.columns if c in INVESTOR_TYPES]:
        vals = conv_df[inv]
        entry = {"investor_type": inv}
        for target in [80, 90, 100]:
            if target in conv_df.index:
                entry[f"mean_{target}"] = float(vals.loc[target])
        if 80 in conv_df.index and 90 in conv_df.index:
            entry["delta_80_90"] = float(vals.loc[90] - vals.loc[80])
        if 90 in conv_df.index and 100 in conv_df.index:
            entry["delta_90_100"] = float(vals.loc[100] - vals.loc[90])
        if 80 in conv_df.index and 100 in conv_df.index:
            entry["abs_delta_80_100"] = float(abs(vals.loc[100] - vals.loc[80]))

        stabilized_from = None
        if 100 in conv_df.index and not np.isnan(vals.loc[100]):
            m100 = float(vals.loc[100])
            for n in ns:
                v = vals.loc[n]
                if not np.isnan(v) and abs(float(v) - m100) < 0.01:
                    stabilized_from = n
                    break
        entry["stabilized_from_n_(<0.01)"] = stabilized_from
        rows.append(entry)

    out = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, "convergence_changes.csv")
    out.to_csv(out_path, index=False)
    try:
        with open(os.path.join(OUTPUT_DIR, "convergence_changes.md"), "w") as f:
            f.write(out.round(3).to_markdown(index=False))
    except Exception:
        pass
    return out


def load_data(result_file: str) -> Dict[str, List[List[str]]]:
    with open(result_file, "r") as f:
        data = json.load(f)
    return data


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convergence test for mean Jaccard similarity (n=50..100)")
    parser.add_argument("--results", default=RESULT_FILE, help="Path to prompts_repetition.json")
    parser.add_argument("--start", type=int, default=50, help="Start n (default 50)")
    parser.add_argument("--stop", type=int, default=100, help="Stop n inclusive (default 100)")
    parser.add_argument("--step", type=int, default=10, help="Step for n (default 10)")
    args = parser.parse_args()


    OUTPUT_DIR = "/recent_code_0905"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading portfolio data from {args.results}...")
    data = load_data(args.results)

    # Log counts
    print(f"Found data for {len(data)} investor types")
    for investor_type in INVESTOR_TYPES:
        if investor_type in data:
            print(f"  - {investor_type}: {len(data[investor_type])} repetitions")
        else:
            print(f"  - {investor_type}: MISSING")

    ns = list(range(args.start, args.stop + 1, args.step))
    print(f"\nComputing convergence table for n={ns} ...")
    conv_df = compute_convergence_table(data, ns=ns)
    print(f"Convergence table saved to {os.path.join(OUTPUT_DIR, 'convergence_table.csv')}")

    print("Computing convergence summary (stabilization metrics)...")
    summary_df = compute_convergence_summary(conv_df)
    print(f"Convergence summary saved to {os.path.join(OUTPUT_DIR, 'convergence_changes.csv')}")

    # Brief textual claim to paste in paper
    try:
        claim_lines = ["Convergence observation (rounded to 3 decimals):"]
        last_row = conv_df.round(3).iloc[-1]
        claim_lines.append(f"  n={conv_df.index[-1]} means: " + ", ".join([f"{c}={last_row[c]:.3f}" for c in conv_df.columns]))
        claim_text = "\n".join(claim_lines)
        with open(os.path.join(OUTPUT_DIR, "convergence_note.txt"), "w") as f:
            f.write(claim_text + "\n")
    except Exception:
        pass

    print("\nDone. Please include convergence_table.md and convergence_changes.md in the paper.")


if __name__ == "__main__":
    main()
