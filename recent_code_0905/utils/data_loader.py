import os
import json
from typing import Dict, List, Sequence, Tuple, Optional

import pandas as pd
import numpy as np

try:
    import yfinance as yf
except Exception:
    yf = None


# ----------------------------
# LLM recommendation data loaders
# ----------------------------

def _load_list_from_json_or_csv(file_path: str) -> Optional[List[str]]:
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".json":
            with open(file_path, "r") as f:
                data = json.load(f)
            # Accept either list of tickers or list of [ticker, weight] pairs
            if isinstance(data, list):
                if len(data) == 0:
                    return []
                if isinstance(data[0], list) or isinstance(data[0], tuple):
                    # assume first element is ticker
                    return [row[0] for row in data]
                else:
                    return list(data)
        elif ext in (".csv", ".txt"):
            try:
                df = pd.read_csv(file_path)
            except Exception:
                df = pd.read_csv(file_path, header=None)
            # Heuristics for ticker column
            for col in ("ticker", "Ticker", "symbol", "Symbol"):
                if col in df.columns:
                    return df[col].dropna().astype(str).tolist()
            # fallback to first column
            return df.iloc[:, 0].dropna().astype(str).tolist()
    except Exception:
        return None
    return None


def load_grouped_recommendations_from_dir(base_dir: str) -> List[List[List[str]]]:
    """
    Expect directory structure:
      base_dir/
        group1/  # any group name
          *.json or *.csv (each file is one repetition: list of tickers)
        group2/
          ...
    Returns: list of groups; each group is a list of repetitions; each repetition is list of tickers.
    Groups are ordered by sorted directory names to keep determinism.
    """
    if not os.path.isdir(base_dir):
        return []
    groups: List[List[List[str]]] = []
    for group_name in sorted(os.listdir(base_dir)):
        gpath = os.path.join(base_dir, group_name)
        if not os.path.isdir(gpath):
            continue
        reps: List[List[str]] = []
        for fname in sorted(os.listdir(gpath)):
            if not any(fname.lower().endswith(ext) for ext in (".json", ".csv", ".txt")):
                continue
            fpath = os.path.join(gpath, fname)
            lst = _load_list_from_json_or_csv(fpath)
            if lst is not None:
                reps.append(lst)
        if reps:
            groups.append(reps)
    return groups


def load_grouped_recommendations_with_names(base_dir: str) -> Tuple[List[str], List[List[List[str]]]]:
    """
    Same as load_grouped_recommendations_from_dir but also returns group directory names
    for labeling results.
    """
    if not os.path.isdir(base_dir):
        return [], []
    names: List[str] = []
    groups: List[List[List[str]]] = []
    for group_name in sorted(os.listdir(base_dir)):
        gpath = os.path.join(base_dir, group_name)
        if not os.path.isdir(gpath):
            continue
        reps: List[List[str]] = []
        for fname in sorted(os.listdir(gpath)):
            if not any(fname.lower().endswith(ext) for ext in (".json", ".csv", ".txt")):
                continue
            fpath = os.path.join(gpath, fname)
            lst = _load_list_from_json_or_csv(fpath)
            if lst is not None:
                reps.append(lst)
        if reps:
            names.append(group_name)
            groups.append(reps)
    return names, groups


# ----------------------------
# JSON-based grouped recommendation loaders (Prompts/Rephrase)
# ----------------------------

def load_prompts_repetition_json(file_path: str) -> Tuple[List[str], List[List[List[str]]]]:
    """
    Load prompts_repetition.json where structure is {prompt_name: [ [tickers...], [tickers...], ... ]}
    Returns (names, groups) with groups[i] being a list of repetitions (each repetition is list of tickers).
    """
    if not os.path.isfile(file_path):
        return [], []
    with open(file_path, "r") as f:
        data = json.load(f)
    names = []
    groups: List[List[List[str]]] = []
    for k in sorted(data.keys()):
        reps = data[k]
        if isinstance(reps, list) and reps:
            # ensure elements are lists of strings
            reps_clean = []
            for rep in reps:
                if isinstance(rep, list):
                    reps_clean.append([str(x) for x in rep])
            if reps_clean:
                names.append(k)
                groups.append(reps_clean)
    return names, groups


def load_rephrase_repetition_json(file_path: str) -> Dict[str, Tuple[List[str], List[List[List[str]]]]]:
    """
    Load Rephrase_Repetition_Result_*.json with structure:
      { investor_type: { rephrase_1: [ [tickers], ... R ], rephrase_2: [...], ... } }
    Returns dict: investor_type -> (rephrase_names, groups) where groups is list of rephrase groups (each group is list of repetitions).
    """
    if not os.path.isfile(file_path):
        return {}
    with open(file_path, "r") as f:
        data = json.load(f)
    out: Dict[str, Tuple[List[str], List[List[List[str]]]]] = {}
    for inv, reph in data.items():
        if not isinstance(reph, dict):
            continue
        names = []
        groups: List[List[List[str]]] = []
        for rname in sorted(reph.keys()):
            reps = reph[rname]
            if isinstance(reps, list) and reps:
                reps_clean = []
                for rep in reps:
                    if isinstance(rep, list):
                        reps_clean.append([str(x) for x in rep])
                if reps_clean:
                    names.append(rname)
                    groups.append(reps_clean)
        if groups:
            out[str(inv)] = (names, groups)
    return out


# ----------------------------
# Market data loaders
# ----------------------------

def download_prices(tickers: Sequence[str], start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted close prices for tickers using yfinance.
    Returns DataFrame indexed by date with columns per ticker.
    """
    if yf is None:
        raise ImportError("yfinance is not available. Please install it to download prices.")
    data = yf.download(list(tickers), start=start, end=end, progress=False, auto_adjust=True)
    # yfinance returns multiindex columns when multiple tickers
    if isinstance(data.columns, pd.MultiIndex):
        adj = data['Close'] if 'Close' in data.columns.levels[0] else data.xs('Adj Close', level=0, axis=1)
    else:
        adj = data
    if isinstance(adj, pd.Series):
        adj = adj.to_frame()
    adj.columns = [c if isinstance(c, str) else c[1] for c in adj.columns]
    return adj.dropna(how='all')


def get_risk_free_daily(start: str, end: str, method: str = "IRX") -> pd.Series:
    """
    Fetch time-varying daily risk-free rate series.
    method="IRX" uses ^IRX (13-week T-bill). Converts quoted annualized percent to daily rate.
    Returns daily rf series aligned to trading days (forward/back-filled to business days).
    """
    if yf is None:
        raise ImportError("yfinance is not available. Please install it to download risk-free rates.")
    if method.upper() == "IRX":
        irx = yf.download("^IRX", start=start, end=end, progress=False)["Adj Close"].dropna()
        # ^IRX is in percent. Convert to fraction.
        y = irx / 100.0
        # Convert to daily rate using 252-day compounding assumption
        rf_daily = (1.0 + y) ** (1.0 / 252.0) - 1.0
        rf_daily.name = "rf_daily"
        # Reindex to business days and forward-fill
        bdays = pd.date_range(start=rf_daily.index.min(), end=rf_daily.index.max(), freq='B')
        rf_daily = rf_daily.reindex(bdays).ffill()
        return rf_daily
    else:
        raise ValueError("Unsupported method for risk-free rate")


def get_sp500_index(start: str, end: str) -> pd.Series:
    """
    Download S&P 500 index (^GSPC) adjusted close.
    """
    if yf is None:
        raise ImportError("yfinance is not available. Please install it to download index data.")
    spx = yf.download("^GSPC", start=start, end=end, progress=False, auto_adjust=True)["Close"].dropna()
    spx.name = "SPX"
    return spx
