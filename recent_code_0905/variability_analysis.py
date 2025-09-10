import math
import numpy as np
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

# ============================================================
# Appendix C 핵심 통계 로직 구현 모듈 (상세 한국어 주석)
# ------------------------------------------------------------
# 본 모듈은 Online Appendix C의 절차를 엄격히 구현합니다.
# - C.1 반복 변동성(Subsampling without replacement, U-statistic 기반)
#   · 전체 평균 \bar{M}^{(R)}
#   · b개 서브샘플 평균과 T_b = √b (\bar{M}^{(b)} − \bar{M}^{(R)}) 분포
#   · CI: [\bar{J}^{(R)} − q_{1−α/2}/√R, \bar{J}^{(R)} − q_{α/2}/√R]
#   · 가설: H0: E[\bar{D}] = 0 vs HA: >0, T_R^{(0)}=√R·\bar{D}^{(R)}, p= mean(T_b ≥ T_R^{(0)})
# - C.2/C.3 리프레이즈/프롬프트 변동성(무작위 순열검정)
#   · 각 그룹 대표 포트폴리오(Top-K 빈도) 구축 → 평균 비유사도 관측치
#   · 전표본 풀링, 무작위 재할당 후 대표 포트폴리오 생성 → 귀무분포
#   · p-value = (1 + #{D_perm ≥ D_obs})/(1 + B)
# 모든 수식과 절차는 코드 주석과 함께 구현되어 있습니다.
# ============================================================

from utils.statistics import (
    calculate_jaccard,
    calculate_mean_pairwise_metric,
    construct_representative_portfolio,
)


# -----------------------------
# Subsampling for Repetition Variability (Appendix C.1)
# -----------------------------

def subsampling_bootstrap(
    samples: Sequence[Iterable],
    R: int,
    b: int,
    B: int,
    metric_func: Callable[[Iterable, Iterable], float],
    random_state: int = 42,
) -> Tuple[float, np.ndarray]:
    """
    Implements subsampling without replacement (U-statistic based) per Appendix C.1.
    - samples: list of R sets/lists (recommendation runs for same prompt)
    - metric_func: pairwise metric (e.g., jaccard similarity or dissimilarity)
    Returns:
      mean_metric_full (\bar{M}^{(R)}), T_b distribution (array length B)
    """
    rng = np.random.default_rng(random_state)
    assert len(samples) >= R, "Insufficient samples supplied"

    # Step 1: full-sample mean metric over all pairs
    mean_metric_full = calculate_mean_pairwise_metric(samples[:R], metric_func)

    T_b = np.empty(B, dtype=float)
    idx_all = np.arange(R)

    for j in range(B):
        # Steps 2&3: sample b without replacement and compute mean, then Tb
        idx_b = rng.choice(idx_all, size=b, replace=False)
        subsample = [samples[i] for i in idx_b]
        mean_metric_b = calculate_mean_pairwise_metric(subsample, metric_func)
        T_b[j] = math.sqrt(b) * (mean_metric_b - mean_metric_full)

    return mean_metric_full, T_b


def compute_ci_from_T_distribution(
    mean_stat_R: float,
    T_b: np.ndarray,
    R: int,
    alpha: float = 0.05,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
) -> Tuple[float, float]:
    """
    Given full-sample mean and T_b distribution from subsampling, compute
    two-sided (1-alpha) CI per Appendix C.1.1
      CI = [mean - q_{1-a/2}/sqrt(R), mean - q_{a/2}/sqrt(R)] clipped to [0,1]
    """
    q_low = np.quantile(T_b, alpha / 2)
    q_high = np.quantile(T_b, 1 - alpha / 2)
    ci_low = mean_stat_R - q_high / math.sqrt(R)
    ci_high = mean_stat_R - q_low / math.sqrt(R)
    ci_low = max(lower_bound, min(upper_bound, ci_low))
    ci_high = max(lower_bound, min(upper_bound, ci_high))
    return ci_low, ci_high


def compute_pvalue_from_T_distribution(
    mean_dissim_R: float, T_b: np.ndarray, R: int
) -> float:
    """
    Hypothesis test C.1.2 for Dissimilarity D:
      H0: E[\bar{D}] = 0 vs HA: E[\bar{D}] > 0
      T_R^(0) = sqrt(R) * \bar{D}^{(R)}
      p = (1/B) sum 1{T_b^(j) >= T_R^(0)}
    """
    T_R_0 = math.sqrt(R) * mean_dissim_R
    p = float(np.mean(T_b >= T_R_0))
    return p


# -----------------------------
# Permutation Test for Rephrase/Prompt Variability (Appendix C.2 & C.3)
# -----------------------------

def permutation_test(
    grouped_data: Sequence[Sequence[Iterable]],
    Q: int,
    R: int,
    B: int,
    K: int,
    random_state: int = 42,
) -> Tuple[float, float, np.ndarray]:
    """
    grouped_data: list of Q groups, each contains R samples (lists/sets of tickers)
    Procedure:
      1) Observed: construct representative portfolios S_q (top-K by frequency),
         compute mean dissimilarity over S_q pairs.
      2) Pool all Q*R samples and permute B times; reassign into Q groups of size R;
         construct reps, compute mean dissimilarity; collect distribution.
      3) p-value = (1 + # {D_b >= D_obs}) / (1 + B)
    Returns (D_obs, p_value, D_perm_distribution)
    """
    rng = np.random.default_rng(random_state)
    assert len(grouped_data) >= Q, "Not enough groups"
    for g in grouped_data:
        assert len(g) >= R, "Group has insufficient samples"

    # Step 1: observed representative portfolios per group
    reps_obs: List[List] = [
        construct_representative_portfolio(group[:R], K) for group in grouped_data[:Q]
    ]

    # Mean dissimilarity among representative portfolios
    def dissim(a, b):
        return calculate_jaccard(a, b, return_dissimilarity=True)

    D_obs = calculate_mean_pairwise_metric(reps_obs, dissim)

    # Step 2: permutation distribution
    pooled: List[Iterable] = [x for group in grouped_data[:Q] for x in group[:R]]
    pooled = list(pooled)
    n_total = len(pooled)
    assert n_total == Q * R

    D_perm = np.empty(B, dtype=float)

    for j in range(B):
        perm_idx = rng.permutation(n_total)
        # reassign into Q groups of size R
        reps_perm = []
        for q in range(Q):
            start = q * R
            end = (q + 1) * R
            group_j = [pooled[idx] for idx in perm_idx[start:end]]
            reps_perm.append(construct_representative_portfolio(group_j, K))
        D_perm[j] = calculate_mean_pairwise_metric(reps_perm, dissim)

    # Step 3: p-value
    p_value = (1.0 + float(np.sum(D_perm >= D_obs))) / (1.0 + B)
    return D_obs, p_value, D_perm
