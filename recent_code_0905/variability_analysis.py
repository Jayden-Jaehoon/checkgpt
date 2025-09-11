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

# tqdm(진행바) 선택적 사용: 설치되어 있지 않으면 우아하게 폴백
try:
    from tqdm import trange
except Exception:  # pragma: no cover
    def trange(n, desc=None):
        return range(n)

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
    비복원 서브샘플링(subsampling without replacement; U‑statistic 기반)을 구현합니다 (Appendix C.1).
    - 목적: 동일 조건(R회) 추천 리스트에서 모든 쌍의 평균 지표(예: Jaccard)를 기준으로
      전체 평균 \bar{M}^{(R)}와 서브샘플 통계량 분포 {T_b}를 추정합니다.
    
    매개변수(Parameters)
    - samples: 길이 ≥ R 인 반복 추천 리스트. 각 원소는 주식 티커의 리스트/집합.
    - R: 전체 샘플 크기(사용할 반복 수). 보통 100.
    - b: 서브샘플 크기(비복원). 보통 R의 절반(예: 50).
    - B: 서브샘플링 반복 횟수(분포 표본 크기). 논문용 5,000 권장.
    - metric_func: 두 추천 집합 간 쌍대(pairwise) 지표 함수 (예: Jaccard 유사도/비유사도).
    - random_state: 난수 시드(재현성).
    
    반환(Returns)
    - mean_metric_full: 전체 R개를 사용한 평균 쌍대 지표 값(\bar{M}^{(R)}).
    - T_b: 길이 B의 ndarray. 각 원소는 T_b = sqrt(b)*(\bar{M}^{(b)} − \bar{M}^{(R)}).
    
    구현 개요
    1) 전체 평균 \bar{M}^{(R)} 계산
    2) j=1..B에 대해, R개에서 b개 비복원 추출 → 서브샘플 평균 \bar{M}^{(b)} 계산
    3) T_b = √b (\bar{M}^{(b)} − \bar{M}^{(R)}) 저장
    
    주의사항
    - samples 길이는 R 이상이어야 합니다.
    - metric_func는 대칭/유계(0~1) 지표를 가정(예: Jaccard).
    """
    rng = np.random.default_rng(random_state)
    assert len(samples) >= R, "Insufficient samples supplied"

    # Step 1: full-sample mean metric over all pairs
    mean_metric_full = calculate_mean_pairwise_metric(samples[:R], metric_func)

    T_b = np.empty(B, dtype=float)
    idx_all = np.arange(R)

    for j in trange(B, desc="Subsampling"):
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
    서브샘플링으로 얻은 T_b 분포를 사용하여 (1−alpha) 양측 신뢰구간을 계산합니다 (Appendix C.1.1).
    
    매개변수
    - mean_stat_R: 전체 표본(R개)의 평균 지표 값(예: \bar{J}^{(R)} 또는 \bar{D}^{(R)}).
    - T_b: 길이 B의 분포 표본. 각 원소는 T_b = √b (\bar{M}^{(b)} − \bar{M}^{(R)}).
    - R: 전체 표본 크기.
    - alpha: 유의수준 (기본 0.05 → 95% CI).
    - lower_bound/upper_bound: 지표의 유효 범위(예: Jaccard는 [0,1])로 결과를 클리핑.
    
    반환
    - (ci_low, ci_high): 신뢰구간 하한/상한. 공식은
      [mean − q_{1−α/2}/√R, mean − q_{α/2}/√R] 이며, 범위 [lower_bound, upper_bound]로 제한합니다.
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
    비유사도 D에 대한 단측 가설검정을 수행합니다 (Appendix C.1.2).
      H0: E[\bar{D}] = 0  대  HA: E[\bar{D}] > 0
    
    매개변수
    - mean_dissim_R: 전체 표본의 평균 비유사도(\bar{D}^{(R)}).
    - T_b: 서브샘플 통계량 분포(길이 B). 각 원소는 T_b = √b (\bar{D}^{(b)} − \bar{D}^{(R)}).
    - R: 전체 표본 크기.
    
    절차 및 반환값
    - 귀무가설 하의 기준 통계량 T_R^(0) = √R · \bar{D}^{(R)} 를 계산하고,
      경험적 분포 {T_b}에 대해 P(T_b ≥ T_R^(0))를 추정하여 p‑value를 반환합니다.
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
    리프레이즈/프롬프트 변동성에 대한 무작위 순열검정(Random Permutation Test)을 수행합니다 (Appendix C.2 & C.3).
    
    매개변수
    - grouped_data: 길이 Q 의 리스트. 각 원소는 해당 그룹(프롬프트/리프레이즈)의 R개 반복 추천 리스트.
    - Q: 그룹 수 (예: 10개의 리프레이즈 or 11개의 프롬프트).
    - R: 그룹당 사용할 반복 수(각 그룹에서 앞 R개를 사용해 크기 균등화).
    - B: 순열 반복 횟수(귀무분포 표본 크기). 논문용 5,000 권장.
    - K: 대표 포트폴리오 구성 시 선택할 Top‑K 빈도 종목 수(보통 30).
    - random_state: 난수 시드(재현성).
    
    반환
    - D_obs: 관측된 대표 포트폴리오들 간 평균 비유사도(\bar{D}_{obs}).
    - p_value: 순열 분포에서 P(D_perm ≥ D_obs)를 추정한 유의확률 (보정식 (1+count)/(1+B)).
    - D_perm: 길이 B 의 순열 분포 표본(ndarray).
    
    절차 요약
    1) 각 그룹의 R개 반복에서 대표 포트폴리오(Top‑K 빈도) S_q 를 구성하고, 이들 간 평균 비유사도 D_obs 계산
    2) 모든 샘플(Q×R)을 풀링하여 무작위로 섞은 뒤, R개씩 Q개 그룹으로 재할당 → 각 그룹에서 대표 포트폴리오 구성 → 평균 비유사도 계산
    3) 2)를 B회 반복하여 분포 {D_perm}을 만들고, p-value = (1 + #{D_perm ≥ D_obs})/(1 + B) 계산
    
    주의사항
    - 각 그룹의 샘플 수가 R 이상인지 확인합니다(크기 불일치 방지).
    - 대표 포트폴리오 구성은 빈도 기반이며, 동률인 경우 Counter.most_common의 안정적 순서에 따릅니다.
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

    for j in trange(B, desc="Permutation"):
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
