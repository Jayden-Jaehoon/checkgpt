import os
import csv
from typing import List, Dict, Optional

# ============================================================
# 변동성 분석 실행 스크립트 (Appendix C 구현 주석)
# ------------------------------------------------------------
# 이 스크립트는 Online Appendix C의 통계 검증 절차를 실행합니다.
# - 반복(Repetition) 변동성: 비복원 서브샘플링(U-statistic) 기반 분포로 Jaccard의 CI, D>0 가설검정
# - 리프레이즈/프롬프트 변동성: 무작위 순열검정으로 대표 포트폴리오 간 평균 비유사도 비교
# - 반복 횟수 B=5000: 교수님 지시대로 최종 분석 시 충분한 반복으로 분위수/유의확률의 안정성을 확보
# - Claude 지원: prompts_repetition_claude.json을 추가로 읽어 별도 접미사 파일에 저장
# ============================================================

# tqdm(진행바) 선택적 사용 준비
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, desc=None, total=None):
        return iterable if iterable is not None else []

from utils.data_loader import (
    load_prompts_repetition_json,
    load_rephrase_repetition_json,
)
from utils.statistics import (
    calculate_jaccard,
)
from variability_analysis import (
    subsampling_bootstrap,
    compute_ci_from_T_distribution,
    compute_pvalue_from_T_distribution,
    permutation_test,
)

# ----------------------
# Configuration (Variability)
# ----------------------

# Input JSON files
REPETITION_JSON = os.path.join(
    "/Users/jaehoon/Alphatross/70_Research/checkgpt", "results", "Prompts", "prompts_repetition.json"
)
REPETITION_JSON_CLAUDE = os.path.join(
    "/Users/jaehoon/Alphatross/70_Research/checkgpt", "results", "Prompts", "prompts_repetition_claude.json"
)
REPHRASE_JSON = os.path.join(
    "/Users/jaehoon/Alphatross/70_Research/checkgpt", "results", "Rephrase", "Rephrase_Repetition_Result_NVG.json"
)
REPHRASE_JSON_CLAUDE = os.path.join(
    "/Users/jaehoon/Alphatross/70_Research/checkgpt", "results", "Rephrase", "Rephrase_Repetition_Result_NVG_claude.json"
)

# Results path
RESULTS_DIR = os.path.join(
    "/Users/jaehoon/Alphatross/70_Research/checkgpt", "results"
)

# Statistical params
# R: 그룹당 반복 수(샘플크기), b: 서브샘플 크기(비복원), B: 재표집/순열 반복 횟수(최종분석은 5000),
# K: 대표 포트폴리오 크기(Top-K 빈도), ALPHA: 유의수준(95% CI)
R = 100
b = 50
# B = 500  # 빠른 테스트용 (논문 최종분석은 5000 권장)
B = 5000  # Appendix C 기준 최종 반복 횟수 (분위수/유의확률 추정 안정화)
K = 30
ALPHA = 0.05


# ----------------------
# Helpers
# ----------------------

def ensure_dir(path: str):
    """
    디렉터리 경로가 없으면 생성합니다.
    - path: 파일을 저장할 디렉터리 경로
    - 존재하지 않을 경우 os.makedirs(..., exist_ok=True)로 안전하게 생성합니다.
    """
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def save_csv(path: str, rows: List[Dict[str, object]], header: List[str]):
    """
    CSV 파일을 새로 생성하여(header 포함) rows를 저장합니다.
    - path: 저장할 파일 경로
    - rows: Dict 형태의 레코드 목록
    - header: CSV 헤더(열) 순서 정의
    동작: 디렉터리를 보장(ensure_dir)하고, 파일을 'w' 모드로 열어 헤더 작성 후 모든 행을 기록합니다.
    """
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_or_write_csv(path: str, rows: List[Dict[str, object]], header: List[str]):
    """
    CSV 파일에 행을 덧붙이거나(append) 없으면 새로 생성합니다.
    - path: 저장할 파일 경로
    - rows: Dict 형태의 레코드 목록
    - header: CSV 헤더(열) 순서 정의
    동작: 파일 존재 여부로 모드('a' 또는 'w')를 결정하고, 새 파일이면 헤더를 먼저 씁니다.
    """
    ensure_dir(os.path.dirname(path))
    mode = "a" if os.path.exists(path) else "w"
    with open(path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if mode == "w":
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ----------------------
# JSON-based runners (Variability)
# ----------------------

def run_repetition_variability_from_prompts_json(file_path: str, suffix: Optional[str] = None):
    """
    프롬프트 반복 결과(prompts_repetition.json)를 입력으로 받아 반복 변동성(Repetition)을 계산합니다.
    - 입력 파일 구조: { prompt_name: [ [tickers], [tickers], ... ] }
    - 처리:
      1) 각 프롬프트별로 R_eff=min(R, 실제 반복수), b_eff=min(b, 실제 반복수) 설정
      2) 유사도(J) 기준 서브샘플링으로 CI 계산, 비유사도(D) 기준으로 단측 검정 p-value 계산
      3) 결과를 repetition_variability{suffix}.csv로 저장 (suffix는 _claude 등 식별용)
    - 매개변수
      * file_path: prompts_repetition.json 경로
      * suffix: 파일명 접미사(옵션). None이면 접미사 없이 저장
    - 산출물 컬럼: [prompt, mean_jaccard, ci_low, ci_high, p_value_D_gt_0]
    """
    names, groups = load_prompts_repetition_json(file_path)
    if not groups:
        print(f"[Repetition] No data found in {file_path}. Skipping.")
        return
    rows = []
    for name, reps in tqdm(list(zip(names, groups)), desc="Prompts (repetition)"):
        R_eff = min(R, len(reps))
        b_eff = min(b, len(reps))
        mean_J_R, T_b_J = subsampling_bootstrap(
            reps,
            R=R_eff,
            b=b_eff,
            B=B,
            metric_func=lambda a, b: calculate_jaccard(a, b, return_dissimilarity=False),
        )
        ci_low, ci_high = compute_ci_from_T_distribution(mean_J_R, T_b_J, R=R_eff, alpha=ALPHA)
        mean_D_R, T_b_D = subsampling_bootstrap(
            reps,
            R=R_eff,
            b=b_eff,
            B=B,
            metric_func=lambda a, b: calculate_jaccard(a, b, return_dissimilarity=True),
        )
        p_value = compute_pvalue_from_T_distribution(mean_D_R, T_b_D, R=R_eff)
        rows.append(
            {
                "prompt": name,
                "mean_jaccard": mean_J_R,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value_D_gt_0": p_value,
            }
        )
    out_name = f"repetition_variability{suffix if suffix else ''}.csv"
    save_csv(
        os.path.join(RESULTS_DIR, out_name),
        rows,
        header=["prompt", "mean_jaccard", "ci_low", "ci_high", "p_value_D_gt_0"],
    )
    print(
        f"[Repetition] Saved results to {os.path.join(RESULTS_DIR, out_name)}"
    )


def run_prompt_permutation_from_prompts_json(file_path: str, suffix: Optional[str] = None):
    """
    프롬프트 그룹 간 변동성(Prompt Variability)에 대한 순열검정을 수행합니다.
    - 입력 파일: prompts_repetition.json (여러 프롬프트 이름 → 각 프롬프트의 R회 반복 추천)
    - 처리:
      1) 그룹 수 Q와 균등화된 반복 수 R_eff를 정하고, 각 그룹의 앞 R_eff개만 사용
      2) 각 그룹에서 대표 포트폴리오(Top-K)를 만들어 관측 평균 비유사도 D_obs 계산
      3) 모든 샘플(Q×R_eff)을 섞어 무작위 재할당하며 B회 순열 분포 {D_perm} 생성
      4) p-value = (1 + #{D_perm ≥ D_obs}) / (1 + B)
      5) 결과를 rephrase_prompt_variability{suffix}.csv에 추가(append)
    - 매개변수
      * file_path: prompts_repetition.json 경로
      * suffix: 출력 파일 접미사(예: _claude)
    - 산출물 컬럼: [test, groups, R_per_group, observed_D, p_value]
    """
    names, groups = load_prompts_repetition_json(file_path)
    if not groups or len(groups) < 2:
        print(f"[Prompt] Not enough prompt groups in {file_path}. Skipping.")
        return
    Q = len(groups)
    R_eff = min(R, min(len(g) for g in groups))
    trimmed = [g[:R_eff] for g in groups]
    D_obs, p_value, _ = permutation_test(trimmed, Q=Q, R=R_eff, B=B, K=K)
    rows = [
        {
            "test": "Prompt",
            "groups": Q,
            "R_per_group": R_eff,
            "observed_D": D_obs,
            "p_value": p_value,
        }
    ]
    out_name = f"rephrase_prompt_variability{suffix if suffix else ''}.csv"
    append_or_write_csv(
        os.path.join(RESULTS_DIR, out_name),
        rows,
        header=["test", "groups", "R_per_group", "observed_D", "p_value"],
    )
    print(
        f"[Prompt] Appended results to {os.path.join(RESULTS_DIR, out_name)}"
    )


def run_rephrase_permutation_from_rephrase_json(file_path: str):
    """
    리프레이즈 그룹 간 변동성(Rephrase Variability)에 대한 순열검정을 수행합니다.
    - 입력 파일: Rephrase_Repetition_Result_*.json
      구조 예) { investor_type: { rephrase_1: [ [tickers], ... R ], rephrase_2: [...], ... } }
    - 처리:
      1) 투자자 유형(inv)별로 리프레이즈 그룹을 수집하고, 각 그룹에서 앞 R_eff개만 사용
      2) 각 리프레이즈 그룹의 대표 포트폴리오(Top-K)로 관측 평균 비유사도 D_obs 계산
      3) 풀링/재할당 순열을 B회 수행하여 분포 {D_perm} 형성, p-value 계산
      4) 결과를 rephrase_prompt_variability.csv에 추가
    - 매개변수
      * file_path: Rephrase_Repetition_Result_*.json 경로 (ChatGPT/Claude 각각 별도 실행 가능)
    - 산출물 컬럼: [test, groups, R_per_group, observed_D, p_value] (test 값은 "Rephrase:{inv}")
    """
    inv_map = load_rephrase_repetition_json(file_path)
    if not inv_map:
        print(f"[Rephrase] No data found in {file_path}. Skipping.")
        return
    for inv, (names, groups) in tqdm(list(inv_map.items()), desc="Investors (rephrase)"):
        if not groups or len(groups) < 2:
            print(f"[Rephrase] Investor {inv}: not enough groups. Skipping.")
            continue
        Q = len(groups)
        R_eff = min(R, min(len(g) for g in groups))
        trimmed = [g[:R_eff] for g in groups]
        D_obs, p_value, _ = permutation_test(trimmed, Q=Q, R=R_eff, B=B, K=K)
        rows = [
            {
                "test": f"Rephrase:{inv}",
                "groups": Q,
                "R_per_group": R_eff,
                "observed_D": D_obs,
                "p_value": p_value,
            }
        ]
        append_or_write_csv(
            os.path.join(RESULTS_DIR, "rephrase_prompt_variability.csv"),
            rows,
            header=["test", "groups", "R_per_group", "observed_D", "p_value"],
        )
        print(f"[Rephrase] {inv} appended to rephrase_prompt_variability.csv")


# ----------------------
# Main (Variability)
# ----------------------

def main():
    """
    Appendix C 변동성 분석 전체 실행 엔트리포인트.
    - 수행 항목:
      1) ChatGPT prompts 반복: 반복 변동성(CI, p-value) 및 프롬프트 순열검정
      2) Claude prompts 반복: 동일 절차 수행, 출력 파일명에 접미사(_claude) 포함
      3) Rephrase 순열검정: ChatGPT, Claude 각각 Rephrase JSON에 대해 수행
    - 출력 위치: results/ 디렉터리
    - 주의: 파일 존재 여부를 확인하여 없으면 스킵하며, 개별 단계는 서로 독립적으로 실행
    """
    ensure_dir(RESULTS_DIR)

    # ChatGPT dataset (default)
    if os.path.isfile(REPETITION_JSON):
        run_repetition_variability_from_prompts_json(REPETITION_JSON)
        run_prompt_permutation_from_prompts_json(REPETITION_JSON)
    else:
        print(f"[Repetition/Prompt] JSON not found: {REPETITION_JSON}")

    # Claude dataset (Prompts repetition)
    if os.path.isfile(REPETITION_JSON_CLAUDE):
        run_repetition_variability_from_prompts_json(REPETITION_JSON_CLAUDE, suffix="_claude")
        run_prompt_permutation_from_prompts_json(REPETITION_JSON_CLAUDE, suffix="_claude")
    else:
        print(f"[Repetition/Prompt] JSON not found: {REPETITION_JSON_CLAUDE}")

    # Rephrase (ChatGPT)
    if os.path.isfile(REPHRASE_JSON):
        run_rephrase_permutation_from_rephrase_json(REPHRASE_JSON)
    else:
        print(f"[Rephrase] JSON not found: {REPHRASE_JSON}")

    # Rephrase (Claude)
    if os.path.isfile(REPHRASE_JSON_CLAUDE):
        run_rephrase_permutation_from_rephrase_json(REPHRASE_JSON_CLAUDE)
    else:
        print(f"[Rephrase] JSON not found: {REPHRASE_JSON_CLAUDE}")


if __name__ == "__main__":
    main()
