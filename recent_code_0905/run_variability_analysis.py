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
# JSON-based runners (Variability)
# ----------------------

def run_repetition_variability_from_prompts_json(file_path: str, suffix: Optional[str] = None):
    names, groups = load_prompts_repetition_json(file_path)
    if not groups:
        print(f"[Repetition] No data found in {file_path}. Skipping.")
        return
    rows = []
    for name, reps in zip(names, groups):
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
