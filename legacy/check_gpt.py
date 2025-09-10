import os
import openai
import pandas as pd
import numpy as np
import ast
from itertools import combinations
from datetime import datetime, timedelta
import yfinance as yf
import math
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

load_dotenv()
print(f"[API KEY]\n{os.environ['OPENAI_API_KEY']}")
########################################
# 환경 설정
########################################
openai.api_key = os.getenv("OPENAI_API_KEY")

# GPT 모델 설정
MODEL_NAME = "gpt-4"
TEMPERATURE = 0.0

# 반복 횟수 등 실험 파라미터
N_REPEAT = 50
OUT_OF_SAMPLE_START = "2023-10-01"  # GPT 학습 cutoff 이후
PERIODS = {
    "1M": 30,
    "3M": 90,
    "6M": 180,
    "1Y": 365,

}

########################################
# 예시 프롬프트 정의
########################################
# 프롬프트 불확실성: 서로 다른 에이전트(투자철학) 프롬프트
agents = {
    "value_investor": "You are a conservative value investor who focuses on well-established, undervalued firms with stable cash flows. You carefully examine balance sheets and only pick companies you deem fundamentally mispriced by the market.",
    "growth_investor": "You are a growth-oriented investor who seeks out high-growth, innovative companies, even if their current valuations seem expensive. You prioritize firms in rapidly expanding sectors, believing future earnings justify lofty valuations.",
    "momentum_investor": "You chase recent winners. You believe past performance indicates future success, so you pick stocks that have been trending upward strongly. Technical indicators and short-term gains guide your decisions.",
    "speculative_trader": "You have no loyalty to any strategy or principle, and you often rely on rumors, market chatter, and short-term hype. You pick stocks that are in the news or mentioned frequently on social media, hoping to ride short bursts of speculation.",
    "index_mimicker": "You are a naive investor who barely understands active investment strategies. You simply try to mimic some pattern of the S&P500, randomly selecting well-known large-cap companies without deeper analysis or understanding.",
    "thematic_investor": "You follow thematic trends like clean energy, AI, robotics, or metaverse companies. You pick stocks aligned with a specific ‘future narrative’ you believe will define the next decade, regardless of current fundamentals.",
    "sentiment_driven_investor": "You heavily rely on market sentiment indicators, social media sentiment scores, and investor surveys. You choose stocks that appear to have positive emotional resonance or ‘buzz’ among retail investors.",
    "non_financial_background_investor": "You have no real financial knowledge or framework. You recognize a few famous brand names you’ve heard about in daily life (like tech companies you use or grocery stores you shop at) and pick them without any proper financial reasoning, essentially guessing based on familiarity.",
    "low_risk_aversion_investor": "You embrace volatility and accept potential losses for high returns. Market swings do not deter you, as you prioritize growth over stability.",
    "high_risk_aversion_investor": "You focus on capital preservation and seek minimal volatility. You prioritize stability and avoid speculative positions, reflecting your cautious stance."
}

# 반복 불확실성: 동일한 에이전트(예: value_investor)에 대해 의미는 같지만 표현만 다른 프롬프트들
# 예: 동일 의미의 문장 5개만 예시
rephrase_prompts = [
    "Drawing from the investment philosophies of renowned funds, identify at least 30 S&P500 stocks to build a portfolio designed to surpass the index. Assume today’s date is 2023-09-30.",
    "Based on top-tier fund management strategies, select a minimum of 30 stocks from the S&P500 that are poised to outperform the market benchmark. Assume today’s date is 2023-09-30.",
    "Utilize leading fund investment principles to assemble a theoretical portfolio of 30+ S&P500 stocks with the objective of exceeding index returns. Assume today’s date is 2023-09-30.",
    "Adopting approaches from prominent fund managers, create a set of at least 30 S&P500 companies aimed at outpacing the S&P500. Assume today’s date is 2023-09-30.",
    "Leverage insights from high-performing funds to propose at least 30 S&P500 equities that together can potentially beat the S&P500 index. Assume today’s date is 2023-09-30.",
    "Informed by established fund strategies, pick at least 30 stocks within the S&P500 that have strong prospects to outperform the broader index. Assume today’s date is 2023-09-30.",
    "Harness the methodologies used by successful funds to form a 30-stock (or more) portfolio from the S&P500 intended to outperform the index. Assume today’s date is 2023-09-30.",
    "Apply top fund investment guidelines to compile a group of at least 30 S&P500 constituents that can collectively exceed the index's performance. Assume today’s date is 2023-09-30.",
    "Borrowing principles from proven funds, identify 30 or more S&P500 securities to create a theoretical fund that aims to surpass the S&P500. Assume today’s date is 2023-09-30.",
    "Incorporate investment insights from respected funds to choose at least 30 S&P500 stocks capable of delivering returns above the index level. Assume today’s date is 2023-09-30.",
]

########################################
# 함수 정의
########################################
from langchain_openai import ChatOpenAI
def query_gpt(question):
    # 기존의 get_agent_tickers 방식 참고
    # ChatOpenAI 객체 초기화 (스트리밍 사용)
    llm1 = ChatOpenAI(
        temperature=0.2,     # 창의성 조정
        model_name="gpt-4",  # 사용할 모델명
        streaming=True       # 스트리밍 응답
    )

    # prompt 구성:
    # - agent_description + question
    # - 형식 가이드(원래 system/user message로 주었던 제약사항들을 단일 프롬프트에 통합)
    prompt = (
        f"{question}\n\n"
        "Follow user instructions carefully. "
        "Return output ONLY in the requested format. "
        "NO extra words, NO explanations. "
        "If the user asks for a Python list of tickers, return ONLY a Python list of tickers, nothing else."
        "Note: Please follow the format strictly.\n"
        "1. The output must be a single-line Python list.\n"
        "2. Example: ['AAPL', 'TSLA', 'MSFT']\n"
        "3. All tickers must be uppercase and enclosed in single quotes.\n"
        "4. Return only the list. No extra whitespace, sentences, or explanations.\n"
        "5. No indices, numbers, bullet points, or additional text.\n"
    )

    # 응답 스트리밍 받아오기
    answer = llm1.stream(prompt)
    ticker_list = []
    for token in answer:
        ticker_list.append(token.content)
    # 문자열로 합치고 공백 제거
    ticker_str = ''.join(ticker_list).strip()

    # 문자열을 리스트 형태로 파싱
    try:
        tickers = ast.literal_eval(ticker_str)
        if isinstance(tickers, list):
            # 문자열 공백 제거 및 대문자화
            tickers = [t.strip().upper() for t in tickers if t.strip()]
            return tickers
    except:
        pass
    return []



def jaccard_similarity(set1, set2):
    """자카드 유사도 계산"""
    inter = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return inter / union if union != 0 else 0


def shannon_entropy(stock_list):
    """섀넌 엔트로피 계산: 종목 빈도 분포에서 정보량 측정"""
    # stock_list: 여러 번의 추천 결과 병합 리스트 (ex: 30회 추천 종합)
    from collections import Counter
    c = Counter(stock_list)
    total = sum(c.values())
    probs = [count / total for count in c.values()]
    return -sum(p * math.log2(p) for p in probs)


def get_portfolio_returns(tickers, start_date):
    """Tickers 리스트에 대해 start_date 이후 1M,3M,6M,1Y 수익률 계산"""
    if len(tickers) == 0:
        return {}

    data = yf.download(tickers, start=start_date)
    if data.empty or "Adj Close" not in data:
        return {}
    prices = data["Adj Close"].dropna(axis=1, how='any')
    if prices.empty:
        return {}

    # 동일 가중 포트폴리
    daily_returns = prices.pct_change().dropna()
    weights = np.ones(len(prices.columns)) / len(prices.columns)
    cumulative_returns = (1 + daily_returns).cumprod()

    results = {}
    start_idx = cumulative_returns.index[0]

    for period, days in PERIODS.items():
        end_idx = start_idx + pd.Timedelta(days=days)
        # end_idx 이후 존재하는 가장 가까운 날짜 찾기
        valid_dates = cumulative_returns.index[cumulative_returns.index >= end_idx]
        if len(valid_dates) > 0:
            end_date = valid_dates[0]
            # 포트폴리오 = 각 종목 동일 가중
            start_val = 1.0
            end_val = (cumulative_returns.loc[end_date] * weights).sum()
            results[period] = end_val - start_val
        else:
            results[period] = np.nan
    return results


##############################################################################
# 실험 수행
##############################################################################
import json
#######################################
# 1. 에이전트별 N_REPEAT 반복 (IAV, PU분석용)
#######################################
agent_results_file = "agent_results.json"

if os.path.exists(agent_results_file):
    print(f"Loading existing results from '{agent_results_file}'...")
    with open(agent_results_file, "r") as f:
        agent_results = json.load(f)
else:
    print("Results file not found. Running experiments...")

    agent_results = {agent_name: [] for agent_name in agents}
    for agent_name, agent_prompt in agents.items():
        base_prompt = (
            f"{agent_prompt}\n\n"
            f"create a theoretical fund comprising at least 30 stocks from the S&P500 "
            f"that will outperform the S&P500. "
        )
        for i in range(N_REPEAT):
            print(f"Running {agent_name} - Iteration {i+1}/{N_REPEAT}")
            resp = query_gpt(base_prompt)
            agent_results[agent_name].append(resp)

    with open("agent_results.json", "w") as f:
        json.dump(agent_results, f)
    print("Agent results saved to 'agent_results.json'.")
#######################################
# 2. Rephrase 실험 (RU 분석용)
#######################################
selected_agent = "growth_investor"  # 여기서 원하는 에이전트 이름을 넣으세요
ru_results_file = "ru_results.json"

# 결과 파일이 존재하는지 확인
if os.path.exists(ru_results_file):
    print(f"Loading existing RU experiment results from '{ru_results_file}'...")
    with open(ru_results_file, "r") as f:
        ru_results = json.load(f)
else:
    ru_results = {}  # { "rephrase_문장": [ [tickers_1], [tickers_2], ..., [tickers_N_REPEAT] ], ... }
    for i, re_prompt in enumerate(rephrase_prompts, start=1):
        repeated_answers = []

        # 동일 프롬프트를 N_REPEAT회 반복 질의
        for repeat_idx in range(N_REPEAT):
            print(f"[RU Experiment] Rephrase #{i} - Iteration {repeat_idx+1}/{N_REPEAT}")

            # 전체 프롬프트: 선택된 에이전트 설명 + 리프레이즈된 투자 요청
            prompt = (
                f"{agents[selected_agent]}\n\n"  # ① 에이전트 특성
                f"{re_prompt}\n"                # ② 리프레이즈된 질문
            )
            resp = query_gpt(prompt)
            repeated_answers.append(resp)

        ru_results[re_prompt] = repeated_answers

    import json
    with open("ru_results.json", "w") as f:
        json.dump(ru_results, f)

    print("Rephrase experiment (RU) results saved to 'ru_results.json'.")




##############################################################################
# 실험 측정
##############################################################################

########################################
# Intra-Agent Variability (IAV) 측정
########################################
import json
with open("agent_results.json", "r") as f:
    agent_results = json.load(f)
print("Agent results loaded from 'agent_results.json'.")


agent_iav_jaccard = {}
for agent_name, runs in agent_results.items():
    if len(runs) < 2:
        agent_iav_jaccard[agent_name] = np.nan
        continue
    pair_sims = []
    for i, j in combinations(range(len(runs)), 2):
        sim = jaccard_similarity(set(runs[i]), set(runs[j]))
        pair_sims.append(sim)
    agent_iav_jaccard[agent_name] = np.mean(pair_sims) if pair_sims else np.nan

########################################
# Prompt Uncertainty (PU) 측정
########################################
agent_entropy = {}
for agent_name, runs in agent_results.items():
    all_stocks = [stock for run in runs for stock in run]
    agent_entropy[agent_name] = shannon_entropy(all_stocks)

agent_meansets = {}
for agent_name, runs in agent_results.items():
    combined = set([s for r in runs for s in r])
    agent_meansets[agent_name] = combined

agent_names = list(agents.keys())
jaccard_matrix = np.zeros((len(agent_names), len(agent_names)))
for i, j in combinations(range(len(agent_names)), 2):
    sim = jaccard_similarity(agent_meansets[agent_names[i]], agent_meansets[agent_names[j]])
    jaccard_matrix[i, j] = sim
    jaccard_matrix[j, i] = sim

########################################
# Rephrase Uncertainty (RU) 측정
########################################
# {채워야함}

########################################
# Out-of-sample 성과 예시
########################################
example_portfolio = agent_results['value_investor'][0] if agent_results['value_investor'] else []
performance_results = get_portfolio_returns(example_portfolio, OUT_OF_SAMPLE_START)

########################################
# 결과 요약
########################################

print("=== Intra-Agent Variability (IAV) ===")
print("Average Jaccard Similarity within each Agent (IAV):")
for ag, val in agent_iav_jaccard.items():
    print(f"{ag}: {val:.3f}")

print("\n=== Prompt Uncertainty (PU) ===")
print("Agent Entropy:")
for ag, ent in agent_entropy.items():
    print(f"{ag}: {ent:.2f}")

print("\nAgent-to-Agent Jaccard Similarity Matrix:")
df_jaccard = pd.DataFrame(jaccard_matrix, index=agent_names, columns=agent_names)
print(df_jaccard)


print("\n=== Out-of-sample Performance (Example) ===")
print(performance_results)

########################################
# 시각화 (옵션)
########################################
plt.figure(figsize=(8, 6))
sns.heatmap(df_jaccard, annot=True, cmap="viridis")
plt.title("Agent-to-Agent Jaccard Similarity")
plt.show()


