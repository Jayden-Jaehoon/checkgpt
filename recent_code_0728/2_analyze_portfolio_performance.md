### 논문 리비전 작업 To-Do List (v7.0 - 업데이트 버전)

이 To-Do List는 이전 v6.0을 기반으로 사용자 지시를 반영하여 업데이트되었습니다. 주요 변경점:
- **쟁점 2 세부화**: 사용자의 "이부분 좀더 자세히 작성해줘" 요청에 따라 쟁점 2( 퍼포먼스 기간 연장)를 더 상세히 확장. 작업 단계에 코드 예시 확장( yfinance 데이터 로드, top 30 frequency 계산, returns/Sharpe 공식 및 코드, 에러 핸들링), 단계별 이유/예상 출력 설명, 도구 활용( code_execution 호출 예시, web_search로 risk-free rate 확인) 포함. 최소 수정 원칙 유지(50 reps 기본, 100 reps 비교, top 30 중심).
- **전체 원칙 유지**: 다른 쟁점은 변경 없음. 최소 노력, 50/100 reps/top 크기 구분 명확. 파일 확장 완료 가정. 일정: 2025.07.28(현재) 기준 즉시 실행 가능, 2025.07.29 미팅 전 공유.
- **도구 활용**: 쟁점 2에서 code_execution 실제 호출 가능( 데이터 로드/계산 테스트). web_search로 risk-free rate 업데이트( 쟁점 3 연계). browse_page로 Yahoo Finance 페이지 확인( 필요 시, e.g., 티커 리스트).

#### 1. Jaccard Similarity 분포 확인 및 멀티모달 여부 (쟁점 1: 분포 그리기, 멀티모달 체크)
(이전 버전과 동일, 변경 없음.)

#### 2. 퍼포먼스 기간 연장 (쟁점 2: 1년 → 2년)
   - **목표**: 리뷰어의 지적( 기존 1년 기간 짧음)에 대응해 2년(2023.10.1 ~ 2025.07.28)으로 연장. 최소 수정: 50 reps로 대표 포트폴리오( top 30) 계산(각 프롬프트/리프레이즈별), Table 4 확장. 100 reps는 top 30 변화 확인으로만( 변화 <5% 시 footnote 무시). top 10/20은 쟁점 6으로 별도( 여기서는 top 30만 집중, 퍼포먼스 중심).
   - **현재 상황**: Yahoo Finance 데이터 수집 코드( `new.py` ) 사용 가능( yfinance 라이브러리 기반). 기존 기간 ~1년 9개월, 2년 연장 시 데이터 업데이트 필요. S&P 500 티커 리스트( ~500개 종목) 가정( JSON 또는 CSV 로드). risk-free rate는 쟁점 3과 연계( 기본 4% 또는 최신 값).
   - **작업 단계** (상세 확장):
     1. **new.py로 2년 데이터 로드 (주가 데이터 업데이트)**:
        - 이유: 기존 1년(2023.10.1 ~ 2024.09.30)에서 2년(2023.10.1 ~ 2025.07.28)으로 연장. yfinance로 daily returns 다운로드( adjusted close 사용, 결측치 처리). 각 프롬프트/리프레이즈별 top 30 종목에만 적용( 전체 S&P 500 다운로드 후 subset).
        - 도구 활용: code_execution 호출로 실제 로드/테스트( yfinance 설치 불필요, 기본 라이브러리). 만약 API 제한 시 web_search( query="S&P 500 historical prices CSV download July 2025" ) 또는 browse_page( url="https://finance.yahoo.com/quote/%5EGSPC/history/", instructions="Extract S&P 500 daily adjusted close from 2023-10-01 to 2025-07-28, summarize top stocks returns." ).
        - 코드 예시(확장: 티커 리스트 가정, 결측치 fill, CSV 저장):
          ```python
          import yfinance as yf  # 기본 라이브러리
          import pandas as pd
          import os  # 파일 확인용

          # S&P 500 티커 리스트 가정 (실제로는 JSON/CSV 로드, e.g., ['AAPL', 'MSFT', ...])
          sp500_tickers = pd.read_csv('/Users/jaehoon/Alphatross/70_Research/checkgpt/sp500_tickers.csv')['Symbol'].tolist()  # 가정 경로

          # 2년 데이터 로드 (start/end 지정, adjusted close)
          start_date = '2023-10-01'
          end_date = '2025-07-28'  # 현재 날짜 기반
          data = yf.download(sp500_tickers, start=start_date, end=end_date, group_by='ticker', progress=False)['Adj Close']
          
          # 결측치 처리 (forward fill, dropna)
          data = data.fillna(method='ffill').dropna(axis=1, how='all')
          
          # daily returns 계산 (pct_change)
          returns = data.pct_change().dropna()
          
          # 데이터 저장 (후속 사용)
          data_path = '/Users/jaehoon/Alphatross/70_Research/checkgpt/results/sp500_2year_returns.csv'
          returns.to_csv(data_path)
          
          # 에러 핸들링 및 확인 출력
          if returns.empty:
              print("Error: No data downloaded. Check internet or dates.")
          else:
              print(f"Data loaded successfully: Shape {returns.shape}, Columns {len(returns.columns)}")  # e.g., (500 days x 500 tickers)
          ```
        - 예상 출력: "Data loaded successfully: Shape (approx 500 x 500)". 에러 시 티커 제한( e.g., top 100만 로드).
        - 시간: 30분 ( code_execution 호출 1회, 다운로드 5-10분).

     2. **50 reps subset (위 1번 연계)으로 각 프롬프트/리프레이즈별 top 30 선정 (frequency sort), returns/Sharpe 계산**:
        - 이유: 최소 수정 위해 50 reps subset 사용( 쟁점 1 subset 파일 연계). frequency로 top 30 선정( repetition 빈도 높은 종목). value-weighted( market cap 비중) 또는 equal-weighted( 쟁점 4 연계) 포트폴리오 returns 계산. 1-month/6-month/1-year/2-year horizons별 cumulative returns, annualized volatility, Sharpe( risk-free 뺀 excess return / vol).
        - 도구 활용: code_execution으로 계산. risk-free rate 업데이트: web_search( query="3-month US Treasury bill rate July 2025 average" ).
        - 코드 예시(확장: Counter로 frequency, pd로 returns, Sharpe 공식):
          ```python
          from collections import Counter
          import numpy as np
          import pandas as pd
          import json

          # 50 reps subset 로드 (쟁점 1 연계)
          with open('/Users/jaehoon/Alphatross/70_Research/checkgpt/results/Rephrase/Rephrase_Repetition_Result_NVG_50_subset.json', 'r') as f:
              subsets = json.load(f)

          # 2년 returns 로드 (위 단계)
          returns = pd.read_csv('/Users/jaehoon/Alphatross/70_Research/checkgpt/results/sp500_2year_returns.csv', index_col=0, parse_dates=True)

          # risk-free rate (web_search 결과 가정, e.g., 0.04 / 252 daily)
          rf_daily = 0.04 / 252  # annualized 4% 예시

          perf_results = {}  # {prompt: {rephrase: {'top30': list, 'cum_return_2y': float, 'sharpe_2y': float, ...}}}
          for prompt in subsets:  # 3개 프롬프트
              perf_results[prompt] = {}
              for rephrase in subsets[prompt]:  # 10개 리프레이즈
                  reps_lists = subsets[prompt][rephrase]  # 50 reps 리스트
                  # frequency sort로 top 30 선정 (Counter flatten)
                  all_stocks = [stock for rep in reps_lists for stock in rep]  # flat list
                  freq = Counter(all_stocks)
                  top30 = [stock for stock, count in freq.most_common(30)]  # top 30 tickers
                  
                  # top30 returns subset (value-weighted 가정, market cap 별도 로드 필요 시)
                  top_returns = returns[top30].copy()  # pd.DataFrame
                  weights = np.ones(len(top30)) / len(top30)  # equal-weighted 예시 (value-weighted 시 market cap 추가)
                  port_returns = (top_returns * weights).sum(axis=1)  # daily portfolio returns
                  
                  # horizons 계산 (cumulative return, annualized vol, Sharpe)
                  horizons = {'1m': 21, '6m': 126, '1y': 252, '2y': len(port_returns)}  # trading days approx
                  for h_name, days in horizons.items():
                      h_returns = port_returns[-days:] if days < len(port_returns) else port_returns
                      cum_return = (1 + h_returns).cumprod()[-1] - 1
                      ann_vol = h_returns.std() * np.sqrt(252)
                      excess = h_returns - rf_daily
                      sharpe = (excess.mean() * 252) / (excess.std() * np.sqrt(252)) if ann_vol > 0 else 0
                      perf_results[prompt][rephrase][f'{h_name}_cum_return'] = cum_return
                      perf_results[prompt][rephrase][f'{h_name}_sharpe'] = sharpe
                  
                  perf_results[prompt][rephrase]['top30'] = top30
          
          # 결과 요약 출력/저장 (Table 4용)
          print(perf_results)  # e.g., {'Neutral': {'rephrase_1': {'2y_cum_return': 0.35, '2y_sharpe': 1.2, ...}}}
          pd.DataFrame(perf_results).to_csv('/Users/jaehoon/Alphatross/70_Research/checkgpt/results/perf_50reps.csv')
          ```
        - 예상 출력: perf_results 딕셔너리( horizons별 값, e.g., 2y_cum_return ~0.3-0.4, sharpe ~2.0). S&P 500 벤치마크 비교 추가( 별도 계산).
        - 에러 핸들링: 티커 없으면 skip( returns[top30] KeyError ), freq.most_common(30) 부족 시 pad.
        - 시간: 30분 ( code_execution 호출 1회).

     3. **100 reps로 top 30 비교만 (변화 최소 시 무시)**:
        - 이유: robustness 확인(50 vs 100 top 30 overlap 체크). 전체 파일 사용, but 계산만( Jaccard로 top 30 유사도 <0.05 변화 시 footnote 무시).
        - 도구 활용: code_execution으로 위 코드 수정( subsets -> data 전체 100 reps, freq from all 100).
        - 코드 예시(간략): 위 freq = Counter([stock for rep in full_reps for stock in rep]), top30_100 = most_common(30), overlap = len(set(top30_50) & set(top30_100)) / 30.
        - 예상 출력: overlap >0.95( 변화 최소), "Delta sharpe <0.1".
        - 시간: 15분.

     4. **top 10/20: 여기서는 top 30만 (쟁점 6 연계 별도)**:
        - 이유: 이 쟁점은 기간 연장/퍼포먼스 중심, top 크기 변형은 쟁점 6으로 별도. 여기서는 top 30만 계산( 쟁점 6에서 top 10/20 퍼포먼스 추가 가능).
        - 작업: N/A, but 확인 시 top 10/20 frequency로 sharpe 테스트( but 최소화로 skip).
   - **예상 결과**: 장기(2년) 성능 안정( cum_return 증가, vol 감소 예상). 50 vs 100 유사( 변화 최소). S&P 500 벤치마크 outperformance 유지.
   - **난이도/시간**: 쉽음, 총 1-1.5시간 ( code_execution 2회 호출).
   - **논문 수정**: Portfolio Performance 섹션 Table 4 확장( new columns for 2y, 50 reps 기반 mean/SD per prompt/rephrase). Footnote: "100 reps top 30 비교: 변화 <5%, 안정 확인". top 크기 언급 없음( 쟁점 6 참조). "Horizons: 1m/6m/1y/2y" 추가 설명.

(이하 쟁점 3~8은 이전 버전과 동일, 변경 없음. 전체 To-Do List 유지.)