# 수치 안정화 변경 사항 요약서 (2025-09-12)

본 문서는 최근 코드(recent_code_0905)에서 수행한 “수치 안정화(Numerical Stabilization)” 관련 변경들을 정리합니다. 주요 목표는 대규모 반복 연산 및 포트폴리오 산출 과정에서 발생할 수 있는 부동소수점 오버플로우/언더플로우, 비정상 가중치(inf/NaN), 단위 불일치로 인한 통계적 오류를 예방·완화하는 것입니다.

대상 주요 파일:
- portfolio_analysis.py
- variability_analysis.py
- utils/data_loader.py

---

## 1) 누적수익·연율수익 계산의 로그-공간 안정화
- 위치: portfolio_analysis.py / backtest_portfolio, regime_performance
- 변경 전: (1 + r).prod() − 1 방식으로 누적수익 계산, 연율수익도 동일한 곱셈 누적 기반
- 변경 후: 
  - 누적수익: log1p(r)을 일별 합산 → expm1(합계)로 변환
    - 코드 개념: `logret = np.log1p(port_rets); cum_return = float(np.expm1(logret.sum()))`
  - 연율수익: 일간 로그수익 평균을 252배 후 expm1로 변환
    - 코드 개념: `ann_return = float(np.expm1(logret.mean() * 252.0))`
- 이유: 곱셈 누적은 긴 기간/변동성 큰 구간에서 오버플로우/언더플로우 위험이 큼. log-공간 합산은 수치적으로 훨씬 안정적.

## 2) 최대낙폭(MDD) 계산의 로그-부(wealth in log) 변환
- 위치: portfolio_analysis.py / max_drawdown
- 변경 전: (1 + r).cumprod()로 wealth를 만든 후 peak 대비 하락률 계산
- 변경 후: 
  - wealth를 로그 누적합(log-wealth)으로 계산한 뒤, `exp(current − peak) − 1` 방식으로 drawdown 산출
  - 코드 개념: 
    ```python
    logw = np.log1p(series).cumsum()
    peak_logw = np.maximum.accumulate(logw.values)
    dd = np.exp(logw.values - peak_logw) - 1.0
    mdd = float(dd.min())
    ```
- 이유: 긴 시계열 또는 큰 수익률 구간에서 누적곱 기반 계산은 overflow에 취약. 로그 변환은 안정성과 정밀도를 개선.

## 3) Sharpe 신뢰구간의 단위 일치 및 연율화 절차 수정 (JK/Lo)
- 위치: portfolio_analysis.py / backtest_portfolio, calculate_sharpe_se
- 핵심: Jobson-Korkie/Lo 근사식은 SR과 표본 수 T가 같은 시간단위여야 유효. 
- 절차(최종):
  1. 일별 초과수익률의 SR(daily) 계산: `sr_daily = mean(excess)/std(excess)`
  2. JK/Lo 근사식으로 일별 SE 계산: `se_daily = sqrt((1/T) * (1 + 0.5 * sr_daily^2))`
  3. SR과 SE 모두를 연율화: `sr_ann = sr_daily * sqrt(252)`, `se_ann = se_daily * sqrt(252)`
  4. 95% CI: `sr_ann ± 1.96 * se_ann`
- 이유: 연율화된 SR에 ‘일별 T’를 바로 대입하면 표준오차가 과대/과소추정될 수 있음. 올바른 단위 일치를 통해 통계적 타당성 확보.

## 4) 역변동성 가중치(inverse-vol) 계산의 예외값 처리
- 위치: portfolio_analysis.py / calculate_inverse_volatility_weights, backtest_portfolio, regime_performance
- 내용:
  - lookback 기간의 표준편차 `vol`에서 0을 NaN으로 치환
  - `inv_vol = 1/vol` 계산 후 inf/NaN 제거
  - 유효 종목이 사라질 경우 equal-weight로 폴백
  - 최종 가중치 합이 0 또는 NaN이면 equal-weight로 폴백
- 이유: 0 변동성 또는 비정상 데이터는 가중치 폭주(inf)와 NaN을 유발. 방어적 처리로 안정적 포트폴리오 가중치 확보.

## 5) 시장가중(market-cap weight) 계산의 방어 로직
- 위치: portfolio_analysis.py / calculate_market_cap_weights
- 내용:
  - 시가총액 매핑에서 비양의(≤0), inf/NaN 값 제거
  - 남은 종목에 대해 합계로 나누어 정규화
  - 최종 가중치 합이 0이면 equal-weight로 폴백
- 이유: 일부 종목의 시총 결측/비정상치가 전체 가중치 불능 상태를 야기할 수 있어, 안전한 재정규화 필요.

## 6) 퍼센트 단위 수익률의 자동 정규화
- 위치: portfolio_analysis.py / _normalize_returns_frame
- 내용:
  - 사전계산 returns 프레임이 “퍼센트(예: 0.5 ⇒ 0.5%)” 단위로 저장된 경우를 탐지(절대 99% 분위수 > 1.5 기준)하면 100으로 나눠 정규화
- 이유: 동일 코드가 퍼센트/소수 단위를 혼용할 때 Sharpe, 변동성, 누적수익이 왜곡되는 문제 예방.

## 7) 위험무위험(rf) 시계열 정렬 및 결측 처리
- 위치: portfolio_analysis.py / backtest_portfolio, regime_performance
- 내용:
  - 포트폴리오 수익률 인덱스에 rf를 reindex, ffill 후 결측을 0으로 대체
- 이유: 거래일 비정합으로 인한 결측/정렬 문제를 방지하여 Sharpe 계산의 안정성 확보.

## 8) Jaccard 기반 신뢰구간의 범위 클리핑
- 위치: variability_analysis.py / compute_ci_from_T_distribution
- 내용:
  - 서브샘플링 기반 CI 계산 후 결과를 [0,1] 범위로 클리핑
- 이유: 수치 오차로 인해 이론적 범위 밖으로 벗어나는 값을 방지.

## 9) 진행 상태 표시(tqdm)로 장기 반복의 가시성 개선
- 위치: variability_analysis.py, run_variability_analysis.py, run_portfolio_analysis.py
- 내용:
  - Subsampling/Permutation, 백테스트/국면분석 루프에 tqdm 진행바 적용
- 이유: 매우 많은 반복(B=5000 등) 실행 시 진행률 가시화로 중간 상태 파악 및 운영 안정성 개선(직접적인 수치 안정화는 아니나 실무 안정성 향상에 기여).

---

## 기타 방어적 처리 요약
- 가중치 재정렬/정규화: reindex 후 dropna, sum 검사 → 0 또는 NaN이면 equal-weight 폴백
- 마켓캡/수익률 입력의 부분 누락: 가용 컬럼만 선택, 전부 결측 시 해당 포트폴리오 스킵 또는 NaN 결과 반환
- Bear 서브샘플 고정 기간 사용: 동적 탐지 대신 기간 고정 시, 데이터 정렬/구간 추출의 일관성 향상(분석 재현성 관점)

---

## 검증 방법(권장)
- 극단적 값 시뮬레이션: 큰 양/음의 일간 수익률을 인위적으로 삽입한 뒤 누적수익, MDD, Sharpe 계산이 유한값으로 유지되는지 확인
- 단위 혼동 테스트: 퍼센트 스케일 returns와 소수 스케일 returns 각각으로 동일 포트폴리오를 돌려 결과가 합리적 배수 관계를 갖는지 확인
- 가중치 특이값 테스트: 변동성 0인 가상 종목, 시총 0/inf/NaN 종목 포함 시 가중치 벡터가 정상화되는지 확인

---

## 변경 영향 범위
- 함수: backtest_portfolio, regime_performance, max_drawdown, calculate_inverse_volatility_weights, calculate_market_cap_weights, calculate_sharpe_se, _normalize_returns_frame
- 파일: portfolio_analysis.py, variability_analysis.py, run_* 스크립트들(진행바/로깅)

---

## 결론
위 변경들로 인해 장기간/고변동 구간에서의 누적계산 안정성, 가중치 계산의 견고성, Sharpe 통계 추정의 일관성을 크게 개선했습니다. 특히 로그-공간 합산과 JK/Lo 단위 일치 절차는 금융 시계열 분석의 표준적 안정화 기법으로, 논문 리비전에서 요구하는 재현성과 신뢰성 제고에 직접 기여합니다.
