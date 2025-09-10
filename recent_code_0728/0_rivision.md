### 논문 리비전 작업 To-Do List (v5.0 - 업데이트 버전)

이 To-Do List는 이전 v4.0을 기반으로 사용자 지시를 반영하여 업데이트되었습니다. 주요 변경점:
- **Rephrase 파일 확장 완료 가정**: `/Users/jaehoon/Alphatross/70_Research/checkgpt/results/Rephrase/Rephrase_Repetition_Result_NVG.json`이 이미 100 reps(기존 50 + 추가 50)으로 확장된 상태로 가정. (만약 필요 시 code_execution으로 확인/재확장 가능, 하지만 완성 상태이므로 진행.)
- **최소 수정 원칙 세부 반영**: 
  - 기존 50 reps 포트폴리오로 대응할 쟁점(쟁점 1-6, 8)에서는 각 프롬프트(Neutral/Value/Growth)/리프레이즈(10개)별 분석을 50 reps만 사용 (기존 데이터 subset 추출로 최소 계산). 100 reps는 비교/footnote로만 (변화 적으면 무시, "robustness check"로 해석).
  - 100 reps 사용 쟁점(쟁점 7): 전체 100 reps 필수, 각 프롬프트/리프레이즈별 분석 시 100 reps 기반.
  - 포트폴리오 크기(쟁점 6): top 10/20/30개 대표 포트폴리오 사용 명확히 반영 (frequency 기반 선정, 50 reps로 기본 분석, 100 reps 비교).
- **파일/코드 참조**: Rephrase 파일 확장 완료로 `/Rephrase_Repetition_Result_NVG_100.json` (가정) 사용. `generate_portfolios_multi_llm.py`로 추가 데이터 생성 최소화. `new.py`로 보조.
- **API/비용 관리**: 추가 API 호출 피함 (기존 확장 완료 가정). 필요 시 Gemini/Claude 50 reps subset.
- **도구 활용**: code_execution으로 JSON subset 추출/분석 (e.g., 50 reps subset). web_search로 데이터 보완 (e.g., market cap). 일정: 2025.07.28(현재) 마무리, 2025.07.29 미팅 공유.
- **논문 영향**: 최소 수정으로 기존 Table/Figure 유지, new/expanded 항목에 50/100/top 크기 명시. 전체 원칙: 최소 노력, "했다" 강조.

#### 1. Jaccard Similarity 분포 확인 및 멀티모달 여부 (쟁점 1: 분포 그리기, 멀티모달 체크)
   - **목표**: 분포 확인. 최소 수정: 50 reps로 기본 분석 (각 프롬프트/리프레이즈별).
   - **현재 상황**: Rephrase 파일 100 reps 완료. 기존 50 reps subset 사용.
   - **작업 단계**:
     1. code_execution으로 Rephrase 파일 로드: 100 reps JSON 로드, 각 프롬프트(3개)/리프레이즈(10개)별 첫 50 reps subset 추출 (기존 데이터 재현). 코드 예:
        ```python
        import json
        with open('/Users/jaehoon/Alphatross/70_Research/checkgpt/results/Rephrase/Rephrase_Repetition_Result_NVG.json', 'r') as f:  # 100 reps 파일
            data = json.load(f)
        subsets = {}
        for prompt in data:
            subsets[prompt] = {}
            for rephrase in data[prompt]:
                subsets[prompt][rephrase] = data[prompt][rephrase][:50]  # 첫 50 reps subset
        with open('/Users/jaehoon/Alphatross/70_Research/checkgpt/results/Rephrase/Rephrase_Repetition_Result_NVG_50_subset.json', 'w') as f:
            json.dump(subsets, f)
        ```
     2. 50 reps subset으로 pairwise Jaccard/히스토그램 계산 (matplotlib.hist, 각 프롬프트/리프레이즈별 플롯).
     3. 100 reps 전체로 비교 (mean/SD 변화 체크, <5%면 footnote).
     4. top 10/20/30 영향 없음 (분포 분석에 불필요).
   - **예상 결과**: Unimodal, 50 vs 100 유사.
   - **난이도/시간**: 쉽음, 1-2시간 (subset 30분, 분석 1시간).
   - **논문 수정**: Results > Repetition Variability에 Figure (50 reps 기반), "100 reps 비교 결과 유사" footnote.

#### 2. 퍼포먼스 기간 연장 (쟁점 2: 1년 → 2년)
   - **목표**: 기간 연장. 최소 수정: 50 reps로 대표 포트폴리오 (top 30) 계산 (각 프롬프트/리프레이즈별).
   - **현재 상황**: 데이터 업데이트 가능.
   - **작업 단계**:
     1. `new.py`로 2년 데이터 로드.
     2. 50 reps subset (위 1번 연계)으로 각 프롬프트/리프레이즈별 top 30 선정 (frequency sort), returns/Sharpe 계산.
     3. 100 reps로 top 30 비교만 (변화 최소 시 무시).
     4. top 10/20: 여기서는 top 30만 (쟁점 6 연계 별도).
   - **예상 결과**: 안정.
   - **난이도/시간**: 쉽음, 1시간.
   - **논문 수정**: Table 4 확장 (50 reps 기반), top 크기 언급 없음.

#### 3. Sharpe Ratio 계산 방법 명확화 (쟁점 3: 설명 추가)
   - **목표**: 설명 추가. 최소 수정: 50 reps 기반 유지, 영향 없음.
   - **작업 단계**:
     1. 텍스트 수정 (web_search: "3-month T-bill rate July 2025").
     2. 100 reps/ top 크기 영향 없음.
   - **예상 결과**: N/A.
   - **난이도/시간**: 매우 쉽음, 30분.
   - **논문 수정**: Footnote 추가.

#### 4. Equal-weighted 포트폴리오 결과 적용 (쟁점 4: Equal vs. Market-weighted)
   - **목표**: Equal 계산. 최소 수정: 50 reps로 분석 (각 프롬프트/리프레이즈별 top 30 사용).
   - **작업 단계**:
     1. 50 reps subset으로 equal weights 적용.
     2. 각 리프레이즈별 top 30 기반 계산.
     3. 100 reps 비교 footnote.
     4. top 10/20: 불필요 (top 30 중심).
   - **예상 결과**: Robust.
   - **난이도/시간**: 쉽음, 1시간.
   - **논문 수정**: Table 5 (50 reps), "Per prompt/rephrase" 명시.

#### 5. 다른 모델 적용과 변동성 확인 (쟁점 5: Gemini/Claude 사용)
   - **목표**: 모델 비교. 최소 수정: 50 reps로 기본 (각 프롬프트/리프레이즈별 Jaccard).
   - **현재 상황**: OpenAI/Claude 100 reps, Gemini 추가 최소.
   - **작업 단계**:
     1. Gemini: 50 reps subset 사용 (기존 확장 파일에서 추출).
     2. 50 reps로 각 프롬프트/리프레이즈별 분석 (temperature 체크).
     3. 100 reps: OpenAI/Claude만 비교.
     4. top 10/20/30: 변동성 확인 시 top 30만.
   - **예상 결과**: Variation.
   - **난이도/시간**: 중간, 1-2시간.
   - **논문 수정**: New 섹션 (50 reps 기반).

#### 6. 포트폴리오 크기 조정 (쟁점 6: 30 → 10/20개, 오버랩 변화)
   - **목표**: top 10/20/30 재계산. 최소 수정: 50 reps로 기본 분석 (각 프롬프트/리프레이즈별 frequency 기반 top 선정).
   - **현재 상황**: 리스트 있음.
   - **작업 단계**:
     1. 50 reps subset으로 각 프롬프트/리프레이즈별 frequency dict 생성.
     2. top 10/20/30 추출 (sorted(freq, reverse=True)[:10], [:20], [:30]).
     3. 각 top 크기별 pairwise Jaccard/overlap 계산 (e.g., heatmap per size).
     4. 100 reps: 비교만 (top 10/20/30 변화 체크, footnote).
   - **예상 결과**: 크기 줄수록 overlap 증가.
   - **난이도/시간**: 쉽음, 1-2시간 (top별 루프 1시간).
   - **논문 수정**: Prompt Uncertainty 확장, New Figure (top 10/20/30 heatmap, 50 reps 기반).

#### 7. Repetition 수 증가 (쟁점 7: 50 → 100)
   - **목표**: 100 reps 필수. 각 프롬프트/리프레이즈별 100 reps 분석.
   - **현재 상황**: 파일 확장 완료.
   - **작업 단계**:
     1. 전체 100 reps JSON 사용 (Value Investor 중심, 다른 프롬프트 확장).
     2. 각 리프레이즈(10개)별 100 reps Jaccard/Entropy.
     3. top 10/20/30: 100 reps frequency로 top 선정, 쟁점 6 연계 비교.
   - **예상 결과**: 유사.
   - **난이도/시간**: 중간, 1시간.
   - **논문 수정**: Repetition Variability 확장 (100 reps Table/Figure).

#### 8. Weighted Jaccard Similarity (쟁점 8: Market cap weighted, 수익률 분포)
   - **목표**: Weighted 계산. 최소 수정: 50 reps로 분석 (각 프롬프트/리프레이즈별 top 30 사용).
   - **작업 단계**:
     1. Market cap: web_search 또는 `new.py` 로드.
     2. 50 reps subset으로 weighted Jaccard (top 30 벡터, sum min/max).
     3. 수익률 분포: 50 reps 플롯 (각 리프레이즈별).
     4. 100 reps 비교, top 10/20: weighted 적용 (top 크기별 계산).
   - **예상 결과**: Uncertainty 감소.
   - **난이도/시간**: 중간, 2시간.
   - **논문 수정**: Uncertainty Quantification 확장, Figure (50 reps, top 10/20/30 포함).

**최종 작업**:
- 결과 취합: Subset JSON 저장, top 크기 CSV.
- 논문 업데이트: 최소 변경, 50/100/top 명시.
- 미팅 준비: 요약 (e.g., "50 reps 최소, 100/top 세부 반영"). 총 시간: 8-10시간 (오늘 마무리).