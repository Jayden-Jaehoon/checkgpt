### 논문 리비전 작업 To-Do List (v6.0 - 업데이트 버전)

이 To-Do List는 이전 v5.0을 기반으로 사용자 지시를 반영하여 업데이트되었습니다. 주요 변경점:
- **쟁점 1 세부화**: 사용자의 "이부분 좀더 자세히 작성해줘" 요청에 따라 쟁점 1( Jaccard Similarity 분포 확인 및 멀티모달 여부)을 더 상세히 확장. 작업 단계에 코드 예시 확장( Jaccard 계산 함수, 히스토그램 플롯 코드 추가), 단계별 이유/예상 출력 설명, 에러 핸들링 팁, 도구 활용( code_execution 호출 예시) 포함. 최소 수정 원칙 유지(50 reps 기본, 100 reps 비교).
- **전체 원칙 유지**: 다른 쟁점은 변경 없음. 최소 노력, 50/100 reps/top 크기 구분 명확. 파일 확장 완료 가정. 일정: 2025.07.28(현재) 기준 즉시 실행 가능, 2025.07.29 미팅 전 공유.
- **도구 활용**: 쟁점 1에서 code_execution 실제 호출 가능(파일 로드/계산 테스트). 필요 시 web_search로 보조 데이터( e.g., Jaccard 문헌).

#### 1. Jaccard Similarity 분포 확인 및 멀티모달 여부 (쟁점 1: 분포 그리기, 멀티모달 체크)
   - **목표**: 리뷰어의 지적( Jaccard의 평균/SD 외 분포를 보고 멀티모달 여부 확인)에 대응. 기존 분석 확장으로 최소 수정: 50 reps로 기본 분포/히스토그램 생성(각 프롬프트/리프레이즈별 pairwise 계산), 100 reps는 robustness check로 비교만(변화 미미 시 footnote). 멀티모달(여러 피크)이면 "분포 특징 인정" 해석, unimodal이면 "문제없음"으로 대응.
   - **현재 상황**: Rephrase 파일( `/Users/jaehoon/Alphatross/70_Research/checkgpt/results/Rephrase/Rephrase_Repetition_Result_NVG.json` )이 100 reps로 확장 완료( Neutral/Value/Growth 3개 프롬프트, 각 10개 리프레이즈 x 100 reps 리스트). 기존 50 reps subset을 추출해 최소 계산.
   - **작업 단계** (상세 확장):
     1. **파일 로드 및 50 reps subset 추출 (기존 데이터 재현)**:
        - 이유: 최소 수정 위해 전체 100 reps 대신 첫 50 reps만 사용(기존 실행 재현, 계산 비용 절감). 각 프롬프트( Neutral, Value, Growth )/리프레이즈(10개)별 subset 생성.
        - 도구 활용: code_execution 호출로 실제 실행/테스트. 만약 파일 접근 에러 시( e.g., 경로 오류) print로 디버깅.
        - 코드 예시(확장: JSON 구조 가정 - data = {'Neutral': {'rephrase_1': [list1, list2, ...], ...}, ...}):
          ```python
          import json
          import os  # 경로 확인용 추가

          # 파일 경로 확인 (에러 방지)
          file_path = '/Users/jaehoon/Alphatross/70_Research/checkgpt/results/Rephrase/Rephrase_Repetition_Result_NVG.json'
          if not os.path.exists(file_path):
              print(f"Error: File not found at {file_path}")
          else:
              with open(file_path, 'r') as f:  # 100 reps 파일 로드
                  data = json.load(f)
              
              subsets = {}  # 50 reps subset 딕셔너리
              for prompt in data:  # 3개 프롬프트 루프 (Neutral, Value, Growth)
                  subsets[prompt] = {}
                  for rephrase in data[prompt]:  # 10개 리프레이즈 루프 (e.g., 'rephrase_1' ~ 'rephrase_10')
                      full_reps = data[prompt][rephrase]  # 100 reps 리스트
                      if len(full_reps) < 100:
                          print(f"Warning: {prompt}/{rephrase} has only {len(full_reps)} reps, expected 100")
                      subsets[prompt][rephrase] = full_reps[:50]  # 첫 50 reps subset (기존 데이터 재현)
              
              # subset 저장 (후속 분석용)
              subset_path = '/Users/jaehoon/Alphatross/70_Research/checkgpt/results/Rephrase/Rephrase_Repetition_Result_NVG_50_subset.json'
              with open(subset_path, 'w') as f:
                  json.dump(subsets, f)
              
              print(f"Subset created successfully. Keys: {list(subsets.keys())}")  # 실행 확인 출력
          ```
        - 예상 출력: "Subset created successfully. Keys: ['Neutral', 'Value', 'Growth']". 에러 시 경로 수정.
        - 시간: 30분 ( code_execution 호출 1회).

     2. **50 reps subset으로 pairwise Jaccard 계산 및 히스토그램 플롯 (각 프롬프트/리프레이즈별)**:
        - 이유: 리뷰어 지적 대응 위해 분포 시각화. pairwise( n=50, pairs=1225=50*49/2 )로 Jaccard 리스트 생성, 히스토그램으로 분포/멀티모달 확인. 각 프롬프트/리프레이즈별 별도 플롯( e.g., 3 프롬프트 x 10 리프레이즈 = 30개 플롯, but 평균 플롯으로 요약).
        - 도구 활용: code_execution으로 계산/플롯 생성( matplotlib 사용, 파일 저장). 멀티모달 체크: 히스토그램 피크 수 세기( e.g., scipy.find_peaks ).
        - 코드 예시(확장: Jaccard 함수 정의, 히스토그램 플롯, 멀티모달 체크 추가):
          ```python
          import json
          import numpy as np
          import matplotlib.pyplot as plt
          from scipy.signal import find_peaks  # 멀티모달 체크용
          from itertools import combinations  # pairwise 조합

          # Jaccard 함수 정의 (set 기반, stock lists 가정)
          def jaccard(set1, set2):
              intersection = len(set1.intersection(set2))
              union = len(set1.union(set2))
              return intersection / union if union != 0 else 0

          # subset 파일 로드
          with open('/Users/jaehoon/Alphatross/70_Research/checkgpt/results/Rephrase/Rephrase_Repetition_Result_NVG_50_subset.json', 'r') as f:
              subsets = json.load(f)

          results = {}  # {prompt: {rephrase: {'jaccards': [list], 'hist_fig': path, 'modes': int}}}
          for prompt in subsets:  # 3개 프롬프트
              results[prompt] = {}
              for rephrase in subsets[prompt]:  # 10개 리프레이즈
                  reps_lists = subsets[prompt][rephrase]  # 50 reps (각 reps는 stock 문자열 리스트 e.g., ['AAPL', 'MSFT', ...])
                  # pairwise Jaccard 계산 (combinations으로 1225 pairs)
                  jaccards = [jaccard(set(reps_lists[i]), set(reps_lists[j])) for i, j in combinations(range(50), 2)]
                  mean_jac = np.mean(jaccards)
                  sd_jac = np.std(jaccards)
                  
                  # 히스토그램 플롯 (bins=20, 멀티모달 체크)
                  fig, ax = plt.subplots()
                  counts, bins, _ = ax.hist(jaccards, bins=20, density=True, alpha=0.6, color='b')
                  ax.set_title(f'{prompt} - {rephrase} Jaccard Distribution (Mean: {mean_jac:.3f}, SD: {sd_jac:.3f})')
                  ax.set_xlabel('Jaccard Similarity')
                  ax.set_ylabel('Density')
                  hist_path = f'/Users/jaehoon/Alphatross/70_Research/checkgpt/results/hist_{prompt}_{rephrase}.png'
                  plt.savefig(hist_path)
                  plt.close()
                  
                  # 멀티모달 체크 (피크 수: height threshold 0.01로)
                  peaks, _ = find_peaks(counts, height=0.01)
                  num_modes = len(peaks)
                  
                  results[prompt][rephrase] = {'mean': mean_jac, 'sd': sd_jac, 'hist_path': hist_path, 'num_modes': num_modes}
          
          # 결과 요약 출력 (논문용)
          print(results)  # e.g., {'Neutral': {'rephrase_1': {'mean': 0.687, 'sd': 0.098, 'num_modes': 1}, ...}}
          ```
        - 예상 출력: results 딕셔너리( mean/SD, modes 수). modes=1이면 unimodal(문제없음), >1이면 멀티모달(추가 해석 필요). PNG 파일 30개 생성(요약 평균 플롯으로 논문 사용).
        - 에러 핸들링: 리스트가 빈 경우 skip, set 변환으로 중복 제거.
        - 시간: 1시간 ( code_execution 호출 1회, 플롯 생성).

     3. **100 reps 전체로 비교 (mean/SD 변화 체크, <5%면 footnote)**:
        - 이유: robustness 확인(50 vs 100 변화 최소화). 전체 파일 사용, but 계산만(플롯 생략).
        - 도구 활용: code_execution으로 위 코드 수정( subsets -> data, range(100), combinations(range(100),2) ).
        - 코드 예시(간략): 위 코드 복사, data 로드 후 jaccards 계산, print(f"Delta mean: {mean_100 - mean_50:.3f}").
        - 예상 출력: 변화 <5%( e.g., delta <0.05), footnote로 "100 reps confirms similar distribution".
        - 시간: 30분.

     4. **top 10/20/30 영향 없음 (분포 분석에 불필요)**:
        - 이유: 이 쟁점은 raw Jaccard 분포 중심, top 대표 포트폴리오는 쟁점 6으로 별도. 여기서는 무시(분포 계산 시 full lists 사용).
        - 작업: N/A, but 확인 시 top 크기 영향 테스트( e.g., top 30으로 Jaccard 재계산, but 최소화로 skip).
   - **예상 결과**: 대부분 unimodal( modes=1 ), 50 vs 100 유사( delta <5% ). 멀티모달 시 "분포에서 여러 피크 관찰, 베리에이션 원인 분석" 추가 파인딩.
   - **난이도/시간**: 중간, 총 2-3시간 ( code_execution 2-3회 호출).
   - **논문 수정**: Results > Repetition Variability에 new Figure( 평균 히스토그램, 50 reps 기반, 각 프롬프트/리프레이즈 요약 테이블). Footnote: "100 reps 비교: 변화 <5%, no multimodality". Table 추가( e.g., Table X: Per Prompt/Rephrase Mean/SD/Modes ).

(이하 쟁점 2~8은 이전 버전과 동일, 변경 없음. 전체 To-Do List 유지.)