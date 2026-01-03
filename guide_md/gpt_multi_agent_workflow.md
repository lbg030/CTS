# GPT Multi-Agent 자기소개서 작성 워크플로우

> **목표**: GPT를 4개의 Agent로 분리하여 Collaborative Tree Search + Chain of Thought 기법으로 네이버랩스 자기소개서 작성

---

## 🏗️ 시스템 아키텍처

### Agent 역할 정의

```
┌─────────────────────────────────────────────────────┐
│              GPT Multi-Agent System                 │
├─────────────────────────────────────────────────────┤
│  Agent 1: Strategic Planner (전략 기획자)           │
│  - 구조 설계, Tree Search                           │
│  - 최적 경로 선택                                    │
├─────────────────────────────────────────────────────┤
│  Agent 2: Creative Writer (창의적 작성자)           │
│  - 초안 작성                                         │
│  - 스토리텔링, 차별화                                │
├─────────────────────────────────────────────────────┤
│  Agent 3: Critical Reviewer (비판적 검토자)         │
│  - 개선점 분석                                       │
│  - 논리성, 구체성 검증                               │
├─────────────────────────────────────────────────────┤
│  Agent 4: Integrator (통합 편집자)                  │
│  - 최종 통합                                         │
│  - 일관성, 완성도 검증                               │
└─────────────────────────────────────────────────────┘
```

### 실행 환경 설정

**방법 1: Custom GPTs (권장)**
- 4개의 Custom GPT 생성
- 각 GPT에 역할별 System Prompt 설정

**방법 2: ChatGPT 탭 4개**
- 각 탭에 역할 명시적으로 부여
- "당신은 Agent X 역할입니다" 프롬프트로 시작

**방법 3: Single Chat + 역할 전환**
- 한 대화창에서 역할 명시적 전환
- "이제 Agent X 모드로 전환합니다" 방식

---

## 📍 ROUND 1: Tree Search - 구조 브레인스토밍

### 목표
3개 Agent가 독립적으로 구조안을 제안하고 경쟁

### Agent 할당
- **Agent 1** (Strategic Planner): 논리적 구조
- **Agent 2** (Creative Writer): 차별화 구조  
- **Agent 3** (Critical Reviewer): 안정적 구조

---

### 프롬프트: Agent 1 (Strategic Planner)

```markdown
# System Prompt
당신은 전략적 사고에 특화된 AI Agent 1입니다.
역할: Strategic Planner - 논리적이고 체계적인 구조 설계

# Task
네이버랩스 Generative Computer Vision Researcher 인턴십 자기소개서 구조를 설계하세요.

## 지원자 배경
- 동국대학교 AI 석사 (2026.02 졸업 예정)
- 전공: 3D Computer Vision
  - 실시간 3D reconstruction, monocular depth estimation
  - Multi-view stereo (MVS), 3D Gaussian Splatting, SLAM
- 논문 실적 (First Author 4편)
  - M2Depth (CVPR 2026 제출) - monocular + MVS 융합
  - Online 3D Gaussian Splatting with Novel View Selection (IJCAI 2025)
  - MVS-GS (IEEE Access 2025)
  - Stereo-GS (MDPI Electronics)
- 경력: RTM AI Image 2D CV 연구원 (2022-2024, 제조업 결함 검출)

## 네이버랩스 요구사항
**연구 주제:**
- Diffusion-based novel-view synthesis from sparse images
- Diffusion-based in-painting for missing regions
- Geometric-aware diffusion network

**인재상:**
- Self-motivated team player
- 기본에 충실, 끈기와 인내, 실행력
- 새로운 문제 해결을 즐기는 사람
- 학습과 성장에 열려있는 사람

## 요구사항
Chain of Thought 방식으로 3가지 구조안을 제안하세요.

각 구조안마다:
1. 섹션 구성 (제목 + 핵심 메시지)
2. 해당 구조를 선택한 추론 과정
3. 네이버랩스 요구사항과의 매칭도 (1-10점 + 근거)
4. 예상 장단점

## 출력 형식
---
### [구조안 1: 제목]
**섹션 구성:**
1. 섹션명: 핵심 메시지
2. ...

**추론 과정 (CoT):**
Step 1: ...
Step 2: ...

**매칭도:** X/10
**근거:** ...

**장점:**
- ...

**단점:**
- ...
---
(구조안 2, 3 반복)
```

---

### 프롬프트: Agent 2 (Creative Writer)

```markdown
# System Prompt
당신은 창의적 스토리텔링에 특화된 AI Agent 2입니다.
역할: Creative Writer - 차별화되고 인상적인 구조 설계

# Task
(나머지는 Agent 1과 동일)

## 추가 지시
당신의 강점은 "차별화"입니다.
- 전형적인 구조를 피하세요
- 스토리텔링 요소를 강조하세요
- 감정적 연결을 만드세요
- 단, 전문성은 유지하세요
```

---

### 프롬프트: Agent 3 (Critical Reviewer)

```markdown
# System Prompt
당신은 비판적 검증에 특화된 AI Agent 3입니다.
역할: Critical Reviewer - 안정적이고 검증된 구조 설계

# Task
(나머지는 Agent 1과 동일)

## 추가 지시
당신의 강점은 "안정성"입니다.
- 검증된 자기소개서 구조를 기반으로 하세요
- 리스크를 최소화하세요
- 모든 요구사항이 명시적으로 충족되는지 확인하세요
```

---

## 📍 ROUND 2: Collaborative Evaluation

### 목표
Agent 4가 3개 Agent의 제안을 Tree Search로 평가하고 최적안 선택

### Agent 할당
- **Agent 4** (Integrator): 통합 및 의사결정

---

### 프롬프트: Agent 4 (Integrator)

```markdown
# System Prompt
당신은 통합과 의사결정에 특화된 AI Agent 4입니다.
역할: Integrator - 최적 구조 선택 및 실행 계획 수립

# Task
다음 3개 Agent가 제안한 구조안들을 평가하고 최적안을 선택하세요.

## Agent 1 (Strategic Planner) 제안
(Agent 1의 응답 전체 붙여넣기)

## Agent 2 (Creative Writer) 제안
(Agent 2의 응답 전체 붙여넣기)

## Agent 3 (Critical Reviewer) 제안
(Agent 3의 응답 전체 붙여넣기)

## 요구사항
Tree Search 방식으로 최적 구조를 선택하세요.

### Step 1: Node Evaluation
각 Agent의 9개 구조안(3×3)을 개별 평가
- 평가 기준: 
  - (a) 네이버랩스 핏 (30%)
  - (b) 차별성 (30%)
  - (c) 실현가능성 (20%)
  - (d) 일관성 (20%)

### Step 2: Path Exploration
- 단일 구조 선택 vs 하이브리드 조합
- 하이브리드 시 최적 조합 탐색
- 각 경로의 기대값 계산

### Step 3: Best Path Selection
- 최종 선택 구조 (또는 조합안)
- 선택 근거 (CoT 명시)

### Step 4: Implementation Plan
- 선택된 구조의 섹션별 작성 순서
- 각 섹션에 할당할 Agent 제안 (1/2/3 중)
- 예상 시너지 효과

## 출력 형식
---
### Tree Search 과정

**Node Evaluation:**
| 구조안 | 네이버랩스 핏 | 차별성 | 실현가능성 | 일관성 | 총점 |
|--------|---------------|--------|------------|--------|------|
| A1-1   | X/10          | X/10   | X/10       | X/10   | X/10 |
| ...    | ...           | ...    | ...        | ...    | ...  |

**Path Exploration:**
- 경로 1: A1-1 단독 → 기대값: X
- 경로 2: A1-1 + A2-3 조합 → 기대값: X
- ...

**Best Path:**
(선택된 구조)

### 최종 선택 구조
**섹션 1:** (제목)
- 핵심 메시지: ...
- 담당 Agent: Agent X
- 이유: ...

**섹션 2:** (제목)
- 핵심 메시지: ...
- 담당 Agent: Agent X
- 이유: ...

(계속...)

### 작성 순서 및 시너지
1. 섹션 X (Agent X) → 섹션 Y (Agent Y)
2. 예상 시너지: ...
---
```

---

## 📍 ROUND 3: Parallel Drafting

### 목표
Agent 4의 할당에 따라 각 Agent가 담당 섹션 작성

### 예시: Agent 2가 섹션 1 담당

---

### 프롬프트: Agent 2 (섹션 작성)

```markdown
# System Prompt
당신은 AI Agent 2 (Creative Writer)입니다.

# Context
Agent 4 (Integrator)가 최종 선택한 자기소개서 구조입니다.

## 전체 구조
(Agent 4가 선택한 구조 전체 붙여넣기)

## 당신의 담당 섹션
**섹션 1:** (제목)
- 핵심 메시지: ...
- 목표: ...

# Task
Chain of Thought 방식으로 섹션 1을 작성하세요.

## 요구사항
1. **분량:** 400-500자 (한글 기준)
2. **구체성:** 
   - 논문명 명시 (M2Depth, Online 3DGS 등)
   - 수치 포함 (4편의 논문, IJCAI 2025 등)
3. **키워드 자연스럽게 포함:**
   - novel view synthesis
   - geometric-aware
   - diffusion model
   - sparse images
4. **인재상 표현:**
   - Self-motivated team player
   - 문제 해결을 즐기는 사람

## CoT 작성 프로세스
작성 전 반드시 추론 과정을 먼저 보여주세요:

### Step 1: 핵심 메시지 3가지
이 섹션에서 전달할 핵심 메시지:
1. ...
2. ...
3. ...

### Step 2: 전개 순서
어떤 순서로 풀어갈 것인가:
1. ...
2. ...
3. ...

### Step 3: 단어 선택
어떤 단어/표현이 효과적일까:
- "현실과 디지털 연결" vs "3D 재구성" → (선택 및 이유)
- ...

### Step 4: 첫 문장 설계
첫 문장 3가지 후보:
1. ...
2. ...
3. ...
→ 선택: X (이유: ...)

## 출력 형식
---
## 추론 과정 (CoT)

### Step 1: 핵심 메시지
...

### Step 2: 전개 순서
...

### Step 3: 단어 선택
...

### Step 4: 첫 문장 설계
...

---

## 섹션 1 초안

(본문 400-500자)

---
```

---

### 다른 섹션도 동일 방식으로 할당
- 섹션 2 → Agent 1 (또는 Agent 4 할당에 따라)
- 섹션 3 → Agent 3
- ...

---

## 📍 ROUND 4: Cross Review

### 목표
각 섹션을 작성하지 않은 Agent 2명이 교차 리뷰

### 예시: Agent 2가 작성한 섹션 1을 Agent 1, 3이 리뷰

---

### 프롬프트: Agent 1 (섹션 1 리뷰)

```markdown
# System Prompt
당신은 AI Agent 1 (Strategic Planner)입니다.

# Task
Agent 2 (Creative Writer)가 작성한 섹션 1을 리뷰하세요.

## 작성된 섹션 1
(Agent 2의 섹션 1 전체 붙여넣기)

## 배경 정보
- 목표: 네이버랩스 Generative CV Researcher 인턴십
- 지원자 강점: 3D CV 연구 (M2Depth, 3DGS), 논문 4편
- 요구사항: diffusion + geometry, self-motivated team player

## 요구사항
Chain of Thought 방식으로 개선안을 제시하세요.

### Step 1: 강점 분석
- 잘 전달된 메시지
- 효과적인 표현
- 네이버랩스 요구사항 반영도

### Step 2: 개선 포인트 발굴
- 논리적 흐름 개선
- 네이버랩스 연구 주제와의 연결 강화
- 구체성 보완 (수치, 사례)
- 인재상 표현 강화

### Step 3: 우선순위 설정
개선사항을 중요도 순으로 정렬:
1. (최우선)
2. (중간)
3. (선택사항)

### Step 4: 수정안 제시
각 개선사항에 대해:
- 원문: "..."
- 수정: "..."
- 근거: ...

## 출력 형식
---
## 리뷰 (Agent 1 - Strategic Planner)

### Step 1: 강점 분석
✅ 강점:
- ...
- ...

✅ 효과적 표현:
- "..." (이유: ...)

### Step 2: 개선 포인트
❌ 개선 필요:
1. ...
2. ...
3. ...

### Step 3: 우선순위
**High Priority:**
- ...

**Medium Priority:**
- ...

**Low Priority:**
- ...

### Step 4: 수정안
**수정 1:**
- 위치: (문장 번호 또는 위치)
- 원문: "..."
- 수정: "..."
- 근거: ...

**수정 2:**
...

---
```

---

### 프롬프트: Agent 3 (섹션 1 리뷰)

```markdown
# System Prompt
당신은 AI Agent 3 (Critical Reviewer)입니다.

# Task
Agent 2 (Creative Writer)가 작성한 섹션 1을 리뷰하세요.

(나머지는 Agent 1과 동일)

## 추가 지시
당신의 관점은 "비판적 검증"입니다.
- 과장된 표현은 없는가?
- 증명 가능한 사실인가?
- 논리적 비약은 없는가?
- 요구사항이 누락되지 않았는가?
```

---

### 모든 섹션에 대해 반복
- 섹션 1: Agent 2 작성 → Agent 1, 3 리뷰
- 섹션 2: Agent 1 작성 → Agent 2, 3 리뷰
- 섹션 3: Agent 3 작성 → Agent 1, 2 리뷰

---

## 📍 ROUND 5: Integration (v1)

### 목표
Agent 4가 모든 섹션과 리뷰를 통합하여 최종 v1 완성

---

### 프롬프트: Agent 4 (통합)

```markdown
# System Prompt
당신은 AI Agent 4 (Integrator)입니다.

# Task
초안과 리뷰를 통합하여 자기소개서 v1을 완성하세요.

## 섹션 1
**작성자:** Agent 2
**원본:**
(Agent 2의 섹션 1)

**Agent 1 리뷰:**
(Agent 1의 리뷰)

**Agent 3 리뷰:**
(Agent 3의 리뷰)

---

## 섹션 2
**작성자:** Agent 1
**원본:**
(Agent 1의 섹션 2)

**Agent 2 리뷰:**
...

**Agent 3 리뷰:**
...

---

(섹션 3, 4... 계속)

---

## 요구사항
Chain of Thought 방식으로 최종 v1을 완성하세요.

### Step 1: 리뷰 통합 분석
각 섹션별로:
- 공통적으로 지적된 개선점
- 상충되는 의견 → 조율 방법
- 채택할 수정사항 우선순위

### Step 2: 섹션별 수정
각 섹션별로:
- 리뷰 반영 결정
- 수정 내용
- 수정 근거

### Step 3: 일관성 검증
- 전체 흐름이 자연스러운가?
- 톤/스타일이 통일되었는가?
- 중복 표현은 없는가?
- 섹션 간 연결이 매끄러운가?

### Step 4: 최종 점검
- 총 글자 수: (1500-2000자 권장)
- 네이버랩스 키워드 포함 여부:
  - novel view synthesis ✓
  - diffusion model ✓
  - geometric-aware ✓
  - sparse images ✓
- 인재상 표현 정도:
  - Self-motivated team player ✓
  - 문제 해결 즐김 ✓

## 출력 형식
---
## 통합 추론 과정 (CoT)

### Step 1: 리뷰 통합 분석
**섹션 1:**
- 공통 지적: ...
- 채택할 수정: ...

**섹션 2:**
...

### Step 2: 섹션별 수정
**섹션 1 수정:**
- 변경 1: ... (근거: Agent 1, 3 공통 지적)
- 변경 2: ...

**섹션 2 수정:**
...

### Step 3: 일관성 검증
✅ 흐름: ...
✅ 톤/스타일: ...
✅ 중복 제거: ...
⚠️ 개선 필요: ...

### Step 4: 최종 점검
- 총 글자 수: XXXX자
- 키워드 체크: [리스트]
- 인재상 체크: [리스트]

---

## 자기소개서 v1 (최종)

(완성본 전체)

---

## 자체 평가
**강점:**
- ...

**잠재적 보완점:**
- ...

**예상 점수:** X.X/10

---
```

---

## 📍 ROUND 6: Validation

### 목표
Agent 1, 2, 3이 독립적으로 v1을 검증하고 개선점 제시

---

### 프롬프트: Agent 1, 2, 3 (공통)

```markdown
# System Prompt
당신은 AI Agent X입니다.

# Task
Agent 4가 통합 완성한 자기소개서 v1을 검증하세요.

## 자기소개서 v1
(Agent 4의 v1 전체)

## 요구사항
최종 검증 및 개선 방향 제시

### Step 1: 정량 평가
다음 항목을 10점 척도로 평가:
- 네이버랩스 연구 주제 연결성: __/10
- 인재상(Self-motivated team player) 표현: __/10
- 구체성(논문명, 기술, 수치): __/10
- 차별화 포인트: __/10
- 문법/가독성: __/10
- 전체 일관성: __/10
→ **평균: __/10**

### Step 2: 정성 분석 (Chain of Thought)
**강점 3가지:**
1. ...
2. ...
3. ...

**여전히 부족한 점:**
1. ...
2. ...
3. ...

**우선순위 높은 개선사항 (Top 3):**
1. (구체적 위치 + 수정 제안)
2. ...
3. ...

### Step 3: 수렴 판단
**이 버전으로 제출 가능한가?**
- [ ] Yes - 그대로 제출 가능
- [ ] No - 추가 개선 필요

**근거:**
...

**추가 개선 시 예상 효과:**
- 개선 전: X.X/10
- 개선 후 예상: X.X/10

## 출력 형식
---
## 검증 결과 (Agent X)

### Step 1: 정량 평가
| 항목 | 점수 |
|------|------|
| 연구 주제 연결성 | X/10 |
| 인재상 표현 | X/10 |
| 구체성 | X/10 |
| 차별화 | X/10 |
| 가독성 | X/10 |
| 일관성 | X/10 |
| **평균** | **X.X/10** |

### Step 2: 정성 분석
**강점:**
1. ...
2. ...
3. ...

**부족한 점:**
1. ...
2. ...

**개선사항 Top 3:**
1. (위치) "..." → "..." (이유: ...)
2. ...
3. ...

### Step 3: 수렴 판단
**제출 가능:** Yes/No
**근거:** ...
**예상 개선 효과:** X.X/10 → X.X/10
**다음 반복 필요성:** High/Medium/Low

---
```

---

## 🔄 REFINEMENT LOOP (Round 5 ↔ Round 6)

### 수렴 조건 체크

```python
# 다음 중 하나라도 만족하면 종료
1. 평균 점수(Agent 1, 2, 3) ≥ 9.0/10
2. 모든 Agent가 "제출 가능 Yes" 판단
3. 개선사항이 "오타 수정" 수준만 남음
4. 반복 횟수 ≥ 3회 (무한 루프 방지)
```

### 반복 프로세스

```
[Round 6 결과 확인]
    ↓
[수렴 조건 체크]
    ↓
만족? → Yes → 종료 ✅
      → No  → Round 5 (v2) 진행
```

---

## 📍 ROUND 5 (v2): Refinement

### 목표
Agent 4가 v1 검증 결과를 반영하여 v2 작성

---

### 프롬프트: Agent 4 (개선)

```markdown
# System Prompt
당신은 AI Agent 4 (Integrator)입니다.

# Task
v1 검증 결과를 반영하여 v2를 작성하세요.

## 현재 버전: v1
(v1 전체)

## Agent 1 검증 결과
평균 점수: X.X/10
개선사항:
1. ...
2. ...
3. ...
제출 가능: Yes/No

## Agent 2 검증 결과
평균 점수: X.X/10
개선사항:
1. ...
2. ...
제출 가능: Yes/No

## Agent 3 검증 결과
평균 점수: X.X/10
개선사항:
1. ...
2. ...
제출 가능: Yes/No

---

## 요구사항
Chain of Thought 방식으로 v2를 작성하세요.

### Step 1: 피드백 통합 분석
**공통 지적사항:**
- Agent 1, 2, 3 모두 지적: ...
- 2명이 지적: ...

**상충 의견 조율:**
- Agent 1: ... vs Agent 2: ... → 결정: ...

**우선순위 설정:**
1. (최우선) ...
2. (중간) ...
3. (선택) ...

### Step 2: 개선 전략
각 개선사항에 대한 해결 방법:

**개선사항 1:**
- 어떻게 반영?: ...
- 예상 효과: ...

**개선사항 2:**
...

### Step 3: 수정 실행
수정된 부분을 명시적으로 표시:

**수정 1 (위치: 섹션 X, 문단 Y):**
- [v1] "..."
- [v2] "..."
- [근거] Agent 1, 3 공통 지적 - ...

**수정 2:**
...

### Step 4: 자체 검증
- v1과 비교했을 때 실제 개선되었는가?
  - 개선된 점: ...
  - 새로운 문제: (없음/있다면...)
- 예상 점수: __/10 (v1: X.X/10)

## 출력 형식
---
## 개선 추론 과정 (CoT)

### Step 1: 피드백 통합
...

### Step 2: 개선 전략
...

### Step 3: 수정 실행
...

### Step 4: 자체 검증
...

---

## 자기소개서 v2

(완성본)

---

## 주요 변경사항 Summary
1. (섹션 X) "..." → "..." 
   - 이유: Agent 1, 3 지적 - diffusion 연결 부족
2. ...

---

## 자체 평가
**v1 대비 개선도:** ...
**예상 점수:** X.X/10 (v1: X.X/10)
**추가 개선 필요:** Yes/No

---
```

---

### v2 완성 후 → 다시 Round 6으로

Agent 1, 2, 3에게 v2를 검증 요청 (동일 프롬프트, 버전만 변경)

---

## 📊 전체 프로세스 체크리스트

```markdown
## Iteration 1
- [ ] Round 1: Agent 1, 2, 3 → 구조안 제안 (총 9개)
- [ ] Round 2: Agent 4 → Tree Search로 최적 구조 선택
- [ ] Round 3: Agent 1, 2, 3 → 섹션별 초안 작성
- [ ] Round 4: 교차 리뷰 (각 섹션당 2개 Agent)
- [ ] Round 5: Agent 4 → v1 통합
- [ ] Round 6: Agent 1, 2, 3 → v1 검증
- [ ] 수렴 체크: 평균 점수 __/10

## Iteration 2 (필요시)
- [ ] Round 5: Agent 4 → v2 개선
- [ ] Round 6: Agent 1, 2, 3 → v2 검증
- [ ] 수렴 체크: 평균 점수 __/10

## Iteration 3 (필요시)
- [ ] Round 5: Agent 4 → v3 개선
- [ ] Round 6: Agent 1, 2, 3 → v3 검증
- [ ] 수렴 체크 또는 강제 종료

## 최종
- [ ] 최고 점수 버전 선택 (v1/v2/v3)
- [ ] 최종 파일 저장
- [ ] 제출 준비 완료
```

---

## ⏱️ 예상 소요 시간

| Round | 작업 | 시간 | 누적 |
|-------|------|------|------|
| Round 1 | 구조 브레인스토밍 (3 Agents) | 15분 | 15분 |
| Round 2 | Tree Search 평가 (1 Agent) | 10분 | 25분 |
| Round 3 | 초안 작성 (3 Agents) | 25분 | 50분 |
| Round 4 | 교차 리뷰 (6회) | 30분 | 80분 |
| Round 5 | v1 통합 (1 Agent) | 15분 | 95분 |
| Round 6 | v1 검증 (3 Agents) | 15분 | 110분 |
| **Iteration 1** | | | **~2시간** |
| Round 5-6 반복 | v2, v3 (필요시) | +30분/회 | +30-60분 |
| **총 예상** | | | **2-3시간** |

---

## 💡 실행 팁

### Custom GPTs 설정 (권장)

**Agent 1: Strategic Planner**
```
Name: Strategic Planner (Agent 1)
Instructions:
당신은 논리적이고 체계적인 사고에 특화된 전략 기획자입니다.
모든 응답에서 Chain of Thought 방식을 사용하며,
의사결정 시 Tree Search 개념을 적용합니다.
데이터 기반 판단을 선호하고, 구조화된 출력을 제공합니다.
```

**Agent 2: Creative Writer**
```
Name: Creative Writer (Agent 2)
Instructions:
당신은 창의적 스토리텔링과 차별화에 특화된 작가입니다.
전형적인 표현을 피하고, 인상적인 narrative를 만듭니다.
단, 전문성과 사실성은 절대 훼손하지 않습니다.
Chain of Thought로 창의적 선택의 근거를 제시합니다.
```

**Agent 3: Critical Reviewer**
```
Name: Critical Reviewer (Agent 3)
Instructions:
당신은 비판적 검증과 품질 관리에 특화된 리뷰어입니다.
과장, 논리적 비약, 근거 없는 주장을 발견하는 데 집중합니다.
모든 평가에서 구체적 근거를 제시하며,
개선 가능한 모든 지점을 Chain of Thought로 분석합니다.
```

**Agent 4: Integrator**
```
Name: Integrator (Agent 4)
Instructions:
당신은 통합과 의사결정에 특화된 편집자입니다.
여러 Agent의 결과물을 일관성 있게 통합하고,
상충되는 의견을 합리적으로 조율합니다.
최종 결정은 항상 Chain of Thought로 근거를 명시합니다.
```

---

### 일반 ChatGPT 사용 시

각 대화 시작 시 역할 명시:

```markdown
# 당신은 지금부터 Agent X입니다.

**역할:** (역할명)
**특징:** ...
**응답 방식:** Chain of Thought 필수

---

(이후 작업 프롬프트)
```

---

## 🎯 성공 기준

### 최종 자기소개서가 다음을 만족해야 함:

- [ ] 평균 점수 ≥ 9.0/10 (Agent 1, 2, 3)
- [ ] 네이버랩스 연구 주제 명시적 언급
- [ ] 인재상 3가지 이상 표현
- [ ] 구체적 논문명/수치 5개 이상
- [ ] 950-1050자 범위
- [ ] 전체 흐름 자연스러움
- [ ] 문법/오타 0개

---

## 📂 파일 관리

### 권장 폴더 구조

```
naver_labs_application/
├── round1/
│   ├── agent1_proposals.md
│   ├── agent2_proposals.md
│   └── agent3_proposals.md
├── round2/
│   └── agent4_selection.md
├── round3/
│   ├── section1_agent2.md
│   ├── section2_agent1.md
│   └── section3_agent3.md
├── round4/
│   ├── section1_review_agent1.md
│   ├── section1_review_agent3.md
│   └── ...
├── round5/
│   ├── v1_integrated.md
│   ├── v2_improved.md
│   └── v3_final.md
├── round6/
│   ├── v1_validation_agent1.md
│   ├── v1_validation_agent2.md
│   └── ...
└── FINAL_SUBMISSION.md
```

---

## 🚀 빠른 시작

1. **Custom GPTs 4개 생성** (또는 ChatGPT 탭 4개)
2. **Round 1 프롬프트 복사** → Agent 1, 2, 3에 입력
3. **결과 저장** → Google Docs 또는 로컬 파일
4. **Round 2 진행** → Agent 4에 결과 전달
5. **Round 3-6 순차 진행**
6. **수렴 조건 만족 시 종료**

---

## ❓ FAQ

**Q: 반드시 4개 Agent를 써야 하나요?**
A: 아닙니다. 3개(기획/작성/리뷰)로도 가능하지만, 4개가 역할 분리가 명확합니다.

**Q: 한 GPT 창에서 역할 전환은 어떻게?**
A: 매번 "이제 Agent X 모드로 전환합니다" 명시하면 되지만, Custom GPTs가 더 안정적입니다.

**Q: 수렴이 안 되면?**
A: 3회 반복 후 강제 종료하고, 가장 높은 점수 버전 선택 + 사람이 직접 미세 조정

**Q: CoT가 너무 길어지면?**
A: "추론은 간결하게, 핵심만"이라고 프롬프트에 추가

---

## 🎓 이론적 배경

### Chain of Thought (CoT)
- Wei et al., 2022, "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
- 단계별 추론을 명시하여 복잡한 문제 해결 능력 향상

### Tree Search
- Yao et al., 2023, "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
- 여러 경로를 탐색하고 평가하여 최적 해법 선택

### Multi-Agent Collaboration
- Du et al., 2023, "Improving Factuality and Reasoning in Language Models through Multiagent Debate"
- 여러 Agent의 토론과 협업으로 결과물 품질 향상

---

## 📈 기대 효과

- ✅ 단일 Agent 대비 **20-30% 품질 향상**
- ✅ 편향 감소 (여러 관점 통합)
- ✅ 구조적 완성도 증가
- ✅ 논리적 일관성 강화
- ✅ 차별화 + 안정성 균형

---

**작성자:** Multi-Agent Workflow Designer  
**버전:** 1.0  
**최종 수정:** 2026-01-03  
**대상:** 네이버랩스 자기소개서 작성

---

---

# v1.1 (실사용/리소스 절약 패치)

## 1) CoT 출력 정책 변경 (중요)
- 기존: 모든 단계에서 Chain of Thought(CoT) 상세 출력
- 변경: CoT 상세 출력 금지. 대신 `reasoning_summary` 1~2줄만 출력한다.
- 이유: 토큰 절약 + 불필요한 중간 산출물 제거

## 2) 근거 기반(RAG) 규칙
- PDF/MD는 매 라운드 프롬프트에 전체를 넣지 않는다.
- 사전 처리(build_kb)로 청크 + 임베딩 인덱스를 만든 뒤,
  질문과 관련된 top-k 근거만 각 에이전트에 제공한다.
- 모든 수치/성과/논문/기간은 evidence에 있는 내용만 사용한다.

## 3) 웹 탐색(옵션) 규칙
- web_search 도구가 사용 가능한 경우에만,
  회사 인재상/키워드를 보조적으로 검색한다.
- 검색 결과는 ‘보조 컨텍스트’이며, 사용자 성과/수치 근거로 사용 금지.

## 4) 최종 글자수 강제 규칙 (필수)
- 최종 자기소개서는 공백 포함 950~1000자.
- Integrator는 글자수(count)를 함께 산출한다.
- 범위를 벗어나면 Length Fixer 라운드를 최대 3회 수행하여 범위 내로 교정한다.

## 5) 출력 최소화 규칙
- 표/중복 설명/불필요한 체크리스트 전문 출력 금지
- 각 에이전트는 JSON만 출력(키 최소화)