# CTS 자기소개서 생성 시스템 - 근본적 재설계 방안

> **작성일**: 2026-01-04
> **목적**: 품질 정체, 출력 차단, 에이전트 상호작용 불명확성 문제 해결

---

## 목차

1. [현재 시스템의 구조적 문제점](#1-현재-시스템의-구조적-문제점)
2. [재설계 핵심 원칙](#2-재설계-핵심-원칙)
3. [에이전트 간 상호작용 스키마](#3-에이전트-간-상호작용-스키마)
4. [점수 정체 해결: COT 기반 Refine 전략](#4-점수-정체-해결-cot-기반-refine-전략)
5. [본문 출력 로직 개선](#5-본문-출력-로직-개선)
6. [프롬프트 재설계](#6-프롬프트-재설계)
7. [구현 가이드](#7-구현-가이드)

---

## 1. 현재 시스템의 구조적 문제점

### 1.1 본문 출력 차단 문제 ❌

**위치**: `run_sop.py:1477-1480`

```python
# ❌ 현재 구조
allow_body_output = True
if score_result and not score_result.passed:
    allow_body_output = False
    md.append("품질 기준 미달로 본문을 출력하지 않습니다.\n")
```

**문제**:
- 9.5점 미만 시 본문 완전 차단
- 사용자는 왜 실패했는지 확인 불가
- 디버깅과 개선 검증 불가능

**파급 효과**:
- Refine 루프가 실행되어도 최종 결과를 볼 수 없음
- 점수 낮은 이유를 파악할 수단 없음
- "블랙박스" 시스템으로 전락

---

### 1.2 품질 점수 정체 문제 ❌

**위치**: `run_sop.py:555-595` (RefineLoop._apply_plan)

```python
# ❌ 현재 개선 프롬프트
prompt = f"""다음 자기소개서 본문을 개선하세요.

## 목표 모듈
{plan.get("module")} (전략: {plan.get("strategy_desc")})

## 필수 규칙
6. 미세 수정 금지: 점수 상승이 가능한 방향으로 내용/구조를 명확히 변경
7. 전체를 새로 쓰지 말고 해당 모듈 관련 문장만 집중 수정
"""
```

**문제**:
1. **추상적 지시**: "점수 상승이 가능한 방향으로" → LLM이 이걸 어떻게 구체화?
2. **점수-개선 단절**: 현재 7.5점인데 왜 낮은지, 어떻게 하면 9.5가 될지 명시 없음
3. **전략 효과 불명확**: `MODULE_STRATEGIES`는 정의되어 있지만 각 전략이 점수에 미치는 영향 분석 없음
4. **실패 원인 미분석**: 2회 연속 개선 실패 시 전략만 바꾸고 왜 실패했는지 분석 안 함

**근본 원인**:
- **COT(Chain-of-Thought) 미활용**: "왜 이 수정이 점수를 올릴 것인가" 추론 단계 부재
- **피드백 루프 부실**: Scorer의 rationale이 Refiner에게 제대로 전달 안 됨

---

### 1.3 에이전트 간 상호작용 문제 ❌

**위치**: `run_sop.py:1670-1835` (전체 파이프라인)

```python
# ❌ 현재 데이터 전달 방식
# Planner → CTS
candidates = [
    {"id": "strategic", "plan": p1},  # p1은 그냥 dict
    {"id": "creative", "plan": p2},
]

# Writer → Reviewer
reviewer = call_agent_json(client, reviewer_model, prompts["reviewer"],
    {"draft_text": writer.get("draft_text", ""), ...})  # 텍스트만 전달
```

**문제**:
1. **암묵적 컨텍스트**: 각 에이전트가 이전 단계의 "의도"를 명시적으로 전달받지 못함
2. **스키마 부재**: 입출력 형식이 프롬프트 문자열에만 의존 (타입 안전성 없음)
3. **CTS 활용 부족**: "Collaborative Tree Search"를 표방하지만 실제로는 단순 점수 비교만
4. **피드백 구조화 부족**: Reviewer 피드백이 "왜 수정이 필요한지" 구조적 근거 부족

**근본 원인**:
- 에이전트 간 **정보 전달이 느슨한 key-value 쌍**에 의존
- 각 에이전트가 받아야 할 **필수 컨텍스트**가 명시되지 않음

---

### 1.4 프롬프트 설계 문제 ❌

**위치**: `run_sop.py:314-342` (QualityScorer._build_scorer_prompt)

```python
# ❌ 현재 Scorer 프롬프트
prompt = """역할: Quality Scorer

평가 기준 (0~10):
- question_focus: 질문 의도에 직접 답하고 불필요한 내용이 없는가
- specificity: 구체적 행동/상황/결과로 설명되어 추상적 표현이 적은가

규칙:
- 점수는 엄격하게 부여하고, 9.5 이상은 매우 뛰어난 경우에만 부여한다.
"""
```

**문제**:
- **점수 상승 전략 부재**: "엄격하게"만 강조, "9.5 도달 방법" 가이드 없음
- **평가 기준 모호**: "자연스러운가", "명확한가" 같은 주관적 기준
- **COT 미활용**: 평가 과정에서 단계별 사고 요구 안 함

---

## 2. 재설계 핵심 원칙

### 원칙 1: 본문은 항상 출력 + 개선 유도 ✅

```
9.5점 미만이어도 반드시 본문 출력
+ "왜 9.5에 미달했는지" 명시적 안내
+ "어떻게 개선해야 하는지" 구체적 지침
```

**목표**: 출력 차단이 아니라 **출력 + 개선 유도**

---

### 원칙 2: 스키마 기반 에이전트 통신 ✅

```python
# ✅ 각 에이전트 입출력을 TypedDict/Dataclass로 정의
@dataclass
class PlannerOutput:
    outline: List[str]
    core_messages: List[str]
    personality_traits: List[str]
    experience_to_use: List[str]
    reasoning: str  # COT 추론 과정
    expected_strengths: List[str]  # 이 플랜의 강점
    expected_weaknesses: List[str]  # 이 플랜의 약점
```

**목표**: 타입 안전성 + 명시적 컨텍스트 전달

---

### 원칙 3: COT + CTS 명시적 활용 ✅

```python
# ✅ Refine 시 COT 추론 단계 추가
class RefineStrategy:
    def analyze_why_low_score(self, text: str, score: float, module: str) -> str:
        """왜 이 모듈 점수가 낮은지 COT로 분석"""

    def propose_change(self, analysis: str) -> str:
        """분석 결과 기반으로 구체적 변경 제안"""

    def predict_score_impact(self, change: str) -> float:
        """이 변경이 점수에 미칠 영향 예측"""
```

**목표**: "왜 이 수정이 점수를 올릴 것인가" 추론

---

### 원칙 4: 점수 상승 지향 프롬프트 ✅

```python
# ✅ 개선 프롬프트 예시
prompt = f"""현재 {module} 점수: {current_score}/10
목표 점수: 9.5/10
점수가 낮은 이유: {rationale}

점수를 9.5로 올리기 위한 구체적 변경:
1. {specific_change_1}
2. {specific_change_2}

변경 후 예상 점수: {predicted_score}
"""
```

**목표**: "X를 Y로 바꾸면 점수가 Z만큼 오른다" 명시

---

## 3. 에이전트 간 상호작용 스키마

### 3.1 스키마 설계 원칙

각 에이전트는 다음을 명시적으로 전달받아야 합니다:

1. **현재 단계의 목적** (purpose)
2. **이전 단계 결과 요약** (previous_context)
3. **수정이 필요한 구체적 지점** (modification_targets)
4. **절대 변경 불가 제약** (immutable_constraints)

---

### 3.2 에이전트별 입출력 스키마

#### A. Planner

**입력 스키마**:
```python
@dataclass
class PlannerInput:
    question: str
    question_type: str  # 자기소개, 지원동기 등
    company_profile: Dict
    evidence: str  # RAG 검색 결과
    constraints: Dict  # 글자수, 문체 등
    purpose: str = "질문에 답하는 자기소개서 구조 설계"
```

**출력 스키마**:
```python
@dataclass
class PlannerOutput:
    # 핵심 계획
    outline: List[str]  # ["도입: 성향 제시", "본론: 경험 1~2개", ...]
    core_messages: List[str]  # ["문제 해결 집요함", "팀 협업 중시"]
    personality_traits: List[str]  # ["끈기", "호기심"]
    experience_to_use: List[str]  # ["CVPR 논문 경험", "인턴 프로젝트"]

    # COT 추론
    reasoning_summary: str  # "자기소개 질문이므로 성향 중심 구조 선택"
    expected_strengths: List[str]  # ["질문 초점 명확", "흐름 자연스러움"]
    expected_weaknesses: List[str]  # ["구체성 부족 가능성"]

    # 개선 방향
    must_avoid: List[str]  # ["논문 나열", "추상적 표현"]

    # 메타 정보
    planner_id: str  # "strategic", "creative", "critical"
    confidence: float  # 0.0~1.0
```

---

#### B. CTS Scorer

**입력 스키마**:
```python
@dataclass
class CTSScorerInput:
    candidates: List[PlannerOutput]  # 3개 플랜
    question: str
    company_profile: Dict
    evidence: str
    constraints: Dict
    purpose: str = "3개 플랜 중 최적 플랜 선택"
```

**출력 스키마**:
```python
@dataclass
class CTSScorerOutput:
    # 각 후보 평가
    scores: List[Dict]  # [{"id": "strategic", "question_focus": 8, ...}, ...]

    # COT 분석
    comparison_reasoning: str  # "strategic은 초점 명확하나 창의성 부족, creative는..."
    best_id: str  # "creative"
    selection_rationale: str  # "질문 유형상 성향 중심 서술이 유리하므로 creative 선택"

    # 선택된 플랜
    selected_plan: PlannerOutput

    # 경고
    warnings: List[str]  # ["strategic 플랜에서 연구 보고서 느낌 우려"]
```

---

#### C. Writer

**입력 스키마**:
```python
@dataclass
class WriterInput:
    selected_plan: PlannerOutput  # CTS가 선택한 플랜
    question: str
    company_profile: Dict
    evidence: str
    constraints: Dict
    purpose: str = "선택된 플랜 기반으로 초안 작성"

    # 명시적 가이드
    focus_on: List[str]  # selected_plan.core_messages에서 추출
    must_include: List[str]  # selected_plan.experience_to_use
    must_avoid: List[str]  # selected_plan.must_avoid
```

**출력 스키마**:
```python
@dataclass
class WriterOutput:
    draft_text: str  # 작성된 초안

    # 자체 평가
    self_assessment: Dict  # {"question_focus": 8, "specificity": 7, ...}

    # 작성 과정 기록
    outline_followed: bool  # 플랜 구조를 따랐는지
    evidence_used: List[str]  # 사용한 근거 ID
    deviations: List[str]  # 플랜에서 벗어난 부분 (있다면)
    deviations_reason: str  # 왜 벗어났는지
```

---

#### D. Reviewer

**입력 스키마**:
```python
@dataclass
class ReviewerInput:
    draft_text: str
    writer_output: WriterOutput  # Writer의 전체 출력
    selected_plan: PlannerOutput  # 원래 계획
    question: str
    company_profile: Dict
    evidence: str
    constraints: Dict
    purpose: str = "초안의 문제점 진단 및 수정 방향 제시"
```

**출력 스키마**:
```python
@dataclass
class ReviewerOutput:
    # 문제 진단
    is_report_style: bool
    has_first_person: bool
    first_person_found: List[str]
    hallucination_risks: List[str]  # 근거 없는 주장

    # 수정 지침 (구조화)
    fixes: List[Fix]  # Fix는 별도 dataclass

    # COT 분석
    overall_assessment: str  # "전반적으로 플랜을 잘 따랐으나 specificity 부족"
    priority_fixes: List[str]  # 우선 수정 사항

@dataclass
class Fix:
    location: str  # "2번째 문단", "도입부"
    issue: str  # "추상적 표현"
    current_text: str  # 문제가 있는 현재 텍스트
    suggested_change: str  # 구체적 수정 제안
    rationale: str  # 왜 이 수정이 필요한지
    expected_score_impact: float  # 이 수정이 점수에 미칠 영향 (+0.5 등)
```

---

#### E. Integrator

**입력 스키마**:
```python
@dataclass
class IntegratorInput:
    draft_text: str
    writer_output: WriterOutput
    reviewer_output: ReviewerOutput
    selected_plan: PlannerOutput
    question: str
    company_profile: Dict
    constraints: Dict
    purpose: str = "Reviewer 피드백 반영하여 최종본 작성"

    # 명시적 수정 지침
    fixes_to_apply: List[Fix]  # Reviewer.fixes
    must_preserve: List[str]  # 유지해야 할 부분
```

**출력 스키마**:
```python
@dataclass
class IntegratorOutput:
    final_text: str

    # 수정 이력
    fixes_applied: List[str]  # 적용한 수정 사항
    fixes_skipped: List[str]  # 적용 안 한 수정 사항 (있다면)
    skipped_reason: str  # 왜 스킵했는지

    # 자체 평가
    self_score: Dict  # {"question_focus": 8.5, ...}
    improvements_made: List[str]  # "추상 표현 → 구체적 행동으로 변경"
```

---

#### F. QualityScorer

**입력 스키마**:
```python
@dataclass
class ScorerInput:
    text: str
    question: str
    question_type: str
    company_profile: Dict
    evidence: str
    purpose: str = "제출 적합성 평가"
```

**출력 스키마** (기존 ScoreResult 강화):
```python
@dataclass
class ScoreResult:
    total_score: float
    criteria_scores: Dict[str, float]
    passed: bool
    failed_criteria: List[str]

    # ✅ COT 추가
    rationales: Dict[str, str]  # 각 모듈별 점수 이유
    gap_to_target: float  # 9.5 - total_score

    # ✅ 구체적 개선 방향
    recommendations: List[Recommendation]

@dataclass
class Recommendation:
    module: str  # "question_focus"
    current_score: float  # 7.5
    target_score: float  # 9.5
    issue: str  # "질문과 무관한 내용 포함"
    specific_change: str  # "2문단의 '논문 3편 게재' 부분 삭제"
    expected_score_after: float  # 8.5
    priority: int  # 1=high, 2=medium, 3=low
```

---

#### G. Refiner (RefineLoop)

**입력 스키마**:
```python
@dataclass
class RefinerInput:
    text: str
    score_result: ScoreResult
    question: str
    question_type: str
    company_profile: Dict
    evidence: str
    constraints: Dict
    purpose: str = "점수 9.5 도달을 위한 반복 개선"

    # ✅ 명시적 개선 전략
    target_module: str  # "question_focus"
    recommendations: List[Recommendation]  # Scorer의 권고
```

**출력 스키마**:
```python
@dataclass
class RefinerOutput:
    improved_text: str

    # COT 추론
    change_reasoning: str  # "2문단 논문 나열이 질문 초점 흐리므로 삭제"
    changes_made: List[str]  # "2문단 3~5행 삭제", "도입부 1문장 추가"

    # 점수 예측
    predicted_score_change: float  # +1.0

    # 메타 정보
    strategy_used: str  # "trim_offtopic"
    iteration_num: int
```

---

### 3.3 스키마 활용 예시

**Before (현재)**:
```python
# ❌ 느슨한 dict 전달
reviewer = call_agent_json(client, model, prompt, {
    "draft_text": writer.get("draft_text", ""),
    "company_profile": company_profile,
})
```

**After (재설계)**:
```python
# ✅ 구조화된 스키마 전달
reviewer_input = ReviewerInput(
    draft_text=writer_output.draft_text,
    writer_output=writer_output,
    selected_plan=selected_plan,
    question=question,
    company_profile=company_profile,
    evidence=evidence,
    constraints=constraints,
    purpose="초안의 문제점 진단 및 수정 방향 제시"
)

reviewer_output = call_agent_with_schema(
    client=client,
    model=model,
    instructions=prompts["reviewer"],
    input_schema=reviewer_input,
    output_schema=ReviewerOutput,
)
```

**장점**:
- 타입 안전성
- 필수 필드 검증
- 다른 AI가 이 구조만 보고 재구현 가능

---

## 4. 점수 정체 해결: COT 기반 Refine 전략

### 4.1 현재 Refine 로직의 문제

```python
# ❌ 현재: 추상적 지시
prompt = f"""다음 자기소개서 본문을 개선하세요.
## 목표 모듈
{module} (전략: {strategy_desc})

## 필수 규칙
6. 미세 수정 금지: 점수 상승이 가능한 방향으로 내용/구조를 명확히 변경
```

**문제**:
- "점수 상승이 가능한 방향으로" ← 너무 추상적
- LLM은 "어떻게"를 모름

---

### 4.2 COT 기반 Refine 3단계 전략

#### 단계 1: WHY - 점수 낮은 이유 분석

```python
# ✅ Refine 전략 1단계: COT로 원인 분석
def analyze_low_score_with_cot(
    text: str,
    score_result: ScoreResult,
    target_module: str
) -> str:
    """왜 이 모듈 점수가 낮은지 COT로 분석"""

    current_score = score_result.criteria_scores[target_module]
    rationale = score_result.rationales.get(target_module, "")

    analysis_prompt = f"""다음 자기소개서의 '{target_module}' 모듈 점수가 {current_score}/10입니다.
목표는 9.5/10입니다.

현재 본문:
{text}

평가자 의견:
{rationale}

단계별로 분석하세요:

1. 점수가 {current_score}인 **구체적 이유** 3가지를 본문에서 찾아 인용하세요.
   예: "2문단의 '열정적으로 임했습니다'는 추상적 표현"

2. 각 이유가 점수에 미친 영향을 추정하세요.
   예: "추상적 표현 3회 사용 → -1.0점"

3. 9.5/10 도달을 위해 **반드시 변경해야 할 부분**을 우선순위로 나열하세요.

출력 JSON:
{{
    "reasons": [
        {{"issue": "추상적 표현", "location": "2문단 3행", "quote": "열정적으로", "score_impact": -0.5}},
        ...
    ],
    "priority_changes": [
        {{"rank": 1, "what": "2문단 '열정적으로' → 구체적 행동", "expected_gain": +0.5}},
        ...
    ],
    "gap_to_target": 2.0
}}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": analysis_prompt}]
    )

    return response.choices[0].message.content
```

**핵심**:
- "왜 낮은지" 구체적으로 본문에서 인용
- 각 이유의 점수 영향 정량화
- 우선순위 명확화

---

#### 단계 2: WHAT - 구체적 변경 제안

```python
# ✅ Refine 전략 2단계: 구체적 변경 제안
def propose_concrete_changes(
    text: str,
    analysis: Dict,
    target_module: str
) -> str:
    """분석 결과 기반으로 구체적 변경 제안"""

    priority_changes = analysis["priority_changes"]

    proposal_prompt = f"""다음 자기소개서를 '{target_module}' 점수 향상을 위해 수정합니다.

현재 본문:
{text}

필수 변경 사항 (우선순위순):
{json.dumps(priority_changes, ensure_ascii=False, indent=2)}

각 변경 사항에 대해:

1. 현재 텍스트를 정확히 인용하세요.
2. 변경 후 텍스트를 작성하세요.
3. 왜 이 변경이 점수를 올릴지 설명하세요.

출력 JSON:
{{
    "changes": [
        {{
            "rank": 1,
            "before": "열정적으로 임했습니다",
            "after": "매일 3시간씩 추가 실험을 진행했습니다",
            "rationale": "추상 표현을 구체적 행동으로 교체 → specificity +0.5",
            "expected_score_impact": +0.5
        }},
        ...
    ],
    "total_expected_gain": +1.5
}}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": proposal_prompt}]
    )

    return response.choices[0].message.content
```

**핵심**:
- Before/After 명시
- 각 변경의 점수 영향 예측
- 총 예상 상승폭 계산

---

#### 단계 3: HOW - 변경 적용 및 검증

```python
# ✅ Refine 전략 3단계: 변경 적용
def apply_changes(
    text: str,
    changes: List[Dict]
) -> str:
    """제안된 변경 사항을 본문에 적용"""

    apply_prompt = f"""다음 자기소개서에 변경 사항을 적용하세요.

현재 본문:
{text}

변경 사항:
{json.dumps(changes, ensure_ascii=False, indent=2)}

규칙:
1. 각 변경 사항의 'before' 텍스트를 정확히 찾아 'after'로 교체
2. 나머지 부분은 그대로 유지
3. 변경 후 문장 연결이 자연스러운지 확인
4. 절대로 제공되지 않은 수치/사실 추가 금지

출력: 수정된 전체 본문 (JSON 아니고 텍스트만)
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": apply_prompt}],
        max_tokens=1500
    )

    return response.choices[0].message.content.strip()
```

---

### 4.3 Refine 루프 전체 흐름 (재설계)

```python
class RefineLoopV2:
    """COT 기반 품질 개선 루프"""

    def refine(
        self,
        text: str,
        score_result: ScoreResult,
        question: str,
        company_profile: Dict,
        evidence: str,
        qtype: QuestionType,
        constraints: Dict
    ) -> Tuple[str, ScoreResult, List[RefineIteration]]:
        """
        3단계 COT 기반 개선:
        1. WHY: 점수 낮은 이유 분석
        2. WHAT: 구체적 변경 제안
        3. HOW: 변경 적용
        """

        if score_result.passed:
            return text, score_result, []

        current_text = text
        current_score = score_result
        iterations = []

        for i in range(self.max_total_iterations):
            # 개선 대상 모듈 선택
            target_module = self._pick_target_module(current_score)
            if not target_module:
                break

            # ✅ 단계 1: WHY - COT로 원인 분석
            self.logger.info(f"[Refine {i+1}] WHY: {target_module} 점수 낮은 이유 분석")
            analysis = self._analyze_low_score_with_cot(
                current_text, current_score, target_module
            )

            # ✅ 단계 2: WHAT - 구체적 변경 제안
            self.logger.info(f"[Refine {i+1}] WHAT: 구체적 변경 사항 제안")
            proposals = self._propose_concrete_changes(
                current_text, analysis, target_module
            )

            # ✅ 단계 3: HOW - 변경 적용
            self.logger.info(f"[Refine {i+1}] HOW: 변경 사항 적용")
            improved_text = self._apply_changes(
                current_text, proposals["changes"]
            )

            # 점수 재평가
            new_score = self.scorer.score(
                improved_text, question, company_profile, evidence, qtype
            )

            # 개선 검증
            actual_gain = new_score.total_score - current_score.total_score
            expected_gain = proposals.get("total_expected_gain", 0)

            self.logger.info(
                f"[Refine {i+1}] {current_score.total_score:.2f} → {new_score.total_score:.2f} "
                f"(예상: +{expected_gain:.2f}, 실제: {actual_gain:+.2f})"
            )

            # 이력 기록
            iteration = RefineIteration(
                iteration=i+1,
                module=target_module,
                module_score_before=current_score.criteria_scores.get(target_module, 0),
                module_score_after=new_score.criteria_scores.get(target_module, 0),
                score_before=current_score.total_score,
                score_after=new_score.total_score,
                improvements_made=[c["rationale"] for c in proposals["changes"]],
                strategy="cot_3step",
                diagnostics={"analysis": analysis, "proposals": proposals},
                text_before=current_text,
                text_after=improved_text
            )
            iterations.append(iteration)

            # 업데이트
            current_text = improved_text
            current_score = new_score

            # 통과 확인
            if current_score.passed:
                self.logger.info(f"[Refine] ✅ 품질 통과! ({current_score.total_score:.2f}/10)")
                break

            # 개선 없음 감지
            if actual_gain < 0.1:
                self.logger.warning(f"[Refine] 개선 폭 미미 ({actual_gain:+.2f}), 다음 모듈로")
                continue

        return current_text, current_score, iterations

    def _analyze_low_score_with_cot(self, text, score_result, module):
        """WHY: 점수 낮은 이유 COT 분석"""
        # (위 analyze_low_score_with_cot 함수 내용)
        pass

    def _propose_concrete_changes(self, text, analysis, module):
        """WHAT: 구체적 변경 제안"""
        # (위 propose_concrete_changes 함수 내용)
        pass

    def _apply_changes(self, text, changes):
        """HOW: 변경 적용"""
        # (위 apply_changes 함수 내용)
        pass
```

---

### 4.4 Before/After 비교

#### Before (현재)
```python
# ❌ 추상적 프롬프트
prompt = """다음 본문을 개선하세요.
목표 모듈: question_focus
전략: 질문 초점 강화

규칙:
- 점수 상승이 가능한 방향으로 수정
"""

# 결과: LLM이 "방향"을 추측 → 점수 정체
```

#### After (재설계)
```python
# ✅ COT 3단계 + 구체적 지침
# 1단계: WHY
analysis = {
    "reasons": [
        {"issue": "질문과 무관한 논문 나열", "location": "2문단", "score_impact": -1.0}
    ],
    "gap_to_target": 2.0
}

# 2단계: WHAT
proposals = {
    "changes": [
        {
            "before": "CVPR 2025에 3편의 논문을 게재했습니다.",
            "after": "(삭제)",
            "rationale": "질문은 '자신에 대해'인데 논문 나열은 무관 → question_focus +1.0",
            "expected_score_impact": +1.0
        }
    ],
    "total_expected_gain": +1.0
}

# 3단계: HOW
improved_text = apply_changes(text, proposals["changes"])

# 결과: 명확한 변경 → 점수 상승
```

---

## 5. 본문 출력 로직 개선

### 5.1 현재 문제

```python
# ❌ run_sop.py:1477-1480
allow_body_output = True
if score_result and not score_result.passed:
    allow_body_output = False
    md.append("품질 기준 미달로 본문을 출력하지 않습니다.\n")
```

**문제**: 9.5 미만 시 본문 완전 차단

---

### 5.2 재설계

```python
# ✅ 항상 본문 출력 + 개선 안내
def write_markdown_v2(
    out_path: str,
    company_name: str,
    role: str,
    question: str,
    qtype: QuestionType,
    final_text: str,
    score_result: Optional[ScoreResult],
    **kwargs
) -> None:

    md = []
    md.append("# 자기소개서 결과\n\n")
    # ... (기존 메타 정보)

    # ✅ 품질 스코어 섹션
    if score_result:
        md.append("\n---\n\n## 📊 품질 스코어\n\n")
        md.append(f"- **총점**: {score_result.total_score:.2f}/10.0\n")
        md.append(f"- **목표**: 9.5/10.0\n")
        md.append(f"- **상태**: {'✅ 통과' if score_result.passed else f'⚠️ 미달 (갭: -{score_result.gap_to_target:.2f})'}\n")

        # 모듈별 점수
        md.append("\n| 모듈 | 점수 | 목표 | 상태 |\n|------|------|------|------|\n")
        for name, score in score_result.criteria_scores.items():
            target = score_result.module_targets.get(name, 9.0)
            status = "✅" if score >= target else "❌"
            md.append(f"| {name} | {score:.2f} | {target:.2f} | {status} |\n")

        # ✅ 9.5 미달 시: 개선 가이드 추가
        if not score_result.passed:
            md.append("\n### ⚠️ 품질 개선 가이드\n\n")
            md.append(f"현재 점수가 목표({score_result.pass_threshold})에 **{score_result.gap_to_target:.2f}점** 미달합니다.\n\n")

            # 우선순위별 개선 권고
            md.append("#### 개선 권고 사항 (우선순위순)\n\n")
            for i, rec in enumerate(score_result.recommendations, 1):
                md.append(f"{i}. **{rec.module}** ({rec.current_score:.2f} → {rec.target_score:.2f})\n")
                md.append(f"   - 문제: {rec.issue}\n")
                md.append(f"   - 수정: {rec.specific_change}\n")
                md.append(f"   - 예상 점수: {rec.expected_score_after:.2f} (+{rec.expected_score_after - rec.current_score:.2f})\n\n")

    # ✅ 본문은 항상 출력
    md.append("\n---\n\n## 📝 제출용 본문\n\n")

    if score_result and not score_result.passed:
        md.append("> ⚠️ **주의**: 이 본문은 현재 품질 기준(9.5/10)에 미달합니다.\n")
        md.append("> 위 개선 권고 사항을 참고하여 수정 후 제출하시기 바랍니다.\n\n")

    md.append(final_text.strip() + "\n")

    # ... (나머지 섹션)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(md))
```

---

### 5.3 Before/After 비교

#### Before (9.5 미만 시)
```markdown
## 제출용 본문

품질 기준 미달로 본문을 출력하지 않습니다.
```

**문제**: 사용자가 아무것도 볼 수 없음

---

#### After (9.5 미만 시)
```markdown
## 📊 품질 스코어

- **총점**: 8.3/10.0
- **목표**: 9.5/10.0
- **상태**: ⚠️ 미달 (갭: -1.2)

| 모듈 | 점수 | 목표 | 상태 |
|------|------|------|------|
| question_focus | 7.5 | 9.0 | ❌ |
| specificity | 8.0 | 9.0 | ❌ |
| logic_flow | 9.0 | 9.0 | ✅ |

### ⚠️ 품질 개선 가이드

현재 점수가 목표(9.5)에 **1.2점** 미달합니다.

#### 개선 권고 사항 (우선순위순)

1. **question_focus** (7.5 → 9.5)
   - 문제: 2문단에 질문과 무관한 논문 나열
   - 수정: "CVPR 2025 논문 3편" 부분 삭제
   - 예상 점수: 8.5 (+1.0)

2. **specificity** (8.0 → 9.5)
   - 문제: "열정적으로 임했습니다" 같은 추상적 표현 3회
   - 수정: "매일 3시간씩 추가 실험 진행" 같은 구체적 행동으로 교체
   - 예상 점수: 9.0 (+1.0)

---

## 📝 제출용 본문

> ⚠️ **주의**: 이 본문은 현재 품질 기준(9.5/10)에 미달합니다.
> 위 개선 권고 사항을 참고하여 수정 후 제출하시기 바랍니다.

(본문 내용...)
```

**장점**:
- 본문을 볼 수 있음
- 무엇이 문제인지 명확
- 어떻게 고쳐야 하는지 구체적

---

## 6. 프롬프트 재설계

### 6.1 QualityScorer 프롬프트 재설계

#### Before (현재)
```python
# ❌ 추상적 평가 기준
prompt = """역할: Quality Scorer

평가 기준:
- question_focus: 질문 의도에 직접 답하고 불필요한 내용이 없는가

규칙:
- 점수는 엄격하게 부여하고, 9.5 이상은 매우 뛰어난 경우에만 부여한다.
"""
```

---

#### After (재설계)
```python
# ✅ COT + 구체적 평가 기준
prompt = """역할: Quality Scorer (제출용 자기소개서 평가)

## 평가 목표
이 자기소개서가 **9.5/10 이상**을 받으려면 어떻게 개선해야 하는지 진단합니다.

## 평가 프로세스 (COT)

### 1단계: 각 모듈 평가 (0~10점)

#### question_focus (질문 초점)
- **10점 기준**: 첫 문장부터 질문에 직접 답하고, 전체가 질문 의도와 일치
- **9점 기준**: 대부분 질문에 집중하나 무관한 문장 1~2개
- **8점 기준**: 질문 관련성 70% 이상
- **7점 이하**: 질문과 무관한 내용이 30% 이상

평가 방법:
1. 질문 키워드 추출 (예: "자신에 대해" → 성향, 가치관)
2. 본문 각 문장이 키워드와 관련 있는지 체크
3. 무관한 문장 개수와 비중 계산
4. 점수 부여 + 근거 작성

#### specificity (구체성)
- **10점 기준**: 모든 주장이 구체적 행동/상황/결과로 뒷받침됨
- **9점 기준**: 추상적 표현 1~2회
- **8점 기준**: 추상적 표현 3~4회
- **7점 이하**: 추상적 표현 5회 이상 또는 근거 없는 주장

평가 방법:
1. 추상적 표현 찾기 (예: "열정적으로", "최선을 다해")
2. 각 추상 표현 옆에 구체적 근거가 있는지 확인
3. 추상 표현 개수 카운트
4. 점수 부여 + 어떤 표현을 어떻게 바꿀지 제안

(나머지 모듈도 동일하게 구체화)

### 2단계: 9.5 도달을 위한 개선 방향

각 모듈이 9.5에 도달하려면:
- 현재 점수와 목표 점수 차이 계산
- 점수 차이를 메우기 위한 **구체적 변경 사항** 제안
- 각 변경의 예상 점수 영향 추정

출력 JSON:
{
  "scores": {
    "question_focus": 7.5,
    "logic_flow": 8.0,
    "specificity": 7.0,
    "expression_quality": 8.5
  },
  "rationales": {
    "question_focus": "2문단의 '논문 3편 게재' 부분이 질문('자신에 대해')과 무관. 1문단은 성향 제시로 적절.",
    "specificity": "'열정적으로 임했습니다'(3회), '최선을 다했습니다'(2회) 등 추상 표현 5회. 구체적 행동 부족."
  },
  "recommendations": [
    {
      "module": "question_focus",
      "current_score": 7.5,
      "target_score": 9.5,
      "issue": "2문단 논문 나열이 질문과 무관",
      "specific_change": "2문단 '논문 3편 게재' 문장 삭제",
      "expected_score_after": 8.5,
      "priority": 1
    },
    {
      "module": "specificity",
      "current_score": 7.0,
      "target_score": 9.5,
      "issue": "추상 표현 5회",
      "specific_change": "'열정적으로 임했습니다' → '매일 3시간씩 추가 실험 진행했습니다'로 교체 (3곳)",
      "expected_score_after": 8.5,
      "priority": 2
    }
  ]
}
"""
```

**차이점**:
- ❌ Before: "엄격하게 평가" (추상적)
- ✅ After: 각 점수 구간의 구체적 기준 + 개선 방향 명시

---

### 6.2 Refiner 프롬프트 재설계

#### Before (현재)
```python
# ❌ 추상적 개선 지시
prompt = f"""다음 본문을 개선하세요.

## 목표 모듈
{module}

## 필수 규칙
6. 미세 수정 금지: 점수 상승이 가능한 방향으로 내용/구조를 명확히 변경
"""
```

---

#### After (재설계)
```python
# ✅ COT 3단계 + 점수 예측
prompt = f"""당신은 자기소개서 개선 전문가입니다.
현재 본문의 '{module}' 점수를 {current_score:.2f}에서 9.5로 올려야 합니다.

## 1단계: WHY - 점수 낮은 이유 진단

현재 본문:
{text}

평가자 의견:
{rationale}

다음을 분석하세요:
1. '{module}' 점수가 {current_score:.2f}인 **구체적 이유** 3가지를 본문에서 찾아 인용
   - 예: "2문단 3행의 '열정적으로'는 추상적 표현"
2. 각 이유가 점수에 미친 영향 추정
   - 예: "추상 표현 3회 → -1.0점"
3. 9.5 도달을 위해 **반드시 변경할 부분** 우선순위

## 2단계: WHAT - 구체적 변경 계획

1단계 분석 기반으로:
1. 각 변경 사항의 Before/After
2. 왜 이 변경이 점수를 올릴지 설명
3. 예상 점수 영향

## 3단계: HOW - 변경 적용

계획된 변경을 본문에 적용하되:
- 나머지 부분은 그대로 유지
- 문장 연결 자연스럽게
- 제공되지 않은 수치/사실 추가 금지

출력 JSON:
{{
  "analysis": {{
    "reasons": [
      {{"issue": "추상 표현", "location": "2문단 3행", "quote": "열정적으로", "score_impact": -0.5}}
    ],
    "priority_changes": [
      {{"rank": 1, "what": "2문단 '열정적으로' → 구체적 행동", "expected_gain": +0.5}}
    ]
  }},
  "changes": [
    {{
      "before": "열정적으로 임했습니다",
      "after": "매일 3시간씩 추가 실험을 진행했습니다",
      "rationale": "추상 표현을 구체적 행동으로 교체 → specificity +0.5",
      "expected_score_impact": +0.5
    }}
  ],
  "improved_text": "(변경 적용된 전체 본문)",
  "total_expected_gain": +1.0
}}
"""
```

**차이점**:
- ❌ Before: "점수 상승 가능한 방향으로" (모호)
- ✅ After: WHY-WHAT-HOW 3단계 + 점수 예측

---

## 7. 구현 가이드

### 7.1 구현 우선순위

#### Phase 1: 본문 출력 로직 수정 (즉시 적용 가능)
- `write_markdown_v2` 함수로 교체
- 9.5 미만이어도 본문 + 개선 가이드 출력

#### Phase 2: 스키마 정의 (타입 안전성)
- `schemas.py` 파일 생성
- 각 에이전트 입출력 스키마 dataclass로 정의
- `call_agent_with_schema` 헬퍼 함수 구현

#### Phase 3: COT 기반 Refine 로직 (점수 상승)
- `RefineLoopV2` 클래스 구현
- WHY-WHAT-HOW 3단계 프롬프트 적용

#### Phase 4: Scorer 프롬프트 개선
- 각 점수 구간 구체적 기준 추가
- Recommendation 스키마 적용

---

### 7.2 코드 구조 (제안)

```
CTS/
├── run_sop.py (main)
├── schemas.py (새로 추가)
│   ├── PlannerInput, PlannerOutput
│   ├── CTSScorerInput, CTSScorerOutput
│   ├── WriterInput, WriterOutput
│   ├── ReviewerInput, ReviewerOutput, Fix
│   ├── IntegratorInput, IntegratorOutput
│   ├── ScorerInput, ScoreResult, Recommendation
│   ├── RefinerInput, RefinerOutput
├── refine_v2.py (새로 추가)
│   ├── RefineLoopV2
│   ├── analyze_low_score_with_cot()
│   ├── propose_concrete_changes()
│   ├── apply_changes()
├── prompts_v2.py (새로 추가)
│   ├── build_scorer_prompt_v2()
│   ├── build_refiner_prompt_v2()
│   ├── build_planner_prompt_v2()
├── output_v2.py (새로 추가)
│   ├── write_markdown_v2()
├── config.yaml (업데이트)
│   ├── scoring.pass_threshold: 9.5
│   ├── scoring.always_output_body: true
│   ├── refine_loop.use_cot: true
```

---

### 7.3 설정 파일 변경

`config.yaml`에 추가:

```yaml
scoring:
  enabled: true
  pass_threshold: 9.5

  # ✅ 새로 추가
  always_output_body: true  # 9.5 미만이어도 본문 출력
  provide_improvement_guide: true  # 개선 가이드 제공

  modules:
    question_focus:
      weight: 0.28
      min_score: 9.0
      # ✅ 새로 추가: 각 점수 구간 설명
      score_levels:
        10: "첫 문장부터 질문 직접 답변, 전체 일치"
        9: "무관한 문장 1~2개"
        8: "관련성 70% 이상"
        7: "무관 내용 30% 이상"

refine_loop:
  enabled: true

  # ✅ 새로 추가
  use_cot: true  # COT 3단계 활성화
  predict_score_impact: true  # 점수 영향 예측

  max_iterations: 5
  max_total_iterations: 20
```

---

### 7.4 마이그레이션 가이드

#### 기존 코드 → 재설계 코드

**Step 1**: `schemas.py` 생성
```python
# schemas.py
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class Recommendation:
    module: str
    current_score: float
    target_score: float
    issue: str
    specific_change: str
    expected_score_after: float
    priority: int

@dataclass
class ScoreResultV2:
    total_score: float
    criteria_scores: Dict[str, float]
    passed: bool
    failed_criteria: List[str]
    rationales: Dict[str, str]  # ✅ 추가
    gap_to_target: float  # ✅ 추가
    recommendations: List[Recommendation]  # ✅ 추가

# (나머지 스키마도 추가)
```

**Step 2**: `run_sop.py` 수정
```python
# run_sop.py

# Before
from dataclasses import dataclass

@dataclass
class ScoreResult:
    total_score: float
    criteria_scores: Dict[str, float]
    passed: bool
    failed_criteria: List[str]
    recommendations: List[str]  # ❌ 단순 문자열

# After
from schemas import ScoreResultV2, Recommendation

# (ScoreResult를 ScoreResultV2로 교체)
```

**Step 3**: `write_markdown_v2` 적용
```python
# run_sop.py:1417

# Before
def write_markdown(...):
    allow_body_output = True
    if score_result and not score_result.passed:
        allow_body_output = False
        md.append("품질 기준 미달로 본문을 출력하지 않습니다.\n")

# After
from output_v2 import write_markdown_v2

# write_markdown → write_markdown_v2로 교체
```

**Step 4**: `RefineLoopV2` 적용
```python
# run_sop.py:1833

# Before
from run_sop import RefineLoop
refine_loop = RefineLoop(client, cfg, logger, scorer, model_selector)

# After
from refine_v2 import RefineLoopV2
refine_loop = RefineLoopV2(client, cfg, logger, scorer, model_selector)
```

---

## 8. 검증 방법

### 8.1 기능 검증

#### 테스트 1: 본문 출력 확인
```bash
# 1. 의도적으로 낮은 품질 본문 생성 (임시로 pass_threshold=10.0 설정)
# 2. 9.5 미만이어도 본문이 출력되는지 확인
# 3. "개선 가이드" 섹션이 있는지 확인
```

#### 테스트 2: Refine 점수 상승 확인
```bash
# 1. 초기 본문 점수 기록
# 2. Refine 루프 실행
# 3. 각 iteration에서 점수가 상승하는지 확인
# 4. 로그에서 "WHY-WHAT-HOW" 단계가 출력되는지 확인
```

#### 테스트 3: 스키마 검증
```python
# schemas.py에 정의한 스키마대로 데이터가 전달되는지 확인
from schemas import PlannerOutput

# LLM 응답을 스키마로 변환
planner_output = PlannerOutput(**json_response)

# 필수 필드 확인
assert planner_output.outline
assert planner_output.reasoning_summary
```

---

### 8.2 품질 검증

#### 메트릭 1: 9.5 도달률
```python
# 10개 질문 테스트
# - Before: 9.5 도달 0/10
# - After: 9.5 도달 7/10 (목표)
```

#### 메트릭 2: Refine 반복 횟수
```python
# - Before: 평균 18회 반복 후에도 미달
# - After: 평균 3~5회 반복 후 도달 (목표)
```

#### 메트릭 3: 사용자 만족도
```python
# - Before: 본문 못 봄 → 불만
# - After: 본문 + 개선 가이드 → 만족
```

---

## 9. 예상 결과

### 9.1 Before (현재)

```
[실행]
→ Planner (3개 후보)
→ CTS (best 선택)
→ Writer
→ Reviewer
→ Integrator
→ Length Fixer
→ Scorer: 8.3/10 ❌
→ Refine 18회 반복
   - iteration 1: 8.3 → 8.4
   - iteration 2: 8.4 → 8.3 (악화)
   - iteration 3: 8.3 → 8.5
   ...
   - iteration 18: 8.9 → 8.8
→ 최종: 8.8/10 (미달)

[출력]
📊 품질 스코어: 8.8/10 (❌ 미달)
📝 제출용 본문: 품질 기준 미달로 출력하지 않습니다.
```

**문제**:
- 18회 반복해도 9.5 도달 실패
- 사용자는 본문을 볼 수 없음
- 왜 실패했는지 알 수 없음

---

### 9.2 After (재설계)

```
[실행]
→ Planner (3개 후보, COT 포함)
→ CTS (구조화된 비교 분석)
→ Writer
→ Reviewer (구조화된 Fix 제안)
→ Integrator (Fix 적용 이력 기록)
→ Length Fixer
→ Scorer: 8.3/10 ❌
→ Refine V2 (COT 3단계)
   - iteration 1:
     WHY: question_focus 7.5 → "2문단 논문 나열 무관"
     WHAT: "논문 3편 부분 삭제" (예상 +1.0)
     HOW: 적용
     → 8.3 → 9.2 (실제 +0.9)

   - iteration 2:
     WHY: specificity 8.0 → "추상 표현 3회"
     WHAT: "'열정적으로' → '매일 3시간 실험'" (예상 +0.8)
     HOW: 적용
     → 9.2 → 9.6 ✅
→ 최종: 9.6/10 (통과)

[출력]
📊 품질 스코어: 9.6/10 (✅ 통과)

📝 제출용 본문:
(본문 전체 출력)
```

**개선**:
- 2회 반복으로 9.5 도달 (18회 → 2회)
- 본문 출력 + 개선 과정 투명
- COT로 각 변경의 이유 명확

---

### 9.3 만약 9.5 미달 시 (After)

```
[실행]
→ (동일)
→ Refine V2 (5회 반복)
   - iteration 1~5: 8.3 → 9.3
→ 최종: 9.3/10 (미달, gap: -0.2)

[출력]
📊 품질 스코어

- 총점: 9.3/10.0
- 목표: 9.5/10.0
- 상태: ⚠️ 미달 (갭: -0.2)

| 모듈 | 점수 | 목표 | 상태 |
|------|------|------|------|
| question_focus | 9.5 | 9.0 | ✅ |
| specificity | 9.0 | 9.0 | ✅ |
| expression_quality | 9.2 | 9.0 | ✅ |
| submission_ready | 10.0 | 10 | ✅ |
| length_fit | 10.0 | 10 | ✅ |
| logic_flow | 8.8 | 9.0 | ❌ |

### ⚠️ 품질 개선 가이드

현재 점수가 목표(9.5)에 0.2점 미달합니다.

#### 개선 권고 사항

1. **logic_flow** (8.8 → 9.0)
   - 문제: 2문단과 3문단 연결이 갑작스러움
   - 수정: 2문단 마지막에 "이러한 경험을 통해" 같은 연결 문구 추가
   - 예상 점수: 9.0 (+0.2)

---

## 📝 제출용 본문

> ⚠️ **주의**: 이 본문은 현재 품질 기준(9.5/10)에 0.2점 미달합니다.
> 위 개선 권고 사항을 참고하여 수정 후 제출하시기 바랍니다.

(본문 전체 출력)
```

**개선**:
- ✅ 본문 출력 (사용자가 볼 수 있음)
- ✅ 무엇이 문제인지 명확 (logic_flow 8.8)
- ✅ 어떻게 고쳐야 하는지 구체적 (연결 문구 추가)
- ✅ 예상 효과 제시 (+0.2점)

---

## 10. 결론

### 10.1 핵심 변경 사항 요약

| 항목 | Before (현재) | After (재설계) |
|------|---------------|----------------|
| **본문 출력** | 9.5 미만 시 차단 ❌ | 항상 출력 + 개선 가이드 ✅ |
| **Refine 전략** | 추상적 프롬프트 ❌ | COT 3단계 (WHY-WHAT-HOW) ✅ |
| **에이전트 통신** | 느슨한 dict ❌ | 스키마 기반 (dataclass) ✅ |
| **점수 상승** | 반복해도 정체 ❌ | 구체적 변경 → 예측 가능 상승 ✅ |
| **프롬프트** | "점수 올릴 방향으로" ❌ | "X를 Y로 → +Z점" ✅ |

---

### 10.2 기대 효과

1. **본문 출력 차단 문제 해결**
   - 사용자가 항상 결과를 확인 가능
   - 9.5 미달 시에도 개선 방향 제시

2. **점수 정체 문제 해결**
   - COT로 "왜 낮은지" 분석
   - 구체적 변경으로 "점수 상승" 유도
   - 18회 → 3~5회 반복으로 효율 향상

3. **에이전트 상호작용 명확화**
   - 스키마로 입출력 구조화
   - 다른 AI가 재구현 가능한 수준

4. **사용자 경험 개선**
   - "블랙박스" → "투명한 개선 과정"
   - 실패 시에도 "왜/어떻게" 안내

---

### 10.3 구현 로드맵

#### Week 1
- Phase 1: 본문 출력 로직 수정
- Phase 2: 스키마 정의

#### Week 2
- Phase 3: COT 기반 Refine 구현
- Phase 4: Scorer 프롬프트 개선

#### Week 3
- 통합 테스트
- 10개 질문으로 Before/After 비교

#### Week 4
- 성능 튜닝
- 문서화

---

**작성자**: Claude Sonnet 4.5
**작성일**: 2026-01-04
