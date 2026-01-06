# CTS 자기소개서 시스템 - 구현 가이드

> **작성일**: 2026-01-04
> **목적**: 재설계 방안의 단계별 구현 가이드
> **선행 문서**: [REDESIGN_PROPOSAL.md](REDESIGN_PROPOSAL.md)

---

## 목차

1. [구현 완료 항목](#1-구현-완료-항목)
2. [다음 구현 항목](#2-다음-구현-항목)
3. [코드 변경 사항 상세](#3-코드-변경-사항-상세)
4. [테스트 방법](#4-테스트-방법)
5. [다음 단계 로드맵](#5-다음-단계-로드맵)

---

## 1. 구현 완료 항목

### ✅ Phase 1: 본문 출력 로직 수정 (완료)

#### 변경 사항

**파일**: [`run_sop.py`](run_sop.py)

**함수**: `write_markdown` (line 1417~)

#### 주요 수정 내용

1. **본문 출력 차단 로직 제거**

**Before**:
```python
# ❌ 9.5 미만 시 본문 차단
allow_body_output = True
if score_result and not score_result.passed:
    allow_body_output = False
    md.append("품질 기준 미달로 본문을 출력하지 않습니다.\n")
```

**After**:
```python
# ✅ 본문은 항상 출력 (9.5 미만이어도 출력)
md.append("\n---\n\n## 📝 제출용 본문\n\n")

# 품질 미달 시 경고 메시지 추가
if score_result and not score_result.passed:
    md.append("> ⚠️ **주의**: 이 본문은 현재 품질 기준(9.5/10)에 미달합니다.\n")
    md.append("> 위 개선 권고 사항을 참고하여 수정 후 제출하시기 바랍니다.\n\n")

if submission_text:
    md.append(submission_text.strip() + "\n")
```

2. **품질 스코어 표시 개선**

**Before**:
```python
md.append(f"- **통과**: {'✅ 예' if score_result.passed else '❌ 아니오'}\n")
```

**After**:
```python
md.append(f"- **총점**: {score_result.total_score:.2f}/10.0\n")
md.append(f"- **목표**: {pass_threshold:.1f}/10.0\n")

if score_result.passed:
    md.append(f"- **상태**: ✅ 통과\n")
else:
    md.append(f"- **상태**: ⚠️ 미달 (갭: -{gap:.2f}점)\n")
```

3. **모듈별 상태 표시 추가**

```python
md.append("\n| 항목 | 점수 | 상태 |\n|------|------|------|\n")
for name, score in score_result.criteria_scores.items():
    if name != "rationale":
        module_cfg = scoring_cfg.get("modules", {}).get(name, {})
        min_score = module_cfg.get("min_score", 9.0)
        status = "✅" if score >= min_score else "❌"
        md.append(f"| {name} | {score:.2f} | {status} |\n")
```

4. **개선 가이드 추가**

```python
if not score_result.passed and score_result.recommendations:
    md.append("\n### ⚠️ 품질 개선 가이드\n\n")
    md.append(f"현재 점수가 목표({pass_threshold:.1f})에 **{gap:.2f}점** 미달합니다.\n\n")
    md.append("#### 개선 권고 사항 (우선순위순)\n\n")
    for i, rec in enumerate(score_result.recommendations[:5], 1):
        md.append(f"{i}. **권고**: {rec}\n")
```

5. **본문 파일(.txt) 생성 로직 개선**

**Before**:
```python
# ❌ 9.5 통과 시에만 파일 생성
if submission_path and submission_text and allow_body_output:
```

**After**:
```python
# ✅ 항상 파일 생성 (미달 시 경고 추가)
if submission_path and submission_text:
    ensure_dir(submission_path)
    with open(submission_path, "w", encoding="utf-8") as f:
        if score_result and not score_result.passed:
            f.write("<!-- ⚠️ 경고: 이 본문은 품질 기준(9.5/10) 미달. 개선 후 제출 필요 -->\n\n")
        f.write(submission_text.strip() + "\n")
```

6. **터미널 출력 로직 개선**

**Before**:
```python
# ❌ 통과 시에만 출력
if cfg.get("output", {}).get("print_final_to_terminal", False) and (not score_result or score_result.passed):
```

**After**:
```python
# ✅ 항상 출력 가능 (미달 시 경고 표시)
if cfg.get("output", {}).get("print_final_to_terminal", False):
    print("\n" + "=" * 60)
    if score_result and not score_result.passed:
        print(f"⚠️ 경고: 품질 기준(9.5/10) 미달 (현재: {score_result.total_score:.2f})")
        print("=" * 60)
    print(final_text)
    print("=" * 60)
```

---

#### 예상 출력 예시

**9.5 미달 시 (예: 8.3점)**:

```markdown
## 📊 품질 스코어

- **총점**: 8.30/10.0
- **목표**: 9.5/10.0
- **상태**: ⚠️ 미달 (갭: -1.20점)

| 항목 | 점수 | 상태 |
|------|------|------|
| question_focus | 7.50 | ❌ |
| logic_flow | 9.00 | ✅ |
| specificity | 8.00 | ❌ |
| expression_quality | 9.20 | ✅ |
| submission_ready | 10.00 | ✅ |
| length_fit | 10.00 | ✅ |

**미달 항목**: question_focus, specificity

### ⚠️ 품질 개선 가이드

현재 점수가 목표(9.5)에 **1.20점** 미달합니다.

#### 개선 권고 사항 (우선순위순)

1. **권고**: 질문에 직접 답하는 문장을 앞부분에 배치하고 관련 없는 내용을 제거
2. **권고**: 추상 표현을 줄이고 행동/상황/결과를 구체적으로 보강

---

## 📝 제출용 본문

> ⚠️ **주의**: 이 본문은 현재 품질 기준(9.5/10)에 미달합니다.
> 위 개선 권고 사항을 참고하여 수정 후 제출하시기 바랍니다.

(본문 내용...)
```

---

### 효과

1. **사용자 경험 개선**
   - ✅ 본문을 항상 확인 가능
   - ✅ 무엇이 문제인지 명확히 파악
   - ✅ 어떻게 고쳐야 하는지 구체적 안내

2. **디버깅 용이**
   - ✅ 점수 낮은 이유 분석 가능
   - ✅ 개선 방향 검증 가능

3. **블랙박스 문제 해결**
   - ❌ Before: "본문 출력 안 됨" → 왜인지 모름
   - ✅ After: "본문 + 개선 가이드" → 투명

---

## 2. 다음 구현 항목

### Phase 2: 스키마 정의 (다음 단계)

#### 목표
에이전트 간 통신을 명시적이고 타입 안전하게 만들기

#### 구현 계획

**새 파일**: `schemas.py`

```python
from dataclasses import dataclass
from typing import List, Dict, Optional

# ============================================================
# 평가 관련 스키마
# ============================================================

@dataclass
class Recommendation:
    """구체적 개선 권고"""
    module: str  # "question_focus"
    current_score: float  # 7.5
    target_score: float  # 9.5
    issue: str  # "2문단에 질문과 무관한 내용 포함"
    specific_change: str  # "2문단의 '논문 3편 게재' 부분 삭제"
    expected_score_after: float  # 8.5
    priority: int  # 1=high, 2=medium, 3=low

@dataclass
class ScoreResultV2:
    """품질 평가 결과 (강화 버전)"""
    total_score: float
    criteria_scores: Dict[str, float]
    passed: bool
    failed_criteria: List[str]

    # ✅ 추가: COT 및 구체적 개선 방향
    rationales: Dict[str, str]  # 각 모듈별 점수 이유
    gap_to_target: float  # 9.5 - total_score
    recommendations: List[Recommendation]  # 구조화된 권고

# ============================================================
# 에이전트 입출력 스키마
# ============================================================

@dataclass
class PlannerInput:
    """Planner 에이전트 입력"""
    question: str
    question_type: str  # "자기소개", "지원동기" 등
    company_profile: Dict
    evidence: str  # RAG 검색 결과
    constraints: Dict
    purpose: str = "질문에 답하는 자기소개서 구조 설계"

@dataclass
class PlannerOutput:
    """Planner 에이전트 출력"""
    # 핵심 계획
    outline: List[str]
    core_messages: List[str]
    personality_traits: List[str]
    experience_to_use: List[str]

    # COT 추론
    reasoning_summary: str
    expected_strengths: List[str]
    expected_weaknesses: List[str]

    # 개선 방향
    must_avoid: List[str]

    # 메타 정보
    planner_id: str  # "strategic", "creative", "critical"
    confidence: float  # 0.0~1.0

@dataclass
class Fix:
    """Reviewer가 제안하는 수정 사항"""
    location: str  # "2번째 문단", "도입부"
    issue: str  # "추상적 표현"
    current_text: str  # 문제가 있는 현재 텍스트
    suggested_change: str  # 구체적 수정 제안
    rationale: str  # 왜 이 수정이 필요한지
    expected_score_impact: float  # 이 수정이 점수에 미칠 영향 (+0.5 등)

@dataclass
class ReviewerInput:
    """Reviewer 에이전트 입력"""
    draft_text: str
    writer_output: Dict  # WriterOutput (순환 참조 방지를 위해 Dict)
    selected_plan: Dict  # PlannerOutput
    question: str
    company_profile: Dict
    evidence: str
    constraints: Dict
    purpose: str = "초안의 문제점 진단 및 수정 방향 제시"

@dataclass
class ReviewerOutput:
    """Reviewer 에이전트 출력"""
    # 문제 진단
    is_report_style: bool
    has_first_person: bool
    first_person_found: List[str]
    hallucination_risks: List[str]

    # 수정 지침 (구조화)
    fixes: List[Fix]

    # COT 분석
    overall_assessment: str
    priority_fixes: List[str]

# (WriterInput, WriterOutput, IntegratorInput, IntegratorOutput 등도 동일하게 정의)
```

---

### Phase 3: COT 기반 Refine 로직 (핵심)

#### 목표
점수 정체 문제를 해결하기 위해 WHY-WHAT-HOW 3단계 COT 적용

#### 구현 계획

**새 파일**: `refine_v2.py`

```python
from dataclasses import dataclass
from typing import List, Dict, Tuple
from openai import OpenAI
import json
import logging

@dataclass
class RefineAnalysis:
    """Refine 1단계: WHY 분석 결과"""
    reasons: List[Dict]  # [{"issue": "추상 표현", "location": "2문단", ...}]
    priority_changes: List[Dict]
    gap_to_target: float

@dataclass
class RefineProposal:
    """Refine 2단계: WHAT 제안 결과"""
    changes: List[Dict]  # [{"before": "...", "after": "...", ...}]
    total_expected_gain: float

class RefineLoopV2:
    """COT 기반 품질 개선 루프"""

    def __init__(self, client: OpenAI, cfg: Dict, logger: logging.Logger,
                 scorer, model_selector):
        self.client = client
        self.cfg = cfg
        self.logger = logger
        self.scorer = scorer
        self.model_selector = model_selector

        refine_cfg = cfg.get("refine_loop", {})
        self.enabled = refine_cfg.get("enabled", True)
        self.use_cot = refine_cfg.get("use_cot", True)
        self.max_total_iterations = refine_cfg.get("max_total_iterations", 20)

        self.model = model_selector.get_model("refiner")
        self.max_tokens = cfg.get("openai", {}).get("max_tokens", {}).get("refiner", 1800)

    def refine(self, text: str, score_result, question: str,
               company_profile: Dict, evidence: str, qtype,
               constraints: Dict) -> Tuple[str, Any, List]:
        """
        3단계 COT 기반 개선:
        1. WHY: 점수 낮은 이유 분석
        2. WHAT: 구체적 변경 제안
        3. HOW: 변경 적용
        """

        if not self.enabled or score_result.passed:
            return text, score_result, []

        if not self.use_cot:
            # 기존 로직으로 폴백
            return self._refine_legacy(text, score_result, question, company_profile, evidence, qtype, constraints)

        current_text = text
        current_score = score_result
        iterations = []

        for i in range(self.max_total_iterations):
            target_module = self._pick_target_module(current_score)
            if not target_module:
                break

            self.logger.info(f"[Refine {i+1}] 대상 모듈: {target_module}")

            # ✅ 단계 1: WHY - COT로 원인 분석
            analysis = self._analyze_low_score_with_cot(
                current_text, current_score, target_module
            )

            # ✅ 단계 2: WHAT - 구체적 변경 제안
            proposals = self._propose_concrete_changes(
                current_text, analysis, target_module
            )

            # ✅ 단계 3: HOW - 변경 적용
            improved_text = self._apply_changes(
                current_text, proposals["changes"]
            )

            # 점수 재평가
            new_score = self.scorer.score(
                improved_text, question, company_profile, evidence, qtype
            )

            actual_gain = new_score.total_score - current_score.total_score
            expected_gain = proposals.get("total_expected_gain", 0)

            self.logger.info(
                f"[Refine {i+1}] {current_score.total_score:.2f} → {new_score.total_score:.2f} "
                f"(예상: +{expected_gain:.2f}, 실제: {actual_gain:+.2f})"
            )

            # 이력 기록
            # (iteration 객체 생성 및 추가)

            current_text = improved_text
            current_score = new_score

            if current_score.passed:
                self.logger.info(f"[Refine] ✅ 품질 통과! ({current_score.total_score:.2f}/10)")
                break

            if actual_gain < 0.1:
                self.logger.warning(f"[Refine] 개선 폭 미미, 다음 모듈로")
                continue

        return current_text, current_score, iterations

    def _analyze_low_score_with_cot(self, text: str, score_result, module: str) -> Dict:
        """단계 1: WHY - 점수 낮은 이유 COT 분석"""

        current_score = score_result.criteria_scores.get(module, 0)
        rationale = score_result.rationales.get(module, "") if hasattr(score_result, 'rationales') else ""

        prompt = f"""다음 자기소개서의 '{module}' 모듈 점수가 {current_score:.2f}/10입니다.
목표는 9.5/10입니다.

현재 본문:
{text}

평가자 의견:
{rationale}

단계별로 분석하세요:

1. 점수가 {current_score:.2f}인 **구체적 이유** 3가지를 본문에서 찾아 인용하세요.
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
        {{"rank": 1, "what": "2문단 '열정적으로' → 구체적 행동", "expected_gain": 0.5}},
        ...
    ],
    "gap_to_target": 1.5
}}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens
            )

            result_text = response.choices[0].message.content.strip()
            # JSON 파싱
            analysis = json.loads(result_text)
            return analysis
        except Exception as e:
            self.logger.warning(f"[Refine] WHY 분석 실패: {e}")
            return {"reasons": [], "priority_changes": [], "gap_to_target": 0}

    def _propose_concrete_changes(self, text: str, analysis: Dict, module: str) -> Dict:
        """단계 2: WHAT - 구체적 변경 제안"""

        priority_changes = analysis.get("priority_changes", [])

        prompt = f"""다음 자기소개서를 '{module}' 점수 향상을 위해 수정합니다.

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
            "expected_score_impact": 0.5
        }},
        ...
    ],
    "total_expected_gain": 1.5
}}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens
            )

            result_text = response.choices[0].message.content.strip()
            proposals = json.loads(result_text)
            return proposals
        except Exception as e:
            self.logger.warning(f"[Refine] WHAT 제안 실패: {e}")
            return {"changes": [], "total_expected_gain": 0}

    def _apply_changes(self, text: str, changes: List[Dict]) -> str:
        """단계 3: HOW - 변경 적용"""

        prompt = f"""다음 자기소개서에 변경 사항을 적용하세요.

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

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens
            )

            improved_text = response.choices[0].message.content.strip()
            return improved_text
        except Exception as e:
            self.logger.warning(f"[Refine] HOW 적용 실패: {e}")
            return text

    def _pick_target_module(self, score_result) -> str:
        """개선 대상 모듈 선택"""
        if score_result.failed_criteria:
            return score_result.failed_criteria[0]
        if not score_result.passed:
            scores = score_result.criteria_scores or {}
            if not scores:
                return None
            return min(scores.keys(), key=lambda k: scores.get(k, 0))
        return None

    def _refine_legacy(self, text, score_result, question, company_profile, evidence, qtype, constraints):
        """기존 Refine 로직 (폴백용)"""
        # (기존 RefineLoop 로직)
        pass
```

---

### Phase 4: Scorer 프롬프트 개선

#### 목표
평가 기준을 구체화하고 COT 활용

#### 구현 계획

**파일**: `run_sop.py`의 `QualityScorer._build_scorer_prompt` 수정

```python
def _build_scorer_prompt(self) -> str:
    return """역할: Quality Scorer (제출용 자기소개서 평가)

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
    }
  ]
}
"""
```

---

## 3. 코드 변경 사항 상세

### 변경 파일 목록

1. ✅ **[run_sop.py](run_sop.py)**: `write_markdown` 함수 수정 (완료)
2. 🔄 **schemas.py**: 에이전트 스키마 정의 (다음 단계)
3. 🔄 **refine_v2.py**: COT 기반 Refine 로직 (다음 단계)
4. 🔄 **prompts_v2.py**: 개선된 프롬프트 (다음 단계)

### 설정 파일 변경

**[config.yaml](config.yaml)**에 추가할 항목:

```yaml
# ✅ 이미 존재하는 설정은 그대로 유지

refine_loop:
  enabled: true
  use_cot: true  # ✅ 추가: COT 3단계 활성화
  predict_score_impact: true  # ✅ 추가: 점수 영향 예측
  max_iterations: 5
  max_total_iterations: 20
  max_iterations_per_module: 4

scoring:
  enabled: true
  pass_threshold: 9.5

  # ✅ 추가: 각 점수 구간 설명
  modules:
    question_focus:
      weight: 0.28
      min_score: 9.0
      score_levels:  # ✅ 추가
        10: "첫 문장부터 질문 직접 답변, 전체 일치"
        9: "무관한 문장 1~2개"
        8: "관련성 70% 이상"
        7: "무관 내용 30% 이상"
```

---

## 4. 테스트 방법

### 테스트 1: 본문 출력 확인

```bash
# 1. 기존 설정으로 실행
python run_sop.py --config config.yaml

# 2. 출력 파일 확인
ls outputs/

# 3. Markdown 파일 열어서 확인
# - "📝 제출용 본문" 섹션이 있는지
# - 9.5 미달 시 경고 메시지가 있는지
# - "⚠️ 품질 개선 가이드" 섹션이 있는지
```

### 테스트 2: 점수 표시 확인

```bash
# 1. outputs/*.md 파일 열기
# 2. "📊 품질 스코어" 섹션 확인
#    - 총점, 목표, 상태가 표시되는지
#    - 모듈별 상태 (✅/❌)가 표시되는지
```

### 테스트 3: 본문 파일(.txt) 확인

```bash
# 1. outputs/*_submission.txt 파일 확인
# 2. 9.5 미달 시 상단에 경고 주석이 있는지 확인
```

---

## 5. 다음 단계 로드맵

### Week 1 (현재)
- [x] Phase 1: 본문 출력 로직 수정 ✅
- [ ] 테스트 및 버그 수정
- [ ] 사용자 피드백 수집

### Week 2
- [ ] Phase 2: 스키마 정의 (`schemas.py`)
- [ ] Phase 3: COT 기반 Refine 로직 (`refine_v2.py`)

### Week 3
- [ ] Phase 4: Scorer 프롬프트 개선
- [ ] 통합 테스트
- [ ] 10개 질문으로 Before/After 비교

### Week 4
- [ ] 성능 튜닝
- [ ] 최종 문서화
- [ ] 릴리스

---

## 6. 문제 해결 가이드

### 문제 1: cfg 파라미터 관련 에러

**증상**:
```
NameError: name 'cfg' is not defined
```

**해결**:
- `write_markdown` 함수 호출 시 `cfg=cfg` 전달 (이미 수정 완료)

### 문제 2: 본문이 여전히 출력되지 않음

**점검 사항**:
1. `run_sop.py`가 최신 버전인지 확인
2. `write_markdown` 함수의 `allow_body_output` 로직이 제거되었는지 확인
3. git에서 최신 변경사항 pull

### 문제 3: 스코어 표시가 이상함

**점검 사항**:
1. `config.yaml`의 `scoring.modules`에 `min_score` 설정 확인
2. `scoring.pass_threshold` 값 확인 (기본 9.5)

---

## 7. 참고 자료

- [재설계 제안서](REDESIGN_PROPOSAL.md)
- [GitHub 저장소](https://github.com/lbg030/CTS)
- [config.yaml](config.yaml)

---

**작성자**: Claude Sonnet 4.5
**최종 수정**: 2026-01-04
