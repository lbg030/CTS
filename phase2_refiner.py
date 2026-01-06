"""
Phase 2: Structural Quality Improvement (9.0-9.2 baseline)

This module implements COT (Chain of Thought) based refinement with
WHY-WHAT-HOW analysis for structural quality improvements.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from schemas import (
    Phase2RefinerInput,
    Phase2DiagnosticResult,
    Phase2ImprovementPlan,
    RefineIteration,
    RootCause,
    SideEffectRisk,
    EvidenceNeed,
    ContentChange,
    RefinementApproach,
    RiskLevel,
)


class Phase2StructuralRefiner:
    """
    Structural quality refiner focusing on achieving 9.0-9.2 baseline.

    Uses COT-based WHY-WHAT-HOW refinement strategy:
    1. WHY: Analyze root causes of low scores
    2. WHAT: Design specific improvement plan
    3. HOW: Execute controlled changes
    """

    def __init__(self, client, cfg: Dict, logger: logging.Logger, scorer, kb_searcher, model_selector):
        self.client = client
        self.cfg = cfg
        self.logger = logger
        self.scorer = scorer
        self.kb_searcher = kb_searcher
        self.model_selector = model_selector

        phase2_cfg = cfg.get("phase2", {})
        self.max_iterations = phase2_cfg.get("max_iterations", 8)
        self.target_score = phase2_cfg.get("target_score", 9.0)
        self.stagnation_threshold = phase2_cfg.get("stagnation_threshold", 3)
        self.enable_evidence_refresh = phase2_cfg.get("enable_evidence_refresh", True)

    def refine(
        self,
        text: str,
        score_result: Any,  # ScoreResult
        question: str,
        qtype: Any,  # QuestionType
        company_profile: Dict,
        evidence: str,
        constraints: Dict
    ) -> Tuple[str, Any, List[RefineIteration]]:
        """
        Phase 2: Structural quality improvement to 9.0-9.2

        Returns:
            (improved_text, final_score, iteration_history)
        """
        # Check entry condition
        if score_result.total_score >= self.target_score:
            self.logger.info(f"[Phase2] Already meets target {self.target_score}, skipping")
            return text, score_result, []

        self.logger.info(f"[Phase2] Starting refinement from {score_result.total_score:.2f} to {self.target_score}")

        current_text = text
        current_score = score_result
        iterations = []
        stagnation_count = 0

        for i in range(self.max_iterations):
            self.logger.info(f"[Phase2-{i+1}/{self.max_iterations}] Starting iteration")

            # 1. Pick target module
            target_module = self._pick_target_module(current_score)
            if not target_module:
                self.logger.info("[Phase2] No module needs improvement")
                break

            self.logger.info(f"[Phase2-{i+1}] Target module: {target_module} ({current_score.criteria_scores[target_module]:.2f})")

            # 2. Build structured input
            refiner_input = Phase2RefinerInput(
                question=question,
                question_type=qtype.value if hasattr(qtype, 'value') else str(qtype),
                current_text=current_text,
                current_score=current_score.total_score,
                score_breakdown={k: v for k, v in current_score.criteria_scores.items()},
                target_module=target_module,
                target_module_score=current_score.criteria_scores[target_module],
                scorer_rationale=current_score.rationales.get(target_module, ""),
                preserve_strengths=self._identify_strengths(current_score),
                forbidden_changes=self._get_forbidden_changes(qtype),
                available_evidence=evidence,
                iteration_num=i + 1,
                previous_attempts=[it.diagnostics for it in iterations],
                current_char_count=len(current_text),
                target_char_min=constraints.get("target_char_min", 950),
                target_char_max=constraints.get("target_char_max", 1000)
            )

            # 3. COT Step 1: WHY - Diagnostic
            self.logger.info(f"[Phase2-{i+1}] WHY: Analyzing root causes for {target_module}")
            diagnostic = self._analyze_why_low_score(refiner_input)

            # 4. Check if evidence refresh is needed
            if diagnostic.requires_new_evidence and self.enable_evidence_refresh:
                self.logger.info(f"[Phase2-{i+1}] Evidence refresh needed: {diagnostic.evidence_need.type}")
                new_evidence = self._refresh_evidence(
                    diagnostic.evidence_need,
                    question
                )
                refiner_input.available_evidence = new_evidence

            # 5. COT Step 2: WHAT - Strategy design
            self.logger.info(f"[Phase2-{i+1}] WHAT: Planning improvement strategy")
            plan = self._design_improvement_plan(refiner_input, diagnostic)

            # 6. Risk check
            if plan.risk_level == RiskLevel.HIGH:
                self.logger.warning(f"[Phase2-{i+1}] High risk plan detected, applying conservative edits")
                plan = self._make_plan_conservative(plan)

            # 7. COT Step 3: HOW - Execute
            self.logger.info(f"[Phase2-{i+1}] HOW: Executing {plan.approach.value}")
            if plan.approach == RefinementApproach.SECTION_REWRITE:
                improved_text = self._rewrite_section(refiner_input, plan)
            else:
                improved_text = self._apply_incremental_edits(refiner_input, plan)

            # 8. Re-score
            self.logger.info(f"[Phase2-{i+1}] Re-scoring improved text")
            new_score = self.scorer.score(
                improved_text, question, company_profile, evidence, qtype
            )

            # 9. Evaluate improvement
            module_delta = new_score.criteria_scores[target_module] - current_score.criteria_scores[target_module]
            total_delta = new_score.total_score - current_score.total_score

            self.logger.info(
                f"[Phase2-{i+1}] Score: {current_score.total_score:.2f} → {new_score.total_score:.2f} "
                f"({target_module}: {module_delta:+.2f})"
            )

            # 10. Record iteration
            iterations.append(RefineIteration(
                iteration=i + 1,
                module=target_module,
                module_score_before=current_score.criteria_scores[target_module],
                module_score_after=new_score.criteria_scores[target_module],
                score_before=current_score.total_score,
                score_after=new_score.total_score,
                improvements_made=plan.changes,
                strategy=plan.approach.value,
                diagnostics={"diagnostic": diagnostic, "plan": plan},
                text_before=current_text,
                text_after=improved_text
            ))

            # 11. Stagnation detection
            if total_delta < 0.05:
                stagnation_count += 1
                self.logger.warning(f"[Phase2-{i+1}] Stagnation count: {stagnation_count}/{self.stagnation_threshold}")

                if stagnation_count >= self.stagnation_threshold:
                    self.logger.warning("[Phase2] Stagnation detected, triggering escalation")

                    # Try section rewrite as last resort
                    if plan.approach != RefinementApproach.SECTION_REWRITE:
                        improved_text = self._escalate_to_rewrite(refiner_input, target_module)
                        new_score = self.scorer.score(
                            improved_text, question, company_profile, evidence, qtype
                        )

                        if new_score.total_score > current_score.total_score:
                            self.logger.info("[Phase2] Escalation successful")
                            current_text = improved_text
                            current_score = new_score
                            stagnation_count = 0
                            continue

                    self.logger.info("[Phase2] Escalation failed, moving to next module")
                    current_text = improved_text
                    current_score = new_score
                    stagnation_count = 0
                    continue
            else:
                stagnation_count = 0

            # 12. Update state
            current_text = improved_text
            current_score = new_score

            # 13. Check exit criteria
            if current_score.total_score >= self.target_score:
                self.logger.info(f"[Phase2] ✅ Target {self.target_score} reached!")
                break

            # 14. Module good enough, move to next
            if new_score.criteria_scores[target_module] >= 9.0:
                self.logger.info(f"[Phase2] {target_module} reached 9.0, moving to next module")
                continue

        return current_text, current_score, iterations

    def _pick_target_module(self, score_result: Any) -> Optional[str]:
        """Pick lowest scoring CONTENT module (skip submission_ready, length_fit)"""
        content_modules = ["question_focus", "logic_flow", "specificity", "expression_quality"]
        scores = score_result.criteria_scores

        # Filter to modules below 9.0
        below_target = {m: scores[m] for m in content_modules if scores.get(m, 0) < 9.0}

        if not below_target:
            return None

        # Return lowest scoring
        return min(below_target, key=below_target.get)

    def _identify_strengths(self, score_result: Any) -> List[str]:
        """Identify what NOT to change"""
        strengths = []
        for module, score in score_result.criteria_scores.items():
            if score >= 9.0:
                strengths.append(f"{module} 모듈 유지 (현재 {score:.2f}점)")
        return strengths

    def _get_forbidden_changes(self, qtype: Any) -> List[str]:
        """Get forbidden changes based on question type"""
        base_forbidden = [
            "KB에 없는 새로운 사실 추가 금지",
            "존댓말 유지 필수",
            "1인칭 주어('저는', '나는') 사용 금지"
        ]
        return base_forbidden

    def _analyze_why_low_score(self, input: Phase2RefinerInput) -> Phase2DiagnosticResult:
        """COT Step 1: WHY - Analyze root causes of low score"""

        model = self.model_selector.get_model("refine_diagnostic")

        # 이전 시도 요약
        previous_summary = ""
        if input.previous_attempts:
            previous_summary = f"""
## 이전 시도 결과
- {len(input.previous_attempts)}번 시도했으나 점수 개선 없음
- 이전 접근법들: {', '.join(set(str(attempt.get('approach', 'unknown')) for attempt in input.previous_attempts[-3:]))}
- **중요**: 이전과 동일한 진단/접근법을 반복하지 마세요
"""

        prompt = f"""당신은 자기소개서 구조 품질 전문가입니다.

## 🎯 이번 반복의 목적
**{input.target_module} 모듈을 {input.target_module_score:.2f}점 → {input.target_module_goal}점으로 개선**
{previous_summary}

## 📊 현재 본문의 문제점
{input.scorer_rationale if input.scorer_rationale else f"{input.target_module} 점수가 낮습니다."}

## 📝 현재 본문 ({len(input.current_text)}자)
{input.current_text}

## 🔍 분석 과제

### 1. 근본 원인 파악 (최소 2가지)
본문에서 **구체적으로 인용**하며 분석하세요:
- **어느 부분**이 문제인가? (예: "2문단 전체", "도입부 첫 문장")
- **왜** 문제인가? (추상적 표현, 근거 부족, 논리 비약 등)
- **점수 영향**은? (예상 감점: -0.5점, -1.0점)

### 2. 개선 전략 선택
다음 중 **하나**를 선택하세요:
- `incremental_edit`: 특정 문장/표현만 수정 (문제가 국지적일 때)
- `section_rewrite`: 전체 문단 재작성 (구조적 문제일 때)
- `evidence_swap`: KB 근거 교체 필요 (현재 근거가 부적절할 때)

### 3. 부작용 예측
- 이 수정이 다른 모듈에 미칠 영향은?
- 현재 잘 되고 있는 부분: {', '.join(input.preserve_strengths[:3]) if input.preserve_strengths else '없음'}

출력 형식: 반드시 유효한 JSON만 출력하세요.

```json
{{
  "root_causes": [
    {{
      "issue": "문제 설명",
      "quote": "본문에서 인용",
      "score_impact": -0.8,
      "location": "위치 설명"
    }}
  ],
  "improvement_approach": "incremental_edit",
  "approach_rationale": "선택 이유",
  "side_effect_risks": [
    {{
      "affected_module": "모듈명",
      "risk": "위험 설명",
      "mitigation": "예방 방법"
    }}
  ],
  "requires_new_evidence": false,
  "evidence_need": {{
    "type": "근거 유형",
    "query": "검색 쿼리"
  }},
  "confidence": 0.7
}}
```

주의사항:
- 반드시 유효한 JSON만 출력 (설명 없이)
- improvement_approach는 "incremental_edit", "section_rewrite", "evidence_swap" 중 하나
- requires_new_evidence는 true 또는 false (따옴표 없이)
- score_impact는 음수 (예: -0.8)
- confidence는 0.0~1.0 사이 숫자
"""

        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.3
        )

        result_text = response.choices[0].message.content.strip()

        # Parse JSON response
        try:
            # Extract JSON from markdown code blocks if present
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()

            result_data = json.loads(result_text)

            # Convert to dataclass
            root_causes = [
                RootCause(**rc) for rc in result_data.get("root_causes", [])
            ]

            side_effect_risks = [
                SideEffectRisk(**ser) for ser in result_data.get("side_effect_risks", [])
            ]

            evidence_need = None
            if result_data.get("requires_new_evidence") and result_data.get("evidence_need"):
                evidence_need = EvidenceNeed(**result_data["evidence_need"])

            approach_str = result_data.get("improvement_approach", "incremental_edit")
            approach = RefinementApproach(approach_str)

            return Phase2DiagnosticResult(
                root_causes=root_causes,
                improvement_approach=approach,
                approach_rationale=result_data.get("approach_rationale", ""),
                side_effect_risks=side_effect_risks,
                requires_new_evidence=result_data.get("requires_new_evidence", False),
                evidence_need=evidence_need,
                confidence=result_data.get("confidence", 0.7)
            )

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse diagnostic JSON: {e}\nResponse: {result_text}")
            # Return minimal diagnostic
            return Phase2DiagnosticResult(
                root_causes=[],
                improvement_approach=RefinementApproach.INCREMENTAL_EDIT,
                approach_rationale="JSON 파싱 실패, 기본 전략 사용",
                side_effect_risks=[],
                requires_new_evidence=False,
                confidence=0.3
            )

    def _design_improvement_plan(
        self,
        input: Phase2RefinerInput,
        diagnostic: Phase2DiagnosticResult
    ) -> Phase2ImprovementPlan:
        """COT Step 2: WHAT - Design specific improvement plan"""

        model = self.model_selector.get_model("refine_planning")

        # 근본 원인을 번호와 함께 명확히 정리
        root_causes_str = "\n".join([
            f"{i+1}. **{rc.issue}** (영향: {rc.score_impact}점)\n"
            f"   위치: {rc.location}\n"
            f"   인용: \"{rc.quote[:100]}...\""
            for i, rc in enumerate(diagnostic.root_causes)
        ])

        prompt = f"""당신은 자기소개서 개선 계획 전문가입니다.

## 🎯 목표
**{input.target_module} 점수를 {input.target_module_score:.2f}점에서 {input.target_module_goal}점으로 올리기**

## 🔍 진단 결과
**선택된 접근법**: {diagnostic.improvement_approach.value}
**이유**: {diagnostic.approach_rationale}

## ❌ 발견된 문제점들
{root_causes_str}

## 📋 개선 계획 작성 지침

각 문제점에 대해 **구체적인 수정 계획**을 작성하세요:

### 필수 항목
1. **위치**: 정확히 어디를 수정할 것인가? (예: "2문단 두 번째 문장")
2. **작업**: replace (교체), delete (삭제), insert (삽입) 중 하나
3. **변경 전**: 현재 텍스트를 **정확히** 인용
4. **변경 후**: 구체적인 수정안 (추상적 지시 금지!)
5. **이유**: 왜 이 변경이 {input.target_module} 점수를 올리는가?

### 좋은 예시
```
location: "도입부 첫 문장"
action: "replace"
before: "다양한 경험을 통해 성장했습니다."
after: "3D 재구성 연구에서 SLAM 기반 포즈 추정 정확도를 15% 향상시킨 경험을 통해..."
rationale: "추상적 표현을 구체적 수치와 기술로 교체하여 specificity 점수 향상"
```

### 나쁜 예시 (하지 마세요)
```
before: "첫 문장"
after: "더 구체적으로 작성"  ← 이런 추상적 지시는 금지!
```

## 📝 현재 본문
{input.current_text}

## 💡 사용 가능한 KB 근거 (필요시 활용)
{input.available_evidence[:800]}...

출력 형식: 반드시 유효한 JSON만 출력하세요.

```json
{{
  "changes": [
    {{
      "location": "변경할 위치 (예: '2문단 첫 문장')",
      "action": "replace",
      "before": "현재 텍스트",
      "after": "수정 텍스트",
      "rationale": "변경 이유"
    }}
  ],
  "expected_score_delta": {{
    "target_module": 0.8,
    "total": 0.5
  }},
  "risk_level": "low"
}}
```

주의:
- 반드시 유효한 JSON 형식으로만 응답
- 중괄호, 따옴표, 쉼표 정확히 사용
- 숫자는 따옴표 없이 작성
- 문자열은 반드시 쌍따옴표 사용
"""

        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.3
        )

        result_text = response.choices[0].message.content.strip()

        try:
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()

            result_data = json.loads(result_text)

            changes = [ContentChange(**ch) for ch in result_data.get("changes", [])]

            risk_str = result_data.get("risk_level", "medium")
            risk_level = RiskLevel(risk_str)

            return Phase2ImprovementPlan(
                changes=changes,
                expected_score_delta=result_data.get("expected_score_delta", {}),
                risk_level=risk_level,
                approach=diagnostic.improvement_approach
            )

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse plan JSON: {e}")
            self.logger.warning(f"LLM Response (first 500 chars):\n{result_text[:500]}")

            # JSON 파싱 실패 시: section_rewrite로 폴백 (JSON 불필요)
            # 대신 진단 결과를 기반으로 직접 재작성 시도
            self.logger.info("[Phase2] JSON parsing failed, falling back to direct rewrite")

            return Phase2ImprovementPlan(
                changes=[],  # 빈 changes로, _rewrite_section 호출하도록 유도
                expected_score_delta={},
                risk_level=RiskLevel.HIGH,
                approach=RefinementApproach.SECTION_REWRITE  # 강제로 section_rewrite
            )

    def _apply_incremental_edits(
        self,
        input: Phase2RefinerInput,
        plan: Phase2ImprovementPlan
    ) -> str:
        """COT Step 3: HOW - Apply incremental edits"""

        model = self.model_selector.get_model("refine_execution")

        # 변경 사항을 단계별로 명확히 나열
        changes_str = "\n\n".join([
            f"### 변경 {i+1}: {ch.location}\n"
            f"**작업**: {ch.action}\n"
            f"**찾을 텍스트**: \"{ch.before[:150]}\"\n"
            f"**교체할 텍스트**: \"{ch.after[:150]}\"\n"
            f"**목적**: {ch.rationale}"
            for i, ch in enumerate(plan.changes)
        ])

        prompt = f"""당신은 자기소개서 편집 전문가입니다.

## 🎯 작업 목표
{input.target_module} 점수를 {input.target_module_score:.2f}점에서 {input.target_module_goal}점으로 올리기

⚠️ **Phase 2의 핵심**: 글자 수 유지하면서 품질 개선
- 현재 {len(input.current_text)}자 → 유지 (분량 확장 금지)
- 지정된 변경 사항만 적용, 불필요한 추가 금지

## ✏️ 적용할 변경 사항 ({len(plan.changes)}개)

{changes_str}

## 🚫 절대 금지 사항
1. **지정되지 않은 부분 수정 금지** - 변경 목록에 없는 부분은 절대 건드리지 마세요
2. **새로운 사실 추가 금지** - KB에 없는 내용을 창작하지 마세요
3. **1인칭 표현 금지** - "저는", "나는", "제가" 사용 불가
4. **존댓말 유지** - 문장 종결은 "~습니다", "~ㅂ니다" 형태 유지
5. **한글 중심 작성** - 영어 문장 절대 금지 (기술 용어는 영어 허용)

## 📝 현재 본문
{input.current_text}

## 📤 출력 형식
- 수정된 전체 본문만 출력하세요
- 설명, 주석, 마크다운 없이 순수 텍스트만
- 변경 사항을 정확히 적용한 결과
"""

        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.2
        )

        return response.choices[0].message.content.strip()

    def _rewrite_section(
        self,
        input: Phase2RefinerInput,
        plan: Phase2ImprovementPlan
    ) -> str:
        """COT Step 3: HOW - Rewrite entire section"""

        model = self.model_selector.get_model("refine_execution")

        # 재작성 대상과 목표 결정
        target_section = plan.changes[0].location if plan.changes else "전체 본문"
        rewrite_goal = plan.changes[0].rationale if plan.changes else f"{input.target_module} 모듈 점수 향상"

        prompt = f"""당신은 자기소개서 재작성 전문가입니다.

## 🎯 재작성 목표
**{input.target_module} 점수를 {input.target_module_score:.2f}점 → {input.target_module_goal}점으로 올리기**

⚠️ **Phase 2의 핵심**: 글자 수를 늘리지 않고 품질만 개선
- 현재 {len(input.current_text)}자 → 목표 범위 유지 ({input.target_char_min}~{input.target_char_max}자)
- 분량 확장 금지, 문장 정제와 표현 개선에만 집중

## 📍 재작성 대상
{target_section}

## 💡 개선 방향
{rewrite_goal}

## 📝 현재 본문 ({len(input.current_text)}자)
{input.current_text}

## 🔧 구체적 지침

### {input.target_module} 점수를 올리려면:
{'- 추상적 표현을 구체적 사례/수치로 교체\n- KB 근거에서 구체적인 프로젝트 경험 활용\n- "다양한", "여러" 같은 모호한 표현 제거' if input.target_module == 'specificity' else ''}
{'- 질문이 요구하는 핵심 내용에 집중\n- 불필요한 배경 설명 최소화\n- 직접적으로 답변하는 구조' if input.target_module == 'question_focus' else ''}
{'- 문단 간 논리적 연결 강화\n- "따라서", "이를 통해" 등 연결어 활용\n- 시간/인과 순서 명확히' if input.target_module == 'logic_flow' else ''}
{'- 전문적이고 세련된 표현 사용\n- 문장 구조 다양화\n- 불필요한 반복 제거' if input.target_module == 'expression_quality' else ''}

### 반드시 유지해야 할 것
- {', '.join(input.preserve_strengths[:3]) if input.preserve_strengths else '현재 잘 작성된 부분'}
- 존댓말 어투 ("~습니다", "~ㅂ니다")
- 글자수: {input.target_char_min}~{input.target_char_max}자 범위

### 절대 금지
- 1인칭 주어 사용 ("저는", "나는", "제가")
- KB에 없는 새로운 사실 창작
- 다른 모듈 점수를 떨어뜨리는 수정
- 영어 문장 사용 (기술 용어는 영어 허용)

## 💼 활용 가능한 KB 근거
{input.available_evidence[:1000]}

## 📤 출력 형식
- 재작성된 전체 본문만 출력
- 설명, 주석 없이 순수 텍스트만
- {input.target_char_min}~{input.target_char_max}자 준수
"""

        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.3
        )

        return response.choices[0].message.content.strip()

    def _make_plan_conservative(self, plan: Phase2ImprovementPlan) -> Phase2ImprovementPlan:
        """Make a risky plan more conservative"""
        # Reduce number of changes
        if len(plan.changes) > 2:
            plan.changes = plan.changes[:2]

        plan.risk_level = RiskLevel.MEDIUM
        return plan

    def _escalate_to_rewrite(self, input: Phase2RefinerInput, target_module: str) -> str:
        """Escalate to full section rewrite as last resort"""

        self.logger.info(f"[Phase2] Escalating {target_module} to section rewrite")

        model = self.model_selector.get_model("refine_execution")

        prompt = f"""## 긴급 재작성 (Escalation)

여러 시도에도 {target_module} 점수가 개선되지 않았습니다.
전체 본문을 {target_module} 관점에서 전면 재작성하세요.

**현재 점수**: {input.target_module_score:.2f}
**목표 점수**: {input.target_module_goal}

**사용 가능한 근거**:
{input.available_evidence[:1500]}

**현재 본문**:
{input.current_text}

**재작성 방향**:
- {target_module}에 집중하여 근본적으로 개선
- 구체적 사례와 근거 활용
- 논리적 흐름 명확히
- 다른 모듈 점수는 유지

출력: 재작성된 본문 (텍스트만)
"""

        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.4
        )

        return response.choices[0].message.content.strip()

    def _refresh_evidence(self, need: EvidenceNeed, question: str) -> str:
        """Re-query KB with refined search"""

        self.logger.info(f"[Phase2] Refreshing evidence: {need.type}")

        new_query = f"{question} {need.type} {need.query}"
        hits = self.kb_searcher.search(new_query, top_k=6)

        # Format evidence
        evidence_parts = []
        for i, hit in enumerate(hits, 1):
            evidence_parts.append(f"[근거 {i}] {hit.get('text', '')[:500]}")

        return "\n\n".join(evidence_parts)
