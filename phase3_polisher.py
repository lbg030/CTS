"""
Phase 3: Final Convergence (9.5+ submission quality)

This module implements targeted polishing focusing on format cleanup,
expression quality, and length precision.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from schemas import Phase3PolisherInput, RefineIteration


class Phase3FinalPolisher:
    """
    Final polisher focusing on achieving 9.5+ submission quality.

    Focus areas:
    1. Format cleanup (submission_ready): Remove STAR/bullets/meta phrases
    2. Expression polish (expression_quality): Eliminate redundancy
    3. Length precision (length_fit): Fine-tune to 950-1000 chars
    """

    def __init__(self, client, cfg: Dict, logger: logging.Logger, scorer, model_selector):
        self.client = client
        self.cfg = cfg
        self.logger = logger
        self.scorer = scorer
        self.model_selector = model_selector

        phase3_cfg = cfg.get("phase3", {})
        self.max_iterations = phase3_cfg.get("max_iterations", 5)
        self.target_score = phase3_cfg.get("target_score", 9.5)
        self.strict_scope_limit = phase3_cfg.get("strict_scope_limit", True)

    def refine(
        self,
        text: str,
        score_result: Any,
        question: str,
        company_profile: Dict,
        evidence: str,
        qtype: Any,
        constraints: Dict,
        phase2_final_score: float
    ) -> Tuple[str, Any, List[RefineIteration]]:
        """
        Phase 3: Final polish to 9.5+

        Args:
            phase2_final_score: Baseline to not fall below

        Returns:
            (polished_text, final_score, iteration_history)
        """

        if score_result.total_score >= self.target_score:
            self.logger.info(f"[Phase3] Already meets target {self.target_score}, skipping")
            return text, score_result, []

        self.logger.info(f"[Phase3] Starting polish from {score_result.total_score:.2f} to {self.target_score}")

        current_text = text
        current_score = score_result
        iterations = []

        for i in range(self.max_iterations):
            self.logger.info(f"[Phase3-{i+1}/{self.max_iterations}] Starting iteration")

            # 1. Diagnose specific issues
            issues = self._diagnose_final_issues(current_text, current_score, constraints)

            if not issues:
                self.logger.info("[Phase3] No fixable issues found")
                break

            primary_issue = issues[0]
            self.logger.info(f"[Phase3-{i+1}] Fixing: {primary_issue['type']} (module: {primary_issue['module']})")

            # 2. Build polisher input
            polisher_input = Phase3PolisherInput(
                question=question,
                current_text=current_text,
                current_score=current_score.total_score,
                score_breakdown={k: v for k, v in current_score.criteria_scores.items()},
                failing_modules=[m for m, s in current_score.criteria_scores.items() if s < self.target_score],
                primary_target=primary_issue['module'],
                format_violations=primary_issue.get('violations', []),
                redundant_phrases=primary_issue.get('redundancies', []),
                length_delta=self._calc_length_delta(current_text, constraints),
                locked_content=self._get_locked_content(current_text),
                iteration_num=i + 1,
                phase2_final_score=phase2_final_score
            )

            # 3. Apply targeted fixes
            polished_text = self._apply_targeted_fixes(polisher_input)

            # 4. Re-score
            self.logger.info(f"[Phase3-{i+1}] Re-scoring polished text")
            new_score = self.scorer.score(
                polished_text, question, company_profile, evidence, qtype
            )

            # 5. Safety check: don't accept if score drops below Phase 2 baseline
            if new_score.total_score < phase2_final_score - 0.1:
                self.logger.warning(
                    f"[Phase3-{i+1}] Score dropped below Phase 2 baseline "
                    f"({new_score.total_score:.2f} < {phase2_final_score:.2f}), reverting"
                )
                continue

            # 6. Evaluate improvement
            total_delta = new_score.total_score - current_score.total_score
            self.logger.info(
                f"[Phase3-{i+1}] Score: {current_score.total_score:.2f} → {new_score.total_score:.2f} "
                f"({total_delta:+.2f})"
            )

            # 7. Record iteration
            iterations.append(RefineIteration(
                iteration=i + 1,
                module=primary_issue['module'],
                module_score_before=current_score.criteria_scores.get(primary_issue['module'], 0),
                module_score_after=new_score.criteria_scores.get(primary_issue['module'], 0),
                score_before=current_score.total_score,
                score_after=new_score.total_score,
                improvements_made=[primary_issue],
                strategy="targeted_fix",
                diagnostics={"issue": primary_issue},
                text_before=current_text,
                text_after=polished_text
            ))

            # 8. Update state
            current_text = polished_text
            current_score = new_score

            # 9. Check exit
            if current_score.total_score >= self.target_score:
                self.logger.info(f"[Phase3] ✅ Target {self.target_score} reached!")
                break

            # 10. Diminishing returns check
            if i > 0 and total_delta < 0.03:
                self.logger.info("[Phase3] Diminishing returns, stopping")
                break

        return current_text, current_score, iterations

    def _diagnose_final_issues(
        self,
        text: str,
        score: Any,
        constraints: Dict
    ) -> List[Dict]:
        """Identify specific, fixable issues in priority order"""

        issues = []

        # 1. Format violations (highest priority)
        if score.criteria_scores.get("submission_ready", 10) < 10:
            violations = self._detect_format_violations(text)
            if violations:
                issues.append({
                    "type": "format_cleanup",
                    "module": "submission_ready",
                    "violations": violations,
                    "fix_type": "remove_patterns"
                })

        # 2. Length issues
        if score.criteria_scores.get("length_fit", 10) < 10:
            delta = self._calc_length_delta(text, constraints)
            if abs(delta) > 20:
                issues.append({
                    "type": "length_adjustment",
                    "module": "length_fit",
                    "delta": delta,
                    "fix_type": "expand" if delta > 0 else "compress"
                })

        # 3. Expression quality
        if score.criteria_scores.get("expression_quality", 10) < 9.5:
            redundancies = self._detect_redundancies(text)
            if redundancies:
                issues.append({
                    "type": "expression_polish",
                    "module": "expression_quality",
                    "redundancies": redundancies,
                    "fix_type": "refine_expressions"
                })

        return issues

    def _detect_format_violations(self, text: str) -> List[str]:
        """Detect format violations like STAR, bullets, meta phrases"""

        violations = []

        # STAR markers
        star_patterns = [
            r'\b(상황|과제|행동|결과)\s*[:：]',
            r'\bSTAR\b',
            r'\b(Situation|Task|Action|Result)\b'
        ]
        for pattern in star_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append(f"STAR 패턴 감지: {pattern}")

        # Bullet points
        if re.search(r'^\s*[-*•]\s', text, re.MULTILINE):
            violations.append("불릿 포인트 사용")

        # Meta phrases
        meta_phrases = [
            r'자기소개서',
            r'지원서',
            r'본문에서는',
            r'이하에서는',
            r'다음과 같이'
        ]
        for phrase in meta_phrases:
            if re.search(phrase, text):
                violations.append(f"메타 표현: {phrase}")

        return violations

    def _detect_redundancies(self, text: str) -> List[str]:
        """Detect redundant expressions"""

        redundancies = []

        # Common redundant patterns
        patterns = [
            (r'할 수 있었습니다\..*?할 수 있었습니다', '반복: "할 수 있었습니다"'),
            (r'~하였습니다\..*?~하였습니다', '반복: "~하였습니다"'),
            (r'경험.*?경험.*?경험', '반복: "경험"'),
            (r'프로젝트.*?프로젝트.*?프로젝트', '반복: "프로젝트"'),
        ]

        for pattern, desc in patterns:
            if re.search(pattern, text):
                redundancies.append(desc)

        # Check for repeated sentence structures
        sentences = text.split('.')
        if len(sentences) >= 3:
            # Simple heuristic: check if many sentences start with same word
            first_words = [s.strip().split()[0] if s.strip() and s.strip().split() else "" for s in sentences]
            word_counts = {}
            for word in first_words:
                if word and len(word) > 1:
                    word_counts[word] = word_counts.get(word, 0) + 1

            for word, count in word_counts.items():
                if count >= 3:
                    redundancies.append(f'반복된 문장 시작: "{word}" ({count}회)')

        return redundancies

    def _calc_length_delta(self, text: str, constraints: Dict) -> int:
        """Calculate how many chars to add/remove"""

        current_len = len(text)
        target_min = constraints.get("target_char_min", 950)
        target_max = constraints.get("target_char_max", 1000)

        if current_len < target_min:
            return target_min - current_len  # Positive: need to add
        elif current_len > target_max:
            return target_max - current_len  # Negative: need to remove
        else:
            return 0

    def _get_locked_content(self, text: str) -> List[str]:
        """Identify content that should not be changed"""

        # Simple heuristic: preserve opening and closing sentences
        sentences = text.split('.')
        locked = []

        if sentences:
            locked.append(f"오프닝: {sentences[0][:50]}...")

        if len(sentences) > 1:
            locked.append(f"클로징: {sentences[-1][-50:]}...")

        return locked

    def _apply_targeted_fixes(self, input: Phase3PolisherInput) -> str:
        """Apply precise, scoped fixes (NOT full rewrite)"""

        model = self.model_selector.get_model("refine_polish")

        prompt = f"""다음 자기소개서를 최종 제출 품질로 마무리하세요.

## 현재 상태
- 점수: {input.current_score:.2f}/10 (목표: {self.target_score}+)
- 주요 이슈: {input.primary_target}
- 현재 글자 수: {len(input.current_text)}자

⚠️ **Phase 3의 핵심**: 글자 수를 유지하면서 표현만 다듬기
- 분량 확장 금지 (Phase 1에서 이미 분량 확보 완료)
- 문장 정제, 중복 제거, 표현 개선에만 집중
- 목표 범위(950~1000자) 내에서 미세 조정만 허용

## 수정 사항 (이것만 정확히 수정)

"""

        if input.format_violations:
            prompt += f"""
### 1. 형식 오류 제거
다음 패턴을 삭제하세요:
{chr(10).join(f"- {v}" for v in input.format_violations)}
"""

        if input.redundant_phrases:
            prompt += f"""
### 2. 중복 표현 정리
{chr(10).join(f"- {r}" for r in input.redundant_phrases)}
→ 다양한 표현으로 교체하거나 불필요한 반복 제거
"""

        if abs(input.length_delta) > 50:
            action = "압축" if input.length_delta < 0 else "미세 조정"
            prompt += f"""
### 3. 길이 미세 조정
현재 {len(input.current_text)}자 → 목표 950-1000자 (약 {input.length_delta:+d}자 {action})
- ⚠️ 중요: 분량을 크게 늘리지 마세요 (Phase 1에서 이미 확보됨)
- 중복/군더더기 제거로 자연스럽게 조정
- 핵심 내용은 유지
"""

        prompt += f"""

## 절대 금지
- 새로운 내용 추가 금지
- 구조 변경 금지
- 영어 문장 사용 금지 (기술 용어는 영어 허용)
- 다음 내용은 반드시 유지: {input.locked_content}
- Phase 2 점수({input.phase2_final_score:.2f}) 아래로 떨어뜨리지 말 것

## 원본
{input.current_text}

출력: 수정된 본문 (텍스트만, 설명 없이)
"""

        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.2
        )

        return response.choices[0].message.content.strip()
