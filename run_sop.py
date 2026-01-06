#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_sop.py - 합격 자소서 패턴 기반 자기소개서 생성 (v2.0 품질 개선)

개선사항:
  R1) 1인칭 표현 금지 - FirstPersonRemover
  R2) 스코어링 + Refine 루프 - QualityScorer, RefineLoop
  R5) 문체/흐름 반영 강화 - 프롬프트 강화

핵심 원칙:
  1. 사람 중심 서술 (성향 → 관심 → 경험 → 직무 연결)
  2. 질문 유형별 구조 적용
  3. KB 근거는 지정된 위치에서만 사용
  4. "연구 보고서"가 아닌 "자기소개서"
  5. 1인칭 주어 반복 제거

사용법:
    python run_sop.py --config config.yaml
"""

import os
import re
import json
import time
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from enum import Enum
from dataclasses import dataclass, asdict, field

import yaml
from openai import OpenAI

from query_kb import KBSearcher


# ============================================================
# 질문 유형 분류
# ============================================================

class QuestionType(Enum):
    SELF_INTRO = "자기소개"
    MOTIVATION = "지원동기"
    COMPETENCY = "직무역량"
    TEAMWORK = "협업/갈등"
    GROWTH = "성장/도전"
    OTHER = "기타"


class QuestionClassifier:
    """질문 유형 분류기"""
    
    PATTERNS = {
        QuestionType.SELF_INTRO: [
            "자기소개", "자유롭게", "소개해", "어떤 사람", "본인을", "자신을",
        ],
        QuestionType.MOTIVATION: [
            "지원동기", "지원 동기", "왜 지원", "선택한 이유", "입사하고 싶",
        ],
        QuestionType.COMPETENCY: [
            "역량", "강점", "능력", "전문성", "기술", "경험을 바탕",
        ],
        QuestionType.TEAMWORK: [
            "협업", "팀워크", "갈등", "의견 충돌", "협력", "소통",
        ],
        QuestionType.GROWTH: [
            "성장", "도전", "실패", "극복", "배운 점", "변화",
        ],
    }
    
    GUIDELINES = {
        QuestionType.SELF_INTRO: {
            "structure": [
                "나의 성향·가치관·관심사 (도입)",
                "성향이 직무/분야와 연결된 이유",
                "대표 경험 1~2개 (여기서만 KB 사용)",
                "경험이 사고방식에 준 영향",
                "회사/직무 연결 (마무리)",
            ],
            "allowed": ["성향", "가치관", "관심", "태도", "사고방식"],
            "limited": ["실적 1~2개만 보조적으로"],
            "forbidden": ["성과 나열", "논문 리스트", "수치 나열"],
            "tone": "나는 어떤 사람인가를 중심으로",
        },
        QuestionType.MOTIVATION: {
            "structure": [
                "회사/직무에 관심 갖게 된 계기",
                "나의 경험/역량과 연결점",
                "구체적 기여 방향",
            ],
            "allowed": ["회사 관심 이유", "적합성", "비전"],
            "limited": ["관련 경험"],
            "forbidden": ["일반적 칭찬", "뻔한 표현", "과도한 자기 PR"],
            "tone": "왜 이 회사인가를 중심으로",
        },
        QuestionType.COMPETENCY: {
            "structure": [
                "핵심 역량 제시",
                "역량을 보여주는 경험 (문제-해결-결과)",
                "해당 역량의 직무 적용 방향",
            ],
            "allowed": ["문제해결 사례", "구체적 상황"],
            "limited": ["정량 성과"],
            "forbidden": ["기술 스택 나열", "논문 리스트"],
            "tone": "어떻게 문제를 해결하는가를 중심으로",
        },
        QuestionType.TEAMWORK: {
            "structure": [
                "협업/갈등 상황 설명",
                "나의 행동과 역할",
                "결과와 배운 점",
            ],
            "allowed": ["상황-행동-결과", "소통 방식"],
            "limited": ["팀 규모/역할"],
            "forbidden": ["자기 PR 위주", "남 탓"],
            "tone": "어떻게 함께 일하는가를 중심으로",
        },
        QuestionType.GROWTH: {
            "structure": [
                "도전/실패 상황",
                "극복 과정",
                "성장과 변화",
            ],
            "allowed": ["솔직한 실패", "배운 점", "변화"],
            "limited": ["결과 수치"],
            "forbidden": ["실패 미화", "자랑"],
            "tone": "어떻게 성장했는가를 중심으로",
        },
        QuestionType.OTHER: {
            "structure": ["질문에 맞는 자유 구조"],
            "allowed": ["질문 의도에 맞는 내용"],
            "limited": ["실적"],
            "forbidden": ["질문과 무관한 내용"],
            "tone": "질문에 직접 답변",
        },
    }
    
    @classmethod
    def classify(cls, question: str) -> QuestionType:
        q_lower = question.lower()
        for qtype, patterns in cls.PATTERNS.items():
            if any(p in q_lower for p in patterns):
                return qtype
        return QuestionType.OTHER
    
    @classmethod
    def get_guideline(cls, qtype: QuestionType) -> Dict:
        return cls.GUIDELINES.get(qtype, cls.GUIDELINES[QuestionType.OTHER])


# ============================================================
# R1) 1인칭 표현 제거기
# ============================================================

class FirstPersonRemover:
    """1인칭 표현을 제거하고 자연스러운 문장으로 변환"""
    
    def __init__(self, cfg: Dict):
        style_rules = cfg.get("style_rules", {})
        self.forbidden = style_rules.get("forbidden_first_person", [
            "저는", "나는", "제가", "내가", "저의", "나의"
        ])
        self.max_count = style_rules.get("max_first_person_count", 0)
    
    def count_violations(self, text: str) -> int:
        """1인칭 표현 개수 카운트"""
        count = 0
        for word in self.forbidden:
            count += text.count(word)
        return count
    
    def remove(self, text: str) -> str:
        """1인칭 표현 제거 (단순 제거 후 정리)"""
        result = text
        
        # 패턴별 제거
        patterns = [
            (r"저는\s+", ""),
            (r"나는\s+", ""),
            (r"제가\s+", ""),
            (r"내가\s+", ""),
            (r"저의\s+", ""),
            (r"나의\s+", ""),
            # 문장 시작 처리
            (r"^\s*저는\s*", ""),
            (r"^\s*나는\s*", ""),
        ]
        
        for pattern, replacement in patterns:
            result = re.sub(pattern, replacement, result, flags=re.MULTILINE)
        
        # 중복 공백 제거
        result = re.sub(r"\s{2,}", " ", result)
        result = re.sub(r"\n\s*\n", "\n\n", result)
        
        return result.strip()
    
    def get_violation_report(self, text: str) -> Dict:
        """위반 상세 리포트"""
        violations = {}
        for word in self.forbidden:
            count = text.count(word)
            if count > 0:
                violations[word] = count
        
        return {
            "total_count": sum(violations.values()),
            "violations": violations,
            "passed": sum(violations.values()) <= self.max_count
        }


# ============================================================
# R2) 품질 스코어링
# ============================================================

@dataclass
class ScoreResult:
    """스코어링 결과

    rationales: 각 평가 모듈별 상세 근거
                QualityScorer.score()에서 항상 제공되어야 함
                Phase 2/3 refinement에서 사용
    """
    total_score: float
    criteria_scores: Dict[str, float]
    passed: bool
    failed_criteria: List[str]
    recommendations: List[str]
    rationales: Dict[str, str] = field(default_factory=dict)


class QualityScorer:
    """자기소개서 품질 스코어링 (모듈 기반)"""
    
    def __init__(self, client: OpenAI, cfg: Dict, logger: logging.Logger, model_selector):
        self.client = client
        self.cfg = cfg
        self.logger = logger
        self.model_selector = model_selector
        
        scoring_cfg = cfg.get("scoring", {})
        self.modules = scoring_cfg.get("modules") or scoring_cfg.get("criteria", {})
        self.module_order = scoring_cfg.get("module_order") or list(self.modules.keys())
        self.pass_threshold = scoring_cfg.get("pass_threshold", 7.0)
        self.strict_pass = scoring_cfg.get("strict_pass", False)
        self.enforce_min = scoring_cfg.get("enforce_min_scores", True)
        app_cfg = cfg.get("application", {})
        self.target_min = int(app_cfg.get("target_char_min", 950))
        self.target_max = int(app_cfg.get("target_char_max", 1000))
        
        self.model = model_selector.get_model("scorer")
        openai_cfg = cfg.get("openai", {})
        self.retry_cfg = openai_cfg.get("retry", {"max_retries": 2, "base_sleep_sec": 1.0})
        self.max_tokens = openai_cfg.get("max_tokens", {}).get("scorer", 800)
        
        self.first_person_remover = FirstPersonRemover(cfg)
        self.meta_phrases = [
            "이 문단에서는", "다음으로", "아래는", "위에서", "이 글에서는",
            "본 문항", "이번 문항", "다음 문단", "첫째", "둘째", "셋째", "마지막으로",
        ]
    
    def diagnose(self, text: str) -> Dict[str, Any]:
        body = self._extract_body(text)
        length = char_len(body)
        status = "ok" if self.target_min <= length <= self.target_max else ("short" if length < self.target_min else "long")
        violations = self._detect_submission_violations(body)
        fp_report = self.first_person_remover.get_violation_report(body)
        paragraphs = [p for p in body.split("\n") if p.strip()]
        return {
            "length": length,
            "target_min": self.target_min,
            "target_max": self.target_max,
            "length_status": status,
            "length_delta": (self.target_min + self.target_max) // 2 - length,
            "submission_violations": violations,
            "first_person_violations": fp_report.get("violations", {}),
            "paragraph_count": len(paragraphs),
        }
    
    def _extract_body(self, text: str) -> str:
        if "[Self-Scoring]" in text:
            return text.split("[Self-Scoring]")[0].strip()
        return text.strip()

    def _detect_submission_violations(self, body: str) -> List[str]:
        violations = set()
        if re.search(r"(?m)^\s*[\-\*\u2022]\s+", body):
            violations.add("bullet_list")
        if re.search(r"(?m)^\s*\(?\d+\)?\s*[\.\)]\s+", body):
            violations.add("numbered_list")
        if re.search(r"(?m)^\s*\*?\s*[STAR]\s*:", body):
            violations.add("star_markers")
        if re.search(r"(?i)\bR-STAR\b|\bSTAR\b", body):
            violations.add("star_terms")
        if re.search(r"한 줄 결론|소제목", body):
            violations.add("structure_terms")
        if re.search(r"(?m)^\s*\[(?!내용 보강 필요\]).*\]\s*$", body):
            violations.add("bracket_heading")
        if "[Self-Scoring]" in body:
            violations.add("self_scoring")
        for phrase in self.meta_phrases:
            if phrase in body:
                violations.add(f"meta:{phrase}")
        fp_report = self.first_person_remover.get_violation_report(body)
        if not fp_report.get("passed", True):
            violations.add("first_person")

        # 영어 문장 체크 (기술 용어는 제외)
        # 3단어 이상의 연속된 영어 단어는 영어 문장으로 간주
        if re.search(r'\b[A-Za-z]+\s+[A-Za-z]+\s+[A-Za-z]+\b', body):
            violations.add("english_sentence")

        return sorted(violations)
    
    def _score_length(self, body: str) -> float:
        count = char_len(body)
        return 10.0 if self.target_min <= count <= self.target_max else 0.0
    
    def _score_submission_ready(self, body: str) -> float:
        return 10.0 if not self._detect_submission_violations(body) else 0.0

    def _build_scorer_prompt(self) -> str:
        return """역할: Quality Scorer (제출용 자기소개서 평가)

평가 대상은 질문에 직접 답하는 제출용 본문이다. 다음 모듈을 0~10점으로 평가한다.
- question_focus: 질문 의도에 직접 답하고 불필요한 내용이 없는가
- logic_flow: 도입-근거-의미-마무리의 흐름이 자연스러운가
- specificity: 구체적 행동/상황/결과로 설명되어 추상적 표현이 적은가
- expression_quality: 문장이 명확하고 매끄러우며 존댓말이 유지되는가

규칙:
- 형식/메타 금지 여부는 별도 모듈로 평가하므로, 여기서는 내용 완성도 중심으로 판단한다.
- 과장/추측/근거 없는 일반론은 감점한다.
- 점수는 엄격하게 부여하고, 9.5 이상은 매우 뛰어난 경우에만 부여한다.

출력 JSON:
{
  "scores": {
    "question_focus": 8.5,
    "logic_flow": 8.0,
    "specificity": 7.5,
    "expression_quality": 8.2
  },
  "rationales": {
    "question_focus": "짧은 이유",
    "logic_flow": "짧은 이유",
    "specificity": "짧은 이유",
    "expression_quality": "짧은 이유"
  }
}"""

    def _score_content_modules(self, text: str, question: str, qtype: QuestionType) -> Tuple[Dict[str, float], Dict[str, str]]:
        prompt = self._build_scorer_prompt()
        payload = {"question": question, "text": text, "question_type": qtype.value}
        try:
            result = call_agent_json(
                self.client, self.model, prompt, payload,
                self.max_tokens, self.retry_cfg, self.logger, "Quality-Scorer",
                self.model_selector
            )
        except Exception as e:
            self.logger.warning("스코어러 호출 실패: %s", e)
            return {}, {}
        if is_error_payload(result):
            return {}, {}
        scores = result.get("scores", {}) if isinstance(result, dict) else {}
        rationales = result.get("rationales", {}) if isinstance(result, dict) else {}
        return scores, rationales

    def _clamp_score(self, value: Any) -> float:
        try:
            v = float(value)
        except Exception:
            v = 0.0
        return max(0.0, min(10.0, round(v, 2)))
    
    def score(self, text: str, question: str, company_profile: Dict, 
              evidence: str, qtype: QuestionType) -> ScoreResult:
        """텍스트 품질 스코어링"""
        
        body = self._extract_body(text)
        llm_scores, rationales = self._score_content_modules(body, question, qtype)
        scores = {k: self._clamp_score(v) for k, v in llm_scores.items()}
        scores["submission_ready"] = self._score_submission_ready(body)
        scores["length_fit"] = self._score_length(body)
        
        # 가중 평균 계산
        total = 0.0
        weight_sum = 0.0
        failed = []

        ordered_modules = [m for m in self.module_order if m in self.modules]
        for m in self.modules:
            if m not in ordered_modules:
                ordered_modules.append(m)
        for name in ordered_modules:
            if name not in scores:
                scores[name] = 0.0
        
        for name in ordered_modules:
            info = self.modules.get(name, {})
            weight = info.get("weight", 0.1)
            min_score = info.get("min_score", 5)
            score = scores.get(name, 0.0)
            
            total += score * weight
            weight_sum += weight
            
            if self.enforce_min and score < min_score:
                failed.append(name)
        
        final_score = total / weight_sum if weight_sum > 0 else 0
        threshold_passed = final_score > self.pass_threshold if self.strict_pass else final_score >= self.pass_threshold
        passed = threshold_passed and len(failed) == 0
        
        # 개선 권고
        recommendations = self._generate_recommendations(scores, failed, rationales)

        return ScoreResult(
            total_score=round(final_score, 2),
            criteria_scores=scores,
            passed=passed,
            failed_criteria=failed,
            recommendations=recommendations,
            rationales=rationales  # Phase 2/3에서 사용
        )
    
    def _generate_recommendations(self, scores: Dict, failed: List[str], rationales: Dict[str, str]) -> List[str]:
        """개선 권고 생성"""
        suggestions = {
            "question_focus": "질문에 직접 답하는 문장을 앞부분에 배치하고 관련 없는 내용을 제거",
            "logic_flow": "도입-근거-의미-마무리 흐름이 자연스럽도록 문장 순서를 재정렬",
            "specificity": "추상 표현을 줄이고 행동/상황/결과를 구체적으로 보강",
            "expression_quality": "중복과 군더더기를 제거하고 문장을 명확하게 다듬기",
            "submission_ready": "소제목/번호/STAR/불릿/메타 문구를 제거하고 본문만 유지",
            "length_fit": f"글자수 {self.target_min}~{self.target_max}자 범위로 조정",
        }
        recs = []
        targets = failed if failed else sorted(scores.keys(), key=lambda k: scores.get(k, 0))[:2]
        for key in targets:
            if key in rationales and rationales.get(key):
                recs.append(f"{suggestions.get(key, key)} (근거: {rationales[key]})")
            else:
                recs.append(suggestions.get(key, key))
        return recs[:3]


# ============================================================
# R2) Refine 루프
# ============================================================

@dataclass
class RefineIteration:
    """Refine 반복 기록"""
    iteration: int
    module: str
    module_score_before: float
    module_score_after: float
    score_before: float
    score_after: float
    improvements_made: List[str]
    strategy: str
    diagnostics: Dict[str, Any]
    text_before: str
    text_after: str


class RefineLoop:
    """품질 미달 시 자동 개선 루프 (R2)"""

    MODULE_STRATEGIES = {
        "question_focus": [
            {"id": "direct_answer", "desc": "첫 문장에서 질문 의도에 바로 답하도록 재구성"},
            {"id": "trim_offtopic", "desc": "질문과 무관한 문장을 제거하고 핵심만 남김"},
            {"id": "role_anchor", "desc": "직무/회사 연결 문장을 앞쪽으로 끌어 질문 초점을 강화"},
        ],
        "logic_flow": [
            {"id": "reorder", "desc": "문장/문단 순서를 재배치해 흐름을 자연스럽게"},
            {"id": "causal_link", "desc": "원인-행동-결과 연결 문장을 보강"},
            {"id": "progression", "desc": "도입-근거-의미-마무리 단계로 재정렬"},
        ],
        "specificity": [
            {"id": "action_detail", "desc": "행동/상황/결과를 구체적으로 보강"},
            {"id": "replace_vague", "desc": "추상 표현을 구체적 행동으로 교체"},
            {"id": "mini_example", "desc": "짧은 구체 사례를 1~2문장 추가"},
        ],
        "expression_quality": [
            {"id": "clarify", "desc": "문장을 명확하고 간결하게 정리"},
            {"id": "remove_redundancy", "desc": "중복·군더더기 제거로 가독성 향상"},
            {"id": "tone_consistency", "desc": "존댓말 문체를 일관되게 유지"},
        ],
        "submission_ready": [
            {"id": "remove_structures", "desc": "소제목/번호/STAR/불릿/메타 문구 제거"},
            {"id": "merge_to_paragraph", "desc": "목록형 표현을 자연스러운 문장으로 통합"},
            {"id": "clean_meta", "desc": "안내 문구와 형식 표기를 삭제"},
        ],
        "length_fit": [
            {"id": "expand_detail", "desc": "짧다면 구체 디테일을 추가해 확장"},
            {"id": "compress", "desc": "길다면 중복을 줄여 압축"},
            {"id": "rebalance", "desc": "불균형한 문장 길이를 조정"},
        ],
    }
    
    def __init__(self, client: OpenAI, cfg: Dict, logger: logging.Logger,
                 scorer: QualityScorer, model_selector):
        self.client = client
        self.cfg = cfg
        self.logger = logger
        self.scorer = scorer
        self.model_selector = model_selector
        
        refine_cfg = cfg.get("refine_loop", {})
        self.enabled = refine_cfg.get("enabled", True)
        self.max_iterations = refine_cfg.get("max_iterations", 3)
        self.max_iterations_per_module = refine_cfg.get("max_iterations_per_module", self.max_iterations)
        self.max_total_iterations = refine_cfg.get(
            "max_total_iterations",
            self.max_iterations_per_module * max(1, len(self.scorer.module_order))
        )
        self.priority = refine_cfg.get("improvement_priority", []) or self.scorer.module_order
        self.save_history = refine_cfg.get("save_iteration_history", True)
        
        self.first_person_remover = FirstPersonRemover(cfg)
        self.history: List[RefineIteration] = []
        self.target_min = int(cfg.get("application", {}).get("target_char_min", 950))
        self.target_max = int(cfg.get("application", {}).get("target_char_max", 1000))

        self.model = model_selector.get_model("refiner")
        self.max_tokens = cfg.get("openai", {}).get("max_tokens", {}).get("refiner", 1800)

    def _module_strategies(self, module: str) -> List[Dict[str, str]]:
        return self.MODULE_STRATEGIES.get(module, [{"id": "focused", "desc": "해당 모듈만 집중 개선"}])

    def _select_strategy(self, module: str, state: Dict[str, Any]) -> Dict[str, str]:
        strategies = self._module_strategies(module)
        if state["no_improve"] >= 2:
            state["strategy_idx"] = (state["strategy_idx"] + 1) % len(strategies)
            state["no_improve"] = 0
        return strategies[state["strategy_idx"]]

    def _build_plan(self, module: str, diag: Dict[str, Any], strategy: Dict[str, str]) -> Dict[str, Any]:
        target_len = (self.target_min + self.target_max) // 2
        reasons = []
        if module == "length_fit":
            reasons.append(f"글자수 {diag.get('length_status')}")
        if module == "submission_ready":
            reasons.extend(diag.get("submission_violations", [])[:4])
        return {
            "module": module,
            "strategy": strategy.get("id", "focused"),
            "strategy_desc": strategy.get("desc", ""),
            "reasons": reasons[:4],
            "target_len": target_len,
            "delta": target_len - diag.get("length", 0),
        }

    def _apply_plan(self, text: str, diag: Dict[str, Any], plan: Dict[str, Any],
                    question: str, company_profile: Dict, evidence: str, qtype: QuestionType,
                    constraints: Dict) -> str:
        delta = plan.get("delta", 0)
        target_len = plan.get("target_len", (self.target_min + self.target_max) // 2)
        reasons = "\n".join(f"- {r}" for r in plan.get("reasons", [])) or "- 없음"

        prompt = f"""다음 자기소개서 본문을 개선하세요.

## 질문
{question}

## 목표 모듈
{plan.get("module")} (전략: {plan.get("strategy_desc")})

## 현재 글자수
{diag.get('length', 0)}자 (목표 {self.target_min}~{self.target_max}자, 목표치 {target_len}자, 변화량 {delta:+d}자)

## 수정 이유
{reasons}

## 필수 규칙
1. 질문에 직접 답하는 본문만 출력
2. 소제목/번호/STAR/불릿/메타 문구 금지
3. 존댓말(합니다/입니다) 유지
4. 1인칭 주어 금지
5. 제공되지 않은 수치/사실 가공 금지
6. 미세 수정 금지: 점수 상승이 가능한 방향으로 내용/구조를 명확히 변경
7. 전체를 새로 쓰지 말고 해당 모듈 관련 문장만 집중 수정

## 원본
{text}

수정된 자기소개서 본문만 출력 (설명 없이 텍스트만):"""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens
            )
            improved = resp.choices[0].message.content.strip()
            improved = self.first_person_remover.remove(improved)
            improved = normalize_submission_text(improved)
            return improved
        except Exception as e:
            self.logger.warning(f"개선 실패: {e}")
            return text

    def refine(self, text: str, score_result: ScoreResult, question: str,
               company_profile: Dict, evidence: str, qtype: QuestionType,
               constraints: Dict) -> Tuple[str, ScoreResult, List[RefineIteration]]:
        """
        텍스트 개선 루프 실행
        
        Returns:
            (개선된 텍스트, 최종 스코어, 반복 기록)
        """
        if not self.enabled or score_result.passed:
            return text, score_result, []
        
        current_text = normalize_submission_text(text)
        current_score = score_result
        iterations = []
        total_attempts = 0
        state = {m: {"attempts": 0, "no_improve": 0, "strategy_idx": 0} for m in self.priority}

        for i in range(self.max_total_iterations):
            target_module = self._pick_target_module(current_score)
            if not target_module:
                break
            if target_module not in state:
                state[target_module] = {"attempts": 0, "no_improve": 0, "strategy_idx": 0}
            if state[target_module]["attempts"] >= self.max_iterations_per_module:
                self.logger.warning("[Refine] 모듈 %s 개선 횟수 초과", target_module)
                break

            diag = self.scorer.diagnose(current_text)
            strategy = self._select_strategy(target_module, state[target_module])
            plan = self._build_plan(target_module, diag, strategy)
            self.logger.info(
                "[Refine %d/%d] module=%s score=%.2f len=%d delta=%+d strategy=%s reasons=%s",
                i + 1, self.max_total_iterations, target_module, current_score.total_score, diag.get("length", 0),
                plan.get("delta", 0), plan.get("strategy"),
                ",".join(plan.get("reasons", []))
            )
            
            improved_text = self._apply_plan(
                current_text, diag, plan, question, company_profile, evidence, qtype, constraints
            )
            
            new_score = self.scorer.score(improved_text, question, company_profile, evidence, qtype)
            module_before = current_score.criteria_scores.get(target_module, 0.0)
            module_after = new_score.criteria_scores.get(target_module, 0.0)
            
            iteration = RefineIteration(
                iteration=i + 1,
                module=target_module,
                module_score_before=module_before,
                module_score_after=module_after,
                score_before=current_score.total_score,
                score_after=new_score.total_score,
                improvements_made=plan.get("reasons", []),
                strategy=plan.get("strategy", "focused"),
                diagnostics=diag,
                text_before=current_text,
                text_after=improved_text
            )
            iterations.append(iteration)
            
            self.logger.info(
                "[Refine %d] %.2f → %.2f | %s %.2f → %.2f",
                i + 1, current_score.total_score, new_score.total_score,
                target_module, module_before, module_after,
            )
            
            if module_after > module_before:
                state[target_module]["no_improve"] = 0
            else:
                state[target_module]["no_improve"] += 1
            
            current_text = improved_text
            current_score = new_score
            state[target_module]["attempts"] += 1
            total_attempts += 1
            
            if current_score.passed:
                self.logger.info("[Refine] 품질 통과!")
                break
        
        return current_text, current_score, iterations

    def _pick_target_module(self, score_result: ScoreResult) -> Optional[str]:
        if score_result.failed_criteria:
            return score_result.failed_criteria[0]
        if not score_result.passed:
            scores = score_result.criteria_scores or {}
            ordered = self.priority or list(scores.keys())
            if not ordered:
                return None
            return min(ordered, key=lambda k: scores.get(k, 0))
        return None


# ============================================================
# 유틸리티
# ============================================================

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def clamp_text(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= max_chars else s[:max_chars].rstrip()


def read_jsonl(path: str, max_lines: int = 0) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def render_flow_guide(flow_guide: Dict[str, Any], max_items: int = 6) -> str:
    if not isinstance(flow_guide, dict):
        return ""
    lines = []
    common_flow = flow_guide.get("common_flow") or []
    if isinstance(common_flow, list) and common_flow:
        lines.append("공통 흐름:")
        for s in common_flow[:max_items]:
            if str(s).strip():
                lines.append(f"- {s}")
    section_roles = flow_guide.get("section_roles") or []
    if isinstance(section_roles, list) and section_roles:
        lines.append("섹션 역할:")
        for s in section_roles[:max_items]:
            if not isinstance(s, dict):
                continue
            name = str(s.get("name", "")).strip()
            role = str(s.get("role", "")).strip()
            pos = str(s.get("position", "")).strip()
            if name:
                label = f"{name} ({pos})" if pos else name
                lines.append(f"- {label}: {role}" if role else f"- {label}")
    tips = flow_guide.get("tips") or []
    if isinstance(tips, list) and tips:
        lines.append("작성 팁:")
        for t in tips[:max_items]:
            if str(t).strip():
                lines.append(f"- {t}")
    return "\n".join(lines).strip()


def summarize_structured_patterns(path: str, max_records: int = 80, top_n: int = 6) -> str:
    records = read_jsonl(path, max_lines=max_records)
    if not records:
        return ""
    
    flow_counter = Counter()
    section_counter = Counter()
    emphasis_counter = Counter()
    paragraph_counts = []
    
    for r in records:
        flow = r.get("flow", [])
        if isinstance(flow, list):
            for item in flow:
                if isinstance(item, str) and item.strip():
                    flow_counter[item.strip()] += 1
        sections = r.get("sections", [])
        if isinstance(sections, list):
            for s in sections:
                if not isinstance(s, dict):
                    continue
                name = str(s.get("name", "")).strip()
                pos = str(s.get("position", "")).strip()
                if name:
                    label = f"{name} ({pos})" if pos else name
                    section_counter[label] += 1
        emphasis = r.get("emphasis_positions", [])
        if isinstance(emphasis, list):
            for e in emphasis:
                if isinstance(e, str) and e.strip():
                    emphasis_counter[e.strip()] += 1
        pc = r.get("paragraph_count", None)
        if isinstance(pc, int):
            paragraph_counts.append(pc)
    
    lines = []
    lines.append(f"샘플 수: {len(records)}")
    if paragraph_counts:
        lines.append(
            f"문단 수 범위: {min(paragraph_counts)} ~ {max(paragraph_counts)}"
            f" (avg: {sum(paragraph_counts) / len(paragraph_counts):.1f})"
        )
    if emphasis_counter:
        parts = [f"{k}:{v}" for k, v in emphasis_counter.most_common(3)]
        lines.append("강조 위치 상위: " + ", ".join(parts))
    
    if flow_counter:
        lines.append("빈도 높은 흐름 요소:")
        for item, count in flow_counter.most_common(top_n):
            lines.append(f"- {item} ({count})")
    if section_counter:
        lines.append("자주 등장한 섹션:")
        for item, count in section_counter.most_common(top_n):
            lines.append(f"- {item} ({count})")
    
    return "\n".join(lines).strip()


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def char_len(s: str) -> int:
    return len(s or "")


def strip_self_scoring(text: str) -> str:
    if "[Self-Scoring]" in text:
        return text.split("[Self-Scoring]")[0].strip()
    return text.strip()


def normalize_submission_text(text: str) -> str:
    s = (text or "").replace("\r\n", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def slugify(name: str) -> str:
    s = (name or "").strip().lower()
    s = re.sub(r"[^a-z0-9가-힣]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "default"


def question_slug(question: str, max_len: int = 28) -> str:
    q = normalize_ws(question)
    q = re.sub(r"[\[\]\(\)\{\}<>\"'`]", "", q)
    q = re.sub(r"[^a-zA-Z0-9가-힣\s]+", " ", q)
    q = normalize_ws(q)[:max_len].strip()
    return slugify(q) if q else "question"


class StepTimer:
    def __init__(self, logger: logging.Logger, name: str):
        self.logger = logger
        self.name = name
        self.t0 = None

    def __enter__(self):
        self.t0 = time.time()
        self.logger.info("▶ %s", self.name)
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - self.t0
        if exc is None:
            self.logger.info("✓ %s (%.2fs)", self.name, dt)
        else:
            self.logger.error("✗ %s 실패 (%.2fs): %s", self.name, dt, exc)


# ============================================================
# 로깅
# ============================================================

def setup_logger(cfg: Dict[str, Any]) -> logging.Logger:
    log_cfg = cfg.get("logging", {})
    level = log_cfg.get("level", "INFO").upper()
    quiet = log_cfg.get("quiet", False)
    to_file = log_cfg.get("to_file", False)
    file_path = log_cfg.get("file_path", "logs/run.log")

    logger = logging.getLogger("sop")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING if quiet else getattr(logging, level, logging.INFO))
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    if to_file:
        ensure_dir(file_path)
        fh = logging.FileHandler(file_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)

    return logger


# ============================================================
# 모델 선택
# ============================================================

class ModelSelector:
    def __init__(self, cfg: Dict[str, Any]):
        models = cfg.get("models", {})
        self.high_quality = models.get("high_quality", "gpt-4o")
        self.standard = models.get("standard", "gpt-4o")
        self.fast = models.get("fast", "gpt-4o-mini")
        
        self.gpt5_compat = cfg.get("openai", {}).get("gpt5_compat", {})
        self.fallback_on_error = self.gpt5_compat.get("fallback_on_error", True)
    
    def get_model(self, task: str) -> str:
        high_quality_tasks = ["integrator", "writer", "cts_scorer", "scorer", "refiner",
                             "refine_execution", "refine_polish"]
        standard_tasks = ["planner", "reviewer", "refine_diagnostic", "refine_planning"]
        fast_tasks = ["length_fixer"]

        task_lower = task.lower()

        if any(t in task_lower for t in high_quality_tasks):
            return self.high_quality
        elif any(t in task_lower for t in standard_tasks):
            return self.standard
        elif any(t in task_lower for t in fast_tasks):
            return self.fast
        return self.standard
    
    def get_fallback(self, model: str) -> str:
        if model == self.high_quality:
            return self.standard
        return self.fast
    
    def is_gpt5(self, model: str) -> bool:
        return model.lower().startswith("gpt-5") or model.lower().startswith("o")
    
    def get_reasoning(self, model: str) -> Optional[Dict]:
        if self.is_gpt5(model) and self.gpt5_compat.get("use_reasoning", True):
            return {"effort": self.gpt5_compat.get("reasoning_effort", "medium")}
        return None


# ============================================================
# OpenAI
# ============================================================

def get_api_key(cfg: Dict[str, Any]) -> str:
    key = cfg.get("openai", {}).get("api_key", "").strip()
    if key:
        return key
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    raise RuntimeError("OPENAI_API_KEY 필요")


def with_retry(fn, max_retries: int, base_sleep: float, logger: logging.Logger, what: str):
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            retryable = any(k in msg for k in ["rate", "timeout", "overloaded", "502", "503"])
            if attempt >= max_retries or not retryable:
                raise
            sleep = base_sleep * (2 ** attempt)
            logger.warning("%s 재시도 %d/%d: %s", what, attempt + 1, max_retries, e)
            time.sleep(sleep)


def is_quota_error_message(msg: str) -> bool:
    m = (msg or "").lower()
    return "insufficient_quota" in m or "exceeded your current quota" in m


def build_error_payload(kind: str, message: str) -> Dict[str, str]:
    return {"__error__": kind, "__error_message__": message}


def is_error_payload(obj: Any) -> bool:
    return isinstance(obj, dict) and bool(obj.get("__error__"))


def build_failure_text(error_type: str, error_message: str) -> str:
    if error_type == "insufficient_quota":
        return (
            "[실패] OpenAI 사용량/요금제 한도 초과로 생성이 중단되었습니다.\n"
            "- 조치: 결제/요금제 확인 또는 API Key 교체\n"
            f"- 상세: {error_message}"
        )
    return f"[실패] API 호출 오류로 생성이 중단되었습니다.\n- 상세: {error_message}"


# ============================================================
# 회사 프로필 보정
# ============================================================

def validate_company_profile(cp: Dict, company_name: str, role: str) -> Dict:
    fixed = dict(cp) if isinstance(cp, dict) else {}

    def ensure_list(key: str):
        v = fixed.get(key, [])
        if not isinstance(v, list):
            v = [v] if v else []
        fixed[key] = [str(x).strip() for x in v if str(x).strip()]

    def ensure_str(key: str, default: str):
        v = fixed.get(key, default)
        fixed[key] = str(v).strip() if v else default

    ensure_str("company_name", company_name)
    ensure_str("role", role)
    ensure_str("writing_tone", "명확하고 구체적인 근거 중심")

    for key in ["values", "talent_traits", "required_qualifications", 
                "preferred_qualifications", "keywords", "do_not_claim"]:
        ensure_list(key)

    if not fixed["do_not_claim"]:
        fixed["do_not_claim"] = ["근거 없는 수치", "과장된 표현", "논문 리스트 나열"]

    return fixed


# ============================================================
# 근거 포맷팅
# ============================================================

def format_evidence(hits: List[Dict], max_chars: int, qtype: QuestionType) -> str:
    if qtype == QuestionType.SELF_INTRO:
        hits = hits[:2]
    elif qtype == QuestionType.MOTIVATION:
        hits = hits[:3]
    elif qtype == QuestionType.COMPETENCY:
        hits = hits[:4]
    else:
        hits = hits[:3]
    
    lines = []
    for h in hits:
        chunk = h.get("chunk", h)
        chunk_id = chunk.get("id", "unknown")
        source = chunk.get("meta", {}).get("title", "")[:30]
        text = clamp_text(normalize_ws(chunk.get("text", "")), 200)
        lines.append(f"- [{chunk_id}] ({source}) {text}")
    
    return clamp_text("\n".join(lines), max_chars)


def detect_weak_evidence(hits: List[Dict], evidence: str, cfg: Dict) -> Tuple[bool, List[str]]:
    wcfg = cfg.get("evidence_quality", {})
    reasons = []
    
    valid = sum(1 for h in hits if len(h.get("chunk", h).get("text", "")) >= 40)
    if valid < wcfg.get("min_hits", 3):
        reasons.append(f"유효 근거 부족: {valid}개")
    
    if len(evidence) < wcfg.get("min_total_chars", 400):
        reasons.append(f"근거 텍스트 부족")
    
    if hits and hits[0].get("score", 0) < wcfg.get("min_score", 0.2):
        reasons.append(f"유사도 낮음")
    
    return len(reasons) > 0, reasons


# ============================================================
# Agent 호출
# ============================================================

def call_agent_json(
    client: OpenAI, model: str, instructions: str, payload: Dict,
    max_tokens: int, retry_cfg: Dict, logger: logging.Logger,
    what: str, model_selector: ModelSelector
) -> Dict:
    
    inp = "json\n" + json.dumps(payload, ensure_ascii=False)
    
    def _build_kwargs(m: str) -> Dict:
        kwargs = {
            "model": m,
            "instructions": instructions,
            "input": inp,
            "max_output_tokens": max_tokens,
            "text": {"format": {"type": "json_object"}},
        }
        reasoning = model_selector.get_reasoning(m)
        if reasoning:
            kwargs["reasoning"] = reasoning
        return kwargs
    
    def _call(kwargs: Dict):
        return client.responses.create(**kwargs)
    
    def _extract(resp) -> str:
        txt = getattr(resp, "output_text", "")
        if txt:
            return txt.strip()
        outputs = getattr(resp, "output", None)
        if outputs:
            for out in outputs:
                if getattr(out, "type", None) == "message":
                    for c in getattr(out, "content", []) or []:
                        if getattr(c, "type", None) == "output_text":
                            return (getattr(c, "text", "") or "").strip()
        return ""
    
    def _parse(txt: str) -> Dict:
        txt = txt.strip()
        if txt.startswith("\ufeff"):
            txt = txt[1:]
        if txt.lower().startswith("json"):
            txt = txt[4:].lstrip()
        try:
            return json.loads(txt)
        except:
            pass
        m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", txt, flags=re.S)
        if m:
            return json.loads(m.group(1))
        start, end = txt.find("{"), txt.rfind("}")
        if start != -1 and end > start:
            return json.loads(txt[start:end+1])
        raise ValueError("JSON 파싱 실패")
    
    def _try_call(m: str, label: str) -> Dict:
        kwargs = _build_kwargs(m)
        try:
            resp = with_retry(
                lambda: _call(kwargs),
                retry_cfg.get("max_retries", 2),
                retry_cfg.get("base_sleep_sec", 1.0),
                logger, label
            )
            return _parse(_extract(resp))
        except TypeError as e:
            if "text" in str(e):
                kwargs.pop("text", None)
            if "reasoning" in str(e):
                kwargs.pop("reasoning", None)
            resp = with_retry(lambda: _call(kwargs), 1, 1.0, logger, f"{label}(compat)")
            return _parse(_extract(resp))
    
    try:
        return _try_call(model, what)
    except Exception as e:
        if is_quota_error_message(str(e)):
            logger.error("[%s] quota exceeded: %s", what, e)
            return build_error_payload("insufficient_quota", str(e))
        if model_selector.fallback_on_error and model_selector.is_gpt5(model):
            fallback = model_selector.get_fallback(model)
            logger.warning("[%s] %s 실패, %s로 폴백", what, model, fallback)
            try:
                return _try_call(fallback, f"{what}(fallback)")
            except:
                pass
        logger.error("[%s] 실패: %s", what, e)
        return {}


# ============================================================
# 프롬프트 빌더 (R1, R5 적용)
# ============================================================

def build_prompts(company_name: str, workflow_guide: str, qtype: QuestionType) -> Dict[str, str]:
    guide = clamp_text(workflow_guide, 4000)
    guideline = QuestionClassifier.get_guideline(qtype)
    
    structure_guide = "\n".join(f"{i+1}. {s}" for i, s in enumerate(guideline["structure"]))
    allowed = ", ".join(guideline["allowed"])
    limited = ", ".join(guideline["limited"])
    forbidden = ", ".join(guideline["forbidden"])
    tone = guideline["tone"]
    
    output_format = """[출력 형식]
- 질문에 직접 답하는 제출용 본문만 출력
- 소제목/번호/STAR/불릿/메타 문구 금지
- 1~3 문단 허용 (줄바꿈만 사용)"""

    # R1: 1인칭 금지 규칙 추가
    first_person_rule = """
## ⚠️ 1인칭 표현 금지 (매우 중요)
다음 표현을 절대 사용하지 마세요: "저는", "나는", "제가", "내가", "저의", "나의"

대체 방법:
- 주어 생략: "~하며", "~했고", "~해왔다"
- 경험 중심: "~한 경험이 있다", "~을 통해"
- 행동 중심: "~하는 과정에서", "~하면서"
"""
    
    common = f"""너는 {company_name} 자기소개서 작성 멀티에이전트다.

## 핵심 원칙
1. 질문에 직접 답하는 제출용 본문을 작성
2. 논리 흐름: 성향/관심 → 경험 → 배운 점 → 직무 연결
3. 존댓말(합니다/입니다) 문체 유지
4. 형용사 단독 사용 금지 → 행동/근거로 증명
5. 제공되지 않은 수치/사실 가공 금지
6. 수치/지표가 없으면 [내용 보강 필요] 표기
7. 문장 길이는 50자 이내를 우선 (길면 분할)
8. 접속사 반복 최소화
9. **1인칭 주어 완전 제거**
10. 메타 발언 금지: "예시/가이드/설명입니다/아래는" 등 금지
11. 질문과 직접 무관한 일반론 금지
12. **한글 중심 작성** (기술 용어 외 영어 사용 금지)
{first_person_rule}

## ⚠️ 한글 위주 작성 규칙 (매우 중요)
- 본문은 한글을 중심으로 작성하세요
- 영어 문장, 영어로만 된 긴 설명은 절대 금지
- **기술 용어는 영어 사용 허용** (예: SLAM, 3D reconstruction, depth estimation)
- 일반 명사는 한글 사용 필수
  ✗ 나쁜 예: "I worked on the project", "The system is good"
  ✓ 좋은 예: "SLAM 기반 3D reconstruction 프로젝트에서 depth estimation 알고리즘을 개선했습니다"

## 출력 형식 (고정)
{output_format}
※ [Self-Scoring] 섹션은 시스템이 추가하므로 출력하지 마라.

## 질문 유형: {qtype.value}
### 참고 구조(내용 선택용)
{structure_guide}
### 허용: {allowed}
### 보조적 사용만: {limited}
### 금지: {forbidden}
### 톤: {tone}

## 워크플로우 가이드
{guide}

출력은 JSON만. 키/문자열은 큰따옴표.
"""

    planner_schema = """출력 JSON:
{{
    "reasoning_summary": "2줄 이내",
    "outline": ["구조1", "구조2", ...],
    "core_messages": ["핵심메시지1", ...],
    "personality_traits": ["성향/가치관"],
    "experience_to_use": ["사용할 경험 1개"],
    "must_avoid": ["피해야 할 것"]
}}"""

    return {
        "planner_strategic": common + f"""
역할: Strategic Planner
- 제출용 본문 형식 준수
- 첫 문장에서 질문에 직접 답하도록 설계
- KB 근거는 경험 증명에만 사용
- 1인칭 표현 배제한 구조 설계
{planner_schema}""",

        "planner_creative": common + f"""
역할: Creative Planner
- 차별화된 스토리텔링
- 성향을 자연스럽게 드러내는 전개
- 과도한 실적 나열 금지
- 1인칭 없이 몰입감 있는 서술
{planner_schema}""",

        "planner_stable": common + f"""
역할: Critical Planner
- 안정적이고 검증된 구조
- 질문 주제에 충실
- 과장 없는 서술
{planner_schema}""",

        "cts_scorer": common + """
역할: CTS Evaluator (중요: 합격 가능성 평가)

평가 기준 (0~10):
- question_focus (0.30): 질문 의도에 직접 답하는가
- logic_flow (0.20): 흐름이 자연스럽고 설득력 있는가
- specificity (0.20): 구체적 행동/상황이 계획에 포함되는가
- expression_quality (0.15): 표현 완성도가 높은 흐름인가
- submission_ready (0.10): 소제목/번호/STAR/불릿 없이 본문으로 구성되는가
- length_fit (0.05): 글자수 범위를 맞출 수 있는 구조인가

출력 JSON:
{{
    "scores": [{{"id": "strategic", "question_focus": 8, "logic_flow": 7, "specificity": 7,
                 "expression_quality": 8, "submission_ready": 8, "length_fit": 7, "total": 7.7, "rationale": "이유"}}, ...],
    "best_id": "선택된 ID",
    "selected_plan": {{...}},
    "warnings": ["연구 보고서 느낌 경고", ...]
}}""",

        "writer": common + f"""
역할: Creative Writer (중요: 사람 중심 작성)

필수 규칙:
1. 제출용 본문 형식 준수 (소제목/번호/STAR/불릿/메타 문구 금지)
2. 존댓말(합니다/입니다)로 작성
3. 수치/지표 없으면 [내용 보강 필요]로 표기
4. **1인칭 표현 완전 금지**

출력 JSON:
{{
    "reasoning_summary": "2줄 이내",
    "draft_text": "초안 텍스트 (제출용 본문, 1인칭 없이)",
    "personality_shown": ["드러난 성향"],
    "evidence_used": ["사용한 근거 (2개 이하)"],
    "self_check": {{"is_person_centered": true, "is_report_style": false, "has_first_person": false, "has_structure_markers": false}}
}}""",

        "reviewer": common + """
역할: Critical Reviewer

검토 항목:
1. 제출용 본문 형식 준수 여부 (소제목/번호/STAR/불릿/메타 문구 금지)
2. 질문 집중도 및 직접성
3. 논리 흐름/구체성
4. 존댓말 유지 여부
5. **1인칭 표현 사용 여부**

출력 JSON:
{{
    "reasoning_summary": "2줄",
    "is_report_style": false,
    "question_focus_score": 8,
    "has_first_person": false,
    "first_person_found": [],
    "format_violations": [],
    "issues": ["문제1", ...],
    "fixes": ["수정제안1", ...],
    "hallucination_risks": ["위험1", ...]
}}""",

        "integrator": common + f"""
역할: Integrator (최종 통합, 가장 중요)

필수:
- 공백 포함 글자수 범위 엄수
- 제출용 본문 형식 준수 (소제목/번호/STAR/불릿/메타 문구 금지)
- 존댓말(합니다/입니다) 유지
- 수치/지표 없으면 [내용 보강 필요] 표기
- **1인칭 표현 완전 제거**

최종 점검:
- [ ] 질문에 직접 답하는가?
- [ ] 논리 흐름이 자연스러운가?
- [ ] 제출용 본문 형식을 지키는가?
- [ ] **1인칭 표현이 없는가?**

출력 JSON:
{{
    "reasoning_summary": "2줄",
    "final_text": "최종 텍스트 (제출용 본문, 1인칭 없이)",
    "char_count": 970,
    "pass": true,
    "first_person_check": true,
    "length_fix_instruction": "필요시 조정 지침",
    "final_check": {{
        "question_answered": true,
        "person_centered": true,
        "not_report_style": true,
        "no_first_person": true,
        "submission_ready": true
    }}
}}""",

        "length_fixer": common + """
역할: Length Fixer (Phase 1 초안 분량 확보)

**Phase 1의 핵심 목표: 최종 제출 분량(950~1000자)에 가까운 초안을 생성**

규칙:
- **목표 글자수에 적극적으로 도달** (950자 이상 필수)
- 분량 확보 방법:
  * 기존 내용을 구체적으로 확장 (추상적 표현 → 구체적 사례로)
  * KB 근거를 활용하여 경험을 상세히 서술
  * 논리적 연결 문장 추가 (원인-행동-결과 연결)
  * 불필요한 반복은 피하되, 필요한 상세 설명은 추가
- 제출용 본문 형식 유지 (소제목/번호/STAR/불릿/메타 문구 금지)
- 존댓말 유지
- 수치/지표 없으면 [내용 보강 필요] 유지
- **1인칭 표현 추가 금지**

⚠️ 중요: Phase 1에서 분량을 확보하지 못하면, Phase 2 이후에는 글자 수를 늘릴 수 없습니다!

출력 JSON:
{{
    "reasoning_summary": "조정 내용 (어떻게 분량을 확보했는지)",
    "final_text": "조정된 텍스트 (제출용 본문, 1인칭 없이, 950자 이상)",
    "char_count": 980
}}""",
    }


def select_cts_plan(cts: Dict, candidates: List[Dict], logger: logging.Logger) -> Dict:
    if isinstance(cts.get("selected_plan"), dict):
        return cts["selected_plan"]
    
    best_id = cts.get("best_id")
    if best_id:
        for c in candidates:
            if c.get("id") == best_id:
                return c.get("plan", {})
    
    scores = cts.get("scores", [])
    if scores:
        try:
            top = max(scores, key=lambda s: float(s.get("total", 0)))
            for c in candidates:
                if c.get("id") == top.get("id"):
                    return c.get("plan", {})
        except:
            pass
    
    logger.warning("CTS 선택 실패 → 첫 후보")
    return candidates[0].get("plan", {}) if candidates else {}


# ============================================================
# Markdown 저장 (R2: 스코어 리포트 추가)
# ============================================================

def write_markdown(
    out_path: str, *,
    company_name: str, role: str, question: str, qtype: QuestionType,
    models_used: Dict[str, str], char_count: int, final_text: str,
    evidence_top: List[Dict], company_profile_path: str, kb_dir: str,
    weak_evidence: bool, weak_reasons: List[str],
    score_result: Optional[ScoreResult] = None,
    refine_iterations: Optional[List[RefineIteration]] = None,
    versions: Optional[List[Dict[str, Any]]] = None,
    submission_text: Optional[str] = None,
    submission_path: Optional[str] = None,
    cfg: Optional[Dict] = None
) -> None:
    ensure_dir(out_path)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = []
    md.append("# 자기소개서 결과\n\n")
    md.append(f"- 생성: {now}\n")
    md.append(f"- 회사: {company_name}\n")
    md.append(f"- 직무: {role}\n")
    md.append(f"- 질문: {question}\n")
    md.append(f"- 질문 유형: {qtype.value}\n")
    md.append(f"- 글자수: {char_count}\n")
    
    md.append("\n### 사용 모델\n")
    for task, model in models_used.items():
        md.append(f"- {task}: `{model}`\n")

    # R2: 스코어 리포트
    if score_result:
        scoring_cfg = cfg.get("scoring", {}) if cfg else {}
        pass_threshold = scoring_cfg.get("pass_threshold", 9.5)
        gap = max(0, pass_threshold - score_result.total_score)

        md.append("\n---\n\n## 📊 품질 스코어\n\n")
        md.append(f"- **총점**: {score_result.total_score:.2f}/10.0\n")
        md.append(f"- **목표**: {pass_threshold:.1f}/10.0\n")

        if score_result.passed:
            md.append(f"- **상태**: ✅ 통과\n")
        else:
            md.append(f"- **상태**: ⚠️ 미달 (갭: -{gap:.2f}점)\n")

        if score_result.criteria_scores:
            md.append("\n| 항목 | 점수 | 상태 |\n|------|------|------|\n")
            for name, score in score_result.criteria_scores.items():
                if name != "rationale":
                    module_cfg = scoring_cfg.get("modules", {}).get(name, {})
                    min_score = module_cfg.get("min_score", 9.0)
                    status = "✅" if score >= min_score else "❌"
                    md.append(f"| {name} | {score:.2f} | {status} |\n")

        if score_result.failed_criteria:
            md.append(f"\n**미달 항목**: {', '.join(score_result.failed_criteria)}\n")

        # ✅ 9.5 미달 시: 개선 가이드 추가
        if not score_result.passed and score_result.recommendations:
            md.append("\n### ⚠️ 품질 개선 가이드\n\n")
            md.append(f"현재 점수가 목표({pass_threshold:.1f})에 **{gap:.2f}점** 미달합니다.\n\n")
            md.append("#### 개선 권고 사항 (우선순위순)\n\n")
            for i, rec in enumerate(score_result.recommendations[:5], 1):
                md.append(f"{i}. **권고**: {rec}\n")
        elif score_result.recommendations:
            md.append("\n**개선 권고**:\n")
            for rec in score_result.recommendations:
                md.append(f"- {rec}\n")

    # R2: Refine 이력
    if refine_iterations:
        md.append("\n---\n\n## 🔄 Refine 이력\n\n")
        for it in refine_iterations:
            md.append(f"- **Iteration {it.iteration}**: {it.score_before:.2f} → {it.score_after:.2f}\n")

    if weak_evidence:
        md.append("\n---\n\n## ⚠️ 근거 부족 경고\n\n")
        for r in weak_reasons:
            md.append(f"- {r}\n")

    # ✅ 본문은 항상 출력 (9.5 미만이어도 출력)
    md.append("\n---\n\n## 📝 제출용 본문\n\n")

    # 품질 미달 시 경고 메시지 추가
    if score_result and not score_result.passed:
        md.append("> ⚠️ **주의**: 이 본문은 현재 품질 기준(9.5/10)에 미달합니다.\n")
        md.append("> 위 개선 권고 사항을 참고하여 수정 후 제출하시기 바랍니다.\n\n")

    if submission_text:
        md.append(submission_text.strip() + "\n")
    else:
        md.append(final_text.strip() + "\n")

    md.append("\n---\n\n## 자기소개서 본문 (버전별)\n\n")
    if versions:
        for v in versions:
            v_idx = v.get("version", 1)
            v_text = strip_self_scoring(str(v.get("text", "")))
            v_score = v.get("score")
            v_recs = []
            if isinstance(v_score, ScoreResult):
                v_recs = v_score.recommendations or []
            md.append(f"[버전 {v_idx}]\n\n")
            md.append(v_text.strip() + "\n\n")
            if isinstance(v_score, ScoreResult):
                md.append("[Self-Scoring]\n")
                md.append(f"* total: {v_score.total_score:.2f} / 10\n")
                recs = v_recs[:3] if v_recs else ["없음"]
                for r in recs:
                    md.append(f"* 개선 포인트: {r}\n")
                md.append("\n")
    else:
        md.append(final_text.strip() + "\n")

    md.append("\n---\n\n## 사용 근거\n\n")
    for h in evidence_top[:3]:
        chunk = h.get("chunk", h)
        md.append(f"- `{chunk.get('id', '')}` ({chunk.get('meta', {}).get('title', '')[:40]})\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(md))

    # ✅ 본문 파일도 항상 생성 (9.5 미만이어도 생성)
    if submission_path and submission_text:
        ensure_dir(submission_path)
        with open(submission_path, "w", encoding="utf-8") as f:
            # 품질 미달 시 파일 상단에 경고 추가
            if score_result and not score_result.passed:
                f.write("<!-- ⚠️ 경고: 이 본문은 품질 기준(9.5/10) 미달. 개선 후 제출 필요 -->\n\n")
            f.write(submission_text.strip() + "\n")


def render_output_path(template: str, company_name: str, q_idx: int, question: str) -> str:
    return template.format(
        company_slug=slugify(company_name),
        timestamp=now_ts(),
        q_idx=q_idx,
        question_slug=question_slug(question),
    )


# ============================================================
# 단일 질문 실행
# ============================================================

def run_for_question(cfg: Dict, logger: logging.Logger, question: str, q_idx: int):
    qtype = QuestionClassifier.classify(question)
    logger.info("[Q%d] 질문 유형: %s", q_idx, qtype.value)
    
    guideline = QuestionClassifier.get_guideline(qtype)
    logger.info("[Q%d] 권장 구조: %s", q_idx, " → ".join(guideline["structure"][:3]))
    
    model_selector = ModelSelector(cfg)
    
    openai_cfg = cfg.get("openai", {})
    max_tokens = openai_cfg.get("max_tokens", {})
    retry_cfg = openai_cfg.get("retry", {"max_retries": 2, "base_sleep_sec": 1.0})

    paths = cfg["paths"]
    kb_dir = paths["kb_dir"]
    company_db_dir = paths["company_db_dir"]
    workflow_md = paths["workflow_md"]

    company = cfg["company"]
    company_name = company["name"]
    role = company["role"]

    app = cfg["application"]
    target_min = app.get("target_char_min", 950)
    target_max = app.get("target_char_max", 1000)
    max_fix_iters = app.get("max_length_fix_iters", 3)

    out_cfg = cfg.get("output", {})
    out_template = out_cfg.get("template", "outputs/{company_slug}_{timestamp}_q{q_idx}_{question_slug}.md")

    # timestamp를 함수 스코프에서 유지 (나중에 report 저장 시 사용)
    timestamp = now_ts()
    out_md = out_template.format(
        company_slug=slugify(company_name),
        timestamp=timestamp,
        q_idx=q_idx,
        question_slug=question_slug(question),
    )

    models_used = {}

    # R1: 1인칭 제거기
    first_person_remover = FirstPersonRemover(cfg)

    # OpenAI
    with StepTimer(logger, f"[Q{q_idx}] 초기화"):
        client = OpenAI(api_key=get_api_key(cfg))
        base_workflow = read_text(workflow_md)
        pass_sop_parts = []
        ps_cfg = cfg.get("pass_sop_patterns", {})
        if ps_cfg.get("enabled", False):
            flow_guide_path = ps_cfg.get("output", {}).get("flow_guide_md", "temp/flow_guide.md")
            if os.path.exists(flow_guide_path):
                flow_guide = read_text(flow_guide_path)
                pass_sop_parts.append("[합격 자소서 흐름 가이드]\n" + clamp_text(flow_guide, 1200))
            patterns_md_path = ps_cfg.get("output", {}).get("patterns_md", "patterns/pass_sop_patterns.md")
            if os.path.exists(patterns_md_path):
                patterns_md = read_text(patterns_md_path)
                pass_sop_parts.append("[합격 자소서 구조 패턴 요약]\n" + clamp_text(patterns_md, 1400))
            patterns_json_path = ps_cfg.get("output", {}).get("patterns_json", "patterns/pass_sop_patterns.json")
            if os.path.exists(patterns_json_path):
                try:
                    patterns_json = load_json(patterns_json_path)
                except Exception:
                    patterns_json = {}
                if isinstance(patterns_json, dict):
                    rendered = render_flow_guide(patterns_json.get("flow_guide", {}), max_items=6)
                    if rendered:
                        pass_sop_parts.append("[합격 자소서 구조 패턴(JSON 요약)]\n" + rendered)
            structured_jsonl_path = ps_cfg.get("output", {}).get(
                "structured_jsonl", "temp/essays_structured.jsonl"
            )
            if os.path.exists(structured_jsonl_path):
                summary = summarize_structured_patterns(structured_jsonl_path, max_records=80, top_n=6)
                if summary:
                    pass_sop_parts.append("[합격 자소서 구조 통계(구조 패턴)]\n" + summary)

        if pass_sop_parts:
            workflow_guide = "\n\n".join(pass_sop_parts) + "\n\n[워크플로우]\n" + base_workflow
        else:
            workflow_guide = base_workflow

    # 회사 프로필
    company_slug = cfg.get("company_profile", {}).get("company_slug") or slugify(company_name)
    company_profile_path = os.path.join(company_db_dir, f"{company_slug}.json")

    if not os.path.exists(company_profile_path):
        raise RuntimeError(f"회사 프로필 없음: {company_profile_path}")

    with StepTimer(logger, f"[Q{q_idx}] 회사 프로필"):
        raw_cp = load_json(company_profile_path)
        company_profile = validate_company_profile(raw_cp, company_name, role)

    # KB
    with StepTimer(logger, f"[Q{q_idx}] KB 로드"):
        kb_searcher = KBSearcher(cfg)

    # RAG
    rag_cfg = cfg.get("rag", {})
    top_k = rag_cfg.get("top_k", 6)
    max_ev_chars = rag_cfg.get("max_evidence_chars", 1500)

    query = f"{question} {company_name} {role}"

    with StepTimer(logger, f"[Q{q_idx}] RAG 검색"):
        hits = kb_searcher.search(query, top_k=top_k)
        evidence = format_evidence(hits, max_ev_chars, qtype)

    weak, reasons = detect_weak_evidence(hits, evidence, cfg)
    if weak:
        logger.warning("[Q%d] 근거 부족: %s", q_idx, reasons)

    prompts = build_prompts(company_name, workflow_guide, qtype)

    constraints = {
        "target_char_min": target_min,
        "target_char_max": target_max,
        "question_type": qtype.value,
        "structure": guideline["structure"],
        "forbidden": guideline["forbidden"],
        "tone": company_profile.get("writing_tone", "명확하고 구체적"),
        "ban": company_profile.get("do_not_claim", []),
    }

    def abort_if_error(payload: Dict[str, Any], step_name: str) -> bool:
        if not is_error_payload(payload):
            return False
        error_type = payload.get("__error__", "api_error")
        error_message = payload.get("__error_message__", "")
        logger.error("[Q%d] %s 중단: %s", q_idx, step_name, error_message)
        final_text = build_failure_text(error_type, error_message)
        write_markdown(
            out_md,
            company_name=company_name, role=role, question=question, qtype=qtype,
            models_used=models_used, char_count=char_len(final_text), final_text=final_text,
            evidence_top=hits[:3], company_profile_path=company_profile_path,
            kb_dir=kb_dir, weak_evidence=weak, weak_reasons=reasons,
            score_result=None, refine_iterations=None,
            cfg=cfg
        )
        return True

    # Planner
    planner_model = model_selector.get_model("planner")
    models_used["Planner"] = planner_model

    with StepTimer(logger, f"[Q{q_idx}] Planner 1/3 [{planner_model}]"):
        p1 = call_agent_json(client, planner_model, prompts["planner_strategic"],
            {"company_profile": company_profile, "question": question, "evidence": evidence, "constraints": constraints},
            max_tokens.get("planner", 700), retry_cfg, logger, "Planner-Strategic", model_selector)
    if abort_if_error(p1, "Planner-Strategic"):
        return

    with StepTimer(logger, f"[Q{q_idx}] Planner 2/3 [{planner_model}]"):
        p2 = call_agent_json(client, planner_model, prompts["planner_creative"],
            {"company_profile": company_profile, "question": question, "evidence": evidence, "constraints": constraints},
            max_tokens.get("planner", 700), retry_cfg, logger, "Planner-Creative", model_selector)
    if abort_if_error(p2, "Planner-Creative"):
        return

    with StepTimer(logger, f"[Q{q_idx}] Planner 3/3 [{planner_model}]"):
        p3 = call_agent_json(client, planner_model, prompts["planner_stable"],
            {"company_profile": company_profile, "question": question, "evidence": evidence, "constraints": constraints},
            max_tokens.get("planner", 700), retry_cfg, logger, "Planner-Critical", model_selector)
    if abort_if_error(p3, "Planner-Critical"):
        return

    candidates = [
        {"id": "strategic", "plan": p1},
        {"id": "creative", "plan": p2},
        {"id": "critical", "plan": p3},
    ]

    # CTS
    cts_model = model_selector.get_model("cts_scorer")
    models_used["CTS Scorer"] = cts_model

    with StepTimer(logger, f"[Q{q_idx}] CTS 평가 [{cts_model}]"):
        cts = call_agent_json(client, cts_model, prompts["cts_scorer"],
            {"candidates": candidates, "company_profile": company_profile, "question": question, "evidence": evidence, "constraints": constraints},
            max_tokens.get("cts_scorer", 800), retry_cfg, logger, "CTS-Scorer", model_selector)
    if abort_if_error(cts, "CTS-Scorer"):
        return

    selected = select_cts_plan(cts, candidates, logger)

    # Writer
    writer_model = model_selector.get_model("writer")
    models_used["Writer"] = writer_model

    with StepTimer(logger, f"[Q{q_idx}] Writer [{writer_model}]"):
        writer = call_agent_json(client, writer_model, prompts["writer"],
            {"outline": selected.get("outline", []), "core_messages": selected.get("core_messages", []),
             "personality_traits": selected.get("personality_traits", []),
             "experience_to_use": selected.get("experience_to_use", []),
             "company_profile": company_profile, "question": question, "evidence": evidence, "constraints": constraints},
            max_tokens.get("writer", 1200), retry_cfg, logger, "Writer", model_selector)
    if abort_if_error(writer, "Writer"):
        return

    # Reviewer
    reviewer_model = model_selector.get_model("reviewer")
    models_used["Reviewer"] = reviewer_model

    with StepTimer(logger, f"[Q{q_idx}] Reviewer [{reviewer_model}]"):
        reviewer = call_agent_json(client, reviewer_model, prompts["reviewer"],
            {"draft_text": writer.get("draft_text", ""), "company_profile": company_profile,
             "evidence": evidence, "constraints": constraints},
            max_tokens.get("reviewer", 650), retry_cfg, logger, "Reviewer", model_selector)
    if abort_if_error(reviewer, "Reviewer"):
        return

    if reviewer.get("is_report_style"):
        logger.warning("[Q%d] ⚠️ '연구 보고서' 느낌 감지됨", q_idx)

    # R1: Reviewer에서 1인칭 감지
    if reviewer.get("has_first_person"):
        logger.warning("[Q%d] ⚠️ 1인칭 표현 감지됨: %s", q_idx, reviewer.get("first_person_found", []))

    # Integrator
    integrator_model = model_selector.get_model("integrator")
    models_used["Integrator"] = integrator_model

    with StepTimer(logger, f"[Q{q_idx}] Integrator [{integrator_model}]"):
        integrator = call_agent_json(client, integrator_model, prompts["integrator"],
            {"draft_text": writer.get("draft_text", ""), "fixes": reviewer.get("fixes", []),
             "hallucination_risks": reviewer.get("hallucination_risks", []),
             "is_report_style": reviewer.get("is_report_style", False),
             "has_first_person": reviewer.get("has_first_person", False),
             "company_profile": company_profile, "question": question, "constraints": constraints},
            max_tokens.get("integrator", 1500), retry_cfg, logger, "Integrator", model_selector)
    if abort_if_error(integrator, "Integrator"):
        return

    final_text = (integrator.get("final_text") or "").strip()
    final_text = strip_self_scoring(final_text)
    final_text = normalize_submission_text(final_text)

    # R1: 1인칭 강제 제거
    final_text = first_person_remover.remove(final_text)
    
    count = char_len(final_text)
    logger.info("[Q%d] 글자수: %d (목표 %d~%d)", q_idx, count, target_min, target_max)

    # Length Fixer
    fixer_model = model_selector.get_model("length_fixer")
    models_used["Length Fixer"] = fixer_model
    target_len = (target_min + target_max) // 2
    prev_delta = None

    for i in range(max_fix_iters):
        if target_min <= count <= target_max:
            break
        delta = target_len - count

        # Phase 1의 목표: 최소 950자 이상 확보
        # 950자 미만이면 반복 계속
        must_continue = count < target_min

        with StepTimer(logger, f"[Q{q_idx}] Length Fixer {i+1}/{max_fix_iters} [{fixer_model}]"):
            fixer = call_agent_json(client, fixer_model, prompts["length_fixer"],
                {"final_text": final_text, "target_min": target_min, "target_max": target_max,
                 "instruction": f"현재 {count}자 → 목표 {target_len}자, 변화량 {delta:+d}자 (반드시 950자 이상 달성). "
                                f"⚠️ Phase 1에서 분량을 확보하지 못하면 Phase 2 이후에는 늘릴 수 없습니다! "
                                f"{integrator.get('length_fix_instruction', '')}",
                 "constraints": constraints},
                max_tokens.get("length_fixer", 1200), retry_cfg, logger, "LengthFixer", model_selector)
            if abort_if_error(fixer, "LengthFixer"):
                return
            final_text = strip_self_scoring((fixer.get("final_text") or "").strip())
            final_text = normalize_submission_text(final_text)
            # R1: 다시 1인칭 제거
            final_text = first_person_remover.remove(final_text)
            count = char_len(final_text)
            logger.info("[Q%d] → %d자", q_idx, count)
            new_delta = target_len - count

            # 조기 종료 조건: 950자 미만이면 계속 진행
            if prev_delta is not None and not must_continue:
                improved = abs(prev_delta) - abs(new_delta)
                if improved < max(20, abs(prev_delta) * 0.2):
                    if count >= target_min:
                        logger.info("[Q%d] 목표 분량 달성 (%d자), 반복 종료", q_idx, count)
                        break
                    else:
                        logger.warning("[Q%d] 개선 폭은 작지만 분량 부족(%d자 < %d자), 계속 진행",
                                      q_idx, count, target_min)
            prev_delta = new_delta

    if not final_text:
        logger.error("[Q%d] 최종 결과가 비어있습니다. fallback 텍스트로 대체합니다.", q_idx)
        final_text = build_failure_text("api_error", "최종 출력이 비어있습니다.")
        count = char_len(final_text)

    # R1: 최종 1인칭 체크
    fp_report = first_person_remover.get_violation_report(final_text)
    if not fp_report["passed"]:
        logger.warning("[Q%d] ⚠️ 최종 결과에 1인칭 표현 남음: %s", q_idx, fp_report["violations"])

    # R2: 스코어링
    score_result = None
    refine_iterations = None
    versions = None
    submission_text = normalize_submission_text(strip_self_scoring(final_text))
    
    scoring_cfg = cfg.get("scoring", {})
    if scoring_cfg.get("enabled", False):
        initial_text = final_text
        with StepTimer(logger, f"[Q{q_idx}] 품질 스코어링"):
            scorer = QualityScorer(client, cfg, logger, model_selector)
            score_result = scorer.score(initial_text, question, company_profile, evidence, qtype)
            logger.info("[Q%d] 품질 점수: %.2f/10 (통과: %s)", 
                       q_idx, score_result.total_score, score_result.passed)

        versions = [{"version": 1, "text": initial_text, "score": score_result}]

        # Phase 2/3 Refinement Pipeline (NEW)
        all_iterations = []
        phase2_enabled = cfg.get("phase2", {}).get("enabled", False)
        phase3_enabled = cfg.get("phase3", {}).get("enabled", False)

        # Phase 2: Structural Quality Improvement (9.0-9.2)
        if phase2_enabled and score_result.total_score < cfg.get("phase2", {}).get("target_score", 9.0):
            with StepTimer(logger, f"[Q{q_idx}] Phase 2: Structural Quality Improvement"):
                from phase2_refiner import Phase2StructuralRefiner
                phase2_refiner = Phase2StructuralRefiner(
                    client, cfg, logger, scorer, kb_searcher, model_selector
                )
                final_text, score_result, phase2_iterations = phase2_refiner.refine(
                    initial_text, score_result, question, qtype,
                    company_profile, evidence, constraints
                )
                count = char_len(final_text)
                all_iterations.extend(phase2_iterations)
                logger.info("[Q%d] Phase 2 완료: %.2f → %.2f (%d회 반복)",
                           q_idx, initial_text and versions[0]["score"].total_score,
                           score_result.total_score, len(phase2_iterations))

                # Track versions
                for idx, it in enumerate(phase2_iterations, len(versions) + 1):
                    versions.append({
                        "version": idx,
                        "text": it.text_after,
                        "score": scorer.score(it.text_after, question, company_profile, evidence, qtype),
                        "phase": "phase2"
                    })

        # Phase 3: Final Convergence (9.5+)
        if phase3_enabled and score_result.total_score < cfg.get("phase3", {}).get("target_score", 9.5):
            phase2_baseline = score_result.total_score
            with StepTimer(logger, f"[Q{q_idx}] Phase 3: Final Convergence"):
                from phase3_polisher import Phase3FinalPolisher
                phase3_polisher = Phase3FinalPolisher(
                    client, cfg, logger, scorer, model_selector
                )
                final_text, score_result, phase3_iterations = phase3_polisher.refine(
                    final_text, score_result, question, company_profile,
                    evidence, qtype, constraints, phase2_baseline
                )
                count = char_len(final_text)
                all_iterations.extend(phase3_iterations)
                logger.info("[Q%d] Phase 3 완료: %.2f → %.2f (%d회 반복)",
                           q_idx, phase2_baseline, score_result.total_score, len(phase3_iterations))

                # Track versions
                for idx, it in enumerate(phase3_iterations, len(versions) + 1):
                    versions.append({
                        "version": idx,
                        "text": it.text_after,
                        "score": scorer.score(it.text_after, question, company_profile, evidence, qtype),
                        "phase": "phase3"
                    })

        # Update refine_iterations for backward compatibility
        refine_iterations = all_iterations if all_iterations else None
        if refine_iterations:
            models_used["Phase2_Refiner"] = model_selector.get_model("refine_execution")
            models_used["Phase3_Polisher"] = model_selector.get_model("refine_polish")
            submission_text = normalize_submission_text(strip_self_scoring(final_text))

    # 저장
    with StepTimer(logger, f"[Q{q_idx}] MD 저장"):
        write_markdown(
            out_md,
            company_name=company_name, role=role, question=question, qtype=qtype,
            models_used=models_used, char_count=count, final_text=final_text,
            evidence_top=hits[:3], company_profile_path=company_profile_path,
            kb_dir=kb_dir, weak_evidence=weak, weak_reasons=reasons,
            score_result=score_result, refine_iterations=refine_iterations,
            versions=versions, submission_text=submission_text,
            submission_path=os.path.splitext(out_md)[0] + "_submission.txt",
            cfg=cfg
        )

    logger.info("[Q%d] ✅ 완료: %s", q_idx, out_md)

    # Phase 2/3 리포트 저장
    if refine_iterations:
        # Save detailed phase report
        report_path = f"outputs/phase_report_q{q_idx}_{timestamp}.json"
        ensure_dir(report_path)

        # Separate iterations by phase
        phase2_iters = [it for it in refine_iterations if hasattr(it, 'diagnostics') and 'diagnostic' in it.diagnostics]
        phase3_iters = [it for it in refine_iterations if it not in phase2_iters]

        # Helper function to convert Enum to string in nested structures
        def serialize_for_json(obj):
            """Recursively convert Enum objects to their values for JSON serialization"""
            if isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, dict):
                return {k: serialize_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize_for_json(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                return serialize_for_json(asdict(obj) if hasattr(obj, '__dataclass_fields__') else obj.__dict__)
            else:
                return obj

        report = {
            "question": question,
            "initial_score": versions[0]["score"].total_score if versions else None,
            "final_score": score_result.total_score if score_result else None,
            "total_improvement": (score_result.total_score - versions[0]["score"].total_score) if score_result and versions else 0,
            "phase2": {
                "enabled": phase2_enabled,
                "iterations": len(phase2_iters),
                "improvement": (phase2_iters[-1].score_after - phase2_iters[0].score_before) if phase2_iters else 0,
                "details": [serialize_for_json(asdict(it)) for it in phase2_iters]
            },
            "phase3": {
                "enabled": phase3_enabled,
                "iterations": len(phase3_iters),
                "improvement": (phase3_iters[-1].score_after - phase3_iters[0].score_before) if phase3_iters else 0,
                "details": [serialize_for_json(asdict(it)) for it in phase3_iters]
            },
            "versions": len(versions)
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info("[Q%d] Phase 리포트 저장: %s", q_idx, report_path)

    # ✅ 터미널 출력도 항상 가능 (단, 미달 시 경고 표시)
    if cfg.get("output", {}).get("print_final_to_terminal", False):
        print("\n" + "=" * 60)
        if score_result and not score_result.passed:
            print(f"⚠️ 경고: 품질 기준(9.5/10) 미달 (현재: {score_result.total_score:.2f})")
            print("=" * 60)
        print(final_text)
        print("=" * 60)


# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="합격 자소서 생성 (v2.0)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"[ERROR] 설정 파일 없음: {args.config}")
        return

    cfg = load_yaml(args.config)
    logger = setup_logger(cfg)

    models = cfg.get("models", {})
    logger.info("모델: high=%s, std=%s, fast=%s",
                models.get("high_quality", "gpt-4o"),
                models.get("standard", "gpt-4o"),
                models.get("fast", "gpt-4o-mini"))

    # R1, R2, Phase 2/3 설정 로그
    style_rules = cfg.get("style_rules", {})
    scoring = cfg.get("scoring", {})
    phase2 = cfg.get("phase2", {})
    phase3 = cfg.get("phase3", {})

    logger.info("R1 1인칭 금지: %s", style_rules.get("forbidden_first_person", [])[:3])
    logger.info("R2 스코어링: %s (임계값: %.1f)",
                "활성화" if scoring.get("enabled") else "비활성화",
                scoring.get("pass_threshold", 7.0))
    logger.info("Phase 2 (구조 개선): %s (목표: %.1f, 최대 %d회)",
                "활성화" if phase2.get("enabled") else "비활성화",
                phase2.get("target_score", 9.0),
                phase2.get("max_iterations", 8))
    logger.info("Phase 3 (최종 다듬기): %s (목표: %.1f, 최대 %d회)",
                "활성화" if phase3.get("enabled") else "비활성화",
                phase3.get("target_score", 9.5),
                phase3.get("max_iterations", 5))

    app = cfg.get("application", {})
    questions = app.get("questions", None)

    if questions and isinstance(questions, list):
        logger.info("배치: %d개 질문", len(questions))
        for i, q in enumerate(questions, 1):
            q = str(q).strip()
            if q:
                with StepTimer(logger, f"배치 [Q{i}/{len(questions)}]"):
                    run_for_question(cfg, logger, q, i)
    else:
        q = str(app.get("question", "")).strip()
        if not q:
            raise RuntimeError("config.yaml에 application.question 필요")
        with StepTimer(logger, "단일 질문"):
            run_for_question(cfg, logger, q, 1)


if __name__ == "__main__":
    main()
