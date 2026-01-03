#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_sop.py - 합격 자소서 패턴 기반 자기소개서 생성

핵심 원칙:
  1. 사람 중심 서술 (성향 → 관심 → 경험 → 직무 연결)
  2. 질문 유형별 구조 적용
  3. KB 근거는 지정된 위치에서만 사용
  4. "연구 보고서"가 아닌 "자기소개서"

모델 선택:
  - gpt-5.2: Integrator, Writer, CTS Scorer
  - gpt-4o: Planner, Reviewer
  - gpt-4o-mini: Length Fixer

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
    
    # 질문 유형별 가이드
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
        """질문 유형 분류"""
        q_lower = question.lower()
        
        for qtype, patterns in cls.PATTERNS.items():
            if any(p in q_lower for p in patterns):
                return qtype
        
        return QuestionType.OTHER
    
    @classmethod
    def get_guideline(cls, qtype: QuestionType) -> Dict:
        """질문 유형별 가이드라인 반환"""
        return cls.GUIDELINES.get(qtype, cls.GUIDELINES[QuestionType.OTHER])


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


def summarize_structured_patterns(
    path: str,
    max_records: int = 80,
    top_n: int = 6,
) -> str:
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
        self.high_quality = models.get("high_quality", "gpt-5.2")
        self.standard = models.get("standard", "gpt-4o")
        self.fast = models.get("fast", "gpt-4o-mini")
        
        self.gpt5_compat = cfg.get("openai", {}).get("gpt5_compat", {})
        self.fallback_on_error = self.gpt5_compat.get("fallback_on_error", True)
    
    def get_model(self, task: str) -> str:
        high_quality_tasks = ["integrator", "writer", "cts_scorer"]
        standard_tasks = ["planner", "reviewer"]
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
    """질문 유형에 맞게 근거 포맷팅"""
    
    # 자기소개: 대표 경험 1~2개만
    if qtype == QuestionType.SELF_INTRO:
        hits = hits[:2]
    # 지원동기: 회사 연결 경험만
    elif qtype == QuestionType.MOTIVATION:
        hits = hits[:3]
    # 직무역량: 문제해결 사례
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
        score = h.get("score", 0)
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
            # SDK 호환성
            if "text" in str(e):
                kwargs.pop("text", None)
            if "reasoning" in str(e):
                kwargs.pop("reasoning", None)
            resp = with_retry(lambda: _call(kwargs), 1, 1.0, logger, f"{label}(compat)")
            return _parse(_extract(resp))
    
    try:
        return _try_call(model, what)
    except Exception as e:
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
# 프롬프트 빌더 (합격 자소서 패턴 적용)
# ============================================================

def build_prompts(company_name: str, workflow_guide: str, qtype: QuestionType) -> Dict[str, str]:
    guide = clamp_text(workflow_guide, 4000)
    guideline = QuestionClassifier.get_guideline(qtype)
    
    structure_guide = "\n".join(f"{i+1}. {s}" for i, s in enumerate(guideline["structure"]))
    allowed = ", ".join(guideline["allowed"])
    limited = ", ".join(guideline["limited"])
    forbidden = ", ".join(guideline["forbidden"])
    tone = guideline["tone"]
    
    common = f"""너는 {company_name} 자기소개서 작성 멀티에이전트다.

## 핵심 원칙
1. "합격 자소서"는 "연구 보고서"가 아니다
2. 사람 중심 서술: 성향 → 관심 → 경험 → 직무 연결
3. 기술/성과는 "증거"이지 "주인공"이 아님
4. 질문에 직접 답변

## 질문 유형: {qtype.value}
### 권장 구조
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
- 합격 자소서 구조 준수
- 사람 중심 도입 설계
- KB 근거는 경험 증명에만 사용
{planner_schema}""",

        "planner_creative": common + f"""
역할: Creative Planner
- 차별화된 스토리텔링
- 성향을 자연스럽게 드러내는 전개
- 과도한 실적 나열 금지
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
- question_focus (0.25): 질문 주제에 직접 답변하는가
- person_centered (0.25): 성향/가치관이 드러나는가
- structure_fit (0.20): 합격 자소서 구조를 따르는가
- evidence_proper (0.15): KB 근거가 적절한 위치에 있는가
- company_connect (0.15): 회사/직무와 자연스럽게 연결되는가

출력 JSON:
{{
    "scores": [
        {{"id": "strategic", "question_focus": 8, "person_centered": 7, "structure_fit": 8, "evidence_proper": 7, "company_connect": 8, "total": 7.7, "rationale": "이유"}},
        ...
    ],
    "best_id": "선택된 ID",
    "selected_plan": {{...}},
    "warnings": ["연구 보고서 느낌 경고", ...]
}}""",

        "writer": common + f"""
역할: Creative Writer (중요: 사람 중심 작성)

필수 규칙:
1. 첫 문장은 "나는 ~한 사람입니다" 류의 성향 제시로 시작
2. 논문/성과는 2개 이하, 경험 증명에서만 사용
3. 수치는 맥락 있게, 나열 금지
4. "연구 보고서"가 아닌 "자기소개서"로 읽히게

출력 JSON:
{{
    "reasoning_summary": "2줄 이내",
    "draft_text": "초안 텍스트",
    "personality_shown": ["드러난 성향"],
    "evidence_used": ["사용한 근거 (2개 이하)"],
    "self_check": {{"is_person_centered": true, "is_report_style": false}}
}}""",

        "reviewer": common + """
역할: Critical Reviewer

검토 항목:
1. 성과 나열 여부 → "연구 보고서" 느낌인가?
2. 질문 주제 집중도 → 질문에 답하고 있는가?
3. 사람 중심 서술 → 성향이 드러나는가?
4. KB 근거 위치 → 적절한 곳에서 사용되는가?

출력 JSON:
{{
    "reasoning_summary": "2줄",
    "is_report_style": false,
    "question_focus_score": 8,
    "issues": ["문제1", ...],
    "fixes": ["수정제안1", ...],
    "hallucination_risks": ["위험1", ...]
}}""",

        "integrator": common + f"""
역할: Integrator (최종 통합, 가장 중요)

필수:
- 공백 포함 950~1000자
- 첫 문장: 성향/가치관 제시
- 논문/수치: 2개 이하
- "연구 보고서" 느낌 제거

최종 점검:
- [ ] 질문에 직접 답변하는가?
- [ ] 성향/가치관이 드러나는가?
- [ ] 회사/직무 연결이 자연스러운가?

출력 JSON:
{{
    "reasoning_summary": "2줄",
    "final_text": "최종 텍스트",
    "char_count": 970,
    "pass": true,
    "length_fix_instruction": "필요시 조정 지침",
    "final_check": {{
        "question_answered": true,
        "person_centered": true,
        "not_report_style": true
    }}
}}""",

        "length_fixer": common + """
역할: Length Fixer

규칙:
- 목표 글자수 맞추기
- 성향 제시 첫 문장 유지
- 논문/수치 추가 금지

출력 JSON:
{{
    "reasoning_summary": "조정 내용",
    "final_text": "조정된 텍스트",
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
# Markdown 저장
# ============================================================

def write_markdown(
    out_path: str, *,
    company_name: str, role: str, question: str, qtype: QuestionType,
    models_used: Dict[str, str], char_count: int, final_text: str,
    evidence_top: List[Dict], company_profile_path: str, kb_dir: str,
    weak_evidence: bool, weak_reasons: List[str]
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

    if weak_evidence:
        md.append("\n---\n\n## ⚠️ 근거 부족 경고\n\n")
        for r in weak_reasons:
            md.append(f"- {r}\n")

    md.append("\n---\n\n## 최종 자기소개서\n\n")
    md.append(final_text.strip() + "\n")

    md.append("\n---\n\n## 사용 근거\n\n")
    for h in evidence_top[:3]:
        chunk = h.get("chunk", h)
        md.append(f"- `{chunk.get('id', '')}` ({chunk.get('meta', {}).get('title', '')[:40]})\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(md))


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
    # 질문 유형 분류
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
    out_md = render_output_path(out_template, company_name, q_idx, question)

    models_used = {}

    # OpenAI
    with StepTimer(logger, f"[Q{q_idx}] 초기화"):
        client = OpenAI(api_key=get_api_key(cfg))
        base_workflow = read_text(workflow_md)
        pass_sop_parts = []
        ps_cfg = cfg.get("pass_sop_patterns", {})
        if ps_cfg.get("enable", False):
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

    # RAG (질문 유형에 맞게)
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

    # 프롬프트 (질문 유형 반영)
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

    # Planner (gpt-4o)
    planner_model = model_selector.get_model("planner")
    models_used["Planner"] = planner_model

    with StepTimer(logger, f"[Q{q_idx}] Planner 1/3 [{planner_model}]"):
        p1 = call_agent_json(client, planner_model, prompts["planner_strategic"],
            {"company_profile": company_profile, "question": question, "evidence": evidence, "constraints": constraints},
            max_tokens.get("planner", 700), retry_cfg, logger, "Planner-Strategic", model_selector)

    with StepTimer(logger, f"[Q{q_idx}] Planner 2/3 [{planner_model}]"):
        p2 = call_agent_json(client, planner_model, prompts["planner_creative"],
            {"company_profile": company_profile, "question": question, "evidence": evidence, "constraints": constraints},
            max_tokens.get("planner", 700), retry_cfg, logger, "Planner-Creative", model_selector)

    with StepTimer(logger, f"[Q{q_idx}] Planner 3/3 [{planner_model}]"):
        p3 = call_agent_json(client, planner_model, prompts["planner_stable"],
            {"company_profile": company_profile, "question": question, "evidence": evidence, "constraints": constraints},
            max_tokens.get("planner", 700), retry_cfg, logger, "Planner-Critical", model_selector)

    candidates = [
        {"id": "strategic", "plan": p1},
        {"id": "creative", "plan": p2},
        {"id": "critical", "plan": p3},
    ]

    # CTS (gpt-5.2)
    cts_model = model_selector.get_model("cts_scorer")
    models_used["CTS Scorer"] = cts_model

    with StepTimer(logger, f"[Q{q_idx}] CTS 평가 [{cts_model}]"):
        cts = call_agent_json(client, cts_model, prompts["cts_scorer"],
            {"candidates": candidates, "company_profile": company_profile, "question": question, "evidence": evidence, "constraints": constraints},
            max_tokens.get("cts_scorer", 800), retry_cfg, logger, "CTS-Scorer", model_selector)

    selected = select_cts_plan(cts, candidates, logger)

    # Writer (gpt-5.2)
    writer_model = model_selector.get_model("writer")
    models_used["Writer"] = writer_model

    with StepTimer(logger, f"[Q{q_idx}] Writer [{writer_model}]"):
        writer = call_agent_json(client, writer_model, prompts["writer"],
            {"outline": selected.get("outline", []), "core_messages": selected.get("core_messages", []),
             "personality_traits": selected.get("personality_traits", []),
             "experience_to_use": selected.get("experience_to_use", []),
             "company_profile": company_profile, "question": question, "evidence": evidence, "constraints": constraints},
            max_tokens.get("writer", 1200), retry_cfg, logger, "Writer", model_selector)

    # Reviewer (gpt-4o)
    reviewer_model = model_selector.get_model("reviewer")
    models_used["Reviewer"] = reviewer_model

    with StepTimer(logger, f"[Q{q_idx}] Reviewer [{reviewer_model}]"):
        reviewer = call_agent_json(client, reviewer_model, prompts["reviewer"],
            {"draft_text": writer.get("draft_text", ""), "company_profile": company_profile,
             "evidence": evidence, "constraints": constraints},
            max_tokens.get("reviewer", 650), retry_cfg, logger, "Reviewer", model_selector)

    # 연구 보고서 느낌 경고
    if reviewer.get("is_report_style"):
        logger.warning("[Q%d] ⚠️ '연구 보고서' 느낌 감지됨", q_idx)

    # Integrator (gpt-5.2)
    integrator_model = model_selector.get_model("integrator")
    models_used["Integrator"] = integrator_model

    with StepTimer(logger, f"[Q{q_idx}] Integrator [{integrator_model}]"):
        integrator = call_agent_json(client, integrator_model, prompts["integrator"],
            {"draft_text": writer.get("draft_text", ""), "fixes": reviewer.get("fixes", []),
             "hallucination_risks": reviewer.get("hallucination_risks", []),
             "is_report_style": reviewer.get("is_report_style", False),
             "company_profile": company_profile, "question": question, "constraints": constraints},
            max_tokens.get("integrator", 1500), retry_cfg, logger, "Integrator", model_selector)

    final_text = (integrator.get("final_text") or "").strip()
    count = char_len(final_text)
    logger.info("[Q%d] 글자수: %d (목표 %d~%d)", q_idx, count, target_min, target_max)

    # Length Fixer (gpt-4o-mini)
    fixer_model = model_selector.get_model("length_fixer")
    models_used["Length Fixer"] = fixer_model

    for i in range(max_fix_iters):
        if target_min <= count <= target_max:
            break
        with StepTimer(logger, f"[Q{q_idx}] Length Fixer {i+1}/{max_fix_iters} [{fixer_model}]"):
            fixer = call_agent_json(client, fixer_model, prompts["length_fixer"],
                {"final_text": final_text, "target_min": target_min, "target_max": target_max,
                 "instruction": integrator.get("length_fix_instruction", ""), "constraints": constraints},
                max_tokens.get("length_fixer", 900), retry_cfg, logger, "LengthFixer", model_selector)
            final_text = (fixer.get("final_text") or "").strip()
            count = char_len(final_text)
            logger.info("[Q%d] → %d자", q_idx, count)

    if not final_text:
        raise RuntimeError("최종 결과가 비어있습니다.")

    # 저장
    with StepTimer(logger, f"[Q{q_idx}] MD 저장"):
        write_markdown(
            out_md,
            company_name=company_name, role=role, question=question, qtype=qtype,
            models_used=models_used, char_count=count, final_text=final_text,
            evidence_top=hits[:3], company_profile_path=company_profile_path,
            kb_dir=kb_dir, weak_evidence=weak, weak_reasons=reasons
        )

    logger.info("[Q%d] ✅ 완료: %s", q_idx, out_md)

    if cfg.get("output", {}).get("print_final_to_terminal", False):
        print("\n" + "=" * 60)
        print(final_text)
        print("=" * 60)


# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="합격 자소서 생성")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"[ERROR] 설정 파일 없음: {args.config}")
        return

    cfg = load_yaml(args.config)
    logger = setup_logger(cfg)

    models = cfg.get("models", {})
    logger.info("모델: high=%s, std=%s, fast=%s",
                models.get("high_quality", "gpt-5.2"),
                models.get("standard", "gpt-4o"),
                models.get("fast", "gpt-4o-mini"))

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
