#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_sop.py - 자기소개서 자동 생성 파이프라인

모델 선택 정책:
  - gpt-5.2 (high_quality): Integrator, Writer, CTS Scorer
  - gpt-4o (standard): Planner, Reviewer
  - gpt-4o-mini (fast): Length Fixer

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

import yaml
from openai import OpenAI

from query_kb import KBSearcher


# ================================
# 1) 유틸리티
# ================================

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


# ================================
# 2) 로깅
# ================================

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


# ================================
# 3) 모델 선택 헬퍼
# ================================

class ModelSelector:
    """작업 복잡도에 따른 모델 선택"""
    
    def __init__(self, cfg: Dict[str, Any]):
        models = cfg.get("models", {})
        self.high_quality = models.get("high_quality", "gpt-5.2")
        self.standard = models.get("standard", "gpt-4o")
        self.fast = models.get("fast", "gpt-4o-mini")
        
        self.gpt5_compat = cfg.get("openai", {}).get("gpt5_compat", {})
        self.use_reasoning = self.gpt5_compat.get("use_reasoning", True)
        self.reasoning_effort = self.gpt5_compat.get("reasoning_effort", "medium")
        self.fallback_on_error = self.gpt5_compat.get("fallback_on_error", True)
    
    def get_model(self, task: str) -> str:
        """작업별 모델 반환"""
        # gpt-5.2 필요: 최종 결과물 품질에 직접 영향
        high_quality_tasks = ["integrator", "writer", "cts_scorer"]
        
        # gpt-4o: 중간 복잡도
        standard_tasks = ["planner_strategic", "planner_creative", "planner_stable", "reviewer"]
        
        # gpt-4o-mini: 단순 작업
        fast_tasks = ["length_fixer", "meta_extract", "keyword_extract"]
        
        task_lower = task.lower()
        
        if any(t in task_lower for t in high_quality_tasks):
            return self.high_quality
        elif any(t in task_lower for t in standard_tasks):
            return self.standard
        elif any(t in task_lower for t in fast_tasks):
            return self.fast
        else:
            return self.standard  # 기본값
    
    def get_fallback_model(self, model: str) -> str:
        """폴백 모델 반환"""
        if model == self.high_quality:
            return self.standard
        elif model == self.standard:
            return self.fast
        return self.fast
    
    def is_gpt5(self, model: str) -> bool:
        """gpt-5.x 계열인지 확인"""
        return model.lower().startswith("gpt-5") or model.lower().startswith("o")
    
    def get_reasoning_config(self, model: str) -> Optional[Dict]:
        """gpt-5.2용 reasoning 설정"""
        if self.is_gpt5(model) and self.use_reasoning:
            return {"effort": self.reasoning_effort}
        return None


# ================================
# 4) OpenAI 클라이언트
# ================================

def get_api_key(cfg: Dict[str, Any]) -> str:
    key = cfg.get("openai", {}).get("api_key", "").strip()
    if key:
        return key
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    raise RuntimeError("OPENAI_API_KEY 필요 (config.yaml 또는 환경변수)")


def with_retry(fn, max_retries: int, base_sleep: float, logger: logging.Logger, what: str):
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            retryable = any(k in msg for k in ["rate", "timeout", "overloaded", "connection", "502", "503"])
            if attempt >= max_retries or not retryable:
                raise
            sleep = base_sleep * (2 ** attempt)
            logger.warning("%s 재시도 %d/%d (%.1fs): %s", what, attempt + 1, max_retries, sleep, e)
            time.sleep(sleep)


# ================================
# 5) 회사 프로필 보정
# ================================

def validate_company_profile(cp: Dict[str, Any], company_name: str, role: str) -> Dict[str, Any]:
    fixed = dict(cp) if isinstance(cp, dict) else {}

    def ensure_list(key: str):
        v = fixed.get(key, [])
        if v is None:
            v = []
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list):
            v = []
        fixed[key] = [str(x).strip() for x in v if str(x).strip()]

    def ensure_str(key: str, default: str):
        v = fixed.get(key, default)
        fixed[key] = str(v).strip() if v else default

    ensure_str("company_name", company_name)
    ensure_str("role", role)
    ensure_str("writing_tone", "명확/간결/근거 중심")

    ensure_list("values")
    ensure_list("talent_traits")
    ensure_list("required_qualifications")
    ensure_list("preferred_qualifications")
    ensure_list("keywords")
    ensure_list("do_not_claim")

    if not fixed["do_not_claim"]:
        fixed["do_not_claim"] = ["근거 없는 수치", "근거 없는 논문/수상", "근거 없는 경력"]

    return fixed


# ================================
# 6) 임베딩 캐시
# ================================

class EmbeddingCache:
    def __init__(self, path: str, enabled: bool, logger: logging.Logger):
        self.path = path
        self.enabled = enabled
        self.logger = logger
        self.data: Dict[str, Any] = {}
        if self.enabled and os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f) or {}
            except Exception:
                self.data = {}

    def get(self, model: str, query: str) -> Optional[List[float]]:
        if not self.enabled:
            return None
        key = f"{model}||{query}"
        v = self.data.get(key)
        return v if isinstance(v, list) and v else None

    def set(self, model: str, query: str, emb: List[float]) -> None:
        if not self.enabled:
            return
        key = f"{model}||{query}"
        self.data[key] = emb
        ensure_dir(self.path)
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False)
        except Exception as e:
            self.logger.warning("캐시 저장 실패: %s", e)


# ================================
# 7) 근거 포맷팅/검증
# ================================

def format_evidence(hits: List[Dict[str, Any]], max_chars: int) -> str:
    lines = []
    for h in hits:
        chunk = h.get("chunk", h)
        chunk_id = chunk.get("id", "unknown")
        source = chunk.get("meta", {}).get("title", chunk.get("source", ""))
        text = clamp_text(normalize_ws(chunk.get("text", "")), 260)
        score = h.get("score", 0)
        lines.append(f"- [{chunk_id}] (score:{score:.2f}, {source[:30]}) {text}")
    return clamp_text("\n".join(lines), max_chars)


def detect_weak_evidence(hits: List[Dict], evidence_text: str, cfg: Dict) -> Tuple[bool, List[str]]:
    wcfg = cfg.get("evidence_quality", {})
    min_hits = wcfg.get("min_hits", 3)
    min_chars = wcfg.get("min_total_chars", 400)
    min_score = wcfg.get("min_score", 0.20)

    reasons = []
    valid = sum(1 for h in hits if len(h.get("chunk", h).get("text", "")) >= 40)
    if valid < min_hits:
        reasons.append(f"유효 근거 부족: {valid} < {min_hits}")
    if len(evidence_text) < min_chars:
        reasons.append(f"근거 텍스트 부족: {len(evidence_text)} < {min_chars}")
    if hits:
        top_score = hits[0].get("score", 0)
        if top_score < min_score:
            reasons.append(f"상위 유사도 낮음: {top_score:.3f} < {min_score}")

    return len(reasons) > 0, reasons


# ================================
# 8) Agent 호출 (모델 선택 + gpt-5.2 호환성)
# ================================

def call_agent_json(
    client: OpenAI,
    model: str,
    instructions: str,
    payload: Dict[str, Any],
    max_tokens: int,
    retry_cfg: Dict,
    logger: logging.Logger,
    what: str,
    model_selector: ModelSelector,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """Agent 호출 (gpt-5.2 호환성 처리 포함)"""
    
    inp = "json\n" + json.dumps(payload, ensure_ascii=False)
    
    def _build_kwargs(model_to_use: str) -> Dict[str, Any]:
        kwargs = {
            "model": model_to_use,
            "instructions": instructions,
            "input": inp,
            "max_output_tokens": max_tokens,
            "text": {"format": {"type": "json_object"}},
        }
        
        # gpt-5.2용 reasoning 설정
        reasoning = model_selector.get_reasoning_config(model_to_use)
        if reasoning:
            kwargs["reasoning"] = reasoning
        
        if temperature is not None and not model_selector.is_gpt5(model_to_use):
            kwargs["temperature"] = temperature

        return kwargs
    
    def _call(kwargs: Dict[str, Any]):
        return client.responses.create(**kwargs)
    
    def _extract(resp) -> Tuple[str, Optional[str]]:
        txt = getattr(resp, "output_text", "")
        incomplete = getattr(resp, "incomplete_details", None)
        reason = getattr(incomplete, "reason", None) if incomplete else None
        if txt:
            return txt.strip(), reason
        outputs = getattr(resp, "output", None)
        if outputs:
            for out in outputs:
                if getattr(out, "type", None) == "message":
                    for c in getattr(out, "content", []) or []:
                        if getattr(c, "type", None) == "output_text":
                            return (getattr(c, "text", "") or "").strip(), reason
        return "", reason

    def _escape_newlines_in_json(s: str) -> str:
        out = []
        in_str = False
        escape = False
        for ch in s:
            if in_str:
                if escape:
                    out.append(ch)
                    escape = False
                    continue
                if ch == "\\":
                    out.append(ch)
                    escape = True
                    continue
                if ch == "\"":
                    in_str = False
                    out.append(ch)
                    continue
                if ch == "\n":
                    out.append("\\n")
                    continue
                if ch == "\r":
                    out.append("\\r")
                    continue
                if ch == "\t":
                    out.append("\\t")
                    continue
                out.append(ch)
            else:
                if ch == "\"":
                    in_str = True
                out.append(ch)
        return "".join(out)
    
    def _parse(txt: str) -> Dict:
        txt = txt.strip()
        if txt.startswith("\ufeff"):
            txt = txt[1:]
        if txt.lower().startswith("json"):
            txt = txt[4:].lstrip()
        try:
            return json.loads(txt)
        except Exception:
            pass
        escaped = _escape_newlines_in_json(txt)
        if escaped != txt:
            try:
                return json.loads(escaped)
            except Exception:
                pass
        m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", txt, flags=re.S | re.I)
        if m:
            fixed = _escape_newlines_in_json(m.group(1))
            return json.loads(fixed)
        start, end = txt.find("{"), txt.rfind("}")
        if start != -1 and end > start:
            fixed = _escape_newlines_in_json(txt[start:end+1])
            return json.loads(fixed)
        raise ValueError("JSON 파싱 실패")
    
    def _try_call(model_to_use: str, label: str) -> Dict[str, Any]:
        kwargs = _build_kwargs(model_to_use)
        
        try:
            resp = with_retry(
                lambda: _call(kwargs),
                max_retries=retry_cfg.get("max_retries", 2),
                base_sleep=retry_cfg.get("base_sleep_sec", 1.0),
                logger=logger,
                what=label
            )
            txt, incomplete = _extract(resp)
            if incomplete == "max_output_tokens":
                raise RuntimeError("max_output_tokens")
            return _parse(txt)
        except TypeError as e:
            # SDK 호환성 문제: 파라미터 제거 후 재시도
            msg = str(e)
            if "text" in msg or "response_format" in msg:
                kwargs.pop("text", None)
            if "reasoning" in msg:
                kwargs.pop("reasoning", None)
            
            resp = with_retry(
                lambda: _call(kwargs),
                max_retries=retry_cfg.get("max_retries", 2),
                base_sleep=retry_cfg.get("base_sleep_sec", 1.0),
                logger=logger,
                what=f"{label}(compat)"
            )
            txt, incomplete = _extract(resp)
            if incomplete == "max_output_tokens":
                raise RuntimeError("max_output_tokens")
            return _parse(txt)
    
    # 1차 시도
    try:
        logger.debug("[%s] 모델: %s", what, model)
        return _try_call(model, what)
    except Exception as e:
        if "max_output_tokens" in str(e).lower():
            bump = max(max_tokens + 600, int(max_tokens * 1.5))
            orig = max_tokens
            max_tokens = bump
            logger.warning("[%s] max_output_tokens 재시도: %d -> %d", what, orig, bump)
            try:
                return _try_call(model, f"{what}(retry)")
            except Exception as e2:
                e = e2
        # 폴백 시도 (gpt-5.2 실패 시 gpt-4o로)
        if model_selector.fallback_on_error and model_selector.is_gpt5(model):
            fallback = model_selector.get_fallback_model(model)
            logger.warning("[%s] %s 실패, %s로 폴백: %s", what, model, fallback, e)
            try:
                return _try_call(fallback, f"{what}(fallback)")
            except Exception as e2:
                logger.error("[%s] 폴백도 실패: %s", what, e2)
                return {}
        else:
            logger.error("[%s] 실패: %s", what, e)
            return {}


# ================================
# 9) 프롬프트 빌더
# ================================

def build_prompts(company_name: str, workflow_guide: str) -> Dict[str, str]:
    guide = clamp_text(workflow_guide, 5000)

    common = (
        f"너는 {company_name} 자기소개서 작성 멀티에이전트다.\n"
        "출력은 JSON 하나만. 키/문자열은 큰따옴표.\n"
        "문자열 내부 줄바꿈은 \\n로 escape하고 실제 줄바꿈을 쓰지 마라.\n"
        "근거(evidence, company_profile)에 없는 사실/수치/논문명은 만들지 마라.\n"
        "긴 CoT 출력 금지. reasoning_summary는 2줄 이내.\n"
        f"가이드라인:\n{guide}\n"
    )

    planner_schema = (
        "출력 JSON: reasoning_summary, outline(list 3~5), core_messages(list <=6), "
        "must_use_company_points(list <=6), must_use_evidence_ids(list <=6)\n"
    )

    return {
        "planner_strategic": common + (
            "역할: Strategic Planner\n논리적 구조, 근거 중심.\n"
            "다른 플래너와 중복되지 않는 구조를 제안해라.\n"
            + planner_schema
        ),
        "planner_creative": common + (
            "역할: Creative Planner\n차별화, 스토리텔링.\n"
            "다른 플래너와 중복되지 않는 구조를 제안해라.\n"
            + planner_schema
        ),
        "planner_stable": common + (
            "역할: Critical Planner\n안정성, 검증 가능한 주장만.\n"
            "다른 플래너와 중복되지 않는 구조를 제안해라.\n"
            + planner_schema
        ),
        "cts_scorer": common + (
            "역할: CTS Evaluator (중요: 최적안 선택)\n"
            "후보를 점수화하고 최적안 선택. 이 판단이 최종 품질에 직접 영향.\n"
            "기준: fit(0.3), differentiation(0.2), evidence(0.2), coherence(0.2), feasibility(0.1)\n"
            "출력: scores(list), best_id, selected_plan, merge_notes\n"
            "scores의 각 항목은 {id, fit, differentiation, evidence, coherence, feasibility, total, rationale}.\n"
            "rationale은 한 문장, 줄바꿈 금지.\n"
            "selected_plan은 {outline, core_messages, must_use_company_points, must_use_evidence_ids}.\n"
            "예시 JSON:\n"
            "{\"scores\":[{\"id\":\"strategic\",\"fit\":0,\"differentiation\":0,\"evidence\":0,"
            "\"coherence\":0,\"feasibility\":0,\"total\":0,\"rationale\":\"...\"}],"
            "\"best_id\":\"strategic\",\"selected_plan\":{\"outline\":[],\"core_messages\":[],"
            "\"must_use_company_points\":[],\"must_use_evidence_ids\":[]},\"merge_notes\":\"...\"}\n"
        ),
        "writer": common + (
            "역할: Creative Writer (중요: 핵심 내용 작성)\n"
            "최종 결과물의 문장 품질에 직접 기여. 신중하게 작성.\n"
            "출력: reasoning_summary, draft_text, used_company_points, used_evidence_ids\n"
            "must_use_company_points/must_use_evidence_ids가 주어지면 반드시 반영해라.\n"
            "draft_text는 한 줄 문자열로 출력하고 줄바꿈은 \\n로만 표현해라.\n"
            "예시 JSON: {\"reasoning_summary\":\"...\",\"draft_text\":\"...\",\"used_company_points\":[],\"used_evidence_ids\":[]}\n"
        ),
        "reviewer": common + (
            "역할: Critical Reviewer\n"
            "출력: reasoning_summary, issues(<=6), fixes(<=6), hallucination_risks(<=4)\n"
        ),
        "integrator": common + (
            "역할: Integrator (최종 통합, 가장 중요)\n"
            "모든 피드백을 통합하여 최종 자기소개서 완성. 품질 최우선.\n"
            "필수: 공백 포함 950~1000자.\n"
            "출력: reasoning_summary, final_text, char_count, pass, length_fix_instruction\n"
        ),
        "length_fixer": common + (
            "역할: Length Fixer (길이 조정)\n"
            "목표 글자수에 맞게 반드시 늘리거나 줄여라.\n"
            "부족하면 새 문장/예시를 추가하고, 초과하면 중복을 줄여라.\n"
            "final_text는 한 줄 문자열로 출력하고 줄바꿈은 \\n로만 표현해라.\n"
            "출력: reasoning_summary, final_text, char_count\n"
        ),
    }


def select_cts_plan(cts_resp: Dict, candidates: List[Dict], logger: logging.Logger) -> Dict:
    if isinstance(cts_resp.get("selected_plan"), dict):
        return cts_resp["selected_plan"]

    best_id = cts_resp.get("best_id") or cts_resp.get("selected_id")
    if best_id:
        for c in candidates:
            if c.get("id") == best_id:
                return c.get("plan", {})

    scores = cts_resp.get("scores", [])
    if scores:
        try:
            top = max(scores, key=lambda s: float(s.get("total", 0)))
            for c in candidates:
                if c.get("id") == top.get("id"):
                    return c.get("plan", {})
        except Exception:
            pass

    logger.warning("CTS 선택 실패 → 첫 후보 사용")
    return candidates[0].get("plan", {}) if candidates else {}


# ================================
# 10) Markdown 저장
# ================================

def write_markdown(
    out_path: str, *,
    company_name: str, role: str, question: str, models_used: Dict[str, str],
    char_count: int, final_text: str, evidence_top: List[Dict],
    company_profile_path: str, kb_dir: str,
    weak_evidence: bool, weak_reasons: List[str]
) -> None:
    ensure_dir(out_path)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = []
    md.append("# 자기소개서 결과\n\n")
    md.append(f"- 생성: {now}\n")
    md.append(f"- 회사: {company_name}\n")
    md.append(f"- 직무: {role}\n")
    md.append(f"- 글자수: {char_count}\n")
    md.append(f"- 질문: {question}\n")
    
    md.append("\n### 사용 모델\n")
    for task, model in models_used.items():
        md.append(f"- {task}: `{model}`\n")
    
    md.append(f"\n- 프로필: `{company_profile_path}`\n")
    md.append(f"- KB: `{kb_dir}`\n")

    if weak_evidence:
        md.append("\n---\n\n## ⚠️ 근거 부족 경고\n\n")
        for r in weak_reasons:
            md.append(f"- {r}\n")

    md.append("\n---\n\n## 최종 자기소개서\n\n")
    md.append(final_text.strip() + "\n")

    md.append("\n---\n\n## 사용 근거 Top 3\n\n")
    for h in evidence_top[:3]:
        chunk = h.get("chunk", h)
        md.append(f"- `{chunk.get('id', 'unknown')}` ({chunk.get('meta', {}).get('title', '')[:40]})\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(md))


# ================================
# 11) 출력 경로
# ================================

def render_output_path(template: str, company_name: str, q_idx: int, question: str) -> str:
    return template.format(
        company_slug=slugify(company_name),
        timestamp=now_ts(),
        q_idx=q_idx,
        question_slug=question_slug(question),
    )


# ================================
# 12) 단일 질문 실행
# ================================

def run_for_question(cfg: Dict, logger: logging.Logger, question: str, q_idx: int):
    # 모델 선택기
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

    # 사용된 모델 추적
    models_used = {}

    # 캐시
    cache_cfg = cfg.get("cache", {})
    emb_cache = EmbeddingCache(
        path=cache_cfg.get("embedding_cache_path", "cache/embeddings.json"),
        enabled=cache_cfg.get("enable_embedding_cache", True),
        logger=logger
    )

    # OpenAI 클라이언트
    with StepTimer(logger, f"[Q{q_idx}] OpenAI 초기화"):
        client = OpenAI(api_key=get_api_key(cfg))

    # 워크플로우 가이드
    with StepTimer(logger, f"[Q{q_idx}] 워크플로우 로드"):
        workflow_guide = read_text(workflow_md)
        ps_cfg = cfg.get("pass_sop_patterns", {})
        if ps_cfg.get("enable", False):
            flow_guide_path = ps_cfg.get("output", {}).get("flow_guide_md", "temp/flow_guide.md")
            if os.path.exists(flow_guide_path):
                flow_guide = read_text(flow_guide_path)
                workflow_guide += "\n\n[합격 자소서 흐름 가이드]\n" + flow_guide

    # 회사 프로필
    company_slug = cfg.get("company_profile", {}).get("company_slug") or slugify(company_name)
    company_profile_path = os.path.join(company_db_dir, f"{company_slug}.json")

    if not os.path.exists(company_profile_path):
        raise RuntimeError(f"회사 프로필 없음: {company_profile_path}\n→ build_company_profile.py 실행 필요")

    with StepTimer(logger, f"[Q{q_idx}] 회사 프로필 로드"):
        raw_cp = load_json(company_profile_path)
        company_profile = validate_company_profile(raw_cp, company_name, role)

    # KB 로드
    with StepTimer(logger, f"[Q{q_idx}] KB 로드"):
        try:
            kb_searcher = KBSearcher(cfg)
        except Exception as e:
            raise RuntimeError(f"KB 로드 실패: {e}\n→ build_kb.py 실행 필요")

    # RAG 검색
    rag_cfg = cfg.get("rag", {})
    top_k = rag_cfg.get("top_k", 6)
    max_ev_chars = rag_cfg.get("max_evidence_chars", 1800)

    query = f"{company_name} {role} {question} 성과 논문 프로젝트"

    with StepTimer(logger, f"[Q{q_idx}] RAG 검색 (top_k={top_k})"):
        hits = kb_searcher.search(query, top_k=top_k)
        evidence = format_evidence(hits, max_ev_chars)

    # 근거 품질
    weak, reasons = detect_weak_evidence(hits, evidence, cfg)
    if weak:
        logger.warning("[Q%d] 근거 부족: %s", q_idx, " | ".join(reasons))

    # 프롬프트
    prompts = build_prompts(company_name, workflow_guide)

    constraints = {
        "target_char_min": target_min,
        "target_char_max": target_max,
        "tone": company_profile.get("writing_tone", "명확/간결"),
        "ban": company_profile.get("do_not_claim", []),
    }

    # ===== Planner (gpt-4o) =====
    planner_model = model_selector.get_model("planner")
    models_used["Planner"] = planner_model
    
    with StepTimer(logger, f"[Q{q_idx}] Planner 1/3 (Strategic) [{planner_model}]"):
        p1 = call_agent_json(client, planner_model, prompts["planner_strategic"],
            {"company_profile": company_profile, "question": question, "evidence": evidence, "constraints": constraints},
            max_tokens.get("planner", 700), retry_cfg, logger, "Planner-Strategic", model_selector,
            temperature=0.2)

    with StepTimer(logger, f"[Q{q_idx}] Planner 2/3 (Creative) [{planner_model}]"):
        p2 = call_agent_json(client, planner_model, prompts["planner_creative"],
            {"company_profile": company_profile, "question": question, "evidence": evidence, "constraints": constraints},
            max_tokens.get("planner", 700), retry_cfg, logger, "Planner-Creative", model_selector,
            temperature=0.7)

    with StepTimer(logger, f"[Q{q_idx}] Planner 3/3 (Critical) [{planner_model}]"):
        p3 = call_agent_json(client, planner_model, prompts["planner_stable"],
            {"company_profile": company_profile, "question": question, "evidence": evidence, "constraints": constraints},
            max_tokens.get("planner", 700), retry_cfg, logger, "Planner-Critical", model_selector,
            temperature=0.1)

    candidates = [
        {"id": "strategic", "plan": p1},
        {"id": "creative", "plan": p2},
        {"id": "critical", "plan": p3},
    ]

    # ===== CTS Scorer (gpt-5.2) =====
    cts_model = model_selector.get_model("cts_scorer")
    models_used["CTS Scorer"] = cts_model
    
    with StepTimer(logger, f"[Q{q_idx}] CTS 평가/선택 [{cts_model}]"):
        cts = call_agent_json(client, cts_model, prompts["cts_scorer"],
            {"candidates": candidates, "company_profile": company_profile, "question": question, "evidence": evidence, "constraints": constraints},
            max_tokens.get("cts_scorer", 800), retry_cfg, logger, "CTS-Scorer", model_selector)

    selected = select_cts_plan(cts, candidates, logger)

    # ===== Writer (gpt-5.2) =====
    writer_model = model_selector.get_model("writer")
    models_used["Writer"] = writer_model
    
    with StepTimer(logger, f"[Q{q_idx}] Writer [{writer_model}]"):
        writer = call_agent_json(client, writer_model, prompts["writer"],
            {"outline": selected.get("outline", []), "core_messages": selected.get("core_messages", []),
             "must_use_company_points": selected.get("must_use_company_points", []),
             "must_use_evidence_ids": selected.get("must_use_evidence_ids", []),
             "company_profile": company_profile, "question": question, "evidence": evidence, "constraints": constraints},
            max_tokens.get("writer", 1200), retry_cfg, logger, "Writer", model_selector)

    # ===== Reviewer (gpt-4o) =====
    reviewer_model = model_selector.get_model("reviewer")
    models_used["Reviewer"] = reviewer_model
    
    with StepTimer(logger, f"[Q{q_idx}] Reviewer [{reviewer_model}]"):
        reviewer = call_agent_json(client, reviewer_model, prompts["reviewer"],
            {"draft_text": writer.get("draft_text", ""), "company_profile": company_profile,
             "evidence": evidence, "constraints": constraints},
            max_tokens.get("reviewer", 650), retry_cfg, logger, "Reviewer", model_selector)

    # ===== Integrator (gpt-5.2) =====
    integrator_model = model_selector.get_model("integrator")
    models_used["Integrator"] = integrator_model
    
    with StepTimer(logger, f"[Q{q_idx}] Integrator [{integrator_model}]"):
        integrator = call_agent_json(client, integrator_model, prompts["integrator"],
            {"draft_text": writer.get("draft_text", ""), "fixes": reviewer.get("fixes", []),
             "hallucination_risks": reviewer.get("hallucination_risks", []),
             "company_profile": company_profile, "question": question, "constraints": constraints},
            max_tokens.get("integrator", 1500), retry_cfg, logger, "Integrator", model_selector)

    final_text = (integrator.get("final_text") or "").strip()
    count = char_len(final_text)
    logger.info("[Q%d] 글자수: %d (목표 %d~%d)", q_idx, count, target_min, target_max)

    # ===== Length Fixer (gpt-4o-mini) =====
    fixer_model = model_selector.get_model("length_fixer")
    models_used["Length Fixer"] = fixer_model
    
    for i in range(max_fix_iters):
        if target_min <= count <= target_max:
            break
        with StepTimer(logger, f"[Q{q_idx}] Length Fixer {i+1}/{max_fix_iters} [{fixer_model}]"):
            fixer = call_agent_json(client, fixer_model, prompts["length_fixer"],
                {"final_text": final_text, "current_len": count, "target_min": target_min, "target_max": target_max,
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
            company_name=company_name, role=role, question=question,
            models_used=models_used, char_count=count, final_text=final_text,
            evidence_top=hits[:3], company_profile_path=company_profile_path,
            kb_dir=kb_dir, weak_evidence=weak, weak_reasons=reasons
        )

    logger.info("[Q%d] 완료: %s", q_idx, out_md)
    
    # 모델 사용 요약
    logger.info("[Q%d] 모델 사용: %s", q_idx, 
                ", ".join([f"{k}={v}" for k, v in models_used.items()]))

    if cfg.get("output", {}).get("print_final_to_terminal", False):
        print("\n" + "=" * 60)
        print(final_text)
        print("=" * 60)


# ================================
# 13) main
# ================================

def main():
    parser = argparse.ArgumentParser(description="자기소개서 생성")
    parser.add_argument("--config", default="config.yaml", help="설정 파일")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"[ERROR] 설정 파일 없음: {args.config}")
        return

    cfg = load_yaml(args.config)
    logger = setup_logger(cfg)

    # 모델 정책 로깅
    models = cfg.get("models", {})
    logger.info("모델 정책: high_quality=%s, standard=%s, fast=%s",
                models.get("high_quality", "gpt-5.2"),
                models.get("standard", "gpt-4o"),
                models.get("fast", "gpt-4o-mini"))

    app = cfg.get("application", {})
    questions = app.get("questions", None)

    if questions and isinstance(questions, list):
        logger.info("배치 실행: %d개 질문", len(questions))
        for i, q in enumerate(questions, 1):
            q = str(q).strip()
            if q:
                with StepTimer(logger, f"배치 [Q{i}/{len(questions)}]"):
                    run_for_question(cfg, logger, q, i)
    else:
        q = str(app.get("question", "")).strip()
        if not q:
            raise RuntimeError("config.yaml에 application.question 설정 필요")
        with StepTimer(logger, "단일 질문 실행"):
            run_for_question(cfg, logger, q, 1)


if __name__ == "__main__":
    main()
