#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_pass_sop_patterns.py
- 목적: 회사/직무 키워드 기반으로 공개 합격 자소서 소스를 자동 수집하고
        문장 복사 없이 구조적 흐름만 추출한다.
- 출력: temp/ 원문/정제 텍스트 + patterns/ 구조 패턴/가이드 저장
"""

import os
import re
import json
import hashlib
import argparse
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, parse_qs, unquote

import yaml
import requests
from bs4 import BeautifulSoup
from openai import OpenAI


OPENAI_API_KEY: str = ""  # 비워두면 환경변수 사용
DEFAULT_NOISE_KEYWORDS = [
    "로그인", "회원", "가입", "구독", "카테고리", "태그", "댓글", "공감",
    "목록", "이전글", "다음글", "저작권", "무단", "신고", "광고",
    "all rights reserved", "privacy", "terms", "cookie", "consent", "subscribe",
]
COMPANY_ALIAS_MAP = {
    "naver labs": ["네이버랩스", "네이버 랩스"],
    "naver": ["네이버"],
    "kakao": ["카카오"],
    "line": ["라인"],
    "coupang": ["쿠팡"],
    "toss": ["토스"],
    "baemin": ["배민", "배달의민족"],
    "samsung": ["삼성", "삼성전자"],
    "lg": ["LG", "엘지"],
    "sk": ["SK", "에스케이"],
    "hyundai": ["현대", "현대자동차", "현대차"],
}
ROLE_ALIAS_MAP = {
    "computer vision": ["컴퓨터 비전", "CV", "비전"],
    "vision": ["비전", "영상"],
    "generative": ["생성", "생성형"],
    "researcher": ["연구원", "연구직"],
    "research": ["연구", "R&D"],
    "artificial intelligence": ["인공지능", "AI"],
    "ai": ["AI", "인공지능"],
    "deep learning": ["딥러닝"],
    "machine learning": ["머신러닝", "ML"],
    "data scientist": ["데이터 사이언티스트", "데이터 과학자"],
    "data science": ["데이터 사이언스", "데이터"],
    "backend": ["백엔드", "서버"],
    "frontend": ["프론트엔드"],
    "full stack": ["풀스택"],
    "software engineer": ["소프트웨어 엔지니어", "개발자"],
    "engineer": ["엔지니어"],
}
QUESTION_CATEGORY_KEYWORDS = {
    "자기소개": ["자기소개", "자신", "자유롭게", "소개", "가치관", "강점", "약점", "성격", "성장과정"],
    "지원동기": ["지원동기", "지원 동기", "입사", "포부", "지원 이유", "동기", "왜"],
    "직무역량": ["직무역량", "역량", "프로젝트", "기술", "전문성", "성과", "문제 해결"],
    "협업": ["협업", "팀워크", "협력", "커뮤니케이션", "소통", "갈등", "리더십"],
    "성장과정": ["성장과정", "성장", "학습", "배움", "자기개발", "변화"],
    "도전": ["도전", "실패", "극복", "위기", "어려움", "개선", "혁신"],
}
QUESTION_CATEGORY_QUERIES = {
    "자기소개": ["자기소개", "성장과정", "강점", "가치관"],
    "지원동기": ["지원동기", "입사 후 포부", "회사 선택 기준"],
    "직무역량": ["직무역량", "프로젝트 경험", "문제 해결", "성과"],
    "협업": ["협업", "팀워크", "커뮤니케이션"],
    "성장과정": ["성장과정", "학습", "자기개발"],
    "도전": ["도전", "실패 극복", "문제 해결"],
}
DEFAULT_QUESTION_CATEGORIES = ["자기소개", "지원동기", "직무역량"]
DEFAULT_STEP_TEMPLATES = {
    "A": [
        "{company} {role} {question_kw} 합격 자소서",
        "{company} {role} {question_kw} 자기소개서 합격",
        "{company} {role} {question_kw} 서류 합격 자기소개서",
        "{company} {role} 합격 자소서",
        "{company} {role} 자기소개서 합격",
    ],
    "B": [
        "{role} {question_kw} 합격 자소서",
        "{role} {question_kw} 자기소개서 합격",
        "{role} {question_kw} 합격 후기 자기소개서",
        "{role} 합격 자소서",
        "{role} 자기소개서 합격",
    ],
    "C": [
        "{role} {question_kw} 합격 자소서",
        "{role} {question_kw} 자기소개서 합격",
        "{role} {question_kw} 합격 후기 자기소개서",
    ],
}


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_api_key(cfg: Dict[str, Any]) -> str:
    key = (cfg.get("openai", {}).get("api_key", "") or "").strip()
    if key:
        return key
    key = (OPENAI_API_KEY or "").strip()
    if key:
        return key
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    raise RuntimeError("OPENAI_API_KEY가 필요합니다. config.yaml 또는 환경변수로 설정하세요.")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def slugify(text: str) -> str:
    s = (text or "").strip().lower()
    s = re.sub(r"[^a-z0-9가-힣]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "item"


def clamp_text(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= max_chars else s[:max_chars].rstrip()


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def has_korean(s: str) -> bool:
    return bool(re.search(r"[가-힣]", s or ""))


def to_list(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(v).strip() for v in val if str(v).strip()]
    if isinstance(val, str):
        return [val.strip()] if val.strip() else []
    return []


def unique_list(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for it in items:
        if it in seen:
            continue
        seen.add(it)
        out.append(it)
    return out


def get_questions(cfg: Dict[str, Any]) -> List[str]:
    app = cfg.get("application", {})
    qs = []
    qs.extend(to_list(app.get("question", "")))
    qs.extend(to_list(app.get("questions", [])))
    return unique_list(qs)


def infer_question_categories(questions: List[str]) -> List[str]:
    found = []
    for q in questions:
        q_norm = q.strip()
        if not q_norm:
            continue
        for cat, keys in QUESTION_CATEGORY_KEYWORDS.items():
            if any(k in q_norm for k in keys):
                found.append(cat)
    return unique_list(found)


def build_question_keywords(
    categories: List[str],
    override_keywords: List[str],
    extra_keywords: List[str],
    max_keywords: int,
) -> List[str]:
    if override_keywords:
        base = override_keywords
    else:
        base = []
        for cat in categories:
            base.extend(QUESTION_CATEGORY_QUERIES.get(cat, []))
    if not base:
        for cat in DEFAULT_QUESTION_CATEGORIES:
            base.extend(QUESTION_CATEGORY_QUERIES.get(cat, []))
    base.extend(extra_keywords)
    base = unique_list([b for b in base if str(b).strip()])
    return base[:max_keywords] if max_keywords > 0 else base


def expand_company_terms(company: str, aliases: List[str], max_terms: int) -> List[str]:
    terms = [company]
    terms.extend(aliases)
    lower = (company or "").lower()
    for key, vals in COMPANY_ALIAS_MAP.items():
        if key in lower:
            terms.extend(vals)
    if " " in (company or ""):
        terms.append(company.replace(" ", ""))
    terms = unique_list([t for t in terms if str(t).strip()])
    if not has_korean(company):
        ko_terms = [t for t in terms if has_korean(t)]
        other_terms = [t for t in terms if not has_korean(t)]
        terms = ko_terms + other_terms
    return terms[:max_terms] if max_terms > 0 else terms


def expand_role_terms(role: str, role_keywords: List[str], max_terms: int) -> List[str]:
    terms = [role]
    terms.extend(role_keywords)
    role_lower = (role or "").lower()
    tokens = [t for t in re.split(r"[\s/_,-]+", role_lower) if t]

    for key, aliases in ROLE_ALIAS_MAP.items():
        if key in role_lower:
            terms.extend(aliases)

    if "computer" in tokens and "vision" in tokens:
        terms.extend(["computer vision", "컴퓨터 비전", "CV"])
    if "researcher" in tokens or "research" in tokens:
        terms.extend(["연구원", "연구직", "R&D"])
    if "ai" in tokens or "artificial" in tokens:
        terms.extend(["AI", "인공지능"])

    if not any(has_korean(t) for t in terms):
        terms.extend(["연구원", "AI 연구"])

    terms = unique_list([t for t in terms if str(t).strip()])
    if not has_korean(role):
        ko_terms = [t for t in terms if has_korean(t)]
        other_terms = [t for t in terms if not has_korean(t)]
        terms = ko_terms + other_terms
    return terms[:max_terms] if max_terms > 0 else terms


def build_queries_from_templates(
    templates: List[str],
    company_terms: List[str],
    role_terms: List[str],
    question_keywords: List[str],
    max_queries: int,
) -> List[str]:
    qs = []
    use_company = any("{company}" in t for t in templates)
    companies = company_terms if use_company else [""]
    for company_term in companies:
        for role_term in role_terms:
            for t in templates:
                if "{question_kw}" in t:
                    for kw in question_keywords:
                        qs.append(t.format(company=company_term, role=role_term, question_kw=kw))
                else:
                    qs.append(t.format(company=company_term, role=role_term))
    qs = unique_list([q.strip() for q in qs if q.strip()])
    if max_queries > 0:
        return qs[:max_queries]
    return qs


def normalize_ddg_url(url: str) -> str:
    if not url:
        return url
    try:
        parsed = urlparse(url)
        if "duckduckgo.com" in parsed.netloc and parsed.path == "/l/":
            qs = parse_qs(parsed.query)
            if "uddg" in qs and qs["uddg"]:
                return unquote(qs["uddg"][0])
    except Exception:
        pass
    return url


def search_duckduckgo(query: str, max_results: int, timeout: int) -> List[Dict[str, str]]:
    headers = {"User-Agent": "Mozilla/5.0"}
    params = {"q": query}
    r = requests.get("https://html.duckduckgo.com/html/", params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    results = []
    for res in soup.select("div.result"):
        a = res.select_one("a.result__a")
        if not a:
            continue
        url = normalize_ddg_url(a.get("href", ""))
        title = a.get_text(" ", strip=True)
        snippet = ""
        sn = res.select_one(".result__snippet")
        if sn:
            snippet = sn.get_text(" ", strip=True)
        results.append({"url": url, "title": title, "snippet": snippet})
        if len(results) >= max_results:
            break
    return results


def get_domain(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def check_stop(stats: Dict[str, int], criteria: Dict[str, int], max_sources: int) -> str:
    if stats["sample_count"] >= criteria["min_samples"]:
        return "min_samples"
    if stats["total_chars"] >= criteria["min_total_chars"]:
        return "min_total_chars"
    if stats["unique_domains"] >= criteria["min_unique_domains"]:
        return "min_unique_domains"
    if stats["sample_count"] >= max_sources:
        return "max_sources"
    return ""


def build_failure_reasons(stats: Dict[str, int], criteria: Dict[str, int]) -> List[str]:
    reasons = []
    if stats["sample_count"] < criteria["min_samples"]:
        reasons.append(f"샘플 수 부족: {stats['sample_count']} < {criteria['min_samples']}")
    if stats["total_chars"] < criteria["min_total_chars"]:
        reasons.append(f"본문 총 길이 부족: {stats['total_chars']} < {criteria['min_total_chars']}")
    if stats["unique_domains"] < criteria["min_unique_domains"]:
        reasons.append(f"도메인 다양성 부족: {stats['unique_domains']} < {criteria['min_unique_domains']}")
    return reasons


def build_fallback_flow_guide() -> Dict[str, Any]:
    return {
        "common_flow": [
            "지원 동기와 관심 계기 제시",
            "문제/목표 인식과 개인 역할 정의",
            "행동/전략/협업 방식 서술",
            "결과/성과와 배운 점 정리",
            "회사/직무와의 연결 및 기여 계획",
        ],
        "section_roles": [
            {"name": "동기", "role": "지원 맥락과 관심 계기를 제시", "position": "intro"},
            {"name": "문제/목표", "role": "해결하려는 과제와 기준을 정의", "position": "mid"},
            {"name": "행동", "role": "접근 방식, 역할, 협업을 설명", "position": "mid"},
            {"name": "결과", "role": "성과와 학습을 간결히 정리", "position": "mid"},
            {"name": "연결", "role": "회사/직무와의 적합성 및 포부", "position": "ending"},
        ],
        "paragraph_count_range": [4, 6],
        "tips": [
            "경험·근거 중심으로 구체성을 확보",
            "결과 뒤에 배운 점과 재현 가능성을 제시",
            "직무 키워드를 마지막에 자연스럽게 연결",
        ],
    }


def hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def noise_ratio(s: str) -> float:
    if not s:
        return 1.0
    noise = re.sub(r"[a-zA-Z0-9가-힣\s]", "", s)
    return len(noise) / max(1, len(s))


def split_paragraphs(text: str) -> List[str]:
    text = re.sub(r"\n{3,}", "\n\n", (text or ""))
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def select_relevant_paragraphs(
    text: str,
    min_chars: int,
    max_chars: int,
    max_paragraphs: int,
    max_total_chars: int,
    noise_keywords: List[str],
) -> str:
    keywords = [str(k).lower() for k in (noise_keywords or []) if str(k).strip()]
    selected = []
    total = 0
    for p in split_paragraphs(text):
        p_clean = p.strip()
        if len(p_clean) < min_chars:
            continue
        if noise_ratio(p_clean) > 0.35:
            continue
        p_lower = p_clean.lower()
        if any(k in p_lower for k in keywords):
            continue
        if re.search(r"[-=*_]{4,}", p_clean):
            continue
        if len(p_clean) > max_chars:
            p_clean = p_clean[:max_chars].rstrip()
        if total + len(p_clean) + 2 > max_total_chars:
            break
        selected.append(p_clean)
        total += len(p_clean) + 2
        if len(selected) >= max_paragraphs:
            break
    if not selected:
        return clamp_text(text, max_total_chars)
    return "\n\n".join(selected)


def iter_strings(obj: Any):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from iter_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from iter_strings(v)
    elif isinstance(obj, str):
        yield obj


def has_verbatim_overlap(obj: Any, source_text: str, min_chars: int) -> bool:
    src = normalize_ws(source_text).lower()
    for s in iter_strings(obj):
        s_norm = normalize_ws(s).lower()
        if len(s_norm) >= min_chars and s_norm in src:
            return True
    return False


def looks_like_login(text: str) -> bool:
    t = (text or "").lower()
    keywords = [
        "로그인", "회원", "가입", "sign in", "log in", "signin", "login required",
        "권한", "접근 권한", "멤버 전용", "유료", "구독",
    ]
    return any(k in t for k in keywords)


def fetch_url(url: str, timeout: int = 15) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text


def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_json_fallback(txt: str) -> Dict[str, Any]:
    txt = (txt or "").strip()
    if txt.startswith("\ufeff"):
        txt = txt[1:]
    if txt.lower().startswith("json"):
        txt = txt[4:].lstrip()
    try:
        return json.loads(txt)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", txt, flags=re.S | re.I)
    if m:
        return json.loads(m.group(1))
    start, end = txt.find("{"), txt.rfind("}")
    if start != -1 and end > start:
        return json.loads(txt[start:end + 1])
    raise ValueError("JSON 파싱 실패")


def call_json_object(
    client: OpenAI,
    model: str,
    instructions: str,
    input_text: str,
    max_tokens: int,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Any] = None,
) -> Dict[str, Any]:
    inp = "json\n" + (input_text or "")
    kwargs = {
        "model": model,
        "instructions": instructions,
        "input": inp,
        "max_output_tokens": max_tokens,
        "text": {"format": {"type": "json_object"}},
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice or "auto"

    resp = client.responses.create(**kwargs)
    txt = (getattr(resp, "output_text", "") or "").strip()
    if not txt:
        outputs = getattr(resp, "output", None)
        if outputs:
            for out in outputs:
                if getattr(out, "type", None) == "message":
                    for c in getattr(out, "content", []) or []:
                        if getattr(c, "type", None) == "output_text":
                            txt = (getattr(c, "text", "") or "").strip()
                            break
    return parse_json_fallback(txt)


def build_queries(company: str, role: str, keywords: List[str], templates: List[str]) -> List[str]:
    qs = []
    base = {"company": company, "role": role}
    for t in templates:
        qs.append(t.format(**base))
    for kw in keywords:
        qs.append(f"{company} {role} {kw} 합격 자소서")
        qs.append(f"{company} {kw} 자기소개서 합격")
    return list(dict.fromkeys([q.strip() for q in qs if q.strip()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ps = cfg.get("pass_sop_patterns", {})
    if not ps.get("enable", True):
        print("[SKIP] pass_sop_patterns 비활성화")
        return

    company = cfg.get("company", {}).get("name", "")
    role = cfg.get("company", {}).get("role", "")
    if not company or not role:
        raise RuntimeError("company.name / company.role 설정이 필요합니다.")

    temp_dir = ps.get("temp_dir", "temp")
    raw_dir = ps.get("output", {}).get("raw_dir", os.path.join(temp_dir, "essays_raw"))
    clean_dir = ps.get("output", {}).get("clean_dir", os.path.join(temp_dir, "essays_clean"))
    structured_jsonl = ps.get("output", {}).get("structured_jsonl", os.path.join(temp_dir, "essays_structured.jsonl"))
    flow_template_json = ps.get("output", {}).get("flow_template_json", os.path.join(temp_dir, "flow_templates.json"))
    flow_guide_md = ps.get("output", {}).get("flow_guide_md", os.path.join(temp_dir, "flow_guide.md"))

    ensure_dir(temp_dir)
    ensure_dir(raw_dir)
    ensure_dir(clean_dir)

    patterns_dir = ps.get("output", {}).get("patterns_dir", "patterns")
    sources_dir = ps.get("output", {}).get("sources_dir", os.path.join(patterns_dir, "sources"))
    patterns_json = ps.get("output", {}).get("patterns_json", os.path.join(patterns_dir, "pass_sop_patterns.json"))
    patterns_md = ps.get("output", {}).get("patterns_md", os.path.join(patterns_dir, "pass_sop_patterns.md"))
    search_log_json = ps.get("output", {}).get("search_log_json", os.path.join(sources_dir, "search_log.json"))
    sources_jsonl = ps.get("output", {}).get("sources_jsonl", os.path.join(sources_dir, "sources.jsonl"))

    ensure_dir(patterns_dir)
    ensure_dir(sources_dir)

    max_sources = int(ps.get("max_sources", 8))
    min_text_chars = int(ps.get("min_text_chars", 800))
    max_input_chars = int(ps.get("max_input_chars", 12000))
    min_extract_chars = int(ps.get("min_extract_chars", max(400, min_text_chars // 2)))
    min_paragraph_chars = int(ps.get("min_paragraph_chars", 80))
    max_paragraph_chars = int(ps.get("max_paragraph_chars", 1200))
    max_paragraphs = int(ps.get("max_paragraphs", 40))
    overlap_min_chars = int(ps.get("overlap_min_chars", 40))
    max_question_keywords = int(ps.get("max_question_keywords", 8))
    max_queries_per_step = int(ps.get("max_queries_per_step", 10))
    max_company_terms = int(ps.get("max_company_terms", 4))
    max_role_terms = int(ps.get("max_role_terms", 6))
    use_openai_search = bool(ps.get("use_openai_search", True))
    enable_ddg_fallback = bool(ps.get("enable_ddg_fallback", True))
    ddg_max_results = int(ps.get("ddg_max_results", 8))
    ddg_timeout = int(ps.get("ddg_timeout", 15))

    stop_cfg = ps.get("stop_criteria", {})
    stop_criteria = {
        "min_samples": int(stop_cfg.get("min_samples", min(6, max_sources))),
        "min_total_chars": int(stop_cfg.get("min_total_chars", 8000)),
        "min_unique_domains": int(stop_cfg.get("min_unique_domains", 3)),
    }

    extra_noise = ps.get("noise_keywords", [])
    if isinstance(extra_noise, str):
        extra_noise = [extra_noise]
    if not isinstance(extra_noise, list):
        extra_noise = []
    noise_keywords = list(dict.fromkeys(
        [k for k in (DEFAULT_NOISE_KEYWORDS + extra_noise) if str(k).strip()]
    ))

    questions = get_questions(cfg)
    question_categories = infer_question_categories(questions)

    role_keywords = to_list(ps.get("role_keywords"))
    if not role_keywords:
        role_keywords = to_list(cfg.get("company", {}).get("role_keywords"))
    role_terms = expand_role_terms(role, role_keywords, max_role_terms)

    company_aliases = to_list(ps.get("company_aliases"))
    if not company_aliases:
        company_aliases = to_list(cfg.get("company", {}).get("aliases"))
    company_terms = expand_company_terms(company, company_aliases, max_company_terms)

    question_keywords_override = to_list(ps.get("question_keywords"))
    question_keywords_extra = to_list(ps.get("keywords"))
    question_keywords = build_question_keywords(
        question_categories,
        question_keywords_override,
        question_keywords_extra,
        max_question_keywords,
    )

    category_keywords = []
    for cat in (question_categories or DEFAULT_QUESTION_CATEGORIES):
        category_keywords.extend(QUESTION_CATEGORY_QUERIES.get(cat, []))
    category_keywords = unique_list([k for k in category_keywords if str(k).strip()])
    if max_question_keywords > 0:
        category_keywords = category_keywords[:max_question_keywords]

    search_templates_cfg = ps.get("search_templates", {})
    if not isinstance(search_templates_cfg, dict):
        search_templates_cfg = {}
    step_a_templates = to_list(search_templates_cfg.get("step_a") or search_templates_cfg.get("A"))
    step_b_templates = to_list(search_templates_cfg.get("step_b") or search_templates_cfg.get("B"))
    step_c_templates = to_list(search_templates_cfg.get("step_c") or search_templates_cfg.get("C"))
    if not step_a_templates:
        step_a_templates = list(DEFAULT_STEP_TEMPLATES["A"])
    if not step_b_templates:
        step_b_templates = list(DEFAULT_STEP_TEMPLATES["B"])
    if not step_c_templates:
        step_c_templates = list(DEFAULT_STEP_TEMPLATES["C"])

    legacy_templates = to_list(ps.get("search_queries"))
    if legacy_templates:
        step_a_templates = unique_list(step_a_templates + legacy_templates)

    step_a_queries = build_queries_from_templates(
        step_a_templates, company_terms, role_terms, question_keywords, max_queries_per_step
    )
    step_b_queries = build_queries_from_templates(
        step_b_templates, company_terms, role_terms, question_keywords, max_queries_per_step
    )
    step_c_queries = build_queries_from_templates(
        step_c_templates, company_terms, role_terms, category_keywords, max_queries_per_step
    )

    models = cfg.get("models", {})
    search_model = models.get("standard", "gpt-4o")
    extract_model = models.get("high_quality", "gpt-5.2")

    client = OpenAI(api_key=resolve_api_key(cfg))

    # 1) web_search로 URL 수집 (계층적 폴백)
    collected: List[Dict[str, Any]] = []
    cleaned: List[Dict[str, Any]] = []
    sources_log: List[Dict[str, Any]] = []
    search_log: Dict[str, Any] = {
        "company": company,
        "role": role,
        "questions": questions,
        "question_categories": question_categories,
        "role_terms": role_terms,
        "company_terms": company_terms,
        "steps": [],
    }

    seen_urls = set()
    seen_text_hash = set()
    domain_set = set()
    stats = {"sample_count": 0, "total_chars": 0, "unique_domains": 0}
    stop_reason = ""

    search_inst = (
        "너는 웹에서 공개 합격 자소서 페이지를 찾는 에이전트다.\n"
        "주어진 검색어로 web_search 도구를 사용해 공개 페이지를 찾아라.\n"
        "반드시 web_search 도구를 호출해라.\n"
        "로그인/유료/비공개 페이지는 제외하고, 합격 자소서/합격 후기/자기소개서 본문이 있는 글만 포함하라.\n"
        "출력은 JSON 하나만: {items:[{url,title,snippet}]}\n"
        "items는 최대 8개로 제한해라.\n"
    )

    steps = [
        {"id": "A", "name": "질문+회사+직무", "queries": step_a_queries},
        {"id": "B", "name": "질문+직무", "queries": step_b_queries},
        {"id": "C", "name": "직무+유사문항", "queries": step_c_queries},
    ]

    for idx, step in enumerate(steps):
        if stop_reason:
            break
        queries = step["queries"]
        if not queries:
            continue

        step_log = {
            "step": step["id"],
            "name": step["name"],
            "queries": [],
            "raw_results": 0,
            "saved_samples": 0,
            "backend_counts": {"openai_web_search": 0, "duckduckgo": 0},
        }

        for q in queries:
            step_log["queries"].append(q)
            items = []
            backend = ""
            if use_openai_search:
                try:
                    resp = call_json_object(
                        client,
                        search_model,
                        search_inst,
                        q,
                        max_tokens=900,
                        tools=[{"type": "web_search"}],
                        tool_choice="required",
                    )
                    items = resp.get("items", []) if isinstance(resp, dict) else []
                    backend = "openai_web_search"
                except Exception:
                    items = []
                    backend = "openai_web_search"

            if not items and enable_ddg_fallback:
                try:
                    items = search_duckduckgo(q, ddg_max_results, ddg_timeout)
                    backend = "duckduckgo"
                except Exception:
                    items = []
                    backend = "duckduckgo"

            step_log["raw_results"] += len(items)
            if backend:
                step_log["backend_counts"][backend] = step_log["backend_counts"].get(backend, 0) + len(items)

            for it in items:
                url = (it.get("url") or "").strip()
                title = it.get("title", "")
                snippet = it.get("snippet", "")
                if not url or not url.startswith("http"):
                    sources_log.append({
                        "url": url,
                        "title": title,
                        "snippet": snippet,
                        "query": q,
                        "step": step["id"],
                        "backend": backend,
                        "status": "skip_invalid_url",
                    })
                    continue
                if url in seen_urls:
                    sources_log.append({
                        "url": url,
                        "title": title,
                        "snippet": snippet,
                        "query": q,
                        "step": step["id"],
                        "backend": backend,
                        "status": "skip_duplicate_url",
                    })
                    continue
                seen_urls.add(url)
                collected.append({"url": url, "title": title, "snippet": snippet, "query": q, "step": step["id"]})

                try:
                    html = fetch_url(url)
                except Exception:
                    sources_log.append({
                        "url": url,
                        "title": title,
                        "snippet": snippet,
                        "query": q,
                        "step": step["id"],
                        "backend": backend,
                        "status": "fetch_error",
                    })
                    continue

                text = extract_text_from_html(html)
                if len(text) < min_text_chars:
                    sources_log.append({
                        "url": url,
                        "title": title,
                        "snippet": snippet,
                        "query": q,
                        "step": step["id"],
                        "backend": backend,
                        "status": "skip_too_short",
                        "text_chars": len(text),
                    })
                    continue
                if looks_like_login(text):
                    sources_log.append({
                        "url": url,
                        "title": title,
                        "snippet": snippet,
                        "query": q,
                        "step": step["id"],
                        "backend": backend,
                        "status": "skip_login",
                    })
                    continue

                focus_text = select_relevant_paragraphs(
                    text,
                    min_chars=min_paragraph_chars,
                    max_chars=max_paragraph_chars,
                    max_paragraphs=max_paragraphs,
                    max_total_chars=max_input_chars,
                    noise_keywords=noise_keywords,
                )
                if len(focus_text) < min_extract_chars:
                    sources_log.append({
                        "url": url,
                        "title": title,
                        "snippet": snippet,
                        "query": q,
                        "step": step["id"],
                        "backend": backend,
                        "status": "skip_low_extract",
                        "text_chars": len(focus_text),
                    })
                    continue

                text_hash = hash_text(normalize_ws(focus_text))
                if text_hash in seen_text_hash:
                    sources_log.append({
                        "url": url,
                        "title": title,
                        "snippet": snippet,
                        "query": q,
                        "step": step["id"],
                        "backend": backend,
                        "status": "skip_duplicate_text",
                    })
                    continue
                seen_text_hash.add(text_hash)

                base = slugify(title or url)
                fid = f"{base}-{hash_text(url)}"
                raw_path = os.path.join(raw_dir, f"{fid}.html")
                clean_path = os.path.join(clean_dir, f"{fid}.txt")

                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(html)
                with open(clean_path, "w", encoding="utf-8") as f:
                    f.write(focus_text)

                domain = get_domain(url)
                if domain:
                    domain_set.add(domain)

                cleaned.append({
                    "url": url,
                    "title": title,
                    "snippet": snippet,
                    "text_path": clean_path,
                    "text_hash": text_hash,
                    "domain": domain,
                    "text_chars": len(focus_text),
                    "query": q,
                    "step": step["id"],
                })
                sources_log.append({
                    "url": url,
                    "title": title,
                    "snippet": snippet,
                    "query": q,
                    "step": step["id"],
                    "backend": backend,
                    "status": "saved",
                    "text_chars": len(focus_text),
                    "domain": domain,
                    "text_path": clean_path,
                })

                stats["sample_count"] += 1
                stats["total_chars"] += len(focus_text)
                stats["unique_domains"] = len(domain_set)
                step_log["saved_samples"] += 1

                stop_reason = check_stop(stats, stop_criteria, max_sources)
                if stop_reason:
                    break
            if stop_reason:
                break
        search_log["steps"].append(step_log)

        print(f"Step {step['id']} 쿼리 {len(step_log['queries'])}개 실행 → 결과 {step_log['raw_results']}개")
        if stop_reason:
            print(f"유효 샘플 {stats['sample_count']}개 확보 → Step {step['id']} 종료")
            break
        if any(s.get("queries") for s in steps[idx + 1:]):
            print(f"Step {step['id']} 실패 → 다음 Step으로 폴백")

    criteria_met = stop_reason in {"min_samples", "min_total_chars", "min_unique_domains"}
    search_log["stop_reason"] = stop_reason or "none"
    search_log["criteria_met"] = criteria_met
    search_log["criteria"] = stop_criteria
    search_log["stats"] = stats
    search_log["collected_count"] = len(collected)
    search_log["cleaned_count"] = len(cleaned)

    if sources_jsonl:
        with open(sources_jsonl, "w", encoding="utf-8") as f:
            for row in sources_log:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if search_log_json:
        with open(search_log_json, "w", encoding="utf-8") as f:
            json.dump(search_log, f, ensure_ascii=False, indent=2)

    # 3) 구조 패턴 추출
    extract_inst = (
        "너는 합격 자소서에서 '구조적 흐름'만 추출하는 에이전트다.\n"
        "문장/표현/사례를 복사하지 말고, 흐름/구성/전개 패턴만 요약해라.\n"
        "고유명사/수치/기간/회사명/학교명/프로젝트명 등 구체 정보 금지.\n"
        "섹션명은 일반명사(동기/문제/행동/결과/회사 연결 등)로만 작성.\n"
        "출력은 JSON 하나만:\n"
        "{flow:[str], sections:[{name, role, position}], paragraph_count:int, emphasis_positions:[str], notes:str}\n"
        "position은 intro/mid/ending 중 하나. notes는 한 문장.\n"
    )
    extract_inst_strict = extract_inst + (
        "추가 조건: 입력 텍스트의 표현을 재사용하지 마라.\n"
        "각 항목은 20자 내외로 짧게.\n"
    )

    records = []
    for it in cleaned:
        with open(it["text_path"], "r", encoding="utf-8") as f:
            text = f.read()
        text = clamp_text(text, max_input_chars)
        try:
            data = call_json_object(client, extract_model, extract_inst, text, max_tokens=900)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if has_verbatim_overlap(data, text, overlap_min_chars):
            try:
                data = call_json_object(client, extract_model, extract_inst_strict, text, max_tokens=900)
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            if has_verbatim_overlap(data, text, overlap_min_chars):
                continue
        data["url"] = it["url"]
        data["title"] = it.get("title", "")
        data["step"] = it.get("step", "")
        data["query"] = it.get("query", "")
        records.append(data)

    if records:
        with open(structured_jsonl, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 4) 템플릿 요약
    summary_inst = (
        "너는 여러 개의 구조 패턴을 통합해 '자기소개서 흐름 가이드'를 만드는 에이전트다.\n"
        "문장 복사 금지. 구조/흐름만 요약해라.\n"
        "고유명사/수치/사례/경험 상세는 절대 포함하지 마라.\n"
        "출력 JSON:\n"
        "{common_flow:[str], section_roles:[{name, role, position}], paragraph_count_range:[min,max], tips:[str]}\n"
    )

    flow_guide = {}
    if records:
        try:
            flow_guide = call_json_object(
                client,
                extract_model,
                summary_inst,
                json.dumps(records, ensure_ascii=False),
                max_tokens=900,
            )
        except Exception:
            flow_guide = {}
    if not isinstance(flow_guide, dict):
        flow_guide = {}

    fallback_flow = build_fallback_flow_guide()
    flow_guide_final = flow_guide if (criteria_met and flow_guide) else fallback_flow
    use_fallback = (not criteria_met) or (not flow_guide)

    with open(flow_template_json, "w", encoding="utf-8") as f:
        json.dump({
            "company": company,
            "role": role,
            "source_count": len(records),
            "criteria_met": criteria_met,
            "stop_reason": stop_reason or "none",
            "flow_guide": flow_guide_final,
        }, f, ensure_ascii=False, indent=2)

    # Markdown 가이드 저장
    md_lines = []
    md_lines.append("# 합격 자소서 흐름 가이드 (구조 요약)\n\n")
    md_lines.append("※ 문장/표현 복사 금지, 구조적 흐름만 참고\n\n")
    if flow_guide_final.get("common_flow"):
        md_lines.append("## 공통 흐름\n")
        for s in flow_guide_final.get("common_flow", []):
            md_lines.append(f"- {s}\n")
        md_lines.append("\n")
    if flow_guide_final.get("section_roles"):
        md_lines.append("## 섹션 역할\n")
        for s in flow_guide_final.get("section_roles", []):
            name = s.get("name", "")
            role = s.get("role", "")
            pos = s.get("position", "")
            md_lines.append(f"- {name} ({pos}): {role}\n")
        md_lines.append("\n")
    if flow_guide_final.get("paragraph_count_range"):
        rng = flow_guide_final.get("paragraph_count_range", [])
        if len(rng) == 2:
            md_lines.append(f"## 문단 수 범위\n- {rng[0]} ~ {rng[1]}\n\n")
    if flow_guide_final.get("tips"):
        md_lines.append("## 작성 팁\n")
        for t in flow_guide_final.get("tips", []):
            md_lines.append(f"- {t}\n")
        md_lines.append("\n")

    with open(flow_guide_md, "w", encoding="utf-8") as f:
        f.write("".join(md_lines))

    # patterns/pass_sop_patterns.json|md 저장
    failure_reasons = build_failure_reasons(stats, stop_criteria) if not criteria_met else []
    patterns_payload = {
        "company": company,
        "role": role,
        "questions": questions,
        "source_count": len(records),
        "criteria_met": criteria_met,
        "stop_reason": stop_reason or "none",
        "stats": stats,
        "failure_reasons": failure_reasons,
        "flow_guide": flow_guide_final,
        "fallback_used": use_fallback,
        "fallback_guide": fallback_flow if use_fallback else {},
        "search_log_path": search_log_json,
        "sources_path": sources_jsonl,
    }
    with open(patterns_json, "w", encoding="utf-8") as f:
        json.dump(patterns_payload, f, ensure_ascii=False, indent=2)

    pm = []
    pm.append("# 합격 자소서 구조 패턴 가이드\n\n")
    pm.append("※ 문장/표현 복사 금지, 구조적 흐름만 참고\n\n")
    pm.append("## 수집 결과 요약\n")
    pm.append(f"- 샘플 수: {stats['sample_count']}\n")
    pm.append(f"- 본문 총 길이: {stats['total_chars']}\n")
    pm.append(f"- 도메인 수: {stats['unique_domains']}\n")
    pm.append(f"- 기준 충족: {'예' if criteria_met else '아니오'}\n")
    pm.append(f"- 종료 사유: {stop_reason or 'none'}\n\n")
    if failure_reasons:
        pm.append("## 수집 실패/부족 원인\n")
        for r in failure_reasons:
            pm.append(f"- {r}\n")
        pm.append("\n")

    pm.append("## 실행한 검색 쿼리\n")
    for step in search_log.get("steps", []):
        pm.append(f"### Step {step.get('step')} ({step.get('name')})\n")
        for q in step.get("queries", []):
            pm.append(f"- {q}\n")
        pm.append("\n")

    if flow_guide:
        pm.append("## 추출된 흐름 요약\n")
        if flow_guide.get("common_flow"):
            pm.append("### 공통 흐름\n")
            for s in flow_guide.get("common_flow", []):
                pm.append(f"- {s}\n")
            pm.append("\n")
        if flow_guide.get("section_roles"):
            pm.append("### 섹션 역할\n")
            for s in flow_guide.get("section_roles", []):
                name = s.get("name", "")
                role = s.get("role", "")
                pos = s.get("position", "")
                pm.append(f"- {name} ({pos}): {role}\n")
            pm.append("\n")

    if use_fallback:
        pm.append("## 대체 가이드 (일반 구조 템플릿)\n")
        if fallback_flow.get("common_flow"):
            pm.append("### 공통 흐름\n")
            for s in fallback_flow.get("common_flow", []):
                pm.append(f"- {s}\n")
            pm.append("\n")
        if fallback_flow.get("section_roles"):
            pm.append("### 섹션 역할\n")
            for s in fallback_flow.get("section_roles", []):
                name = s.get("name", "")
                role = s.get("role", "")
                pos = s.get("position", "")
                pm.append(f"- {name} ({pos}): {role}\n")
            pm.append("\n")
        if fallback_flow.get("paragraph_count_range"):
            rng = fallback_flow.get("paragraph_count_range", [])
            if len(rng) == 2:
                pm.append(f"### 문단 수 범위\n- {rng[0]} ~ {rng[1]}\n\n")
        if fallback_flow.get("tips"):
            pm.append("### 작성 팁\n")
            for t in fallback_flow.get("tips", []):
                pm.append(f"- {t}\n")
            pm.append("\n")

    with open(patterns_md, "w", encoding="utf-8") as f:
        f.write("".join(pm))

    print("[OK] pass_sop_patterns collected:", len(records))
    print("-", structured_jsonl)
    print("-", flow_template_json)
    print("-", flow_guide_md)
    print("-", patterns_json)
    print("-", patterns_md)


if __name__ == "__main__":
    main()
