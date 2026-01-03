#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_pass_sop_patterns.py
- 목적: 회사/직무 키워드 기반으로 공개 합격 자소서 소스를 자동 수집하고
        문장 복사 없이 구조적 흐름만 추출한다.
- 출력: temp/ 하위 폴더에 원문/정제 텍스트/구조 패턴/가이드 저장
"""

import os
import re
import json
import hashlib
import argparse
from typing import List, Dict, Any, Optional

import yaml
import requests
from bs4 import BeautifulSoup
from openai import OpenAI


OPENAI_API_KEY: str = ""  # 비워두면 환경변수 사용


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


def hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


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
        kwargs["tool_choice"] = "auto"

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

    max_sources = int(ps.get("max_sources", 8))
    min_text_chars = int(ps.get("min_text_chars", 800))
    max_input_chars = int(ps.get("max_input_chars", 12000))

    templates = ps.get("search_queries", [
        "{company} {role} 합격 자소서",
        "{company} {role} 자기소개서 합격",
        "{company} {role} 합격 후기 자기소개서",
    ])
    keywords = ps.get("keywords", [])
    queries = build_queries(company, role, keywords, templates)

    models = cfg.get("models", {})
    search_model = models.get("standard", "gpt-4o")
    extract_model = models.get("high_quality", "gpt-5.2")

    client = OpenAI(api_key=resolve_api_key(cfg))

    # 1) web_search로 URL 수집
    collected: List[Dict[str, Any]] = []
    seen = set()

    search_inst = (
        "너는 웹에서 공개 합격 자소서 페이지를 찾는 에이전트다.\n"
        "주어진 검색어로 web_search 도구를 사용해 공개 페이지를 찾아라.\n"
        "로그인/유료/비공개 페이지는 제외하고, 합격 자소서/합격 후기/자기소개서 본문이 있는 글만 포함하라.\n"
        "출력은 JSON 하나만: {items:[{url,title,snippet}]}\n"
        "items는 최대 8개로 제한해라.\n"
    )

    for q in queries:
        try:
            resp = call_json_object(
                client,
                search_model,
                search_inst,
                q,
                max_tokens=900,
                tools=[{"type": "web_search"}],
            )
        except Exception:
            continue

        items = resp.get("items", []) if isinstance(resp, dict) else []
        for it in items:
            url = (it.get("url") or "").strip()
            if not url or url in seen:
                continue
            if not url.startswith("http"):
                continue
            seen.add(url)
            collected.append({"url": url, "title": it.get("title", ""), "snippet": it.get("snippet", ""), "query": q})
            if len(collected) >= max_sources:
                break
        if len(collected) >= max_sources:
            break

    # 2) 페이지 다운로드 + 텍스트 정제
    cleaned: List[Dict[str, Any]] = []
    for it in collected:
        url = it["url"]
        try:
            html = fetch_url(url)
        except Exception:
            continue
        text = extract_text_from_html(html)
        if len(text) < min_text_chars:
            continue
        if looks_like_login(text):
            continue

        base = slugify(it.get("title") or url)
        fid = f"{base}-{hash_text(url)}"
        raw_path = os.path.join(raw_dir, f"{fid}.html")
        clean_path = os.path.join(clean_dir, f"{fid}.txt")

        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(html)
        with open(clean_path, "w", encoding="utf-8") as f:
            f.write(text)

        cleaned.append({
            "url": url,
            "title": it.get("title", ""),
            "snippet": it.get("snippet", ""),
            "text_path": clean_path,
        })

    # 3) 구조 패턴 추출
    extract_inst = (
        "너는 합격 자소서에서 '구조적 흐름'만 추출하는 에이전트다.\n"
        "문장/표현/사례를 복사하지 말고, 흐름/구성/전개 패턴만 요약해라.\n"
        "출력은 JSON 하나만:\n"
        "{flow:[str], sections:[{name, role, position}], paragraph_count:int, emphasis_positions:[str], notes:str}\n"
        "position은 intro/mid/ending 중 하나. notes는 한 문장.\n"
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
        data["url"] = it["url"]
        data["title"] = it.get("title", "")
        records.append(data)

    if records:
        with open(structured_jsonl, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 4) 템플릿 요약
    summary_inst = (
        "너는 여러 개의 구조 패턴을 통합해 '자기소개서 흐름 가이드'를 만드는 에이전트다.\n"
        "문장 복사 금지. 구조/흐름만 요약해라.\n"
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

    with open(flow_template_json, "w", encoding="utf-8") as f:
        json.dump({
            "company": company,
            "role": role,
            "source_count": len(records),
            "flow_guide": flow_guide,
        }, f, ensure_ascii=False, indent=2)

    # Markdown 가이드 저장
    md_lines = []
    md_lines.append("# 합격 자소서 흐름 가이드 (구조 요약)\n\n")
    md_lines.append("※ 문장/표현 복사 금지, 구조적 흐름만 참고\n\n")
    if flow_guide.get("common_flow"):
        md_lines.append("## 공통 흐름\n")
        for s in flow_guide.get("common_flow", []):
            md_lines.append(f"- {s}\n")
        md_lines.append("\n")
    if flow_guide.get("section_roles"):
        md_lines.append("## 섹션 역할\n")
        for s in flow_guide.get("section_roles", []):
            name = s.get("name", "")
            role = s.get("role", "")
            pos = s.get("position", "")
            md_lines.append(f"- {name} ({pos}): {role}\n")
        md_lines.append("\n")
    if flow_guide.get("paragraph_count_range"):
        rng = flow_guide.get("paragraph_count_range", [])
        if len(rng) == 2:
            md_lines.append(f"## 문단 수 범위\n- {rng[0]} ~ {rng[1]}\n\n")
    if flow_guide.get("tips"):
        md_lines.append("## 작성 팁\n")
        for t in flow_guide.get("tips", []):
            md_lines.append(f"- {t}\n")
        md_lines.append("\n")

    with open(flow_guide_md, "w", encoding="utf-8") as f:
        f.write("".join(md_lines))

    print("[OK] pass_sop_patterns collected:", len(records))
    print("-", structured_jsonl)
    print("-", flow_template_json)
    print("-", flow_guide_md)


if __name__ == "__main__":
    main()
