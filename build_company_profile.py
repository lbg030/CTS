

import os
import re
import json
import argparse
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

import yaml
import requests
from bs4 import BeautifulSoup
from openai import OpenAI


# ================================
# 0) 사용자 설정 (코드에서 직접 입력 가능)
# ================================
OPENAI_API_KEY: str = ""
OPENAI_MODEL: str = "gpt-5.2"  # 기본 GPT-5.2


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_api_key() -> str:
    key = (OPENAI_API_KEY or "").strip()
    if key:
        return key
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    raise RuntimeError("OPENAI_API_KEY가 필요합니다. 코드 상단에 넣거나 환경변수로 설정하세요.")


def slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9가-힣]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "company"


def clamp_text(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= max_chars else s[:max_chars].rstrip()


def fetch_url_text(url: str, timeout: int = 15) -> str:
    """
    공고 URL에서 텍스트를 가져옵니다.
    - robots/로그인/스크립트 기반 페이지는 텍스트가 빈약할 수 있습니다.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    html = r.text
    soup = BeautifulSoup(html, "html.parser")

    # 불필요 요소 제거
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def web_search_if_enabled(client: OpenAI, query: str, enabled: bool) -> str:
    """
    OpenAI Responses web_search tool (가능한 계정에서만 동작).
    실패하면 빈 문자열.
    """
    if not enabled:
        return ""
    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=query,
            tools=[{"type": "web_search"}],
            tool_choice="auto",
            max_output_tokens=900,
        )
        return (getattr(resp, "output_text", "") or "").strip()
    except Exception:
        return ""


def call_json_object(client: OpenAI, instructions: str, payload: Dict[str, Any], max_tokens: int = 1400) -> Dict[str, Any]:
    """
    JSON 한 덩어리만 받도록 강제 + SDK/모델 호환성 방어형 처리
    - text.format 미지원 SDK -> 자동 제거 후 재시도
    - 최종적으로는 JSON 추출 fallback
    """
    import json as _json

    # text.format json_object requires the word "json" to appear in input
    inp = "json\n" + _json.dumps(payload, ensure_ascii=False)

    # 기본 호출 파라미터 (안전하게 dict로 관리)
    base_kwargs = {
        "model": OPENAI_MODEL,
        "instructions": instructions,
        "input": inp,
        "max_output_tokens": max_tokens,
    }

    # ✅ text.format: 지원하면 쓰고, 아니면 제거될 것
    base_kwargs["text"] = {"format": {"type": "json_object"}}

    def _extract_output_text(resp) -> str:
        txt = (getattr(resp, "output_text", "") or "").strip()
        if txt:
            return txt

        outputs = getattr(resp, "output", None)
        if outputs:
            parts = []
            refusals = []
            for out in outputs:
                if getattr(out, "type", None) == "message":
                    for content in getattr(out, "content", []) or []:
                        ctype = getattr(content, "type", None)
                        if ctype == "output_text":
                            parts.append(getattr(content, "text", "") or "")
                        elif ctype == "refusal":
                            refusals.append(getattr(content, "refusal", "") or "")
            if parts:
                return "".join(parts).strip()
            if refusals:
                raise RuntimeError("모델이 거절했습니다: " + " | ".join([r for r in refusals if r]))

        err = getattr(resp, "error", None)
        if err:
            raise RuntimeError(f"OpenAI 응답 오류: {err}")
        incomplete = getattr(resp, "incomplete_details", None)
        if incomplete and getattr(incomplete, "reason", None):
            raise RuntimeError(f"모델 출력이 불완전합니다: {getattr(incomplete, 'reason', None)}")

        return ""

    def _parse_json_text(txt: str) -> Dict[str, Any]:
        txt = (txt or "").strip()
        if txt.startswith("\ufeff"):
            txt = txt.lstrip("\ufeff")
        if txt.lower().startswith("json"):
            rest = txt[4:].lstrip()
            if rest.startswith("{"):
                txt = rest
        if not txt:
            raise ValueError("empty output")
        try:
            return _json.loads(txt)
        except Exception:
            pass

        m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", txt, flags=re.S | re.I)
        if m:
            return _json.loads(m.group(1))

        start = txt.find("{")
        end = txt.rfind("}")
        if start != -1 and end != -1 and end > start:
            return _json.loads(txt[start:end + 1])

        raise ValueError("json parse failed")

    def _parse_or_raise(txt: str, label: str) -> Dict[str, Any]:
        try:
            return _parse_json_text(txt)
        except Exception as e:
            snippet = txt[:300].replace("\n", "\\n")
            raise RuntimeError(f"{label}: JSON 파싱 실패. raw='{snippet}'") from e

    def _try_create(kwargs: Dict[str, Any]):
        # kwargs는 항상 복사본으로 넘겨서 꼬임 방지
        return client.responses.create(**kwargs)

    # 1) 1차 시도
    try:
        resp = _try_create(dict(base_kwargs))
        txt = _extract_output_text(resp)
        return _parse_json_text(txt)

    except TypeError as e:
        # SDK 미지원 인자(text/response_format 등) 처리
        msg = str(e)

        kwargs = dict(base_kwargs)

        if "text" in msg or "response_format" in msg:
            kwargs.pop("text", None)
            kwargs.pop("response_format", None)

        resp = _try_create(kwargs)
        txt = _extract_output_text(resp)
        return _parse_or_raise(txt, "회사 프로필 JSON 파싱 실패 (TypeError fallback)")

    except Exception as e:
        if isinstance(e, RuntimeError) and any(
            k in str(e) for k in ("모델이 거절했습니다", "OpenAI 응답 오류", "모델 출력이 불완전합니다")
        ):
            # max_output_tokens면 더 큰 토큰으로 1회 재시도
            if "max_output_tokens" in str(e).lower():
                kwargs = dict(base_kwargs)
                kwargs["max_output_tokens"] = max(max_tokens + 800, int(max_tokens * 1.5))
                kwargs["instructions"] = (
                    instructions
                    + "\n\n반드시 유효한 JSON 객체 하나만 출력해라. 키/문자열은 큰따옴표, 코드펜스/설명 금지."
                    + " 각 리스트는 최대 8개, 각 항목 80자 이내. references는 최대 8개."
                )
                resp = _try_create(kwargs)
                txt = _extract_output_text(resp)
                return _parse_or_raise(txt, "회사 프로필 JSON 파싱 실패 (max_output_tokens retry)")
            raise
        # 2) 모델/SDK 호환 이슈 fallback 재시도
        msg = str(e).lower()
        kwargs = dict(base_kwargs)

        # text/response_format 관련 에러면 제거 후 재시도
        if "text" in msg or "response_format" in msg or "unexpected keyword" in msg:
            kwargs.pop("text", None)
            kwargs.pop("response_format", None)

        # JSON 강제 지시를 더 강하게
        kwargs["instructions"] = instructions + "\n\n반드시 JSON 객체 하나만 출력해라. 다른 텍스트 금지."

        resp = _try_create(kwargs)
        txt = _extract_output_text(resp)
        return _parse_or_raise(txt, "회사 프로필 JSON 파싱 실패 (Exception fallback)")


def build_company_profile(
    client: OpenAI,
    *,
    company_name: str,
    role: str,
    job_posting_urls: List[str],
    enable_web_search: bool,
    max_source_chars: int = 12000,
) -> Dict[str, Any]:
    """
    회사 프로필(JSON) 생성:
    - values / talent_traits / required / preferred / keywords / tone / references
    """
    sources: List[Dict[str, Any]] = []

    # 1) 공고 URL 텍스트(최우선 근거)
    for url in job_posting_urls[:5]:
        try:
            raw = fetch_url_text(url)
            sources.append({
                "type": "job_posting_url",
                "url": url,
                "text": clamp_text(raw, max_source_chars)
            })
        except Exception as e:
            sources.append({
                "type": "job_posting_url",
                "url": url,
                "text": f"(가져오기 실패: {e})"
            })

    # 2) web_search 보조(옵션)
    if enable_web_search:
        q1 = f"{company_name} 인재상 핵심가치"
        q2 = f"{company_name} 채용 {role} 자격요건 우대사항"
        w1 = web_search_if_enabled(client, q1, True)
        w2 = web_search_if_enabled(client, q2, True)
        if w1:
            sources.append({"type": "web_search", "query": q1, "text": clamp_text(w1, 6000)})
        if w2:
            sources.append({"type": "web_search", "query": q2, "text": clamp_text(w2, 6000)})

    instructions = (
        "너는 채용공고/기업정보를 구조화하는 에이전트다.\n"
        "출력은 JSON 하나만. 불필요한 설명 금지.\n"
        "반드시 유효한 JSON만 출력해라(키/문자열은 큰따옴표, 코드펜스/설명 금지).\n"
        "각 리스트는 최대 8개, 각 항목 80자 이내로 간결하게 써라.\n"
        "references는 최대 8개만 출력해라.\n"
        "근거(sources)에 없는 내용을 단정하지 마라. 불확실하면 'unknown' 또는 빈 리스트로 둬라.\n"
        "아래 스키마를 최대한 채워라:\n"
        "{\n"
        "  company_name: str,\n"
        "  role: str,\n"
        "  values: [str],               # 기업 핵심가치/인재상 키워드\n"
        "  talent_traits: [str],        # 선호 역량/태도(협업/주도성 등)\n"
        "  required_qualifications: [str],\n"
        "  preferred_qualifications: [str],\n"
        "  keywords: [str],             # 자기소개서에 녹일 키워드(기술/도메인)\n"
        "  writing_tone: str,           # 문체 가이드(예: 간결/근거중심/정량)\n"
        "  do_not_claim: [str],         # 근거 없으면 쓰면 안 되는 주장 유형\n"
        "  references: [ {type, url|query, snippet} ]  # 추적 가능한 스니펫(짧게)\n"
        "}\n"
        "references.snippet은 200자 이내로 압축해라.\n"
    )

    payload = {
        "company_name": company_name,
        "role": role,
        "sources": sources,
    }

    prof = call_json_object(client, instructions, payload)

    # references.snippet 축약(안전)
    refs = []
    for r in prof.get("references", [])[:12]:
        snip = clamp_text((r.get("snippet") or ""), 200)
        rr = dict(r)
        rr["snippet"] = snip
        refs.append(rr)
    prof["references"] = refs

    # 최소 필드 보정
    prof.setdefault("company_name", company_name)
    prof.setdefault("role", role)
    prof.setdefault("values", [])
    prof.setdefault("talent_traits", [])
    prof.setdefault("required_qualifications", [])
    prof.setdefault("preferred_qualifications", [])
    prof.setdefault("keywords", [])
    prof.setdefault("writing_tone", "명확/간결/근거 중심")
    prof.setdefault("do_not_claim", ["근거 없는 수치", "근거 없는 수상/논문", "근거 없는 경력/직책"])

    return prof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--company", default=None)
    ap.add_argument("--role", default=None)
    ap.add_argument("--job_url", action="append", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config)
    company_name = args.company or cfg["company"]["name"]
    role = args.role or cfg["company"]["role"]

    job_urls = args.job_url[:] if args.job_url else cfg["company_profile"].get("job_posting_urls", [])
    # 요구 조건: web search는 항상 활성화
    enable_web = True

    out_dir = cfg["company_profile"].get("output_dir", "company_db")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{slugify(company_name)}.json")

    client = OpenAI(api_key=resolve_api_key())

    prof = build_company_profile(
        client,
        company_name=company_name,
        role=role,
        job_posting_urls=job_urls,
        enable_web_search=enable_web,
        max_source_chars=int(cfg["company_profile"].get("max_source_chars", 12000)),
    )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(prof, f, ensure_ascii=False, indent=2)

    print("[OK] company profile saved:", out_path)


if __name__ == "__main__":
    main()
