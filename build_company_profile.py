#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_company_profile.py - 회사 프로필 자동 생성

채용공고 URL 또는 웹 검색을 통해 회사 정보를 수집하고
company_db/<slug>.json으로 저장합니다.

사용법:
    python build_company_profile.py --config config.yaml
"""

import os
import re
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

import yaml
from openai import OpenAI


# ============================================================
# 유틸리티
# ============================================================

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def slugify(name: str) -> str:
    s = (name or "").strip().lower()
    s = re.sub(r"[^a-z0-9가-힣]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "company"


def setup_logger(cfg: Dict) -> logging.Logger:
    log_cfg = cfg.get("logging", {})
    level = log_cfg.get("level", "INFO").upper()
    quiet = log_cfg.get("quiet", False)

    logger = logging.getLogger("company_profile")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING if quiet else getattr(logging, level, logging.INFO))
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    return logger


def get_api_key(cfg: Dict) -> str:
    key = cfg.get("openai", {}).get("api_key", "").strip()
    if key:
        return key
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    raise RuntimeError("OPENAI_API_KEY 필요")


# ============================================================
# URL 크롤러
# ============================================================

class JobPostingCrawler:
    """채용공고 URL에서 텍스트 추출"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def fetch(self, url: str) -> Optional[str]:
        """URL에서 텍스트 추출"""
        try:
            import requests
            from bs4 import BeautifulSoup
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            
            soup = BeautifulSoup(resp.text, "html.parser")
            
            # 스크립트/스타일 제거
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            
            text = soup.get_text(separator="\n", strip=True)
            
            # 정리
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            text = "\n".join(lines)
            
            self.logger.info(f"URL 크롤링 성공: {len(text)}자")
            return text[:15000]  # 최대 15000자
            
        except ImportError:
            self.logger.warning("requests/beautifulsoup4 미설치 - URL 크롤링 불가")
            return None
        except Exception as e:
            self.logger.warning(f"URL 크롤링 실패: {e}")
            return None


# ============================================================
# 프로필 추출기
# ============================================================

class ProfileExtractor:
    """LLM으로 회사 프로필 추출"""
    
    def __init__(self, client: OpenAI, cfg: Dict, logger: logging.Logger):
        self.client = client
        self.cfg = cfg
        self.logger = logger
        
        models = cfg.get("models", {})
        self.model = models.get("standard", "gpt-4o")
    
    def extract(self, company_name: str, role: str, raw_text: str) -> Dict[str, Any]:
        """텍스트에서 회사 프로필 추출"""
        
        prompt = f"""다음 채용공고/회사 정보에서 자기소개서 작성에 필요한 정보를 추출하세요.

회사: {company_name}
직무: {role}

원문:
{raw_text[:12000]}

JSON으로 응답:
{{
    "company_name": "{company_name}",
    "role": "{role}",
    "company_description": "회사 소개 2-3문장",
    "team_description": "팀/부서 소개 (있으면)",
    "values": ["핵심가치1", "핵심가치2"],
    "talent_traits": ["인재상1", "인재상2"],
    "required_qualifications": ["필수자격1", "필수자격2"],
    "preferred_qualifications": ["우대사항1", "우대사항2"],
    "responsibilities": ["담당업무1", "담당업무2"],
    "keywords": ["기술키워드1", "기술키워드2"],
    "writing_tone": "명확하고 구체적인 근거 중심",
    "do_not_claim": ["근거 없는 수치", "과장된 표현"]
}}

규칙:
- 원문에 없는 내용은 추측하지 말고 빈 리스트로
- 각 항목은 간결하게 (20자 이내)
- keywords는 기술/도메인 키워드 중심"""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=1500
            )
            
            result = json.loads(resp.choices[0].message.content)
            self.logger.info(f"프로필 추출 완료: {len(result.get('keywords', []))}개 키워드")
            return result
            
        except Exception as e:
            self.logger.error(f"프로필 추출 실패: {e}")
            return self._default_profile(company_name, role)
    
    def _default_profile(self, company_name: str, role: str) -> Dict[str, Any]:
        """기본 프로필 템플릿"""
        return {
            "company_name": company_name,
            "role": role,
            "company_description": "",
            "team_description": "",
            "values": [],
            "talent_traits": [],
            "required_qualifications": [],
            "preferred_qualifications": [],
            "responsibilities": [],
            "keywords": [],
            "writing_tone": "명확하고 구체적인 근거 중심",
            "do_not_claim": ["근거 없는 수치", "근거 없는 논문/수상", "과장된 표현"]
        }
    
    def enrich_with_search(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """웹 검색으로 프로필 보강 (선택적)"""
        
        company = profile.get("company_name", "")
        role = profile.get("role", "")
        
        if not company:
            return profile
        
        search_prompt = f"""다음 회사/직무에 대해 자기소개서 작성에 도움될 정보를 제공하세요.

회사: {company}
직무: {role}

기존 정보:
- 핵심가치: {profile.get('values', [])}
- 인재상: {profile.get('talent_traits', [])}

JSON으로 응답:
{{
    "additional_values": ["추가 핵심가치"],
    "additional_traits": ["추가 인재상"],
    "company_culture": "회사 문화 특징",
    "industry_focus": "주요 사업 분야"
}}

규칙: 공식적으로 알려진 정보만 사용"""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": search_prompt}],
                response_format={"type": "json_object"},
                max_tokens=800
            )
            
            extra = json.loads(resp.choices[0].message.content)
            
            # 병합
            profile["values"] = list(set(profile.get("values", []) + extra.get("additional_values", [])))
            profile["talent_traits"] = list(set(profile.get("talent_traits", []) + extra.get("additional_traits", [])))
            profile["company_culture"] = extra.get("company_culture", "")
            profile["industry_focus"] = extra.get("industry_focus", "")
            
            self.logger.info("웹 검색으로 프로필 보강 완료")
            
        except Exception as e:
            self.logger.warning(f"프로필 보강 실패: {e}")
        
        return profile


# ============================================================
# 메인
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="회사 프로필 빌더")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"[ERROR] 설정 파일 없음: {args.config}")
        return
    
    cfg = load_yaml(args.config)
    logger = setup_logger(cfg)
    
    logger.info("=" * 50)
    logger.info("회사 프로필 빌드 시작")
    logger.info("=" * 50)
    
    # 설정 로드
    company_cfg = cfg.get("company", {})
    profile_cfg = cfg.get("company_profile", {})
    
    company_name = company_cfg.get("name", "")
    role = company_cfg.get("role", "")
    
    if not company_name:
        logger.error("config.yaml에 company.name 필요")
        return
    
    company_slug = profile_cfg.get("company_slug") or slugify(company_name)
    job_urls = profile_cfg.get("job_posting_urls", [])
    enable_search = profile_cfg.get("enable_web_search", True)
    max_chars = profile_cfg.get("max_source_chars", 12000)
    
    # 출력 경로
    company_db_dir = cfg.get("paths", {}).get("company_db_dir", "company_db")
    output_path = os.path.join(company_db_dir, f"{company_slug}.json")
    ensure_dir(output_path)
    
    # OpenAI 클라이언트
    client = OpenAI(api_key=get_api_key(cfg))
    
    # 크롤러 & 추출기
    crawler = JobPostingCrawler(logger)
    extractor = ProfileExtractor(client, cfg, logger)
    
    # URL에서 텍스트 수집
    all_text = []
    
    for url in job_urls:
        if not url or not url.strip():
            continue
        logger.info(f"URL 크롤링: {url[:60]}...")
        text = crawler.fetch(url.strip())
        if text:
            all_text.append(text)
    
    if all_text:
        combined_text = "\n\n---\n\n".join(all_text)[:max_chars]
        logger.info(f"총 {len(combined_text)}자 수집")
    else:
        logger.warning("URL에서 텍스트 수집 실패 - 기본 프로필 생성")
        combined_text = f"회사: {company_name}\n직무: {role}"
    
    # 프로필 추출
    logger.info("프로필 추출 중...")
    profile = extractor.extract(company_name, role, combined_text)
    
    # 웹 검색으로 보강 (선택적)
    if enable_search:
        logger.info("프로필 보강 중...")
        profile = extractor.enrich_with_search(profile)
    
    # 메타데이터 추가
    profile["_meta"] = {
        "created_at": datetime.now().isoformat(),
        "source_urls": job_urls,
        "slug": company_slug
    }
    
    # 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    
    logger.info("=" * 50)
    logger.info(f"회사 프로필 저장 완료: {output_path}")
    logger.info(f"  - 핵심가치: {len(profile.get('values', []))}개")
    logger.info(f"  - 인재상: {len(profile.get('talent_traits', []))}개")
    logger.info(f"  - 필수자격: {len(profile.get('required_qualifications', []))}개")
    logger.info(f"  - 우대사항: {len(profile.get('preferred_qualifications', []))}개")
    logger.info(f"  - 키워드: {len(profile.get('keywords', []))}개")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
