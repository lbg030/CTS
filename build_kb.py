#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_kb.py - Knowledge Base 자동 빌더

모델 선택 정책:
  - gpt-4o (standard): 논문 분석, 프로필 생성
  - gpt-4o-mini (fast): 메타데이터 추출, 키워드

사용법:
    python build_kb.py --config config.yaml
"""

import os
import re
import json
import yaml
import logging
import hashlib
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import Counter

import numpy as np


# ============================================================
# 로깅
# ============================================================

class QuietFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno == logging.INFO:
            return f"[{record.levelname}] {record.getMessage()}"
        return super().format(record)


def setup_logging(config: Dict) -> logging.Logger:
    log_cfg = config.get("logging", {})
    level_str = log_cfg.get("level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    quiet = log_cfg.get("quiet", False)
    
    logger = logging.getLogger("build_kb")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING if quiet else level)
    console.setFormatter(QuietFormatter())
    logger.addHandler(console)
    
    if log_cfg.get("to_file", False):
        log_path = log_cfg.get("kb_log_path", "logs/build_kb.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)
    
    return logger


# ============================================================
# 데이터 구조
# ============================================================

@dataclass
class PaperMeta:
    title: str
    authors: List[str]
    year: int
    venue: str
    citations: int
    abstract: str
    source: str
    pdf_path: Optional[str] = None
    scholar_url: Optional[str] = None


@dataclass
class PaperAnalysis:
    paper_id: str
    meta: PaperMeta
    sections: Dict[str, str]
    key_contributions: List[str]
    problem_addressed: str
    proposed_method: str
    results_summary: str
    keywords: List[str]
    technologies: List[str]
    importance_score: float


@dataclass
class KBChunk:
    id: str
    paper_id: str
    section: str
    text: str
    keywords: List[str]
    importance: float
    meta: Dict[str, Any]


@dataclass
class BuildReport:
    status: str
    started_at: str
    completed_at: str
    duration_seconds: float
    sources: Dict[str, int]
    papers_processed: int
    chunks_created: int
    embedding_dim: int
    warnings: List[str]
    errors: List[str]
    models_used: Dict[str, str]


# ============================================================
# 설정
# ============================================================

def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_api_key(config: Dict) -> str:
    key = config.get("openai", {}).get("api_key", "").strip()
    if key:
        return key
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    raise RuntimeError("OPENAI_API_KEY 필요")


# ============================================================
# 모델 선택
# ============================================================

class ModelSelector:
    """KB 빌드용 모델 선택 (R3: config.yaml의 model_policy 지원)"""
    
    def __init__(self, config: Dict):
        models = config.get("models", {})
        self.high_quality = models.get("high_quality", "gpt-4o")
        self.standard = models.get("standard", "gpt-4o")
        self.fast = models.get("fast", "gpt-4o-mini")
        
        # KB 모델 정책 (config에서 override 가능)
        kb_policy = config.get("kb", {}).get("model_policy", {})
        self.policy = {
            "paper_analysis": kb_policy.get("paper_analysis", "high_quality"),
            "key_extraction": kb_policy.get("key_extraction", "high_quality"),
            "summary_generation": kb_policy.get("summary_generation", "high_quality"),
            "meta_extraction": kb_policy.get("meta_extraction", "fast"),
            "keyword_extraction": kb_policy.get("keyword_extraction", "fast"),
        }
    
    def get_model(self, task: str) -> str:
        """작업별 모델 반환 (R3 정책 적용)"""
        task_lower = task.lower()
        
        # 정책 기반 매핑
        if "analysis" in task_lower or "analyzer" in task_lower:
            tier = self.policy.get("paper_analysis", "high_quality")
        elif "key" in task_lower and "extract" in task_lower:
            tier = self.policy.get("key_extraction", "high_quality")
        elif "summary" in task_lower or "profile" in task_lower:
            tier = self.policy.get("summary_generation", "high_quality")
        elif "meta" in task_lower:
            tier = self.policy.get("meta_extraction", "fast")
        elif "keyword" in task_lower:
            tier = self.policy.get("keyword_extraction", "fast")
        else:
            tier = "fast"  # KB 기본
        
        # tier -> 실제 모델
        if tier == "high_quality":
            return self.high_quality
        elif tier == "standard":
            return self.standard
        else:
            return self.fast


# ============================================================
# Google Scholar 크롤러
# ============================================================

class ScholarCrawler:
    def __init__(self, config: Dict, logger: logging.Logger):
        self.config = config.get("kb", {}).get("scholar", {})
        self.delay = self.config.get("rate_limit_delay", 2.0)
        self.max_papers = self.config.get("max_papers", 50)
        self.logger = logger
    
    def fetch_profile(self, url: str) -> List[PaperMeta]:
        try:
            from scholarly import scholarly
        except ImportError:
            self.logger.error("scholarly 미설치: pip install scholarly")
            return []
        
        author_id = self._extract_author_id(url)
        if not author_id:
            self.logger.warning(f"Scholar URL에서 author ID 추출 실패")
            return []
        
        papers = []
        try:
            self.logger.info(f"Scholar 프로필 크롤링: {author_id}")
            author = scholarly.search_author_id(author_id)
            author = scholarly.fill(author, sections=['publications'])
            
            pubs = author.get('publications', [])
            if self.max_papers > 0:
                pubs = pubs[:self.max_papers]
            
            for i, pub in enumerate(pubs):
                time.sleep(self.delay)
                try:
                    filled = scholarly.fill(pub)
                    paper = PaperMeta(
                        title=filled.get('bib', {}).get('title', 'Unknown'),
                        authors=filled.get('bib', {}).get('author', '').split(' and '),
                        year=int(filled.get('bib', {}).get('pub_year', 0)),
                        venue=filled.get('bib', {}).get('venue', ''),
                        citations=int(filled.get('num_citations', 0)),
                        abstract=filled.get('bib', {}).get('abstract', ''),
                        source="scholar",
                        scholar_url=filled.get('pub_url', '')
                    )
                    papers.append(paper)
                    self.logger.debug(f"  [{i+1}/{len(pubs)}] {paper.title[:40]}...")
                except Exception as e:
                    self.logger.warning(f"논문 수집 실패: {e}")
            
            self.logger.info(f"Scholar에서 {len(papers)}개 논문 수집")
        except Exception as e:
            self.logger.error(f"Scholar 접근 실패: {e}")
        
        return papers
    
    def _extract_author_id(self, url: str) -> Optional[str]:
        match = re.search(r'user=([^&]+)', url)
        return match.group(1) if match else None


# ============================================================
# PDF 프로세서
# ============================================================

class PDFProcessor:
    def __init__(self, client, config: Dict, logger: logging.Logger, model_selector: ModelSelector):
        self.client = client
        self.config = config
        self.logger = logger
        self.model_selector = model_selector
        self.model = model_selector.get_model("meta_extract")  # gpt-4o-mini
        self.min_text = config.get("kb", {}).get("quality", {}).get("min_text_per_pdf", 500)
    
    def process_directory(self, dir_path: str) -> Tuple[List[Tuple[str, str]], List[str]]:
        pdf_files = list(Path(dir_path).glob("*.pdf"))
        results = []
        warnings = []
        
        self.logger.info(f"PDF 폴더 처리: {dir_path} ({len(pdf_files)}개)")
        
        for i, pdf_path in enumerate(pdf_files):
            self.logger.debug(f"  [{i+1}/{len(pdf_files)}] {pdf_path.name}")
            text = self._extract_text(str(pdf_path))
            
            if text:
                if len(text) < self.min_text:
                    warnings.append(f"{pdf_path.name}: 텍스트 부족 ({len(text)}자)")
                results.append((str(pdf_path), text))
            else:
                warnings.append(f"{pdf_path.name}: 추출 실패")
        
        self.logger.info(f"PDF 처리 완료: {len(results)}/{len(pdf_files)}개")
        return results, warnings
    
    def _extract_text(self, pdf_path: str) -> Optional[str]:
        try:
            from pypdf import PdfReader
            reader = PdfReader(pdf_path)
            texts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    texts.append(text)
            return "\n\n".join(texts)
        except Exception as e:
            self.logger.debug(f"PDF 읽기 실패: {e}")
            return None
    
    def extract_paper_meta(self, pdf_path: str, text: str) -> PaperMeta:
        """메타데이터 추출 (gpt-4o-mini 사용)"""
        prompt = f"""학술 논문 PDF에서 메타데이터를 추출하세요.

텍스트 (처음 3000자):
{text[:3000]}

JSON 응답:
{{"title": "논문 제목", "authors": ["저자1", "저자2"], "year": 2024, "venue": "학회/저널명", "abstract": "초록"}}"""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=800
            )
            data = json.loads(resp.choices[0].message.content)
            return PaperMeta(
                title=data.get("title", Path(pdf_path).stem),
                authors=data.get("authors", []),
                year=data.get("year", 0),
                venue=data.get("venue", ""),
                citations=0,
                abstract=data.get("abstract", ""),
                source="pdf",
                pdf_path=pdf_path
            )
        except Exception as e:
            self.logger.warning(f"메타 추출 실패: {e}")
            return PaperMeta(
                title=Path(pdf_path).stem,
                authors=[], year=0, venue="", citations=0,
                abstract="", source="pdf", pdf_path=pdf_path
            )


# ============================================================
# 논문 분석기
# ============================================================

class PaperAnalyzer:
    def __init__(self, client, config: Dict, logger: logging.Logger, model_selector: ModelSelector):
        self.client = client
        self.config = config
        self.logger = logger
        self.model_selector = model_selector
        self.model = model_selector.get_model("analyzer")  # gpt-4o
        self.importance_cfg = config.get("kb", {}).get("importance", {})
    
    def analyze(self, meta: PaperMeta, full_text: Optional[str] = None) -> PaperAnalysis:
        paper_id = hashlib.md5(meta.title.encode()).hexdigest()[:12]
        
        if full_text:
            analysis_text = full_text[:12000]
        else:
            analysis_text = f"Title: {meta.title}\nAbstract: {meta.abstract}"
        
        analysis = self._llm_analyze(analysis_text, meta)
        importance = self._calculate_importance(meta, analysis)
        
        return PaperAnalysis(
            paper_id=paper_id,
            meta=meta,
            sections=analysis.get("sections", {}),
            key_contributions=analysis.get("key_contributions", []),
            problem_addressed=analysis.get("problem_addressed", ""),
            proposed_method=analysis.get("proposed_method", ""),
            results_summary=analysis.get("results_summary", ""),
            keywords=analysis.get("keywords", []),
            technologies=analysis.get("technologies", []),
            importance_score=importance
        )
    
    def _llm_analyze(self, text: str, meta: PaperMeta) -> Dict:
        """논문 분석 (gpt-4o 사용)"""
        prompt = f"""학술 논문을 분석하세요.

제목: {meta.title}
학회/저널: {meta.venue}
연도: {meta.year}

본문:
{text[:10000]}

JSON 응답:
{{"sections": {{"abstract": "초록 요약", "problem": "문제", "method": "방법", "results": "결과"}}, "key_contributions": ["기여1", "기여2"], "problem_addressed": "문제 한줄", "proposed_method": "방법 한줄", "results_summary": "결과 수치 포함", "keywords": ["키워드"], "technologies": ["기술"]}}"""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=1500
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            self.logger.warning(f"분석 실패: {e}")
            return {}
    
    def _calculate_importance(self, meta: PaperMeta, analysis: Dict) -> float:
        cfg = self.importance_cfg
        score = 0.0
        
        if meta.citations > 0:
            score += min(meta.citations / 100, 1.0) * cfg.get("citation_weight", 0.3)
        
        venue_lower = meta.venue.lower()
        top_venues = cfg.get("top_venues", [])
        good_venues = cfg.get("good_venues", [])
        
        if any(v in venue_lower for v in top_venues):
            score += cfg.get("venue_weight", 0.3)
        elif any(v in venue_lower for v in good_venues):
            score += cfg.get("venue_weight", 0.3) * 0.5
        
        current_year = datetime.now().year
        if meta.year >= current_year:
            score += cfg.get("recency_weight", 0.2)
        elif meta.year >= current_year - 1:
            score += cfg.get("recency_weight", 0.2) * 0.75
        elif meta.year >= current_year - 2:
            score += cfg.get("recency_weight", 0.2) * 0.5
        
        if len(analysis.get("key_contributions", [])) >= 2:
            score += cfg.get("content_weight", 0.2) * 0.5
        if analysis.get("results_summary") and any(c.isdigit() for c in analysis.get("results_summary", "")):
            score += cfg.get("content_weight", 0.2) * 0.5
        
        return min(score, 1.0)


# ============================================================
# KB 빌더
# ============================================================

class KBBuilder:
    def __init__(self, client, config: Dict, logger: logging.Logger, model_selector: ModelSelector):
        self.client = client
        self.config = config
        self.logger = logger
        self.model_selector = model_selector
        self.model = model_selector.get_model("profile")  # gpt-4o
    
    def build_chunks(self, analyses: List[PaperAnalysis]) -> List[KBChunk]:
        chunks = []
        
        for analysis in analyses:
            # 메타 청크
            meta_text = self._create_meta_chunk(analysis)
            chunks.append(KBChunk(
                id=f"{analysis.paper_id}_meta",
                paper_id=analysis.paper_id,
                section="meta",
                text=meta_text,
                keywords=analysis.keywords,
                importance=analysis.importance_score,
                meta={"title": analysis.meta.title, "year": analysis.meta.year,
                      "venue": analysis.meta.venue, "citations": analysis.meta.citations}
            ))
            
            # 섹션 청크
            for section, text in analysis.sections.items():
                if text and len(text) > 50:
                    chunks.append(KBChunk(
                        id=f"{analysis.paper_id}_{section}",
                        paper_id=analysis.paper_id,
                        section=section,
                        text=text,
                        keywords=analysis.keywords,
                        importance=analysis.importance_score * 0.8,
                        meta={"title": analysis.meta.title}
                    ))
            
            # 기여 청크
            if analysis.key_contributions:
                contrib_text = f"논문 \"{analysis.meta.title}\"의 핵심 기여:\n"
                contrib_text += "\n".join(f"- {c}" for c in analysis.key_contributions)
                contrib_text += f"\n\n제안 방법: {analysis.proposed_method}"
                
                chunks.append(KBChunk(
                    id=f"{analysis.paper_id}_contributions",
                    paper_id=analysis.paper_id,
                    section="contributions",
                    text=contrib_text,
                    keywords=analysis.keywords,
                    importance=analysis.importance_score,
                    meta={"title": analysis.meta.title}
                ))
        
        self.logger.info(f"청크 생성: {len(chunks)}개")
        return chunks
    
    def _create_meta_chunk(self, analysis: PaperAnalysis) -> str:
        return f"""논문: {analysis.meta.title}
저자: {', '.join(analysis.meta.authors[:3])}
발표: {analysis.meta.venue} ({analysis.meta.year})
인용: {analysis.meta.citations}회
키워드: {', '.join(analysis.keywords)}
문제: {analysis.problem_addressed}
방법: {analysis.proposed_method}
결과: {analysis.results_summary}"""

    def generate_summary(self, analyses: List[PaperAnalysis]) -> Dict:
        """프로필 생성 (gpt-4o 사용)"""
        all_keywords = []
        for a in analyses:
            all_keywords.extend(a.keywords)
        keyword_counts = Counter(all_keywords).most_common(20)
        
        research_areas = self._extract_areas(keyword_counts)
        
        sorted_papers = sorted(analyses, key=lambda x: x.importance_score, reverse=True)
        representative = [
            {"title": p.meta.title, "venue": p.meta.venue, "year": p.meta.year,
             "contribution": p.key_contributions[0] if p.key_contributions else "",
             "importance": p.importance_score}
            for p in sorted_papers[:5]
        ]
        
        all_techs = []
        for a in analyses:
            all_techs.extend(a.technologies)
        tech_expertise = [t for t, _ in Counter(all_techs).most_common(10)]
        
        profile = self._generate_profile(analyses, keyword_counts, research_areas)
        
        return {
            "research_areas": research_areas,
            "top_keywords": keyword_counts,
            "representative_papers": representative,
            "researcher_profile": profile,
            "technical_expertise": tech_expertise
        }
    
    def _extract_areas(self, keyword_counts: List[Tuple[str, int]]) -> List[str]:
        areas = set()
        area_mapping = {
            "3d reconstruction": ["3d", "reconstruction", "gaussian", "splatting", "nerf"],
            "depth estimation": ["depth", "monocular", "stereo", "mvs"],
            "computer vision": ["vision", "image", "visual", "detection"],
            "deep learning": ["neural", "network", "learning", "transformer"],
        }
        
        keyword_set = set(kw.lower() for kw, _ in keyword_counts[:15])
        for area, keywords in area_mapping.items():
            if any(kw in keyword_set for kw in keywords):
                areas.add(area)
        return list(areas)
    
    def _generate_profile(self, analyses: List[PaperAnalysis],
                          keyword_counts: List, research_areas: List[str]) -> str:
        papers_info = "\n".join([
            f"- {a.meta.title} ({a.meta.venue}, {a.meta.year})"
            for a in sorted(analyses, key=lambda x: x.importance_score, reverse=True)[:5]
        ])
        
        prompt = f"""연구자 프로필을 3-4문장으로 작성하세요.

연구 분야: {', '.join(research_areas)}
키워드: {', '.join([k for k, _ in keyword_counts[:10]])}
대표 논문:
{papers_info}

과장 없이 사실 기반으로."""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return f"연구 분야: {', '.join(research_areas)}"


# ============================================================
# 임베딩 인덱서
# ============================================================

class EmbeddingIndexer:
    def __init__(self, client, config: Dict, logger: logging.Logger):
        self.client = client
        self.config = config
        self.logger = logger
        
        embed_cfg = config.get("kb", {}).get("embedding", {})
        self.model = embed_cfg.get("model", "text-embedding-3-large")
        self.batch_size = embed_cfg.get("batch_size", 32)
        self.cache_enabled = embed_cfg.get("cache_enabled", False)
        self.cache_path = embed_cfg.get("cache_path", "cache/kb_embeddings.json")
        
        self.cache = self._load_cache() if self.cache_enabled else {}
    
    def _load_cache(self) -> Dict:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r") as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _save_cache(self):
        if self.cache_enabled:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f)
    
    def create_index(self, chunks: List[KBChunk]) -> Tuple[Any, int]:
        import faiss
        
        texts = [c.text for c in chunks]
        vectors = []
        
        uncached_indices = []
        uncached_texts = []
        
        for i, text in enumerate(texts):
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in self.cache:
                vectors.append(self.cache[text_hash])
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
                vectors.append(None)
        
        if uncached_texts:
            self.logger.info(f"임베딩 생성: {len(uncached_texts)}개 (캐시: {len(texts) - len(uncached_texts)}개)")
            
            new_vectors = []
            for i in range(0, len(uncached_texts), self.batch_size):
                batch = uncached_texts[i:i+self.batch_size]
                resp = self.client.embeddings.create(model=self.model, input=batch)
                new_vectors.extend([d.embedding for d in resp.data])
            
            for idx, vec in zip(uncached_indices, new_vectors):
                vectors[idx] = vec
                text_hash = hashlib.md5(texts[idx].encode()).hexdigest()
                self.cache[text_hash] = vec
            
            self._save_cache()
        else:
            self.logger.info(f"임베딩: 전체 캐시 히트 ({len(texts)}개)")
        
        vec = np.array(vectors, dtype=np.float32)
        faiss.normalize_L2(vec)
        
        index = faiss.IndexFlatIP(vec.shape[1])
        index.add(vec)
        
        return index, vec.shape[1]


# ============================================================
# 메인 파이프라인
# ============================================================

class KBPipeline:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = setup_logging(config)
        
        from openai import OpenAI
        self.client = OpenAI(api_key=get_api_key(config))
        
        self.model_selector = ModelSelector(config)
        
        self.scholar = ScholarCrawler(config, self.logger)
        self.pdf_proc = PDFProcessor(self.client, config, self.logger, self.model_selector)
        self.analyzer = PaperAnalyzer(self.client, config, self.logger, self.model_selector)
        self.kb_builder = KBBuilder(self.client, config, self.logger, self.model_selector)
        self.indexer = EmbeddingIndexer(self.client, config, self.logger)
        
        self.warnings: List[str] = []
        self.errors: List[str] = []
    
    def run(self) -> BuildReport:
        import faiss
        
        started_at = datetime.now()
        kb_cfg = self.config.get("kb", {})
        kb_dir = self.config.get("paths", {}).get("kb_dir", "kb")
        os.makedirs(kb_dir, exist_ok=True)
        
        # 사용 모델 추적
        models_used = {
            "meta_extract": self.model_selector.get_model("meta_extract"),
            "analyzer": self.model_selector.get_model("analyzer"),
            "profile": self.model_selector.get_model("profile"),
        }
        
        self.logger.info("=" * 50)
        self.logger.info("KB 빌드 시작")
        self.logger.info("모델: meta=%s, analyzer=%s, profile=%s",
                        models_used["meta_extract"], models_used["analyzer"], models_used["profile"])
        self.logger.info("=" * 50)
        
        # 1. 데이터 수집
        self.logger.info("[1/4] 데이터 수집")
        
        papers: List[PaperMeta] = []
        pdf_texts: Dict[str, str] = {}
        source_counts = {"scholar": 0, "pdf": 0}
        
        scholar_url = kb_cfg.get("sources", {}).get("scholar_url", "").strip()
        if scholar_url:
            scholar_papers = self.scholar.fetch_profile(scholar_url)
            papers.extend(scholar_papers)
            source_counts["scholar"] = len(scholar_papers)
        
        assets_dir = kb_cfg.get("sources", {}).get("assets_dir", "").strip()
        if assets_dir and os.path.exists(assets_dir):
            pdf_results, pdf_warnings = self.pdf_proc.process_directory(assets_dir)
            self.warnings.extend(pdf_warnings)
            
            for pdf_path, text in pdf_results:
                meta = self.pdf_proc.extract_paper_meta(pdf_path, text)
                is_dup = any(self._is_same_paper(meta.title, p.title) for p in papers)
                if is_dup:
                    for p in papers:
                        if self._is_same_paper(meta.title, p.title):
                            pdf_texts[p.title] = text
                            break
                else:
                    papers.append(meta)
                    pdf_texts[meta.title] = text
            
            source_counts["pdf"] = len(pdf_results)
        
        if not papers:
            self.errors.append("수집된 논문 없음")
            return self._create_report(started_at, source_counts, 0, 0, 0, models_used)
        
        self.logger.info(f"  → 총 {len(papers)}개 논문")
        
        # 2. 분석
        self.logger.info("[2/4] 논문 분석")
        
        analyses = []
        for i, paper in enumerate(papers):
            self.logger.debug(f"  [{i+1}/{len(papers)}] {paper.title[:40]}...")
            full_text = pdf_texts.get(paper.title)
            analysis = self.analyzer.analyze(paper, full_text)
            analyses.append(analysis)
        
        self.logger.info(f"  → {len(analyses)}개 분석 완료")
        
        # 3. KB 구축
        self.logger.info("[3/4] KB 구축")
        
        chunks = self.kb_builder.build_chunks(analyses)
        summary = self.kb_builder.generate_summary(analyses)
        index, dim = self.indexer.create_index(chunks)
        
        # 4. 저장
        self.logger.info("[4/4] 파일 저장")
        
        output_cfg = kb_cfg.get("output", {})
        
        # chunks
        chunks_path = os.path.join(kb_dir, output_cfg.get("chunks_file", "kb_chunks.jsonl"))
        with open(chunks_path, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps({
                    "id": c.id, "paper_id": c.paper_id, "section": c.section,
                    "text": c.text, "keywords": c.keywords, "importance": c.importance,
                    "meta": c.meta
                }, ensure_ascii=False) + "\n")
        
        # index
        index_path = os.path.join(kb_dir, output_cfg.get("index_file", "kb_index.faiss"))
        faiss.write_index(index, index_path)
        
        # papers
        papers_path = os.path.join(kb_dir, output_cfg.get("papers_file", "kb_papers.json"))
        with open(papers_path, "w", encoding="utf-8") as f:
            papers_data = [
                {"paper_id": a.paper_id, "title": a.meta.title, "authors": a.meta.authors,
                 "year": a.meta.year, "venue": a.meta.venue, "citations": a.meta.citations,
                 "importance_score": a.importance_score, "keywords": a.keywords,
                 "key_contributions": a.key_contributions, "problem_addressed": a.problem_addressed,
                 "proposed_method": a.proposed_method, "results_summary": a.results_summary}
                for a in analyses
            ]
            json.dump(papers_data, f, ensure_ascii=False, indent=2)
        
        # summary
        summary_path = os.path.join(kb_dir, output_cfg.get("summary_file", "kb_summary.json"))
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # meta
        meta_path = os.path.join(kb_dir, output_cfg.get("meta_file", "kb_meta.json"))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "total_papers": len(papers),
                "total_chunks": len(chunks),
                "embedding_dim": dim,
                "sources": source_counts,
                "models_used": models_used,
                "created_at": datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)
        
        # report
        report = self._create_report(started_at, source_counts, len(papers), len(chunks), dim, models_used)
        report_path = os.path.join(kb_dir, output_cfg.get("report_file", "build_report.json"))
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, ensure_ascii=False, indent=2)
        
        self.logger.info("=" * 50)
        self.logger.info(f"KB 빌드 완료: {kb_dir}/")
        self.logger.info(f"  논문: {len(papers)}개, 청크: {len(chunks)}개")
        self.logger.info(f"  소요시간: {report.duration_seconds:.1f}초")
        self.logger.info("=" * 50)
        
        return report
    
    def _is_same_paper(self, t1: str, t2: str) -> bool:
        def norm(t):
            return re.sub(r'[^a-z0-9]', '', t.lower())
        n1, n2 = norm(t1), norm(t2)
        return n1 == n2 or n1 in n2 or n2 in n1
    
    def _create_report(self, started_at: datetime, sources: Dict,
                       papers: int, chunks: int, dim: int, models_used: Dict) -> BuildReport:
        completed_at = datetime.now()
        return BuildReport(
            status="success" if not self.errors else "failed",
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            duration_seconds=(completed_at - started_at).total_seconds(),
            sources=sources,
            papers_processed=papers,
            chunks_created=chunks,
            embedding_dim=dim,
            warnings=self.warnings,
            errors=self.errors,
            models_used=models_used
        )


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="KB 빌더")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"[ERROR] 설정 파일 없음: {args.config}")
        return
    
    try:
        import faiss
        from openai import OpenAI
        from pypdf import PdfReader
    except ImportError as e:
        print(f"[ERROR] 패키지 설치 필요: {e}")
        print("pip install openai pypdf faiss-cpu scholarly numpy pyyaml")
        return
    
    config = load_config(args.config)
    pipeline = KBPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
