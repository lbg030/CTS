#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
query_kb.py - KB 검색 유틸리티 (run_sop.py 연동용)

역할:
  1. build_kb.py가 생성한 KB를 검색
  2. run_sop.py가 Agent에게 근거를 제공할 때 사용
  3. 쿼리 → 관련 청크 반환 (RAG)

사용법:
  # CLI 테스트
  python query_kb.py --config config.yaml --query "3D reconstruction 경험"
  
  # Python (run_sop.py에서)
  from query_kb import KBSearcher
  searcher = KBSearcher(config)
  results = searcher.search("diffusion model", top_k=5)
"""

import os
import json
import yaml
import argparse
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from openai import OpenAI


class KBSearcher:
    """Knowledge Base 검색기"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: config.yaml 전체 또는 필요한 부분
        """
        self.config = config
        
        # API 클라이언트
        api_key = config.get("openai", {}).get("api_key", "").strip()
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY", "")
        self.client = OpenAI(api_key=api_key)
        
        # 경로
        kb_dir = config.get("paths", {}).get("kb_dir", "kb")
        
        # RAG 설정
        rag_cfg = config.get("rag", {})
        self.embed_model = rag_cfg.get("embed_model", "text-embedding-3-large")
        self.default_top_k = rag_cfg.get("top_k", 6)
        self.max_evidence_chars = rag_cfg.get("max_evidence_chars", 1800)
        self.min_importance = rag_cfg.get("min_importance", 0.0)
        
        # KB 로드
        self.kb_dir = kb_dir
        self.chunks = self._load_chunks()
        self.index = self._load_index()
        self.summary = self._load_summary()
        self.papers = self._load_papers()
    
    def _load_chunks(self) -> List[Dict]:
        path = os.path.join(self.kb_dir, "kb_chunks.jsonl")
        chunks = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
        return chunks
    
    def _load_index(self):
        path = os.path.join(self.kb_dir, "kb_index.faiss")
        return faiss.read_index(path)
    
    def _load_summary(self) -> Dict:
        path = os.path.join(self.kb_dir, "kb_summary.json")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _load_papers(self) -> List[Dict]:
        path = os.path.join(self.kb_dir, "kb_papers.json")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    # ─────────────────────────────────────────
    # 핵심 검색 메서드
    # ─────────────────────────────────────────
    
    def search(self, query: str, top_k: Optional[int] = None,
               min_importance: Optional[float] = None,
               section_filter: Optional[List[str]] = None) -> List[Dict]:
        """
        쿼리로 관련 청크 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 최대 결과 수
            min_importance: 최소 중요도 필터
            section_filter: 섹션 필터 (예: ["contributions", "meta"])
        
        Returns:
            [{"score": float, "chunk": Dict}, ...]
        """
        top_k = top_k or self.default_top_k
        min_importance = min_importance if min_importance is not None else self.min_importance
        
        # 쿼리 임베딩
        resp = self.client.embeddings.create(model=self.embed_model, input=[query])
        query_vec = np.array([resp.data[0].embedding], dtype=np.float32)
        faiss.normalize_L2(query_vec)
        
        # 검색
        k = min(top_k * 3, len(self.chunks))
        scores, indices = self.index.search(query_vec, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            
            chunk = self.chunks[idx]
            
            if chunk.get("importance", 0) < min_importance:
                continue
            if section_filter and chunk.get("section") not in section_filter:
                continue
            
            results.append({"score": float(score), "chunk": chunk})
            
            if len(results) >= top_k:
                break
        
        return results
    
    def get_evidence_for_claim(self, claim: str, top_k: int = 3) -> List[Dict]:
        """
        주장에 대한 근거 검색 (자기소개서 작성용)
        
        Returns:
            [{"text": str, "source": str, "relevance": float, "importance": float}, ...]
        """
        results = self.search(claim, top_k=top_k, min_importance=0.2)
        
        evidences = []
        for r in results:
            chunk = r["chunk"]
            evidences.append({
                "text": chunk["text"],
                "source": chunk.get("meta", {}).get("title", "Unknown"),
                "section": chunk.get("section"),
                "relevance": r["score"],
                "importance": chunk.get("importance", 0)
            })
        
        return evidences
    
    # ─────────────────────────────────────────
    # 요약 정보 접근
    # ─────────────────────────────────────────
    
    def get_researcher_profile(self) -> str:
        return self.summary.get("researcher_profile", "")
    
    def get_top_keywords(self, n: int = 10) -> List[str]:
        return [kw for kw, _ in self.summary.get("top_keywords", [])[:n]]
    
    def get_research_areas(self) -> List[str]:
        return self.summary.get("research_areas", [])
    
    def get_representative_papers(self, n: int = 3) -> List[Dict]:
        return self.summary.get("representative_papers", [])[:n]
    
    def get_technical_expertise(self) -> List[str]:
        return self.summary.get("technical_expertise", [])
    
    # ─────────────────────────────────────────
    # run_sop.py 연동용
    # ─────────────────────────────────────────
    
    def get_context_for_requirements(self, requirements: List[str]) -> Dict[str, Any]:
        """
        회사 요구사항에 대한 컨텍스트 생성
        
        Args:
            requirements: ["diffusion model", "self-motivated", ...]
        
        Returns:
            Agent 프롬프트에 삽입할 컨텍스트
        """
        context = {
            "researcher_profile": self.get_researcher_profile(),
            "research_areas": self.get_research_areas(),
            "top_keywords": self.get_top_keywords(15),
            "representative_papers": self.get_representative_papers(5),
            "technical_expertise": self.get_technical_expertise(),
            "requirement_matches": {}
        }
        
        for req in requirements:
            evidences = self.get_evidence_for_claim(req, top_k=3)
            context["requirement_matches"][req] = evidences
        
        return context
    
    def format_context_for_prompt(self, context: Dict[str, Any]) -> str:
        """컨텍스트를 프롬프트 삽입용 문자열로 변환"""
        lines = []
        
        lines.append("## 연구자 프로필")
        lines.append(context["researcher_profile"])
        
        lines.append("\n## 연구 분야")
        lines.append(", ".join(context["research_areas"]))
        
        lines.append("\n## 핵심 키워드")
        lines.append(", ".join(context["top_keywords"]))
        
        lines.append("\n## 기술 전문성")
        lines.append(", ".join(context["technical_expertise"]))
        
        lines.append("\n## 대표 논문")
        for p in context["representative_papers"]:
            lines.append(f"- {p['title']} ({p['venue']}, {p['year']})")
            if p.get("contribution"):
                lines.append(f"  → {p['contribution']}")
        
        lines.append("\n## 요구사항별 근거")
        for req, evidences in context["requirement_matches"].items():
            lines.append(f"\n### {req}")
            for ev in evidences:
                text = ev["text"][:300] + "..." if len(ev["text"]) > 300 else ev["text"]
                lines.append(f"- [{ev['source']}] {text}")
        
        # 최대 길이 제한
        result = "\n".join(lines)
        if len(result) > self.max_evidence_chars:
            result = result[:self.max_evidence_chars] + "\n...(truncated)"
        
        return result
    
    def get_evidence_text(self, query: str, top_k: int = 5) -> str:
        """
        쿼리에 대한 근거를 텍스트로 반환 (Agent 프롬프트 삽입용)
        """
        results = self.search(query, top_k=top_k)
        
        if not results:
            return "[관련 근거 없음]"
        
        parts = []
        total_chars = 0
        
        for i, r in enumerate(results, 1):
            chunk = r["chunk"]
            text = chunk["text"]
            
            # 길이 제한
            if total_chars + len(text) > self.max_evidence_chars:
                remaining = self.max_evidence_chars - total_chars
                if remaining > 200:
                    text = text[:remaining] + "..."
                else:
                    break
            
            parts.append(f"[근거 {i}] (관련도: {r['score']:.2f})\n"
                        f"출처: {chunk.get('meta', {}).get('title', 'Unknown')}\n"
                        f"{text}")
            total_chars += len(text)
        
        return "\n\n---\n\n".join(parts)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="KB 검색")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--mode", type=str, default="search",
                        choices=["search", "evidence", "profile", "context"])
    args = parser.parse_args()
    
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    searcher = KBSearcher(config)
    
    if args.mode == "search":
        results = searcher.search(args.query, top_k=args.top_k)
        print(f"\n🔍 검색: '{args.query}' ({len(results)}개 결과)")
        print("=" * 50)
        for i, r in enumerate(results, 1):
            chunk = r["chunk"]
            print(f"\n[{i}] Score: {r['score']:.3f} | Importance: {chunk.get('importance', 0):.2f}")
            print(f"Source: {chunk.get('meta', {}).get('title', 'Unknown')}")
            print(f"Text: {chunk['text'][:200]}...")
    
    elif args.mode == "evidence":
        evidences = searcher.get_evidence_for_claim(args.query, top_k=args.top_k)
        print(f"\n📋 근거: '{args.query}'")
        print("=" * 50)
        for i, ev in enumerate(evidences, 1):
            print(f"\n[{i}] {ev['source']}")
            print(f"  관련도: {ev['relevance']:.3f}, 중요도: {ev['importance']:.2f}")
            print(f"  {ev['text'][:300]}...")
    
    elif args.mode == "profile":
        print("\n👤 연구자 프로필")
        print("=" * 50)
        print(searcher.get_researcher_profile())
        print(f"\n연구 분야: {', '.join(searcher.get_research_areas())}")
        print(f"키워드: {', '.join(searcher.get_top_keywords(10))}")
    
    elif args.mode == "context":
        # 쿼리를 요구사항 리스트로 파싱
        requirements = [r.strip() for r in args.query.split(",")]
        context = searcher.get_context_for_requirements(requirements)
        print("\n📝 컨텍스트")
        print("=" * 50)
        print(searcher.format_context_for_prompt(context))


if __name__ == "__main__":
    main()
