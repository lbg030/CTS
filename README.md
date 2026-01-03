# Multi-Agent SOP Writer (Company Profile + RAG KB)

이 프로젝트는 **“회사/공고 사전정보(인재상·필수·우대) + 사용자 자료(RAG KB)”**를 결합해,  
기업 질문만 입력해도 **공백 포함 950~1000자 자기소개서(.md)** 를 자동 생성하는 파이프라인입니다.

구성은 4단계입니다.

1) **회사/공고 사전정보 구축**: `build_company_profile.py` → `company_db/<slug>.json`  
2) **사용자 자료 RAG KB 구축**: `build_kb.py` (PDF + workflow md) → `kb/`  
3) **합격 자소서 구조 패턴 수집**: `collect_pass_sop_patterns.py` → `temp/`, `patterns/`  
4) **자기소개서 생성**: `run_sop.py` (회사 프로필 + 근거 top-k + 구조 가이드/패턴 요약) → `outputs/*.md`

> ✅ 최신 `run_sop.py`는  
> - **임베딩 캐시**로 비용 절약  
> - **근거 부족 감지** 시 결과 md에 경고 섹션 자동 삽입  
> - **출력 파일명에 질문 슬러그 포함**(옵션)  
> - **회사 프로필 스키마 보정**으로 누락/타입 이상에 강함  
> - **합격 자소서 패턴 요약(md/json/structured) 자동 반영**  
> - **진행 로그를 촘촘하게 출력**(단계별 + 소요시간)

---

## 0) 요구사항

- Python 3.10+ 권장

### 설치
```bash
pip install openai pyyaml requests beautifulsoup4 pypdf faiss-cpu numpy
```

### OpenAI API Key 설정
권장: **환경변수**
```bash
export OPENAI_API_KEY="sk-..."
```

또는 (비권장): `config.yaml`에 직접 입력  
- `openai.api_key`에 붙여넣기 (공유/깃 업로드 시 유출 주의)
> 참고: `build_company_profile.py`/`build_kb.py`는 환경변수(또는 코드 상단 `OPENAI_API_KEY`)를 사용합니다.  
> `run_sop.py`는 `config.yaml`의 `openai.api_key` 또는 환경변수를 사용합니다.

---

## 1) 권장 폴더 구조

```
project/
  assets/
    Lee_Byeonggwon_Portfolio_final.pptx.pdf

  gpt_multi_agent_workflow.md
  config.yaml

  build_company_profile.py
  build_kb.py
  collect_pass_sop_patterns.py
  run_sop.py

  kb/            # build_kb.py 실행 후 생성
  company_db/    # build_company_profile.py 실행 후 생성
  temp/          # 합격 자소서 구조 패턴(선택)
  patterns/      # 합격 자소서 구조 패턴 요약/로그(선택)
    sources/     # 수집 URL/검색 로그
  outputs/       # run_sop.py 결과(.md)
  cache/         # 임베딩 캐시(옵션)
  logs/          # 실행 로그(옵션)
```

---

## 2) config.yaml 한 번에 관리하기 (중요)

`build_company_profile.py`/`collect_pass_sop_patterns.py`/`run_sop.py`는 **config.yaml**을 사용합니다.  
`build_kb.py`는 CLI 인자(`--pdf`, `--workflow_md`, `--outdir`)로 실행합니다.

필수 섹션(요약):
- `paths`: workflow/md, kb, company_db 경로
- `openai`: model/토큰/재시도
- `company`: 회사명/직무
- `pass_sop_patterns`: 합격 자소서 구조 패턴 수집 설정
- `rag`: embed_model/top_k/max_evidence_chars
- `application`: 질문(단일 또는 배치) + 글자수
- `output`: md 파일명 템플릿 + 터미널 본문 출력 여부
- `logging`: 진행 로그 레벨/파일 저장
- `cache`: 임베딩 캐시 on/off
- `evidence_quality`: 근거 부족 감지 임계값

`pass_sop_patterns`에서는 **계층적 폴백 검색**, **회사/직무 별칭**, **중단 기준**, **검색 백엔드(DDG 폴백)** 등을 조절할 수 있습니다.

---

## 3) 1단계: 회사/공고 사전정보 저장 (Company Profile)

### 왜 필요한가?
- run_sop 실행 때마다 회사 인재상/요건을 재탐색하면 비용/시간이 증가합니다.
- 공고/인재상은 자주 변하지 않으므로 **사전 JSON으로 저장**해 재사용하는 편이 효율적입니다.

### 설정
`config.yaml`의 `company_profile.job_posting_urls`에 **채용공고 URL**을 넣으세요(가장 정확).
`enable_web_search`는 보조 수집 옵션이며 계정/모델에서 web_search 지원이 필요합니다.

예:
```yaml
company_profile:
  job_posting_urls:
    - "https://careers.example.com/jobs/12345"
  enable_web_search: false
```

### 실행
```bash
python build_company_profile.py --config config.yaml
```

### 결과
- `company_db/<company_slug>.json` 생성  
- `company_slug`를 비워두면 company.name에서 자동 slugify 됩니다.

---

## 4) 2단계: 사용자 포트폴리오 기반 RAG KB 구축

### 왜 필요한가?
- 포트폴리오 PDF 전체를 매번 프롬프트에 넣으면 토큰 낭비/비용 증가합니다.
- 임베딩 인덱스를 만들어 질문 관련 근거만 top-k로 가져오면 효율적입니다.

### 실행
```bash
python build_kb.py \
  --pdf assets/Lee_Byeonggwon_Portfolio_final.pptx.pdf \
  --workflow_md guide_md/gpt_multi_agent_workflow.md \
  --outdir kb
```

### 결과
- `kb/kb_chunks.jsonl`
- `kb/kb_index.faiss`
- `kb/kb_meta.json`

---

## 5) 3단계: 합격 자소서 구조 패턴 수집 (선택)

회사명/직무 키워드를 기반으로 공개 합격 자소서를 자동 수집하고, 문장 복사 없이 구조적 흐름만 추출합니다.  
결과는 `temp/`/`patterns/`에 저장되며, `run_sop.py`에서 **구조 가이드로만** 사용됩니다.

### 실행
```bash
python collect_pass_sop_patterns.py --config config.yaml
```

### 결과
- `temp/essays_raw/` (원문 HTML)
- `temp/essays_clean/` (정제 텍스트)
- `temp/essays_structured.jsonl` (구조 패턴)
- `temp/flow_guide.md` (구조 가이드)
- `patterns/pass_sop_patterns.json` (구조 요약 + 상태)
- `patterns/pass_sop_patterns.md` (가이드 문서)
- `patterns/sources/search_log.json` (검색 로그)
- `patterns/sources/sources.jsonl` (수집 URL 메타)

> 참고: `run_sop.py`는 위 결과 중 **구조 요약만** 사용하며, 원문 텍스트(`temp/essays_raw`, `temp/essays_clean`)는 직접 사용하지 않습니다.

## 6) 4단계: 자기소개서 생성(run_sop.py)

### 단일 질문 실행
`config.yaml`의 `application.question`을 사용합니다.

```bash
python run_sop.py --config config.yaml
```

### 배치 실행(여러 질문)
`config.yaml`에 `application.questions` 리스트를 넣으면 질문 개수만큼 md가 생성됩니다.

예:
```yaml
application:
  questions:
    - "[필수] 자신에 대해 자유롭게 표현해주세요."
    - "지원 동기와 직무 역량을 구체적 사례로 설명해 주세요."
```

실행:
```bash
python run_sop.py --config config.yaml
```

---

## 7) 출력(.md) 파일명 템플릿(질문 슬러그 포함)

최신 `run_sop.py`는 `{question_slug}` 변수를 지원합니다.

예:
```yaml
output:
  template: "outputs/{company_slug}_{timestamp}_q{q_idx}_{question_slug}.md"
  print_final_to_terminal: false
```

- `{company_slug}`: 회사명 slug
- `{timestamp}`: 실행 시각 (YYYYmmdd_HHMMSS)
- `{q_idx}`: 배치 실행 인덱스
- `{question_slug}`: 질문 앞부분을 축약한 slug

---

## 8) 임베딩 캐시(비용 절약)

동일 query를 반복 실행할 경우 임베딩 호출을 줄이기 위해 캐시를 사용할 수 있습니다.

```yaml
cache:
  enable_embedding_cache: true
  embedding_cache_path: "cache/embeddings.json"
```

---

## 9) 근거 부족 감지(결과 md에 경고 자동 삽입)

KB에서 가져온 근거가 부족하면 결과 md 상단에 **⚠️ 근거 부족 경고**가 자동으로 추가됩니다.

임계값 조정:
```yaml
evidence_quality:
  min_hits: 3
  min_total_chars: 400
  min_score: 0.20
```

---

## 10) 로깅(진행 상황 촘촘하게 보기)

```yaml
logging:
  level: "INFO"      # DEBUG로 올리면 더 자세
  quiet: false       # true면 WARNING 이상만 콘솔 출력
  to_file: true
  file_path: "logs/run.log"
```

---

## 11) Troubleshooting

### Q1) 회사 프로필 JSON을 못 찾는다고 나옵니다.
- 먼저 1단계를 실행했는지 확인:
```bash
python build_company_profile.py --config config.yaml
```
- `company_db/`에 `<slug>.json` 파일이 생성됐는지 확인하세요.

### Q2) KB 파일을 못 찾는다고 나옵니다.
- 먼저 2단계를 실행했는지 확인:
```bash
python build_kb.py --pdf ... --workflow_md ... --outdir kb
```
- `kb/` 폴더에 `kb_chunks.jsonl`, `kb_index.faiss`가 있는지 확인하세요.

### Q3) 결과 md에 근거 부족 경고가 뜹니다.
- 포트폴리오 PDF에 해당 질문 관련 내용이 충분히 포함되어 있는지 확인하세요.
- 채용공고 URL을 추가한 뒤 `build_company_profile.py`를 다시 실행하세요.
- 질문을 더 구체화해서 다시 실행해보세요.

---

## 12) 빠른 시작(요약)

```bash
export OPENAI_API_KEY="sk-..."

# 1) 회사 프로필 구축 (공고 URL을 config.yaml에 넣고 실행)
python build_company_profile.py --config config.yaml

# 2) 사용자 KB 구축
python build_kb.py --pdf assets/Lee_Byeonggwon_Portfolio_final.pptx.pdf --workflow_md guide_md/gpt_multi_agent_workflow.md --outdir kb

# 3) 합격 자소서 구조 패턴 수집(선택)
python collect_pass_sop_patterns.py --config config.yaml

# 4) 자기소개서 생성 (결과는 outputs/*.md)
python run_sop.py --config config.yaml

# 전체 파이프라인 실행 (build_kb → 회사 프로필 → 합격 자소서 → run_sop)
./run.sh --config config.yaml --query "3D reconstruction"
```
