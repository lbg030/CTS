# CTS 자기소개서 시스템 재설계 - 최종 요약

> **작성일**: 2026-01-04
> **작성자**: Claude Sonnet 4.5
> **목적**: 품질 정체, 출력 차단, 에이전트 상호작용 불명확성 문제의 근본적 해결

---

## 📌 핵심 요약

본 작업은 CTS 기반 자동 자기소개서 생성 파이프라인에서 발생하는 **3가지 핵심 문제**를 근본적으로 해결하기 위한 재설계입니다.

### 해결한 문제

1. ✅ **본문 출력 차단** → 9.5점 미만이어도 본문 + 개선 가이드 출력
2. 📋 **점수 정체** → COT 3단계 (WHY-WHAT-HOW) Refine 전략 설계
3. 📋 **에이전트 상호작용 불명확** → 스키마 기반 구조화된 통신 설계

---

## 📂 생성된 문서

### 1. [REDESIGN_PROPOSAL.md](REDESIGN_PROPOSAL.md)
**포괄적 재설계 제안서 (56KB)**

#### 내용:
- 현재 시스템의 구조적 문제점 4가지 상세 분석
- 재설계 핵심 원칙 4가지
- 에이전트 간 상호작용 스키마 (A~G, 7개 에이전트)
- COT 기반 Refine 3단계 전략 (WHY-WHAT-HOW)
- 본문 출력 로직 개선 방안
- 프롬프트 재설계 (Before/After 비교)
- 구현 로드맵 (Week 1~4)
- 예상 결과 시뮬레이션

#### 핵심 개선 사항:

| 항목 | Before (현재) | After (재설계) |
|------|---------------|----------------|
| 본문 출력 | 9.5 미만 시 차단 ❌ | 항상 출력 + 개선 가이드 ✅ |
| Refine 전략 | 추상적 프롬프트 ❌ | COT 3단계 ✅ |
| 에이전트 통신 | 느슨한 dict ❌ | 스키마 기반 ✅ |
| 점수 상승 | 반복해도 정체 ❌ | 구체적 변경 → 예측 가능 ✅ |

---

### 2. [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)
**단계별 구현 가이드 (28KB)**

#### 내용:
- ✅ Phase 1 구현 완료: 본문 출력 로직 수정
  - `run_sop.py` 수정 내역 상세
  - Before/After 코드 비교
  - 예상 출력 예시
- 📋 Phase 2~4 구현 계획
  - 스키마 정의 (`schemas.py`)
  - COT 기반 Refine (`refine_v2.py`)
  - Scorer 프롬프트 개선
- 테스트 방법
- 문제 해결 가이드
- Week별 로드맵

---

## 🔧 실제 코드 변경 사항

### ✅ 완료: [run_sop.py](run_sop.py) 수정

#### 1. `write_markdown` 함수 시그니처 변경
```python
# ✅ cfg 파라미터 추가
def write_markdown(
    ...,
    cfg: Optional[Dict] = None
) -> None:
```

#### 2. 품질 스코어 표시 개선
```python
# ✅ 목표 점수, 갭, 모듈별 상태 표시
md.append(f"- **총점**: {score_result.total_score:.2f}/10.0\n")
md.append(f"- **목표**: {pass_threshold:.1f}/10.0\n")
md.append(f"- **상태**: {'✅ 통과' if score_result.passed else f'⚠️ 미달 (갭: -{gap:.2f}점)'}\n")

md.append("\n| 항목 | 점수 | 상태 |\n|------|------|------|\n")
for name, score in score_result.criteria_scores.items():
    status = "✅" if score >= min_score else "❌"
    md.append(f"| {name} | {score:.2f} | {status} |\n")
```

#### 3. 개선 가이드 추가
```python
# ✅ 9.5 미달 시 개선 가이드 표시
if not score_result.passed and score_result.recommendations:
    md.append("\n### ⚠️ 품질 개선 가이드\n\n")
    md.append(f"현재 점수가 목표({pass_threshold:.1f})에 **{gap:.2f}점** 미달합니다.\n\n")
    md.append("#### 개선 권고 사항 (우선순위순)\n\n")
    for i, rec in enumerate(score_result.recommendations[:5], 1):
        md.append(f"{i}. **권고**: {rec}\n")
```

#### 4. 본문 출력 차단 로직 제거
```python
# ❌ Before: allow_body_output = False로 차단
# ✅ After: 항상 출력 + 미달 시 경고

md.append("\n---\n\n## 📝 제출용 본문\n\n")

if score_result and not score_result.passed:
    md.append("> ⚠️ **주의**: 이 본문은 현재 품질 기준(9.5/10)에 미달합니다.\n")
    md.append("> 위 개선 권고 사항을 참고하여 수정 후 제출하시기 바랍니다.\n\n")

if submission_text:
    md.append(submission_text.strip() + "\n")
```

#### 5. 본문 파일(.txt) 생성 로직 개선
```python
# ✅ 항상 파일 생성 (미달 시 경고 주석 추가)
if submission_path and submission_text:
    ensure_dir(submission_path)
    with open(submission_path, "w", encoding="utf-8") as f:
        if score_result and not score_result.passed:
            f.write("<!-- ⚠️ 경고: 이 본문은 품질 기준(9.5/10) 미달. 개선 후 제출 필요 -->\n\n")
        f.write(submission_text.strip() + "\n")
```

#### 6. 터미널 출력 로직 개선
```python
# ✅ 항상 출력 가능 (미달 시 경고 표시)
if cfg.get("output", {}).get("print_final_to_terminal", False):
    print("\n" + "=" * 60)
    if score_result and not score_result.passed:
        print(f"⚠️ 경고: 품질 기준(9.5/10) 미달 (현재: {score_result.total_score:.2f})")
        print("=" * 60)
    print(final_text)
    print("=" * 60)
```

---

## 📊 예상 효과

### Before (현재 시스템)

```
[실행]
→ 18회 Refine 반복
→ 최종: 8.8/10 (미달)

[출력]
📊 품질 스코어: 8.8/10 (❌ 미달)
📝 제출용 본문: 품질 기준 미달로 출력하지 않습니다.

[문제]
❌ 본문을 볼 수 없음
❌ 왜 실패했는지 모름
❌ 어떻게 고쳐야 할지 모름
```

### After (재설계 시스템)

#### Phase 1 적용 후 (현재)

```
[실행]
→ (동일한 Refine 로직)
→ 최종: 8.8/10 (미달)

[출력]
📊 품질 스코어

- 총점: 8.80/10.0
- 목표: 9.5/10.0
- 상태: ⚠️ 미달 (갭: -0.70점)

| 항목 | 점수 | 상태 |
|------|------|------|
| question_focus | 8.50 | ❌ |
| specificity | 8.00 | ❌ |
| logic_flow | 9.00 | ✅ |

### ⚠️ 품질 개선 가이드

현재 점수가 목표(9.5)에 0.70점 미달합니다.

#### 개선 권고 사항

1. **권고**: 질문에 직접 답하는 문장을 앞부분에 배치하고 관련 없는 내용을 제거
2. **권고**: 추상 표현을 줄이고 행동/상황/결과를 구체적으로 보강

---

## 📝 제출용 본문

> ⚠️ **주의**: 이 본문은 현재 품질 기준(9.5/10)에 미달합니다.
> 위 개선 권고 사항을 참고하여 수정 후 제출하시기 바랍니다.

(본문 전체 출력...)

[개선]
✅ 본문을 확인 가능
✅ 무엇이 문제인지 파악 가능
✅ 어떻게 고쳐야 할지 안내
```

#### Phase 2~4 적용 후 (예상)

```
[실행]
→ COT 3단계 Refine
  - WHY: question_focus 8.5 → "무관한 내용 2문장"
  - WHAT: "2문단 삭제" (예상 +0.7)
  - HOW: 적용
→ 최종: 9.6/10 ✅

[출력]
📊 품질 스코어: 9.6/10 (✅ 통과)
📝 제출용 본문: (본문 전체...)

[개선]
✅ 2~5회 반복으로 9.5 도달 (18회 → 2~5회)
✅ COT로 개선 과정 투명
✅ 구체적 변경 → 예측 가능한 점수 상승
```

---

## 🗺️ 다음 단계 로드맵

### ✅ Week 1 (완료)
- [x] 문제 분석 및 재설계 방안 수립
- [x] REDESIGN_PROPOSAL.md 작성
- [x] Phase 1: 본문 출력 로직 수정 구현
- [x] IMPLEMENTATION_GUIDE.md 작성

### 📅 Week 2 (다음)
- [ ] Phase 2: 스키마 정의 (`schemas.py`)
  - `PlannerInput/Output`
  - `ReviewerInput/Output`, `Fix`
  - `ScoreResultV2`, `Recommendation`
  - 등 7개 에이전트 스키마
- [ ] Phase 3: COT 기반 Refine 구현 (`refine_v2.py`)
  - WHY: `_analyze_low_score_with_cot()`
  - WHAT: `_propose_concrete_changes()`
  - HOW: `_apply_changes()`

### 📅 Week 3
- [ ] Phase 4: Scorer 프롬프트 개선
- [ ] 통합 테스트
- [ ] 10개 질문으로 Before/After 비교

### 📅 Week 4
- [ ] 성능 튜닝
- [ ] 최종 문서화
- [ ] 릴리스

---

## 📖 문서 구조

```
CTS/
├── REDESIGN_PROPOSAL.md       # 재설계 제안서 (56KB)
│   ├── 1. 현재 시스템의 구조적 문제점
│   ├── 2. 재설계 핵심 원칙
│   ├── 3. 에이전트 간 상호작용 스키마
│   ├── 4. 점수 정체 해결: COT 기반 Refine 전략
│   ├── 5. 본문 출력 로직 개선
│   ├── 6. 프롬프트 재설계
│   └── 7. 구현 가이드
│
├── IMPLEMENTATION_GUIDE.md    # 구현 가이드 (28KB)
│   ├── 1. 구현 완료 항목 (Phase 1)
│   ├── 2. 다음 구현 항목 (Phase 2~4)
│   ├── 3. 코드 변경 사항 상세
│   ├── 4. 테스트 방법
│   └── 5. 다음 단계 로드맵
│
├── SUMMARY.md                 # 최종 요약 (본 문서)
│   ├── 핵심 요약
│   ├── 생성된 문서
│   ├── 실제 코드 변경 사항
│   ├── 예상 효과
│   └── 다음 단계 로드맵
│
└── run_sop.py                 # ✅ 수정 완료
    └── write_markdown() 함수 수정 (6곳)
```

---

## 🎯 핵심 메시지

### 사용자에게

**현재 상태**:
- ✅ 9.5점 미만이어도 **본문을 확인**할 수 있습니다.
- ✅ **무엇이 문제인지** 명확히 알 수 있습니다.
- ✅ **어떻게 고쳐야 하는지** 구체적인 안내를 받을 수 있습니다.

**다음 단계**:
- 📋 점수 정체 문제 해결 (COT 3단계 Refine)
- 📋 에이전트 간 통신 구조화 (스키마 기반)
- 📋 2~5회 반복으로 9.5 도달 (현재 18회 → 목표 2~5회)

### 개발자에게

**현재 변경 사항**:
- `run_sop.py`의 `write_markdown()` 함수 6곳 수정
- 본문 출력 차단 로직 완전 제거
- 품질 스코어 표시 개선 (목표, 갭, 모듈별 상태)
- 개선 가이드 자동 생성

**다음 구현 항목**:
1. `schemas.py`: 에이전트 스키마 정의
2. `refine_v2.py`: COT 3단계 Refine 로직
3. `prompts_v2.py`: Scorer 프롬프트 개선
4. `config.yaml`: `use_cot`, `predict_score_impact` 추가

**구현 원칙**:
- 추상적 방향 제시 지양
- 코드/프롬프트/스키마 관점에서 구체적 설명
- 다른 AI가 재구현 가능한 수준의 명시성

---

## 🔗 참고 링크

- **GitHub 저장소**: https://github.com/lbg030/CTS
- **재설계 제안서**: [REDESIGN_PROPOSAL.md](REDESIGN_PROPOSAL.md)
- **구현 가이드**: [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)
- **설정 파일**: [config.yaml](config.yaml)

---

## ✍️ 작성 정보

- **작성자**: Claude Sonnet 4.5 (model ID: claude-sonnet-4-5-20250929)
- **작성일**: 2026-01-04
- **작업 시간**: 약 2시간
- **생성 문서**: 3개 (총 ~100KB)
- **코드 수정**: 1개 파일, 6곳

---

## 📋 체크리스트

### 완료 항목
- [x] 현재 시스템 문제점 분석
- [x] 재설계 방안 수립
- [x] 에이전트 스키마 설계
- [x] COT 기반 Refine 전략 설계
- [x] 본문 출력 로직 수정 구현
- [x] 문서화 (REDESIGN_PROPOSAL.md)
- [x] 구현 가이드 (IMPLEMENTATION_GUIDE.md)
- [x] 최종 요약 (SUMMARY.md)

### 다음 항목
- [ ] Phase 2: 스키마 구현
- [ ] Phase 3: COT Refine 구현
- [ ] Phase 4: Scorer 프롬프트 개선
- [ ] 통합 테스트
- [ ] 성능 검증

---

**이 문서는 CTS 자기소개서 시스템 재설계 작업의 최종 요약본입니다.**
**상세 내용은 [REDESIGN_PROPOSAL.md](REDESIGN_PROPOSAL.md)와 [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)를 참조하시기 바랍니다.**
