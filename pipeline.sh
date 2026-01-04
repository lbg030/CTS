#!/usr/bin/env bash
set -e

echo "======================================"
echo " CTS Pipeline (v2.0 품질 개선)"
echo "======================================"
echo ""
echo " 개선사항:"
echo "   R1) 1인칭 표현 금지"
echo "   R2) 스코어링 + Refine 루프"
echo "   R3) KB 모델 정책"
echo "   R4) 크롤링 품질 개선"
echo "   R5) 문체/흐름 강화"
echo ""
echo "======================================"

CONFIG_PATH="${CONFIG_PATH:-config.yaml}"
QUERY_TEXT="${QUERY_TEXT:-3D reconstruction}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --query)
      QUERY_TEXT="$2"
      shift 2
      ;;
    *)
      echo "Usage: $0 [--config path] [--query text]"
      exit 1
      ;;
  esac
done

# ─────────────────────────────────────────
# [0/5] 사전 검증
# ─────────────────────────────────────────
echo ""
echo "[0/5] 사전 검증"
echo "--------------------------------------"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ $CONFIG_PATH 파일이 없습니다."
    exit 1
fi
echo "✓ Config: $CONFIG_PATH"

if [ -z "${OPENAI_API_KEY:-}" ]; then
    if ! python3 - <<PY
import yaml
cfg = yaml.safe_load(open("${CONFIG_PATH}", "r", encoding="utf-8")) or {}
key = (cfg.get("openai", {}) or {}).get("api_key", "")
raise SystemExit(0 if str(key).strip() else 1)
PY
    then
        echo "❌ OPENAI_API_KEY 환경변수 또는 config.yaml의 openai.api_key가 필요합니다."
        exit 1
    fi
fi
echo "✓ API Key 설정됨"

for f in build_kb.py query_kb.py build_company_profile.py collect_pass_sop_patterns.py run_sop.py; do
    if [ ! -f "$f" ]; then
        echo "❌ $f 파일이 없습니다."
        exit 1
    fi
done
echo "✓ 필수 파일 존재"

# 디렉토리 생성
mkdir -p kb company_db patterns temp outputs logs cache
echo "✓ 디렉토리 준비 완료"

echo "✓ 사전 검증 완료"

# ─────────────────────────────────────────
# [1/5] KB 빌드
# ─────────────────────────────────────────
echo ""
echo "[1/5] KB 빌드 시작 (R3 모델 정책 적용)"
echo "--------------------------------------"
python3 build_kb.py --config ${CONFIG_PATH}

if [ ! -f "kb/kb_chunks.jsonl" ]; then
    echo "⚠️  KB 빌드 결과 없음 (PDF/Scholar 설정 확인)"
else
    echo "✓ KB 빌드 완료"
fi

# ─────────────────────────────────────────
# [2/5] KB 검색 테스트
# ─────────────────────────────────────────
echo ""
echo "[2/5] KB 검색 테스트"
echo "--------------------------------------"
if [ -f "kb/kb_chunks.jsonl" ]; then
    python3 query_kb.py --config ${CONFIG_PATH} --query "${QUERY_TEXT}" --mode search || true
    echo "✓ KB 검색 테스트 완료"
else
    echo "⚠️  KB 없음 - 검색 테스트 건너뜀"
fi

# ─────────────────────────────────────────
# [3/5] 회사 프로필 생성
# ─────────────────────────────────────────
echo ""
echo "[3/5] 회사 프로필 생성"
echo "--------------------------------------"
python3 build_company_profile.py --config ${CONFIG_PATH}
echo "✓ 회사 프로필 생성 완료"

# ─────────────────────────────────────────
# [4/5] 합격 자소서 패턴 수집 (R4)
# ─────────────────────────────────────────
echo ""
echo "[4/5] 합격 자소서 패턴 수집 (R4 품질 개선)"
echo "--------------------------------------"
python3 collect_pass_sop_patterns.py --config ${CONFIG_PATH} || echo "⚠️  패턴 수집 실패 (폴백 사용)"
echo "✓ 합격 자소서 패턴 수집 완료"

# ─────────────────────────────────────────
# [5/5] 자기소개서 생성 (R1, R2, R5)
# ─────────────────────────────────────────
echo ""
echo "[5/5] 자기소개서 생성"
echo "      R1) 1인칭 표현 제거"
echo "      R2) 스코어링 + Refine 루프"
echo "      R5) 문체/흐름 반영 강화"
echo "--------------------------------------"
python3 run_sop.py --config ${CONFIG_PATH}
echo "✓ 자기소개서 생성 완료"

# ─────────────────────────────────────────
# 완료
# ─────────────────────────────────────────
echo ""
echo "======================================"
echo " CTS Pipeline Finished Successfully"
echo "======================================"
echo ""
echo "결과물:"
echo "  - KB: kb/"
echo "  - 회사 프로필: company_db/"
echo "  - 패턴: patterns/"
echo "  - 자기소개서: outputs/*.md"
if [ -f "outputs/refine_report.json" ]; then
echo "  - Refine 리포트: outputs/refine_report.json"
fi
echo ""
