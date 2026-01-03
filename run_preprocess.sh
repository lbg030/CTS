#!/usr/bin/env bash
set -e

echo "======================================"
echo " CTS Pipeline Start"
echo "======================================"

CONFIG_PATH="config.yaml"

# ─────────────────────────────────────────
# [0/4] 사전 검증
# ─────────────────────────────────────────
echo ""
echo "[0/4] 사전 검증"
echo "--------------------------------------"

# API 키 확인
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다."
    echo "   export OPENAI_API_KEY=\"sk-...\""
    exit 1
fi
echo "✓ OPENAI_API_KEY 설정됨"

# 필수 파일 확인
for f in build_kb.py query_kb.py build_company_profile.py run_sop.py; do
    if [ ! -f "$f" ]; then
        echo "❌ $f 파일이 없습니다."
        exit 1
    fi
done
echo "✓ 필수 파일 존재"

# config 확인
if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ $CONFIG_PATH 파일이 없습니다."
    exit 1
fi
echo "✓ $CONFIG_PATH 존재"

# workflow 가이드 확인
if [ ! -f "guide_md/gpt_multi_agent_workflow.md" ]; then
    echo "⚠️  guide_md/gpt_multi_agent_workflow.md 없음 (기본값 사용)"
fi

echo "✓ 사전 검증 완료"

# ─────────────────────────────────────────
# [1/4] KB 빌드
# ─────────────────────────────────────────
echo ""
echo "[1/4] KB 빌드 시작"
echo "--------------------------------------"
python build_kb.py --config ${CONFIG_PATH}

if [ ! -d "kb" ]; then
    echo "❌ KB 빌드 실패: kb/ 폴더가 생성되지 않았습니다."
    exit 1
fi
echo "✓ KB 빌드 완료 (kb/ 폴더 확인)"

# ─────────────────────────────────────────
# [2/4] KB 검색 테스트
# ─────────────────────────────────────────
echo ""
echo "[2/4] KB 검색 테스트"
echo "--------------------------------------"
python query_kb.py --config ${CONFIG_PATH} --query "3D reconstruction" --mode search
echo "✓ KB 검색 테스트 완료"

# ─────────────────────────────────────────
# [3/4] 회사 프로필 생성
# ─────────────────────────────────────────
echo ""
echo "[3/4] 회사 프로필 생성"
echo "--------------------------------------"
python build_company_profile.py --config ${CONFIG_PATH}

if [ ! -d "company_db" ]; then
    echo "❌ 회사 프로필 생성 실패: company_db/ 폴더가 생성되지 않았습니다."
    exit 1
fi
echo "✓ 회사 프로필 생성 완료 (company_db/*.json 확인)"

# ─────────────────────────────────────────
# [4/4] 자기소개서 생성
# ─────────────────────────────────────────
echo ""
echo "[4/4] 자기소개서 생성"
echo "--------------------------------------"
python run_sop.py --config ${CONFIG_PATH}

if [ ! -d "outputs" ]; then
    echo "❌ 자기소개서 생성 실패: outputs/ 폴더가 생성되지 않았습니다."
    exit 1
fi
echo "✓ 자기소개서 생성 완료 (outputs/*.md 확인)"

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
echo "  - 자기소개서: outputs/"
echo ""
