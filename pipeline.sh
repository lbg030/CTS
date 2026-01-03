#!/usr/bin/env bash
set -e

echo "======================================"
echo " CTS Pipeline Start"
echo "======================================"

CONFIG_PATH="config.yaml"

# 사전 검증
echo "[0/5] 사전 검증"
echo "--------------------------------------"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다."
    exit 1
fi

for f in build_kb.py query_kb.py build_company_profile.py collect_pass_sop_patterns.py run_sop.py; do
    if [ ! -f "$f" ]; then
        echo "❌ $f 파일이 없습니다."
        exit 1
    fi
done

if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ $CONFIG_PATH 파일이 없습니다."
    exit 1
fi

echo "✓ 사전 검증 완료"

echo ""
echo "[1/5] KB 빌드 시작"
echo "--------------------------------------"
python build_kb.py --config ${CONFIG_PATH}
echo "✓ KB 빌드 완료 (kb/ 폴더 확인)"

echo ""
echo "[2/5] KB 검색 테스트"
echo "--------------------------------------"
python query_kb.py --config ${CONFIG_PATH} --query "3D reconstruction" --mode search
echo "✓ KB 검색 테스트 완료"

echo ""
echo "[3/5] 회사 프로필 생성"
echo "--------------------------------------"
python build_company_profile.py --config ${CONFIG_PATH}
echo "✓ 회사 프로필 생성 완료 (company_db/*.json 확인)"

echo ""
echo "[4/5] 합격 자소서 구조 패턴 수집"
echo "--------------------------------------"
python collect_pass_sop_patterns.py --config ${CONFIG_PATH}
echo "✓ 합격 자소서 구조 패턴 수집 완료 (temp/ 확인)"

echo ""
echo "[5/5] 자기소개서 생성"
echo "--------------------------------------"
python run_sop.py --config ${CONFIG_PATH}
echo "✓ 자기소개서 생성 완료 (outputs/*.md 확인)"

echo ""
echo "======================================"
echo " CTS Pipeline Finished Successfully"
echo "======================================"
