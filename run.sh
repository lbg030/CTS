#!/usr/bin/env bash
set -euo pipefail

# ===== 기본값 (필요 시 인자/환경변수로 덮어쓰기) =====
PIPELINE_SCRIPT="${PIPELINE_SCRIPT:-./pipeline.sh}"
CONFIG_PATH="${CONFIG_PATH:-./config.yaml}"
LOG_DIR="${LOG_DIR:-./logs}"
QUERY_TEXT="${QUERY_TEXT:-3D reconstruction}"
# ===================================================

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
    --pipeline)
      PIPELINE_SCRIPT="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    *)
      echo "Usage: $0 [--config path] [--query text] [--pipeline path] [--log-dir dir]"
      exit 1
      ;;
  esac
done

# 현재 경로 기준으로 실행(스크립트 위치로 이동)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p "${LOG_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/pipeline_${TS}.log"

echo "======================================"
echo " Master Runner"
echo " - pipeline script : ${PIPELINE_SCRIPT}"
echo " - config          : ${CONFIG_PATH}"
echo " - query           : ${QUERY_TEXT}"
echo " - log file        : ${LOG_FILE}"
echo "======================================"

# 실행 파일 존재 확인
if [[ ! -f "${PIPELINE_SCRIPT}" ]]; then
  echo "[ERROR] ${PIPELINE_SCRIPT} 파일을 찾을 수 없습니다."
  exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] ${CONFIG_PATH} 파일을 찾을 수 없습니다."
  exit 1
fi

# 실행 권한 부여
echo "[STEP] 실행 권한 부여: ${PIPELINE_SCRIPT}"
chmod +x "${PIPELINE_SCRIPT}"

# 파이프라인 실행 + 로그 저장
echo "[STEP] 파이프라인 실행 (로그 저장 시작)"
set +e
bash "${PIPELINE_SCRIPT}" --config "${CONFIG_PATH}" --query "${QUERY_TEXT}" 2>&1 | tee "${LOG_FILE}"
EXIT_CODE=${PIPESTATUS[0]}
set -e

if [[ ${EXIT_CODE} -ne 0 ]]; then
  echo "======================================"
  echo "[FAIL] Pipeline exited with code: ${EXIT_CODE}"
  echo "Log: ${LOG_FILE}"
  echo "======================================"
  exit ${EXIT_CODE}
fi

echo "======================================"
echo "[OK] Pipeline finished successfully"
echo "Log: ${LOG_FILE}"
echo "======================================"
