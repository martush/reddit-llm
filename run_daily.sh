#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/home/martina/Desktop/Git/reddit-llm"
PYTHON="/home/martina/anaconda3/envs/reddit-llm/bin/python"

SCRAPER="${BASE_DIR}/scripts/reddit_scraper.py"
POSTPROCESS="${BASE_DIR}/scripts/reddit_postprocess.py"
EMBEDDING_QUEUE="${BASE_DIR}/scripts/build_embedding_queue.py"
EMBEDDING="${BASE_DIR}/scripts/embed_to_chroma.py"


LOG_DIR="${BASE_DIR}/logs"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/daily_$(date -u +%Y-%m-%d).log"

{
  echo "=================================================="
  echo "$(date -u)  Starting daily pipeline"
  echo "BASE_DIR=${BASE_DIR}"
  echo "PYTHON=${PYTHON}"
  echo "=================================================="

  echo "$(date -u)  [1/4] Scraping Reddit"
  "${PYTHON}" "${SCRAPER}"

  echo "$(date -u)  [2/4] Postprocessing"
  "${PYTHON}" "${POSTPROCESS}" --top

  echo "$(date -u)  [3/4] Build embedding queue"
  "${PYTHON}" "${EMBEDDING_QUEUE}"

  echo "$(date -u)  [4/4] Embed to Chroma"
  "${PYTHON}" "${EMBEDDING}"

  echo "$(date -u)  Pipeline finished OK"
} >> "${LOG_FILE}" 2>&1
