#!/usr/bin/env bash
set -euo pipefail

# Download PartSLIP data directly from raw HTTPS URLs.
# No huggingface-cli / hf client is used.

OUT_DIR="${1:-/root/data/PartSLIP}"
MODE="${2:-eval}"   # eval | data-all

BASE_URL="https://huggingface.co/datasets/minghua/PartSLIP/resolve/main"

download_one() {
  local rel="$1"
  local url="${BASE_URL}/${rel}?download=true"
  local dst="${OUT_DIR}/${rel}"
  mkdir -p "$(dirname "${dst}")"

  if command -v aria2c >/dev/null 2>&1; then
    aria2c \
      -c \
      -x 16 \
      -s 16 \
      -k 1M \
      --max-tries=0 \
      --retry-wait=8 \
      --timeout=60 \
      --file-allocation=none \
      -o "$(basename "${dst}")" \
      -d "$(dirname "${dst}")" \
      "${url}"
    return
  fi

  if command -v wget >/dev/null 2>&1; then
    wget -c --tries=0 --waitretry=8 -O "${dst}" "${url}"
    return
  fi

  curl -L --retry 100 --retry-delay 8 -C - -o "${dst}" "${url}"
}

mkdir -p "${OUT_DIR}"

COMMON_FILES=(
  "PartNetE_meta.json"
)

EVAL_FILES=(
  "split/test.txt"
  "data/test.zip"
)

DATA_ALL_FILES=(
  "split/train.txt"
  "split/test.txt"
  "split/few_shot.txt"
  "data/test.zip"
  "data/few_shot.zip"
)

FILES=("${COMMON_FILES[@]}")

case "${MODE}" in
  eval)
    FILES+=("${EVAL_FILES[@]}")
    ;;
  data-all)
    FILES+=("${DATA_ALL_FILES[@]}")
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Usage: $0 [OUT_DIR] [eval|data-all]" >&2
    exit 2
    ;;
esac

echo "Downloading PartSLIP to: ${OUT_DIR}"
echo "Mode: ${MODE}"

for rel in "${FILES[@]}"; do
  echo
  echo "==> ${rel}"
  download_one "${rel}"
done

echo
echo "Done."
