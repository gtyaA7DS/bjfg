#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-/root/data/ScanObjectNN}"
KEEP_ARCHIVE="${KEEP_ARCHIVE:-0}"
ARIA2_CONN="${ARIA2_CONN:-16}"
ARIA2_SPLIT="${ARIA2_SPLIT:-16}"
ARIA2_MIN_SPLIT_SIZE="${ARIA2_MIN_SPLIT_SIZE:-1M}"
ARIA2_CONN_MAX=16

URL="https://hkust-vgd.ust.hk/scanobjectnn/raw/object_dataset_complete_with_parts%20.zip"
ARCHIVE_NAME="object_dataset_complete_with_parts.zip"
ARCHIVE_PATH="${OUT_DIR}/${ARCHIVE_NAME}"
DATA_DIR="${OUT_DIR}/object_dataset_complete_with_parts_"

download_one() {
  local url="$1"
  local dst="$2"

  if command -v aria2c >/dev/null 2>&1; then
    aria2c \
      -c \
      -x "${ARIA2_CONN}" \
      -s "${ARIA2_SPLIT}" \
      -k "${ARIA2_MIN_SPLIT_SIZE}" \
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

validate_positive_int() {
  local name="$1"
  local value="$2"
  if ! [[ "${value}" =~ ^[0-9]+$ ]] || (( value < 1 )); then
    echo "Invalid ${name}: ${value}. It must be a positive integer." >&2
    exit 2
  fi
}

mkdir -p "${OUT_DIR}"

validate_positive_int "ARIA2_CONN" "${ARIA2_CONN}"
validate_positive_int "ARIA2_SPLIT" "${ARIA2_SPLIT}"

if (( ARIA2_CONN > ARIA2_CONN_MAX )); then
  echo "ARIA2_CONN=${ARIA2_CONN} is too high for aria2c; capping to ${ARIA2_CONN_MAX}." >&2
  ARIA2_CONN="${ARIA2_CONN_MAX}"
fi

echo "Downloading ScanObjectNN part-seg archive to: ${OUT_DIR}"
echo "Target data dir: ${DATA_DIR}"
echo "Only files used by cops/data/ScanObjectNN.py will be kept."
echo
echo "Archive URL: ${URL}"
echo "Archive size on server: 10347548625 bytes (~9.64 GiB)"
echo "Kept extracted subset: ~21.20 GiB"
if command -v aria2c >/dev/null 2>&1; then
  echo "aria2c parallelism: -x ${ARIA2_CONN} -s ${ARIA2_SPLIT} -k ${ARIA2_MIN_SPLIT_SIZE}"
fi
echo

download_one "${URL}" "${ARCHIVE_PATH}"

python - "${ARCHIVE_PATH}" "${OUT_DIR}" <<'PY'
import sys
import shutil
import zipfile
from pathlib import Path

archive_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
target_root = out_dir / "object_dataset_complete_with_parts_"
target_root.mkdir(parents=True, exist_ok=True)

kept = 0
skipped = 0
reused = 0
bytes_written = 0

def normalize_parts(name: str):
    parts = [part.rstrip() for part in Path(name).parts if part not in ("", ".")]
    if not parts:
        return []
    parts[0] = "object_dataset_complete_with_parts_"
    return parts

def should_keep(rel_path: Path) -> bool:
    rel = rel_path.as_posix()
    if rel == "split_new.txt":
        return True
    if rel.endswith("_indices.bin"):
        return False
    if rel.endswith("_part.xml"):
        return False
    if rel.endswith(".bin"):
        return True
    return False

with zipfile.ZipFile(archive_path) as zf:
    members = zf.infolist()
    total_members = len(members)
    for idx, info in enumerate(members, start=1):
        if info.is_dir():
            continue
        parts = normalize_parts(info.filename)
        if len(parts) < 2:
            continue
        rel_path = Path(*parts[1:])
        if not should_keep(rel_path):
            skipped += 1
            continue

        dst = target_root / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists() and dst.stat().st_size == info.file_size:
            reused += 1
            continue

        with zf.open(info, "r") as src, dst.open("wb") as out:
            shutil.copyfileobj(src, out, length=1024 * 1024)
        kept += 1
        bytes_written += info.file_size

        if kept % 200 == 0:
            print(f"[extract] kept {kept} files | scanned {idx}/{total_members}", flush=True)

print(f"[extract] done | newly_written={kept} | reused={reused} | skipped={skipped}", flush=True)
print(f"[extract] bytes_written={bytes_written}", flush=True)
print(f"[extract] dataset_root={target_root}", flush=True)
PY

if [[ "${KEEP_ARCHIVE}" != "1" ]]; then
  rm -f "${ARCHIVE_PATH}"
  echo
  echo "Removed archive: ${ARCHIVE_PATH}"
else
  echo
  echo "Kept archive: ${ARCHIVE_PATH}"
fi

echo
echo "Final dataset path expected by the loader:"
echo "  ${DATA_DIR}"
echo
echo "Disk usage:"
du -sh "${DATA_DIR}" || true
if [[ -f "${ARCHIVE_PATH}" ]]; then
  du -sh "${ARCHIVE_PATH}" || true
fi
