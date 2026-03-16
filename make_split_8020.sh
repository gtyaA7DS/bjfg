#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/root/data/core_subset}"
SPLIT_DIR="$ROOT/labeled/split"
RENDER_DIR="$ROOT/labeled/rendered"
POINTS_DIR="$ROOT/labeled/points"

if [[ ! -d "$RENDER_DIR" ]]; then
  echo "[error] render directory not found: $RENDER_DIR" >&2
  exit 1
fi

mkdir -p "$SPLIT_DIR"

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

all_items="$tmpdir/all_items.txt"
valid_items="$tmpdir/valid_items.txt"
shuffled_items="$tmpdir/shuffled_items.txt"

find "$RENDER_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort > "$all_items"
: > "$valid_items"

while IFS= read -r item; do
  uid="${item##*_}"
  mask2points="$RENDER_DIR/$item/oriented/masks/merged/mask2points.pt"
  mask_labels="$RENDER_DIR/$item/oriented/masks/merged/mask_labels.txt"
  patch_feats="$RENDER_DIR/$item/oriented/patch_dino/patch_features.pt"
  points="$POINTS_DIR/$uid/points.pt"

  if [[ -f "$mask2points" && -f "$mask_labels" && -f "$patch_feats" && -f "$points" ]]; then
    echo "$item" >> "$valid_items"
  fi
done < "$all_items"

total="$(wc -l < "$valid_items")"
if [[ "$total" -eq 0 ]]; then
  echo "[error] no valid items found under $ROOT" >&2
  exit 1
fi

if command -v shuf >/dev/null 2>&1; then
  shuf "$valid_items" > "$shuffled_items"
else
  cp "$valid_items" "$shuffled_items"
fi

train_count=$(( total * 8 / 10 ))
val_count=$(( total - train_count ))

head -n "$train_count" "$shuffled_items" > "$SPLIT_DIR/train.txt"
tail -n "$val_count" "$shuffled_items" > "$SPLIT_DIR/val.txt"

echo "[done] root=$ROOT"
echo "[done] total=$total train=$train_count val=$val_count"
echo "[done] wrote $SPLIT_DIR/train.txt"
echo "[done] wrote $SPLIT_DIR/val.txt"

