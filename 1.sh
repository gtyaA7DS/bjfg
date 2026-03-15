POINTS_SHARD=/root/data/core/core_part-00001.tar.zst
RENDER_SHARD=/root/data/core/core_part-00009.tar.zst
OUT=/root/data/core_subset
N=2000
tmp=$(mktemp -d)
mkdir -p "$OUT"

# 1) 从渲染分片选 item_id
( tar --zstd -tf "$RENDER_SHARD" \
  | grep -E '^labeled/rendered/.+/oriented/masks/merged/mask_labels[.]txt$' \
  | sed -E 's#^labeled/rendered/(.+)/oriented/masks/merged/mask_labels.txt$#\1#' \
  | head -n "$N" > "$tmp/item_ids.txt" ) || true

# 2) 分开生成“渲染文件清单”和“点云文件清单”
: > "$tmp/render_files.txt"
: > "$tmp/point_files.txt"
while IFS= read -r item; do
  uid="${item##*_}"
  echo "labeled/rendered/$item/oriented/masks/merged/mask2points.pt" >> "$tmp/render_files.txt"
  echo "labeled/rendered/$item/oriented/masks/merged/mask_labels.txt" >> "$tmp/render_files.txt"
  echo "labeled/rendered/$item/oriented/patch_dino/patch_features.pt" >> "$tmp/render_files.txt"
  echo "labeled/points/$uid/points.pt" >> "$tmp/point_files.txt"
done < "$tmp/item_ids.txt"

sort -u "$tmp/render_files.txt" -o "$tmp/render_files.txt"
sort -u "$tmp/point_files.txt" -o "$tmp/point_files.txt"

# 3) 分开解压（关键）
tar --zstd -xf "$RENDER_SHARD" -C "$OUT" -T "$tmp/render_files.txt"
tar --zstd -xf "$POINTS_SHARD" -C "$OUT" -T "$tmp/point_files.txt"
