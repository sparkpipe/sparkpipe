#!/bin/bash
# Rename legacy stagepack dirs to the canonical naming scheme (STAGEPACK_NAMING.md).
# Usage: rename_stagepacks.sh <nodename>
# Idempotent — skips dirs already at the canonical path.
set -u
NODE="$1"
HOME_DIR="/home/$NODE"
SD="$HOME_DIR/sparkdata"

# legacy_dir|canonical_dir
MAPPINGS=(
  "glm5_next.tp16|glm53flash.bf16.tp16"
  "glm5_next.tp4pp4|glm53flash.fp8.tp4pp4"
  "glm5_next.tp8.fp8|glm53flash.fp8.tp8"
  "glm5_next.bf16.tp16|glm53flash.bf16.tp16"
  "qwenflash.tp8|qwen3flash.bf16.tp8"
  "qwenflash.tp8.fp8|qwen3flash.fp8.tp8"
  "qwenflash.tp8.nvfp4|qwen3flash.nvfp4.tp8"
  "qwenflash.tp4pp4|qwen3flash.bf16.tp4pp4"
  "qwenflash.tp4pp4.fp8|qwen3flash.fp8.tp4pp4"
  "qwen4_flash.tp4|qwen3flash.bf16.tp4"
  "dsv4flash.tp16|dsv4flash.fp8.tp16"
  "dsv4flash.tp4pp4|dsv4flash.fp8.tp4pp4"
  "dsv4_pro.tp16|dsv4pro.fp8.tp16"
  "dsv4_pro.tp4pp4|dsv4pro.fp8.tp4pp4"
  "qwen27b.tp4pp4|qwen27b.bf16.tp4pp4"
  "qwen38_27b.tp4pp4|qwen27b.nvfp4a16.tp4pp4"
  "qwen38-27b.nvfp4a16.tp4|qwen27b.nvfp4a16.tp4"
)

for mapping in "${MAPPINGS[@]}"; do
  legacy="${mapping%%|*}"
  canon="${mapping##*|}"
  old_dir="$SD/$legacy"
  new_dir="$SD/$canon"

  if [ ! -d "$old_dir" ]; then
    [ -d "$new_dir" ] && echo "SKIP $legacy (already canonical)"
    continue
  fi

  # clear immutable bits in the old tree
  find "$old_dir" -exec sudo chattr -i {} \; 2>/dev/null

  # rename the directory
  if mv "$old_dir" "$new_dir" 2>/dev/null; then
    echo "RENAMED $legacy -> $canon"
  else
    echo "FAIL mv $legacy"
    continue
  fi

  # rename inner pack files to .sp extension and normalise rank padding
  packs_dir="$new_dir/packs"
  if [ -d "$packs_dir" ]; then
    for f in "$packs_dir"/*; do
      [ -f "$f" ] || continue
      base=$(basename "$f")
      new_base="$base"
      for old_ext in .g5nsp .glm52sp .qwen36sp .safetensors .pack; do
        new_base="${new_base%$old_ext}.sp"
      done
      # normalise rank padding: rank00/rank-00 -> rank<h>
      new_base=$(echo "$new_base" | sed 's/rank00/rank0/g; s/rank-00/rank0/g')
      new_path="$packs_dir/$new_base"
      if [ "$f" != "$new_path" ]; then
        mv "$f" "$new_path"
      fi
    done
  fi
done
echo "RENAME-DONE for $NODE"
