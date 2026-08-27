#!/usr/bin/env bash
# Build the 16 TP4xPP4 rank packs from the AMD Quark MXFP4 checkpoint, verify
# each against the checkpoint, and deploy one pack per spark node.
#
# Stage p covers layers [p*23, (p+1)*23); rank r holds the tp-4 shard. Node
# sparkN receives world rank N (pp = N/4, tp = N%4). Everything streams from
# warm storage; each pack lands in the node-local NVMe deploy directory.
#
# Parameterized per the lane rules: --spark selects the build hub, nothing is
# hardcoded. Resumable: whole, size-verified checkpoint shards are skipped.
set -euo pipefail

SPARK="${SPARK_HOST:-spark0}"
CHECKPOINT="/mnt/model-warm/packbuild/qwen38max/amd-mxfp4"
REPO_REMOTE="\$HOME/sparkpipe-lane"
DEPLOY_DIR="sparkdata/qwen38max.tp4/packs"
WORKERS="${BUILD_WORKERS:-4}"

usage() { echo "usage: $0 [--spark N] [--checkpoint PATH] [--verify-only] [--deploy-from DIR]" >&2; exit 2; }
VERIFY_ONLY=0
DEPLOY_FROM=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --spark) SPARK="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --verify-only) VERIFY_ONLY=1; shift ;;
    --deploy-from) DEPLOY_FROM="$2"; shift 2 ;;
    *) usage ;;
  esac
done

remote_script=$(cat <<'REMOTE'
set -euo pipefail
CHECKPOINT="__CHECKPOINT__"
REPO="__REPO__"
DEPLOY_DIR="__DEPLOY_DIR__"
VERIFY_ONLY=__VERIFY_ONLY
DEPLOY_FROM="__DEPLOY_FROM__"
LAYERS_PER_STAGE=23
TP_DEGREE=4
cd "$REPO"

build_one() {
  local pp="$1" rank="$2"
  local first=$(( pp * LAYERS_PER_STAGE ))
  local name="qwen38max.pp${pp}.tp4-rank${rank}.qwen38sp"
  local out="$HOME/${DEPLOY_DIR}/${name}"
  mkdir -p "$HOME/${DEPLOY_DIR}"
  if [[ "$VERIFY_ONLY" == 1 || -n "$DEPLOY_FROM" ]]; then
    out="${DEPLOY_FROM:-$HOME/${DEPLOY_DIR}}/${name}"
  fi
  if [[ "$VERIFY_ONLY" != 1 ]]; then
    echo "$(date -Is) BUILD pp=${pp} rank=${rank} -> ${out}"
    python3 tools/qwen38_stagepack.py \
      --checkpoint "$CHECKPOINT" --source-format quark-mxfp4 \
      --first-layer "$first" --layer-count "$LAYERS_PER_STAGE" \
      --tp-degree "$TP_DEGREE" --tp-rank "$rank" --output "$out"
  fi
  python3 tools/qwen38_pack_verify.py --pack "$out" --checkpoint "$CHECKPOINT" \
    --source-format quark-mxfp4 --tp-degree "$TP_DEGREE" --tp-rank "$rank" --sample 12
}

for pp in 0 1 2 3; do
  for rank in 0 1 2 3; do
    build_one "$pp" "$rank"
  done
done
echo "$(date -Is) ALL_RANKS_DONE"
REMOTE
)
remote_script="${remote_script//__CHECKPOINT__/$CHECKPOINT}"
remote_script="${remote_script//__REPO__/$REPO_REMOTE}"
remote_script="${remote_script//__DEPLOY_DIR__/$DEPLOY_DIR}"
remote_script="${remote_script//__VERIFY_ONLY__/$VERIFY_ONLY}"
remote_script="${remote_script//__DEPLOY_FROM__/$DEPLOY_FROM}"

echo "building 16 rank packs on ${SPARK} from ${CHECKPOINT}"
ssh -o BatchMode=yes "${SPARK}" bash -s <<< "$remote_script"
