#!/usr/bin/env bash
# Download the AMD Quark MXFP4 checkpoint of Qwen3.8-2.4T-A95B to warm storage.
#
# The checkpoint quantizes ONLY the routed experts to MXFP4-E2M1 with E8M0
# group-32 scales (config.json quantization_config.exclude keeps every GDN,
# attention, router, shared-expert and MTP tensor BF16) - exactly the ladder
# model_contracts/qwen38_authoritative.json pins. Consumed verbatim by
# tools/qwen38_stagepack.py --source-format quark-mxfp4.
#
# Parameterized per the lane rules: --spark selects the download node, nothing
# is hardcoded. Size: 1.248 TiB across 213 shards + aux files.
set -euo pipefail

SPARK="${SPARK_HOST:-spark7}"
WORKERS="${DOWNLOAD_WORKERS:-10}"
REPO="amd/Qwen3.8-2.4T-A95B-Quark-MXFP4"
DEST="/mnt/model-warm/packbuild/qwen38max/amd-mxfp4"
# Split the file list across concurrent download nodes: --shard-mod N/M
# takes files whose list index satisfies index % M == N. Default takes all.
SHARD_N=0
SHARD_M=1

usage() { echo "usage: $0 [--spark N] [--workers N] [--dest PATH] [--shard-mod N/M]" >&2; exit 2; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --spark) SPARK="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --dest) DEST="$2"; shift 2 ;;
    --shard-mod) SHARD_N="${2%%/*}"; SHARD_M="${2##*/}"; shift 2 ;;
    *) usage ;;
  esac
done

remote_script=$(cat <<'REMOTE'
set -euo pipefail
REPO="__REPO__"
DEST="__DEST__"
WORKERS=__WORKERS__
SHARD_N=__SHARD_N__
SHARD_M=__SHARD_M__
mkdir -p "$DEST"
cd "$DEST"

# File list with exact blob sizes from the HF API (the size is the integrity
# check - every shard must land whole). The shard-mod filter splits the list
# across concurrent download nodes.
curl -s --retry 5 "https://huggingface.co/api/models/${REPO}?blobs=true" \
  | SHARD_N="${SHARD_N}" SHARD_M="${SHARD_M}" python3 -c '
import json,os,sys
d=json.load(sys.stdin)
n=int(os.environ["SHARD_N"]); m=int(os.environ["SHARD_M"])
files=[s for s in d.get("siblings",[]) if not s["rfilename"].startswith(".")
       and s["rfilename"] not in ("README.md","LICENSE")]
for i,s in enumerate(files):
    if i % m == n:
        print(str(s.get("size",0)) + "\t" + s["rfilename"])' > files.tsv
TOTAL=$(awk -F'\t' '{s+=$1} END{print s+0}' files.tsv)
COUNT=$(wc -l < files.tsv)
echo "$(date -Is) download start repo=${REPO} slice=${SHARD_N}/${SHARD_M} files=${COUNT} bytes=${TOTAL} workers=${WORKERS}"

fetch_one() {
  local size="$1" name="$2"
  local url="https://huggingface.co/${REPO}/resolve/main/${name}"
  # Skip only whole, size-matched files so a re-run resumes.
  if [[ -f "$name" ]] && [[ "$(stat -c %s "$name")" == "$size" ]]; then
    echo "SKIP ${name}"; return 0
  fi
  local rc=0
  for attempt in 1 2 3 4 5; do
    if curl -sL --fail --retry 3 --retry-delay 5 \
         -o "${name}.part" -C - "$url"; then
      if [[ "$(stat -c %s "${name}.part")" == "$size" ]]; then
        mv "${name}.part" "$name"
        echo "OK ${name}"; return 0
      fi
    fi
    rc=$?
    sleep $((attempt * 10))
  done
  echo "FAIL ${name} rc=${rc}" >&2
  return 1
}
export -f fetch_one
export REPO

awk -F'\t' '{print $1" "$2}' files.tsv | xargs -P "$WORKERS" -n 2 bash -c 'fetch_one "$0" "$1"'

GOT=$(awk -F'\t' '{s+=$1} END{print s+0}' <(while read -r sz nm; do
  [[ -f "$nm" ]] && [[ "$(stat -c %s "$nm")" == "$sz" ]] && echo -e "$sz\t$nm"
done < files.tsv))
echo "$(date -Is) slice complete verified_bytes=${GOT} expected=${TOTAL}"
[[ "$GOT" == "$TOTAL" ]] || { echo "MISSING BYTES: $((TOTAL-GOT))" >&2; exit 1; }
REMOTE
)
remote_script="${remote_script//__REPO__/$REPO}"
remote_script="${remote_script//__DEST__/$DEST}"
remote_script="${remote_script//__WORKERS__/$WORKERS}"
remote_script="${remote_script//__SHARD_N__/$SHARD_N}"
remote_script="${remote_script//__SHARD_M__/$SHARD_M}"

echo "launching download on ${SPARK} -> ${DEST} slice ${SHARD_N}/${SHARD_M} with ${WORKERS} workers"
ssh -o BatchMode=yes "${SPARK}" bash -s <<< "$remote_script"
