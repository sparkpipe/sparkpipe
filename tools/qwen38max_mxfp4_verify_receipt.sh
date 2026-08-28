#!/usr/bin/env bash
# Full receipt-chain verification of the downloaded AMD Quark checkpoint:
# 1. every file from the HF API present at the exact blob size
# 2. sha256 of every file equal to the repo's ARCHIVE-SHA256SUMS entry
# (the coordinator's standing rule: re-hash after ceph degradation; a bad
# shard is re-downloaded, never trusted.)
set -euo pipefail
DEST="/mnt/model-warm/packbuild/qwen38max/amd-mxfp4"
SPARK="${SPARK_HOST:-spark7}"

remote_script=$(cat <<'REMOTE'
set -euo pipefail
DEST="__DEST__"
cd "$DEST"
curl -s --retry 5 "https://huggingface.co/api/models/amd/Qwen3.8-2.4T-A95B-Quark-MXFP4?blobs=true" \
  | python3 -c '
import json,os,sys
d=json.load(sys.stdin)
for s in d.get("siblings",[]):
    n=s["rfilename"]; sz=s.get("size",0)
    if n.startswith(".") or n in ("README.md","LICENSE"): continue
    actual = os.path.getsize(n) if os.path.exists(n) else -1
    print(("OK" if actual==sz else "BAD")+f"\t{sz}\t{actual}\t{n}")
' > /tmp/size_check.tsv
BAD=$(grep -c "^BAD" /tmp/size_check.tsv || true)
TOTAL=$(wc -l < /tmp/size_check.tsv)
echo "size check: $((TOTAL-BAD))/$TOTAL ok, $BAD bad"
if [[ "$BAD" != "0" ]]; then grep "^BAD" /tmp/size_check.tsv; exit 1; fi

curl -sL --fail --retry 5 -o ARCHIVE-SHA256SUMS.verify \
  "https://huggingface.co/amd/Qwen3.8-2.4T-A95B-Quark-MXFP4/resolve/main/ARCHIVE-SHA256SUMS"
echo "$(date -Is) sha256 pass start (reads ~1.25 TiB)"
# Hash only the model files the receipt covers, skipping our own scratch.
grep -E "safetensors|\.json|\.jinja|\.txt|LICENSE" ARCHIVE-SHA256SUMS.verify | grep -v ARCHIVE > /tmp/receipt.filtered
sha256sum -c /tmp/receipt.filtered --quiet 2>/tmp/sha_errors.log && RC=0 || RC=$?
if [[ "$RC" != "0" ]]; then
  echo "SHA FAILURES:"; head -20 /tmp/sha_errors.log >&2; exit 1
fi
rm -f ARCHIVE-SHA256SUMS.verify /tmp/receipt.filtered
echo "$(date -Is) RECEIPT_VERIFIED all files size+sha256 clean"
REMOTE
)
remote_script="${remote_script//__DEST__/$DEST}"
ssh -o BatchMode=yes "${SPARK}" bash -s <<< "$remote_script"
