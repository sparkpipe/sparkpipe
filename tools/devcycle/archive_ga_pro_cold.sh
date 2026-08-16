#!/usr/bin/env bash
# Archive the DSV4 Pro GA checkpoint to spark0's RAID6 cold storage.
# Runs ON spark0. Two phases:
#   phase 1 (--initial): copy whatever has finished downloading so far.
#   phase 2 (no args): wait for the spark3 download to complete, verify
#       every shard against the HF tree sizes, final rsync, full sha256
#       manifest + receipts (ds4-model-cold-archive-v1 convention).
set -u

SOURCE_HOST="spark3"
SOURCE_DIR="/home/spark3/extnvme/models/hf/deepseek-ai/DeepSeek-V4-Pro-0813"
DEST="/mnt/cold-raid6/models/deepseek-v4-pro-0813-ga"
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

mkdir -p "$DEST"

if [[ "${1:-}" == "--initial" ]]; then
    echo "archive_ga_pro_cold: initial rsync (partial snapshot) at $STARTED_AT"
    rsync -a --info=progress2 --exclude '*.part'         "$SOURCE_HOST:$SOURCE_DIR/" "$DEST/" 2>&1 | tail -3
    echo "archive_ga_pro_cold: initial pass done; final pass waits for the download"
    exit 0
fi

echo "archive_ga_pro_cold: waiting for the GA download completion pass on $SOURCE_HOST"
while true; do
    if ssh -o BatchMode=yes "$SOURCE_HOST"         "grep -q 'FINISH-GA-DOWNLOAD' /tmp/finish-ga-download.log 2>/dev/null"; then
        break
    fi
    sleep 300
done
echo "archive_ga_pro_cold: waiting for the stage-1 rsync to settle"
while pgrep -f 'rsync -a --info=progress2' >/dev/null 2>&1; do sleep 60; done
echo "archive_ga_pro_cold: download complete; verifying shard sizes against the HF tree"

ssh -o BatchMode=yes "$SOURCE_HOST" 'python3 - <<"PYEOF"
import json, os
base = "/home/spark3/extnvme/models/hf/deepseek-ai/DeepSeek-V4-Pro-0813"
tree = json.load(open("/tmp/ga-tree.json"))
expected = {f["path"]: f["size"] for f in tree if f["type"] == "file"}
bad = []
for path, size in expected.items():
    full = os.path.join(base, path)
    if not os.path.exists(full) or os.path.getsize(full) != size:
        bad.append((path, os.path.getsize(full) if os.path.exists(full) else -1, size))
if bad:
    print("SIZE-MISMATCH", len(bad))
    for path, got, want in bad[:10]:
        print(path, got, want)
    raise SystemExit(1)
print("SIZES-OK", len(expected))
PYEOF' || { echo "archive_ga_pro_cold: source size verification FAILED"; exit 1; }

echo "archive_ga_pro_cold: final rsync with checksums"
rsync -a --checksum --info=progress2     "$SOURCE_HOST:$SOURCE_DIR/" "$DEST/" || exit 1

echo "archive_ga_pro_cold: full sha256 pass over the archive"
( cd "$DEST" && find . -type f -print0 | sort -z |     xargs -0 sha256sum > ARCHIVE-SHA256SUMS ) || exit 1

total_bytes="$(awk '{print $1}' "$DEST/ARCHIVE-SHA256SUMS" | wc -l)"
total_files="$(wc -l < "$DEST/ARCHIVE-SHA256SUMS")"


