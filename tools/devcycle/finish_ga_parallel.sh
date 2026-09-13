#!/usr/bin/env bash
# Finish the GA download with 4 parallel shard fetches.
# Runs ON spark3. Regenerates the missing list, fetches remaining shards
# concurrently, verifies nothing is left, then writes the FINISH-GA-DOWNLOAD
# marker that the spark0 archive watcher waits on.
set -u
REPO="https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813/resolve/main"
export REPO
DEST="/home/spark3/extnvme/models/hf/deepseek-ai/DeepSeek-V4-Pro-0813"
cd "$DEST" || exit 1

python3 /tmp/ga_missing_files.py
grep -E '^model-' /tmp/ga-missing.txt > /tmp/ga-shards.txt || true
echo "shards to fetch: $(wc -l < /tmp/ga-shards.txt)"

cat /tmp/ga-shards.txt | xargs -P 4 -I{} bash -c '
    curl -sL --max-time 10800 "$REPO/{}" -o "{}.part" \
        && mv "{}.part" "{}" && echo "OK {}" \
        || { echo "FAIL {}"; rm -f "{}.part"; }
'

python3 /tmp/ga_missing_files.py
left=$(grep -c . /tmp/ga-missing.txt || true)
if [ "$left" -eq 0 ]; then
    echo "FINISH-GA-DOWNLOAD fail=0 (parallel, all files complete)"
else
    echo "FINISH-GA-DOWNLOAD fail=1 remaining=$left"
    echo "still missing:"
    cat /tmp/ga-missing.txt
fi
