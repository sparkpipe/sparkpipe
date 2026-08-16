#!/usr/bin/env bash
# Complete the GA download: fetch any file from the HF tree that is missing
# or has the wrong size on spark3 (resumable, skips correct files).
set -u
REPO="https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813/resolve/main"
DEST="/home/spark3/extnvme/models/hf/deepseek-ai/DeepSeek-V4-Pro-0813"
mkdir -p "$DEST"
cd "$DEST" || exit 1

curl -sL --max-time 300 "https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4-Pro-0813/tree/main?recursive=true&expand=true" -o /tmp/ga-tree.json.new || true
if python3 - <<'PYEOF'
import json
try:
    t = json.load(open("/tmp/ga-tree.json.new"))
    big = [e for e in t if e.get("type") == "file" and "safetensors" in e["path"] and e.get("size", 0) > 1_000_000_000]
    ok = len(big) >= 50
except Exception:
    ok = False
print("tree-ok" if ok else "tree-bad")
raise SystemExit(0 if ok else 1)
PYEOF
then
    mv /tmp/ga-tree.json.new /tmp/ga-tree.json
    echo "tree refresh ok"
else
    echo "tree refresh FAILED; keeping previous /tmp/ga-tree.json"
fi
python3 /tmp/ga_missing_files.py

fail=0
while IFS= read -r path; do
    [ -n "$path" ] || continue
    mkdir -p "$(dirname "$path")"
    if curl -sL --max-time 7200 "$REPO/$path" -o "$path.part"; then
        mv "$path.part" "$path"
    else
        echo "FAIL $path"; fail=1; rm -f "$path.part"
    fi
    echo "fetched $path ($(wc -c < "$path" 2>/dev/null || echo 0) bytes)"
done < /tmp/ga-missing.txt
echo "FINISH-GA-DOWNLOAD fail=$fail"
