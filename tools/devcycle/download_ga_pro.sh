#!/usr/bin/env bash
# Download the DSV4 Pro GA checkpoint (deepseek-ai/DeepSeek-V4-Pro-0813)
# into a fresh directory on spark3. Resumes: skips files that already have
# the expected size.
set -u
REPO="https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813/resolve/main"
DEST="/home/spark3/extnvme/models/hf/deepseek-ai/DeepSeek-V4-Pro-0813"
mkdir -p "$DEST"
cd "$DEST" || exit 1

for name in config.json generation_config.json tokenizer.json tokenizer_config.json model.safetensors.index.json; do
    curl -sL --max-time 300 "$REPO/$name" -o "$name"
    echo "fetched $name ($(wc -c < "$name") bytes)"
done

fail=0
for i in $(seq 1 66); do
    file="model-$(printf '%05d' "$i")-of-00066.safetensors"
    if [[ -s "$file" ]]; then
        echo "skip $file (present)"
        continue
    fi
    rm -f "$file"
    curl -sL --max-time 7200 "$REPO/$file" -o "$file" || { echo "FAIL $file"; fail=1; }
done
echo "GA-DOWNLOAD-DONE fail=$fail"
