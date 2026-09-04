#!/usr/bin/env bash
set -euo pipefail

require_environment_value()
{
    local variable_name="$1"

    if [[ -z "${!variable_name:-}" ]]; then
        printf 'missing required environment variable: %s\n' "$variable_name" >&2
        exit 2
    fi
}

require_environment_value GLM52_PREFILL_TOKEN_IDS_FILE
require_environment_value GLM52_LOCAL_PIPELINE_OUTPUT_DIR
require_environment_value GLM52_LOCAL_PIPELINE_MAX_PREFILL_TOKENS

if [[ "${GLM52_LOCAL_PIPELINE_PREFILL_ONLY:-0}" != "1" ]]; then
    printf '%s\n' \
        'full local CUDA pipeline execution is unavailable in this source package; use GLM52_LOCAL_PIPELINE_PREFILL_ONLY=1 for the host scheduling gate' >&2
    exit 3
fi

mkdir -p "$GLM52_LOCAL_PIPELINE_OUTPUT_DIR"
SCHEDULE_PATH="$GLM52_LOCAL_PIPELINE_OUTPUT_DIR/prefill_schedule.tsv"

python3 - \
    "$GLM52_PREFILL_TOKEN_IDS_FILE" \
    "$GLM52_LOCAL_PIPELINE_MAX_PREFILL_TOKENS" \
    "$SCHEDULE_PATH" <<'PY'
from __future__ import annotations

from pathlib import Path
import sys


def positive_integer(text: str, name: str) -> int:
    try:
        value = int(text, 10)
    except ValueError as error:
        raise SystemExit(f"{name} must be an integer") from error
    if value <= 0:
        raise SystemExit(f"{name} must be positive")
    return value


def load_token_ids(path: Path) -> list[int]:
    if not path.is_file():
        raise SystemExit(f"missing token-id file: {path}")
    tokens: list[int] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        text = line.strip()
        if not text:
            continue
        try:
            token_id = int(text, 10)
        except ValueError as error:
            raise SystemExit(
                f"invalid token id at {path}:{line_number}: {text}"
            ) from error
        if token_id < 0:
            raise SystemExit(
                f"negative token id at {path}:{line_number}: {token_id}"
            )
        tokens.append(token_id)
    if not tokens:
        raise SystemExit(f"token-id file is empty: {path}")
    return tokens


token_path = Path(sys.argv[1])
maximum_prefill_tokens = positive_integer(sys.argv[2], "maximum prefill tokens")
schedule_path = Path(sys.argv[3])
tokens = load_token_ids(token_path)

rows = [
    "step\tkind\ttoken_offset\ttoken_count\tremaining\tcommit_after\tprefill_blocks\tkv_blocks\tflags"
]
token_offset = 0
step_index = 0
while token_offset < len(tokens):
    token_count = min(maximum_prefill_tokens, len(tokens) - token_offset)
    commit_after = token_offset + token_count
    remaining = len(tokens) - commit_after
    prefill_blocks = (token_count + maximum_prefill_tokens - 1) // maximum_prefill_tokens
    kv_blocks = (commit_after + maximum_prefill_tokens - 1) // maximum_prefill_tokens
    rows.append(
        f"{step_index}\tprefill\t{token_offset}\t{token_count}\t{remaining}\t"
        f"{commit_after}\t{prefill_blocks}\t{kv_blocks}\t0"
    )
    token_offset = commit_after
    step_index += 1
rows.append(f"{step_index}\tdecode_ready\t0\t0\t0\t0\t0\t0\t0")
schedule_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

print(f"glm52_local_pipeline_prefill_schedule={schedule_path}")
print(f"glm52_local_pipeline_prefill_steps={step_index}")
print(f"glm52_local_pipeline_prefill_tokens={len(tokens)}")
print("glm52_local_pipeline_prefill_only=1")
PY
