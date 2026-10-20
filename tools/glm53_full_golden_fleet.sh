#!/usr/bin/env bash
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
HUB=rtx5090
SSH="ssh -o BatchMode=yes -o ConnectTimeout=8"
LABEL="${1:?label}"
PORT="${2:?api port}"
SWAP_MODEL="${3:-}"
WORK='~/glmfull-port'
OUT="$WORK/golden-$LABEL"
TOKENIZER=/mnt/cold-raid6/models/glm-5.3-fp8/tokenizer.json

. "$HERE/../scripts/rollover_lib.sh"

activate() {
  local model="$1" deadline state
  swap "-X POST localhost:8440/activate -d '{\"model\":\"$model\"}'" > /dev/null || return 1
  deadline=$((SECONDS + 900))
  while [ $SECONDS -lt $deadline ]; do
    state="$(swap localhost:8440/status | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['state'], d['active'], d['error'])")"
    case "$state" in
      "idle $model None") echo "activated $model"; return 0 ;;
      idle*) echo "activate $model failed: $state" >&2; return 1 ;;
    esac
    sleep 5
  done
  echo "activate $model timed out" >&2
  return 1
}

run() {
  $SSH $HUB "cd $WORK/repo && python3 tools/glm53_full_golden.py $*"
}

if [ -n "$SWAP_MODEL" ]; then
  activate "$SWAP_MODEL" || { activate qwen3.8-max; exit 1; }
fi
$SSH $HUB "mkdir -p $OUT"
status=0
run capture --prompts ../golden_prompts.json --port "$PORT" --max-tokens 128 --logprobs 4 --label "$LABEL" --out "$OUT/lp.json" || status=1
run batch --prompts ../golden_prompts.json --names capital,arith,code,story --port "$PORT" --max-tokens 64 --label "$LABEL" --out "$OUT/batch.json" || status=1
run compsec --prompts qualification/glm53full/compsec17_prompts_off.json --port "$PORT" --max-tokens 512 --concurrency 2 --tokenizer "$TOKENIZER" --label "$LABEL" --out "$OUT/compsec17_off.json" || status=1
if [ -n "$SWAP_MODEL" ]; then
  activate qwen3.8-max || status=1
fi
exit $status
