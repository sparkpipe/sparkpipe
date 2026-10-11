#!/usr/bin/env bash
set -u
. "$(dirname "$0")/rollover_lib.sh"
[ "$(controller)" = "idle qwen3.8-max None" ] || { echo "PRECHECK-FAILED: $(controller)"; exit 1; }
nodes_up || exit 9
models=$(hub "python3 -c \"import json; print(' '.join(m['id'] for m in json.load(open('/home/spec/sparkpipe-swap/model_swap.json'))['models']))\"")
for m in $models qwen3.8-max; do
  [ "$m" = qwen3.8-max ] && [ "${seen_max:-}" = "" ] && { seen_max=1; continue; }
  if activate "$m"; then
    reply=$(hub "curl -s -m 600 localhost:4000/v1/chat/completions -H 'content-type: application/json' -d '{\"model\":\"$m\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? Answer in one word.\"}],\"max_tokens\":512,\"temperature\":0,\"reasoning_effort\":\"low\"}'" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin); m = d['choices'][0]['message']
    print(repr((m.get('content') or '')[:60]), 'reasoning', len(m.get('reasoning_content') or ''), d['choices'][0].get('finish_reason'))
except Exception as e:
    print('REQUEST-FAILED', e)")
    echo "SPOT $m: $reply; $(hub "grep swap_memory ~/sparkpipe-swap/swap.log | tail -1 | cut -c1-160")"
  else
    echo "SPOT $m: ACTIVATE-FAILED"
  fi
done
[ "$(controller)" = "idle qwen3.8-max None" ] || restore_max
nodes_up || exit 9
stamp "spotcheck done ($(controller))"
