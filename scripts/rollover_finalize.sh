#!/usr/bin/env bash
set -u
. "$(dirname "$0")/rollover_lib.sh"
[ "$(controller)" = "idle qwen3.8-max None" ] || { echo "PRECHECK-FAILED: $(controller)"; exit 1; }
UTC=$(date -u +%Y%m%dT%H%M%SZ)
hub "cp -p ~/sparkpipe-swap/model_swap.json ~/sparkpipe-swap/model_swap.json.prev-$UTC && python3 - <<'PY'
import json
path = '/home/spec/sparkpipe-swap/model_swap.json'
swap = json.load(open(path))
measured = {'glm-5.3': 57 << 30, 'glm-5.3-flash': 25 << 30}
for model in swap['models']:
    if model['id'] in measured:
        model['node_memory_bytes'] = measured[model['id']]
open(path, 'w').write(json.dumps(swap, indent=1) + '\n')
print({m['id']: m['node_memory_bytes'] for m in swap['models'] if m['id'] in measured})
PY
systemctl --user restart sparkpipe-swap; sleep 8"
echo "finalize: backup model_swap.json.prev-$UTC; controller $(controller)"
