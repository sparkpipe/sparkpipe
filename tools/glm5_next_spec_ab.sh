#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: $0 ARM OUT_DIR [BASELINE_RUN_JSON]" >&2
    echo "  env: SPEC_AB_ENDPOINT (default http://127.0.0.1:8433), SPEC_AB_LOG_HOST (default sparkf, empty skips the log)," >&2
    echo "       SPEC_AB_ROOT (default sparkdata/glm53flash.fp8.tp16), SPEC_AB_MAX_TOKENS (default 512)" >&2
    exit 2
fi

arm="$1"
out="$2"
baseline="${3:-}"
endpoint="${SPEC_AB_ENDPOINT:-http://127.0.0.1:8433}"
log_host="${SPEC_AB_LOG_HOST-sparkf}"
root="${SPEC_AB_ROOT:-sparkdata/glm53flash.fp8.tp16}"
max_tokens="${SPEC_AB_MAX_TOKENS:-512}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! "$arm" =~ ^[A-Za-z0-9._+-]+$ ]]; then
    echo "ARM must be a plain label such as off, oracle, mtp or mtp+lookup" >&2
    exit 2
fi
mkdir -p "$out"

python3 "$here/spec_verify_bench.py" run --endpoint "$endpoint" --label "$arm" --max-tokens "$max_tokens" --out "$out/$arm.json"

if [[ -n "$log_host" ]]; then
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$log_host" "cat \$HOME/$root/residentd.log" > "$out/$arm.residentd.log"
    python3 "$here/spec_verify_bench.py" log "$out/$arm.residentd.log" > "$out/$arm.acceptance.json"
    grep -E "GLM verify regime|GLM verify MTP pack|GRAPH-VERIFY-TABLE|VERIFY-RANK-LOCAL|VERIFY-MTP-(UNSUPPORTED|DRAFT-FAILED)|VERIFY-PLAIN-FRAME" "$out/$arm.residentd.log" | tail -n 40 > "$out/$arm.verify-lines.txt" || true
fi

status=0
if [[ -n "$baseline" ]]; then
    python3 "$here/spec_verify_bench.py" compare "$baseline" "$out/$arm.json" > "$out/$arm.compare.json" || status=$?
fi

python3 - "$out" "$arm" "$status" <<'EOF'
import json
import sys
from pathlib import Path

out, arm, status = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
run = json.loads((out / f"{arm}.json").read_text())
summary = {"arm": arm, "decode_tok_s": {}}
for entry in run["results"]:
    total = summary["decode_tok_s"].setdefault(entry["class"], {"tokens": 0, "seconds": 0.0})
    total["tokens"] += entry["decode_tokens"]
    total["seconds"] += entry["decode_s"]
summary["decode_tok_s"] = {name: round(v["tokens"] / v["seconds"], 2) if v["seconds"] else None for name, v in summary["decode_tok_s"].items()}
acceptance = out / f"{arm}.acceptance.json"
if acceptance.exists():
    summary["acceptance"] = json.loads(acceptance.read_text())
compare = out / f"{arm}.compare.json"
if compare.exists():
    report = json.loads(compare.read_text())
    summary["exact_vs_baseline"] = report["exact"]
    summary["speedup_vs_baseline"] = report["speedup"]
(out / f"{arm}.summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
sys.exit(status)
EOF
