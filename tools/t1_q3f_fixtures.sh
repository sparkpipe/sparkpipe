#!/bin/sh
# Q3F-T1 fixture window: runs INSIDE the ceph lease. Tokenize the canonical
# prompts, generate fixtures twice (determinism proof), negative control,
# manifest verify, sanity-decode the continuations, release.
set -eu
WARM=/mnt/model-warm/qwen3.8-flash-next-fp8
TREE=${TREE:-/tmp/t1q3f_stage/tree}
OUT=$TREE/qualification/t1_reference/qwen4_flash
PROMPTS=$OUT/prompts_qwen4flash.json

cd "$TREE"
python3 tools/t1_q3f_tokenize.py --tokenizer "$WARM/tokenizer.json" --prompts "$TREE/tools/t1_q3f_prompts_source.json" --output "$PROMPTS"
for run in 1 2; do
	PYTHONHASHSEED=0 python3 tools/t1_reference_decoder.py --family qwen4_flash \
		--checkpoint "$WARM" \
		--header model-families/qwen4_flash/include/sparkpipe/llm_defines.h \
		--prompts "$PROMPTS" --output /tmp/t1q3f_fixtures_run$run > /tmp/t1q3f_fixture_run$run.log 2>&1
	tail -1 /tmp/t1q3f_fixture_run$run.log
done
python3 - <<'EOF'
import hashlib, json, sys
def digests(run):
    out = {}
    for name in ("capital_of_france.t1r", "count_up.t1r"):
        path = f"/tmp/t1q3f_fixtures_run{run}/qwen4_flash/{name}"
        out[name] = hashlib.sha256(open(path, "rb").read()).hexdigest()
    return out
a, b = digests(1), digests(2)
print(json.dumps({"run1": a, "run2": b, "deterministic": a == b}, indent=1))
sys.exit(0 if a == b else 1)
EOF
cp /tmp/t1q3f_fixtures_run1/qwen4_flash/*.t1r "$OUT/"
cp /tmp/t1q3f_fixtures_run1/qwen4_flash/MANIFEST.json "$OUT/"
python3 tools/t1_reference_compare.py corrupt-fixture --source "$OUT/capital_of_france.t1r" \
	--target /tmp/t1q3f_negative.t1r --array pos0000_layer0000_streams --offset 9
set +e
python3 tools/t1_reference_compare.py compare --reference "$OUT/capital_of_france.t1r" \
	--candidate /tmp/t1q3f_negative.t1r > /tmp/t1q3f_negative.log 2>&1
neg_rc=$?
set -e
[ "$neg_rc" != 0 ] && echo "NEGATIVE CONTROL CONVICTED (expected)" || { echo "NEGATIVE CONTROL FAILED"; exit 1; }
python3 tools/t1_reference_compare.py verify-manifest --manifest "$OUT/MANIFEST.json" 2>/dev/null || \
	python3 - <<'EOF'
import hashlib, json, sys
root = "/tmp/t1q3f_stage/tree/qualification/t1_reference/qwen4_flash"
manifest = json.load(open(f"{root}/MANIFEST.json"))
bad = 0
for name, want in manifest["fixtures"].items():
    got = hashlib.sha256(open(f"{root}/{name}", "rb").read()).hexdigest()
    if got != want["sha256"]:
        bad += 1
        print(f"{name}: sha mismatch")
print(json.dumps({"verified": len(manifest["fixtures"]), "bad": bad}))
sys.exit(1 if bad else 0)
EOF
python3 - <<'EOF'
import json, sys
sys.path.insert(0, "/tmp/t1q3f_stage/tree/tools")
import t1_q3f_tokenize as tok
tokenizer = json.load(open("/mnt/model-warm/qwen3.8-flash-next-fp8/tokenizer.json"))
model = {"vocab": tokenizer["model"]["vocab"], "ranks": {}}
for rank, pair in enumerate(tokenizer["model"]["merges"]):
    parts = pair.split(" ") if isinstance(pair, str) else list(pair)
    model["ranks"][(parts[0], parts[1])] = rank
root = "/tmp/t1q3f_stage/tree/qualification/t1_reference/qwen4_flash"
for name in ("capital_of_france", "count_up"):
    meta, arrays = None, None
    from t1_reference_common import read_fixture
    meta, arrays = read_fixture(f"{root}/{name}.t1r")
    gen = arrays["generated_token_ids"].tolist()
    print(json.dumps({"prompt": name, "generated_ids": gen,
                      "generated_text": tok.decode_ids(gen, model)}))
EOF
echo "FIXTURE WINDOW COMPLETE"
