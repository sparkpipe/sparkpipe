# Qwen3.8-27B DFlash2 serving — ops runbook

Everything a developer needs to find the model, build, deploy, launch,
benchmark, and verify the release tagged `qwen38-dflash2-20260821`.
Deep internals (kernel ledgers, experiment history, perf analysis) live in
[`docs/DFLASH2_HANDOFF.md`](DFLASH2_HANDOFF.md); this file is the
from-zero operational path.

## 1. Where everything is

| What | Where |
|---|---|
| Repo / release | `github.com/sparkpipe/sparkpipe`, tag `qwen38-dflash2-20260821` (main `76a92f8`) |
| Serving host | `spark2` (DGX Spark, GB10, sm_121a), model data on local disk |
| Working clone on spark2 | `/home/spark2/sparkpipe` (⚠ may be dirty — see §3 step 0) |
| Deployment root | `/home/spark2/sparkdata/qwen38.fp8.tp1/` |
| Target model pack (FP8, production) | `…/packs/qwen38-fp8.tp1.qwen38_27bsp` (29 GB, MX E4M3+E8M0/128, format 6) |
| Target model pack (F32B128, original) | `…/packs/qwen38-fp8-F32B128.orig.qwen38_27bsp` (28 GB) |
| Target model pack (BF16) | `…/packs/qwen38.tp1.bf16.qwen38_27bsp` (51 GB) |
| DFlash2 drafter pack | `/home/spark2/sparkdata/qwen38-dflash2-drafter.qwen38_27bsp` (3.6 GB, 5-layer BF16) |
| Deployment configs | `…/config/model_resident.json` (daemon/deployment), `…/config/qwen38_27b_tp1_rank0.json` (adapter: pack path, revision, 8192 max positions) |
| Deployed binaries | `…/bin/sparkpipe_model_residentd`, `…/bin/sparkpipe_model_batch`; `…/lib/model_driver.so`, `model_serving_adapter.so`, `hidden_transport.so` |
| Daemon log | `/tmp/qwen38.log` (truncated by the launcher each start) |
| Reference server (vLLM, for comparisons) | `spark3` |
| TP4 fabric hosts (DSV4 work, not qwen38) | `spark4`–`spark7` |

Model identity: `Qwen/Qwen3.8-27B`, revision
`bf16-h5120-l64-gdn48-full16-v248320-mtp1-v1` — 64 layers (48 GDN +
16 attention), hidden 5120, vocab 248320, +1 MTP layer.
The FP8 pack was produced from the F32B128 original by
`tools/qwen38_27b_stagepack_mx_repack.py` (both known bugs fixed: per-entry
rows/cols gate; header `file_bytes` rewrite at offset 112). Re-running the
repack is NOT needed unless you regenerate weights.

## 2. Prerequisites / gotchas (read once, saves hours)

- **One daemon at a time.** GPU memory is ~78 GiB and a stale daemon makes
  the next start fail with `phase=adapter_initialize capacity_exceeded`.
  Always kill first: `pgrep -f "[s]parkpipe_model_residentd" | xargs -r kill;
  sleep 3; nvidia-smi --query-compute-apps=pid --format=csv,noheader |
  xargs -r kill -9; sleep 4`.
- **Re-run benchmarks on a fresh daemon** for clean numbers.
- **The deployment JSON must declare page capacities** now that prefix
  caching is on: `kv_logical_page_capacity: 256` (≥
  `resident_sequence_capacity`) and `kv_physical_page_capacity: 64` (≥
  `max_active_sequences`). Zeros fail deployment validation with
  `invalid_argument`.
- **Push auth**: the Mac's `gh` token is dead; pushes work from spark2
  (`gh` logged in as `sparkpipe`). The `/tmp/launch_*.sh` and `/tmp/*.py`
  helper scripts on spark2 are SESSION-EPHEMERAL — the launcher below is
  reproduced in §5 so you don't depend on them.
- Batch benchmark files also live in `/tmp` on spark2 and vanish; §6 has
  the generator.

## 3. Build (on spark2, from the release tag)

```bash
# step 0: clean clone at the tag (the long-lived /home/spark2/sparkpipe
# tree has carried working-tree edits during development)
cd /home/spark2 && git clone -b qwen38-dflash2-20260821 \
  https://github.com/sparkpipe/sparkpipe.git sparkpipe-rel && cd sparkpipe-rel

# step 1: module (GPU validator MUST print PASS)
pgrep -f "[s]parkpipe_model_residentd" | xargs -r kill; sleep 3
make -C modules/qwen38_27b_resident_decode_stage -j8 publish \
  NVCC=/usr/local/cuda/bin/nvcc CUDA_ARCH=sm_121a \
  STAGE_PACK_PATH=/home/spark2/sparkdata/qwen38.fp8.tp1/packs/qwen38-fp8.tp1.qwen38_27bsp \
  STAGE_COUNT=1 STAGE_INDEX=0 STAGE_FIRST_LAYER=0 STAGE_LAYER_COUNT=64 \
  TP_DEGREE=1 TP_RANK=0 TP_STANDALONE=1 MTP_LAYER_COUNT=1 \
  GDN_SNAPSHOT_SLOT_COUNT=16 MAX_ACTIVE_SEQUENCES=8 KV_BLOCK_COUNT=8 \
  ALLOW_UNQUALIFIED_EXECUTION=1
#   → expect: "qwen38_27b_validation PASS"

# step 2: node binaries (daemon, bench tool, and the driver compiler)
make -j8 build/sparkpipe_model_residentd build/sparkpipe_model_batch \
  build/sparkpipe_model_compile

# step 3: serving adapter (TP1)
rm -f build/libqwen38_27b_serving_adapter.so
make build/libqwen38_27b_serving_adapter.so \
  CC="cc -DSPARK_QWEN38_27B_SERVING_TP_DEGREE=1u" -j8

# step 4: driver compile
rm -rf /tmp/qwen38-driver-new && mkdir -p /tmp/qwen38-driver-new
env SPARK_QWEN38_27B_STAGE_COUNT=1 SPARK_QWEN38_27B_STAGE_INDEX=0 \
  SPARK_QWEN38_27B_STAGE_FIRST_LAYER=0 SPARK_QWEN38_27B_STAGE_LAYER_COUNT=64 \
  SPARK_QWEN38_27B_TP_DEGREE=1 SPARK_QWEN38_27B_TP_RANK=0 SPARK_QWEN38_27B_TP_STANDALONE=1 \
  SPARK_QWEN38_27B_STAGE_MTP=1 SPARK_QWEN38_27B_STAGE_GDN_SNAPSHOT_SLOTS=16 \
  SPARK_QWEN38_27B_STAGE_MAX_ACTIVE_SEQUENCES=8 SPARK_QWEN38_27B_STAGE_KV_BLOCKS=8 \
  SPARK_QWEN38_27B_STAGE_KV_STORE=none SPARK_QWEN38_27B_STAGE_KV_SERVICE=none \
  SPARK_QWEN38_27B_STAGE_KV_SOCKET=none SPARK_QWEN38_27B_STAGE_KV_POOL_BYTES=0 \
  SPARK_QWEN38_27B_STAGE_KV_WORKER_COUNT=0 SPARK_QWEN38_27B_ALLOW_UNQUALIFIED_EXECUTION=1 \
  build/sparkpipe_model_compile \
  --model examples/model_descriptions/qwen38_27b_resident_decode_stage_firmware.json \
  --library build/module_library --output /tmp/qwen38-driver-new \
  --include include --cc-arg -L/usr/local/cuda/lib64 --cc-arg -lcudart --cc-arg -lstdc++

# step 5: deploy ALL FOUR artifacts (stale-mix = subtle numerics bugs)
D=/home/spark2/sparkdata/qwen38.fp8.tp1
cp /tmp/qwen38-driver-new/stages/stage_000/model_driver.so $D/lib/
cp build/libqwen38_27b_serving_adapter.so $D/lib/model_serving_adapter.so
cp build/sparkpipe_model_residentd $D/bin/
cp build/sparkpipe_model_batch $D/bin/
```

Local dev machines (Mac): `python3 tests/test_code_size.py` gates authored
code size (ceiling ratchets require justification in the same change).

## 4. Deployment config (already correct on spark2 — for regenerating)

`config/model_resident.json` essentials: single node/rank/stage 0,
`adapter.shared_object_path = lib/model_serving_adapter.so`,
`driver.program_name = resident_decode`,
`transport.mode = host-rdma`, `control_port_base 58700`,
`runtime_limits`: submissions 2, active 64, rows 128, resident 64,
**kv_logical 256 / kv_physical 64**, node `runtime_root` =
`/home/spark2/sparkdata/qwen38.fp8.tp1`, node_target
`cuda.sm121.qwen38_27b.resident_decode_stage.bf16`,
`adapter_configuration_path = config/qwen38_27b_tp1_rank0.json`, endpoint
`tcp spark2:17480`.

`config/qwen38_27b_tp1_rank0.json`: `stage_pack_path` → the FP8 pack,
`model_revision` as above, `max_sequence_positions: 8192`.

## 5. Launch (production daemon)

```bash
#!/bin/bash
# /tmp/launch_prod.sh — daemon with the release configuration
pgrep -f "[s]parkpipe_model_residentd" | xargs -r kill 2>/dev/null; sleep 3
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null; sleep 4
cd /home/spark2/sparkdata/qwen38.fp8.tp1
: > /tmp/qwen38.log
export LD_LIBRARY_PATH=$PWD/lib:${LD_LIBRARY_PATH:-}
export SPARK_QWEN38_27B_PROFILE=1                                  # optional: phase counters
export SPARK_QWEN38_27B_SERVING_SPECULATE=1 SPARK_QWEN38_27B_SERVING_SPEC_METHOD=dflash2
export SPARK_QWEN38_27B_SERVING_SPECULATIVE_DRAFT_COUNT=8          # k=8 = all drafter mask slots
export SPARK_QWEN38_27B_DSPARK_PACK_PATH=/home/spark2/sparkdata/qwen38-dflash2-drafter.qwen38_27bsp
export SPARK_QWEN38_27B_DFLASH2_STATE_SELECT=1 SPARK_QWEN38_27B_DFLASH2_BONUS_FOLD=2
export SPARK_QWEN38_27B_DFLASH2_BLOCK_KV=0                          # MUST be 0 (see handoff: E 4.81 vs 5.66)
export SPARK_QWEN38_27B_DFLASH2_WINDOW=2048 SPARK_QWEN38_27B_DFLASH2_CTX_CACHE=1
setsid nohup bin/sparkpipe_model_residentd --deployment config/model_resident.json \
  --rank-index 0 > /tmp/qwen38.log 2>&1 < /dev/null &
for i in $(seq 1 90); do grep -q "model_residentd ready" /tmp/qwen38.log 2>/dev/null && break; sleep 1; done
grep -q "model_residentd ready" /tmp/qwen38.log && echo READY || { echo FAILED; tail -3 /tmp/qwen38.log; }
```

No-spec daemon: same script minus the `SPECULATE/SPEC_METHOD/DRAFT_COUNT/
DSPARK_PACK_PATH/DFLASH2_*` exports. Default-ON kill-switches (rarely
needed): `SPARK_QWEN38_27B_FRAME_GRAPH=0`, `SPARK_QWEN38_27B_WS_PLAIN=0`,
`SPARK_QWEN38_27B_WS_GEMM=0`.

Ready line to expect:
`… lanes=64 kv_blocks=8192 device_gib=77.7` then
`model_residentd ready rank=0 … tcp=spark2:17480`.

## 6. Benchmark

Generate the standard O512 batch (real-text prompt; the o128 token ids
below are the canonical ones — do not substitute random tokens or
acceptance collapses):

```python
# writes /tmp/o512_seq8.json — the CANONICAL 128-token real-text prompt is
# embedded here (do not substitute random tokens: drafter acceptance
# collapses and the expected streams/rounds below stop matching)
import json
PROMPT = [0, 3476, 477, 18068, 260, 3375, 35312, 3417, 16, 38074, 13254, 16, 455, 4087, 3287, 2231, 1605, 270, 21361, 8786, 9045, 16, 128803, 79418, 2317, 566, 8130, 345, 14866, 3312, 2019, 16, 983, 1142, 469, 1142, 554, 6242, 260, 31191, 603, 19905, 418, 270, 4031, 2455, 2562, 1167, 1479, 270, 6074, 15398, 344, 10097, 16, 2052, 270, 15398, 344, 1353, 4521, 538, 260, 2395, 2740, 294, 18885, 6243, 14, 20430, 418, 270, 19904, 50098, 5898, 1789, 638, 1341, 294, 6319, 2562, 3737, 603, 25529, 223, 18, 855, 270, 2019, 344, 7681, 1202, 270, 10844, 22283, 339, 671, 2019, 109029, 260, 716, 15, 10554, 30347, 112566, 1936, 14327, 436, 304, 270, 489, 5927, 7104, 339, 9945, 1137, 9854, 69, 201, 223, 19, 28, 1823, 11006, 334, 30557, 32684, 16617]
json.dump({
  "schema_version": 1, "connect_timeout_ms": 30000, "request_capacity": 2,
  "max_context_tokens": 4096, "max_prefill_rows_per_submission": 8,
  "maximum_messages_per_rank_per_progress": 8,
  "maximum_new_submissions_per_progress": 2, "stop_token_ids": [],
  "requests": [{"request_id": 760130, "sequence_id": 760130, "priority": 0,
    "output_token_budget": 512, "prompt_token_ids": PROMPT}]},
  open("/tmp/o512_seq8.json","w"))
```

Run + verify:

```bash
cd /home/spark2/sparkdata/qwen38.fp8.tp1 && export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
/usr/bin/time -f "WALL %e" bin/sparkpipe_model_batch \
  --deployment config/model_resident.json --runtime-root $PWD \
  --batch /tmp/o512_seq8.json > /tmp/run.out 2>/dev/null
grep -c '"event":"token"' /tmp/run.out          # 512
python3 - <<'EOF'
import json,hashlib
t=[json.loads(l)["token_id"] for l in open("/tmp/run.out") if json.loads(l).get("event")=="token"]
print(len(t), hashlib.sha256(str(t).encode()).hexdigest()[:16])
EOF
```

**Release acceptance numbers** (same daemon, fresh start):

| Check | Expected |
|---|---|
| Spec O512 wall | 20.9–21.0 s (24.5 tok/s), stream `d7f798801a6e43a6` |
| Rerun on same daemon | identical stream (live-daemon rerun is supported) |
| No-spec O512 | 66.4 s (7.7 tok/s), stream `5d6ee525deb999f5` |
| Rounds | 77 (`grep -ac spec_diag /tmp/qwen38.log`), mean accepted ≈ 5.66 |
| Prefix cache | `grep -a "qwen38_27b_prefix borrow" /tmp/qwen38.log` fires on a repeat prompt; output bit-identical to cold |

Sequential-arrival mode (prefix-cache testing): run the batch tool with
`SPARK_MODEL_BATCH_SEQUENTIAL=1` and a 2-request file — request 2 must
arrive after request 1 terminates (the scheduler co-prefills simultaneous
lanes from zero, so same-batch requests never hit).

## 7. Troubleshooting quick table

| Symptom | Cause / fix |
|---|---|
| `phase=adapter_load … invalid_argument` at start | adapter/driver build mismatch — rebuild step 3 AND 4, deploy both |
| `phase=deployment_validation … invalid_argument` | missing/zero `kv_*_page_capacity` in model_resident.json (§2) |
| `phase=adapter_initialize capacity_exceeded` | stale daemon holding GPU — full kill sequence (§2) |
| `route_failed … reason=2` on RELEASE submissions | adapter release path returned bad residency — don't strip the `pending->residency = submission->residency` line |
| Request errors immediately, log shows `qwen38_27b_prefix miss … recomputing` | adapter prefix store missed (client cache outlives it) — correct-but-slow fallback; expected after daemon restarts mid-batch |
| Acceptance ~4.8 instead of 5.66 | `BLOCK_KV` left at 1 — must be 0 |
| Wall jumped on "identical" build | stale artifact mix in `lib/` (deploy all four, step 5) |
| Bench hangs at 0 tokens | client raced a daemon restart — rerun after READY |

## 8. Where to go next (pointers, not instructions)

- Perf analysis + next levers: `docs/DFLASH2_HANDOFF.md` (round structure,
  FFN in-situ locality, kernel ledgers).
- Kernel benches: `tools/qwen38_27b_native_warp_specialized_bench.cu`,
  `tools/qwen38_27b_multirow_dot_bench.cu` (build line in each header).
- TP4×PP4 direction: TP4 fabric characterization + measured allreduce
  latencies in `PERFORMANCE_STATUS.md`; DSV4 TP4 B1 production experience
  in `DSV4_TP4_B1_HANDOFF.md`; qwen38 TP4 build scaffolding in
  `qwen38_tp4_build.sh` (BF16, never carried to a decode measurement).
- Estimated at TP4×PP4 (16 nodes, single stream): ~48–55 spec tok/s;
  see the release notes' scaling section for the model behind it.
