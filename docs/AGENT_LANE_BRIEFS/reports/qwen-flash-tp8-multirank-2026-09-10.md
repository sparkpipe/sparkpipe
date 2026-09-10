# qwen-flash: step 3 multi-rank TP8 functional PASS 8/8 — step 4 staged, wave blocked by sparkd outage — 2026-09-10

## Step 3 result (PASS)

Queue v2 job `qwen-flash-tp8fn-1` (attempt 017ec646c44f41b69566afd25d565884,
--per-node spark0-7, gpu, 10240 MiB, ttl 15): all 8 ranks ran the whole-stack
ladder (`qwen4_flash_validation PASS`, 17/17) against each node's OWN rank
shard of the placed `qwen3flash.bf16.tp8` arm, fresh 31383ad sync
`qwen-flash-cuda5c`, MTP-carrying packs, TP_DEGREE=8 TP_RANK=0
TP_STANDALONE=1 (the harness's whole-stack admission tier).

| node | rank | rc | wall | pack sha16 | .experts sha16 |
|---|---|---|---|---|---|
| spark0 | 0 | 0 | 130s | 44b78acc6af814b5 | df02369ac7535b69 |
| spark1 | 1 | 0 | 152s | 65a44e9b173512b2 | 6cc493cca674bae9 |
| spark2 | 2 | 0 | 132s | f84ad41b631261ae | ca1c6f36fcc6ac7f |
| spark3 | 3 | 0 | 136s | 9d4d447050288133 | 46e72729c52c547f |
| spark4 | 4 | 0 | 122s | 5b0d8ffcd65c7405 | 8656369f24b1caf2 |
| spark5 | 5 | 0 | 134s | 6aa25ab0ed538d7f | 634fb6ebc29952fd |
| spark6 | 6 | 0 | 121s | 52e9cb42c75f01a9 | e776d3899f02984f |
| spark7 | 7 | 0 | 123s | 1c3875cdc507e6ff | edd398ad8f604e5d |

decode_vs_prefill bit_exact=1, module_determinism bit_exact=1,
mtp_draft in_vocab=1, ple_hash_gather bit_exact=1 on every rank
(per-node unit logs /tmp/sparkqueue-017ec646*.log, rc files
`~/q4f_tp8fn_rank<r>.rc`).

## Rank-pack + sidecar verification (all green)

- bf16.tp8: 8/8 packs present (46333527808 B each, distinct rank shas) each
  WITH its 27656 B `.experts` sidecar.
- spark0 rank0 `.experts` GAP: CLOSED before this session — sidecar present
  (Sep 5 08:03), byte-identical to spark4's rank0 copy (both
  df02369ac7535b69); verified, no replacement write needed.
- nvfp4.tp8: 8/8 packs match their `.sha256` sidecars; rank5 = 3cbad2bc…,
  matching the packer receipt from the coexistence cell.
- spark4's duplicate rank0 pair is byte-identical to spark0's (44b78acc… /
  df02369a…).
- v4 TP4xPP4 arm: M4 sha gate 16/16 PASS vs build receipts
  (spark4:~/q4f_w2_gate_16xPASS.txt).

## Step 4 (TP4xPP4 wave + B1) — STAGED, launch blocked by sparkd outage

Done:
- 16 deploy trees staged (`~/sparkdata/qwen4_flash.tp4/deploy_v4`: bin from
  spark5 q4f_serving 07:12 build, lib incl FIXED adapter, deployment.json +
  launch_table.json + config from the 31383ad generator, packs_v4 symlink to
  the moved `qwen3flash.bf16.tp4/packs_v4` placement, B1 batches:
  q4f_smoke_batch.json (80-token canon, max 128) + qwen4_flash_b1_32k_batch.json
  (32640 prompt ids + max 128)).
- Defect found by the wave bring-up and FIXED (commit 5486d68): the serving
  adapter descriptor reported `cache_block_token_count = 0`; the batch engine
  fails closed (`initialize=invalid_argument phase=adapter_load`). The
  descriptor now carries
  `SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS` (64u) — the same
  constant the adapter already used for the model contract block size.
  Rebuilt from the cuda5c tree on spark5, restaged to all reachable deploy
  trees (sparkd unreachable at restage time — re-stage its lib/ when the node
  returns).
- Launcher gain (commit 8c1b61e): `SPARK_FLEET_RUNNER` knob wraps each
  residentd spawn; recorded pid now comes from an in-scope pidfile so
  teardown still TERMs the real process.

Blocker: **sparkd is DOWN** (fabric 10.10.100.23 and mgmt 10.20.0.23 both
unreachable; alive at ~09:04, down by 09:5x). A 16-rank TP4xPP4 chain cannot
form without stage-3 rank 1. Fleet-ops action needed; when sparkd returns:
re-stage its lib/ (fixed adapter), then relaunch `qwen-flash-w2wave-3` with
the existing orchestrator `spark4:~/q4f_w2_wave.sh` and the watcher job
pattern (exit42=re-add).

## sparkcap mandate compliance

- No pack-copy or pack-write occurred: the spark0 rank0 sidecar already
  existed; the M4 gate and rank sweeps are read-side verification; the only
  staging copies were ~17 MB/node of binaries/configs (no pack bytes).
- sparkcap verified deployed (spark4/spark0/spark8 sampled). GAP reported per
  mandate: `sparkcap` cannot create its user scope over headless ssh
  ("Interactive authentication required" from systemd-run --scope); the
  launch therefore wraps each residentd with the mandate's fallback form
  `sudo -n systemd-run --scope -q -p MemoryMax=10240M -p MemoryHigh=7168M --`
  (10240 MiB = the queue-capped envelope the publication/coexistence cells
  proved for reading the 46 GB pack). Queue --memory-mib does not cover the
  detached residentd processes, hence the wrap.

## Weights law audit (direct pack-mmap paths; deletion targets)

| path | verdict | action |
|---|---|---|
| ladder/validator pack load (chunked, TP_STANDALONE cells) | test tooling — exempt | none |
| residentd serving-stack pack load with no weightd member in deployment.json (current v4 generator output) | direct-load, deprecated-with-removal | DELETE at step 5: generator emits mandatory weightd identity; stack fails closed without it |
| module eager/chunked pack loader inside the resident decode stage | direct-load, deprecated-with-removal | DELETE at step 5: load ONLY via shared seam (stage_module_common + spark_weightd*, SparkStageModuleLoadDeviceRegion) |
| weightd lazy-attach path (SPARK_WEIGHTD_SOCKET/ATTACH, proven in publication cell) | sanctioned sole weight path | becomes the only path at step 5 |
| this PR's changes (launcher runner knob, descriptor cache_block_token_count, deploy restage tooling) | no new direct-load code | none |

Interim steps 3-4 used the already-built direct path for receipts only; no
new direct-load code was introduced.

## Commits (lane/qwen38flash-wave2 @ 31383ad + 2)

- 8c1b61e qwen4_flash wave launcher: SPARK_FLEET_RUNNER knob wraps each
  residentd spawn; recorded pid from an in-scope pidfile
- 5486d68 qwen4_flash serving adapter: descriptor sets cache_block_token_count

## Next (exact)

1. Fleet-ops: bring sparkd back; then `ssh sparkd 'cat /tmp/…so >
   ~/sparkdata/qwen4_flash.tp4/deploy_v4/lib/libqwen4_flash_serving_adapter.so'`
   (or re-copy from spark4:/tmp/libqwen4_flash_serving_adapter_fixed.so).
2. Re-add the wave job (w2wave-3, same cmd as w2wave-2) — orchestrator
   `spark4:~/q4f_w2_wave.sh` already exports SPARK_FLEET_RUNNER to the sudo
   systemd-run fallback and runs launch + B1 smoke ×2 determinism + exact-32K,
   receipts into `deploy_v4/wave_cells-*.receipt.md`.
3. On wave PASS: step 5 lazy-qualified gate (weightd sole authority,
   fail-closed, SPINE vs EXPERT bytes reported separately) + the deletion of
   the direct-load paths in the same change.
