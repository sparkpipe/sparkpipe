# DSV4 Flash DSpark k-sweep — receipt (v4 pack, k=7 first)

Baseline anchor: **40.19 tok/s no-spec B1, 3/3 exact** (driver `3d962820`).

## What landed this session (all validated)

1. **Code changes** (buckets 6/9/11 + SPEC_STEP override):
   `spark_dsv4_batch_tuning.h` (allowlist + B6/B9/B11 module IDs + #elif + switch),
   `spark_dsv4_model.h` (#ifndef SPEC_STEP guard + B6/B9/B11 sha constants),
   `tools/generate_dsv4_contracts.py` (bucket list + flash_module_id + SPEC_STEP guard),
   `modules/.../spark_dsv4_serving_adapter.c` (B6/B9/B11 → sha map),
   `modules/.../Makefile` + `build_remote.sh` + `devcycle.sh` (DSPARK_SPEC_STEP knob).
   Pinning tests green: test_dry_law, test_dsv4_contracts, test_dsv4_module_host_syntax.
2. **4 drivers built** (archive validation PASS):
   k=5 (bucket6) `41a241e4…`, k=7 (bucket8) `9d7ac21a…`, k=8 (bucket9) `ba3c0a88…`,
   k=10 (bucket11) `77f43b1d…`, staged at tools/devcycle/drivers/lean-dspark-k{5,7,8,10}/.
3. **Checkpoint** downloaded (48 shards, 0 fail) + full index re-fetched (the
   partial index only covered layers 12-14 — that was the earlier re-pack failure).
   Full index: 72317 tensors, 43 layers + **4705 mtp.* draft tensors** (the DSpark
   draft weights ARE in DeepSeek-V4-Flash-0731 under `checkpoint_namespace: mtp`).
4. **Re-pack v4**: full pack 166.9 GB, 1409 tensors, sha `56a07b2d…`, validated.
5. **Sharded** via `dsv4_tp16_stagepack.py --model flash --tp-degree 4` into 4 rank
   packs (~51 GB each, format_version 4, validated) — +11 GB vs the old 40 GB v3
   packs is the 3 MTP draft layers. Deployed rank0→spark4 … rank3→spark7.

## k=7 spec result — INITIALIZES, runs one verify step, then errors

The v4 pack FIXED the `pack_geometry_mismatch` init failure. The b8 spec driver
(k=7, `9d7ac21a…`) now reaches `model_residentd ready` on all 4 ranks with
`device_gib=48.6` (v3 no-spec was 38.5).

First decode submission then logs:

```
dspark_staging tp_rank=0 prefill=1 enabled=1 rows=1   (prefill)
dspark_staging tp_rank=0 prefill=0 enabled=1 rows=1   (first decode)
dspark_accept tp_rank=0 lane=0 accepted=0 rows=8 chain=1
model_residentd run=io_error status=4 rank=0 stage=0 reason=9 submission=0 kind=0 route_state=0
```

`dspark_accept accepted=0 rows=8` proves the DSpark verify loop EXECUTED (8 verify
rows, greedy accept of 0 draft positions → 1 bonus token). Then the run loop fails
with `reason=9` = `SPARK_MODEL_RESIDENTD_FAILURE_CLIENT_LEASE_DISCONNECT` (the
client's continuation lease disconnected before the stream completed). Client sees
`sparkpipe_model_batch_status=4 tokens=0 terminal=0`. Consistent across retries —
**a continuation-lease bug in the DSpark spec path, not a transient timeout**.

## Remaining blocker (needs speculation/kv-cache/coordinator)

The spec verify path emits its first burst then the continuation lease disconnects.
Likely the spec lane continuation (tap store publish / lane advance / next-submit
lease) is not wired the way the no-spec continuation lease expects. Greedy-only,
k=7 only so far (k=5/8/10 drivers are built and ready once this is fixed).