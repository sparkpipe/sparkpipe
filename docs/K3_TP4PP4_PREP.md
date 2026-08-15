# K3 TP4xPP4 — concurrency contract and preparation notes

## Fleet slotting (multi-model registry)

K3 TP4xPP4 is a FULL-16 topology, so it rides Big slot B: all hosts,
control 20480, collective 64620-64623, transport control 61700.
It never runs concurrently with another measured window; per the fleet
rule, residentd processes may be dev-active (up, idle) anytime, but a K3
measured window is EXCLUSIVE on the whole fleet — the always-on
DSV4 Flash TP4 (spark4-7) and Qwen 27B PP16 (spark0-3) dev-active
residentds pause for the window and are restored after. Promotion is
under 60 s, so pause/restore is cheap.

My assigned dev node is sparka: single-node CUDA smoke tests, kernel
probes, and packer runs there need no residentd and no ports, so they
are always allowed regardless of slot state.

## Preparation state

- Checkpoint: moonshotai/Kimi-K3-MXFP4, 96 shards (~1.6 TB), already on
  13 of 16 sparks under /home/<user>/srcdata/kimi_k3.mxfp4.pp13/ (the
  PP13-era layout: shards 1-93 split 7-per-node, 94-96 replicated).
  sparkd, sparke, sparkf are empty.
- TP4xPP4 shard map (from the pinned index): stage 0 (layers 0-23) =
  shards 1-24; stage 1 (layers 24-46) = 25-47; stage 2 (47-69) = 48-70;
  stage 3 (70-92) = 71-93; tail 94-96 = final norm/head/MTP, stage 3.
- Copy plan: every node gets its stage's full shard set + the tail on
  stage 3 + metadata; sha256-verified against the concatenated
  .sparkpipe-dataset.json table.

## Progress log (2026-08-15)

- Shard fetch COMPLETE: 16/16 nodes hold their stage's full shard set
  plus the 94-96 tail, sha256-verified (0 mismatches).
- k3_pack.py reconciled with the released checkpoint (text_config,
  language_model prefix, block_sparse_moe MoE subtree, full-rank gates,
  slice support). Verified: layer-0 pack (18 tensors, 4.7 GB) and
  layers-1-3 pack (69 tensors, 50.6 GB) build from real shards.
- k3_shard.py TP4 classification updated for the full-rank gates;
  rank00 pack produced, full 4-rank slice running.
- Driver gate path reconciled (layer.cuh/slice.cuh); the K3 host gate
  (tests/test_k3_layer_host.py) passes.
- Stage-0 pack (layers 0-23, ~350 GB) packing on spark0.
- OPEN: the serving-tier module (manifest loader for the new tensor
  names + resident stage driver) does not exist in this revision; it is
  the next build item, followed by the TP4xPP4 deployment and the
  torch-reference numerical gate.
- Fleet coordination adopted (PR #649): K3 registry entry filled
  (owner, verified shard/pack layout, K3 port block 21480/65620/62700,
  fleet scope); fleet_swap.sh status verified; branch pushed to origin.
- All four PP stage packs RUNNING in parallel on their stage nodes with
  the fixed packer: stage0 (0+24) spark1, stage1 (24+23) spark4,
  stage2 (47+23) spark8, stage3 (69+23) sparkc. ~350 GB each, Python
  expert interleave loops dominate; multi-hour ETA. spark0 was dropped
  for stage0 because the Qwen session's sha256/rsync saturates its disk.
- Single-spark CUDA compile gate PASSED on sparka: bind.cu and unity.cu
  compile clean for compute_121a/sm_121a (nvcc 13.0) with the full-rank
  gate reconciliation. tools/k3_sm121a_compile_gate.sh preserves the
  exact flags (gencode arch=compute_121a,code=sm_121a; no forced model
  header, which drags spark_kv_cache.h into the host pass).
