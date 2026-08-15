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
- Serving-tier binder written (modules/k3_resident_decode_stage/
  spark_k3_bind.c): per-layer-kind name tables (every-layer, KDA 11,
  MLA 8, MoE 9, dense 2) resolving pack entries by name; host test
  8/8 PASS against the real layer-0 pack (dense KDA: gate present,
  dense payload resolves, experts correctly absent).
- Packer RESUME added: a side journal records each tensor only after its
  bytes are on disk; a killed run re-walks the journal, truncates to the
  last complete tensor, and skips re-generating emitted tensors.
  tools/k3_watchdog.sh restarts a killed pack on the stage nodes, so the
  externally-killed stage packs now survive and finish (the packs were
  being killed by an unknown external process; the watchdog + journal
  make the pack build self-healing).

## Per-spark data budget (TP4 x PP4, 1/16 per rank)

Weights: the full PP-stage pack (~350 GB, 1/4 of the model) is sliced by
k3_shard.py into four rank packs (~87 GB each, 1/16); each spark's
internal NVMe holds exactly its rank pack + its stage's checkpoint shards
(~400 GB) until packing is retired. The rank pack is the deployment unit.

KV and recurrent state, per rank (its 23-layer slice: 17 KDA + 6 MLA):

  - KDA slot pool: 6.59 MB per layer (96x128x128 f32 state + conv
    windows) x 17 layers = ~112 MB per sequence per rank (the full-model
    number is 448 MB; PP4 quarters it).
  - MLA token arena: 1152 B/token/layer (512 latent + 64 unrotated, bf16)
    x 6 layers = ~6.9 KB/token per rank. The kv_a latent replicates within
    the TP group (512-wide cannot shard usefully) - the rank-local saving
    is the PP slice (6 of 24 layers), not a TP split of the latent.

The serving module must size both pools from ITS slice layer counts
(model-families/k3/spark_k3_kv_geometry.h holds the full-model constants;
the module's capacity request scales them by first_layer/layer_count).
- Full binder test 16/16 PASS against the real 0-4 pack (dense-KDA,
  routed-KDA, routed-MLA). Fixed the pack loader to read manifest offsets
  with SparkJsonGetUInt64 - pack offsets exceed 2^32 for the ~350 GB stage
  packs (the common JSON API already exported the uint64 getter; no
  common-code change was needed).
- Rank-slicing recipe ready: tools/k3_tp4_slice.sh runs k3_shard.py over a
  completed stage pack and emits the four TP4 rank packs (~87 GB each).
  Deployment layout: stage s ranks t (spark[s*4+t]) get
  /home/<user>/sparkdata/k3.mxfp4.tp4pp4/packs/<stage>.<s>.rank0<t>.pack,
  matching the fleet_registry pack_dir.
