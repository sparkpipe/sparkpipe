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

## CUDA dispatch (built this round, sm_121a compile-verified)

- `modules/k3_resident_decode_stage/` gained the serving-tier bridge:
  `spark_k3_resident_decode_stage_cuda.{h,cu}` (commits 2155b3f7, c481a008).
  It fills the device `K3LayerWeights` table from the binder's name tables
  (offsetof table over slice.cuh fields), carves the `K3SliceState` recurrent
  pools exactly as `tests/host_cuda/k3_slice_host.cu` does, builds the MLA
  `LmKvView` array, allocates the per-step scratch blob, and launches
  `K3StageSlice` from bind.cu's ABI.
- `tools/k3_dispatch_compile_gate.sh` compiles it for sm_121a on sparka
  (worktree /home/sparka/sparkpipe-k3): PASS. The `K3StageSlice` extern "C"
  declaration matches bind.cu exactly.
- Pack mmap is `cudaHostRegister`ed for UVA weight access (GB10 unified
  memory, no device copy of the ~87 GB rank pack).
- Two facts the dispatch encodes that the end-to-end run will surface:
  1. Pack V2 has NO scale tensors outside the interleaved experts (BF16
     spine), so every `K3LayerWeights` `*_scale` field stays NULL and
     `expert_interleave` is 1 on every MoE layer.
  2. `K3LayerLatentMoe` currently FAILS CLOSED on `expert_interleave=1`
     (layer.cuh comment: the grouped GEMM has not learned the interleave
     cell). The first live MoE launch errors until that kernels wave lands;
     the dispatch still encodes the pack truth so nothing runs silently wrong.
- The head step signature surfaced while compiling: `K3Head(buffers,
  head_norm_weight, head_weight, token_ids, vocabulary, rows, stream)` at
  layer.cuh:909 - the PP-last rank's serving tier calls it after the slice.

## Final round state (round 16)

- `SPARKPIPE_K3_SOURCES` added to sources.mk (pack_load, bind, module .c
  files; the .cu dispatch TU follows the DSV4 module's CUDA pattern).
- `tools/k3_autoslice.sh` watchers are LIVE on spark1/4/8/c: each polls for
  its stage pack's final `.pack` (payload streamed in and unlinked on
  close), then runs `k3_shard.py` to emit the four TP4 rank packs.
- `tools/k3_deploy_ranks.sh` copies the 16 rank packs to
  /home/<host>/sparkdata/k3.mxfp4.tp4pp4/packs/k3.stage<s>.rank0<t>.pack
  with sha256 verification (stage s ranks t on spark[s*4+t]).
- Stage pack progress at last check: sparkc layer 91/92 (~356 GB), spark4
  44/46 (~339 GB), spark8 65/69, spark1 19/24 - all four packers healthy.
- REMAINING to a live run, in dependency order:
  1. Rank deployment (running: stage-node-local rsync pushes, ~500 MB/s,
     sha256-verified per pack via tools/k3_deploy_stage.sh).
  2. ~~Serving adapter + stage runner~~ LANDED: the runner (embed -> slice ->
     head + TP wiring, the collective hook) and the serving adapter
     (SparkModelServingAdapterInterface, example config) both compile-gate
     on sm_121a.
  3. ~~The expert_interleave grouped-GEMM wave~~ LANDED + gated.
  4. ~~The rank-sliced dimensions~~ LANDED: the layer calls read the rank
     pack's shapes through K3_RANK_DIM; the single-spark real-weight gate
     now PASSES bit-deterministic (0 mismatches, 6/6 identical dumps) and
     the warm step runs in ~56 ms.
  5. The full-cluster end-to-end run itself: needs a spark ring reservation
     (live runs are not allowed without one), then `tools/fleet_swap.sh k3`.

## Device-direct collective + TP16 round (2026-08-16)

- The adapter parses a `device_collective` JSON object (backend nccl |
  hidden_transport, `backend_module_path`, `local_host`,
  `collective_identifier`, `listen_port`, timeouts, `peer_hosts`),
  builds the topology, calls SparkTpDeviceCollectiveApplyTopology, and
  hands the runner the completed config. The runner attaches the K3
  combine kernels ONLY for the hidden-transport backend (NCCL's config
  validation rejects non-null combine functions).
- The per-layer hook's device branch packs attention_out | hidden |
  shared_out into ONE fused rows x 3x7168 buffer and issues one
  stream-ordered all-reduce whose completion folds the summed segments
  into the AttnRes partial on the same stream - no sync, no host staging.
- NCCL lib staged: every spark's runtime root now has
  `lib/runtime_libs/libnccl.so.2` (identical sha256 bf23e731102d8ec2,
  copied from the DSV4 Flash runtime), and the regenerated per-rank
  configs point `backend_module_path` at it. All 16 adapter.json
  restaged with `world_size`: 16 and the device tier section; every rank
  shares control port 64620 (the fetchers dial rank 0's host on their own
  control_port_base value).
- Adapter .so rebuilt (nvcc sm_121a gate PASS), restaged to all 16 sparks
  (identical sha256 037628ca7dc8ea40). Single-spark step gate re-PASS:
  bit-deterministic, warm step 54.8 ms.
- Two-phase layer collective landed: the hook fires after the attention
  half AND after the MLP half - required for correctness (the MLP-side
  AttnRes retrieval reads the POST-attention partial, which the old single
  post-MLP hook could not provide to the sharded path), and each phase's
  fold lands on the submission stream (no legacy default stream).
- Per-shape CUDA-graph capture landed: warm-direct first submit, capture
  on the second, replay after; gated on a non-default stream and a
  capture-safe tier (NCCL or tp_degree 1). The single-spark gate now
  validates it: graph-replay vs direct launch 0 mismatches, pure replay
  54.2 ms (the path is weight-bandwidth-bound; the capture removes the
  ~55 ms serialized host enqueue per step). Configs carry
  `capture_graphs`: 1.
- TP16 sharder audit: the earlier "unbalanced w2" split was INCONSISTENT -
  the gate|up output split gives each rank 192 intermediate elements
  (96+96, the halves must correspond element-wise), while the unbalanced
  down split consumed 128 or 256 k-elements, so a rank would read
  intermediate elements it never computed. No whole-tile scheme fixes
  TP16 (the gate half is 12 tiles, < 16 ranks); the sharder now REFUSES
  TP16 loudly and the fix is the 64-element half-tile repack (pack V3 +
  TILE_K=64 interleave variant). Everything else TP16 (configs, NCCL
  degree 16, geometry) stays staged.
- Test suite reconciled with the released checkpoint: the packer's top_k
  fallback no longer KeyErrors on num_experts_per_tok-only configs (the
  eager .get default), and the mini fixtures now carry the
  language_model. prefix, the dense layer-0 MLP, the full-rank KDA/MLA
  g_proj, and the block_sparse_moe nesting. test_k3_pack (66 tensors,
  byte-exact) and test_k3_shard (66 x 2 ranks, every class reassembles,
  the k-tile-indivisible degree refused) PASS on sparka; the adapter
  smoke PASSES through the PP1 derive path (world_size 1) on the real
  rank pack.
- TP16-ready geometry landed: the adapter derives the PP stage split from
  world_size/tp_degree (16/16 -> PP1), the module takes slice bounds from
  the pack manifest via SPARK_K3_MODULE_DERIVE_SLICE, the bound-layer cap
  is 93, and the generator emits TP16 configs (16 peers, no host tier -
  its 4-rank cap). NCCL + hidden transport both support degree 16.
