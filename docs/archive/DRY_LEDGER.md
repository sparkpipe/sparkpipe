# The DRY ledger

Every unit in the tree, classified by what it actually depends on - not what
its name claims. The probe is mechanical: a `SparkGlm52X` symbol whose body
reads no `GLM52_` shape constant and no glm tensor name is generic machinery
wearing a model's name, and the name is the defect. Counts below are from
that probe (2026-07-28); re-run it before trusting this file over the code.

## The law

A model's name may appear in exactly two places: `inference/llms/<model>/`
(kernels, layer, slice, config) and `model-families/<model>/` (host-side
tables: shard classification, tokenizer/template data, shape config). Nothing
in `node/`, `ring/`, `include/sparkpipe/` or a generic tool carries a model
name. A `Spark<Model>X` whose body reads no `<MODEL>_` constant is a bug of
the same class as a wrong constant - it hides where the seams are, and the
next model copies 10,000 lines instead of writing a table.

## Already common (correctly named, no action)

`include/sparkpipe/`: status, json, sha256, tokenizer, kv_store, memlink,
hidden_transport, **tp_collective** (the all-reduce: log2-N halving-doubling,
degrees to 16, handshaked, tested - the exemplar this ledger generalises
from), module abi/library, driver compiler/loader, model_driver(+support),
model_description, orchestrator, stage_kv_client, stage_module_common,
sideband. `ring/transport/`: tcp, rdma, memlink, hidden_transport,
tp_collective.

## Generic wearing glm names (rename + move to common; shape-constant hits)

The serving stack, ~13,700 lines. Ordered by size; "hits" is real coupling,
not the prefix:

| unit | lines | hits | verdict |
|---|---|---|---|
| node/backend.c | 3814 | 7 | ring service backend; the 7 hits are KV-prefetch sizing to pull from the kv-cache seam |
| node/rank_daemon.c | 3088 | 1 | rank process; generic |
| node/residentd.c | 2658 | 5 | resident weight daemon; hits are sizing |
| node/rank_runtime.c | 994 | 0 | generic |
| spark_glm52_request_api.h | 537 | 2 | request slots/dispatch; 8 type refs into the kv seam |
| spark_glm52_serving_engine.h | 464 | 0 | continuous batching engine; pure prefix |
| spark_glm52_scheduler.h | 412 | 0 | decode cohorts + prefill reserve; pure prefix |
| spark_glm52_service.h | 393 | 0 | service wiring |
| spark_glm52_ring_work_control.h | 353 | 1 | work packets |
| spark_glm52_prefix_cache.h | 336 | 0* | radix prefix cache (21 prefix-name refs, 0 shape) |
| spark_glm52_cuda_resident_ipc.{h,c} | 634 | 1 | resident IPC protocol |
| spark_glm52_prompt_pipeline.h | - | 0 | chunked prefill dispatch |
| spark_glm52_ring_runtime.h | 252 | 0 | per-rank runtime |
| spark_glm52_mtp_tree.h | 245 | 0 | speculation tree - K3's DSpark wants this |
| spark_glm52_stage_plan.h | 225 | 0 | PP stage planning |
| spark_glm52_long_context.h | 152 | 0 | context tiering |
| spark_glm52_production_topology.{h,c} | 701 | 0 | ring topology tables |
| spark_glm52_batch_sequence_table.c | 138 | 0 | sequence slots |
| spark_glm52_row_allocator.c | - | 0 | row budget |
| spark_glm52_ring_node_context_builder.{h,c} | 434 | 0 | node context |
| spark_glm52_http_gateway / chat_template / text_prompt / compat_api | - | 0 | front door; template DATA is per-model, the engine is not |
| spark_glm52_shape_config.{h,c} | ~250 | 0 | reads shapes from a contract file - already model-agnostic mechanism, model-specific data |

Plan: rename `SparkGlm52` -> `Spark` symbol-for-symbol, move headers to
`include/sparkpipe/`, sources under `serving/` (new) or `node/`. The two real
seams get interfaces instead of renames:

  **kv-cache seam.** `spark_glm52_kv_cache.h` (506 lines, 58 hits) is the one
  genuinely glm serving header: page geometry, prefetch lanes, NVMe tiering
  sized in glm terms. Generic core (paging, tiering, prefetch machinery)
  splits from a per-model geometry struct - K3 needs the same machinery for
  MLA latent pages plus a second, fixed-size pool for KDA state (and the
  SGLang unified-pool design is the known end state).

  **tp_shard seam.** `spark_glm52_tp_shard.{h,c}` (415 lines, 0 shape hits,
  geometry caller-supplied BY DESIGN) is generic engine + a glm
  classification table (the 26 tensor-name refs). Split: `spark_tp_shard`
  engine + `model-families/<m>/` tables. `tools/k3_shard.py` is the same
  classification written pack-side in python; once the C engine takes tables,
  the python slicer stays as the CPU-gated reference and offline
  pre-sharder, and the loader path uses the C module - two mechanisms, one
  table, checked against each other by a gate.

  **expert_queue** (154 lines, 17 hits) sizes from glm expert counts -
  parameterise, low effort.

## Genuinely glm (stays, correctly named)

`spark_glm52_sm121_flashinfer_b12x_moe.h`, `spark_glm52_rope.h`,
`spark_glm52_kv_cache.h` geometry half, `spark_glm52_dspark.h` (drafter
wiring; the TREE it drives is generic mtp_tree), stagepack tensor tables,
chat template data, `spark_glm52_model.h`, everything in
`modules/glm52_resident_decode_stage/source/cuda/`.

## Dead, deleted (this commit)

`model-families/k3/` - a 2026-07-17 pre-release GUESS ledger whose constants
contradict the authoritative `inference/llms/kimi_k3/config.h` (2048 vs 3072
intermediate, 71 vs 92 routed layers). Nothing included it. A wrong-constants
header waiting to be included is the config-drift incident class, on a
timer. `model-families/dsv4/`, `model-families/qwen38_27b/` - stubs and a
work-control nothing called; the real models live in `inference/llms/`.

## Kernel-layer duplication (inference/llms)

Structural similarity after normalising model prefixes and constants:

| function | copies | similarity | verdict |
|---|---|---|---|
| Head | 5 | 0.93 | one `LmHead` template over a constants struct: candidate/commit vocab-shard sampler is model-free |
| LayerDenseMlp | 4 | 0.98 | one `LmDenseMlp` |
| LayerAttention | 4 | 0.50 | genuinely divergent (GQA vs windowed vs compressed); leave |
| LayerMoe | 3 | unmeasured | measure before touching; SiTU vs SwiGLU and latent vs direct differ for real |
| Quantise | 1 (qwen) | - | K3 deleted its own for the recipe; qwen's is its recipe - leave |
| slice.cuh pattern | 1 (k3) | - | the host-executable slice loop; when a second model needs one, extract the scaffold, not before |

## Tooling duplication (tools/)

`k3_pack.py` embeds a raw-bytes safetensors reader;
`glm52_resident_pack_common.py` has a torch-tensor one AND loads the glm
model contract at import - a side effect that makes it unimportable by any
other model's tool. Different contracts (raw vs torch), same index/shard
resolution: extract `tools/pack_common.py` with the resolution + framing,
both wrap it. B-tier.

## Order of work (all CPU-gateable this week)

1. **S - done in this commit**: dead stubs deleted; this ledger.
2. **A1 - the great renaming**: serving core `SparkGlm52`->`Spark`, headers
   to `include/sparkpipe/`, `node/*.c` symbol sweep. Mechanical, huge, zero
   behaviour: gate is the existing suite plus a grep gate enforcing the law
   (no model name in common paths; no `Spark<Model>` symbol without a
   `<MODEL>_` constant in its unit).
3. **A2 - tp_shard split**: engine + per-model tables; K3 table checked
   against `tools/k3_shard.py` by a reassembly gate.
4. **A3 - kv-cache seam**: geometry struct out of the machinery; K3 pool pair
   behind it.
5. **B - LmHead/LmDenseMlp extraction; pack_common.py; expert_queue
   parameterisation.**

The law gets a gate with A1, and this ledger stops being prose and starts
being enforced.
