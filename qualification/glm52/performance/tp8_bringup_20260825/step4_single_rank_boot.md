# STEP 4 — single-rank boot test PASS (spark8 rank0) — 2026-08-25

## Root cause of the earlier schema_error(6) @ adapter_initialize
- Deployed `model_driver.so` (built 2026-08-25 13:0x UTC on sparke from
  `~/build/glm52_unified`, workaround pkg path after publish link failure)
  carried a **shared-reader drift**: `runtime/spark_stagepack_reader.h`
  `SparkStagePackShapeWords()` stamps expected `weight_codec =
  SPARK_WEIGHT_CODEC_BF16(1)` for word tensors, while the sha-gated Aug-15
  packs store `weight_codec=0 (NONE)` on every ROUTER_CORRECTION (kind 21)
  entry (builder `tools/glm52_resident_decode_stage` add_f32 writes
  CODEC_NONE). Exact-shape compare in module PackLoad → SCHEMA_ERROR at the
  first router_correction entry (rank00 dir entry[65], layer 3).
- Packs audited offline (python struct walk, rank00 + rank06): header fv=3,
  tp=8/r, 1344 entries, bounds/alignment/inventory classes ALL clean — packs
  are ground truth; code drifted.
- Fix: `ShapeWords` sets `SPARK_WEIGHT_CODEC_NONE` (matches builder + struct
  comment "meaningful when packed"). Only glm52 builders emit F32/U32 word
  tensors → no cross-family impact.

## Method
1. Traced adapter (.so) built on sparke from glm52_unified + GLM52TRACE
   fprintf at every schema site → pinpointed LoadDriver/create.
2. Traced module compiled into model_driver.so via make variants → publish
   (GPU validator PASS per bucket) → glm52_compile_pkg.sh; deployed to
   spark8 ONLY (backups: lib/*.bringup-bak).
3. Iterations: words-codec fix alone broke BF16 entries (sed hit both shape
   helpers); restored + scoped edit to ShapeWords only → final driver sha
   dd2ba28c7d936a48ac15dd95b2d235acd85521600bdd1c669d3d1ddaed27e13f.

## Solo boot result (spark8, --rank-index 0)
```
GLM52TRACE lc members_exact rc=0
GLM52TRACE lc LoadTpCollective rc=0
GLM52TRACE LoadConfiguration rc=0 tp_degree=8 tp_rank=0 max_seq=4096 row_cap=16 speculate=0
GLM52MODTRACE init configure rc=0
GLM52MODTRACE init pack_load rc=0 path=.../packs/glm52_tp8_rank00.fp8.glms52sp
GLM52MODTRACE init caches rc=0
GLM52MODTRACE init slots rc=0
hidden_spark_rdma_fabric_ready local_host=spark8 device=rocep1s0f1 port=1 gid_index=7
```
Process stayed alive in TP-collective peer-wait (expected without the ring);
no listener yet by design. Binary/adapter/driver/pack/CUDA init all green.

Note: deployed driver+adapter carry diagnostic fprintf (init-time stderr
only). Clean rebuild queued after bring-up if time allows.
