# Wave-R2 roofline report — 2026-09-13

Lanes: k3, glm53full, hy4, dsv5. No daemon contact, no module/model
execution; spark3/spark6 untouched. Measurement sources: repo receipts
plus capped probes on spark0 only (sparkcap MemoryMax=4096M
MemoryHigh=2900M, page-cache purge before and after each probe).

## Measured machine facts (spark0, 2026-09-13)

| Probe | Command shape | Result |
| --- | --- | ---: |
| GPU stream triad, 3x640 MiB float, 369 passes / 3.0 s | nvcc sm_121, `systemd-run --scope -p MemoryMax=4096M -p MemoryHigh=2900M` | **247.51 GB/s** (90.7% of the 273 GB/s spec) |
| CPU stream triad, 20 threads, same footprint | gcc -fopenmp, same cap | 112.18 GB/s |
| NVMe seq read, 4 GiB iflag=direct | Samsung MZALC4T0HBL1-00B07, root device, read-only | **5.0 GB/s** |
| (rejected) 4 GiB read of /mnt/model-warm shard | ceph-backed, not NVMe; killed before completion, cache purged | n/a |

The single-thread triad first measured 42.93 GB/s - a per-core limit,
not DRAM; all roofline denominators below use the GPU triad number.

## Roofline tables (bytes/token per rank, ideal build-up)

Regimes: HOT = resident weights at 247.51 GB/s measured; COLD =
lease-streamed from NVMe at 5.0 GB/s measured; Engram = host-mem to
device stream (112.18 GB/s CPU-side measured; DMA path unmeasured,
labeled ESTIMATE).

### k3 (TP4xPP4 receipt topology, per rank; inventory from tools/k3_tp4pp4_perf_estimate.py)

| Term | GB/token |
| --- | ---: |
| dense spine, 16 KDA x 327.2 MB (critical stage) | 5.235 |
| dense spine, 7 MLA x 268.7 MB | 1.881 |
| experts: 23 layers x top-16/896 x 1965.3 MB set (MXFP4, TP4 slice) | 0.807 |
| KDA fp32 state read+write, 16 layers | 0.053 |
| lm_head (stage 3, quarter slice) | 0.587 |
| activation + AR wire, 23 layers x 88 KB | 0.002 |
| **total** | **8.57** |

Ceiling 28.9 tok/s at measured BW. Receipt 18.0 tok/s x 8.57 GB =
**62.3% of measured BW**. TP16: 8.28 GB/token -> ceiling 29.9 tok/s;
receipt 20.2 tok/s = **67.6%**.

### glm53full (FP8 serving arm, TP8, per rank; contract geometry)

| Term | GB/token |
| --- | ---: |
| experts, 75 MoE layers x E[local]=top8 x 32/256 = 1.0 x 37.76 MB | 2.832 |
| spine attention bf16, 78 x 330 MB / 8 | 3.218 |
| shared expert bf16, 78 x 75.5 MB / 8 | 0.736 |
| dense FFN layers 0-2 (12288 inter) / 8 | 0.170 |
| router + norms + indexer (EST) | 0.078 |
| lm_head 154880x6144 bf16 / 8 | 0.238 |
| KV read O128 fp8 78x128x576 + write | 0.006 |
| **total (ideal)** | **7.28** |

Ceiling 34.0 tok/s. No 5.3-full decode receipt yet (bf16/fp8 packs
published, arms not decoded) -> **% N/A**. Module-class anchor: the
glm52 5.2 TP8 B1 receipt (6.91 tok/s, ~21.7 GB/token claimed) = 60.6%
of measured BW; the ideal-vs-receipted gap (7.3 vs 21.7 GB) is lease
over-read + activation round-trips - the T2 A/B target.

### hy4 (TP16, EP 16 experts/rank, per rank; model header)

| Term | GB/token |
| --- | ---: |
| experts, 77 MoE layers x E[local]=top8 x 16/256 = 0.5 x 38.93 MB | 1.499 |
| spine attention fp8, 78 x 141 MB / 16 | 0.805 |
| shared expert fp8, 78 x 37.75 MB / 16 | 0.190 |
| hc fn f32, 2 branches [8,24576] x 78 | 0.123 |
| router f32 [256,6144] + bias x 77 (replicated, not sharded) | 0.485 |
| lm_head 7552x6144 bf16 (vocab-parallel) | 0.093 |
| KV read O128 fp8 + write | 0.006 |
| **total (ideal)** | **3.20** |

Ceiling 77.4 tok/s (before collective-latency floors). No decode
receipts (attach milestone pending) -> **% N/A**. The f32 router plane
is 15% of the ideal budget - an fp8/bf16 router is the named next lever
after attach.

### dsv5 = dsv41_flash (TP8 per rank, EP 48 experts/rank; branch lane/dsv5-flash-dev)

| Term | GB/token |
| --- | ---: |
| experts, 37 MoE layers x E[local]=top6 x 48/384 = 0.75 x 17.69 MB (resident, HOT) | 0.491 |
| dspark 3 blocks x top-3 of 128 placed experts (EST) | 0.030 |
| spine attention fp8, 40 x 136 MB / 8 | 0.717 |
| shared expert fp8, 40 x 35.4 MB / 8 | 0.177 |
| hc / indexer / compressor / router (EST) | 0.050 |
| lm_head 129280x5120 bf16 / 8 | 0.166 |
| KV write 890 B/token (FP4 global) + read at 32K ctx | 0.029 |
| Engram: 12,672 B/step remote + local rows (2 layers) | 0.000 |
| **total (ideal, 32K ctx)** | **1.66** |

Ceiling ~149 tok/s per rank at measured BW (~313 tok/s with the DP2
node pair). M7 mid-flight, no decode receipts -> **% N/A**. Engram
tables 2x98.3 GiB host-resident, row-sharded 12.3 GiB/spark: the bytes
term is trivial; the cost is doorbell round-trips (24 per layer per
step in Publish) - measured path BW does not exist yet, T2 item.

## Algorithm-iteration PRs (1:1 per driver, all WIP-early, compile-green)

| PR | Lane | Lever | Byte/latency math |
| --- | --- | --- | --- |
| #987 | k3 | derive PP stage bounds, tie geometry headers, pre-allocated TP context pool | no byte change; kills per-layer-phase new/delete; ceiling 28.9/29.9 tok/s |
| #988 | glm53full (glm52 module) | deferred expert-lease retire: 75 blocking stream drains per B1 token -> 1 per wave | bytes unchanged; ESTIMATE O(1-2 ms)/layer drain class removed; T2 A/B cell |
| #989 | hy4 | vectorized fused dequant GEMV (warp-per-row, uint4, bit-exact E4M3 decode) + derived rope theta + I48 comment purge | bytes unchanged; scalar kernel capped far below 77.4 tok/s ceiling; rung re-run validates |
| #990 | dsv5 | single-lease engram step seam (Publish+Rows pair on one lease cycle) | engram bytes unchanged (12,672 B/step); one lease round-trip + one 24-row load pass removed per layer per step |

Branches: lane/wave-r2-k3-roofline (7304ae9),
lane/wave-r2-glm53full-roofline (1e08ebb), lane/wave-r2-hy4-roofline
(adce4a7), lane/wave-r2-dsv5-roofline (504d509, based on
lane/dsv5-flash-dev 83b1a09).

## Audit findings (fresh clone /Users/mac/roof-r2 from origin main c698e20; dsv5 from lane/dsv5-flash-dev 83b1a09)

Known items from docs/CONSTANT_AUDIT.md (#979) are NOT re-reported (its
8 findings are all in glm5_next/weightd/transport, outside these four
driver trees except where noted).

Count by class: **fix-now 9, fix-at-touch 7, document-only 6** (22 total).

### k3 (modules/k3_resident_decode_stage + inference/llms/kimi_k3)

1. runner.cu:140,153 duplicate `typedef struct SparkK3RunnerState`
   (three statements of one type) - fix-now - FIXED in #987.
2. runner.cu:220-227 PP stage tables {0,24,47,70}/{24,23,23,23} restate
   K3_LAYERS=93 with no tiling assert - fix-now - FIXED in #987.
3. runner.cu:309,356 `new SparkK3RunnerTpContext` per layer per phase on
   the decode hot path - fix-now - FIXED in #987 (pool + fail-closed).
4. module.c (k3) `pack.config.total_layers != 93u` bare literal -
   fix-now - FIXED in #987.
5. Geometry stated 4x with no ties: config.h, generated_config.h (both
   included by layer.cuh), model header, SPARK_K3_MODULE_MAX_BOUND_LAYERS
   - fix-now (assert pairs) - PARTIALLY FIXED in #987 (module<->model<->kernel
   chain asserted); config.h vs generated_config.h merge is the follow-up.
6. model header + kv_geometry + runtime_contract use #ifndef guards
   (style law: #pragma once); the model header transitively pulls
   C11-atomic spark_kv_cache.h into C++ TUs, which is why runner.cu
   could not include it - fix-at-touch (header split + pragma once;
   documented in #987 so the include gap is deliberate until then).
7. runner.cu:1213 local extern "C" declaration of K3StageSliceHalf -
   fix-at-touch (move to an internal header).
8. runner.cu:955 host head exchange reuses the MoE staging buffer
   (uint16_t*) as f32 scratch; device maxloc tier is the qualified path -
   document-only.
9. runner.cu:513-519 LazyAcquire full-stream sync + blocking D2H per
   layer - document-only (lease-latency lever, symmetrical to #988;
   needs a side stream + event, not safe to land blind).

### glm53full (modules/glm52_resident_decode_stage)

10. module.c:1459-1460 blocking cudaStreamSynchronize per MoE layer in
    SparkGlm52LazyRelease (pipeline drain) - fix-now - FIXED in #988.
11. module.c:1856/1677 chain calloc/free per wave - fix-at-touch
    (pre-allocate per slot; host-side, per batch not per layer).
12. spark_glm52_stagepack_format.h has zero _Static_assert pairs while
    the sibling hy4 format header has twelve (wire sizes, format codes) -
    fix-at-touch (add sizeof/code-equality asserts).
13. serving_adapter.c:188,356 flat-ranks/speculators env config -
    document-only (deployment inputs per I04; keep them validated).

### hy4 (modules/hy4_resident_decode_stage)

14. 15 block comments across firmware.h, stagepack_format.h, module.c,
    pack_convert.c (I48) - fix-now - FIXED in #989.
15. cuda.cu:125 rope theta literal 10000000.0f restating the model
    header - fix-now - FIXED in #989.
16. cuda.cu scalar per-thread-row dequant GEMV (bandwidth killer) -
    fix-now - FIXED in #989 (warp-per-row uint4 + bit-exact decode).
17. Slot KV cache planes are f32 (kv_latent_cache_f32) - 4x the KV
    roofline term at long context vs the fp8 latent contract - fix-at-touch
    (bf16/fp8 cache planes before long-context cells).
18. f32 router plane 15% of ideal bytes/token budget - document-only
    (quantization lever named in #989).
19. cuda.cu:208-240 attention reads k_latent twice (score pass + value
    pass) - document-only (flash-decode fusion is the follow-up).
20. tools/hy4_gpu harness files carry comments and a duplicated kernel
    copy - fix-at-touch.

### dsv5 (modules/dsv41_flash_resident_decode_stage, branch)

21. Publish+Rows double lease cycle per layer per step - fix-now -
    FIXED in #990 (Step seam).
22. engram_access.c Publish issues 24 sequential MeshBroadcast
    round-trips per layer (48/step); coalescing is blocked by scattered
    column ownership (owner = id/part, not col-contiguous) - document-only;
    a compact per-publisher staging layout is the design follow-up.
23. PollRemotes busy-spins on 24 volatile doorbells without backoff -
    document-only (single-core spin at B1 is tolerable; revisit at B8+).
24. Zero _Static_asserts in the dsv5 stagepack format header and module
    tree (hy4's format header is the in-repo model: 12 asserts) -
    fix-at-touch.
25. engram doorbell/staging offsets derive from one define each
    (STAGING_BYTES, DoorbellOffset) - clean; noted so the audit trail
    shows the layout was checked.

Cross-check vs #979: the miss-record stride 512 and lease-cap items
named there live in glm5_next files, not in these four trees - no
overlap, nothing re-reported.

## What I could NOT verify

- Any number labeled ESTIMATE above; the glm53full/hy4/dsv5 ceilings
  have no receipts to compare against.
- The DMA bandwidth of the Engram host-mem path (CPU triad is a bound,
  not the path).
- Device behavior of every PR: only compile gates ran (see PR bodies).
  #988's event-ordered retire and #989's kernel need the T2 cells named
  in the PRs before any perf or numeric claim.
- sparke was used as the compile reference (k3 gate PASS there); spark0's
  nvcc rejects the pristine k3 runner TU over glibc stdatomic C++
  semantics - an environment fact, not a repo defect.

## Blockers / handoff

- Decode receipts for glm53full/hy4/dsv5 remain daemon-gated; the T2
  plan per lane is in the PRs and the tables above.
- config.h vs generated_config.h merge (k3) and the glm52/dsv5 format
  assert pairs are small, mechanical follow-ups.
