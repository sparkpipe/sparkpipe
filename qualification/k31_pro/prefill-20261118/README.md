# Kimi K3.1 Pro long-prompt prefill, 2026-11-18

Receipts for [docs/K3_PREFILL_PLAN.md](../../../docs/K3_PREFILL_PLAN.md).

## What ran

- Release `a242ef38e`: residentd, `libk31_pro_tp16_serving_adapter.so`
  (sha16 `e430d1f72200b6f7`) and the hidden transport, with the device
  library of #1945 (sha16 `8b69d85be11a1a29`). weightd `ee01d0d5665e4464` as
  deployed. The x86 API of the release on the hub.
- All 16 Sparks, TP16, mesh lane 2, 2,048-row waves, 8 sequences, 131,072
  logical KV pages, the K3.1 pack `k31_pro.cells.tp16` (100.6 GB a rank),
  experts warmed with `weightd_warm --shared 93 896` after ready.
- Production deactivated, the fleet hold taken for each run, MemAvailable
  114-116 GiB on every node before the start and after the reclaim.
- Prompts are token ranges of the hub's long-context corpus, sent through
  `/v1/completions` with `max_tokens` 1, `temperature` 0, speculation off, the
  API's prefix index cleared at its start (no cached tokens).

| Run | Rank 0 | Prompts | 262,144-token prefill |
| --- | --- | --- | --- |
| KP1 | CUPTI activity trace (`tools/tp_cupti_trace_build.py` library), `TRACE_EXIT` present, 2,808,577 kernel and copy records | 4,096 at offset 15,031, then 262,144 at offset 0 | 656.8 s, 399.1 tok/s |
| KP2 | no tracer | 4,096 at offset 15,031, then 262,144 at offset 170,000 | 592.1 s, 442.7 tok/s |

## Pass engine runs, 2026-11-19

The same lane, firmware and hold rules, with each rank's adapter config
extended for the pass engine and restored afterwards: `streamed_prefill`
with `node_rows` 32,768, `dense_buffers` 3, `expert_slots` 28, the rank ^ 1
partner pack from `sparkdata/k31_pro.cells.tp16/partner/`, the fabric and pair
devices, GID 3 and traffic class 106 of the 2026-11-07 K3.1 record, and
`expert_pool_cap_bytes` 8 GiB. The API's `streamed_prefill_min_tokens` was
65,536. No expert warm.

| Run | Rank 0 | Prompt | Result |
| --- | --- | --- | --- |
| KQ1 | CUPTI, residentd MemoryMax 16G | none | rank 0 OOM-killed by its unit while the engine was created |
| KQ2 | CUPTI, MemoryMax 80G | 262,144 tokens at corpus offset 170,000 | one pass, 156.9 s |
| KQ3 | no tracer, MemoryMax 80G | pass-key needle, 41,600 filler units, key at the middle (998,469 prompt positions) | two passes, 764.9 s |
| KQ4 | no tracer, MemoryMax 80G, the K3.1 adapter rebuilt from `a242ef38e` with the staged index kernel (sha16 `fcea2a673f2e30a9`) on every rank, the release adapter put back after | the KQ3 needle | two passes, 687.0 s |
| KQ6 | as KQ4, adapter rebuilt from `a242ef38e` with the staged index kernel, the row-owned index and attention and the 96-head attend (sha16 `81a48b5bf34460f6`) | the KQ3 needle | two passes, 595.3 s |
| KQ7 | as KQ6 plus the sampled top-2,048 select (sha16 `ebb1911c453b4907`) | the KQ3 needle | two passes, 529.8 s |
| KQ8 | as KQ7, rank 0 under CUPTI | the KQ3 needle | two passes, 529.4 s |
| KQ9 | as KQ7 plus the tail chunks on the tile kernel, the causal-bound skip, the row-tile value projection and the FP8 option (sha16 `5743fbd4e597bf1d`), BF16 | the KQ3 needle | did not start: the adapter required `streamed_prefill.index_fp8`, and rank 3 stopped at configuration |
| KQ10 | the KQ9 adapter with `index_fp8` 1 in every rank's config | the KQ3 needle | two passes, 473.4 s, but the engine cleared the flag when it set up its buffers and scored in BF16 (`index BF16` in the log), so this is the BF16 run of the KQ9 build |
| KQ11 | the fixed adapter (sha16 `de4e0fbc5d2b0f50`), BF16 | the KQ3 needle | two passes, 475.3 s |
| KQ12 | as KQ11 with `index_fp8` 1 (`index FP8 keys and queries (ue8m0 scales)` in the log) | the KQ3 needle | two passes, 438.9 s |

## Mode-switch runs, 2026-11-19

Branch `perf/k31-pass-mode-switch` (stacked on #1950): the lane starts with
the whole arena warmed (`weightd_warm --shared 93 896` after ready, as the
TP16 runs), each rank's `streamed_prefill` names `expert_pool_cap_bytes`
8 GiB and the lane config carries no top-level cap, so the runner switches
to the slot pool for each streamed prefill and back to the whole arena
after it. Adapter sha16 `8aaef0a6b450b0a7` (release `a242ef38e` with the
#1950 exact levers and the switch). Hub script `switchrun.sh` (the passrun
steps plus the warm; it waits for the answer).

| Run | Prompt | Result |
| --- | --- | --- |
| KS1 | none | did not start: one rank's config copy failed (sparkcp), the script now retries |
| KS2 | pass-key needle, 4,200 filler units (100,869 tokens) | pass pool switch 3.9 s, prefill 110.4 s (one pass), whole arena 19.6 s (warm 14.6), answer `60669` correct, 142.9 s end to end |

- `switch-KS2-rank0.log`: rank 0's pool, spine, switch, pass and graph
  capture lines; `needle-KS2.json`: the client's result.

After each prefill the one-row decode step of the last token stopped rank 0
(`decode step key=1 rows=1 before every expert group is resident`), so the
needle has no answer; the prefill times come from rank 0's log.

- `pass-KQ2-rank0.log`, `pass-KQ3-rank0.log`, `pass-KQ4-rank0.log`,
  `pass-KQ6-rank0.log`, `pass-KQ7-rank0.log`, `pass-KQ10-rank0.log`,
  `pass-KQ11-rank0.log`, `pass-KQ12-rank0.log`: rank 0's pool, engine,
  per-pass and sequence lines, and the per-layer hidden absmax of each pass
  (identical across KQ3, KQ4, KQ6, KQ7, KQ10 and KQ11).
- `absmax-KQ11-vs-KQ12.txt`: the FP8 index against BF16, layer by layer
  (`absmax_compare.py`). The absmax is the largest value over rank 0's
  31,232 rows, logged to three digits: pass 1 moves at most 1.9%, pass 2
  (contexts to 1M) up to +15.1% at the last layer, so FP8 changes some rows'
  hidden states at long context; how much it changes answers needs decode.
- `pass-KQ2-rank0-kernels.tsv`: rank 0's GPU seconds and launches by kernel
  over the KQ2 trace, the 40 largest.
- `pass-KQ8-rank0.log`, `pass-KQ8-rank0-kernels.txt`: rank 0's lines and its
  GPU seconds and launches by kernel over the KQ8 trace (`pass_trace.py`,
  run on the hub: the prefill window, GPU busy and idle time, the gaps, the
  45 largest kernels).
- `pass-KQ8-rank0-mesh.txt`: the mesh kernels by the compute kernel before
  them (`mesh_attr.py`) and the convolution-tail waits by shard
  (`tail_waits.py`).

## Files

- `probes.jsonl`: the client's result for every prompt.
- `waves-KP1.tsv`, `waves-KP2.tsv`: every prefill wave from the API log
  (`batch_step kind=prefill`): request, rows, service and wall time, the
  context after the wave, the gap before it.
- `rank0-stages-KP1.tsv`: rank 0's GPU time in each wave of KP1. A wave runs
  from one `LmStageEmbeddingKernel` start to the next; its rows match
  `waves-KP1.tsv` one to one. Kernel intervals are merged per stage before
  they are summed, so overlap inside a stage is counted once; stages overlap
  each other on the two pair streams. `compute_ms` is the union of every
  kernel that is neither a mesh kernel nor a copy.

Stages by kernel name prefix:

| Stage | Kernels |
| --- | --- |
| collective | `SparkTpMesh*`, host-to-host copies |
| copies | other copies and memsets |
| experts | `LmCellMma*` |
| gemm | `LmStreamGemm*`, `LmGemm*`, `LmSkinny*`, `LmDense*` |
| attnres | `LmAttnRes*` |
| mla_attention | `LmLatentShard*`, `LmPerHeadProjectSplit*`, `LmLatentExpand*`, `LmLatentRope*`, `LmAttention*` |
| kda | `LmDeltaRule*`, `LmCausalConv*`, `LmL2NormalisePerHead*`, `LmBoundedDecay*`, `K3SplitFusedProjections*`, `LmSigmoidRows*` |
| index | `LmWeightedSparseScore*`, `LmIndexSelect*`, `LmIndexShard*`, `LmLayerNorm*`, `LmTopkExact*` |
| moe_route | `LmMoeFinalize*`, `LmRouteBuild*`, `LmSituMul*`, `K3RunnerLatentLogits*`, `LmTopk*`, `K3LatentSlice*` |
| rowwise | `LmAddRows*`, `K3RunnerFusedPack*`, `LmFusedResidualRmsNorm*`, `LmCopyRows*`, `LmOutputGate*`, `LmKvShardStore*`, `K3Runner*`, `LmStage*`, `StageRunner*` |
| head | `SparkLmHead*` |

The tracer cost about 11% (KP1 against KP2). Use KP2 for time and KP1 for
the split.
