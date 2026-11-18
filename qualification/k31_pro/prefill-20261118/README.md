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
