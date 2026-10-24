# Kimi K3: SparkPipe against gb10-vllm

Current as of 2026-10-24. SparkPipe's side comes from
[`PERFORMANCE_STATUS.md`](../PERFORMANCE_STATUS.md), the only current
performance ledger. gb10-vllm's side is its published `kimi-k3/v5` README
(github.com/ciprianveg/gb10-vllm, image `v5-prd`, "current production"),
rechecked on 2026-10-24 and unchanged since the 2026-09-21 audit. Its numbers
are ANNOUNCED: no receipt for them is in this tree. The 2026-09-21 audit,
written when SparkPipe had no fleet K3 number, is kept as
[`archive/K3_VS_GB10_VLLM_2026-09-21.md`](archive/K3_VS_GB10_VLLM_2026-09-21.md).

Both stacks run K3 on sixteen DGX Spark nodes at TP16.

## Numbers

| | SparkPipe | gb10-vllm v5-prd |
| --- | --- | --- |
| Speculation | lookup (suffix match, frames up to 255 drafts) plus DFlash2 model drafts from the hub's RTX 5090 | DSpark, static depth 6, fused TILE8 verify |
| Context split | KV sharded by context across all 16 ranks | DCP8 |
| B1, no speculation | 16.2 tok/s; 15.1 at 73K | not published |
| B1, code rewrite | 126.4 tok/s whole-file (best 140.6); 28.0 SEARCH/REPLACE | C1 29.81 tok/s (coding game-bench, 3,000 tokens) |
| B1, long context | whole-file rewrite 76.8 tok/s at 73K, 41.8 at 337K; reasoning prose 1.07x plain at 137.7K | tg2048 23.59 t/s at depth 4,000, 21.13 at 100K, 20.03 at 200K (coherent corpus, lower acceptance) |
| Eight streams | 63.7-63.9 tok/s aggregate (model drafts change nothing measurable) | C8 87.12 aggregate (136 peak) |
| Prefill | 590 tok/s at 8K; 137.7K in 299 s (460 tok/s) | pp2048 880 t/s at depth 4,000, 849 at 200K |
| Window | 1,048,576 tokens | not published for v5 (v3: 2M "total ctx capacity") |

Each row compares what the two sides publish, and the conditions differ:

- gb10-vllm publishes no no-speculation number, so its rates include
  speculation at its benchmark's acceptance.
- Code that a model rewrites from its own context is where lookup drafts win.
  SparkPipe is far ahead there (126 against 30 tok/s at B1).
- On prose at long context gb10-vllm is ahead. It runs 20-24 tok/s; SparkPipe
  runs about 16 tok/s, because DFlash2 drafts gain little on sampled prose.
- gb10-vllm is ahead at eight streams (87 against 64 tok/s aggregate) and in
  prefill (about 1.5-1.9 times).

## Lever by lever

### Collectives: even

gb10-vllm runs one-shot RoCE all-reduce and all-gather ("RoCEnante"), with an
init-time capability vote and an NCCL fallback above a size cap. SparkPipe
runs its own mesh transport (`ring/transport/tp_device_collective.c`): a
16-rank all-gather takes 16 µs and a 7 KB all-reduce 33-38 µs, with in-graph
device waits. Prefill uses a pipelined reduce-scatter (in production since
2026-10-22). At eight rows, decode spends about 20% of each step in
collective latency, not stragglers.

### Speculation: ahead on code and edits, behind on prose

- gb10-vllm uses DSpark at a static depth of 6 with a fused verify kernel.
  Adaptive depth (4-6) is opt-in.
- SparkPipe uses two draft sources, and the target verifies every draft:
  - **Lookup** copies drafts from the request's own context by suffix match
    and verifies up to 255 of them in one frame.
  - **Model drafts** come from the lightseek DFlash2 drafter on the hub's
    RTX 5090, fed by hidden states from five K3 layers, when lookup finds no
    match of at least 8 tokens. It has been in production since 2026-10-19.
    Draft time is 6.9 ms at 4K and 12.0 ms at 737K.
  - Verify frames are exact: drafted output equals plain decode at every
    layer (since 2026-10-21).
- The engine charges each frame its measured cost, including the draft wait,
  and backs a request off when drafting does not pay.
- Where gb10-vllm is ahead:
  - Its verify is fused into attention. SparkPipe's long verify frames still
    re-read the dense weights for every 16-row chunk.
  - On sampled prose its drafter's acceptance at depth 6 beats what DFlash2
    adds over lookup.

### Context split (their DCP): deployed, at degree 16

gb10-vllm splits decode attention by context across 8 ranks (DCP8) and merges
the partial softmax results with log-sum-exp legs over RoCE.

SparkPipe shards every sequence's MLA latent by context across all 16 ranks:
each node holds 1/16 of the KV. Each decode step, the K3 dispatch
(`modules/k3_resident_decode_stage/source/spark_k3_resident_decode_stage_cuda.cu`)
chooses one of two exchanges by a cost model:

- **Gather:** gather the keys every rank needs.
- **Scatter:** all-gather the queries, have each rank attend its own shard,
  exchange the float partials all-to-all and merge them.

The scatter is the same partial-softmax structure as DCP's LSE legs
(`K3_COLLECTIVE_MLA_PARTIALS` in `spark_k3_stage_model.cu`). It is what lets
K3 serve a 1,048,576-token window on a per-node KV pool.

The range-scan and merge kernels that the 2026-09-21 audit landed
compile-gated (`LmAttentionDecodeRangeKernel` and
`LmAttentionRangeMergeKernel` in `inference/kernels/attn.cuh`, with
`K3MlaDecodeRange` and `K3MlaRangeMerge` in `inference/llms/kimi_k3/bind.cu`)
were never wired. The shard attention replaced them, and they have no caller.

### MoE padding: not applicable

gb10-vllm patches out a Marlin pad from 192 to 256 that inflates MoE weights.
SparkPipe packs the K3 experts as MXFP4 at native width with its own GEMMs, so
there is no pad.

### Kernels: even on architecture, behind on upstream polish

gb10-vllm ports upstream vLLM kernel work: the B12X MLA attention backend,
Triton MXFP4 MoE on SM12x, and router-prefetch, KDA and low-M MoE tail
patches. SparkPipe owns its kernel stack and replays decode through CUDA
graphs:

- the GEMM slice;
- the KDA delta rule and causal convolution;
- MLA latent attention over the shard;
- tensor-core experts and shard attention for prefill.

K3 B1 reads within 4% of its ideal bytes per token and runs at 58% of the
floor for the bytes it reads.

### Dynamic FP8 at load: behind

gb10-vllm converts the dense linears to MXFP8 and the shared experts to FP8
per block at load, keeping attention, lm_head and the gate in higher
precision. Its README credits this with about a third of its v4-to-v5 gain.

SparkPipe streams a BF16 spine with MXFP4 experts. An FP8 spine is the largest
lever SparkPipe does not have; it would cut the dense bytes per verify row in
half. It needs a new pack format and an accuracy A/B on the BF16 spine before
it can serve.

### Stability guards: even

gb10-vllm's fixes (NaN-gumbel, multistream record-stream, grammar fences,
draft-noeplb) target vLLM runtime hazards. SparkPipe's equivalents are the
exact-key chain waits, the QP repair, the dispatch gate and the DMA guard.

## Where SparkPipe goes next on K3

- Prefill and eight-stream throughput, where gb10-vllm leads.
- Fused or chunk-free verify for long frames.
- An FP8 spine, gated on an accuracy A/B.
- A no-speculation and B1-to-B8 sweep at the full window from one cached
  prefix. It needs requests that share a cached prefix to reserve its pages
  once (PR #1542).
