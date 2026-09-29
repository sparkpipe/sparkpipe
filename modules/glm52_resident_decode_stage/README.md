# GLM52 resident decode-stage firmware

This directory is an exact model-specific CUDA firmware link unit. It is not a generic serving backend and it is not a reusable graph interpreter.

The fixed stage program is specialized for SM121 GLM 5.2 BF16 decode with 6144 hidden elements, 64 MLA heads, 512 latent elements, 64 adjacent-pair RoPE elements, 2048 selected context tokens, 64-token KV blocks, a 256-token restricted vocabulary head, and depth-2 MXFP4 MTP draft verification.

One submission executes this stream-ordered sequence:

```text
attention RMSNorm
BF16 Q latent / Q RoPE / K RoPE / KV latent projections
driver-owned sparse-token selection
RoPE + final-layout current KV write
resident sparse MLA attention
attention output projection + residual
final RMSNorm + restricted-vocabulary logits + argmax
MXFP4 E2M1/E8M0 MTP draft logits + argmax
MTP verify / commit / rollback counters
BF16 dense MLP layer progression for GLM 5.2's first dense layers
external completion
```

The live layer-0 dense gate is:

```sh
GLM52_MODEL_DIR=/home/spark1/models/hf/nvidia/GLM-5.2-NVFP4 \
PATH=/usr/local/cuda-13.0/bin:$PATH \
make -C modules/glm52_resident_decode_stage validate_layer0_dense_bf16 MAX_STAGE_MICROSECONDS=10000
```

The package-level gate, which uses a distinct validation recipe and then runs
the generated driver/orchestrator path, is:

```sh
GLM52_MODEL_DIR=/home/spark1/models/hf/nvidia/GLM-5.2-NVFP4 \
PATH=/usr/local/cuda-13.0/bin:$PATH \
make -C modules/glm52_resident_decode_stage package_layer0_dense_bf16 MAX_STAGE_MICROSECONDS=10000
```

The stronger combined layer-0 BF16 gate is:

```sh
GLM52_MODEL_DIR=/home/spark1/models/hf/nvidia/GLM-5.2-NVFP4 \
PATH=/usr/local/cuda-13.0/bin:$PATH \
make -C modules/glm52_resident_decode_stage package_layer0_bf16 MAX_STAGE_MICROSECONDS=10000
```

That gate loads the real BF16 attention tensors, `post_attention_layernorm`, `gate_proj`,
`up_proj`, and `down_proj` tensors for layer 0, runs the dense layer body on
device, and then checks restricted logits against real checkpoint
`lm_head.weight` rows.

The strongest current B1 layer-0 gate also replaces the synthetic input hidden
with one checkpoint embedding row:

```sh
GLM52_MODEL_DIR=/home/spark1/models/hf/nvidia/GLM-5.2-NVFP4 \
GLM52_INPUT_TOKEN_ID=1000 \
PATH=/usr/local/cuda-13.0/bin:$PATH \
make -C modules/glm52_resident_decode_stage package_layer0_embedding_bf16 MAX_STAGE_MICROSECONDS=10000
```

That gate loads `model.embed_tokens.weight[token]`, the layer-0 attention
tensors, the layer-0 dense tensors, and restricted `lm_head.weight` rows, then
runs the package/generated-driver path. It still seeds the previous KV cache;
the next correctness gate is checkpoint-derived prefill/KV and reference
activation comparison.

The current strongest B1 layer-0 gate removes that seeded prior-KV assumption
for the checked context window:

```sh
GLM52_MODEL_DIR=/home/spark1/models/hf/nvidia/GLM-5.2-NVFP4 \
GLM52_INPUT_TOKEN_ID=1000 \
PATH=/usr/local/cuda-13.0/bin:$PATH \
make -C modules/glm52_resident_decode_stage package_layer0_prefill_bf16 MAX_STAGE_MICROSECONDS=10000
```

That gate runs prior embedding rows through the same resident CUDA stage to
populate remapped KV cache slots, restores the final input embedding row, and
then checks the final attention result against the actual cache rows written by
prefill. It still is not full GLM token equivalence; the next gate needs
external reference activation comparison for the checkpoint-backed layer body.

The strongest current sampled-reference gate is:

```sh
GLM52_MODEL_DIR=/home/spark1/models/hf/nvidia/GLM-5.2-NVFP4 \
GLM52_INPUT_TOKEN_ID=1000 \
PATH=/usr/local/cuda-13.0/bin:$PATH \
make -C modules/glm52_resident_decode_stage package_layer0_reference_bf16 MAX_STAGE_MICROSECONDS=10000
```

That gate keeps the prefilled KV setup and adds sampled CPU references for the
checkpoint-backed layer-0 projection, RMSNorm, residual, and dense MLP
boundaries. It intentionally stays validator-local; the model driver ABI does
not learn GLM tensor names or reference math.

The stronger layer-0 output-side full-reference gate is:

```sh
GLM52_MODEL_DIR=/home/spark1/models/hf/nvidia/GLM-5.2-NVFP4 \
GLM52_INPUT_TOKEN_ID=1000 \
PATH=/usr/local/cuda-13.0/bin:$PATH \
make -C modules/glm52_resident_decode_stage package_layer0_full_reference_bf16 MAX_STAGE_MICROSECONDS=10000
```

That gate preserves the sampled q/kv/gate/up coverage, then checks every
output element of `o_proj`, the attention residual, post-attention RMSNorm,
SwiGLU activation, and dense-down residual against validator CPU reference
math. It still does not replace an external activation artifact for full-model
equivalence.

The same full-reference gate can be run over the other dense pre-MoE layers:

```sh
GLM52_MODEL_DIR=/home/spark1/models/hf/nvidia/GLM-5.2-NVFP4 \
GLM52_INPUT_TOKEN_ID=1000 \
PATH=/usr/local/cuda-13.0/bin:$PATH \
make -C modules/glm52_resident_decode_stage package_layer1_full_reference_bf16 MAX_STAGE_MICROSECONDS=10000

GLM52_MODEL_DIR=/home/spark1/models/hf/nvidia/GLM-5.2-NVFP4 \
GLM52_INPUT_TOKEN_ID=1000 \
PATH=/usr/local/cuda-13.0/bin:$PATH \
make -C modules/glm52_resident_decode_stage package_layer2_full_reference_bf16 MAX_STAGE_MICROSECONDS=10000
```

These targets load `model.layers.1.*` and `model.layers.2.*` respectively,
matching the live GLM config's `first_k_dense_replace=3`. They are per-layer
checks; they do not yet chain hidden activations through layers 0, 1, and 2.

The node context binds resident weight pointers, paged KV cache, streams, workspaces, RoPE tables, token maps, and output buffers once when the driver instance is created. Per-submission inputs are only dynamic decode facts such as active sequence count, requested token count, sequence identity, deadline, priority, and residency token. The firmware admission function chooses the opaque pipeline slot; SparkPipe does not assign or interpret CUDA stream/KV ownership.

The module also publishes direct admission and snapshot symbols. They expose only neutral scheduling data: accepted/rejected, dispatch slot, dispatch generation/cookies, private queue pressure, resident token capacity, active submissions, CUDA graph capture/replay counts, stale-admission count, and zero memcpy/host-staging counters.

Sparse-token selection has three modes. `preselected` is the intended production path: DSA or another model-specific policy produces sparse rows before graph replay. `copy_context_prefix` is a parallel bring-up path. `debug_serial_topk` is deliberately not a production mode.

Each pipeline slot may hold one captured CUDA graph for its fixed active-sequence shape. Production launch checking should be disabled; peek/sync modes exist for target bring-up. Completion is enqueued after the graph launch so the arithmetic graph stays reusable and the completion mechanism can later move to event polling or a doorbell without changing the captured graph.

Normal publication validates a new archive exactly once:

```sh
make -C modules/glm52_resident_decode_stage publish \
    CUDA_ARCH=sm_121a \
    MAX_STAGE_MICROSECONDS=<qualified-limit>
```

The source is correctness-first until hardware profiling says which fused pieces should be replaced by tensor-core or persistent-kernel implementations. It must not be published unless the hardware validator passes the numerical checks and the maximum full-stage submission-to-completion latency ceiling.

## Lazy consumer and serving adapter

Moved from `docs/WEIGHTD_DESIGN.md` on 2026-09-28; the shared weightd contract
stays there.

LAZY CONSUMER: the lazy pack qualifies the FP8 codec only; BF16 keeps the
resident eager load. Routed experts stay in the arena's sparse address space
and materialize through per-wave acquisition. The module retains pack offsets
for lease binding, while non-expert tensors come from the compact spine.
Kernel-side, expert pointers are consumer-local leased VMM addresses; the
weightd map exposes only acquired extents. Route results publish to host
storage (event plus pinned host mirror) only for slots wired for lazy
acquisition; resident and validator slots skip it. Retained lazy chains
awaiting lease recovery are keyed by pipeline slot in the module state.

FLAT RANKS: the adapter exposes FLAT_RANKS flat ranks, one per TP rank, in a
single PP stage. residentd fans each submission out to every rank
(PARALLEL_FANOUT) and the firmware stage stays STAGE_COUNT=1; the adapter maps
flat rank to tp_rank and pins the firmware stage to 0. The rank count is a
per-deployment environment selection (`SPARK_GLM52_SERVING_FLAT_RANKS`, 8 or 16),
so one adapter artifact serves TP8 and TP16 while each keeps its own adapter
identity. ValidateForAdapter pins deployment node_count == stage_count and the
stage configs' tp_degree == TP_DEGREE. Unset, empty or malformed values leave
the descriptor unconfigured, and the host's adapter-load validation fails
closed.

LAUNCHER: `tools/fleet_serve.sh` forwards `SPARK_GLM52_SERVING_FLAT_RANKS` to
residentd verbatim, with no default. It is a legacy manual launcher: it starts
residentds over SSH outside the fleet agent, defaults the API host to spark0,
and sets none of the fleet's serving variables. Do not run it on a root the
fleet agent manages (`docs/FLEET_RELEASE_RUNBOOK.md` §4).

DRIVER MODEL ID: the expected DRIVER model id must equal the model.id of the
firmware the driver was compiled from (ServingAdapterTemplateLoadDriver compares
them). The bf16 arm's firmware pins the 5.3-full identity; every other codec's
firmware keeps the 5.2 identity. GLM52_EXPERT_WEIGHT_CODEC is a numeric define.

## GLM-5.3 Full at TP16 (lane tools and expert residency)

GLM-5.3 Full (`GlmMoeDsaForCausalLM`) is served by this module at TP16 from
the `glm53full.fp8.tp16` packs. `tools/glm53full_lane_build.sh` builds a
bucketed firmware on a Spark, `tools/glm53full_lane.py` renders the
deployment and sixteen adapter configurations for one weightd lane
(`tests/test_glm53full_lane.py` pins them to the adapter's member set and the
build identity), and `tools/glm53full_lane.sh` stages, starts, stops and
serves the lane. The packs must carry the current contract digest;
`tools/glm52_pack_restamp_contract.py` rewrites a stale digest when the
contracts differ only in identity, after checking the pack against its
sidecar.

Expert residency has two modes:

- Lazy (default with `SPARK_WEIGHTD_ATTACH_LAZY=1`): each routed layer leases
  its experts through weightd. With a per-chunk pool (pool smaller than the
  pack) every lease maps and unmaps its chunks, about 20-40 ms per layer, so
  B1 is well under 1 token per second. It fits beside other lanes.
- Pinned (`SPARK_GLM52_PIN_EXPERTS=1`): with an expert pool larger than the
  pack, weightd premaps the whole arena once; the module then leases every
  routed expert at attach (`spark_module_pin_experts.h`, 512 keys per lease)
  and binds all layers to that base, so decode never calls weightd. It needs
  about 50.4 GiB of arena per rank for fp8. Attach prints
  `EXPERT-RESIDENCY mode=pinned keys=19200 leases=38`, or
  `EXPERT-PIN-FAILED` and fails startup.

Serving rules for the lane:

- A residentd serves one client at a time and resets the adapter on every
  new client's hello. The module answers the reset admission
  (`SPARK_MODEL_DRIVER_ADMISSION_FLAG_RESET`) through the shared
  `spark_module_reset_page_cache.h`: it claims every pipeline slot and
  sequence lane (BUSY while one is in flight), drains the execution stream,
  releases every page-cache lane with `SparkKvPageCacheReleaseAll` and
  unbinds the lanes. Before this the admission fell through to the shape
  check, was rejected, and every rank exited `status=9` on the second client
  (`tests/test_module_page_cache_reset.py`, which also covers ling, whose
  module had the same gap). A stream failure during the drain returns
  IO_ERROR and keeps the slots and lanes claimed.
- `tools/glm53full_lane.sh api`, `api-stop` and `decode` run only on the
  rtx5090. Any other `GLMFULL_API_HOST` is refused before a remote command
  runs (`tests/test_glm53full_lane.py`). Measure on the fleet with
  `sparkpipe_model_batch`, which the lane build stages next to the residentd.
- The model's EOS ids always stop a request (invariant I49); an empty
  `stop_token_ids` does not disable them. Fixed-length decode measurements
  need a prompt that does not reach EOS within the budget, and a runner that
  refuses a case whose token count is below its budget.
- COMPSEC-17 uses the checkpoint's own chat template:
  `tools/glm53full_compsec17.py render` gives
  `[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>{question}<|assistant|><think>`,
  plus `</think>` for thinking off. The committed prompts are
  `qualification/glm53full/compsec17_prompts_{off,on}.json`
  (`tests/test_glm53full_compsec17.py`). The GLM-5.3 Flash framing
  (`<|user|>\n...<|assistant|>\n<think></think>\n`, no system turn) is not
  the Full template.

## Chain modes (`SPARK_GLM52_CHAIN_MODE`)

The module runs one token step as a chain: the embedding reduce, then per
layer attention, a hidden all-reduce, the MLP and a second all-reduce, then the
head and its max-loc reduce (158 collectives per token at 78 layers).

- `eager` (default): every collective ends in a host completion callback that
  launches the next stage. Required for `SPARK_GLM52_T1` traces and for lazy
  expert leases (the route has to reach the host before the experts launch).
- `linear`: `SparkGlm52RunChain` enqueues the whole chain, every wave of the
  batch, from the submitting thread. Collectives are stream ordered with no
  host completion (the device waits on the mesh); the lazy-pack worker then
  waits for the stream, checks the deferred rounds
  (`SparkTpDeviceCollectiveVerifyDeferred`) and completes the slot.
- `graph`: as linear, but a single-wave step is captured once per slot, row
  count and attention regime, then replayed. The captured context bound is
  the largest context of its regime (`spark_glm52_graph_regime.h`: unsplit
  below `decode_split_context_threshold`, split up to the 2048 selected
  tokens), so a replay launches the same kernels eager launches for any
  context of that regime. Steps the graph does not cover run linear and log
  `GLM52-GRAPH-GATE` with the reason (`multi-wave` prefill waves,
  `selected-context` above 2048 tokens, `rows` above 64); the counts appear in
  every `GLM52-CHAIN-TIME` line.

`linear` and `graph` refuse to start (`GLM52-CHAIN-MODE-REFUSED`) without the
lazy attach worker, pinned experts (`SPARK_GLM52_PIN_EXPERTS=1`), stream-ordered
collectives (`SPARK_TP_WAIT_MODE=hardware`) or with `SPARK_GLM52_T1`. One chain
is in flight at a time; a second submission returns BUSY until the first has
settled. If the lazy worker refuses the settle, the submitting thread settles
the chain itself and logs `GLM52-CHAIN-SETTLE-INLINE`. The shared pieces (mode parse, arm/disarm, capture, pre-launch seed,
settle) are model-neutral in `include/sparkpipe/spark_tp_chain_graph.h`.

## Split q_a/kv_a projections (`SPARK_GLM52_PROJECTION_SPLIT=1`)

q_a (2048 x 6144) and kv_a (576 x 6144) are replicated on every rank and were
read in full every token, 2.34 GiB per token per rank. With the split each
rank reads an 8-element-aligned slice (q_a 128 rows at TP16; kv_a 40 rows on
ranks 0-7 and 32 on ranks 8-15), writes it into a zeroed gather row, and one
bf16 sum all-reduce per layer assembles q_compressed and the KV latent before
their norms (`GlmLayerAttentionProject`, reduce, `GlmLayerAttentionCore`).
Each gathered element has exactly one nonzero contributor, so the sum is the
replicated projection. It adds 78 collectives per token and removes about
2.2 GiB of reads per token per rank.

## Single-GPU parity

`validation/run_glm52_chain_graph_parity.sh fp8 16` builds the b16 archive and
`glm52_chain_graph_parity`, which walks a dense and a routed layer for 96
decode steps (split threshold 64) with the validator's synthetic weights, at
TP1 and at TP16 rank-local shapes, as staged launches (a host sync after every
stage), a linear walk, replayed graphs, and the projection split with the
16-rank gather emulated on one GPU. Hidden and residual outputs must be
bit-identical to the staged walk. Two controls must differ: every step replayed
with the other regime's bound, and the split gather missing one rank.
`tests/test_glm52_chain_modes.py` drives the real chain runner on the host:
walk order, capture and replay per regime, the selected-context and multi-wave
gates, settle, walk and stream failures, a refused worker, and the busy gate
(including its release when a submission fails before the chain starts).

## Multi-row prefill waves (`SPARK_GLM52_PREFILL_WAVE_ROWS`)

Without the variable (or with `0`) a prefill submission runs one round per
wave: a single prompt of 16 rows is 16 full 78-layer waves, so a 300-token
prompt costs about 300 B1 steps. With `SPARK_GLM52_PREFILL_WAVE_ROWS=N`
(1 up to the execution row capacity; any other value refuses startup) a
prefill wave spans consecutive whole rounds up to N rows. The rows of one
sequence in a wave are causal: every layer stores the wave's KV and DSA index
rows before attention, and attention, index scoring and top-k mask positions
after each row's own position.

A wave never mixes attention regimes. The regime of a round is the largest of
its rows' regimes, where a row's regime is its graph regime (split or unsplit
attention at `decode_split_context_threshold`) plus whether its context passes
the 2048-token DSA selection width. The batch's final round is always a wave
of its own, so the token a prefill emits comes from the same head path (the
certified B1 head for a single sequence) as before.

The startup line `GLM52-PREFILL-WAVE-ROWS rows=N exact_rows=8 exact=yes|no`
names the bound. Up to 8 rows every linear takes the skinny path and each row
is bit-identical to one-row prefill; above 8 rows the linears take the GEMM and
the rows are not bit-equal, which the line states.

Two kernel changes make multi-row waves exact:
- `LmLatentAttentionDecodeSplitLaunch<..., true>` sizes the split partitions
  per row (from the head count, as for one row) and bounds each row's span by
  its own position, so a row's attention sums do not depend on the other rows
  in the wave. At one row per sequence this is the previous launch.
- The glm52 wave metadata kernel sets each sequence's context length to the
  maximum over its rows in the wave (the previous kernel let the last writer
  win).

The DSA indexer stores every position's index key, also while the context is
within the selection width (the query projection, scoring and top-k still run
only past it). Before this, positions 0..2047 of every sequence had no index
key, so any sequence that grew past 2048 tokens selected from zero keys.

`validation/run_glm52_prefill_rows_parity.sh fp8 16` builds
`glm52_prefill_rows_parity`: on the validator's synthetic weights (a dense and
a routed layer, TP1, split threshold 64) it prefills 2112 positions one row per
wave as the reference, then with waves of up to 2, 4, 8 and 16 rows, with and
without the regime split, and compares every row's hidden and residual, the KV
pool and the DSA index pool. It then runs verify waves of up to 8 rows at
anchors in the unsplit, split and selected regimes on the reference cache: an
oracle wave (reference tokens) must be bit-exact, an adversary wave (wrong
draft tokens) must leave the anchor row equal and change every drafted row,
and a one-row replay after the adversary must equal the reference. The run
also prints FNV hashes of the reference rows before and after 2048 for a
comparison against another build.
