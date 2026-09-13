# SparkPipe serving charter — from driver to system (2026-08-23)

Charter: make sparkpipe a USEFUL serving system — arbitrary requests in
through a standard API, optimally shaped internally, at B1024 and 256k
context — not just a fast driver with a private IPC.

## Where we are (measured, release qwen38-dflash2-20260821 + fixes)

| Capability | State |
|---|---|
| Single-stream serving, DFlash2 speculation | 24.5 tok/s wall, ~28 decode-only, deterministic bit-identical streams |
| Prefix caching (hybrid GDN+KV) | bit-identical repeat outputs, ~all prefill skipped |
| Robustness (session death, concurrent clients, dump envs) | fixed + logged (PRs #682-#684) |
| Batch B>1 | WORKS but prefill inefficient pre-fix, decode unoptimized |
| API | HTTP skeleton exists (model_api); completion path has one known bug |
| B1024 / 256k | NOT supported today — see the link-by-link audit below |

## The B1024 / 256k audit (what breaks, today, in order)

1. Deployment JSON: `max_active_sequences: 64`, `max_input_rows: 128`,
   `kv_logical/physical page capacities` — must rise to 1024 (capacity
   arithmetic, config change).
2. Module build: `MAX_ACTIVE_SEQUENCES=8` sizes the GDN state pool, lane
   arrays, slot pools per lane. At 1024 lanes the GDN state pool alone is
   1024 x 48 layers x state — must be sized by residency, not lane count.
3. Adapter: lane-indexed arrays (`host_block_indices` per lane, prefix
   bookkeeping per lane) — fine at 1024 but resident capacity must mean
   RESIDENT (admitted), not CONFIGURED.
4. KV pool: 8192 blocks x 64 tokens = 524k tokens TOTAL. B1024 x 256k =
   268M tokens - 512x the pool. THE HARD LIMIT: B1024 at 256k context
   cannot be resident simultaneously. Requires paged KV with block
   eviction/restore (the KV store/service config plumbing exists:
   KV_STORE/KV_SERVICE/KV_POOL_BYTES build vars, unused today).
5. max_sequence_positions: 8192 (adapter config) → 262144; position
   arrays, taps history (2048 window), dflash ctx caches sized by
   context — several grow linearly and need re-sizing or windowing.
6. Engine scheduler: lane arrays capped at
   SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT (compile-time);
   the spec path is B1-gated; decode batching at high lane counts is
   untested beyond B16.
7. The no-spec path is the B>1 vehicle (plain batched decode, one frame
   for all lanes) - at B1024 an M=1024 decode frame is
   compute-favorable, but attention context sums across lanes and the
   2-submission inflight pipeline needs deepening.

REALITY CHECK: B1024 with bounded aggregate residency (e.g. 1024 lanes x
small context, or 64 lanes x 256k) is an engineering lift on existing
structures. B1024 x 256k simultaneously REQUIRES the paged-KV project.
Say this in every plan; do not paper over it.

## The API architecture (arbitrary requests -> optimal shape)

The principle: CALLERS describe intent; THE SYSTEM owns shaping. Nobody
outside sends lanes, prefill rows, or slot ids.

                     +-----------------------------------------+
   arbitrary HTTP -> | model_api (the ONLY daemon client)       |
                     |  - request queue, priorities, cancellation|
                     |  - continuous batcher: admission control  |
                     |    by resident capacity + KV budget       |
                     |  - shapes prefill (full-width frames,     |
                     |    prefix-cache aware) and decode (lane   |
                     |    packing; spec for B1, batched plain    |
                     |    decode for Bn)                        |
                     |  - tokenizer/detokenizer sidecar (the    |
                     |    python stack on-host) for text in/out |
                     |  - SSE streaming per OpenAI shape         |
                     +-----------------------------------------+
                          | one long-lived engine session

Work items, in order:
1. FIX the model_api completion path (currently stalls post-prefill,
   then exits; the delta vs model_batch's Progress+poll loop is one
   CloseAdmission-shaped bug away - health + admission + prefill already
   verified working).
2. Continuous batching in the API worker: N HTTP requests <-> one engine
   session, admission by capacity, per-request streaming.
3. Tokenizer sidecar: text in / text out (v0 is token-id in/out).
4. Optimal shaping: prefill concentration + prefix reuse already exist
   in the engine; the API chooses request_capacity/prefill rows from the
   deployment descriptor, not from the caller.
5. THEN the B1024/256k ladder: limits -> pools-by-residency -> paged KV.

## Non-goals (explicit)

- No second engine implementation: model_api drives the SAME
  model_batch_engine the CLI tool uses.
- No multi-node in the API layer: TP/PP belongs to the deployment
  config (see the TP4xPP4 estimates in the release notes).

## The 2.5TB backing-store budget (measured arithmetic, 2026-08-24)

Qwen3.8-27B per-token attention KV: 16 attn layers x (2 x 4 GQA KV
heads x 256 dim x 2B BF16) = 64 KiB/token. The 48 GDN layers are
recurrent - fixed state per LANE (order MB/lane), not per token; the
hybrid layout is what makes this model 4x cheaper per token than a
standard 64-layer attention model. Sanity check: the current in-VRAM
pool (8192 blocks x 64 tokens x 64 KiB = 34 GB) + 28 GB weights matches
the observed device_gib=77.7.

| Workload | KV bytes | vs 2.5TB budget |
|---|---|---|
| B1024 x 256k (this model, BF16 KV) | 17.6 TB | 7.0x OVER - 14% fits |
| B1024 x 37k / B512 x 74k / B156 x 256k | 2.5 TB | exactly the horizon |
| B1024 x 256k, FP8 KV | 8.8 TB | 3.5x over |
| B1024 x 256k, std 64-layer 8-KV-head model, BF16 | 70 TB | 28x over |

Conclusions: (1) 2.5TB does NOT cover full B1024x256k for this model in
BF16 KV; treat it as the LRU horizon, not the workload promise. (2) The
pager must apply BACKPRESSURE (admit fewer lanes / shorter contexts /
evict cold prefixes) rather than thrash - a thrashing round reads its KV
from the backing store at disk bandwidth (~7-14 GB/s NVMe) vs 273 GB/s
VRAM: a 20-40x cliff on evicted blocks. (3) FP8 KV halves everything
(kv_cache_codec exists) and should be part of the B1024 plan. (4) The
per-token cost formula scales any future model: attn_layers x 2 x
kv_heads x head_dim x bytes_per_element.
