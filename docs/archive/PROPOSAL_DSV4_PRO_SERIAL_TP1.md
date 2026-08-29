# DSV4 Pro — serial-TP (TP1) replay plan for one idle spark + harness spec

Owner: dsv4-pro · status: PLAN ONLY (no launch, no download). Complements
`docs/PROPOSAL_DSV4_PRO_SINGLE_SPARK.md` (the scouting + size check). This doc is
the serial-TP replay plan, the TP16 collective spec, the golden-ref inventory, and
the Pro geometry/collective spec the cuda-kernels harness core consumes.

**Host pick after scouting: spark2** — same healthy TP4xPP4 deployment as spark0
(pack 92.76 GiB, adapter/driver/residentd rebuilt Aug 17 14:23), but cleaner: 3 GiB
used vs spark0's 13 GiB, and 115 GiB available (vs 106). Both are GB10 (sm_121a),
119 GiB unified RAM, cgroup MemoryMax=108G.

---

## 0. Scouting recap (shards + sizes, measured on-host)

| Artifact | Where | Size |
|---|---|---|
| Authoritative full pack (61 layers + GA MTP) | sparkb (not idle hosts) | 805.4 GiB |
| TP4xPP4 rank pack (rank 0 / rank 2) | spark0 / spark2 | **92.76 GiB** (backbone ~48 GiB + MTP replication ~40 GiB) |
| TP16 PRO shards | **none exist** | — |
| Pro model source / val slices / validator | spark3 / sparkb (not idle hosts) | — |
| cgroup | both hosts | MemoryHigh=100G, MemoryMax=108G |

**Consequence for the replay:** a TP4xPP4 rank pack is PP+TP sharded and cannot be
replayed as a clean TP16 all-reduce; the serial-TP replay needs **TP16 backbone
shards generated from the full pack** (which lives on sparkb and is a deferred
download). Each TP16 backbone shard is ~48 GiB and fits under 108G with ~60 GiB
headroom; the MTP block must be excluded from the replay because it is **replicated
full-width on every rank** (zero draft collectives) and is validated separately.

---

## 1. Serial-TP replay plan (rank-major, 16 loads, host-side all-reduce)

Goal: functionally validate the **TP16 sharder + the all-reduce pattern + per-rank
expert routing** at TP1-equivalent precision on ONE host, without ever holding more
than one shard in memory. Slow is fine; correctness is the goal.

### 1.1 Shard sequence and sizes

Produce 16 backbone-only TP16 shards via the existing parameterized sharder
(`tools/dsv4_tp16_stagepack.py --model pro`, MODEL_GEOMETRY["pro"] at :59-63),
with one addition: **a `--exclude-mtp` flag** that drops the replicated MTP
entries (they are byte-identical on every rank — dropping them from the TP replay
is lossless for the backbone and halves each shard). Sequence rank 0 -> 15.

| Step | Artifact | Size (est.) | Peak resident |
|---|---|---|---|
| per rank r | `dsv4_pro.tp16.rank{r:02d}.spstage` (backbone, MTP stripped) | **~47.8 GiB** | ~48 GiB |
| per rank r | per-layer partial hiddens (128 tok x 61 layers x 2 all-reduce x 7168 BF16) | ~112 MB | ~0.2 GiB |
| **total peak** | one shard + its partials + harness | | **~48.5 GiB << 108G** |

- Backbone per-shard = 765.4 GiB / 16 ≈ 47.8 GiB (full pack 805.4 GiB minus ~40 GiB
  MTP, divided by TP16).
- Serialization is **rank-major**: load rank r -> run the full 61-layer forward over
  the 128-token golden prompt -> store the per-layer partial hidden (attn-side and
  FFN-side) + per-layer indexer/KV partials -> unload. 16 loads total (NOT 61x16).
- After all 16 ranks, the host-side reducer all-reduces (sums) the 16 partials
  per layer, feeding the reduced hidden to the next layer's rank-partials — i.e.
  the layer-boundary all-reduce is replayed from saved partials, never recomputed.

### 1.2 TP16 collective spec (all-reduce pattern + expert routing per rank)

From `tools/dsv4_tp16_stagepack.py` sharding rules (:206-245, :331-351):

| Tensor (kind id) | Shard | Pattern | Combine |
|---|---|---|---|
| WQ_A(1)/WQ_B(3)/WKV(4) | rows (128 query heads / 512 KV -> 8 / 32 per rank) | column-parallel | partial q/kv |
| WO_A(6) | rows by output group (16 groups, 1 group/rank at TP16) | column-parallel | partial attn out |
| WO_B(7) | cols by output group | row-parallel | partial attn out -> **sum** |
| ATTN_SINK(0) | cols /16 | row-parallel | sum |
| EXPERTS_W1/W3(19/21) | rows = expert_width/16 = 192 per expert, ALL 384 experts | column-parallel (fused) | partial FFN |
| EXPERTS_W2(20) | cols = expert_width/16 | row-parallel (fused) | partial FFN |
| SHARED_W1/W3(22/24) / SHARED_W2(23) | rows / cols | col / row | partial FFN |
| COMPRESS_WKV/WGATE(26/27), INDEX_WKV/WGATE(32/33) | rows | column-parallel | partial compressor/indexer score -> **sum before top-k** |
| EMBEDDING(35), norms, gate, HC params | full | replicated | none |
| LM_HEAD(37) | vocab rows (129280/16 = 8080) | vocab-sharded | **cross-rank U64 max (argmax)**, not a sum |
| MTP block (kinds 41-49 + 3 draft layers) | **replicated full** | zero draft collectives | none (out of replay scope) |

**Expert routing per rank:** the gate is replicated (bias-gate, noaux_tc, top-6);
every rank routes to the **same top-6 expert ids** (no expert-parallel all-to-all,
no EP sharding — `dsv4_tp16_stagepack.py` shards W1/W3 by *expert width*, not by
expert id). Each rank computes a 1/16-width expert FFN; the fused W1/W3(column) +
W2(row) split elides the intermediate all-reduce, so there is **one all-reduce per
FFN side** (after shared-expert add + residual).

**All-reduce points per decode token (61 layers):** 2 hidden sums per layer
(attn-side after WO_B, FFN-side after shared+residual) = **122 sums** — matches the
comms inventory `dsv4_pro_inference_path_audit.md:51`; plus 1 head cross-rank
U64 max (vocab-sharded argmax); plus the CSA indexer score sum before top-k.

### 1.3 Hard parts the harness must get right (GLM52 lesson)

- **CSA/HCA indexer + sparse attention are not a pure sum.** The per-rank
  compressor/indexer projection is column-parallel, but the top-k *selection* and
  the DSA attention read the combined KV/index. The harness must replay the
  cross-rank indexer score sum **before** top-k, and reproduce each layer's
  selected indices exactly. GLM52's serial-vs-ring equivalence already bit on a
  missing per-layer index (`docs/archive/GLM52_PP13_DSA_SERIAL_RING_EQUIVALENCE_20260710.md`:
  "layer_index = layer_index" fixed a serialized path that attended token-0 2048x).
  The Pro replay must set the same per-layer index/context the live ring sets.
- **KV band layout caveat** (unresolved in `dsv4_pro_kv_cache_audit.md:43-50`):
  whether a rank's cache holds the full 512-dim KV or only its 128/32-dim WKV band
  at a column offset. The harness must pin this read pattern before trusting any
  attention partial. This is a first concrete correctness gate, independent of TP.

---

## 2. Golden reference (what exists, what is missing, what to generate)

**Usable now (Pro):**
- **valtail output token = 48774** — the first real Pro token through the real
  final head (`tools/devcycle/dsv4_pro_single_spark_receipts.md:25`), 57+4 slice,
  1 row, max_seq 4096.
- **val4 nonzero_hidden = 28672** — hidden-sum invariant, 0+4 slice (:24).

**Reusable input (model-agnostic):**
- `qualification/dsv4/reference_vectors/ga_stage0_compsec076_p128/prompt_tokens.u32le`
  — 128 token ids (sha256 f2f860f7...). Vocab 129280 is shared Flash<->Pro, so this
  prompt is valid for Pro.

**NOT usable for Pro (Flash geometry):**
- `.../ga_stage0_compsec076_p128/after_layer_2.bf16le` — shape [1,128,4,**4096**],
  hidden 4096 (Flash). Cannot compare against Pro's 7168.

**MISSING — must generate (first concrete step):**
- A **Pro GA-0813 stage-boundary golden**: run the reference model over the 128-token
  prompt and emit the BF16 hyper-connection boundary vector (shape [1,128,4,7168])
  + the final token. `tools/dsv4_ga_reference_vector.py` is Flash-pinned
  (`CHECKPOINT_REVISION=7872f01b`, hidden 4096, first 3 layers) — extend it with a
  Pro geometry target (7168 hidden, 61 layers, 384 experts, top-6, HC 4-stream) to
  emit `after_layer_{N}.bf16le` for Pro. This golden is the TP1 reference the 16
  rank-partials must reproduce exactly after all-reduce.

**Note:** the referenced `docs/dsv4_pro_single_spark_receipts.md` does **not** exist;
the real file is `tools/devcycle/dsv4_pro_single_spark_receipts.md`.

---

## 3. What I supply to the cuda-kernels harness core (and what I wire)

**I supply (geometry + collective + golden):**
1. Pro geometry: hidden 7168, layers 61, experts 384, expert_width 3072, experts/token 6,
   query heads 128, kv heads 1, head_dim 512, qk_rope 64, output_groups 16,
   query_lora 1536, output_lora 1024, vocab 129280, HC streams 4, routed_scale 2.5
   (`model_contracts/dsv4_pro_authoritative.json:6-58`).
2. The TP16 collective spec above (§1.2) — all-reduce point map + expert routing.
3. Golden: `prompt_tokens.u32le` + valtail token 48774 + (to-generate) Pro
   `after_layer_{N}.bf16le`.

**I wire in my lane (per-model):**
- The `--exclude-mtp` sharder flag + a Pro `dsv4_pro_ga_reference_vector.py` target.
- The Pro DSA/indexer replay context (per-layer index, KV band layout) — the
  GLM52-equivalence bug class, pinned against the live ring's context.
- The vocab-sharded head reconstruction (U64 max -> token id) for Pro's 129280.

**Deferred / blocked (state clearly):** the DSpark draft path itself is NOT in this
replay — it is replicated (not TP-sharded) and its kernels are unimplemented
(`spark_dsv4_resident_decode_stage_module.c:47-48`). The MTP weights load
(`..._module.c:1066-1074`) but the draft forward is a separate, later validation.

---

## 4. Per-rank timing findings grounding the ~12-13 tok/s claim (owed)

- **Measured DRAM bandwidth** on sparkb (`tools/devcycle/bw_probe.cu`): 272.7 GB/s
  @ 64 MB, 250.7 GB/s @ 256 MB (DRAM-resident); the 273 GB/s pricing comment is the
  L2-resident number — streaming is ~250 GB/s (`dsv4_pro_performance_estimate.md:6-9`).
- **Weight traffic per token**: ~315 MB/layer (210 MB top-6 experts + 75 MB attn +
  30 MB shared/gate/HC/compressor) -> ~19.2 GB/token over 61 layers (:17-25).
- **Per-token weight streaming**: 19.2 GB / 250 GB/s = **~77 ms/token** (:27).
  Non-weight overheads (122 TP all-reduces + 3 PP hops + control) ~4-7 ms — second
  order (:33-35).
- **Main-model-only decode = ~77-82 ms/token -> 12-13 tok/s** (:39-40). Cross-check:
  the same model predicts 36 tok/s for Flash vs its **measured 40.46 tok/s** (within
  ~12%), validating the DRAM-bound framing (:10-13). Independent local datum: our
  own Flash TP4 B1 median decode = **37.79 tok/s** (no speculation,
  `qualification/dsv4/performance/tp4_b1_20260814/vllm-dsv4-b12x-tp4-b1.json`).

---

## 5. What I need from you
- **Go + host clear** to (a) fetch the full pack to spark2 (805 GiB, deferred) and
  (b) generate the 16 backbone TP16 shards + the Pro golden vector. I will not
  download or launch until you say so.
- Confirm the harness-core handshake with the cuda-kernels agent (my geometry +
  collective spec above is the input; I wire the per-model DSA/indexer/head parts).
- Acknowledge the two blockers I cannot clear alone: the DSpark draft kernels
  (not implemented) and the TP16 shards + Pro golden (both need the full pack).
