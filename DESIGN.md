# DESIGN.md — laguna-s-2.1 driver (lane/laguna-driver @ 8f3a6f2)

Designer stage output. Untracked by law. The coder implements THIS file.
HF truth fetched 2026-09-09 from poolside/Laguna-S-2.1 (config.json,
modeling_laguna.py 886 lines, configuration_laguna.py 249 lines — all read in
full; pin shas at contract freeze). Warm copy exists and is still
downloading (16 of ~219 GiB landed at check time).

---

## 0. THE PROBLEM IN ONE SENTENCE

Port Poolside Laguna-S-2.1 (48-layer GQA/SWA hybrid MoE, per-layer head
counts, two rope regimes, yarn 1M) onto the donor resident-decode-stage
anatomy using ONLY existing shared kernels plus three minimal flagged
shared-code extensions.

## 0.1 REFRAMES THAT DISSOLVE THE COMPLEXITY

1. **"Per-head-gated MoE" is FALSE — the manager memory and the brief are
   wrong.** HF truth (modeling_laguna.py :352-461): `gating: "per-head"` is
   the ATTENTION OUTPUT gate (`g_proj: H -> heads`, softplus, broadcast over
   head_dim, applied BEFORE o_proj). The MoE router is a completely standard
   per-token sigmoid top-10 with correction bias (:144-184) — byte-for-byte
   the DeepSeek noaux_tc shape the donor already runs via
   `LmTopkSmallKernel`. No new router. No new MoE mechanism.
2. **Per-layer head variation is a TABLE, not a mechanism.** 48 heads on
   full-attention layers (12 layers, i%4==0), 72 on sliding layers (36).
   The shared GQA decode kernel already takes `heads` as a RUNTIME argument
   (gqa.cuh :77) — the variation rides dispatch constants, two per-layer
   variants, zero new kernels.
3. **TP16 is impossible for this geometry; the fleet shape is TP8 x PP2.**
   72 q-heads/16 and 8 kv-heads/16 are non-integer; 72/8=9, 8/8=1, 48/8=6
   are exact. PP is native donor machinery (STAGE_INDEX,
   stage_layer_counts). No padding hacks, no KV replication.
4. **Yarn does not need a yarn kernel.** Yarn = non-uniform per-index
   frequencies + one cos/sin scale. That is the EXISTING per-head rope
   kernel plus (a) an optional frequency table and (b) a scale factor —
   both backward-compatible defaulted parameters, not a new mechanism.

---

## 1. GEOMETRY TABLE (HF-verified; citations = config.json fields)

| Constant | Value | Config field |
|---|---|---|
| architecture | LagunaForCausalLM, model_type `laguna` (custom_code) | architectures |
| hidden | 3072 | hidden_size |
| layers | 48 | num_hidden_layers |
| vocab | 100352 | vocab_size |
| max context | 1,048,576 (matches platform norm: glm52/k3/glm5_next headers all state 1048576) | max_position_embeddings |
| rms eps | 1e-6 | rms_norm_eps |
| head_dim | 128 (explicit) | head_dim |
| kv heads | 8 total, 1 per rank at TP8 | num_key_value_heads |
| q heads, FULL layers (12: idx 0,4,...,44) | 48 (6/rank at TP8) | num_attention_heads_per_layer[i], i%4==0 |
| q heads, SLIDING layers (36) | 72 (9/rank at TP8) | num_attention_heads_per_layer[i], i%4!=0 |
| layer types | full at i%4==0, sliding elsewhere (48-entry list; first layer is full) | layer_types |
| sliding window | 512 | sliding_window |
| rope FULL layers | yarn: theta 5e5, factor 128, orig 8192, beta_fast 32, beta_slow 1.0, **attention_factor 1.4852030263919618 (stated — never recompute)**, partial_rotary_factor 0.5 (rotary_dim 64) | rope_parameters.full_attention |
| rope SLIDING layers | default: theta 1e4, partial 1.0 (rotary_dim 128), scale 1.0 | rope_parameters.sliding_attention (reaches the model as `swa_rope_parameters` via configuration_laguna.py :193-194 — VERIFIED, sliding layers do NOT share the yarn table) |
| rope pairing | NeoX half-split (`rotate_half`), NOT interleaved; rotary dims are the FIRST 64 of the head on full layers (modeling :263-306) | modeling source |
| QK norm | RMSNorm(head_dim, eps) with weight, per head, BEFORE rope, q and k | modeling :398-399, :421-422 |
| attention scale | 128^-0.5 = 0.08838834764831845 (stated constant) | modeling :362 |
| attention output gate | g_proj H->heads (per-head), **softplus in fp32**, broadcast over head_dim, applied BEFORE o_proj; ALL 48 layers | config gating "per-head" + gating_types; modeling :386-391, :452-459 |
| attention bias | none (q/k/v/o/g) | attention_bias false |
| attention sinks | none (swa_attention_sink_enabled absent) | config absence + modeling :394 |
| MoE layers | 1..47 (layer 0 dense) | mlp_only_layers [0], decoder_sparse_step 1 |
| experts | 256, top-10 | num_experts, num_experts_per_tok |
| expert intermediate | 1024 (swiglu: silu(gate)*up) | moe_intermediate_size, hidden_act default "silu" (configuration_laguna.py :147) |
| shared expert | 1, intermediate 1024, standard MLP, ADDED AFTER routed scaling | shared_expert_intermediate_size; modeling :252-258 |
| router | sigmoid scores; selection on score + e_score_correction_bias; weights = UNBIASED sigmoid gathered, renormalised (norm_topk_prob), then routed_scaling **2.5** applied to routed output; softcap 0.0 (disabled); no groups | norm_topk_prob, moe_routed_scaling_factor, moe_router_logit_softcapping, modeling :169-184 |
| e_score_correction_bias | SHIPPED in checkpoint per routed layer (`model.layers.N.mlp.experts.e_score_correction_bias`, 47 tensors) | safetensors index (census below) |
| dense layer 0 MLP | intermediate 12288 | intermediate_size |
| norms | pre-norm residual stack (input_layernorm, post_attention_layernorm, final model.norm) — standard | modeling :486-487, :500-518 |
| embeddings | untied (lm_head separate), no special padding-row handling (padding_idx affects training init only) | tie_word_embeddings false |
| tokens | bos 2, eos [2, 24], pad 9 | bos/eos/pad_token_id |
| MTP / drafter | NONE in base checkpoint. DFlash drafter is a SEPARATE repo (poolside/Laguna-S-2.1-DFlash), sysadmin-owned, later | safetensors census: 36769 tensors, 23 name patterns, zero drafter patterns |

Checkpoint census (from model.safetensors.index.json, 46 shards):
`model.embed_tokens.weight`, `lm_head.weight`, `model.norm.weight`, and per
layer: `input_layernorm.weight`, `post_attention_layernorm.weight`,
`self_attn.{q,k,v,o,g}_proj.weight`, `self_attn.{q,k}_norm.weight`,
`mlp.gate.weight` (router [256,3072]),
`mlp.experts.e_score_correction_bias`,
`mlp.experts.{0..255}.{gate,up,down}_proj.weight` (UNFUSED per-expert),
`mlp.shared_expert.{gate,up,down}_proj.weight`; layer 0 additionally
`mlp.{gate,up,down}_proj.weight` (dense 12288).

Weights math (bf16): total **218.9 GiB** (q 2.32 + o 2.32 + kv 0.56 +
g 0.02 + experts 211.5 + shared 0.83 + dense0 0.21 + emb/head 1.15).
Per rank TP8: **27.4 GiB**; per stage at PP2: **13.7 GiB**.
KV: 512 B/token/layer/rank (1 kv head x (128+128) x bf16) = 24 KiB/token/rank
full model; 1M context = 23.4 GiB/rank. Weights + 1M KV ~= 51 GiB/rank at
TP8 single stage — inside the 110 GiB ceiling with margin.

## 2. ARCHITECTURE MAPPING — every tensor/mechanism -> existing shared kernel or flagged new code

Shared kernels audited: inference/kernels/{gqa,attn,project,norm,topk,kv,
route,head,scale,activation,layout,linear_attn,hc,gemm,mma,tma,graph,
layer_kind,speculate,tp_reduce,frame_error,tensor_map,tile,dtype}.cuh.

| Laguna mechanism | Maps to | Status |
|---|---|---|
| GQA KV store (1 kv head/rank, 128+128 bf16 slot) | `LmGqaKvStoreKernel<Geometry,THREADS,1,128,128>` — inference/kernels/gqa.cuh:56 | EXISTS — direct use (template KV_HEADS=1 is per-rank; runtime `heads` = 6 or 9) |
| Sliding window 512 | `LmBuildSlidingWindowPositionsKernel` gqa.cuh:13 + `selected_positions` path of decode; full layers pass selected=0 + row_position | EXISTS — direct use |
| GQA attention decode | `LmGqaAttentionDecodeKernel` gqa.cuh:77 (runtime heads, runtime qk_scale, row_position causal bound) | EXISTS — direct use |
| QK norm (per-head RMS, weight, eps) | muse lane's `LmHeadRmsNormKernel` landing in norm.cuh | CONSUME — flag A below |
| rope, sliding layers (theta 1e4, full 128) | `LmRopePerHeadKernel` project.cuh:426 (rope_dim=head_dim -> placement moot) | EXISTS |
| rope, full layers (yarn, first-64 placement, x1.4852 scale) | extended `LmRopePerHeadKernel` (inv_freq table + attention_scale + explicit rope offset) | FLAG B below — exact diff |
| attention output gate (softplus, per-head, broadcast) | new tiny shared kernel in norm.cuh (activation-parameterised head gate) | FLAG C below — exact diff |
| router (sigmoid + selection bias + renorm x 2.5) | `LmTopkSmallKernel<THREADS,10,true,1,1,LM_TOPK_SCORE_SIGMOID>` topk.cuh:43 with `selection_bias`=e_score_correction_bias, `mixture_scale`=2.5; n=256 <= LM_TOPK_SMALL_LIMIT(1024); identical instantiation already proven by glm5_next layer.cuh:2118 (and k3 llms: K,1,1,SIGMOID) | EXISTS — direct use |
| route build + grouped expert GEMM (fused gate_up W1, down W2) | donor `LmRouteBuild` + expert grouped GEMM path (glm5_next layer.cuh:2145+) | EXISTS — donor port |
| silu*up expert activation | `LmSiluMulKernel` norm.cuh:127 (gate_first per packer layout) | EXISTS |
| shared expert + routed output combine, x2.5 inside router weights | donor MoE finalize path | EXISTS — donor port |
| qkv projection | packer fuses q,k,v row-wise per rank into one GEMM; `LmSplitQkvKernel` project.cuh:387 splits (g_proj stays a separate tiny GEMM — its per-head layout does not fit SplitQueryGate) | EXISTS — direct use |
| qk-norm placement | after split, before rope (grid (heads,rows)) | ordering per modeling :412-425 |
| pre-norm residual stack | `LmFusedResidualRmsNormKernel` norm.cuh:50 | EXISTS — donor port |
| dense layer 0 MLP | donor dense MLP path | EXISTS — donor port |
| lm_head | donor head path (untied) | EXISTS — donor port |
| KV layout | SPARK_KV_CACHE_LAYOUT_FULL_KEY_VALUE, layer_count 48, head_count 1/rank, dims 128/128, bytes/scalar 2, page slots 64 | EXISTS — family kv_geometry header |
| DFlash drafter payloads | stagepack FLAG_DFLASH in KNOWN_FLAGS; loader accepts, records, SKIPS payloads (glm5_next MTP load-and-ignore pattern). No drafter module, no speculate wiring | format-level only |

No laguna-private copies of any >=2-family mechanism. No new attention,
router, norm, or window code outside inference/kernels/.

## 3. FILE PLAN (exact; donor + substitution sketch)

Port tool: `/Users/mac/batch-ling/tools/dev/port_family.py` (committed on
lane/ling-driver, NOT on main — copy it into tools/dev/ untracked-local or
reference batch-ling; flag to manager). Donor = glm5_next on MAIN at
8f3a6f2 (self-contained anatomy with source/cuda/, _internal.h — the
mission anatomy). The ling branch (origin/lane/ling-driver) is the
SURGERY REFERENCE for: MLA->replacement deletions, name_map/tensor_patterns
scaffolding, contract + references pinning, test header BINDINGS pattern,
firmware.json shape. Do not build on ling's files; port from main.

Substitution map (longest-first; port_family.py prints per-file counts — a
zero-count port cannot pass):

- `glm5_next` -> `laguna`, `GLM5_NEXT` -> `LAGUNA`, `Glm5Next` -> `Laguna`,
  `glm53_flash` -> `laguna` (contract refs)

New files:

```
model-families/laguna/include/sparkpipe/spark_laguna_model.h          (donor: spark_glm5_next_model.h)
  - all Section-1 constants as SPARK_LAGUNA_MODEL_* macros
  - SPARK_LAGUNA_MODEL_LAYER_HEAD_COUNT(i)   48-entry dispatch table
  - SPARK_LAGUNA_MODEL_LAYER_IS_SLIDING(i)   48-entry dispatch table
  - static_asserts: 48%8==0, 72%8==0, 8%8==0, head_dim 128, vocab 100352
model-families/laguna/include/sparkpipe/spark_laguna_kv_geometry.h    (donor: glm5_next kv_geometry)
model-families/laguna/name_map.json        (generated from the HF index census; 23 patterns, count 36769 total)
model-families/laguna/tensor_patterns.json (dtype+shape per pattern; per-kind head counts 48/72)
model_contracts/laguna_authoritative.json  (pre-freeze: publisher config pinned poolside/Laguna-S-2.1 @ <revision> + config sha; shard shas pinned at freeze when warm download completes)
model_contracts/references/modeling_laguna.py       (byte copy + pinned sha256)
model_contracts/references/configuration_laguna.py  (byte copy + pinned sha256)
modules/laguna_resident_decode_stage/Makefile                        (donor Makefile; STAGE_INDEX/PIPELINE_SLOT_COUNT pattern kept)
modules/laguna_resident_decode_stage/include/sparkpipe/
  spark_laguna_resident_decode_stage_firmware.h
  spark_laguna_serving_adapter.h
  spark_laguna_batch_tuning.h
modules/laguna_resident_decode_stage/source/
  cuda/api.h, cuda/config.h, cuda/launch_shape.h, cuda/layer.cuh, cuda/unity.cu
  spark_laguna_resident_decode_stage_module.c
  spark_laguna_resident_decode_stage_cuda.cu
  spark_laguna_resident_decode_stage_internal.h
  spark_laguna_serving_adapter.c          (stage_layer_counts {24,24}; TP8; eos {2,24})
  spark_laguna_stagepack_format.h         (magic 0x334C4147 — grep-assert unique; enum+shape table DRIVES pack load + synthesize; FLAG_DFLASH 0x2 in KNOWN_FLAGS)
modules/laguna_resident_decode_stage/tools/laguna_pack_synthesize.c
modules/laguna_resident_decode_stage/validation/
  spark_laguna_reference.c                (host C reference: qk-norm, yarn table, rope, softplus gate, router+experts — independent math, written FROM modeling_laguna.py, zero driver imports)
  spark_laguna_resident_decode_stage_cuda_validation.cu
  validate_laguna_resident_decode_stage_cuda.sh
examples/model_descriptions/laguna_resident_decode_stage_firmware.json
tests/test_laguna_model_header.py         (BINDINGS: every macro <-> contract field, ALL 48 per-layer indices bound)
tests/host_cuda/laguna_layer_host.cu + tests/test_laguna_layer_host.py
tools/laguna_layer_reference.py           (python fp32 oracle; numpy, no torch, no driver imports; consumes pack tensors or synthetic fixtures)
tools/laguna_stagepack.py                 (family packer, donor glm5_next_resident_stagepack.py; fuses q|k|v, fuses expert gate_up per expert, TP8 slices, PP2 stage slices)
tools/laguna_gen_deployment.py            (donor glm5_next_gen_deployment.py; SESSION BASE 64800, TP8: base + src*8 + sink)
```

DELETED relative to donor (grep-verify clean, the ling-port technique):
hyper-connections + wide "hc" collective, DSA indexer + sideband, ALL KDA
mechanisms (decay/gate/conv/state), MLA latent path (kv_a/kv_b/absorbed),
MTP layer inventory (no MTP in checkpoint), TP16 head tables. Also delete
the donor's rope theta constants and any per-layer-theta machinery — laguna
carries exactly two rope regimes.

## 4. PACK PLAN

- **bf16 driver first.** Payload types BF16/F32/U32 (+PACKED_WEIGHT for
  future arms; bf16 pack uses raw BF16 payloads). Packer: memmap headers
  only (donor pattern, zero read amplification), fuse q|k|v per rank,
  fuse expert gate|up per expert into W1 [2048,3072] (gate rows first,
  LmSiluMulKernel gate_first=true), down stays [3072,1024] W2.
- **TP8 x PP2 fleet shape (decision).** stage 0 = layers 0..23 (6 full +
  18 sliding), stage 1 = 24..47 (6 full + 18 sliding). Per-rank slices:
  q heads 6 (full) / 9 (sliding); o columns matching; kv 1 head/rank;
  experts 32/rank; both stages 8 ranks = all 16 nodes busy. Session port
  base **64800** (allocation law: 61500/62500 glm53flash, 63500 glm53full,
  64500 glm52, 64630 k3, 64700 qwen4flash). Fallback if PP2 bring-up
  blocks: TP8 single-stage on 8 nodes — same packs minus stage split;
  decide at bring-up, not before.
- **Context: 1048576 stated** (platform norm; config max). KV request:
  full layout, 48 layers, 1 head/rank, 128/128, bf16, page slots 64.
  Practical serving caps are deployment config (runtime limits), not
  driver geometry.
- **fp8/nvfp4: LATER, coexistence smoketests only.** Official sources
  (poolside/Laguna-S-2.1-FP8, -NVFP4 on warm when landed). Same module
  recompiled with the codec trait; smoketest = load + 8 tokens; NO
  multi-arm serving machinery, NEVER self-quantize. The packer must
  fail-closed on dtype surprises (e.g. e_score_correction_bias dtype
  asserted, expected bf16 or f32 — verify from shard header at freeze).
- **DFlash: load-and-ignore at pack level.** Base checkpoint carries NO
  drafter tensors (census-proven). stagepack_format.h defines
  FLAG_DFLASH + an optional ignored-section rule (accept, record in
  inventory, skip payload load — glm5_next MTP precedent). No drafter
  module. Per the operator MTP-sequencing ruling: accuracy first, spec
  later, never in this lane.

## 5. VALIDATION PLAN (anchors independent of the driver; gates cpu-class on sparkb, never the mac)

1. **Header/contract bind**: tests/test_laguna_model_header.py — parses
   spark_laguna_model.h, binds EVERY macro (and head-count/layer-type
   dispatch outputs at all 48 indices) against laguna_authoritative.json.
   RED on any mismatch. Pure python, runs anywhere.
2. **Yarn table anchor**: tools/laguna_layer_reference.py computes the 64
   yarn frequencies from the HF yarn formula (beta_fast 32, beta_slow 1,
   factor 128, orig 8192) INDEPENDENTLY; the module's host table builder
   must match <= 1e-6 relative per element. attention_factor is the
   STATED 1.4852030263919618 (asserted equal to 0.1*ln(128)+1 as a
   comment-free test, never recomputed in product code).
3. **Layer-state oracle**: laguna_layer_host.cu runs the ported layer
   path (both variants: full-48/yarn/partial-64 and sliding-72/plain/
   window) on synthetic bf16 inputs; the SAME inputs go through
   laguna_layer_reference.py. Gates: hidden-state max abs diff <= 0.02;
   attention softmax weights rel err <= 1e-2; router expert SETS exact
   match (ties asserted deterministic); MoE combine rel err <= 1e-2.
   Tolerance constants reuse the existing validation common conventions
   (modules/spark_resident_decode_stage_cuda_validation_common.sh).
4. **Window boundary ranks**: layer host cases at context positions
   0, 1, 511, 512, 513, and window-edge straddles; decode rows with
   row_position == context-1; two-sequence interleave. The sliding path
   must select EXACTLY the last min(pos+1, 512) positions.
5. **Real-weight boundary-rank pack checks vs /mnt/model-warm/laguna-s-2.1**:
   per-rank verifier re-derives every rank slice from the warm shards
   (sha + shape): rank boundaries at head offsets 0/6/.../42 (full) and
   0/9/.../63 (sliding), expert offsets 0/32/.../224, stage boundary
   layers 23|24, first/last shard. Two-pass placement proof on re-run.
   Runs as a queued task; warm download must be COMPLETE (marker) first.
6. **Gates (PR = green by exit code)**: `make offline-gates` on sparkb
   (build-all, run-tests incl. the new tests, package-manifest) — plus
   dry-law + code-size ratchet as for every family. CUDA compile +
   synthesized-weight validator receipts come from the standard
   module_build_release.sh publish flow (GB10), not from the PR gate.
7. **Census lock**: packer asserts the 36769-tensor / 23-pattern census;
   unknown checkpoint tensor = fail-closed (except DFlash-flagged
   sections, which are recorded-and-skipped).

## 6. SHARED-CODE FLAGS FOR THE MANAGER (the only touches outside the family)

- **A (consume, muse lane)**: `LmHeadRmsNormKernel` (norm.cuh, muse lands
  it) — laguna qk-norm requires: per-head RMS over head_dim=128, bf16
  in/out, PLAIN weight multiply (laguna's q_norm/k_norm weights ship in
  the checkpoint; NOT the centered (1+w) semantic — that is
  LmCenteredRmsNormKernel and must NOT be used here), fp32 epilogue
  pass-through 1.0. If the landed signature forces the epilogue, pass 1.0;
  do not fork.
- **B (minimal extension, exact diff)**: inference/kernels/project.cuh
  `LmRopePerHeadKernel` (:426) gains three TRAILING defaulted parameters:
  `const float *inv_freq_table = 0` (nonzero -> angle = position *
  inv_freq_table[index]; else today's `__powf(theta, ...)` path),
  `float attention_scale = 1.0f` (multiplies cos and sin inside the pair
  rotate), `uint32_t rope_offset = 0xffffffffu` (0xffffffff resolves to
  today's `head_dimension - rope_dimension`). Existing call sites compile
  unchanged; behavior identical when defaults are used. Justification:
  laguna is family 1 of N for yarn; the alternative (family-private yarn
  kernel) violates the DRY law. Packer-permutation fallback (permute
  q/k/qk-norm/o channels so yarn dims sit at the tail) exists if the
  interface freeze rejects even this — it is tables-only, but it is the
  SECOND choice: harder to verify.
- **C (minimal new kernel, exact shape)**: inference/kernels/norm.cuh —
  head-wise output gate, activation-parameterised so ling's sigmoid
  variant and laguna's softplus variant are ONE kernel:
  `LmHeadGateBroadcastKernel<THREADS, LM_GATE_SIGMOID|LM_GATE_SOFTPLUS>`
  (output [rows, heads, 128] in place; gate [rows, heads]; softplus =
  x > 20 ? x : log1p(exp(x)), computed in f32 — matches F.softplus
  threshold semantics). Coordinate with the ling branch
  (LmHeadWiseGateKernel, sigmoid-only) so exactly ONE lands.
- **D (consume)**: gqa.cuh — NO changes needed. Runtime `heads` covers
  48/72; template KV_HEADS=1; qk_scale runtime constant. Confirmed by
  direct read of the frozen interface.
- **E (tool)**: tools/dev/port_family.py is not on main; copy from
  /Users/mac/batch-ling/tools/dev/port_family.py (or the ling branch).

## 7. RISKS + THE 3 HARDEST PROBLEMS

Hard:
1. **Yarn + partial-rope + placement correctness.** Three interacting
   offsets (first-64-of-128, 64 non-uniform frequencies, x1.4852 scale)
   multiply silently into wrong-but-plausible attention. Mitigation: the
   independent python table anchor (5.2) + layer host oracle vs fp32
   reference on BOTH layer variants (5.3) + boundary positions (5.4).
2. **Two attention variants across the TP/PP lattice.** 6-head full and
   9-head sliding layers alternate; GEMM tile shapes, split offsets, and
   KV slots differ per variant; a swap of constants compiles fine and
   answers garbage. Mitigation: per-variant static_asserts in config.h,
   the 48-index header bind, and boundary-rank pack checks (5.5).
3. **Expert pack explosion.** 256 experts x 47 layers x 3 unfused tensors
   (36k tensors) must fuse into W1/W2 with codec-aligned, rank-sliced,
   stage-sliced layouts without dropping or duplicating one expert.
   Mitigation: census lock + count asserts per rank pack + verifier
   recomputing every slice from source.

Risks (watch, don't architect):
- Warm download incomplete (16/~219 GiB at check) — real-weight gates
  block on the marker; everything else proceeds on synthetic fixtures.
- e_score_correction_bias dtype surprise at freeze — fail-closed assert,
  packer upcast is a repack decision made once, loudly.
- Prefill is decode-kernel-per-token (O(chunk x context)) on full layers
  — platform-wide existing property, perf-debt note, NOT this lane's job.
- Sliding-layer decode walks only 512 positions: cheap; full layers walk
  full context: same as every GQA family on the platform.

## 8. ACCEPTANCE CRITERIA FOR THE CODER (ordered, testable)

1. `tools/dev/port_family.py` present; glm5_next->laguna port executed
   with per-file substitution counts printed; every mapped file nonzero.
2. `model-families/laguna/` complete (model header with per-layer tables
   + static_asserts, kv_geometry, name_map.json, tensor_patterns.json).
3. `model_contracts/laguna_authoritative.json` + both reference .py
   copies with pinned sha256; `tests/test_laguna_model_header.py` green.
4. Module tree complete per Section 3; stagepack magic unique (grep
   across all families); `make offline-gates` green on sparkb by exit
   code (build-all, run-tests, package-manifest, ratchets).
5. Donor deletions grep-verified absent: hc/DSA/KDA/MLA-latent/MTP/TP16.
6. Attention path uses ONLY inference/kernels/gqa.cuh + project.cuh
   kernels + flagged A/B/C shared kernels; grep proves no laguna-private
   attention/router/norm kernel exists in the module.
7. Layer host tests green: both variants vs python oracle within 5.3
   tolerances; window boundary cases exact.
8. Yarn table test green: C builder vs python <= 1e-6 rel per element.
9. Packer: census lock (36769/23) passes on a warm-prefix smoke or a
   synthetic fixture; TP8xPP2 slice set verifies; boundary-rank checks
   green once warm download completes (blocking, tracked).
10. Deployment generator emits TP8 tables at base 64800; adapter carries
    stage_layer_counts {24,24}, eos {2,24}.
11. FLAG_DFLASH accepted-and-skipped path proven by a synthetic pack
    carrying the flag + one dummy payload section (loads, records, skips,
    forward identical).
12. PACKAGE_MANIFEST.json + SHA256SUMS regenerated LAST (post-freeze);
    everything above re-run green after regeneration.

Nothing in production code has comments (the law), nothing derives what
the config states, everything fails closed.
