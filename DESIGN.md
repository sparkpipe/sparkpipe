# MiniMax H3 — driver design (lane/minimax-driver)

Status: DESIGN — coder contract. Branch `lane/minimax-driver` @ origin/main 8f3a6f2.
Weights truth: `/mnt/model-warm/minimax-h3` on `sparke` (landed complete, 465G, `PUBLISHED`
+ `.WARM-COPY-COMPLETE` present). Upstream: `MiniMaxAI/MiniMax-H3` (diffusers
`MiniMaxH3ModularPipeline`, diffusers 0.36.0.dev0, custom library `minimax-h3`).

---

## 1. What the platform already has, and the chosen integration shape

**Existing findings.** `docs/MODEL_SUPPORT.md` carries a one-line H3 row
("model-profiled TP/PP placement"); `TECHDEBT.md` (Model contracts) lists
"exact checkpoint-derived contracts and native execution packages for MiniMax H3"
as open work; `ARCHITECTURE.md` names H3 in the product set. There is **no
archived diffusion/generative-media design anywhere in docs/ or docs/archive/**
(the "diffusion" hits are DFlash2 block-diffusion *text drafters* — unrelated).
`archive/SPECULATION_AUDIT.md` has an H3-MTP row from an era when H3 was assumed
to be a text LLM; stale, ignore. Conclusion: the platform intended H3 as a
first-class resident model but nobody has designed the shape yet. This document
is that design.

**The reframe.** The reflex is "resident_decode_stage doesn't fit diffusion, so
we need a new serving plane." Wrong. The serving adapter ABI
(`include/sparkpipe/spark_model_serving_adapter.h`) is not a decode-loop
contract — it is a **generic step-pump contract with model-extension payload
slots already built in**:

- Submission: `work_kind` (PREFILL/DECODE/RELEASE), `token_ids`, and
  `model_extension_kind` / `model_extension_bytes` / `model_extension` blob
  (job params ride here — no struct changes).
- Completion: `SPARK_MODEL_SERVING_COMPLETION_FLAG_MODEL_EXTENSION` (0x2) is an
  already-known flag with `model_extension[MAX_EXTENSION_BYTES]` — a completion
  can be a model-defined receipt instead of token ids today.
- `runtime/model_batch_engine.c` schedules kinds 1..3 round-robin and never
  interprets completion payload semantics. It is a pump, not a text decoder.

So H3 is **one new module implementing the SAME adapter ABI**, with this
mapping:

| ABI element | H3 meaning |
| --- | --- |
| PREFILL work (job admission, prompt in `token_ids`, params in `model_extension`) | encoder prefill → text hidden states; latent init; text token-refiner |
| DECODE work | **one denoise step** (engine decode cadence = denoise cadence; `tokens_per_sequence=1`; steps/s falls out of existing telemetry) |
| RELEASE work | job teardown / cancellation |
| Completion with `FLAG_MODEL_EXTENSION` | artifact receipt (path, sha256, bytes, container properties, step/service times); zero tokens |
| Snapshot counters | job/step counters for the API poll |

Media bytes never ride IPC. Rank 0's adapter writes artifacts to a spool dir
under `runtime_root` (`<runtime_root>/media/jobs/<id>.json` + `<id>.media`),
the HTTP door streams files. Job state is **one spool JSON file, 5 states**:
`queued, running, completed, failed, cancelled`. State transitions are written
by rank 0's adapter only (deterministic, no choreography).

Endpoint: the publisher's own local-serving contract
(`scripts/readme/reproducible-768p-t2va-request.sh` on the landed copy) is
`POST /v1/videos` (`task/prompt/conditions/target{short_edge,aspect_ratio,duration_seconds}/seed`)
→ `{id}`, `GET /v1/videos/{id}` → status, `GET /v1/videos/{id}/content` → mp4.
We mirror it exactly in `node/model_api.c` (the deployment's OpenAI-*shaped*
door, same precedent as `prompt_token_ids`). Prompt text is tokenized by the
in-tree BPE tokenizer (`text/tokenizer.c`, loads Qwen2-style `tokenizer.json`;
H3 tokenizer = `Qwen2TokenizerFast`); `prompt_token_ids` accepted directly.

This reuses: resident process (one per rank, no second daemon — hard boundary
#4), batch engine as the pump, resident IPC, serving-adapter template's TP
collective + session-ports parse + weightd attach/lease, tree allreduce,
stagepack loader, fleet UPDATE cycle, validator receipt pattern, tokenizer.
**What is genuinely new is only the model math** (section 3) and the API routes.

## 2. Component map and placement

Landed layout (measured on sparke): top-level components `text_encoder/` 63G,
`transformer/` 62G, `transformer_ref/` 62G, `vae/` 9.8G, `audio_vae/` 578M,
`scheduler/` + `audio_scheduler/` (config only), plus two task-checkpoint
bundles `FL2VA/` 135G and `Ref2VA/` 135G. Vertical slice = **t2va only**
(text→video+audio), which per README belongs to the FL2VA checkpoint whose
t2va path uses: text encoder, transformer, vae, audio_vae, scheduler.
`transformer_ref` is the first/last-frame branch — **not in the t2va path, not
packed**. FL2VA/Ref2VA bundles: out of slice entirely.

| Component | Identity (from landed configs) | Packed content | Placement |
| --- | --- | --- | --- |
| Text encoder | `Qwen3VLForConditionalGeneration`, text tower 64L, hidden 5120, 64q/8kv heads, head_dim 128, inter 25600, vocab 151936, mrope interleaved [24,20,20], theta 5e6 | **text tower only** (vision tower 27L/1152 excluded — not in t2va path), prefill-only, no LM head | TP4-sharded, ~15GB/rank, once per job |
| DiT | `MiniMaxH3Transformer3DModel`, 50 blocks + 2 text-refiner blocks, hidden 5376, 56 heads × 128, qkv dim 7168, ffn 14336 (fused 28672), adaLN proj [96768,2688] per block, `proj_in` [5376,96], `audio_proj_in` [5376,32], `context_embedder` [5376,5120], rope_freq_dim 16, theta 1e4 | full (incl. refiner, embedders) | TP4-sharded, ~15.5GB/rank |
| Video VAE | `AutoencoderKLMiniMaxH3`, latent 24ch, conv arm 6 levels, **decoder = 36L transformer (32 heads × 64, ffn_mult 4, 4 register tokens, rope theta 100, rope_dim_ratio 0.75) + conv/upsample head**, clip_length 17, token_drop 3 | **decoder only** (encoder arm excluded — never encode source video in t2va) | 36L transformer head TP4-sharded; conv head replicated (~4GB total) |
| Audio VAE | DAC-style, latent 32ch @ 40Hz, decoder_rates [5,5,2,2,2,2,2]=800×, 32kHz, stereo via per-channel processing | decoder only | **replicated** (578MB), decode executes on rank 0 |

**TP/PP profile (stated config):** `h3.bf16.tp4` — one island of 4 sparks,
TP4 × PP1, single tree-allreduce session (port base **64800**, ledger:
61500/62500 glm53flash, 63500 glm53full, 64500 glm52, 64630 k3, 64700 qwen4flash),
credit 8, SUM_BF16, via the shared template parse. Per-rank weights ≈ 33GB ≪
110GiB ceiling; both remaining TP4 islands stay schedulable (co-residency).
Encoder and DiT share ONE communicator (uniform, no mixed-TP-group complexity).
Video VAE transformer head reuses the same TP4 collective. Audio decode is
rank-0-local (other ranks idle at the step boundary — deterministic).

**5090 box: not used.** 32GB VRAM < any component, sm_120 vs the tree's
sm_121a driver targets, 10Gb/s single private link, no remote RPC protocol
exists (per RTX5090_SPECULATION_NODE.md runtime boundary). Building transport
for it is a new subsystem with no need. Revisit only if a future audio-only
tier ever matters — recorded, not designed.

Rationale for one island vs 5090-front-end vs split: every split adds a
transport and a failure domain; the fleet holds all weights 6× over; TP4 is
the platform's proven collective shape. `MODULE_MAP` hard boundary #5 (batch
execution enters through `spark_model_batch_engine` and resident IPC) is
satisfied with zero runtime changes.

## 3. Architecture mapping — DiT blocks / attention / conditioning / VAE decode

Measured tensor shapes (safetensors headers, shard 1/3 of
`transformer/diffusion_pytorch_model-*`):

```
transformer_blocks.N.attn.to_q/k/v  [7168, 5376]   ← FULL joint self-attention:
transformer_blocks.N.attn.to_out.0  [5376, 7168]     text + video + audio tokens
                                                    in ONE sequence (MM-DiT style);
                                                    NO cross-attention. qk-norm RMS(128).
transformer_blocks.N.ff.net.0.proj  [28672, 5376]   fused gate+up, silu-mul
transformer_blocks.N.ff.net.2       [5376, 14336]
transformer_blocks.N.adaln_proj.linear [96768, 2688] per-block modulation from
                                                    the (shared-per-step) time
                                                    embedding; 96768 = 18 × 5376
token_refiner.refiner_blocks.{0,1}  same attn/ff shape as main blocks
proj_in [5376,96] (24ch × 1×2×2 patch), audio_proj_in [5376,32],
proj_out/audio_proj_out mirror; context_embedder [5376,5120];
time_embedder.linear_1 [5376,256] (freq_dim 256), linear_2 [2688,5376];
norm_out.linear [10752,2688] final modulation.
```

Kernel mapping (grep of `inference/kernels/` done; DRY ruling applied —
anything existing is reused, single-family mechanisms stay family-local):

| Mechanism | Verdict |
| --- | --- |
| All DiT/encoder/VAE-transformer linears (bf16) | **REUSE** `LmGemmKernel` (`runtime/gemm.cuh`, `inference/kernels/gemm.cuh`) — dense bf16, TMA+tensor-map, existing instantiation pattern (`inference/llms/mimo_2_5/unity.cu` is the template) |
| FFN silu-mul | **REUSE** `LmSiluMulKernel` |
| QK-norm, RMSNorms, GroupNorm(VAE) | RMS **REUSE** (`inference/kernels/norm.cuh`); GroupNorm is NEW (family-local, trivial) |
| Encoder rope (text-only: mrope degenerates to standard interleaved rope — verify against pinned `transformers` source, Qwen3VL text-only path) | **REUSE** `LmRopeKernel` with `LM_ROPE_INTERLEAVED`, theta 5e6, rope_dim 64 |
| DiT 3D axial rope (rope_freq_dim 16 over T/H/W axes) | **NEW** family-local `LmH3Rope3DKernel` (~150 lines; per-token (t,h,w) positions precomputed host-side as stated config) |
| Dense bidirectional S×S attention, no KV, no mask, head_dim 128, bf16/fp32-accum | **NEW** `LmH3DenseAttentionKernel` — tiled online-softmax flash-style kernel over the existing `mma/tile` primitives. This is the one significant new CUDA kernel. All existing attention kernels are KV-cache decode kernels (`LmAttentionDecodeKernel` takes `LmKvView`) — not reusable, do not force-fit |
| adaLN modulation | NEW family host/device glue (per-step time-embed GEMMs + broadcast apply; per-block `adaln_proj` is a [96768,2688] GEMM = existing GEMM) |
| Flow-match scheduler (shift 12.0) | NEW family host math, scalar C, reference-anchored (pinned diffusers `MiniMaxH3Scheduler`) |
| Video VAE 36L transformer head | REUSE DiT block machinery at different template geometry (32×64 heads, rope theta 100) |
| Video VAE conv/upsample head, reflect padding | NEW small family kernels: conv2d 3×3 (+reflect), nearest-upsample+conv, silu |
| Audio VAE conv1d decoder (resblocks k=3/7/11 dil 1/3/5, transposed conv upsample k=9/4, snake activation, 8-head attention block) | NEW small family kernels (1D convs; straightforward) |
| mp4 muxing | NOT CODED: rank 0 pipes raw yuv420p frames + wav through the node's `ffmpeg` when present (stated config, checked at startup); absent → artifact = `.tar`(raw video `.yuv` + `.wav`), `/content` serves it, job metadata names the format. No codec code enters the tree |

## 4. File plan

New (family — geometry + tables + dispatch + contracts ONLY, per DRY ruling):

```
model-families/minimax_h3/
  include/sparkpipe/spark_minimax_h3_model.h          (encoder+DiT+VAE geometry constants from landed configs)
  include/sparkpipe/spark_minimax_h3_kv_geometry.h    (encoder prefill KV only)
  name_map.json, tensor_patterns.json                 (safetensors name tables)
modules/minimax_h3_resident_media_stage/
  Makefile                                            (MODULE_FAMILY=minimax_h3, rules.mk pattern, CUDA_ARCH sm_121a)
  include/sparkpipe/spark_minimax_h3_resident_media_stage_firmware.h
  include/sparkpipe/spark_minimax_h3_serving_adapter.h
  source/spark_minimax_h3_stagepack_format.h          (enum+shape table, drives loader+pack_synthesize)
  source/spark_minimax_h3_resident_media_stage_module.c
  source/spark_minimax_h3_resident_media_stage_cuda.cu
  source/spark_minimax_h3_serving_adapter.c           (job spool, step pump, receipts)
  tools/minimax_h3_pack_synthesize.c
  validation/spark_minimax_h3_reference.c             (CPU oracle — own math from pinned diffusers source, NO driver imports)
  validation/spark_minimax_h3_resident_media_stage_cuda_validation.cu
  validation/validate_minimax_h3_resident_media_stage_cuda.sh
tools/minimax_h3_stagepack.py                         (slicing-only bf16 packer, ≤1GB RSS streaming)
model_contracts/h3.json, h3_authoritative.json        (geometry from landed configs + pinned revisions)
```

Donor ports via the rename porter (`tools/dev/port_family.py`, as used in the
driver campaign — confirm its path in the shared checkout; if absent, the
rename is mechanical): **encoder ← `qwen4_flash`** family/module structure
(nearest shape: GQA + RMS + fused QKV; deltas = geometry tables, interleaved
rope theta 5e6, prefill-only + final-hidden harvest via the existing
`hidden_output_address` boundary mechanism, no sliding window, no decode
chain). DiT/VAE kernels are new (no donor exists for diffusion) — the
instantiation/launch *pattern* donor is `inference/llms/mimo_2_5/unity.cu`.

Shared edits (each is a shared-code flag with cross-model justification):

1. `node/model_api.c`: + `/v1/videos` routes (capability-discoverable from the
   deployment descriptor; no family names — generic media-job door).
2. `docs/STAGEPACK_NAMING.md` + `tools/stagepack_naming.json`: registry token
   `h3` (arm `h3.bf16.tp4`, packs `h3.bf16.tp4.rank<h>.sp` + `.sha256`).
3. Root `Makefile` / `sources.mk`: module target + module_library entry.
4. `examples/model_descriptions` JSON + `PACKAGE_MANIFEST.json` +
   `SHA256SUMS` regenerated LAST (standard end-of-lane mechanics).
5. Port base ledger line for 64800 (COORDINATION/ENGAGEMENT notes).

No changes to: batch engine, resident IPC, adapter ABI structs, collective
engine, pack loader, weightd. Zero new work kinds, zero new capability bits
(`FLAG_MODEL_EXTENSION` already exists and is in `KNOWN_FLAGS`).

## 5. Data plan

- **Arm**: `h3.bf16.tp4` — bf16 is the source precision; **never requantize**
  (operator law). Packer slices/shards only.
- **Packed set** (≈127GB total, ≈32GB/rank): encoder text tower (63G minus
  vision tower), DiT full (62G), video-VAE decoder + conv head (≈4G of the
  9.8G; encoder arm dropped), audio-VAE decoder (0.6G, replicated). Excluded:
  `transformer_ref`, vision tower, VAE encoder arm, FL2VA/Ref2VA bundles
  (≈338GB deliberately not packed).
- **Pack shape**: family stagepack format v1, four sections (encoder/dit/
  video_vae/audio_vae), enum-driven shape table in
  `spark_minimax_h3_stagepack_format.h`; one `rank<h>.sp` per rank; weightd
  attach + lease per the standard seam (restart contract comes free).
- **Packer tooling law**: `tools/minimax_h3_stagepack.py` streams
  safetensors tensor-by-tensor (header-safetensor offsets; no whole-shard
  reads), RSS ≤ 1GB, per-stage fan-out via one queue task (qmax_build2.sh
  shape), `.receipt.json` per pack, sha256 sidecars, two-pass placement proof.
- **Source identity**: top-level `transformer/` vs `FL2VA/transformer/` must
  be verified identical (file digests, one cheap comparison) before packing —
  guards against packing the wrong branch copy.
- **Runtime**: all weights memory-resident (the platform's residency model;
  promotion <1min law is trivially met at 32GB/rank). Latents are per-job
  device buffers (≈10⁴ tokens × 5376 × 2B ≈ 100MB/step, video+audio latent
  tensors a few hundred MB) — no KV streaming, no weight streaming in the
  serving path.
- **Spool**: `<runtime_root>/media/jobs/` — job JSON (params, state, step k/N,
  receipt) + media file; rank-0-writer, atomic rename, fail-closed on sha
  mismatch.

## 6. Vertical slice definition

**Smallest end-to-end path: one prompt → one 4-second clip with audio.**

Stated slice geometry (product defaults stay in config; the slice run states
smaller numbers): `duration_seconds=4`, `fps=24`, `short_edge=480`,
`aspect_ratio=16:9` → 480×832 px; latent grid ≈ 25 × 15 × 26 → ≈9.8k video
tokens + 160 audio tokens + ≤512 text tokens ≈ **10.5k tokens/step**; denoise
steps = pinned reference default (expected 50; read from
`MiniMaxH3ModularPipeline` defaults at pin time); CFG per reference default.

Acceptance numbers (measured on the 4-spark cell, not promised):
- E2E wall for the slice job ≤ 10 min (back-of-envelope ≈ 0.8 PFLOP/step ×
  50 steps at ≈240 effective TFLOPS ⇒ 3–6 min; gate at 10).
- Telemetry: dashboard shows the 4-rank deployment busy for the run;
  steps/s recorded; `nvidia-smi` + dashboard line = proof pair.
- Artifact properties exact: 96 or 97 frames @ 24fps, 480×832, 32kHz stereo
  wav, duration 4.0s ±1 frame.

## 7. Validation plan

Anchors are independent: the CPU oracle and CUDA checks implement **the pinned
diffusers source math** (`diffusers.ModularPipeline` +
`MiniMaxH3Transformer3DModel` + `AutoencoderKLMiniMaxH3(Audio)` +
`MiniMaxH3Scheduler` at recorded revisions), never driver imports. Record
pinned revisions + landed-file sha256s in `h3_authoritative.json` (source
identity law). All CPU-class gates and ≤10GB CUDA checks run on **sparke**
(its local GB10 for CUDA), never the mac; full-DiT runs only on the 4-spark
cell via the queue.

- **V1 pin + identity**: revisions, config digests, `transformer/`≡`FL2VA/transformer/` digest equality.
- **V2 CPU oracle** (`validation/spark_minimax_h3_reference.c`): one DiT block
  + time-embed + scheduler step + VAE transformer-head block + conv head at
  tiny stated geometry, synthesized weights, scalar C. Gate: CUDA (sparke,
  ≤10GB, subset tensors) vs oracle within the family numerical-gate
  tolerances (pattern: `docs/GLM_NUMERICAL_GATES.md`; bf16 latent tensors
  rel ≤ 2e-2, per-tensor abs gates stated in the contract).
- **V3 real-weights block check**: DiT block 0 loaded from real shards on
  sparke (≈2.4GB bf16, inside 10GB) vs diffusers block-0 forward, same input:
  rel ≤ 1e-2. Encoder: text-tower final-hidden vs transformers reference on a
  ≤1k-token prompt (tower-only ≈ within 10GB with offload-free slicing? if
  not, gate on 8 of 64 layers + rank-0 full-tower on the cell).
- **V4 VAE decode**: fixed latent → diffusers decode vs ours, final pixels
  abs ≤ 2/255; audio wav sample-level tolerance stated in contract.
- **V5 determinism/TP equality**: same seed 2× ⇒ **bit-identical** latents
  (tree allreduce is deterministic); 2-block mini-DiT TP4 vs TP1 ⇒
  bit-identical (allreduce order fixed by the tree).
- **V6 E2E cell run** (section 6 numbers) + cancellation (RELEASE mid-run →
  state `cancelled`, no artifact) + oversize-prompt rejection (stated
  `max_prompt_tokens`, fail-closed 400).
- Gates: `make offline-gates` by exit code; code-size ratchet; receipts in
  `docs/AGENT_LANE_BRIEFS/reports/minimax-h3-*.md`.

## 8. Shared-code flags

1. `node/model_api.c` media-job routes — generic, capability-discoverable, no
   family names (dry-law clean); tested with a contract-exact mock the same
   way the LiteLLM passthrough was.
2. Stagepack registry token `h3` (naming doc + json map).
3. Root Makefile / module_library wiring.
4. Port base 64800 ledger entry.
5. `PACKAGE_MANIFEST` + `SHA256SUMS` regenerate last.
Nothing else. Engine, IPC, adapter ABI, collective engine, pack loader,
weightd: untouched (verified: `FLAG_MODEL_EXTENSION` ∈ `KNOWN_FLAGS`;
`model_extension` present on both submission and completion; engine bounds
kinds 1..3 which is exactly PREFILL/DECODE/RELEASE = admit/step/release).

## 9. Risks and the three hardest problems

**Hardest three:**
1. **Exact 3D rope + token-order semantics** (video/audio/text interleaving,
   axial rope split of rope_freq_dim 16 across T/H/W, clip_length 17 /
   token_drop 3 windowing in the *decode* direction). None of this is in
   configs — it lives in the pinned diffusers source. Wrong = garbage video
   that still passes shape checks. Mitigation: V2/V3 gates compare against
   real diffusers forwards, not just shapes; the coder reads the pinned
   source line-by-line before writing kernels (operator ruling 2).
2. **Dense attention kernel** — first S×S non-causal kernel in the tree
   (S≈10–20k). Correctness is gate-covered (V2/V5); perf risk is bounded by
   the fact that linears dominate FLOPs at these shapes.
3. **VAE decode fidelity at bf16** — conv stacks with reflect padding,
   groupnorm, transposed convs, snake activations; error compounds through
   the 36-layer head + conv chain. Tolerance gates (V4) with per-stage
   checkpoints; fp32 accumulation in conv/groupnorm reductions.

**Other risks:** CFG/empty-prompt convention not in configs (read pipeline
source; state it); mp4 muxing depends on node ffmpeg (fail-closed raw+wav
fallback — flagged, revisit-able); long prompts (H3-Context-IR style ≈5k
tokens) drive encoder prefill cost — stated cap, reject beyond; HF source for
the exact pipeline (library `minimax-h3`) may live outside diffusers core —
pin whatever the publisher's docs reference (SGLang support + diffusers
modular classes) and record the resolution in the contract.

**Irreversible/flagged:** none blocking. The pack-component selection
(decoder-only VAE, text-tower-only encoder) is a repack to change. Port base
64800 needs ledger coordination (cheap). API shape mirrors the publisher's
`/v1/videos` — an OpenAI-videos alias later is trivial.

## 10. Acceptance criteria for the coder (ordered, testable)

1. `h3_authoritative.json` exists; contains every constant in section 2/3
   with the landed file's sha256 + pinned diffusers/transformers revisions;
   `transformer/` ≡ `FL2VA/transformer/` digest proof recorded.
2. `model-families/minimax_h3/` headers compile standalone (`cc -std=c11
   -Wall -Wextra -Werror`), static asserts bind every packed tensor shape
   from section 3 to the stagepack format enum table.
3. Stagepack format header compiles with its test main; pack_synthesize
   builds; `tools/minimax_h3_stagepack.py` produces a rank-0 pack for the
   DiT section from real shards on sparke with RSS ≤ 1GB (measure with
   `/usr/bin/time -v`), receipt + sha256 sidecars present.
4. CPU oracle passes its own self-check (synthetic tiny geometry) and V2
   CUDA-vs-oracle gate green on sparke (≤10GB, exact tolerance table in the
   contract).
5. V3 real-weights DiT block-0 gate green on sparke (rel ≤ 1e-2 vs pinned
   reference at identical inputs).
6. V4 VAE decode gates green (video ≤2/255; audio stated tolerance).
7. Module builds through `tools/module_build_release.sh` shape; validator
   receipt recorded; `make offline-gates` exit 0; code-size ratchet clean.
8. V5: 2× same-seed bit-identical latents; TP4-vs-TP1 mini-DiT bit-identical.
9. Four-spark cell: packs placed (two-pass placement proof), deployment up
   4/4 ranks, `/v1/videos` accepts the slice job, job reaches `completed`,
   artifact properties exact (section 6), wall ≤ 10 min, dashboard + nvidia-smi
   proof pair captured.
10. Cancellation + oversize-prompt fail-closed tests green; report with all
    receipts in `docs/AGENT_LANE_BRIEFS/reports/minimax-h3-<date>.md`;
    manifest+sums regenerated last.

Out of scope (explicitly): fl2va, ref2va, 2K regeneration, image inputs,
streaming/partial delivery, multi-job batching beyond CFG pairs, the 5090
box, any quantized arm.
