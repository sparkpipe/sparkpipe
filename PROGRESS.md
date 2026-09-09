# MiniMax H3 lane progress (lane/minimax-driver)

Lane: minimax-h3 resident media stage (t2va vertical slice).
Branch: `lane/minimax-driver` @ origin/main 8f3a6f2. Contract: `DESIGN.md` (untracked, in-tree).
Weights: `/mnt/model-warm/minimax-h3` on sparke (465G, PUBLISHED, WARM-COPY-COMPLETE).

Manager rulings applied on top of DESIGN.md:

1. Port base **65000** (DESIGN text says 64800; ledger: 64800 laguna, 64900 ling, 65000 minimax).
2. New kernels (bidirectional S×S flash attention, 3D axial rope, adaLN glue, VAE conv heads,
   flow-match scheduler math) are module-local in
   `modules/minimax_h3_resident_media_stage/` with clean generic names; promotable later,
   not promoted preemptively.
3. Operator ABI ruling: vertical slice builds on the existing step-pump ABI unchanged;
   the seam (per-step submission/completion round-trip vs step compute, extension payload
   at admission, receipt shape, spool I/O cadence) is instrumented and MEASURED numbers
   land here; any measured loss gets a minimal additive version-gated ABI extension,
   quantified and flagged — never a silent engine redesign. Same treatment for the
   queue-window/TTL vs multi-minute video-job lifetime question at criterion 9.
4. Donor porter: `/Users/mac/batch-ling/tools/dev/port_family.py` (verified present).
5. ffmpeg: fail-closed raw+wav fallback (`.tar` of `.yuv` + `.wav`) when node ffmpeg absent.
6. OPERATOR TOPOLOGY RULING (mid-lane): fill 16 sparks; TP16 preferred, TP4×PP4 is the
   standard, exemptions need a measured advantage. Folded in BEFORE placement (no packs
   existed yet).

## Topology decision (ruling 6): TP4×PP4 on 16 sparks — arm `h3.bf16.tp4pp4`

Numeric reasoning (single media job, batch=1, S≈10.5k tokens, hidden 5376):

- Per-rank compute wall is IDENTICAL for TP8×PP2 and TP4×PP4: 2 serial stages × 8 ranks
  and 4 serial stages × 4 ranks both put F/16 of the step on every rank. TP16 fails the
  head split (56 heads / 16 = 3.5) and needs head_dim splitting — rejected up front.
- The whole difference is traffic. One activation payload is 113 MB
  (10500 × 5376 × 2 B); a denoise step carries 104 tree allreduces (2 per DiT block).
  TP8 tree = 3 hops vs TP4 tree = 2 hops ⇒ TP8×PP2 costs 104 × 113 MB ≈ **11.7 GB/step
  extra allreduce traffic**. TP4×PP4 pays 2 extra inter-stage handoffs ≈ **0.226 GB/step**.
  TP4×PP4 wins the trade by ~50×.
- TP4×PP4 keeps the platform's proven TP4 collective shape and reuses the existing
  `glm5_next` TP4×PP4 deployment-generator precedent; TP8×PP2 would need new generator
  machinery for no measured advantage.
- Shard exactness: DiT 56/4 = 14 heads/rank, 52 blocks / 4 = 13/stage; encoder 64q → 16,
  8kv → 2 per rank; video-VAE transformer 32 → 8 per rank. All exact.
- Per-rank RAM: stage 0 ≈ 19.6 GB (encoder 63 G/4 + 13 DiT blocks ≈ 3.9 GB), stages 1–3
  ≈ 3.9 GB — far under the 110 GiB ceiling; 16 sparks carry a uniform load.
- Stagepack naming follows the arm: `h3.bf16.tp4pp4.rank<h>.sp`, h = 0..15; registry
  token `h3` unchanged; family stagepack format v1 unchanged (format is rank-count
  agnostic).
- Measured follow-up stays honest per ABI ruling 3: criterion-9 instrumentation records
  per-step submission/completion round-trip vs step compute; if PP handoff or TP4
  collective cost shows a measured regression vs the projection, the number lands here
  and the exemption question is reopened with data.
- The 4-spark TP4×PP1 island is dropped (weights at 32 GB/rank × 4, 12 sparks empty —
  exactly the flagged case).

## Pinned-source contract (diffusers 3c221246, minimax_h3 modular pipeline) — read line-by-line

These facts come from the pinned source and OVERRIDE the design's guesses where they
differ. They bind the kernels (criterion 4+).

- **No CFG.** The checkpoint is guidance-distilled: no negative prompt, no unconditional
  branch, one forward per step (`denoise.py`, `modular_pipeline.py`).
- **Packed sequence order**: `[text | keyframe conditions | target audio | target video]`
  — audio BEFORE video. t2va has zero conditioning rows. Audio rows are channel-major:
  channel 0 latents then channel 1 latents (stereo = 2 mono batch items at the VAE).
- **Modality tags**: text=1, video=0, audio=2.
- **t convention**: t=1 clean, t=0 noise; `x_t = t·x0 + (1−t)·noise`. Scheduler: sigma
  grid = linspace(1,0,N) → σ' = s·σ/(1+(s−1)·σ), s=12 video / 3 audio; timesteps =
  1−σ[:-1]; N sigma points ⇒ N−1 model evals. Step: `x0 = x_t + (1−t)·v` (data-ward
  velocity, PLUS sign), then `x_next = r·x_t + (1−r)·x0`, r = σ_next/σ, in float32.
- **Text conditioning**: prompt verbatim, no chat template, no special tokens
  (add_special_tokens=False); harvest = `hidden_states[50]` (output of the 50th decoder
  layer; the final layer is post-norm and NOT the conditioning).
- **Rope**: one fp32 inv_freq[16] = 1/θ^(arange(0,32,2)/32), θ=1e4, shared by t/h/w.
  Per row: freqs = fp64 positions (cast fp32) × inv_freq per axis → concat(t,h,w)[48] →
  concat(self)[96] cos/sin. Rotates the FIRST 96 of 128 head channels, rotate-half
  convention (x1=[:48], x2=[48:96], out = x·cos + cat(−x2,x1)·sin). Positions fp64:
  text rows t=row index; video t from non-uniform grid 5/3·(1,4,4,4,4) cumulative from
  num_text_tokens; spatial axes aspect-normalized: linspace((1−ratio)/2, +ratio,
  dim/patch, endpoint=False)·32 with ratio = dim/sqrt(h·w); audio rows t = num_text +
  arange(latents) repeated per channel, h=0, w pinned to width grid extremes.
- **adaLN**: per block `adaln_proj.linear` [96768,2688]+bias on `silu(temb)` cast to
  bf16; output [T,96768] → [T·3, 6·5376] → six [T·3,5376] (shift_msa, scale_msa,
  gate_msa, shift_mlp, scale_mlp, gate_mlp); row = timestep_idx·3 + tag. temb:
  diffusers Timesteps(256, flip_sin_to_cos, shift 0) on UNSCALED t∈[0,1] → fp32
  TimestepEmbedding (linear_1 [5376,256] → silu → linear_2 [2688,5376]).
  `norm_out`: RMS(h)·(1+scale)+shift with shift FIRST chunk of [10752,2688] linear on
  silu(temb), indexed by timestep_idx (no modality).
- **Attention**: no biases, RMS q/k per head BEFORE rope, softmax scale 1/sqrt(128),
  non-causal, no mask, one document.
- **Mixed precision is part of the checkpoint**: proj_in, audio_proj_in, time_embedder,
  proj_out, audio_proj_out (+ the whole video/audio VAE) are FLOAT32; block stack +
  context_embedder bf16. Packs carry F32 entries for those tensors — never requantize.
- **Token refiner**: 2 plain pre-norm blocks (no rope, no adaln) + final RMSNorm,
  applied to context_embedder(text) before the scatter.
- **Noise**: video randn [1,24,T,H,W] fp32 drawn FIRST from the request generator,
  then audio randn [latents·2, 32] fp32 in row layout. Initial state = pure noise
  (first timestep t=0).
- **Canvas/frame law**: canvas_multiple = 32; short_edge 480 → 480×864 (not 832).
  num_frames snaps UP to 17n+5 (clip_length 17, tokens_chunk 5); latent frames =
  5n+2; duration floor is 5s. Slice: request 5s → 120 → 124 frames → 37 latent
  frames → grid 37×15×27 = 14985 video rows; audio 207 latents × 2 = 414 rows;
  ≈15.9k tokens/step with a 512-token prompt. Artifact: 124 frames @ 24 fps,
  480×864, ~165.6k audio samples/channel.
- **Decode**: denorm latents (×std+mean) → VAE decode (reference runs fp16 autocast;
  we keep bf16 compute with fp32 accumulation and gate on tolerance) → ImageNet
  denorm → clamp [0,1]. Audio: denorm → mono decode ×2 batch → interleave to stereo
  at 32 kHz.

## Work log

- 2026-09-09: lane start. Previous coder died pre-write (verified: `git status` clean at
  8f3a6f2, only DESIGN.md untracked). Survey done: serving adapter ABI v22 read in full
  (`include/sparkpipe/spark_model_serving_adapter.h`), batch engine read in full
  (`runtime/model_batch_engine.c` — kind round-robin, extension slots on submission +
  completion, `FLAG_MODEL_EXTENSION` in KNOWN_FLAGS), donor module
  `modules/qwen4_flash_resident_decode_stage` (Makefile, adapter, stagepack format header
  pattern), shared stagepack common (`include/sparkpipe/spark_stagepack_format.h`),
  `model_contracts/qwen4_flash_authoritative.json` shape, queue v2 contract.
- 2026-09-09: dispatched `minimax-inv-001` (sparke, cpu, 15 min): warm-copy layout,
  receipts, all component configs, transformer≡FL2VA/transformer digest proof, per-file
  digests, python env + minimax-h3 library revision discovery, publisher scripts,
  tokenizer location. Feeds criterion 1 (h3_authoritative.json pins).

## Criterion status (DESIGN.md §10)

| # | criterion | status |
| --- | --- | --- |
| 1 | h3_authoritative.json + identity proof | in progress (inv job on sparke) |
| 2 | family headers compile, static asserts | in progress |
| 3 | stagepack format + synthesize + real-shard rank-0 pack ≤1GB RSS | pending |
| 4 | CPU oracle + V2 gate | pending |
| 5 | V3 real-weights block-0 gate | pending |
| 6 | V4 VAE decode gates | pending |
| 7 | module build + offline-gates + ratchet | pending |
| 8 | V5 determinism + TP4≡TP1 | pending |
| 9 | cell E2E slice + instrumentation | pending |
| 10 | cancellation + oversize fail-closed + report + manifest last | pending |

## ABI seam instrumentation (ruling 3) — to be filled with measured numbers

- per-denoise-step submission/completion round-trip vs step compute: TBD
- model_extension payload handling at admission: TBD
- receipt shape: TBD
- spool I/O cadence: TBD
- queue-window/TTL vs video-job lifetime (criterion 9): TBD
