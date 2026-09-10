# MiniMax H3 lane progress (lane/minimax-driver)

Lane: minimax-h3 resident media stage (t2va vertical slice).
Branch: `lane/minimax-driver` @ origin/main 8f3a6f2. Contract: `DESIGN.md` (untracked, in-tree).
Weights: `/mnt/model-warm/minimax-h3` on sparke (465G, PUBLISHED, WARM-COPY-COMPLETE).

Manager rulings applied on top of DESIGN.md:

1. Port base: **parameterized, pending fleet renumber**. History: 65000 (ledger) →
   collided with ling's TP16 session matrix 64900..65155 (session_ports[a][b] =
   BASE + a*TP + b is degree² wide) → 65160 rejected too (route-kind offsets
   D2A +256 / TREE_ACK +512 / D2A_ACK +768 with a hard 65535 rejection put ACK
   variants at 65687; the whole region above ~64700 is structurally out of room).
   The deployment generator takes the base from the environment with NO frozen
   numeric default; the manager is escalating the allocation plan to coredev.
   Ledger entries are width-aware from here on: a TP4 deployment reserves
   TP² = 16 session ports plus the route-kind offsets.
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

- 2026-09-10 (round 4): ADOPTION + REBASE CHAIN. Adopted the coredev-aligned
  origin/lane/minimax-driver 5462630 (reset onto it; family content verified intact —
  audio-census 914 static assert + all v3/v4 anchor fixtures survived; the only family
  delta was the module Makefile dropping the deleted nccl source). First rebased onto
  the transport-rewrite main 14df85a, then SUPERSEDED: PR #913 landed main 50bd0d3
  (E2E-proven mesh dataflow, direct-broadcast B1 allreduce, old-engine test deletions)
  and the lane was rebased onto it (lane tip at rebase: 5d2bdca). Shared files toward
  main; the aligned branch's mesh cherry-picks were skipped (main carries the proven
  originals). Ceiling re-measured twice: 237412 on the aligned tip, 238164 on the
  50bd0d3 rebase. TRANSPORT FINDING on the adopted tree: the collective data path is
  weightd-owned mesh QPs over shared memfd slot bands — tp_device_collective.c no
  longer dials session ports at all; the config parser still REQUIRES session_ports
  matrices for the hidden_transport backend (serving_adapter_template.c treats an
  absent member as SCHEMA_ERROR), so the deployment generator keeps the frozen official
  block 17408-18431 but those ports are parse-compat only, not dialed. C7 root wiring
  landed (Makefile contract/archive/publish hooks mirroring the glm52 flow). C8 V5
  mini-DiT determinism gate written (2x same-seed LCG latents through 2 scheduler
  steps x real blocks 0,1; TP1 vs TP4-segmented GEMM compared bitwise per step).
  V3 GATE COMPILE REPAIR: r3's final scratch-struct refactor left device_segments in
  three BlockForward call sites (nvcc: too many arguments) — fixed; the refactor was
  never compile-verified in r3. V4 VIDEO BLOCKER (anchor-side, not ours): the anchor
  lane's own vae_video_real.npz carries 67584/67584 nonfinite values in the decoded
  tensor (verified against batch-minimax-val's npz, not a transfer artifact) — the
  video gate cannot bind until that lane regenerates; the gate now fails closed
  naming this attribution. V4 AUDIO: anchor waveform fixture is finite; last r3
  receipt (v3gate13) showed rel=1.0 vs expected AFTER the convs2 residual fix was
  already applied — the fix did not close the gap or was never exercised; re-run
  pending on the repaired tree. QUEUE FINDING: spark_queue dispatch claims only
  kind=run jobs — r3's kind=gate submissions (minimax-c5-v3gate1 et al.) sat queued
  forever and were silently lost by the serving loop; all lane jobs from now on are
  kind=run. Also: job cmds must end with the real gate status propagated (a trailing
  echo masks the gate exit; read the Vx_EXIT line in the node log).
- 2026-09-10 (round 3, coordinator directive): OPERATOR MEMORY MECHANISM — all heavy-IO
  node commands (gate validators, shard slice reads, pack runs) now run under
  `sudo -n sparkcap [--mem MB] cmd` (sysadmin /usr/local/sbin/sparkcap on all sparks;
  cgroup v2 MemoryMax default 4096MB + 70% MemoryHigh, page cache counted). Inside a
  queue job the invocation must be `sudo -n sparkcap`: scopes cannot nest, so the
  unprivileged systemd-run --scope fails with "Interactive authentication required";
  under sudo -n sparkcap resolves SUDO_USER and creates the scope root-side. Queue
  jobs for this lane wrap the gate scripts from round 3 onward. Tooling RSS receipts
  already measured ≤1GB stay well inside the default cap.
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
- 2026-09-10 (round 2): C3 closed (d07e51a receipt + audio-count reconciliation;
  PROGRESS.md table updated). Slice geometry reconciled to the pinned contract
  (fe80d68) — DESIGN §6's 480x832/96f guess was pre-source and the header asserts now
  bind the 17n+5 law itself. C4 landed: oracle full V2 surface (9c88261), kernels +
  validator + module host source (098502f), then three sparke-driven fixes: sha256sum
  portability (ee28237), nvcc C++17 header compatibility + gate-residual shadow
  (6786e8e), oracle include cleanup (2366558), scheduler extern-C guard (f3d86d3),
  honest input scale + 2-ulp scheduler gate (c090041). V2 green as minimax-c4-v2-gate6.
  Lane flow note: source now stages via /mnt/model-warm/staging/minimax-lane/<sha>.tgz
  (scp + git archive) instead of ~100KB base64 cmd payloads — queue cmds stay tiny and
  the 15-min TTL no longer bounds payload transfer. The TTL cap (15 min, queue rejects
  anything larger) remains the binding constraint for long jobs and is the C9 data
  point: a multi-minute video job cannot own a 15-min window end-to-end with setup;
  the detached-chain/cursor pattern (round 1) or an idle-window reservation is needed.
  C4 sparke runs observed: core+runtime lib build ~fast on GB10 (-j4, well under TTL);
- 2026-09-10 (round 3): ANCHOR FINDINGS absorbed and acted on. (a) AUDIO CENSUS:
  header-level census job on sparke (minimax-census1) gave audio_vae shard = 1087
  tensors = 914 decode path + 22 pre_block + 4 mean/logs_proj + 147 encoder.* — the
  decode-path pack count is now BOUND at 914 by static assert; pre_block is encode-only
  (pinned AutoencoderKLMiniMaxH3Audio.encode calls it, decode never does — proven
  bit-exact by the anchor decode without it); RESBLOCKS_PER_STAGE 9→3 (21 resblocks,
  6 alias-free activations x 4 tensors + 6 weight-norm convs x 3 = 42 each);
  conv_post has no bias. The C kind enum was ALSO wrong in a way round 2 could not
  see: sparse hand codes (DIT_PROJ_IN 0x2003) never matched the packer's dense
  appearance-order assignment (0x2000 in every written pack) — enum re-derived from
  tensor_patterns.json, PRE_* kinds deleted, dry-run census green on sparke
  (minimax-census2/3: 705/638/585/914, strict fail-closed coverage, 22.6 MiB RSS).
  Packer now fails closed on any checkpoint tensor matching no pattern and no
  exclusion. (b) transformers pin re-pinned to 4815a0a6 (5.18.0.dev0) per anchors.
  (c) C5 V3 gate built: 4 new module kernels (weighted RMS norm, deterministic
  K-segment GEMM whose fp32 partials combine ((s0+s1)+(s2+s3)) so a 4-way TP
  partition is bit-identical to TP1 by construction, row-indexed adaLN affine and
  gate-residual) + spark_minimax_h3_v3_gate.cu consuming the anchor fixture
  fixtures/real/dit_blocks01_real.npz (converted to raw LE binaries and committed).
  (d) C6 V4 gates built: video ViT decoder (36 blocks, biased attn, bare qk rms-norm,
  48-dim rope theta 100, scale1/scale2 sandwich, chunk assembly 7 tokens -> 22
  frames) and audio BigVGAN decoder (legacy weight_norm, inverted channel ladder
  1024->8 while time x800, 21 AMP resblocks with 6 alias-free activations each,
  checkpoint kaiser filters, bias-free conv_post) vs the real anchor fixtures.
  (e) CRITICAL swiglu fix: diffusers SwiGLU = first_chunk * silu(second_chunk); the
  round-2 oracle and CUDA kernel silu'd the FIRST chunk (self-consistent, so the
  kernel-vs-oracle V2 gate could not see it); oracle, kernel, and V4 validator all
  corrected — the real-fixture gates are the checks that bind. (f) C9: deployment
  generator tools/minimax_h3_gen_deployment.py pins the FROZEN official block
  17408-18431 (PORT_LEDGER v2): session cells BASE+group*64+a*4+b, max used port
  with +768 route offsets = 18383, generator refuses non-official bases; 16 stage
  configs + deployment_manifest.json committed. Live 16-spark cell + seam numbers
  remain (needs the 135GB pack set placed — the pending C9 work).
  module archive + validator run well under TTL.

## Criterion status (DESIGN.md §10)

| # | criterion | status |
| --- | --- | --- |
| 1 | h3_authoritative.json + identity proof | **DONE** — commit c590436; identity REFUTED and recorded (per-tensor proof, pack top-level transformer/ only); revisions pinned (HF 42ed227e cross-evidenced, diffusers 3c221246, transformers 4.57.0.dev0); text_encoder shard digests still to append at pack time |
| 2 | family headers compile, static asserts | **DONE** — 8ecf333 + 39081ce; cc -Werror clean; 61 kinds resolve; counts bind measured census (705+638+585+936) |
| 3 | stagepack format + synthesize + real-shard rank-0 pack ≤1GB RSS | **DONE** — format header + pack_synthesize + packer committed; rank-0 DiT-section pack from real shards: 187 tensors, 12378247520 bytes, sha256 9edfe594…, peak RSS 617.0 MiB (≤1GB law), 19.6s elapsed (~590 MB/s once cross-lane pool contention cleared; the 2.8 MB/s probe was contention, not the pool's rate). Audio count reconciled: packer-measured 936 is authoritative, 937 overcounted the pre_block norms (d07e51a) |
| 4 | CPU oracle + V2 gate | **DONE** — oracle full V2 surface (adaLN affine, gate-residual, silu-mul, conv2d 3x3 reflect, dilated conv1d, snake + round-1 rope/attention/scheduler) with self-check green; 8 module-local CUDA kernels (rope3d, dense bidirectional flash-style attention with DETERMINISTIC accumulation — no atomics, adaln affine, gate-residual, silu-mul, conv2d3x3 reflect, conv1d dil 1/3, snake) + validator; **V2 gate GREEN on sparke GB10** (queue minimax-c4-v2-gate6, exit 0): all 9 kernel comparisons within rel_l2≤2e-2 / cosine≥0.9999 (worst rel_l2 0.0018, worst max_abs 0.0077 at unit scale); scheduler parity shift 3.0 bit-identical, shift 12.0 within 2 ulp (cc vs nvcc host fma contraction — honest-labeled, not memcmp-faked); slice geometry reconciled to pinned contract 480x864/124f/37x15x27/414 audio rows (fe80d68) |
| 5 | V3 real-weights block gate | **BLOCKED ON ANCHORS** — the independent anchor-debugger lane (batch-minimax-val) has staged `validation/anchors/{pinned_source,landed_snapshot}` but no diffusers-derived fixture tensors yet. Fixture contract (proposed, for that lane to produce; do NOT let this lane write its own diffusers reference — independence is the point): `validation/anchors/fixtures/h3_v3/` containing `block0_io.npz` (keys: `hidden_in` [S,5376] bf16/f32, `temb` [2688] f32, `hidden_out` [S,5376] f32, for one real DiT block 0 forward at a stated seed/timestep) + `encoder_last_hidden.npz` (prompt ids + `hidden_states[50]` harvest) + a `manifest.json` (diffusers commit, input hashes, tolerances). Our consumer gate: load fixture, run OUR block-0 path on the same bf16 inputs, rel ≤ 1e-2. Structure the validator to read these files when present; until then V3 stays blocked and every other criterion proceeds. |
| 6 | V4 VAE decode gates | pending (video ≤2/255, audio stated tolerance; needs V3-class anchors for the decode path + our decode kernels) |
| 7 | module build + offline-gates | **PARTIAL** — module archive builds through the repo flow on sparke (`make archive`, nvcc sm_121a object in archive, proven by the V2 runs); remaining: root module_library wiring + `make offline-gates` exit 0 + code-size ratchet |
| 8 | V5 determinism | pending — attention kernel accumulation order is deterministic by construction (per-element serial tile sums, no atomics); 2x same-seed latents + TP4-vs-TP1 mini-DiT need the cell |
| 9 | cell E2E | pending — 16-spark TP4×PP4, port base env-parameterized pending fleet renumber; ABI seam instrumentation numbers to be captured at the live run |
| 10 | fail-closed tests + report + manifest | pending |

## ABI seam instrumentation (ruling 3) — to be filled with measured numbers

- per-denoise-step submission/completion round-trip vs step compute: TBD (C9)
- model_extension payload handling at admission: TBD (C9)
- receipt shape: FLAG_MODEL_EXTENSION model_extension[512] blob = {artifact path id,
  sha256[32], byte count, width/height/frames/sample-rate} — layout to be frozen in
  spark_minimax_h3_serving_adapter.h
- spool I/O cadence: rank-0 adapter writes job JSON at admission + per-step (state,
  step k/N) + terminal receipt; media file written once, atomic rename
- queue-window/TTL vs video-job lifetime: measured pool contention (2.8 MB/s) already
  forced the ttl-15 detached-chain pattern for a single rank pack; a full 16-rank pack
  set (135 GB) at contended rates needs either an idle-pool window or the chain pattern
  at scale — record actuals when the pack chain completes

## Handoff notes (next coder session)

- Pinned-source contract section above is the ground truth for all kernels; the
  FP32-module list (proj_in, audio_proj_in, time_embedder, proj_out, audio_proj_out,
  both VAEs) means those pack entries carry weight_format F32 (1), everything else BF16.
- The module host source + CUDA kernel file + serving adapter do not exist yet; the
  Makefile already declares their paths. The oracle primitives in
  modules/minimax_h3_resident_media_stage/validation/spark_minimax_h3_reference.c are
  the V2 comparison target — the CUDA validation includes the same math at identical
  inputs and compares to the GLM numerical gates (bf16 rel ≤ 2e-2 for latents).
- Audio tensor count: fix SparkMinimaxH3StagePackAudioTensorCount 937 -> 936 and re-run
  the format test (the header's expected tensor_count assert must match the measured
  census; the discrepancy is in the audio globals — recount from the shard dump in the
  lane transcript before editing).
- The /v1/videos door (node/model_api.c shared edit) is not started. Receipt layout
  note above is the design; freeze it when writing the adapter header.
- transformer_ref/ and FL2VA remain untouched, per design.
