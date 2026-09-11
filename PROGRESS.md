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

- 2026-09-11 (round 7): REAL TP16 PACK EMIT (in flight) + TWO EMIT-TIME DRIVER
  BUGS CAUGHT + C9 TABLES + FIRMWARE FLIP. Work dir moved back to
  /Users/mac/batch-minimax (r6 ran in /Users/mac/lane-minimax); lane branch
  fast-forwarded to the r6 tip 13b410f, tree staged to sparke
  (/mnt/model-warm/staging/minimax-lane/13b410f.tgz + extracted). PASS A
  placement proof GREEN: all 16 per-rank dry-run censuses print the identical
  strict census encoder 705 / dit 638 / video_vae 585 / audio_vae 914 (the r6
  note's 936 was the superseded r2 number; the header static assert binds 914
  = decode path, pre_block excluded as encode-only), stage split {0: N} on
  every rank, fail-closed extent guard silent. PASS B real emit: chain
  (nohup sparkcap per rank, skip-if-receipt-complete, per-tensor progress
  resume) wrote ranks 00-03 GREEN before two host events intervened:
  BUG 1 (commit 0e9afdd): match_name built the matched entry WITHOUT the tp16
  tag, so every heads_rows/heads_cols/kv_rows pattern degraded to plain
  rows/cols and the real emit aborted at the first encoder k_proj (1024 rows
  guard) instead of silently packing 448-row uniform slices of the 7168 DiT
  qkv — the r6 unit proofs exercised tp_slice directly and never the name
  routing; routing test added (pure-name, no warm IO, binds tag survival at
  tp16 + tp4 fallback + kv r//2). BUG 2 (commit f96bc23): read_tensor_blob
  sliced rows only and returned row_count × FULL width — (a) the payload
  region ran ~2x the directory's recorded bytes (rank00 first attempt: 63.9
  GiB file, 31.5 GiB payload sum, 304 offset jumps) and (b) every tp_rank > 0
  silently carried RANK-0's columns on all cols/heads_cols tensors (heads_cols
  start != 0 reads the wrong half of the row-major payload). Undetected
  historically because only rank-0 packs ever fed the real numeric gates and
  the r5 tp4 rank>0 packs were cancelled before use. Fix: in-memory column
  slice from a full-row-span read (rows fast path unchanged), 4-way extent in
  the entry, resume signature v2; BytesIO blob unit in the routing test.
  MEASURED SIZES with both fixes (receipts, post-fix chain): rank00 33,829,
  381,672 B = 120 header + 31.51 GiB payload + padding + 2842×56 directory —
  matches the computed layout byte-exactly; uniform 4-head ranks 01-03
  identical 33,569,479,480 B, 1928 tensors (audio rank-0-only), peak RSS
  587-610 MiB (≤1GB law), ~136 s/rank on healthy IO. FLAG (plan figure
  refuted, not tuned): per-rank packs are ~31.5 GiB, not ~8.4 GB — the r5
  135 GB total assumed adaLN sliced 1/16, but DIT_ADALN (n=100, 24.23 GiB on
  rank0) MUST stay replicated: the post-to_out allreduce replicates the full
  hidden on every rank and modulate applies elementwise over all 5376
  channels locally; slicing would add per-block collectives (rejected). Full
  16-rank set ≈ 500 GiB on /mnt/model-warm (11 T free). HOST EVENT (emit
  HALTED deliberately at r7 close): sparke crashed REPEATEDLY — reboot 00:38
  (all net paths down ~7 min; ceph warm storage survived, 4 receipts intact),
  again 01:00, again ~01:36, and a fourth down at ~02:0x ("host is down" on
  connect). Correlation flagged for the manager/sysadmin: ranks 00-03
  emitted cleanly in the single healthy window (15:22-15:31Z, ~136 s/rank);
  every post-crash rank04 restart (15:43:48Z, 16:21:34Z, 16:42:21Z) stalled
  in D-state on the warm pool within the first tensors (progress frozen at
  next_index 2-3 of 2842) and the host went down 7-20 minutes later — either
  a failing ceph client/pool on a host that was already dying, or (less
  likely but possible) the emit's ~33 GB/rank read+write with per-tensor
  os.sync() exposure triggering a client hang. The chain is left DEAD (not
  relaunched) to stop feeding a crash loop; resume when sparke is declared
  healthy is one command: rerun
  /mnt/model-warm/staging/minimax-lane/13b410f/emit_tp16_chain.sh (skips
  ranks with receipt+sha256, resumes partial ranks from progress files; the
  rank04 partial carries resume signature v2).
  Receipts so far: minimax-r7-passA (16/16 dry-run), minimax-r7-pack00..03.
  C9 TABLES LANDED (commit f5c0b02): tools/minimax_h3_gen_deployment.py
  rewritten PP1-only (TP4xPP4 placement code deleted), 16 stage configs +
  deployment_manifest.json + model_resident.json (schema v2 root: adapter/
  driver/transport host-rdma control base 60920 via MINIMAX_H3_TRANSPORT_BASE,
  per-rank tcp endpoints 19560+r, kv pages 0) regenerated under
  deploy/minimax_h3. PORT-MATRIX FINDING pinned in the manifest: session
  cells 17409..17663; cells + route-kind offset +768 reach 18431 = EXACTLY
  BLOCK_LIMIT, zero margin. And the matrix is DEAD PARSE-COMPAT on the
  post-#913 tree — verified by reading the consumers: ring/transport/
  tp_device_collective.c (413 lines, mesh_buffer slot bands) has ZERO
  session_port references; runtime/spark_weightd.c zero; the glm52 reference
  TP16 adapter zero; serving_adapter_template.c requires the members
  (absent = SCHEMA_ERROR) and SparkTpCollectiveLoadSessionPorts fills the
  topology struct, which nothing dials. Noted, proceeded per mission. FIRMWARE
  FLIP (commit dd255e0): SPARK_MINIMAX_H3_TP_DEGREE 4→16, PP_STAGE_COUNT 4→1,
  DIT_BLOCKS_PER_STAGE = the whole 50-block stack (13x3+11 split + two PP4
  asserts deleted, single-stage-coverage assert added); module
  StageBlockCount loses the stage_index<3 branch; format test green locally
  AND on sparke (57 kinds resolve), module.c strict -Werror clean on both.
  ABI-SEAM PLAN ready to fire (PROGRESS §ABI seam r7): sample struct + receipt
  shape frozen in spark_minimax_h3_serving_adapter.h, fire points = existing
  batch-engine extension slots, cell procedure = 2-step smoke then full 5s
  clip vs the 15-min window; measured numbers land at the cell round (module
  host is still the metadata stub). BOUNDARY CHECKER committed
  (tools/minimax_h3_tp16_boundary_check.py): directory extents per rank,
  payload sha of 9 representative tensors vs fresh warm slices, rank07+08
  concat proof vs warm q [3584,4480) — fires when the pack set completes.
  QUEUE untouched this round (zero jobs; direct nohup sparkcap only, per the
  ttl-min-5 contention concern). BOUNDARY CHECK GREEN on the four written
  packs (tools/minimax_h3_tp16_boundary_check.py --ranks 0,1,2,3): directory
  extents correct on every entry (rank00 2842 entries / 4-head rows 512 kv
  head 0; rank01-03 1928 entries; rank02/03 kv heads 1), and 9 representative
  tensors per rank (DiT q/k/v/to_out blocks 0+49, encoder k/v, DiT ffn
  gate_up/down) hash-match fresh warm-copy slices through the same plan
  math. NOT DONE this round: ranks 04-15 (host halted, see above), the
  rank07+08 concat proof (fires once those ranks exist), cell bring-up.

- 2026-09-10 (round 6, PAUSED mid-round per operator): REBASE + TP16 PACK SUPPORT.
  REBASE landed and pushed: lane/minimax-driver e4af723 = old bc45d75 lineage rebased
  onto origin/main f6db50a (182 main commits: muse merge, #919/#925 mesh safety,
  mgr2 waves, k3 #907-era adapter state). Resolution record: PROGRESS.md add/add vs
  muse's doc - ours kept as the lane receipt doc (muse's content preserved in main
  history; the same-path clash re-surfaces at PR merge time for the manager);
  PACKAGE_MANIFEST.json/SHA256SUMS/test_code_size.py took main's side at every
  conflict commit, then manifest+sums REGENERATED and CEILING re-pinned to measured
  245453 at this landing (gate green); Makefile auto-merged (minimax contract/archive/
  publish hooks intact, main's k3/muse wiring kept); 8 manifest/sums-only lane commits
  became empty and were skipped. K3 ADAPTER VERDICT: the lane never textually forked
  the k3 adapter (the r5 "newer discipline" note referred to the lane's own r5
  A-0041 application of the same rule) - main's post-#907/swept
  spark_k3_serving_adapter.c taken wholesale; union verified: main's swept
  spark_error_site.h kept the exact SPARK_FAIL(status_value) contract (now #pragma
  once), minimax module.c/pack_synthesize/stagepack shape protocol conform as-is;
  format test green on the rebased tree (57 declared kinds resolve, per-site
  ERREPORT lines firing). PUSH GOTCHA for every lane: /usr/local/bin/git 2.10.1
  (first on PATH) gets HTTP 400 from GitHub receive-pack; use
  PATH=/usr/bin:$PATH tools/sparkpipe_github_pat.sh git push ... (git 2.39.5).
  TP16 PACK SUPPORT landed (code complete, unit-proven, NOT yet run against the
  real shards): tensor_patterns.json gains the tp16 spec block (arm h3.bf16.tp16,
  pp_degree 1, dit_head_counts [4x8, 3x8], head_dim 128, encoder kv replication 2)
  and 10 pattern tags - DiT main-block AND token-refiner to_q/to_k/to_v = heads_rows,
  their to_out = heads_cols (columns follow the same rank ownership), encoder
  k_proj/v_proj = kv_rows (rank r carries kv head r//2; its 4 q heads 4r..4r+3 all
  sit in kv group r//2, so GQA pairing stays local and exact). tools/minimax_h3_stagepack.py
  gains the three plans, --pp-degree placement (default 4; tp16 spec's 1 collapses
  every section onto the single stage), arm-named outputs (h3.bf16.tp16.rankNN.sp +
  .sha256 + .receipt.json), tp16 placement block in the receipt, resume signature
  widened with pp_degree, and a FAIL-CLOSED extent guard: any tensor whose
  rows/columns equal the DiT qkv extent 7168 must use the head plans and any
  encoder 1024-row tensor must use kv_rows (a missed tag can no longer silently
  pack 3.5-head slices). Unit proofs (this transcript, all green): rank0 = 4 heads
  rows [0,512), rank7 = 4 heads [3584,4096), rank8 = 3 heads [4096,4480), rank15 =
  3 heads [6784,7168), to_out columns mirror the same boundaries, sum = 7168; kv
  head map = r//2 exact with rank pair (0,1) sharing head 0 and (14,15) sharing 7;
  uniform-16 divisibility verified exact for encoder q 8192 (4 heads), o 8192 cols,
  ffn 25600, vocab 151936 (9496/rank), DiT ffn 28672/14336, video VAE 2048 (2 heads
  x 64) and ffn 16384/8192. RESUME POINT (next session, in order): (1) stage the
  rebased tree to sparke (/mnt/model-warm/staging/minimax-lane/<sha>.tgz via scp +
  git archive), (2) run the two-pass proof + pack chain: pass A per-rank dry-run
  census (tools/minimax_h3_stagepack.py --tp-degree 16 --sections
  encoder,dit,video_vae,audio_vae --rank R --dry-run) summed over 16 ranks must
  equal the strict census (705+638+585+936 placements with replication), pass B
  the real packs under sudo -n sparkcap, one queue job per small rank batch with
  --ttl-min 5 and --resources exclusive, falling back to the nohup sparkcap chain
  pattern if a rank exceeds the window at contended IO, (3) boundary checks on the
  WRITTEN packs: rank00/rank15 directory entries (q 512/384 rows, k/v 128 rows at
  kv head r//2, ffn 1792/896) plus a 4-head vs 3-head rank pair (07 vs 08) diffed
  per tensor kind, (4) C9 PP1 deployment tables via tools/minimax_h3_gen_deployment.py
  extended for TP=16/PP=1 (port base 17408 parse-compat; matrix cells BASE+a*16+b
  put the +768 route-kind offset at 18431 = exactly BLOCK_LIMIT, zero margin -
  note for the ledger), (5) ABI-seam instrumentation plan fill-in, (6) manifest+sums
  LAST again, PROGRESS receipts, push (PATH=/usr/bin). NOTHING from the r5 pack
  set exists yet: 0 of 16 ranks generated, no pack bytes written this round; the
  queue was NOT touched this round (no jobs dispatched, none in flight).

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
- 2026-09-10 (round 5): THREE driver-side numeric defects fixed, all gates green.
  (a) V3: mlp gate-residual chained the block input instead of post-attention h
  (error == gate_msa*attn exactly; bisected with per-stage anchor dumps ref_s01..s18:
  s01..s17 all <=6.8e-3, s18 == block0 error). Also V3 block1 needed block-1 adaln
  mods (ref_b1_mod_* fixtures added). (b) V4 audio: 4 stacked causes - inverted AMP
  structure (r4 carryover), single-batch dec_in/conv_pre, FLIPPED Conv1d kernel
  (source = pos+pad-t*d instead of pos-pad+t*d; k1 ops masked it), and pass-1
  activation reading the dilation input instead of the conv1 output. (c) V4 video:
  rope applied IN-PLACE (second half rotated the rotated first half) and swiglu
  silu-mid written at fused stride 2*ffn but read packed at ffn by the down GEMM.
  Fixture correction (anchor-side, proven): gen raw_s_ff dropped the video VAE ff
  biases (non-zero trained tensors; V2 bit-exact xcheck via h3_reference proves the
  pinned module applies them) - decoded fixture regenerated ff-bias-inclusive.
  S4 A-0041: SPARK_FAIL now carries SPARK_STATUS_* (module.c, pack_synthesize.c,
  stagepack shape protocol; SPARK_FAIL(SCHEMA_ERROR) at the 5 real validation sites,
  probe misses return SPARK_STATUS_NOT_FOUND) - format test green, per-site ERREPORT
  firing. C9 pack set generation started. Receipts: minimax-r5-v3bisect4,
  minimax-r5-gates-green4..8, minimax-r5-v4fix6, minimax-r5-final-gates2,
  minimax-r5-c9-packset.
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
| 5 | V3 real-weights block gate | **GREEN (r5)** — block0 rel_l2=1.786e-3 cosine=0.99999841 max_abs=8.0; block1 rel_l2=2.080e-3 cosine=0.99999784 max_abs=16.0 (gate rel<=1e-2); deterministic rerun bit-identical; TP4-partials vs TP1 bit-identical. ROOT CAUSE (bisected per-stage vs the anchor slab path, receipt minimax-r5-v3bisect4): the mlp gate-residual chained the BLOCK INPUT instead of the post-attention h — error = exactly gate_msa*attn (norm ratio 0.057 ~= rel 5.49e-2, max 1024.9 = s12 max). Fix: residual for gate_mlp add is scratch->normed (h_mid) in v3+v5 gates. Anchor per-stage fixtures ref_s01..s18 + ref_b1_mod_* (block-1 adaln mods, which the gate previously took from block 0) committed; dbg_v3_stages.py dumps them |
| 6 | V4 VAE decode gates | **GREEN (r5)** — video_vae rel=1.293e-6 max_abs=0.000009 (gate: rel<=5e-4, abs<=2/255); audio_vae rel=2.129e-6 max_abs=0.000002 (before_clamp max 0.401314 vs anchor 0.401315). Audio root causes (bisected via refa_* per-op dumps): (1) r4-era AMP structure inverted -> r5 rewrote to per-dilation residual chain + one 3-block average per stage; (2) dec_in_proj/conv_pre ran single-batch -> second mono channel was stale zeros; (3) gate Conv1d ran the kernel FLIPPED vs cross-correlation (k1 ops masked it); (4) pass-1 activation must chain from conv1 output. Video root cause: rope applied IN-PLACE (second half rotated the already-rotated first half) + swiglu wrote the silu-mid at fused stride 2*ffn while down GEMM read it packed. Anchor-side fixture correction (documented, not driver-tuning): gen_v3_v4_real.raw_s_ff dropped the video VAE ff.net.0.proj/ff.net.2 biases (non-zero trained tensors; V2 bit-exact xcheck via h3_reference proves the pinned module applies them) -> decoded fixture regenerated ff-bias-inclusive; video fix6 rel went 3.735e-2 -> 1.293e-6 |
| 7 | module build + offline-gates | **PARTIAL** — root Makefile wiring landed (contract/archive/publish hooks mirroring the glm52 flow); module archive builds through the repo flow on sparke; remaining: `make offline-gates` exit 0 on sparke cpu-class (deferred to the C10 landing because the package-manifest gate requires the final manifest regen LAST) |
| 8 | V5 determinism | **DONE at mini-DiT scale** — spark_minimax_h3_v5_gate.cu on real weights: 2 scheduler steps x real blocks 0,1 from a fixed 64-bit LCG seed; same-seed rerun bit-identical at TP1 and at TP4-segmented GEMM, and TP4-vs-TP1 bit-identical per step (receipt minimax-r4-v5gate4, sparke GB10). Full-pipeline 2x same-seed latents across 16 ranks still needs the cell |
| 9 | cell E2E | **TP16 PACK EMIT IN FLIGHT (r7)** — first cell runs TP16 (PP1, no pipeline bubbles). Head arithmetic resolved: encoder 64q/16=4.0 and video VAE 32q/16=2.0 divide cleanly; DiT 56q/16=3.5 does NOT - scheme chosen: MIXED HEAD COUNTS, 8 ranks x 4 heads + 8 ranks x 3 heads (=56), standard TP attention per rank (full 128 head_dim per head, exact all-reduce after to_out; no head-dim split, which would break softmax without gather). TP4xPP4 pack generation was started then CANCELLED - receipts showed 45GB/rank (PP0) x 4 + 11.5GB (PP1) x N ~ 300GB total, inconsistent with the 135GB figure AND obsolete under TP16-first; partial packs deleted. TP16 pack support landed r6; r7 pass-A placement proof 16/16 GREEN and the real emit fixed two driver bugs (tp16 tag dropped by match_name; blob reader had no column slicing so tp_rank>0 got rank-0 columns) before writing ranks 00-03 GREEN (33.57-33.83 GB, layout byte-exact vs computed); sizes ~31.5 GiB/rank FLAG (r5 8.4 GB figure refuted by the required- repl DIT_ADALN 24.2 GiB); ranks 04-15 halted deliberately after sparke crashed 4x (each post-crash restart D-stalled on the warm pool then the host died - flagged for sysadmin, chain left dead, resume = rerun emit_tp16_chain.sh); deployment tables TP16xPP1 + model_resident root regenerated; route matrix pinned DEAD PARSE-COMPAT (zero dialers) with 18431 = BLOCK_LIMIT zero-margin noted |
| 10 | fail-closed tests + report + manifest | pending |

## ABI seam instrumentation (ruling 3) — plan ready to fire at the cell (r7)

- instrumented shape is FROZEN in
  `modules/minimax_h3_resident_media_stage/include/sparkpipe/spark_minimax_h3_serving_adapter.h`:
  `SparkMinimaxH3AbiSeamSample` = {submission_to_dispatch_ns,
  dispatch_to_completion_ns, step_compute_ns, receipt_write_ns,
  spool_bytes_written, admission_payload_bytes}; the terminal receipt
  (`SparkMinimaxH3MediaReceipt`, static-asserted to ride
  SPARK_MODEL_SERVING_ADAPTER_MAX_EXTENSION_BYTES) carries step_compute_ns.
- fire points, zero engine edits per the ruling: the batch engine's existing
  extension slots stamp submission (admission, extension payload present) and
  completion; the module host fills one sample per denoise step; per-step node
  log line `h3 step k/N submit_us/compute_us/roundtrip_us`; seam overhead =
  (submission_to_dispatch + dispatch_to_completion − step_compute) /
  step_compute, reported as mean + max over the job.
- cell procedure: (1) 2-step kind=run smoke job → receipt seam sample +
  admission payload handling time; (2) full 5s-clip job (37 latent frames,
  N−1 evals) → total dispatch→terminal wall vs the 15-min queue window; if a
  single job cannot own the window, FLAG the measured lifetime here and take
  the detached-chain/cursor or idle-window reservation path — never a TTL
  hack.
- spool I/O cadence: rank-0 adapter writes job JSON at admission + per-step
  (state, step k/N) + terminal receipt; media file written once, atomic
  rename; receipt_write_ns measures the per-step spool cost.
- status: measurement points pending the module host implementation (module.c
  is still the metadata stub; the step-pump wiring is the remaining cell-round
  work) — measured samples land here at the cell round.

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
