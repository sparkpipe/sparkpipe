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
