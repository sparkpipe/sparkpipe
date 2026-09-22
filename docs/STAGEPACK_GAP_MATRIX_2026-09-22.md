# Stagepack gap matrix + running inventory — 2026-09-22

Lane: `lane/stagepack-gaps` (worktree `/Users/cem/sparkpipe-worktrees/stagepack-gaps`).
Method: live census of `/home/<host>/sparkdata/` on all 16 sparks (spark0..sparkf)
2026-09-22/23, diffed against per-family stagepack tooling capability
(`tools/*_stagepack*.py`) and warm-source availability (`/mnt/model-warm`,
spark8 client). Companion docs: `docs/STAGEPACK_AUDIT_2026-08-31.md` (fleet
map doctrine), `docs/STAGEPACK_NAMING.md`.

## Fleet map doctrine (from the 2026-08-31 audit, still binding)

- TP16 / TP4PP4 / PP16 / TP8PP2: 16 rank packs, rank h on spark h.
- TP8: 8 distinct ranks x 2 replicas = 16 nodes.
- TP4: 4 distinct ranks x 4 replicas = 16 nodes.
- Placement without a sha256 receipt is not placement; no symlinks; topology
  variants are first-class (never removal candidates).

## Complete sets (no action) — verified 16/16 with pack bytes

glm53flash bf16.tp16, fp8.tp16, fp8.tp8, fp8.tp4pp4, nvfp4.tp16, nvfp4.tp8,
nvfp4.tp4pp4; glm53full bf16/fp8/nvfp4 x tp16/tp4pp4 (all six); k3 mxfp4.tp16,
mxfp4.tp4pp4; hy4 fp8.tp16, ud-iq1m.tp16; ling bf16.tp16, bf16.tp16.t1ling,
fp8.tp16; laguna-s-2.1 bf16.tp8pp2 (stage0 ranks 0-7 on spark0-7, stage1
ranks 8-15 on spark8-f); muse tp16.bf16 (rank h on spark h; sparkc also holds
a full 16-rank staging duplicate, ~54G, reclaim candidate); qwen27b.tp4
(legacy pre-rename fp8, 16/16); qwen38-27b.nvfp4a16.tp4 (16/16, the registry
serving set); qwen38_27b.tp4pp4 (16/16); qwenmax pp16, pp16-stripped,
nvfp4.tp16, tp4pp4; qwen3flash bf16.tp4, bf16.tp8, bf16.tp4pp4, fp8.tp8,
fp8.tp4pp4, nvfp4.tp8, nvfp4.tp4pp4, nvfp4.tp8pp2; gemma4_31b bf16.tp16,
bf16.tp4pp4, nvfp4.tp16; gemma4_26b tp4pp4.t1 (sparkd holds 16 staging
duplicates, 48G, reclaim candidate); dsv41flash mxfp4.tp8pp2.

## GAP LIST

### P0 — broken/partial EXISTING sets (close first)

| # | Set | Missing | Size/node | Source (warm) | Notes |
|---|-----|---------|-----------|---------------|-------|
| 1 | dsv41flash.mxfp4.tp8 | spark6 rank pack (15/16) | 37G | none needed | r2 regen already staged ALL 8 ranks on spark6 `packs-r2/` (292G, rank6 has receipt+sha, stage.log 8/8 OK); fan-out was interrupted. Place rank6 -> `packs/`, then staging-clean decision. Also fixes the spark6 disk alert. |
| 2 | lingfin.bf16.tp16 | rank7 on spark7 (15/16) | 15.7G | ling-3.0-flash-fin (245G) ✓ | spark7 ceph client degraded -> build via spark8 client, rsync to spark7. |
| 3 | qwen27b.fp8.tp4pp4 | spark7 (15/16) | 1.7G | qwen3.8-27b-fp8 (29G) ✓ | twin qwen38_27b.tp4pp4 is 16/16; verify sha-equality first (may be a copy, not a regen). |
| 4 | dsv41flash.mxfp4.tp4 | rank2 on spark6 (3/4 ranks: 0@spark4, 1@spark5, 3@spark7) | 71G | deepseek-v4.1-flash (476G) ✓ | lane was actively building (rank0 placed Sep 23 00:15); confirm not mid-flight before building rank2. |

### P1 — interrupted generations (journals/staging present)

| # | Set | Missing | Size/rank | Source | Notes |
|---|-----|---------|-----------|--------|-------|
| 5 | mimo26pro.mxfp4.tp8 | ranks 4-7 (+ all second replicas; 4/16 node-slots placed: rank0-3 on spark0-3) | 67G | mimo-v2.6-pro-rl (535G) ✓ | emit staging: spark7 271G (ranks 0-6 partial), spark4 39G, spark5 38G; `journal.jsonl` resumable; packs/ dirs empty on 4-7. |
| 6 | mimo26flash.mxfp4.tp4 | ALL ranks (0/16 slots) | ~35-40G est | mimo-v2.6-flash-rl (166G) ✓ | emit staging only: spark7 emit/rank3 (12G partial). |
| 7 | qwen27b.fp8.tp4 | 8/16 nodes (present: 0,1,2,3,9,a,b,e) | 9.9G | qwen3.8-27b-fp8 (29G) ✓ | legacy qwen27b.tp4 (same fp8 set, pre-rename) is 16/16 — likely copy/rename (verify sha per rank), not regen. |

### P2 — missing ARMS (tool-supported, warm source ready, zero packs)

| # | Arm | Tool | ~Size | Source |
|---|-----|------|-------|--------|
| 8 | laguna.fp8.tp8pp2 | laguna_stagepack.py --expert-codec fp8 | ~8G/rank x16 ≈ 123G | laguna-s-2.1-fp8 (123G) ✓ |
| 9 | laguna.nvfp4.tp8pp2 | laguna_stagepack.py --expert-codec nvfp4 | ~6G/rank x16 ≈ 93G | laguna-s-2.1-nvfp4 (93G) ✓ |
| 10 | glm53flash.bf16.tp8 | glm5_next_resident_stagepack.py (source-driven bf16) | ~74G/rank x16 ≈ 1.2T | glm-5.3-flash-bf16-official (599G) ✓ |
| 11 | glm53flash.bf16.tp4pp4 | same | ~37G/rank x16 ≈ 600G | same ✓ |
| 12 | qwen3flash.bf16.tp16 | qwen4_flash_stagepack.py --expert-format bf16 | ~21G/rank x16 | qwen3.8-flash-next (336G) ✓ |
| 13 | qwen3flash.fp8.tp4 / fp8.tp16 | --expert-format fp8-official | ~21G / ~11G per rank | qwen3.8-flash-next-fp8 (173G) ✓ |
| 14 | qwen3flash.nvfp4.tp4 / nvfp4.tp16 | --expert-format nvfp4-official | ~15G / ~8G per rank | qwen3.8-flash-next-nvfp4-nvidia (126G) ✓ |
| 15 | gemma4_31b.nvfp4.tp4pp4 | gemma4_stagepack.py (nvfp4 source + stage slices) | ~8-10G/rank x16 | gemma-4-31b-it-nvfp4 ✓ (queue job `gemma4-fleet-build-13` queued fleet-wide — coordinate before building) |

### Blocked (report-only; NOT generatable under the no-new-packers law)

- hy4.fp8.tp4pp4 — hy4 packer hard-codes TP=16 (`hy4_fp8_stagepack.py`);
  TP4PP4 needs the lane's packer extension. Source hy4-preview-fp8-official ✓.
- lingfin.fp8.tp16 — no fp8 fin checkpoint on warm; the ling packer never
  quantizes. No source.
- ling.bf16/.fp8.tp8 — packer takes --tp-degree but the lane's resident arm
  is TP16; TP8 unruled.
- gemma4_26b non-t1 arms (bf16.tp4pp4 from the 49G bf16 source) — the T1
  encoding lane owns the 26b format; ruling needed.
- minimax.text.bf16.tp4 — 8/16 nodes (0-3, 8-b); ACTIVE lane work: queue job
  `minimax-l10-serve-run7` running on spark8-b right now; spark9/a/b packs
  freshly sha'd Sep 23 03:31. Coordinate with the minimax lane; do not
  collide. (spark8 holds 4 ranks / 82G — 3 are staging extras.)

## Fleet health / constraints (2026-09-22)

- **spark6: 151G free (<200G threshold)** — 96% full. Cause: dsv41 tp8
  `packs-r2` 292G + `engram-r2` 25G staging. Closing gap #1 (place rank6,
  then clean r2 after distribution) restores ~292G. No other node below
  200G (next-lowest free: spark0 595G).
- Staging/internediate debris (cleanup candidates, bytes logged, per pack
  rules "delete your scratch when packs are deployed"): spark7 mimo26pro
  emit 271G + mimo26flash emit 12G, spark4 39G, spark5 38G, sparkc muse
  staging ~54G, sparkd gemma4_26b staging 48G, spark8 minimax extras ~65G.
  Owners: the respective interrupted generations (mine to finish or
  coordinate).
- Ceph/warm: 38T free of 59T. spark8 client HEALTHY (verified read).
  spark1/spark3 clients dead, spark7 degraded (per operator brief);
  laguna tool additionally warns: never run from sparke (stale negative
  cache). One-time generation reads per the ceph law are sanctioned; single
  reader, spark8 preferred.
- Queue (controller mac-studio, wk-sparkpipe-queue-main-20260922): active
  now — `minimax-l10-serve-run7` (gpu-shared, spark8-b), `gate-1169c-run`
  (spark0); queued fleet-wide gpu-shared builds: k3-m3-cold12,
  ling9-m3-build-r14, qmax-m3-build-r23p6, qmax-m3-attach-r13,
  gemma4-fleet-build-13. My jobs: kind=run, --resources cpu, explicit
  --memory-mib (packers are slab-streaming, ≤1G; MemoryMax wrap ~1.5G),
  ttl-safe split rounds (≤15-min deadline; per-rank/per-stage/per-window
  chunks with resumable journals).

## Generation inventory (running log)

| When (UTC) | Gap | Action | Result |
|---|---|---|---|
| 2026-09-22 | — | census (this doc) | see above |
