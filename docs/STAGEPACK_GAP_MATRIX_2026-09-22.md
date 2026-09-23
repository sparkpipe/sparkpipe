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
| 1 | dsv41flash.mxfp4.tp8 | ~~spark6 rank pack~~ CLOSED 2026-09-23 | 37G | none needed | rank6 placed in spark6 `packs/` (sha 58a93a87…, receipt OK); `packs-r2` 292G cleaned; spark6 back to 395G free (engram-r2 25G remains, dsv41-lane scratch). 16/16. |
| 2 | lingfin.bf16.tp16 | rank7 on spark7 (15/16) | 15.7G | ling-3.0-flash-fin (245G) ✓ | build on spark8 TTL-killed at 10.16/15.7G (single-stream, no resume). `ling_stagepack.py --resume` landed (journal + partial, ck128 per region); build resubmits as ttl-split resume rounds on spark8, then rsync to spark7. |
| 3 | qwen27b.fp8.tp4pp4 | ~~spark7 (15/16)~~ CLOSED 2026-09-23 | 1.7G | qwen3.8-27b-fp8 (29G) ✓ | rank07 built on spark8, placed on spark7 `packs/` (sha dc02eff9…, verify OK); identical bytes to the qwen38_27b.tp4pp4 twin — no regen needed. 16/16. |
| 4 | dsv41flash.mxfp4.tp4 | rank2 on spark6 (3/4 ranks placed: 0@spark4, 1@spark5, 3@spark7) | 71G | deepseek-v4.1-flash (476G) ✓ | lane idle (no active build, no emit staging left). Manager green-lit rank2: build on spark8 via headers/plan/copy windows (provenance extracted from placed rank1 header: revision dba1be0a…, contract 44fcba0b…, config 8be45ce0…, recipe d58a0258…), verify + experts manifest + receipt, rsync to spark6. |

### P1 — interrupted generations (journals/staging present)

| # | Set | Missing | Size/rank | Source | Notes |
|---|-----|---------|-----------|--------|-------|
| 5 | mimo26pro.mxfp4.tp8 | ranks 4-7 (+ all second replicas; 4/16 node-slots placed: rank0-3 on spark0-3) | 67G | mimo-v2.6-pro-rl (535G) ✓ | 2026-09-22 round failed: emit died Errno 24 (systemd soft fd limit; the packer holds one fd per output plane), replica places used the wrong source layout. Fixed (fd raise + flat `packs/` source) — chain resubmitted 2026-09-23: emits rank4-7 on spark8 (healthy ceph reader, single-reader serial), assemble, place r4-7 -> spark4-7, all 8 second replicas -> spark8-f. |
| 6 | mimo26flash.mxfp4.tp4 | ALL ranks (0/16 slots) | ~35-40G est | mimo-v2.6-flash-rl (166G) ✓ | 2026-09-22 round never started (chained behind the pro emit failure). Resubmitted behind the pro chain: emit r0-3 on spark8, place primaries on spark0-3, 12 replicas on spark4-f. spark7 flash emit partial (12G, rank3) is dead scratch once the spark8 chain lands. |
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

## Fleet health / constraints (2026-09-23 update)

- **spark6: 395G free** — gap #1 closed (rank6 placed, packs-r2 292G cleaned).
  Only engram-r2 25G staging remains (dsv41 lane scratch). No node below
  200G free; spark7 1.1T free, spark8 1.1T free at chain start.
- Staging/intermediate debris remaining: spark7 mimo26pro emit 268G
  (rank0-3 full, superseded — their packs are placed) + 8G rank4-6 partials
  + mimo26flash 12G; sparkc muse staging ~54G, sparkd gemma4_26b 48G,
  spark8 minimax extras ~65G. The spark7 mimo staging is deleted once the
  spark8 re-emit chain lands (scratch law).
- Ceph/warm: reads for all gap generation go through spark8's client
  (healthy), single-reader, serialized as queue `--after` chains. spark1/3
  clients dead, spark7 degraded — spark7 only receives rsyncs.
- Queue (controller mac-studio, wk-sparkpipe-queue-main-20260922): 2026-09-23
  lane jobs are kind=run, cpu, --memory-mib 1024 (MemoryMax wrap), ttl 14m,
  chained emit -> assemble -> place -> replica per rank (ids g5p-*, g6f-*).
  A 2026-09-22 round failed on the fd limit + wrong replica source layout
  (root causes fixed in tools/stagepack_gap/*, PR #1179).

## Generation inventory (running log)

| When (UTC) | Gap | Action | Result |
|---|---|---|---|
| 2026-09-22 | — | census (this doc) | see above |
| 2026-09-22 | #1 | place dsv41 tp8 rank6 spark6 + clean packs-r2 | rank6 sha 58a93a87… OK; 292G freed; 16/16 |
| 2026-09-22 | #3 | build qwen27b fp8 tp4pp4 rank07 (spark8) + place spark7 | sha dc02eff9… verify OK on spark7; 16/16 |
| 2026-09-22 | #2 | lingfin rank7 build round 1 (spark8) | TTL kill at 10.16/15.7G; superseded by --resume rounds |
| 2026-09-22 | #5/#6 | first pro/flash chain round | failed: Errno 24 fd limit (emit), flat-layout source bug (replicas), cleaned checkout (cd fail); all dependency-cascaded |
| 2026-09-23 | #5/#6 | runner fixes (fd, layout, sha check) + resubmit g5p-*/g6f-* chains on spark8 reader | in flight |
