# Stagepack Fleet Audit — 2026-08-31

Scope: every model dir in warm storage ↔ placed stagepacks on all 16
sparks; deprecated/misleading set removal. Trigger: operator directive
("audit all the stagepacks … a completely clean set of 16 sparks with all
the stagepacks for all the models; audit EVERY model in warm storage").
Companion law: **NO SYMLINKS FOR STAGEPACKS** (see coordinator-log 12:1x).

## 1. The serving matrix (KEEP set — the only packs a clean fleet holds)

| Set (per node dir) | Source (warm) | Rank bytes | Coverage |
|---|---|---|---|
| glm53full.bf16.tp16 | glm-5.3-bf16 (1.4T) | ~98G | 16/16 (spark5 rank5 REBUILD in flight — see §5) |
| glm53full.fp8.tp16 | glm-5.3-fp8 (704G) | ~50G | 16/16 |
| glm53full.nvfp4.tp16 | glm-5.3-nvfp4-radixark (433G) | ~30G | 16/16 |
| glm5_next.tp16 (flash engagement set) | glm-5.3-flash era | ~21.7G | 16/16, materialized real bytes + sha receipts |
| glm5_next.tp8.fp8 (flash serving set) | glm-5.3-flash (306G) | ~21.7G | 16/16, materialized real bytes + sha receipts |
| dsv4flash.tp16 | deepseek-v4-flash-0731 (156G) | ~20G | 16/16 |
| dsv4_pro.tp4pp4 | deepseek-v4-pro-0813-ga (832G) | ~88G | 10/16 — 6 ranks pending packer rank-path |
| qwen27b.tp4 | qwen3.8-27b-fp8 (29G) | ~9G | 16/16 (TP4 ruling) |
| qwenflash.tp8 | qwen3.8-flash-next (336G bf16) | ~43G | 16/16 (TP8 ruling) |
| qwenmax.pp16 | qwen3.8-max-nvfp4-radixark-bf16-spine (1.4T) | ~90G | 16/16 |
| k3.mxfp4.tp16 | kimi-k3 (1.5T) | ~97G | IN FLIGHT — per-node slice→verify→place; receipts land via 10-min automation |
| k3.mxfp4.tp4pp4 | kimi-k3 (1.5T) | ~91G | 16/16 INTACT — **KEEP: valid topology variant, the proven serving topology. Never a removal candidate.** (CORRECTION 2026-08-31: an earlier draft of this audit listed it for post-TP16 removal — WRONG. The matrix keeps topology variants; TP16 adds, it does not replace.) |

## 2. Warm-storage model dirs → coverage verdict

| Warm dir | Size | Verdict |
|---|---|---|
| glm-5.3-bf16 | 1.4T | COVERED (bf16.tp16 16/16) |
| glm-5.3-fp8 | 704G | COVERED (fp8.tp16 16/16) |
| glm-5.3-nvfp4-radixark | 433G | COVERED (nvfp4.tp16 16/16) |
| glm-5.3-flash | 306G | COVERED (glm5_next.tp8.fp8 + tp16 engagement) |
| glm-5.3-flash-bf16-official | 599G | **RULED IN (operator, 2026-08-31): build stagepacks.** The rule is every model in warm storage gets stagepacks — arms are not optional. To build: ~37G/rank × 16 |
| glm-5.3-flash-nvfp4-redhatai | 185G | **RULED IN (operator, 2026-08-31): build stagepacks.** ~12G/rank × 16 |
| glm-5.3-flash-dflash2 | 2.2G | drafter spec module — keep (small) |
| kimi-k3 | 1.5T | IN FLIGHT (k3.mxfp4.tp16) |
| kimi-k3-{dflash2-lightseek,dflash-modal,dspark-inferact,dspark-radixark,dspark-redhatai} | 4-9G | drafter/dspark spec variants — keep (small, experimental) |
| qwen3.8-27b-fp8 | 29G | COVERED (qwen27b.tp4 16/16) |
| qwen3.8-27b-nvfp4a16-bf16-spine | 29G | **RULED IN (operator, 2026-08-31): build stagepacks.** 4-bit experts + 16-bit spine is exactly the compressed-experts+full-spine policy shape. To build |
| qwen3.8-27b-{dflash2-incoai,dspark-radixark} | ~7G | drafters — keep |
| qwen3.8-flash-next | 336G | COVERED (qwenflash.tp8) |
| qwen3.8-flash-next-fp8 | 173G | **GAP — FP8 arm packs not built** (per-arm policy says build: ~11G/rank TP16-equiv) |
| qwen3.8-flash-next-nvfp4-radixark | 126G | **GAP — NVFP4 arm packs not built** (~8G/rank) |
| qwen3.8-max-nvfp4-radixark-bf16-spine | 1.4T | COVERED (qwenmax.pp16) |
| qwen3.8-max-{dflash-modal,dspark-radixark} | ~11G | drafters — keep |
| deepseek-v4-flash-0731 | 156G | COVERED (dsv4flash.tp16) |
| deepseek-v4-flash-dflash-redhatai | 3.4G | drafter — keep |
| deepseek-v4-pro-0813-ga | 832G | PARTIAL (tp4pp4 10/16; last 6 ranks need packer rank-path) |
| deepseek-v4-pro-0813-nvfp4-jarrelscy | 877G | **GAP — dsv4-pro TP16 set** (the last board rung) |
| archive-not-a-source | 1.5T | archive — not a serving source |

## 3. Deprecated removals — WAVE 1 EXECUTED 2026-08-31 (per-node logs: ~/wave1_removal.log)

Exact-name removal across all 16 nodes (~2.5-3TB reclaimed):
dsv4_compress_ingest_{control,fuse}_da7f910 (74G/node × 4);
dsv4_flash.fp8.{pp13,pp13.b16,pp13.b8} (old PP13 era);
dsv4_flash.fp8.tp16.b1{,.128,.gpudirect,.hostrdma} (bench variants —
serving set is dsv4flash.tp16); dsv4_flash.fp8.tp4.b1{,.hostrdma};
dsv4_flash.fp8.tp4_pp4.b1; dsv4_flash.v4.tp16.b1 (bench gen);
qwen38.bf16.{pp16,pp4,tp4,tp1} (BF16 serving eras incl. 78G tp1);
qwen38.fp8.tp1{,.official,.auditk8,.audrb,.q27b} (TP1 era — q27b lane
mission-closed, 27B serves TP4); qwen27b.tp16 (spark2, 38G interrupted
partials — TP16 superseded by the TP4 ruling); glm52.tp8.fp8 (95G × 3 —
GLM-5.2 is frozen/deprecated, kernel donor only); qwen38max.tp4pp4 (573G
orphan on one node — **REMOVED IN ERROR under the same misclassification;
under the topology-variant doctrine it should have been kept. Rebuildable
from the warm max source; rebuild queued pending operator priority.**);
glm53full.bf16.tp16 full-set dump on
spark5 (1.37T of foreign ranks — pruned to rank5, which I then wrongly
deleted and am REBUILDING, see §5).

## 4. HOLDS — no sets on removal lists; duplicates/corrupt only

Wave-2 cleanup targets are limited to EXACT DUPLICATES and corrupt
artifacts, never topology variants or sources:
- ~/glm53_packs{,_fixed,_fixed2,_fixed_r4} staging (~65G/node) —
  byte-identical duplicates of what is now materialized (with sha
  receipts) inside the deployment dirs; reclaim after 16/16 symlinkfix
  receipts verify.
- Zero-byte files, interrupted `.tmp` partials, empty work dirs.
- qwenmax.pp16 size outliers (85-131G) — inspect for extra stages before
  touching anything.
- History: the spark2/spark3 k3.mxfp4.tp4pp4 residentd processes were
  TERMed (27h+ silent, NCCL-erroring). Their PACKS remain intact on every
  node (verified 91-92G × 16). Redeploying working tp4pp4 daemons is a
  serving-lane task, not a storage one.

## 5. Incidents during this audit (honesty ledger)

1. spark6: qwen38.bf16.tp1 packs deleted by earlier cleanup — variant-dir
   dangling links removed; set is LOST (rebuildable from warm if ever
   needed).
2. spark5: my prune of the foreign-rank dump used a wrong glob
   (`^rank5\.` vs the real prefix `glm53full.bf16.tp16-rank5`) and removed
   spark5's OWN rank5 with the extras. Rebuild launched from
   /mnt/model-warm/glm-5.3-bf16 (glm52_resident_stagepack --tp-rank 5,
   expert-codec bf16); verifies against the set contract on completion.
   Lesson: deletion scripts must match FULL file names and print their
   kill list for approval-shaped review before rm.

## 6. Audit gates going forward

- `find ~/sparkdata -type l ! -path "*/.venv/*"` EMPTY on every node.
- Every placed pack has a sha256 receipt beside it (placement without
  receipt is not placement).
- Every warm ACTIVE model dir maps to a placed set — a missing mapping is
  a BUILD TASK, not a question. The only exception is an explicit operator
  ruling recorded verbatim in this file.
- Topology variants (TP16, TP8, TP4PP4, PP16, ...) are each first-class:
  an existing variant is never a removal/deprecation candidate. Removals
  require either corruption, exact-duplicate status, or a verbatim
  operator ruling.

## 7. Build queue from the corrected rulings (operator, 2026-08-31)

1. glm-5.3-flash-bf16-official stagepack set (~37G/rank × 16).
2. glm-5.3-flash-nvfp4-redhatai stagepack set (~12G/rank × 16).
3. qwen3.8-27b-nvfp4a16-bf16-spine stagepack set (TP4 and/or TP16 per the
   topology-variant doctrine).
4. qwen-flash FP8 + NVFP4 arm packs (already queued).
5. dsv4-pro TP16 set (rank-path extension) and remaining tp4pp4 ranks.
6. qwen38max.tp4pp4 rebuild (573G, one node) — priority pending operator.

## 8. hy4 ADDED TO THE ACTIVE SET (operator, 2026-09-01)

- Source: /mnt/model-warm/hy4-preview-fp8-official — 766G FP8 official
  (modelopt MXFP8), HYV4ForCausalLM, 78 layers, hidden 6144, 64 attn
  heads / 8 KV heads, 256-expert MoE top-8 (moe_inter 2048),
  hyper-connections (hc_head / hc_attn_layer), MTP PRESENT (39
  mtp_layers.0.* tensors, deepseek-style) → per the MTP law, hy4 packs
  MUST carry MTP.
- Geometry pre-check (the law): TP16 heads 64/16=4 ✓, ffn 144 blocks
  %16=0 ✓, moe 16 blocks %16=0 ✓, KV-heads 8 → replicated (glm/dsv4
  precedent); ~48G/rank, fits the 110GiB law. TP4: all clean too.
- Targets: TP16 set + TP4×PP4 set, both MTP-carrying. New architecture →
  packer/module vertical = the hy4 dev lane's build (agent started by the
  operator); closest in-tree relative: glm5_next (hc kinds exist in its
  kind table) and dsv4 (mtp.0.* naming class). Coordinator provides
  board slots + queue coordination; the lane owns packer/descriptor.
- Note: dsv4-pro's spec layers and hy4's MTP share the deepseek MTP
  shape — the dsv4 packer's mtp handling is a reusable reference.

## 9. THE COMPLETE-SET CHECKLIST (operator-formatted; standing automation reference)

Definition of done per set: all ranks real bytes on canonical nodes (TP4 4x,
TP8 2x maps; rank r on spark-r), uniform per-rank size, sha256 receipted,
MTP-carrying if the source ships it, verified against source.

In flight (finish first):
1. k3 TP16 — 14/16 placed; left: spark1 rank1 (slicing), spark4 rank4 (after its qwenflash job); then 16/16 sha placement audit.
2. k3 cleanup — after 16/16: remove 1.56TB warm base + all slice/deploy work dirs + spark2 27B temps (bytes logged).
3. qwen-flash TP8 (bf16) — rank4 rebuilding (29/46G) → place on spark4 + copy to sparkc; then MTP audit of placed ranks (copy_mtp_fc) → rebuild if absent.
4. glm5_next TP8 FP8 (true, MTP) — COMPLETE: 8 ranks x 2 targets = 16/16 nodes, one canonical rank each (43,479,544,832 B, 1187 tensors, flags=1, sha-receipted); wrong-topology tp16-named files removed fleet-wide (~272G).

Strays (fix-as-found):
5. k3 duplicate ranks on spark2/spark3 (185G each) — digest-identify; remove own-rank duplicates, report foreign copies.
6. Truncated 19G qwenflash partials — delete once rank4 replacements are placed.

Per-set completions:
7. 27B TP4 (fp8) — placed; confirm MTP via the running verifier on spark6; upgrade if absent.
8. 27B TP4 nvfp4a16-bf16-spine arm — build from warm source (packer vertical per the decoded fused layout).
9. 27B TP4xPP4 — build (with MTP).
10. qwen-max PP16 (nvfp4) — placed; MTP check (source has mtp.fc + layer) → upgrade if absent.
11. qwen-max TP4xPP4 — build (with MTP; supersedes the deleted-in-error qwen38max.tp4pp4 — its rebuild folds in here).
12. dsv4-flash TP16 — placed; audit mtp.0.* spec layers actually in the packs.
13. dsv4-flash TP4xPP4 — build.
14. dsv4-pro TP4xPP4 — 10/16; build the last 6 ranks (packer rank-path extension).
15. dsv4-pro TP16 — build from the nvfp4-pro source (877G; splicer rank-path).
16. glm5_next (flash) TP4xPP4 — build (with MTP).
17. glm53full bf16 TP4xPP4 — build. (TP16 done.)
18. glm53full fp8 TP4xPP4 — build. (TP16 done.)
19. glm53full nvfp4 TP4xPP4 — build. (TP16 done.)
20. qwen-flash TP4xPP4 — build.
21. qwen-flash TP8 fp8 arm — build (with MTP).
22. qwen-flash TP8 nvfp4 arm — build (with MTP).
23. glm-flash bf16-official arm TP16 — build (~37G/rank).
24. glm-flash nvfp4-redhatai arm TP16 — build (~12G/rank).
25. hy4 TP16 — packer/module vertical (hy4 dev lane owns; geometry clean; MTP 39 tensors must ride).
26. hy4 TP4xPP4 — build (MTP-carrying).
27. k3 TP4xPP4 — COMPLETE (kept, verified intact) — no work.
28. Drafter variants audit — the small dflash2/dspark dirs on warm: each has its drafter pack + descriptor, or gets one.

Close-out (only after 1-27):
29. Canonical sanity audit — every node holds exactly its map's sets; per-rank digest == master; uniform per-rank sizes; set-total vs source arithmetic; zero symlinks/temps/duplicates; stray list filed before any removal.
30. Reclaim + final board — ~/glm53_packs* staging duplicates (~65G/node); log the final matrix; commit, push; idle.
