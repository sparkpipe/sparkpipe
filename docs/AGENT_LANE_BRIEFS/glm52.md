# Lane brief: GLM 5.2 serving bring-up (vehicle for GLM 5.3 full)

Worktree: /tmp/lane-glm52 (git worktree, branch lane/glm52, synced to main)

CORRECTIONS 2026-08-27 (coordinator, binding):
- spark1 was REMOVED from this lane. Nodes: spark9, sparka, sparkb, sparkc,
  sparkd, sparke, sparkf (7 nodes). TP8 needs 8 ranks: EITHER borrow spark8's
  idle GPU for one rank ONLY IF `nvidia-smi --query-compute-apps=pid` shows no
  coordinator job there (record the check + coordinate via the lane report),
  OR run TP4/PP across the 7 nodes. Do NOT touch spark1, spark2 (prod),
  spark3 (sysadmin), spark4-7 (qwen-flash), spark8-heavy-jobs.
- GLM 5.3 (the FULL model) shares 5.2's process/architecture (same module,
  different weights). GLM 5.3 FLASH is a DIFFERENT architecture
  (Glm5NextForConditionalGeneration: hybrid linear/deepseek-sparse attention,
  MLA kv_lora 512, MoE 288+1 top-8, 45 layers, hidden 4096 vs 5.2's
  GlmMoeDsaForCausalLM hidden 6144, 78 layers) — NEVER repack flash against
  the glm52 module; geometry mismatch is expected, not a finding. Flash is a
  separate future lane (new kernel work: sparse + hybrid attention), out of
  scope here.
- M5 corrected: "GLM 5.3 (full) repack readiness" — verify the glm52 module
  parameterization covers the 5.3-full geometry when its source lands;
  document what a 5.2 -> 5.3-full repack needs.

Cap concurrent heavy jobs per node at TWO (a prior lane rebooted spark5
with three pack builds).

## Mission
Get the GLM 5.2 FP8 model serving end-to-end on your 8 nodes, then prove
the lane is a re-pack vehicle: GLM 5.3 Flash (same architecture per the
vendor, different weights) at /mnt/model-warm/glm-5.3-flash should need
only new packs + config once 5.2 is healthy.

## Existing assets (in-tree on main)
- modules/glm52_resident_decode_stage/ (module + adapter, ~5.9k LOC) with
  TP8 history: B1 6.91 tok/s, B8 43.5, B16 75.55 aggregate measured
  (PERFORMANCE_STATUS.md). Known state: module builds, served before.
- modules/glm52_dspark_draft_backend/ (draft path, separate).
- model-families/glm52/ (geometry).
- Source: cold RAID6 GLM-5.2-FP8 704G complete at
  /mnt/cold-raid6/models/hf/zai-org/GLM-5.2-FP8 - cold is mounted on
  spark0 only; if not reachable, stage via spark0 over the fabric or pack
  from spark0. The NVFP4 5.2 copy is INCOMPLETE - do not use.
- GLM 5.3 Flash source (warm, 306G, Glm5NextForConditionalGeneration):
  hidden 4096, 45 layers, MLA kv_lora 512, MoE 288+1 top-8, MTP-1,
  hybrid linear/deepseek-sparse attention, FP8 e4m3 dynamic. If 5.2's
  geometry proves incompatible, report the delta - do not force it.

## Known landmines (from the audit trail)
- The module's page-store copy callback was just fixed on main (it
  previously raced async copies) - build from main.
- GLM 5.2 logical==physical KV capacity (no NVMe overflow) - accepted
  limitation for this lane; note it, do not fix.
- The adapter/Makefile topology flags pattern (see shared README build
  chain) - the glm52 module has its own Makefile quirks; the
  glm52_dspark host gate was recently restored.
- daemon single-client rule, DFlash2-style env discipline (GLM's spec
  env differs - check the module source for its env names), canonical
  launcher pattern.

## Milestone ladder
M1 Source + contract: hash-verify the GLM-5.2-FP8 archive (or at
   minimum config.json + index + sampled shards), freeze
   model_contracts/glm52_authoritative.json if not present.
M2 Module build + validation: publish the glm52 module on your node
   (validator + synthesized or real pack per the shared build chain).
   Exit: validation PASS.
M3 Packs: build TP8 rank packs from the 5.2 FP8 source (packer in
   tools/ - glm52 packers exist; adapt as needed).
M4 TP8 serving: 8-rank deployment on your nodes, B1 correctness, then
   B8/B16 batches. Record stream hashes + throughput.
M5 GLM 5.3 (full) repack readiness: verify the glm52 module parameterization covers the 5.3-full geometry when its source lands; document the 5.2->5.3-full repack delta. glm-5.3-flash is OUT OF SCOPE (different architecture, future lane).
   against the same module. If geometry mismatches, stop and report
   the exact deltas (that is a finding, not a failure).

## Report
docs/AGENT_LANE_BRIEFS/reports/glm52-<date>.md after every milestone,
INTEGRATION REQUEST section for anything outside your write set.
