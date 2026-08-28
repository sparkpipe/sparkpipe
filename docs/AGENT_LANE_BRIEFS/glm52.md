# Lane brief: GLM 5.2 serving bring-up (vehicle for GLM 5.3 full)

Worktree: /tmp/lane-glm52 (git worktree, branch lane/glm52, synced to main)

CORRECTIONS 2026-08-27 (coordinator, binding; second update supersedes the first):
- FINAL node set: spark8, spark9, sparka, sparkb, sparkc, sparkd, sparke,
  sparkf (8 nodes, full TP8, no borrowing). spark8 was handed over by the
  coordinator: /home/spark8/sparkpipe-rel (instrumented 27B repo, do not rely
  on its build state) and /home/spark8/sparkdata/qwen38.fp8.tp1 (27B
  deployment, ignore/leave) live there; a leftover residentd may be killed
  freely if it holds GPU memory. Do NOT touch spark1 (coordinator 27B),
  spark2 (prod), spark3 (sysadmin), spark4-7 (qwen-flash).
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
Get the GLM 5.2 FP8 model serving end-to-end on your nodes. The re-pack
target is GLM 5.3 (the FULL model - same process/architecture as 5.2,
different weights). GLM 5.3 FLASH is a DIFFERENT model logic
(Glm5NextForConditionalGeneration: hybrid linear/deepseek-sparse
attention, MLA kv_lora 512, MoE 288+1 top-8, hidden 4096) and is NOT
a repack of this module - it is a separate future lane with new kernel
work. Do not attempt it here.

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
