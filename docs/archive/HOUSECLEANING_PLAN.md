# Housecleaning plan — "the kimi auditor finds nothing in common code"

Operator directive 2026-08-29: all stagepacks to all sparks; core
codebase squeaky clean — no redundancy, everything DRY, automated
tools so repetitive work stops burning tokens and time. CI rejected
at this stage (coordinator judgment drives; CI adds latency and false
security here). The DRY consolidation (accelerator #2) and the
memory-M1 hardware-independence start (accelerator #3) ride this plan.

## Workstreams (owners; status)

### W1 — Stagepacks everywhere (staging lane; RUNNING)
- 16/16 placement per fleet table; staging_manifest.py in gate (DONE,
  merged); qwen38max 16th pack + dsv4_flash@sparkf generation drift
  close-out; warm cleanup (186G dsv4pro stash, 366G k3 copies).

### W2 — Shared-code consolidation (NEW agent: DRY-template lane)
The repo's own DRY_CONSOLIDATION_PLAN.md inventory, executed:
1. Module-lifecycle library: one SparkFirmwareModule lifecycle behind
   the ABI (5 families' init/bind/destroy converge). Gate: qwen38_27b
   + dsv4 rebuilt on it pass all existing cells unchanged.
2. Serving-adapter TEMPLATE: descriptor constants + shared capability
   chain + ONE TP-collective config parser (kills ~3,500 pasted lines
   + ~500 config lines). The admission-default-reject bug class died
   twice (glm52 AND glm5_next) because of this paste — the template
   makes it die once. Pasted lifecycle in new code: refused (rule).
3. Stagepack format/verifier library: one pack format impl (v2 header
   + sharded axis), family packers call it. Kills per-family drift
   (the magic byte-order + payload_bytes bug classes).
4. Memory-M1 rides the template: typed buffer handles {ptr, space,
   bytes} + space-aware copy as the template's allocation path — every
   family inherits the inference-OS memory model for free (the HAL's
   first real code; the o_norm-overread and 110GiB-OOM classes get
   their compile-time teeth).
Merge protocol: per the plan (one family at a time, cell-unchanged
gates, ratchet + cyclomatic + value metric per merge).

### W3 — Hygiene: the auditor's open list (hygiene lane; RUNNING)
Red C gates, red python gates, pipeline flake, Makefile dep,
qwen38max harness wiring, memlink %n + dflash2 bounds diffs (to
coordinator), LICENSE closed-by-decision.

### W4 — Redundancy inventory + automated prevention (NEW agent)
1. Duplicate-code detector across modules/ + runtime/ (token-clone
   tool or simhash over functions) — every hit >N lines either
   consolidates into W2's libraries or justifies existence in-tree.
2. Dead code: unreachable functions, bitrotted tests (the K3
   decay|gate fusion class), unused fields — delete with receipts.
3. Cyclomatic hotspots: max 157 -> named plan; no new function > 25
   without in-commit justification (the slop gate, now mechanical).
4. The TOKEN-SAVER toolkit (operator ask): generators so repetitive
   artifacts stop being hand-written — contract->geometry-header
   generator first (the recipe compiler v0), then adapter-descriptor
   emission. Each generator lands with its generated files diffed
   byte-identical to hand-written originals before cutover.

### W5 — liteLLM architectural integration (COORDINATOR, personal)
The proxy exists (passthrough proven). The architectural cut is in
node/model_api.c (coordinator write set): /v1/models endpoint (liteLLM
model-list + health need it), proper HTTP-status mapping for our
status codes (upstream errors must surface as HTTP, not 200+body),
health-probe compatibility, and the island-catalog registration shape
(GOALS.md abstraction). Findings feed the island design.

## Sequencing
- RUNNING: staging (W1), qwen-flash PACK BUILD ONLY (wave HELD per
  operator pause), K3 pack build (sparke), glm53-diag (one bug from
  first tokens; zero conflict with repo-side cleaning — completes the
  scoreboard while we clean), hygiene (W3).
- NEW: W2 agent, W4 agent (the two big parallel cleaners).
- Coordinator: W5 + merges (per the merge gates) + the dflash2/memlink
  diffs arriving from W3.
