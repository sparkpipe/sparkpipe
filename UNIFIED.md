# Unified branch

`unified` is the reference tree for the six model sessions. It takes
`origin/main` as input and publishes a DRY, gate-green consolidation of it.

## Contract

- **Input:** `origin/main`. New main work is integrated into `unified` as
  it lands; the integration commits never diverge on purpose from main's
  functionality, only on structure.
- **Output:** one DRY tree. Model facts live in tables (family headers,
  authoritative contracts, model-geometry dicts); shared machinery lives
  once. No parallel copies of tooling, adapters, or kernels.
- **Gates:** `test_dry_law.py`, `test_code_size.py`,
  `test_must_work_targets.py`, the contract `--check` generators, and the
  CUDA 13 sm_121a compile gate all pass on `unified`. CI evidence: the
  draft tracking PR (#670, never merged) runs the full compile gate on
  every `unified` push; it is green at `a83ce4e` (4m38s). The DRY gates
  are additionally wired into the CI workflow on the `unified-ci-wiring`
  companion commit; that needs a `workflow`-scoped PAT to push, so it
  lands separately.
- **Convergence:** as the six drivers stabilise and their consolidations
  land on main, `unified` and `main` become the same tree. Until then,
  sessions may rebase onto `unified` at any point; `unified` is the
  eventual reference directory.

## What unified owns

- Duplication removal across model tooling (packers, sharders, generators,
  adapters) — parameterize first, delete the copy.
- The strict general-vs-model-specific law: shared code never names a model;
  model facts stay in `inference/llms/<model>/`, `model-families/<model>/`,
  `modules/<model>_*`, and per-model tools.
- Coexistence data (fleet registry, port tables) and the deprecations the
  directives call for, folded in once main accepts them.
- The code-size ceiling: consolidation shrinks it; new main machinery moves
  it only with a named justification.

## What unified does not own

- Model kernels, contracts, and qualification receipts — those belong to
  their sessions and flow in through main unchanged.
- New features. `unified` only restructures what main already contains.

## Known gaps

- CI wiring for the DRY gates is prepared on `unified-ci-wiring` and
  blocked only on a `workflow`-scoped PAT.

## Landing so far

- Moved `deployment/fleet_registry.json` to `tools/devcycle/` (it is
  fleet-state data, not deployment code) so `test_dry_law.py` passes;
  `tools/fleet_swap.sh` and `COORDINATION.md` point at the new path.
- The CI wiring for `test_dry_law.py` and `test_code_size.py` is prepared
  on the `unified-ci-wiring` companion commit; it needs a
  `workflow`-scoped PAT to push, so it lands separately.
- Extended `test_dry_law.py` to scan `inference/kernels/` and swept the
  shared kernel headers' model-provenance comments to model-neutral
  phrasing (55 comment lines; no code change).
- Consolidated the DSV4 Pro TP16 stagepack copy into the parameterized
  `tools/dsv4_tp16_stagepack.py` (`--model flash|pro`; 426 duplicated
  lines removed) and the TP4xPP4 driver pair into
  `tools/dsv4_tp4_pp4_stagepacks.py` (`--model`), updating the devcycle
  pro scripts to the single sharder.
- Reconciled `tools/generate_dsv4_contracts.py` with the merged Pro
  reality (first-light BF16 activations, FP8-E4M3 expert codec
  selectability, the Pro alias guard in the Flash header, the first-light
  note): all generated files now reproduce byte-exact, so the
  verify-contracts CI step that was failing every session PR passes again.
  Same fix proposed to main as PR #671 (CLEAN). Refreshed the two stale
  DSV4 host tests to the current packer contract (3 packed draft layers,
  format version 4).
- Integrated the six-session wave by merging the active session PRs
  directly (K3 TP4 layer-0, DSV4 DSpark speculative loop, DSV4 Pro GA 0813
  migration, Qwen 3.8 Max phase 2, Qwen TP4 phase 2, DSpark design docs;
  rejected/record-only candidates excluded), then reconciled the generated
  DSV4 contracts to the GA 0813 checkpoint (3 packed draft layers, KV
  codec selectability, consistent -0813 identity), aligned the K3 and
  DSV4 host tests with checkpoint reality, bucket-guarded the DSpark
  speculation cluster for the gate's bucket-1024 archive build, and
  re-proved the whole tree through the tracking CI (green, 4m40s).
