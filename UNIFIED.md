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
  CUDA 13 sm_121a compile gate all pass on `unified`. The DRY gates are
  wired into the CI workflow on the `unified-ci-wiring` companion commit;
  it needs a `workflow`-scoped PAT to push, so it lands separately.
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

- `tests/test_dry_law.py` does not scan `inference/kernels/`, where the
  shared kernel headers carry model-provenance comments. Extending the gate
  there needs the comment sweep first; it is coordinated with the sessions
  whose branches edit those files, and lands once they are quiet.

## Landing so far

- Moved `deployment/fleet_registry.json` to `tools/devcycle/` (it is
  fleet-state data, not deployment code) so `test_dry_law.py` passes;
  `tools/fleet_swap.sh` and `COORDINATION.md` point at the new path.
- The CI wiring for `test_dry_law.py` and `test_code_size.py` is prepared
  on the `unified-ci-wiring` companion commit; it needs a
  `workflow`-scoped PAT to push, so it lands separately.
- Consolidated the DSV4 Pro TP16 stagepack copy into the parameterized
  `tools/dsv4_tp16_stagepack.py` (`--model flash|pro`; 426 duplicated
  lines removed) and the TP4xPP4 driver pair into
  `tools/dsv4_tp4_pp4_stagepacks.py` (`--model`), updating the devcycle
  pro scripts to the single sharder.
