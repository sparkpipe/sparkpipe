#!/usr/bin/env python3
"""THE COMPLEXITY CEILING — the merge gate that makes the plan bind.

The 2026-08 audits' sharpest line: "the max grew 157->158 while the plan
sat in a doc." This gate is the doc growing teeth. The committed ceiling
is the maximum cyclomatic complexity of the PRODUCTION scope (functions
under any */validation/* harness tree are budgeted separately below and
never inflate the production metric — scoping per
docs/CLEANUP_PROGRAM.md's cyclomatic section).

Rules, mirroring tests/test_code_size.py (the size ratchet):
- The ceiling moves DOWN whenever a landing shrinks the real max; run
  `python3 tools/complexity_report.py`, read the production max, and set
  CEILING to the exact number in the same change.
- The ceiling moves UP only with an in-commit justification: a ledger
  entry naming the function, the number, and why the growth is the
  cheapest safe design — the same discipline the size ratchet requires.
- A landing that raises the max without moving this constant FAILS the
  gate; a landing that moves the constant without justification fails
  review.

Ledger (exact counts, newest last):
- 2026-08-28 complexity lane: gate created. Production scope max at the
  baseline scan was 159 (SparkQwen38_27bModuleRunDsparkBlockForward;
  the audit line "max grew 157->158" measured the same function). This
  lane: stage zero on that function (getenv flags -> typed config read
  once at configure, inline /tmp dumps deleted, verbatim-motion helper
  extraction; 159 -> 75) and the conjunction-soup conversions
  (ValidateRankPlan 87 -> rows, ValidateDescriptor 80 -> rows,
  Dsv4ModuleConfigure 75 -> 46, glm5_next StagePackExpectedShape
  84 -> 21). Production max at landing: 75, all receipts in
  docs/AGENT_LANE_BRIEFS/reports/ccn-2026-08-28.md. Ceiling commits to
  75. The named P1 decomposition plan (qwen38_27b
  SubmitSpeculativeDecode 73, glm5_next InitializeTpCollective 70)
  continues from here — the next landing moves this number DOWN again.
- 2026-08-29 jit-safety lane: production mean 7.81 -> 7.84 over the
  same 2948-function scope; max unchanged at 75. The mean moves only
  because the four named safety hazards each add irreducible decision
  points to exactly the functions that own them: SparkKvCacheArenaEvict
  ResidentBlock branches IO-class vs loud write-back failures (the B1
  degrade-not-wedge contract; splitting the branch into a helper moves
  the decision, not removes it), SparkNvmeTierReserveWrite/Pump verify
  presented and landed SHA-256 digests (B3 collision/corruption fail-
  loud; the checks ARE the feature), SparkKvBackingResolvePath rejects
  path traversal per component (B4), and the glm5_next KV init fence
  (B2) is one guard on one identity. Every added branch returns a
  status; none nests. Receipts in
  docs/AGENT_LANE_BRIEFS/reports/jit-safety-2026-08-29.md. Mean
  ceiling commits to 7.84.
- 2026-08-29 W1 loader merge: production mean 7.84 -> 7.85 over a scope
  grown 2948 -> 2963 functions; max unchanged at 75. The +15 functions are
  the pipelined pack loader and the FEAT_SHA2 sha256 path, and their
  branches are the fail-closed contract itself: SparkStageModuleLoad
  PipelineRegion/Finish propagate worker-read and copy errors by poisoning
  the pipeline (Region-after-poison refuses; splitting moves the decision,
  not removes it), the per-slot cudaEvent guards decide buffer reuse, the
  env-and-size dispatch selects the retained synchronous body, and
  SparkSha256Update carries the block-boundary chunk cases plus the runtime
  FEAT_SHA2 selection (portable fallback must stay reachable). Every added
  branch returns a status or selects a path; none nests. Digest identity
  receipts in docs/AGENT_LANE_BRIEFS/reports/w1-loader-2026-08-29.md.
  Mean ceiling commits to 7.85.
- 2026-08-29 kernel-crew merge (K1-K4): production mean 7.85 -> 7.86;
  max unchanged at 75. The movement is the fail-frame error record
  itself: every integrated module's ExecuteFrame now clears the per-frame
  error word stream-ordered ahead of launches and checks+publishes it at
  completion (one clear + one check per frame path - the K1 contract),
  and the bounds fixes add one guard each at exactly the three call
  sites they own. Every added branch returns a status or publishes a
  fail-frame record; none nests. Mean ceiling commits to 7.86.
- 2026-08-29 r2-prefill R2c (dsv4 bulk causal prefill): production mean
  7.86 -> 7.87; max unchanged at 75. The movement is the bulk path
  replacing the wavefront: SparkDsv4ModuleRunCausalAttention gains the
  shadow/scatter/bulk-launch sequence with per-stage validation guards
  (each a fail-closed cudaError), the attention kernel gains the
  staged-row/shadow window resolution (one uniform branch pair
  selecting the value source - a decision the wavefront used to make
  by launch order), the window-source table fill (one serial pass,
  one comparison per frame row), and the snapshot kernel plus the two
  launchers carry their argument-validation guards. Every added branch
  returns a status or selects a value source; none nests. Receipts in
  docs/AGENT_LANE_BRIEFS/reports/r2-prefill-2026-08-29.md. Mean
  ceiling commits to 7.87.
  Coordinator merge #757 (kimi-k3 TP16 wave): the runner's submit
  gained the device-tier head-exchange branch and its host-tier
  fallback else (the U64Max winner path with output copies, guarded
  by the collective tier) - flat, status-returning branches in an
  nvcc-gated file. Mean 7.87 -> 7.88 (+~3 decision points over 3088
  functions); max CCN unchanged at 75.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from complexity_report import scan, scope_rows  # noqa: E402

# The committed production max-CCN ceiling. MOVES DOWN with every
# shrinking landing; moves up ONLY with a ledger entry above.
CEILING = 75

# The committed production MEAN-CCN ceiling (secondary guard: complexity
# may not silently spread). At landing: 7.81.
MEAN_CEILING_X100 = 788

# The validation harnesses (control-vs-candidate CUDA units, never merged
# into production) carry their OWN budget. It does not gate the production
# metric; it exists so harness complexity is at least VISIBLE and bounded.
# At landing: max 90 (SparkGlm52ValFixtureSetup), mean 8.98.
VALIDATION_MAX_BUDGET = 90
VALIDATION_MEAN_CEILING_X100 = 950


def main() -> int:
    rows = scan()
    production = scope_rows(rows, "production")
    validation = scope_rows(rows, "validation")

    production_max = max(row["ccn"] for row in production)
    production_mean_x100 = int(round(
        sum(row["ccn"] for row in production) * 100 / len(production)))
    validation_max = max(row["ccn"] for row in validation)
    validation_mean_x100 = int(round(
        sum(row["ccn"] for row in validation) * 100 / len(validation)))

    worst = production[0]
    print(f"production scope: {len(production)} functions, max CCN {production_max} "
          f"(ceiling {CEILING}), mean {production_mean_x100 / 100:.2f}")
    print(f"worst: {worst['file']}:{worst['function']}")
    print(f"validation scope (own budget): {len(validation)} functions, "
          f"max CCN {validation_max} (budget {VALIDATION_MAX_BUDGET})")

    failures = []
    if production_max > CEILING:
        failures.append(
            f"production max CCN {production_max} exceeds the committed "
            f"ceiling {CEILING}. Shrink it, or justify a new ceiling in the "
            f"same change: add a ledger entry to this test naming the "
            f"function, the number, and why the growth is the cheapest safe "
            f"design (see tests/test_code_size.py for the pattern).")
    if production_mean_x100 > MEAN_CEILING_X100:
        failures.append(
            f"production mean CCN {production_mean_x100 / 100:.2f} exceeds the "
            f"committed ceiling {MEAN_CEILING_X100 / 100:.2f}. Same rule: "
            f"shrink or justify in-commit.")
    if validation_max > VALIDATION_MAX_BUDGET:
        failures.append(
            f"validation harness max CCN {validation_max} exceeds its own "
            f"budget {VALIDATION_MAX_BUDGET}. The harnesses are deliberately "
            f"independent (control-vs-candidate doctrine) but they are not "
            f"exempt from arithmetic: shrink or justify in-commit.")
    if validation_mean_x100 > VALIDATION_MEAN_CEILING_X100:
        failures.append(
            f"validation harness mean CCN {validation_mean_x100 / 100:.2f} "
            f"exceeds its budget {VALIDATION_MEAN_CEILING_X100 / 100:.2f}.")

    if failures:
        for failure in failures:
            print("FAIL " + failure, file=sys.stderr)
        return 1

    if production_max < CEILING:
        print(f"note: ceiling is {CEILING - production_max} above reality; "
              f"lower CEILING to {production_max} with the next landing")
    print("the complexity ceiling held")
    return 0


if __name__ == "__main__":
    sys.exit(main())
