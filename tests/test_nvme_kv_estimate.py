#!/usr/bin/env python3
"""The NVMe KV sizing model must keep reproducing the roadmap it derives from.

tools/nvme_kv_estimate.py is the arithmetic behind docs/archive/NVME_KV_SIZING.md -
the dedicate-the-external-NVMe decision and the resume-after-topology-switch
guarantee both stand on its verdict table. If the estimator drifts from the
per-batch roofline in docs/archive/PERF_ROADMAP_2026-08-01.md, the doc's tables go
stale while the drive purchase and the 20 s switch budget stay justified by
them. This gate holds the estimator to every number the doc claims it
reproduces:

  - the roadmap's per-model 2K step-byte tables (B8+ to rounding, B1 to the
    roadmap's own expert-lump slop - see the tolerance note below),
  - the roadmap's no-residency prefetch-budget table (roadmap:496-511),
  - the doc's verdict-table spot rows, admission caps, and capacity table,
  - the admission-cap invariant directly (cap = last feasible B), which the
    binary search's own docstring defers to this test.

Every constant the estimator assumes is flagged ESTIMATE in its docstring;
this test asserts DERIVED agreement, not the assumptions. When the ring
reports measured bandwidth or residency, the estimator's constants change
and the verdict expectations here move with them - that is the test doing
its job, not a failure to route around.
"""
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import nvme_kv_estimate as est

failures = []


def check(condition, label):
    print(f"  {'ok  ' if condition else 'FAIL'} {label}")
    if not condition:
        failures.append(label)


# The roadmap's per-batch 2K step tables (docs/archive/PERF_ROADMAP_2026-08-01.md:300-414).
# B8+ reproduces to rounding (0.3 GB covers the roadmap's 0.1 GB print
# precision stacked on the estimator's 0.1 GB model inputs). B1 does not:
# the roadmap's B1 expert column is a lump (Pro prints 14.1 where uniform
# coverage prices 12.0), so B1 rows get 2.5 GB - enough for the lump, tight
# enough that a real geometry drift (a wrong pool size, a dropped state
# term) still fails. Demand at B1 is always resident, so this tolerance
# never hides a verdict flip.
STEP_GB_2K = {
    "k3":        {1: 136.4, 8: 310.2, 64: 1157.9, 256: 1792.0, 1024: 2558.7},
    "glm52":     {1: 53.0, 8: 192.3, 64: 669.1, 256: 801.9, 1024: 943.4},
    # Qwen's table prints B1-B64 only; B256+ is compute-roofed, not stepped.
    "qwen36":    {1: 50.4, 8: 51.7, 64: 62.1},
    "dsv4_pro":  {1: 54.5, 8: 131.5, 64: 530.6, 256: 804.0, 1024: 832.1},
    "dsv4_flash": {1: 14.3, 8: 34.4, 64: 119.1, 256: 152.1, 1024: 162.0},
    "mimo25":    {1: 15.4, 8: 73.5, 64: 271.9, 256: 324.8, 1024: 373.7},
}

# The roadmap's prefetch-budget table (roadmap:501-511): no residency, 2K.
HEADLINE_GBPS = {
    ("glm52", 1): 9.9, ("glm52", 64): 50.1, ("glm52", 256): 166.9,
    ("k3", 256): 23.0, ("k3", 1024): 64.4, ("dsv4_pro", 1024): 63.7,
}

# Verdict-table spot rows (docs/archive/NVME_KV_SIZING.md:110-127), 13-node ring:
# (model, batch, context, per-node GB/s, verdict class).
VERDICT_ROWS = [
    ("glm52", 64, 131072, 1.74, "external-ok"),
    ("glm52", 8, 1048576, 0.84, "external-ok"),
    ("k3", 256, 131072, 50.92, "infeasible"),
    ("dsv4_pro", 64, 1048576, 0.08, "external-ok"),
]

# Admission caps at the 5 GB/s line (docs/archive/NVME_KV_SIZING.md:110-127).
CAPS_5GBPS = {
    ("k3", 131072): 111, ("k3", 1048576): 14,
    ("glm52", 131072): 130, ("glm52", 1048576): 95,
    ("qwen36", 131072): 49, ("qwen36", 1048576): 6,
    ("dsv4_pro", 131072): 379, ("dsv4_pro", 1048576): 1899,
    ("dsv4_flash", 131072): 537, ("dsv4_flash", 1048576): 66,
    ("mimo25", 131072): 182, ("mimo25", 1048576): 22,
}

# One node's drive, state included (docs/archive/NVME_KV_SIZING.md:173-193). Exact: the
# formula is the roadmap's own and the floor is deterministic.
SEQS_1TB = {
    ("k3", 8192): 1467, ("k3", 131072): 245, ("k3", 1048576): 33,
    ("glm52", 8192): 1358, ("glm52", 131072): 84, ("glm52", 1048576): 10,
    ("qwen36", 8192): 1776, ("qwen36", 131072): 116, ("qwen36", 1048576): 14,
    ("dsv4_pro", 8192): 13686, ("dsv4_pro", 131072): 855,
    ("dsv4_pro", 1048576): 106,
    ("dsv4_flash", 8192): 19950, ("dsv4_flash", 131072): 1253,
    ("dsv4_flash", 1048576): 156,
    ("mimo25", 8192): 5757, ("mimo25", 131072): 420, ("mimo25", 1048576): 53,
}


def main():
    print("nvme kv estimate\n")

    print("the 2K step-byte law reproduces the roadmap's per-model tables")
    for name, rows in STEP_GB_2K.items():
        model = est.MODELS_BY_NAME[name]
        for batch, roadmap_says in rows.items():
            got = est.step_gb(model, batch, 2048)
            tolerance = 2.5 if batch == 1 else 0.3
            check(abs(got - roadmap_says) <= tolerance,
                  f"{name} B{batch}: {got:.1f} GB vs roadmap {roadmap_says}")

    print("\nthe no-residency mode reproduces the roadmap's prefetch table")
    for (name, batch), roadmap_says in HEADLINE_GBPS.items():
        got = est.headline_prefetch_gbps(est.MODELS_BY_NAME[name], batch)
        check(abs(got - roadmap_says) <= 0.15,
              f"{name} B{batch}: {got:.1f} GB/s vs roadmap {roadmap_says}")

    print("\n2K chat never touches the drive through B1024")
    for model in est.MODELS:
        for batch in est.BATCHES:
            demand = est.nvme_demand_gbps(model, batch, 2048)
            if model.name == "k3" and batch == 1024:
                # fp32 KDA state alone exceeds the resident budget: the one
                # 2K exception, and inf - not 0.0 - is the honest answer.
                check(math.isinf(demand),
                      "k3 B1024 fp32 state is STATE-OVER-RESIDENT, not 0 GB/s")
            else:
                check(demand == 0.0,
                      f"{model.name} B{batch} x 2K is resident (0.00 GB/s)")

    print("\nthe verdict table's load-bearing rows hold")
    for name, batch, context, per_node, klass in VERDICT_ROWS:
        model = est.MODELS_BY_NAME[name]
        ring = est.nvme_demand_gbps(model, batch, context)
        got_per_node = ring / est.NODES_DEFAULT
        _, _, rest = est.residency_split(model, batch, context)
        check(abs(got_per_node - per_node) <= 0.01
              and est.verdict(got_per_node, rest >= 0.0) == klass,
              f"{name} B{batch} x {context}: {got_per_node:.2f} GB/s "
              f"{est.verdict(got_per_node, rest >= 0.0)} "
              f"(doc says {per_node} {klass})")

    print("\nadmission caps are the doc's, and each is the LAST feasible B")
    for (name, context), cap_says in CAPS_5GBPS.items():
        model = est.MODELS_BY_NAME[name]
        cap = est.admission_cap(model, context)
        check(cap == cap_says, f"{name} x {context}: cap {cap} "
                               f"(doc says {cap_says})")
        if cap > 0:
            fits_at = (est.nvme_demand_gbps(model, cap, context)
                       / est.NODES_DEFAULT <= est.USER_THRESHOLD_GBPS)
            over = est.nvme_demand_gbps(model, cap + 1, context)
            _, _, rest_over = est.residency_split(model, cap + 1, context)
            fits_above = (rest_over >= 0.0
                          and over / est.NODES_DEFAULT <= est.USER_THRESHOLD_GBPS)
            check(fits_at and not fits_above,
                  f"{name} x {context}: B{cap} fits, B{cap + 1} does not")

    print("\nthe capacity table is the roadmap's formula, floored")
    for (name, context), seqs_says in SEQS_1TB.items():
        got = est.seqs_fit(est.MODELS_BY_NAME[name], context, 1000.0)
        check(got == seqs_says,
              f"{name} x {context} on 1 TB: {got} (doc says {seqs_says})")

    print("\nthe K3 bf16-state lever does what the roadmap says it does")
    k3 = est.MODELS_BY_NAME["k3"]
    check(est.admission_cap(k3, 131072, bf16_state=True) == 118,
          "bf16 state moves the K3 128K cap 111 -> 118")
    check(est.nvme_demand_gbps(k3, 1024, 2048, bf16_state=True) == 0.0,
          "bf16 state puts K3 B1024 x 2K back inside the resident budget")

    print("\nthe resume-after-switch arithmetic the 20 s budget stands on")
    # The paused cohort's working set fits the tier with ~30x headroom and
    # reloads inside the overlap window: <= 32 GB/node resident budget at
    # the 5-7 GB/s internal range is 4.6-6.4 s (docs/archive/NVME_KV_SIZING.md:222-233).
    reload_lo = est.RESIDENT_KV_GB_PER_NODE / est.INTERNAL_GBPS_HI
    reload_hi = est.RESIDENT_KV_GB_PER_NODE / est.INTERNAL_GBPS_LO
    check(4.0 <= reload_lo <= 5.0 and 6.0 <= reload_hi <= 7.0,
          f"warm reload window {reload_lo:.1f}-{reload_hi:.1f} s "
          "stays inside the overlap budget")

    if failures:
        print(f"\n{len(failures)} expectation(s) broke between the estimator "
              "and the roadmap:")
        for label in failures:
            print(f"  {label}")
        return 1
    print("\nthe estimator still reproduces the roadmap it derives from")
    return 0


if __name__ == "__main__":
    sys.exit(main())
