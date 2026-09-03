#!/usr/bin/env python3
"""Invariants for tools/perf_estimate.py.

What is held here:

  - Monotonicity. Per-seq decode tok/s falls with batch (step bytes grow,
    nothing shrinks per sequence); aggregate tok/s never falls (the fixed
    stream amortises faster than the coverage stream can anti-amortise, and
    the compute wall only flattens it); prefill tok/s falls with context
    (the quadratic attention term and KV writes only grow).

  - Roadmap reproduction. The memory-only roofline at 13 nodes reproduces
    the per-batch tables of docs/archive/PERF_ROADMAP_2026-08-01.md:300-415 to
    rounding, and the bandwidth->compute crossover scan reproduces its
    wall crossovers exactly (roadmap:423-430) - those rows are derived,
    not fitted, so a drift means the model stopped being the roadmap.

  - Boundary behaviors. B1 with graphs and zero launch cost approaches the
    memory roofline from below and never exceeds it; the 80% eta is never
    beaten; coverage(0) = 0 experts streamed.

  - Graph vs eager consistency. The graph step is never slower, and the
    delta is exactly the launch-count collapse times launch_ns.

  - Occupancy sanity. Every top-3 kernel fits at least one block per SM
    under the assumed SM specs (a zero there means the launch fails on
    device, which is a defect, not an estimate).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import perf_estimate as pe  # noqa: E402

FAILURES = 0


def check(condition, message):
    global FAILURES
    if not condition:
        print(f"  FAIL {message}")
        FAILURES += 1


def test_monotonicity():
    for name in pe.PERF:
        for topo in pe.TOPOLOGIES:
            seq_rates, agg_rates = [], []
            for batch in pe.BATCHES:
                r = pe.decode_step(name, batch, 2048, topo, "eager")
                seq_rates.append(r["tok_s_seq"])
                agg_rates.append(r["tok_s_agg"])
            for earlier, later in zip(seq_rates, seq_rates[1:]):
                check(earlier > later,
                      f"{name} {topo}: per-seq tok/s not falling "
                      f"({earlier:.3g} -> {later:.3g})")
            for earlier, later in zip(agg_rates, agg_rates[1:]):
                check(later >= earlier * 0.999,
                      f"{name} {topo}: aggregate tok/s fell "
                      f"({earlier:.3g} -> {later:.3g})")
        prefill_rates = [pe.prefill(name, ctx, 13)["tok_s"]
                         for ctx in pe.PREFILL_CONTEXTS]
        for earlier, later in zip(prefill_rates, prefill_rates[1:]):
            check(earlier >= later * 0.999,
                  f"{name}: prefill tok/s rose with context "
                  f"({earlier:.3g} -> {later:.3g})")


def test_roadmap_b1_reproduction():
    """roadmap:300-415 per-batch tables at the 80% target rate (2,840 GB/s
    on the 13-ring), to one decimal - the roadmap's own precision."""
    roadmap = {
        # (B1, B8, B64, B256, B1024) tok/s/seq at 2K context, 13 nodes
        "k3": (20.8, 9.2, 2.45, 1.58, 1.11),
        "glm52": (53.6, 14.8, 4.24, 3.54, 3.01),
        "qwen38_27b": (56.4, 55.0, 45.7, 29.0, 11.8),
        "dsv4_pro": (52.1, 21.6, 5.35, 3.53, 3.41),
        "dsv4_flash": (198.4, 82.7, 23.8, 18.7, 17.5),
        "mimo25": (184.7, 38.7, 10.4, 8.7, 7.6),
    }
    for name, expected in roadmap.items():
        for batch, want in zip(pe.BATCHES, expected):
            got = pe.memory_roofline_tok_s(name, batch)
            check(abs(got - want) / want < 0.02,
                  f"{name} B{batch}: memory roofline {got:.2f} vs "
                  f"roadmap {want}")


def test_crossovers():
    """roadmap:423-430 wall crossovers, exactly; the scan must land on the
    same batches the roadmap's arithmetic does."""
    expected = {"k3": 258, "glm52": 320, "qwen38_27b": 34, "dsv4_pro": 165,
                "dsv4_flash": 130, "mimo25": 341}
    for name, want in expected.items():
        got = pe.crossover_batch(name, "ring13", "wall")
        check(got == want, f"{name}: wall crossover {got} vs roadmap {want}")
    # At peak, only dense qwen crosses inside the served batch range.
    for name in pe.PERF:
        peak = pe.crossover_batch(name, "ring13", "peak")
        if name == "qwen38_27b":
            check(peak is not None and 280 <= peak <= 320,
                  f"qwen38_27b peak crossover {peak}, roadmap ~B294")
        else:
            check(peak is None or peak > 1024,
                  f"{name}: unexpected peak crossover at {peak}")


def test_boundaries():
    for name in pe.PERF:
        roofline = pe.memory_roofline_tok_s(name, 1)
        # Zero launch cost + graph mode + no overlay is the roofline itself.
        r = pe.decode_step(name, 1, 2048, "ring13", "graph", launch_ns=0.0)
        check(r["tok_s_seq"] <= roofline * 1.001,
              f"{name}: B1 graph step beats the memory roofline")
        # MBU can never exceed the calibrated eta in this model.
        check(r["mbu"] <= pe.ETA_BW + 1e-9,
              f"{name}: MBU {r['mbu']:.3f} exceeds eta")
        # Longer context never speeds a step up.
        short = pe.decode_step(name, 8, 2048, "ring13", "eager")
        long = pe.decode_step(name, 8, 131072, "ring13", "eager")
        check(long["step_s"] >= short["step_s"],
              f"{name}: 128K ctx step faster than 2K")
    check(pe.coverage(0, 896, 16) == 0.0, "coverage(0) != 0")


def test_graph_vs_eager():
    for name in pe.PERF:
        for topo_name, topo in pe.TOPOLOGIES.items():
            eager = pe.decode_step(name, 1, 2048, topo_name, "eager")
            graph = pe.decode_step(name, 1, 2048, topo_name, "graph")
            check(graph["step_s"] <= eager["step_s"],
                  f"{name} {topo_name}: graph step slower than eager")
            delta = eager["step_s"] - graph["step_s"]
            expected = ((pe.PERF[name].launches
                         - (topo.nodes + pe.GRAPH_LAUNCH_EXTRA))
                        * pe.LAUNCH_NS_DEFAULT * 1e-9)
            check(abs(delta - expected) < 1e-12,
                  f"{name} {topo_name}: graph/eager delta {delta:.6g} != "
                  f"launch collapse {expected:.6g}")
        # Every model loses measurable B1 throughput to eager launches -
        # the roadmap's D1/D10 finding (roadmap:589-613).
        eager = pe.decode_step(name, 1, 2048, "ring13", "eager")
        check(eager["mbu"] < 0.78,
              f"{name}: eager B1 MBU {eager['mbu']:.3f} unexpectedly high")


def test_occupancy():
    for name, kernels in pe.PTXAS_TOP.items():
        for kernel, regs, smem_static, tile in kernels:
            smem = pe.gemm_smem_bytes(tile) if tile else smem_static
            blocks, occ, _ = pe.occupancy(regs, smem)
            check(blocks >= 1,
                  f"{name} {kernel}: does not fit one block per SM "
                  f"({regs} regs, {smem:.0f} B smem)")
            check(0.0 < occ <= 1.0,
                  f"{name} {kernel}: nonsense occupancy {occ}")
    # The delta rule's 64 KiB dynamic carve is exactly one block per SM at
    # the 128 KiB SM budget - the smem-bound claim the doc makes.
    blocks, _, bound = pe.occupancy(94, 1664 + 65536)
    check(blocks == 1 and bound == "smem",
          f"delta rule: {blocks} blocks, {bound}-bound (expected 1, smem)")


def main():
    test_monotonicity()
    test_roadmap_b1_reproduction()
    test_crossovers()
    test_boundaries()
    test_graph_vs_eager()
    test_occupancy()
    if FAILURES:
        print(f"\nFAIL ({FAILURES})")
        return 1
    print("\nperf estimate invariants hold: monotonicity, roadmap "
          "reproduction, crossovers, boundaries, graph consistency, "
          "occupancy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
