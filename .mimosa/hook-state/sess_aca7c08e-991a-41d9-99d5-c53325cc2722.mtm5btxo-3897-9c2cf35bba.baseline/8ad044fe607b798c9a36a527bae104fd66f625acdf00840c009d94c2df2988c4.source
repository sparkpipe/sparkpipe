#!/usr/bin/env python3
"""NVMe bandwidth sizing for the JIT KV tier (cache/nvme_tier.c).

The decision this prices: internal NVMe is ~2x the external drive's bandwidth;
if 5 GB/s per node covers the KV tier's steady-state demand, a 4-8 TB external
NVMe becomes the KV backing store. Every output is an ESTIMATE built from the
per-batch roofline of docs/archive/PERF_ROADMAP_2026-08-01.md; docs/archive/NVME_KV_SIZING.md
is the write-up and carries the same numbers.

== ASSUMED CONSTANTS - measured nowhere, re-run with receipts ================

  INTERNAL_GBPS_LO/HI   5.0 / 7.0   GB10-class internal NVMe, assumed
                                    (roadmap:496-498 pins it nowhere)
  EXTERNAL_GBPS_LO/HI   2.5 / 3.5   the user's "external is 2x slower"
  RESIDENT_KV_GB_PER_NODE  32.0     assumed GPU+host KV budget per node: the
                                    ~29 GiB production GPU pool (GLM52_B1024_
                                    JIT_KV_INTEGRATION.md:53-58) plus host
                                    headroom. THE sensitive constant: spill
                                    crossovers move linearly with it.
  USER_THRESHOLD_GBPS   5.0         the decision line from the task: below
                                    this per node, dedicate the external drive
  NODES_DEFAULT         13          the ring every cited doc prices; --nodes
                                    for the 16-node TP16/PP16 end state

== DERIVED CONSTANTS - from model contracts and the roadmap, cited ===========

  RING_AGGREGATE_GBPS   2840.0      13 x 273 GB/s x eta 0.80 (roadmap:32-33);
                                    step time = step bytes / this, so demand
                                    is priced at the 80% target rate. The
                                    6.5 TFLOPS compute wall makes B256+ steps
                                    LONGER (roadmap:417-452), which lowers
                                    demand: these numbers are conservative.

Per-model geometry (kv bytes/token stored, fixed stream, expert pool, E, k,
state) is taken from the roadmap's per-batch tables and capacity table
(docs/archive/PERF_ROADMAP_2026-08-01.md:289-494) and cross-checks against
model_contracts/*.json; each row cites its line. GB is decimal (1e9 bytes):
the roadmap's own capacity arithmetic is decimal (GLM 1e12 / (89856 x 8192)
~ 1359, roadmap:483).

== THE MODEL =================================================================

Demand side. A decode step at batch B and context C reads every active
sequence's KV again (minus whatever token selection prunes), so KV traffic
scales with B x C. Placement is per SEQUENCE - that is what the JIT tier's
admission control actually does (GLM52_B1024_JIT_KV_INTEGRATION.md:62-66):
a sequence's blocks are either all upstairs or all on the drive.

  resident KV (ring) = nodes x resident_per_node - B x per-seq STATE
  spill sequences    = B - floor(resident KV / per-seq KV footprint)
  NVMe read demand   = spill sequences x per-seq KV read / step time

Recurrent state (K3 KDA, Qwen GDN) is READ AND WRITTEN every step; paging it
is absurd, so it is charged against resident memory for every active sequence
and is never NVMe-served. The lookahead hides latency, not bandwidth: demand
below supply means prefetch lands early, demand above means the step stalls -
there is no third outcome, which is why this is a bandwidth sizing and not a
latency one.

Supply side. Internal 5-7 GB/s per node, external 2.5-3.5, times --nodes.
"""
from __future__ import annotations

import argparse
import math
import sys

GB = 1e9

# -- assumed constants (see module docstring; change here, not in the math) --
INTERNAL_GBPS_LO = 5.0
INTERNAL_GBPS_HI = 7.0
EXTERNAL_GBPS_LO = 2.5
EXTERNAL_GBPS_HI = 3.5
RESIDENT_KV_GB_PER_NODE = 32.0
USER_THRESHOLD_GBPS = 5.0
NODES_DEFAULT = 13

# -- derived constant ---------------------------------------------------------
RING_AGGREGATE_GBPS = 2840.0   # 13 nodes x 273 GB/s x 0.80 (roadmap:32-33)


class Model:
    """One row of the roadmap's per-batch section, in keyword form.

    kv_store_bpt    whole-model KV bytes stored per token (roadmap:481-489,
                    unsharded: each ring rank holds only its layers' share)
    read_bpt        KV bytes read per token per decode step; equal to stored
                    unless compression stores less than a full read needs
    read_cap        GLM-style selection: read only the top slots, every step
                    (glm52.json dsa_selected_token_count, roadmap:315-318)
    select_after    DSv4-Pro-style: full read until the context passes this
                    many slots, then only selected_slots (roadmap:365-369)
    state_*         per-seq recurrent state: stream = R+W bytes per step,
                    store = resident bytes; never NVMe-served (rewritten each
                    step), bf16_factor = the halving lever (roadmap:454-471)
    swa_read_gb     fixed sliding-window re-read per seq per step (MiMo,
                    roadmap:404-406) - context-independent
    extra_read_gb   per-seq per-step reads that are not tier material (K3
                    AttnRes bank, roadmap:294-297): in the step time, never
                    in the NVMe demand
    """

    def __init__(self, name, kv_store_bpt, read_bpt, fixed_gb, pool_gb,
                 n_experts, top_k, state_stream_gb=0.0, state_store_gb=0.0,
                 bf16_state_factor=1.0, read_cap=None, select_after=None,
                 selected_slots=None, swa_read_gb=0.0, extra_read_gb=0.0):
        self.name = name
        self.kv_store_bpt = kv_store_bpt
        self.read_bpt = read_bpt
        self.fixed_gb = fixed_gb
        self.pool_gb = pool_gb
        self.n_experts = n_experts
        self.top_k = top_k
        self.state_stream_gb = state_stream_gb
        self.state_store_gb = state_store_gb
        self.bf16_state_factor = bf16_state_factor
        self.read_cap = read_cap
        self.select_after = select_after
        self.selected_slots = selected_slots
        self.swa_read_gb = swa_read_gb
        self.extra_read_gb = extra_read_gb


MODELS = [
    # roadmap:289-297 (fixed 109.65, pool 1446.7 = 896 x 17.55 MB x 92 L,
    # state 69 x 6.59 MB x 2 R+W, KV 24 x 1152 B x ctx, AR ~0.013)
    Model("k3", kv_store_bpt=27648, read_bpt=27648, fixed_gb=109.65,
          pool_gb=1446.7, n_experts=896, top_k=16,
          state_stream_gb=0.909, state_store_gb=0.455, bf16_state_factor=0.5,
          extra_read_gb=0.013),
    # roadmap:313-318 (KV 78 x 1152 B x min(ctx, 2048 selected))
    Model("glm52", kv_store_bpt=89856, read_bpt=89856, fixed_gb=30.2,
          pool_gb=724.8, n_experts=256, top_k=8, read_cap=2048),
    # roadmap:332-339 (dense: no pool; KV 16 L x 4 heads x 256 x K+V x 2 B;
    # GDN state already bf16, 0.052 GB/seq/step R+W, 26 MB stored)
    Model("qwen38_27b", kv_store_bpt=65536, read_bpt=65536, fixed_gb=50.2,
          pool_gb=0.0, n_experts=0, top_k=0,
          state_stream_gb=0.052, state_store_gb=0.026),
    # roadmap:363-369 (~8.9 KB/token compressed; top-k 1024 + window 128
    # selection binds only past ~128K compressed slots)
    Model("dsv4_pro", kv_store_bpt=8919, read_bpt=8919, fixed_gb=40.4,
          pool_gb=773.2, n_experts=384, top_k=6,
          select_after=131072, selected_slots=1152),
    # roadmap:384-390 (~6.1 KB/token; 2 sliding-window layers bounded at 128
    # slots are a small downward correction at long ctx, not modelled)
    Model("dsv4_flash", kv_store_bpt=6082, read_bpt=6082, fixed_gb=10.5,
          pool_gb=138.7, n_experts=256, top_k=6, state_store_gb=0.0003),
    # roadmap:401-406 (7 full layers x 2560 B + 41 SWA layers x 128 slots x
    # 5120 B fixed re-read)
    Model("mimo25", kv_store_bpt=17920, read_bpt=17920, fixed_gb=5.9,
          pool_gb=302.8, n_experts=256, top_k=8,
          state_store_gb=0.0269, swa_read_gb=0.0268),
]
MODELS_BY_NAME = {m.name: m for m in MODELS}

BATCHES = (1, 8, 64, 256, 1024)
CONTEXTS = (2048, 131072, 1048576)   # 2K chat, 128K agent, 1M worst case


def coverage(batch, n_experts, top_k):
    """Uniform-routing expert coverage: an upper bound against measured skew
    (roadmap:248-254), so the expert stream - and the step time, and so the
    demand - is never understated at B8-B128."""
    if n_experts == 0 or batch == 0:
        return 0.0
    return 1.0 - (1.0 - 1.0 / n_experts) ** (batch * top_k)


def kv_read_gb_per_seq(model, context_tokens):
    """What one sequence's attention reads from the KV store per decode step.
    Selection (GLM's cap, Pro's cliff past 128K) prunes the READ only; the
    stored footprint is always context_tokens x kv_store_bpt."""
    effective = context_tokens
    if model.select_after is not None and context_tokens > model.select_after:
        effective = model.selected_slots
    elif model.read_cap is not None:
        effective = min(context_tokens, model.read_cap)
    return model.read_bpt * effective / GB + model.swa_read_gb


def state_stream_gb(model, bf16_state):
    stream = model.state_stream_gb
    if bf16_state:
        stream *= model.bf16_state_factor
    return stream


def state_store_gb(model, bf16_state):
    store = model.state_store_gb
    if bf16_state:
        store *= model.bf16_state_factor
    return store


def step_gb(model, batch, context_tokens, bf16_state=False):
    """Decode step bytes = fixed stream + coverage x pool + per-seq terms,
    the roadmap's per-batch law (roadmap:241-247). Reproduces every per-model
    table at 2K to rounding; the test holds it to them."""
    per_seq = (state_stream_gb(model, bf16_state)
               + kv_read_gb_per_seq(model, context_tokens)
               + model.extra_read_gb)
    return (model.fixed_gb
            + model.pool_gb * coverage(batch, model.n_experts, model.top_k)
            + batch * per_seq)


def step_time_s(model, batch, context_tokens, bf16_state=False,
                ring_gbps=RING_AGGREGATE_GBPS):
    return step_gb(model, batch, context_tokens, bf16_state) / ring_gbps


def footprint_gb_per_seq(model, context_tokens):
    return model.kv_store_bpt * context_tokens / GB


def residency_split(model, batch, context_tokens, nodes=NODES_DEFAULT,
                    resident_per_node=RESIDENT_KV_GB_PER_NODE,
                    bf16_state=False):
    """(resident_sequences, spill_sequences, resident_kv_gb). State for every
    active sequence is charged first; a negative remainder means the batch is
    infeasible at any drive speed - admission must cut B or shrink state."""
    resident_kv = nodes * resident_per_node - batch * state_store_gb(
        model, bf16_state)
    footprint = footprint_gb_per_seq(model, context_tokens)
    if resident_kv < 0.0 or batch == 0:
        return 0, batch, resident_kv
    if footprint <= 0.0:
        return batch, 0, resident_kv
    resident = min(batch, int(resident_kv // footprint))
    return resident, batch - resident, resident_kv


def nvme_demand_gbps(model, batch, context_tokens, nodes=NODES_DEFAULT,
                     resident_per_node=RESIDENT_KV_GB_PER_NODE,
                     bf16_state=False, ring_gbps=RING_AGGREGATE_GBPS):
    """Ring-aggregate GB/s that MUST come from the tier in steady state:
    the spilled sequences' per-step KV read at the 80%-target step rate.
    inf when the per-seq state alone exceeds the resident budget: no drive
    speed fixes that, and 0.0 would be a lie the verdict table believes."""
    _, spill, resident_kv = residency_split(
        model, batch, context_tokens, nodes, resident_per_node, bf16_state)
    if resident_kv < 0.0:
        return math.inf
    if spill == 0:
        return 0.0
    return (spill * kv_read_gb_per_seq(model, context_tokens)
            / step_time_s(model, batch, context_tokens, bf16_state, ring_gbps))


def nvme_writeback_gbps(model, batch, context_tokens, nodes=NODES_DEFAULT,
                        resident_per_node=RESIDENT_KV_GB_PER_NODE,
                        bf16_state=False, ring_gbps=RING_AGGREGATE_GBPS):
    """Second-order but not zero: every spilled sequence appends one token of
    KV per step, and publish is the tier's only eviction path."""
    _, spill, resident_kv = residency_split(
        model, batch, context_tokens, nodes, resident_per_node, bf16_state)
    if resident_kv < 0.0:
        return math.inf
    if spill == 0:
        return 0.0
    return (spill * model.kv_store_bpt / GB
            / step_time_s(model, batch, context_tokens, bf16_state, ring_gbps))


def verdict(per_node_gbps, state_fits=True):
    """The decision classes. external-ok is the high end of the external
    range; internal-5 is the user's dedicate-the-external threshold."""
    if not state_fits:
        return "STATE-OVER-RESIDENT"
    if per_node_gbps <= 0.0:
        return "resident"
    if per_node_gbps <= EXTERNAL_GBPS_HI:
        return "external-ok"
    if per_node_gbps <= USER_THRESHOLD_GBPS:
        return "internal-5"
    if per_node_gbps <= INTERNAL_GBPS_HI:
        return "internal-only"
    return "infeasible"


def admission_cap(model, context_tokens, supply_per_node=USER_THRESHOLD_GBPS,
                  nodes=NODES_DEFAULT, resident_per_node=RESIDENT_KV_GB_PER_NODE,
                  bf16_state=False, max_batch=65536):
    """Largest batch whose steady-state tier demand fits the supply: the
    number admission control must enforce when the working set spills.
    Step time grows with B, so demand is not monotone at the far end in
    principle; over the served range (B <= 65536) it is, and the binary
    search's invariant (cap is the last feasible B) is what the test checks
    directly rather than trusting the search."""
    _, _, resident_kv = residency_split(
        model, 1, context_tokens, nodes, resident_per_node, bf16_state)
    if resident_kv < 0.0:
        return 0

    def fits(batch):
        demand = nvme_demand_gbps(model, batch, context_tokens, nodes,
                                  resident_per_node, bf16_state)
        _, _, rest = residency_split(model, batch, context_tokens, nodes,
                                     resident_per_node, bf16_state)
        return rest >= 0.0 and demand / nodes <= supply_per_node

    if not fits(1):
        return 0
    lo, hi = 1, 2
    while hi <= max_batch and fits(hi):
        lo, hi = hi, hi * 2
    hi = min(hi, max_batch + 1)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid
    return lo


def seqs_fit(model, context_tokens, budget_gb):
    """Whole sequences one node's drive holds: the roadmap capacity table's
    formula (roadmap:479-489), state included, decimal GB, floor."""
    per_seq = (model.kv_store_bpt * context_tokens
               + model.state_store_gb * GB)
    if per_seq <= 0.0:
        return 0
    return int(budget_gb * GB // per_seq)


def headline_prefetch_gbps(model, batch, context_tokens=2048,
                           ring_gbps=RING_AGGREGATE_GBPS):
    """The roadmap's prefetch-budget table (roadmap:496-511): KV demand with
    NO residency, i.e. the whole working set served live from the drive. Kept
    as the cross-check row set - this estimator reproduces those six numbers,
    then refines them with the residency model above."""
    return (batch * kv_read_gb_per_seq(model, context_tokens)
            / step_time_s(model, batch, context_tokens))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nodes", type=int, default=NODES_DEFAULT,
                        help="ring size (13 = the priced ring, 16 = TP16/PP16)")
    parser.add_argument("--resident-gb-per-node", type=float,
                        default=RESIDENT_KV_GB_PER_NODE)
    parser.add_argument("--k3-bf16-state", action="store_true",
                        help="halve K3 KDA state (roadmap:454-471; numerics "
                             "review is the gate, not this script)")
    parser.add_argument("--threshold-gbps", type=float,
                        default=USER_THRESHOLD_GBPS,
                        help="per-node supply line for the admission caps")
    args = parser.parse_args(argv)
    if args.nodes <= 0:
        parser.error("--nodes must be positive")

    common = dict(nodes=args.nodes, resident_per_node=args.resident_gb_per_node,
                  bf16_state=args.k3_bf16_state)
    print(f"ASSUMED: internal {INTERNAL_GBPS_LO}-{INTERNAL_GBPS_HI} GB/s, "
          f"external {EXTERNAL_GBPS_LO}-{EXTERNAL_GBPS_HI} GB/s per node, "
          f"resident KV {args.resident_gb_per_node} GB/node, "
          f"{args.nodes} nodes, ring {RING_AGGREGATE_GBPS:.0f} GB/s @80%"
          + (" (K3 bf16 state)" if args.k3_bf16_state else ""))
    print()
    print("DEMAND - steady-state NVMe read, model x batch x context")
    print(f"{'model':<10} {'B':>5} {'ctx':>8} {'step ms':>8} {'B_res':>6} "
          f"{'B_spill':>8} {'ring GB/s':>10} {'per-node':>9}  verdict")
    for context in CONTEXTS:
        for model in MODELS:
            for batch in BATCHES:
                res, spill, rest = residency_split(
                    model, batch, context, **common)
                ring = nvme_demand_gbps(model, batch, context, **common)
                per_node = ring / args.nodes
                step_ms = step_time_s(model, batch, context,
                                      args.k3_bf16_state) * 1000.0
                print(f"{model.name:<10} {batch:>5} {context:>8} "
                      f"{step_ms:>8.1f} {res:>6} {spill:>8} {ring:>10.1f} "
                      f"{per_node:>9.2f}  {verdict(per_node, rest >= 0.0)}")
        print()
    print("ADMISSION CAPS - largest B with demand <= "
          f"{args.threshold_gbps} GB/s per node")
    print(f"{'model':<10} {'ctx':>8} {'cap B':>7}")
    for model in MODELS:
        for context in CONTEXTS:
            cap = admission_cap(model, context, args.threshold_gbps, **common)
            suffix = "+" if cap >= 65536 else ""
            print(f"{model.name:<10} {context:>8} {cap:>6}{suffix}")
    print()
    print("CAPACITY - whole sequences on one node's drive (state included)")
    print(f"{'model':<10} {'ctx':>8} {'1 TB':>8} {'4 TB':>8} {'8 TB':>8}")
    for model in MODELS:
        for context in (8192, 131072, 1048576):
            row = [str(seqs_fit(model, context, tb * 1000.0))
                   for tb in (1, 4, 8)]
            print(f"{model.name:<10} {context:>8} {row[0]:>8} {row[1]:>8} "
                  f"{row[2]:>8}")
    print()
    print("HEADLINE CROSS-CHECK - no residency, 2K ctx, vs roadmap:501-511")
    for name, batch, roadmap_says in (
            ("glm52", 1, 9.9), ("glm52", 64, 50.1), ("glm52", 256, 166.9),
            ("k3", 256, 23.0), ("k3", 1024, 64.4), ("dsv4_pro", 1024, 63.7)):
        got = headline_prefetch_gbps(MODELS_BY_NAME[name], batch)
        print(f"  {name:<10} B{batch:<5} {got:>7.1f} GB/s   "
              f"(roadmap {roadmap_says})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
