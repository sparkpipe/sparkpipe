#!/usr/bin/env python3
"""Lattice-stage localizer: WHICH stage of the DFlash2 selector diverges, and whether
that divergence means anything.

The trajectory bisect says WHICH steps' module drafts differ from the oracle over the
same target taps. It cannot say which stage, and it cannot say whether a difference is
a defect. This tool answers both from the per-step lattice dumps, which carry the
module's own intermediates:

    step_<pos>_unary.f32    (B-1) x K      top-K unary logits (BF16-truncated head)
    step_<pos>_gate.bf16    (B-1) x R      context gate = bf16(final hidden @ Wproj^T)
    step_<pos>_edges.f32    (B-1) x K x K  unary[c] + sum_r pred_gate[p,r]*succ[c,r]
    step_<pos>_hidden.bf16  (B-1) x H      the final-normed hidden BOTH read (optional)
    step_<pos>_cands.u32    (B-1) x K      the module's own top-K ids       (optional)

STAGE (default mode). The oracle recomputes every stage from the taps and the diff
walks the pipeline in order, naming the first stage that differs:

    hidden differs                     -> FORWARD   (the 5 layers / bf16(rms_norm(x, norm.weight)))
    hidden equal, unary differs        -> W4 HEAD   (top-K over the target lm_head)
    unary equal, gate differs          -> W3 GATE   (hidden_projection)
    unary+gate equal, edges differ     -> W3 EDGE   (the rank-R lattice sum)
    all equal, drafts differ           -> W3 WALK   (argmax-first-max tie-break)

Without the optional hidden dump the FORWARD row cannot fire, and unary+gate differing
TOGETHER is the signature of their shared input rather than of two kernels; the summary
says which case it is looking at. Two counterfactual walks separate blame: the HYBRID
walk (oracle ids and oracle edge sum, the MODULE's unary and gate) reproducing the
module's drafts means the module's lattice INPUTS explain the flip, and the MODULE-EDGE
walk (the module's own lattice under oracle ids) reproducing them means the module's
candidate set agrees along the walked path.

--noise-floor. The arbiter for "is this a defect". It runs the ORACLE against ITSELF
with only the accumulation width of its dot products changed (fp32 -> fp64; identical
math, identical truncation points) and reports the same statistics. Measured on the
o128 dumps: 300353 of 358400 final-hidden BF16 words move, the top-16 candidate set
changes at every step, and the oracle's own drafts change at 5/10 steps. The drafter
truncates to BF16 after every projection in five layers, so its low bits are chaotic
and its draft ids are ONE SAMPLE from a noise band - a module flip rate below the
oracle's own self-flip rate carries no information about correctness.

--sensitivity. The dtype-free control for the same claim: one BF16 ULP on ONE tap
element, same fp32 forward. Measured: 118459/179200 hidden words move, the top-K set
changes at 4/5 steps, the drafts at 1/5. Also self-tests forward_at(fp32) against
rail.forward (must be bit-equal, else the fp64 column means nothing).

--exact. The contract computed in float64 from the oracle's hidden: proves the oracle's
own fp32 head and projection ARE the exactly-rounded BF16 values, so the noise is in
the forward, not in the selector tail. The module columns of that block compare the
module's values against a truth derived from the ORACLE's hidden, so they are only a
verdict on the module once the hidden dump shows the two hiddens agree.

usage: qwen36_dspark_lattice_stage.py DUMP_DIR [--noise-floor] [--sensitivity] [--exact]
                                      [POSITION ...]   (no positions = every step)
"""
from __future__ import annotations

import re
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen36_dspark_reference as ref  # noqa: E402
import qwen36_dspark_e2e_parity as rail  # noqa: E402

_CACHE: dict = {}
SELECTOR_NAMES = ("candidate_selector.predecessor_codebook",
                  "candidate_selector.successor_codebook",
                  "candidate_selector.hidden_projection.weight")


def _cached(name, loader):
    """Load the 18 GB of weights ONCE; rail.forward would otherwise reload per step."""
    def wrapper():
        if name not in _CACHE:
            _CACHE[name] = loader()
        return _CACHE[name]
    return wrapper


ref.load_drafter = _cached("drafter", ref.load_drafter)
ref.load_target_shared = _cached("target", ref.load_target_shared)


def selector_weights():
    if "selector" not in _CACHE:
        _CACHE["selector"] = {
            name: ref.bf16_to_f32(ref.read_safetensors_tensor(
                ref.DRAFTER / "model.safetensors", name)).copy()
            for name in SELECTOR_NAMES}
    return _CACHE["selector"]


def bf16_ulp(value: np.ndarray) -> np.ndarray:
    """Spacing of the BF16 grid at |value| - the unit a truncation difference costs."""
    words = ref.f32_to_bf16(np.abs(np.asarray(value, dtype=np.float32))).astype(np.uint32)
    spacing = np.abs(ref.bf16_to_f32((words + 1).astype(np.uint16)) -
                     ref.bf16_to_f32(words.astype(np.uint16))).astype(np.float32)
    return np.where(spacing > 0.0, spacing, np.float32(np.finfo(np.float32).tiny))


def ulps(a: np.ndarray, b: np.ndarray) -> float:
    """Max |a - b| in BF16 ULPs at that magnitude. 1.0 = one truncation step."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b) / bf16_ulp(np.maximum(np.abs(a), np.abs(b)))))


def drafter_at(dtype):
    """The drafter weights cast to dtype, cached. Only the ACCUMULATION width of the
    projections changes: every truncation point still rounds to BF16, and RoPE and the
    softmax already run in fp32 on both paths, so this isolates exactly what the module
    and numpy genuinely do differently - the order and width of the long dot products."""
    key = f"drafter_{np.dtype(dtype).name}"
    if key not in _CACHE:
        _CACHE[key] = {name: value.astype(dtype) for name, value in ref.load_drafter().items()}
    return _CACHE[key]


def forward_at(taps: np.ndarray, c0: int, base_pos: float, dtype):
    """rail.forward's math with the drafter weights at dtype. Returns (logits, hidden)."""
    weights = drafter_at(dtype)
    lm_head, embed_tokens = ref.load_target_shared()
    positions_q = np.arange(base_pos, base_pos + ref.BLOCK, dtype=np.float32)
    ctx = ref.bf16(weights["fc.weight"] @ taps.reshape(-1).astype(dtype))
    ctx = ref.bf16(ref.rms_norm(ctx, weights["hidden_norm.weight"]))
    block = np.empty((ref.BLOCK, ref.HIDDEN), dtype=np.float32)
    block[0] = embed_tokens[c0]
    block[1:] = embed_tokens[ref.MASK_TOKEN_ID]
    x = block
    for layer in range(ref.N_LAYERS):
        lw = {name.split(f"layers.{layer}.")[1]: weights[name]
              for name in weights if f"layers.{layer}." in name}
        x = ref.forward_layer(x, ctx, lw, positions_q, base_pos - 1.0)
    hidden = ref.bf16(ref.rms_norm(x, weights["norm.weight"]))
    logits = (hidden @ lm_head.T).astype(np.float32)
    return ref.bf16_to_f32(ref.f32_to_bf16(logits)), hidden


def lattice_from(hidden_rows, logits_rows, c0, weights):
    """The selector tail (top-K, gate, edges, drafts) for one forward's outputs."""
    top_ids = np.argsort(-logits_rows, axis=-1, kind="stable")[:, :ref.SELECTOR_TOP_K]
    unary = np.take_along_axis(logits_rows, top_ids, axis=-1).astype(np.float32)
    gate_words = ref.f32_to_bf16((hidden_rows @ weights[SELECTOR_NAMES[2]].T).astype(np.float32))
    gate = ref.bf16_to_f32(gate_words).astype(np.float32)
    edges = ref._score_edges(weights[SELECTOR_NAMES[0]], weights[SELECTOR_NAMES[1]],
                             top_ids[None], unary[None], gate[None], np.array([c0]),
                             ref.SELECTOR_TOP_K)[0]
    return {"ids": top_ids, "unary": unary, "gate": gate, "gate_words": gate_words,
            "edges": edges, "drafts": ref._greedy_walk(edges, top_ids)}


def read_step(directory: Path, position: int):
    taps_raw = np.fromfile(directory / f"step_{position}_taps.bin", dtype=np.uint16)
    if taps_raw.size != ref.TAPS * ref.HIDDEN:
        return None
    taps = ref.bf16_to_f32(taps_raw).reshape(ref.TAPS, ref.HIDDEN)
    c0 = struct.unpack("<I", (directory / f"step_{position}_c0.bin").read_bytes()[:4])[0]
    payload = (directory / f"step_{position}_drafts.bin").read_bytes()
    drafts = list(struct.unpack("<%dI" % (len(payload) // 4), payload))
    steps = len(drafts)
    lattice = {}
    required = (("unary.f32", np.float32, (steps, ref.SELECTOR_TOP_K)),
                ("gate.bf16", np.uint16, (steps, ref.SELECTOR_RANK)),
                ("edges.f32", np.float32, (steps, ref.SELECTOR_TOP_K, ref.SELECTOR_TOP_K)))
    # hidden/cands are optional: older dumps lack them and the table says so rather
    # than silently inferring what they would have proven.
    optional = (("hidden.bf16", np.uint16, (steps, ref.HIDDEN)),
                ("cands.u32", np.uint32, (steps, ref.SELECTOR_TOP_K)))
    for suffix, dtype, shape in required + optional:
        path = directory / f"step_{position}_{suffix}"
        if not path.exists():
            if (suffix, dtype, shape) in required:
                return None
            continue
        raw = np.fromfile(path, dtype=dtype)
        if raw.size != int(np.prod(shape)):
            if (suffix, dtype, shape) in required:
                return None
            continue
        lattice[suffix.split(".")[0]] = raw.reshape(shape)
    return taps, c0, drafts, lattice


def positions_in(directory: Path) -> list[int]:
    found = []
    for taps_path in directory.glob("step_*_taps.bin"):
        match = re.match(r"step_(\d+)_taps\.bin$", taps_path.name)
        if match is not None:
            found.append(int(match.group(1)))
    return sorted(found)


def exact_truth(hidden_rows: np.ndarray, logits_rows: np.ndarray, c0: int, weights: dict):
    """The contract computed in float64 - the arbiter neither fp32 path is.

    Every input to the head and the projection is BF16, exact in f64, so a float64
    accumulation IS the real value and bf16(exact) is what the contract asks for. The
    true top-K SET would need the whole vocabulary, so this re-ranks the fp32 top-64 in
    f64 and prints the K-th place margin, which is what makes that shortcut checkable.
    """
    lm_head, _embed = ref.load_target_shared()
    pool = np.argsort(-logits_rows, axis=-1, kind="stable")[:, :64]
    hidden64 = hidden_rows.astype(np.float64)
    exact = np.empty(pool.shape, dtype=np.float64)
    for row in range(pool.shape[0]):
        exact[row] = lm_head[pool[row]].astype(np.float64) @ hidden64[row]
    exact_bf16 = ref.bf16_to_f32(ref.f32_to_bf16(exact.astype(np.float32))).astype(np.float32)
    ids = np.empty((pool.shape[0], ref.SELECTOR_TOP_K), dtype=np.int64)
    unary = np.empty((pool.shape[0], ref.SELECTOR_TOP_K), dtype=np.float32)
    boundary = []
    for row in range(pool.shape[0]):
        # BF16 truncation first, THEN top-K with id-ascending ties: the contract's order.
        order = sorted(range(pool.shape[1]),
                       key=lambda index: (-exact_bf16[row][index], int(pool[row][index])))
        ids[row] = pool[row][order[:ref.SELECTOR_TOP_K]]
        unary[row] = exact_bf16[row][order[:ref.SELECTOR_TOP_K]]
        boundary.append(float(exact_bf16[row][order[ref.SELECTOR_TOP_K - 1]] -
                              exact_bf16[row][order[ref.SELECTOR_TOP_K]]))
    gate = ref.bf16_to_f32(ref.f32_to_bf16(
        (hidden64 @ weights[SELECTOR_NAMES[2]].astype(np.float64).T).astype(np.float32))).astype(np.float32)
    edges = ref._score_edges(weights[SELECTOR_NAMES[0]].astype(np.float64),
                             weights[SELECTOR_NAMES[1]].astype(np.float64), ids[None],
                             unary[None].astype(np.float64), gate[None].astype(np.float64),
                             np.array([c0]), ref.SELECTOR_TOP_K)[0]
    return {"ids": ids, "unary": unary, "gate": gate, "edges": edges,
            "drafts": ref._greedy_walk(edges, ids), "topk_boundary": min(boundary)}


def analyse(position: int, taps, c0, drafts, lattice, exact: bool = False):
    weights = selector_weights()
    logits, hidden = rail.forward(ref.bf16(taps), c0, float(position))[:2]
    rows = slice(1, 1 + len(drafts))
    oracle = lattice_from(hidden[rows], logits[rows], c0, weights)
    top_ids, unary_r, gate_r, edges_r = (oracle["ids"], oracle["unary"],
                                         oracle["gate"], oracle["edges"])
    drafts_r = oracle["drafts"]
    anchor = np.array([c0])

    def score(ids, unary, gate):
        return ref._score_edges(weights[SELECTOR_NAMES[0]], weights[SELECTOR_NAMES[1]],
                                ids[None], unary[None], gate[None], anchor,
                                ref.SELECTOR_TOP_K)[0]

    unary_m = lattice["unary"].astype(np.float32)
    gate_m = ref.bf16_to_f32(lattice["gate"]).astype(np.float32)
    edges_m = lattice["edges"].astype(np.float32)

    # Counterfactuals under the ORACLE's ids, so the only thing that changes is which
    # stage's numbers are the module's.
    edges_hybrid = score(top_ids, unary_m, gate_m)
    hybrid_matches = ref._greedy_walk(edges_hybrid, top_ids) == list(drafts)
    module_edge_matches = ref._greedy_walk(edges_m, top_ids) == list(drafts)
    # edges_m vs edges_hybrid mixes a candidate-id difference (whole rows/columns
    # permuted) with the rank-R sum order; only step_<pos>_cands.u32 separates them,
    # which is why edge_own_abs below is the edge kernel's real fidelity number.
    edge_input_rel = 0.0
    if edges_hybrid.size and np.max(np.abs(edges_hybrid)) > 0.0:
        edge_input_rel = float(np.max(np.abs(edges_m - edges_hybrid)) /
                               np.max(np.abs(edges_hybrid)))

    hidden_bad = hidden_ulp = None
    if "hidden" in lattice:
        # The ONE input the head and the projection share. unary AND gate differing
        # together cannot be caused by any single kernel below it, so this field is
        # what turns that inference into an observation.
        words_r = ref.f32_to_bf16(hidden[rows])
        hidden_bad = int(np.count_nonzero(lattice["hidden"] != words_r))
        hidden_ulp = ulps(ref.bf16_to_f32(lattice["hidden"]), ref.bf16_to_f32(words_r))
    ids_bad = edge_own_abs = None
    if "cands" in lattice:
        ids_module = lattice["cands"].astype(np.int64)
        ids_bad = int(np.count_nonzero(ids_module != top_ids))
        edge_own_abs = float(np.max(np.abs(edges_m - score(ids_module, unary_m, gate_m))))

    unary_bad = int(np.count_nonzero(unary_m != unary_r))
    gate_bad = int(np.count_nonzero(lattice["gate"] != oracle["gate_words"]))
    edge_bad = int(np.count_nonzero(edges_m != edges_r))
    if hidden_bad:
        stage = "FORWARD"
    elif unary_bad:
        stage = "W4-HEAD"
    elif gate_bad:
        stage = "W3-GATE"
    elif edge_bad:
        stage = "W3-EDGE"
    elif list(drafts) != list(drafts_r):
        stage = "W3-WALK"
    else:
        stage = "-"

    flip = next((slot for slot, (a, b) in enumerate(zip(drafts, drafts_r)) if a != b), None)
    margin_ulp = None
    if flip is not None:
        previous = 0
        for slot in range(flip):
            previous = int(np.where(top_ids[slot] == drafts_r[slot])[0][0])
        row = np.sort(edges_r[flip, previous])[::-1]
        margin_ulp = float((row[0] - row[1]) /
                           bf16_ulp(np.array([row[0]], dtype=np.float32))[0])
    return {
        "position": position, "c0": c0, "stage": stage,
        "parity": list(drafts) == list(drafts_r),
        "unary_bad": unary_bad, "unary_ulp": ulps(unary_m, unary_r),
        # A BF16-truncated head must emit BF16-representable logits; anything else means
        # the module truncates at a different point than truncate-then-top-K.
        "unary_not_bf16": int(np.count_nonzero(unary_m != ref.bf16(unary_m))),
        # Same K values in a different order = a pure tie-order permutation; a different
        # multiset = the head really disagrees.
        "unary_permutation": bool(np.array_equal(np.sort(unary_m, axis=-1),
                                                 np.sort(unary_r, axis=-1))),
        "gate_bad": gate_bad, "gate_ulp": ulps(gate_m, gate_r),
        "edge_bad": edge_bad, "edge_ulp": ulps(edges_m, edges_r),
        "edge_input_rel": edge_input_rel, "flip_slot": flip, "flip_margin_ulp": margin_ulp,
        "hidden_bad": hidden_bad, "hidden_ulp": hidden_ulp,
        "ids_bad": ids_bad, "edge_own_abs": edge_own_abs,
        "hybrid_matches_module": hybrid_matches,
        "module_edge_matches_module": module_edge_matches,
        "drafts_outside_oracle_topk": [slot for slot, draft in enumerate(drafts)
                                       if draft not in set(top_ids[slot].tolist())],
        "truth": None if not exact else exact_summary(
            exact_truth(hidden[rows], logits[rows], c0, weights),
            drafts, drafts_r, top_ids, unary_m, unary_r, gate_m, gate_r),
    }


def exact_summary(truth, drafts, drafts_r, top_ids, unary_m, unary_r, gate_m, gate_r):
    return {
        "module_matches_truth": list(drafts) == list(truth["drafts"]),
        "oracle_matches_truth": list(drafts_r) == list(truth["drafts"]),
        "ids_equal_oracle": bool(np.array_equal(truth["ids"], top_ids)),
        "unary_module_exact": int(np.count_nonzero(unary_m == truth["unary"])),
        "unary_oracle_exact": int(np.count_nonzero(unary_r == truth["unary"])),
        "gate_module_exact": int(np.count_nonzero(gate_m == truth["gate"])),
        "gate_oracle_exact": int(np.count_nonzero(gate_r == truth["gate"])),
        "unary_words": int(unary_m.size), "gate_words": int(gate_m.size),
        "topk_boundary": truth["topk_boundary"],
    }


def noise_floor(position: int, taps, c0, drafts, lattice):
    """The oracle against ITSELF, fp32 vs fp64 accumulation, same truncation points."""
    weights = selector_weights()
    logits32, hidden32 = forward_at(ref.bf16(taps), c0, float(position), np.float32)
    logits64, hidden64 = forward_at(ref.bf16(taps), c0, float(position), np.float64)
    rows = slice(1, 1 + len(drafts))
    lattice32 = lattice_from(hidden32[rows], logits32[rows], c0, weights)
    lattice64 = lattice_from(hidden64[rows], logits64[rows], c0, weights)
    words32 = ref.f32_to_bf16(hidden32[rows])
    words64 = ref.f32_to_bf16(hidden64[rows])
    return {
        "position": position, "hidden_words": int(words32.size),
        "hidden_bad": int(np.count_nonzero(words32 != words64)),
        "hidden_ulp": ulps(hidden32[rows], hidden64[rows]),
        "ids_equal": bool(np.array_equal(lattice32["ids"], lattice64["ids"])),
        "unary_bad": int(np.count_nonzero(lattice32["unary"] != lattice64["unary"])),
        "unary_ulp": ulps(lattice32["unary"], lattice64["unary"]),
        "gate_bad": int(np.count_nonzero(lattice32["gate_words"] != lattice64["gate_words"])),
        "gate_ulp": ulps(lattice32["gate"], lattice64["gate"]),
        "edge_ulp": ulps(lattice32["edges"], lattice64["edges"]),
        "self_parity": list(lattice32["drafts"]) == list(lattice64["drafts"]),
        "module_matches_f32": list(drafts) == list(lattice32["drafts"]),
        "module_matches_f64": list(drafts) == list(lattice64["drafts"]),
    }


def sensitivity(position: int, taps, c0, drafts, lattice):
    """The dtype-free control: ONE BF16 ULP on ONE tap element, same fp32 forward. The
    smallest change the target could hand the drafter, with no implementation difference
    at all, so whatever it moves is the contract's own sensitivity. Also self-tests
    forward_at(fp32) against rail.forward - they must be bit-equal for the fp64
    noise-floor column to mean anything."""
    weights = selector_weights()
    taps_a = ref.bf16(taps)
    words = ref.f32_to_bf16(taps_a).copy()
    words[0][0] = np.uint16(words[0][0] + 1)
    taps_b = ref.bf16_to_f32(words).astype(np.float32)
    logits_a, hidden_a = rail.forward(taps_a, c0, float(position))[:2]
    logits_b, hidden_b = rail.forward(taps_b, c0, float(position))[:2]
    logits_c, hidden_c = forward_at(taps_a, c0, float(position), np.float32)
    rows = slice(1, 1 + len(drafts))
    lattice_a = lattice_from(hidden_a[rows], logits_a[rows], c0, weights)
    lattice_b = lattice_from(hidden_b[rows], logits_b[rows], c0, weights)
    words_a = ref.f32_to_bf16(hidden_a[rows])
    return {
        "position": position,
        "self_test": bool(np.array_equal(ref.f32_to_bf16(hidden_a), ref.f32_to_bf16(hidden_c))
                          and np.array_equal(logits_a, logits_c)),
        "hidden_words": int(words_a.size),
        "hidden_bad": int(np.count_nonzero(words_a != ref.f32_to_bf16(hidden_b[rows]))),
        "hidden_ulp": ulps(hidden_a[rows], hidden_b[rows]),
        "ids_equal": bool(np.array_equal(lattice_a["ids"], lattice_b["ids"])),
        "unary_ulp": ulps(lattice_a["unary"], lattice_b["unary"]),
        "gate_ulp": ulps(lattice_a["gate"], lattice_b["gate"]),
        "drafts_equal": list(lattice_a["drafts"]) == list(lattice_b["drafts"]),
    }


def stage_table(directory: Path, wanted: list[int], exact: bool):
    print(f"steps            = {len(wanted)} from {directory}")
    print(f"{'pos':>6} {'stage':>8} {'parity':>7} {'unaryd':>7} {'unULP':>7} {'gated':>6} "
          f"{'gtULP':>7} {'edged':>6} {'edgeULP':>8} {'flip':>5} {'margULP':>9} "
          f"{'hybrid':>7} {'modedge':>8} {'edgeInRel':>9}")
    rows = []
    for position in wanted:
        step = read_step(directory, position)
        if step is None:
            print(f"{position:>6} {'SKIP':>8}  (incomplete dump - no lattice files)")
            continue
        row = analyse(position, *step, exact=exact)
        rows.append(row)
        print(f"{row['position']:>6} {row['stage']:>8} "
              f"{('PARITY' if row['parity'] else 'FLIP'):>7} {row['unary_bad']:>7} "
              f"{row['unary_ulp']:>7.2f} {row['gate_bad']:>6} {row['gate_ulp']:>7.2f} "
              f"{row['edge_bad']:>6} {row['edge_ulp']:>8.2f} "
              f"{(row['flip_slot'] if row['flip_slot'] is not None else -1):>5} "
              f"{(row['flip_margin_ulp'] if row['flip_margin_ulp'] is not None else float('nan')):>9.3f} "
              f"{str(row['hybrid_matches_module']):>7} "
              f"{str(row['module_edge_matches_module']):>8} {row['edge_input_rel']:>9.2e}")
    return rows


def stage_summary(rows: list[dict]):
    flips = [row for row in rows if not row["parity"]]
    print()
    print(f"steps analysed    = {len(rows)}   flipping = {len(flips)} "
          f"({100.0 * len(flips) / len(rows):.1f}%)")
    print(f"flipping positions= {[row['position'] for row in flips]}")
    for name in ("FORWARD", "W4-HEAD", "W3-GATE", "W3-EDGE", "W3-WALK", "-"):
        count = sum(1 for row in rows if row["stage"] == name)
        if count:
            print(f"  first differing stage {name:>8} : {count} steps")
    print(f"unary not BF16-representable (truncate-then-top-K violated) : "
          f"{sum(row['unary_not_bf16'] for row in rows)}")
    print(f"max unary {max(row['unary_ulp'] for row in rows):.2f} ULP   "
          f"max gate {max(row['gate_ulp'] for row in rows):.2f} ULP   "
          f"max edge {max(row['edge_ulp'] for row in rows):.2f} ULP")
    print(f"unary difference that is only a tie-order permutation : "
          f"{sum(1 for row in rows if row['unary_permutation'] and row['unary_bad'])}"
          f" of {sum(1 for row in rows if row['unary_bad'])} steps with a unary difference")
    both = sum(1 for row in rows if row["unary_bad"] and row["gate_bad"])
    observed = [row for row in rows if row["hidden_bad"] is not None]
    if observed:
        total = sum(row["hidden_bad"] for row in observed)
        print(f"final-normed hidden vs oracle (OBSERVED) : {total} differing BF16 words over "
              f"{len(observed)} steps"
              + (f", max {max(row['hidden_ulp'] for row in observed):.2f} ULP -> the differing "
                 "truncation point is bf16(rms_norm(x, norm.weight)), upstream of both "
                 "selector stages" if total else
                 " -> BIT-EQUAL, so any divergence above is inside the selector kernels"))
    else:
        print(f"final-normed hidden : NOT DUMPED. unary AND gate differ together on "
              f"{both}/{len(rows)} steps, and they share exactly one input, so the stage "
              f"above unary is INFERRED to be the forward; re-run with a module that "
              f"writes step_<pos>_hidden.bf16 to observe it")
    identified = [row for row in rows if row["ids_bad"] is not None]
    if identified:
        print(f"module top-K ids vs oracle : {sum(row['ids_bad'] for row in identified)} "
              f"differing slots; edges rescored over the MODULE's own ids : max abs "
              f"{max(row['edge_own_abs'] for row in identified):.6f} (the edge kernel's "
              f"own fidelity, unlike edgeInRel)")
    else:
        print("module top-K ids : NOT DUMPED - edgeInRel mixes id differences with "
              "accumulation order (re-run with step_<pos>_cands.u32)")
    outside = [(row["position"], row["drafts_outside_oracle_topk"]) for row in rows
               if row["drafts_outside_oracle_topk"]]
    print(f"steps whose drafts leave the oracle top-K set : {outside if outside else 'none'}")
    if flips:
        print(f"flips explained by the module's lattice INPUTS : "
              f"{sum(1 for row in flips if row['hybrid_matches_module'])}/{len(flips)}   "
              f"reproduced from the module's own edges : "
              f"{sum(1 for row in flips if row['module_edge_matches_module'])}/{len(flips)}")
        margins = [row["flip_margin_ulp"] for row in flips
                   if row["flip_margin_ulp"] is not None]
        if margins:
            print(f"oracle decision margin at the flip slot : min {min(margins):.3f} ULP, "
                  f"median {sorted(margins)[len(margins) // 2]:.3f}, max {max(margins):.3f}")
        print("first flip slot histogram : "
              f"{ {slot: sum(1 for row in flips if row['flip_slot'] == slot) for slot in sorted({row['flip_slot'] for row in flips})} }")


def main() -> int:
    arguments = sys.argv[1:]
    flags = {name for name in arguments if name.startswith("--")}
    arguments = [name for name in arguments if not name.startswith("--")]
    unknown = flags - {"--exact", "--noise-floor", "--sensitivity"}
    if not arguments or unknown:
        raise SystemExit(__doc__ if not unknown else f"unknown flag(s): {sorted(unknown)}")
    directory = Path(arguments[0])
    wanted = [int(argument) for argument in arguments[1:]] or positions_in(directory)
    if not wanted:
        raise SystemExit(f"no step_*_taps.bin dumps in {directory} "
                         f"(run the module with SPARK_QWEN36_DSPARK_DUMP_DIR set)")
    rows = stage_table(directory, wanted, "--exact" in flags)
    if not rows:
        raise SystemExit("no complete lattice dumps to analyse")
    stage_summary(rows)
    if "--noise-floor" in flags:
        print()
        print("NOISE FLOOR - the oracle against ITSELF, fp32 vs fp64 accumulation")
        print(f"{'pos':>6} {'hiddend':>8} {'hidULP':>7} {'ids=':>5} {'unaryd':>7} "
              f"{'unULP':>7} {'gated':>6} {'gtULP':>7} {'edgeULP':>8} {'self':>7} "
              f"{'mod=f32':>8} {'mod=f64':>8}")
        floors = []
        for row in rows:
            step = read_step(directory, row["position"])
            entry = noise_floor(row["position"], *step)
            floors.append(entry)
            print(f"{entry['position']:>6} {entry['hidden_bad']:>8} {entry['hidden_ulp']:>7.2f} "
                  f"{str(entry['ids_equal']):>5} {entry['unary_bad']:>7} "
                  f"{entry['unary_ulp']:>7.2f} {entry['gate_bad']:>6} {entry['gate_ulp']:>7.2f} "
                  f"{entry['edge_ulp']:>8.2f} "
                  f"{('PARITY' if entry['self_parity'] else 'FLIP'):>7} "
                  f"{str(entry['module_matches_f32']):>8} {str(entry['module_matches_f64']):>8}")
        words = sum(entry["hidden_words"] for entry in floors)
        print(f"oracle self-disagreement: hidden {sum(e['hidden_bad'] for e in floors)}/{words} "
              f"BF16 words, max {max(e['hidden_ulp'] for e in floors):.2f} ULP; unary max "
              f"{max(e['unary_ulp'] for e in floors):.2f} ULP; gate max "
              f"{max(e['gate_ulp'] for e in floors):.2f} ULP; top-K id set equal in "
              f"{sum(1 for e in floors if e['ids_equal'])}/{len(floors)} steps")
        print(f"the oracle's OWN drafts change with accumulation width in "
              f"{sum(1 for e in floors if not e['self_parity'])}/{len(floors)} steps - that "
              f"fraction is this contract's irreducible flip rate")
        print(f"module drafts == oracle fp32 : {sum(1 for e in floors if e['module_matches_f32'])}"
              f"/{len(floors)}   == oracle fp64 : "
              f"{sum(1 for e in floors if e['module_matches_f64'])}/{len(floors)}")
    if "--sensitivity" in flags:
        print()
        print("SENSITIVITY CONTROL - one BF16 ULP on ONE tap element, same fp32 forward")
        print(f"{'pos':>6} {'selftest':>9} {'hiddend':>8} {'hidULP':>7} {'ids=':>5} "
              f"{'unULP':>7} {'gtULP':>7} {'drafts=':>8}")
        probes = []
        for row in rows:
            step = read_step(directory, row["position"])
            entry = sensitivity(row["position"], *step)
            probes.append(entry)
            print(f"{entry['position']:>6} {str(entry['self_test']):>9} "
                  f"{entry['hidden_bad']:>8} {entry['hidden_ulp']:>7.2f} "
                  f"{str(entry['ids_equal']):>5} {entry['unary_ulp']:>7.2f} "
                  f"{entry['gate_ulp']:>7.2f} {str(entry['drafts_equal']):>8}")
        print(f"self-test (forward_at(fp32) == rail.forward) : "
              f"{sum(1 for e in probes if e['self_test'])}/{len(probes)} bit-equal")
        print(f"ONE tap ULP moves {sum(e['hidden_bad'] for e in probes)}/"
              f"{sum(e['hidden_words'] for e in probes)} final-hidden BF16 words (max "
              f"{max(e['hidden_ulp'] for e in probes):.2f} ULP), changes the top-K id set in "
              f"{sum(1 for e in probes if not e['ids_equal'])}/{len(probes)} steps and the "
              f"drafts in {sum(1 for e in probes if not e['drafts_equal'])}/{len(probes)}")
    truths = [row["truth"] for row in rows if row["truth"] is not None]
    if truths:
        print()
        print("FLOAT64 ARBITER (the contract computed exactly from the ORACLE's hidden)")
        print(f"  worst top-K boundary margin in f64 : {min(t['topk_boundary'] for t in truths):.6f}"
              " (0 = a genuine tie at the K-th place; the re-rank pool is 64 wide)")
        print(f"  oracle top-K id set == truth       : "
              f"{sum(1 for t in truths if t['ids_equal_oracle'])}/{len(truths)}")
        print(f"  unary BF16-exact vs truth          : module "
              f"{sum(t['unary_module_exact'] for t in truths)}/{sum(t['unary_words'] for t in truths)}"
              f", oracle {sum(t['unary_oracle_exact'] for t in truths)}"
              f"/{sum(t['unary_words'] for t in truths)}")
        print(f"  gate  BF16-exact vs truth          : module "
              f"{sum(t['gate_module_exact'] for t in truths)}/{sum(t['gate_words'] for t in truths)}"
              f", oracle {sum(t['gate_oracle_exact'] for t in truths)}"
              f"/{sum(t['gate_words'] for t in truths)}")
        print(f"  drafts == the f64 truth            : module "
              f"{sum(1 for t in truths if t['module_matches_truth'])}/{len(truths)}"
              f", oracle {sum(1 for t in truths if t['oracle_matches_truth'])}/{len(truths)}")
        print("  the module columns are computed against a truth derived from the ORACLE's "
              "hidden, so they only judge the module once the hidden dump shows the two "
              "hiddens agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
