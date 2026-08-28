#!/usr/bin/env python3
"""Execute the real K3 slice loop on a CPU and check the stream it carries.

The per-kernel and per-layer harnesses passed while the LOOP held six defects:
retrieval folded into a residual, a dead MLP-side retrieval, a partial missing
the attention output, a partial that never reset at a block boundary, no output
retrieval, and one state slot shared by every KDA layer. All of them live
between layers, so this is the harness that executes between layers.

The recorder GEMM writes 0.125 * call index, constant across rows, so the whole
partial/bank trajectory is closed-form: the checker replays the reference's
schedule -

    P = embedding
    boundary: bank[l/12] = P, then P RESTARTS from the attention output
    otherwise P += a;  always P += m after the MLP

- from the emitted per-layer gemm log, and compares the emitted partial, the
banked entries, and the MLP-side retrieval (visible as the layer's first norm
output) against it. A loop that carries the old sum across a boundary, adds
where it should copy, or ignores the retrieval it computed disagrees here.
"""
import math
import re
import subprocess
import sys
from pathlib import Path

from host_cuda_compiler import host_cuda_cxx

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "tests" / "host_cuda" / "k3_slice_host.cu"
BINARY = Path("/tmp") / "lm_k3_slice_host"
ROWS, HIDDEN, BLOCK, LAYERS = 2, 7168, 12, 14
EPS = 1e-5


def rms_norm(vec):
    inverse = 1.0 / math.sqrt(sum(x * x for x in vec) / len(vec) + EPS)
    return [x * inverse for x in vec]


def attn_res(candidates, weight):
    scores = []
    for candidate in candidates:
        normed = rms_norm(candidate)
        scores.append(sum(x * w for x, w in zip(normed, weight)))
    top = max(scores)
    mix = [math.exp(s - top) for s in scores]
    total = sum(mix)
    out = [0.0] * len(candidates[0])
    for m, candidate in zip(mix, candidates):
        for e, x in enumerate(candidate):
            out[e] += (m / total) * x
    return out


def compare(name, want, got, failures, tolerance):
    worst = max(abs(w - g) / max(abs(w), 1.0) for w, g in zip(want, got))
    if worst > tolerance:
        print(f"  FAIL {name}: relative error {worst:.3e} over {tolerance:.0e}")
        return failures + 1, worst
    return failures, worst


def main():
    build = subprocess.run(
        [host_cuda_cxx(), "-std=c++17", "-O1", f"-I{ROOT}/tests/host_cuda/shim", f"-I{ROOT}",
         f"-I{ROOT}/tests/host_cuda", "-x", "c++", str(SOURCE), "-o", str(BINARY)],
        capture_output=True, text=True)
    if build.returncode != 0:
        errors = [l for l in build.stderr.split("\n") if "error" in l]
        print("FAIL host build:", (errors or [build.stderr])[0][:240])
        return 1
    run = subprocess.run([str(BINARY)], capture_output=True, text=True)
    if run.returncode != 0 or "done" not in run.stdout:
        print(f"FAIL the slice faulted (returncode {run.returncode})")
        print(run.stdout[-400:])
        return 1

    series = {}
    per_layer = [dict(gemms=[]) for _ in range(LAYERS)]
    current, offsets, caches = -1, {}, {}
    for line in run.stdout.split("\n"):
        match = re.match(r"layer (\d+) state_offset (\d+) cache (\d+)", line)
        if match:
            current = int(match.group(1))
            offsets[current] = int(match.group(2))
            caches[current] = int(match.group(3))
            continue
        match = re.match(r"gemm (\d+) layer (\d+) dest (\w+) wgt (\w+)", line)
        if match:
            per_layer[int(match.group(2))]["gemms"].append(
                (int(match.group(1)), match.group(3), match.group(4)))
            continue
        match = re.match(r"(\w+) ([\d.eE+\-]+)$", line)
        if match:
            tag = match.group(1) if current < 0 or match.group(1) in (
                "embedding", "attnw", "mlpw", "bank0", "bank1") else \
                f"{match.group(1)}{current}"
            series.setdefault(tag, []).append(float(match.group(2)))
    failures = 0

    # F5: every KDA layer must sit on its own state slot, every MLA layer on
    # its own cache. The offsets are kda_index * rows * slot, so distinctness
    # plus monotonicity is the whole claim.
    kda_offsets = [offsets[l] for l in range(LAYERS) if l % 4 != 3]
    if len(set(kda_offsets)) != len(kda_offsets):
        print("  FAIL KDA layers share a state slot")
        failures += 1
    for l in range(LAYERS):
        if l % 4 == 3 and caches[l] != l // 4:
            print(f"  FAIL MLA layer {l} bound cache {caches[l]}, wanted {l // 4}")
            failures += 1

    # Rebuild the stream from the gemm log. a is the projection that wrote
    # attention_out; m is the MLP's writes to hidden (plus shared_out on MoE).
    embedding = series["embedding"]
    partial = list(embedding)
    bank = [list(embedding)]
    for l in range(LAYERS):
        gemms = per_layer[l]["gemms"]
        # PACK V2 CONSUMPTION, PER KDA LAYER: the projections that read the
        # normed input are the fused qkv|beta wide GEMM plus the gate
        # reconciliation pair (commit 55cd2f9): the standalone 128-wide
        # kda_decay_down landing in latent and the checkpoint's full-rank
        # gate. Any other destination here is the six-launch block come
        # back.
        if l % 4 != 3:
            wide = [(d, w) for _, d, w in gemms[:3]]
            if wide != [("fused_qkvb", "qkvb"),
                        ("latent", "decay_gate"),
                        ("gate", "decay_gate")]:
                print(f"  FAIL layer {l}: normed-input projection GEMMs are "
                      f"{wide}, not qkv|beta fused, decay_down, and the "
                      f"full-rank gate once each")
                failures += 1
            stale = [d for _, d, _ in gemms
                     if d in ("query", "key", "value", "beta")]
            if stale:
                print(f"  FAIL layer {l}: projections write {stale} directly; "
                      f"the sections come from the split, not a GEMM")
                failures += 1
        # a is the projection that wrote the attention output: the KDA out
        # projection (first hidden write) on KDA layers, the MLA out GEMM on
        # the every-fourth MLA layer.
        if l % 4 != 3:
            a = 0.125 * next(i for i, d, _ in gemms if d == "hidden")
        else:
            a = 0.125 * next(i for i, d, _ in gemms if d == "attention_out")
        hidden_writes = [i for i, d, _ in gemms if d == "hidden"]
        shared_writes = [i for i, d, _ in gemms if d == "shared_out"]
        m = 0.125 * (hidden_writes[-1] + sum(shared_writes))
        boundary = (l % BLOCK) == 0
        if boundary and l > 0:
            bank.append(list(partial))
        mlp_candidates = [list(c) for c in bank]
        if boundary:
            partial = [a] * (ROWS * HIDDEN)
        else:
            partial = [p + a for p in partial]
        mlp_candidates.append(list(partial))
        partial = [p + m for p in partial]
        failures, _ = compare(f"partial layer {l}", partial,
                              series[f"partial{l}"], failures, 5e-2)
        failures, _ = compare(f"stream layer {l}", partial,
                              series[f"stream{l}"], failures, 5e-2)
        # The MLP-side retrieval must be CONSUMED: the layer's first norm output
        # is RMS(retrieval), so a loop that computed the retrieval and normed
        # attention_out instead - which this loop did - disagrees here.
        want = []
        for row in range(ROWS):
            row_candidates = [c[row * HIDDEN:(row + 1) * HIDDEN]
                              for c in mlp_candidates]
            want.extend(rms_norm(attn_res(row_candidates, series["mlpw"])))
        failures, _ = compare(f"mlp retrieval layer {l}", want,
                              series[f"normed{l}"], failures, 5e-2)

    aux = re.search(r"aux_mismatch (\d+)", run.stdout)
    if aux is None or int(aux.group(1)) != 0:
        print("  FAIL the aux capture after layer 7 differs from the stream")
        failures += 1
    untouched = re.search(r"verify_untouched (\d+)", run.stdout)
    fold = re.search(r"fold_mismatch (\d+)", run.stdout)
    if untouched is None or int(untouched.group(1)) != 0:
        print("  FAIL the verify pass advanced state it must not touch")
        failures += 1
    if fold is None or int(fold.group(1)) != 0:
        print(f"  FAIL fold differs from the committed truth "
              f"({fold.group(1) if fold else 'missing'} bytes)")
        failures += 1
    failures, _ = compare("bank slot 0", embedding, series["bank0"], failures, 5e-2)
    failures, _ = compare("bank slot 1", bank[1], series["bank1"], failures, 5e-2)

    if failures:
        print(f"\n{failures} failures")
        return 1
    print(f"\nrows {ROWS}  layers {LAYERS}  boundaries at 0 and 12")
    print("the slice carries the reference's stream: restart at the boundary, "
          "bank the prefix, consume both retrievals")
    return 0


if __name__ == "__main__":
    sys.exit(main())
