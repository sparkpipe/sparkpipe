#!/usr/bin/env python3
"""Execute the real GLM 5.2 layer on a CPU and check the stream it carries.

The per-kernel tests and the launch-site contracts passed while the layer
itself had no harness at all. This one runs Glm52LayerAttention,
Glm52LayerMoe, Glm52LayerDenseMlp in both gate/up forms, and Glm52Head - the
shipping code, unmodified - and
replays the whole stream closed-form:

    the recorder GEMM writes 0.125 * call index, so every projection output
    is a known constant; the checker recomputes the norm chain, both rope
    rotations, the cache slot the kv projections must write DIRECTLY (latent
    at [0,512), rotated rope at [512,576) - the join the layer used to need
    is gone, and a reintroduced join or a swapped offset disagrees here), the
    attention softmax over four positions, the counting sort and its tile
    tables, the indirect routed-row read, the finalize, and the head's norm over the folded
    hidden + residual stream.
"""
import math
import re
import struct
import subprocess
import sys
from pathlib import Path

from host_cuda_compiler import host_cuda_cxx

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "tests" / "host_cuda" / "glm52_layer_host.cu"
BINARY = Path("/tmp") / "lm_glm52_layer_host"
CODECS = (
    "SPARK_WEIGHT_CODEC_INT6",
    "SPARK_WEIGHT_CODEC_INT7",
    "SPARK_WEIGHT_CODEC_INT8",
    "SPARK_WEIGHT_CODEC_FP8_E4M3",
    "SPARK_WEIGHT_CODEC_NVFP4_E2M1",
    "SPARK_WEIGHT_CODEC_MXFP4_E2M1",
)

ROWS, HIDDEN, HEADS = 2, 6144, 64
LATENT, ROPE, LATENT_ROW = 512, 64, 576
EXPERTS, TOP_K, INTER, GATE_UP = 256, 8, 2048, 4096
DENSE_INTER, PACKED, HEAD_VOCAB = 12288, 16, 128
CONTEXT, POSITION = 4, 3
SEED = 13579

INCLUDES = [
    f"-I{ROOT}/tests/host_cuda/shim",
    f"-I{ROOT}",
    f"-I{ROOT}/tests/host_cuda",
    f"-I{ROOT}/include",
    f"-I{ROOT}/model-families/glm52/include",
    f"-I{ROOT}/modules/glm52_resident_decode_stage/include",
    f"-I{ROOT}/deployment/include",
]


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bf16(value):
    bits = struct.unpack("<I", struct.pack("<f", f32(value)))[0]
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFFFFFF
    return struct.unpack("<f", struct.pack("<I", (rounded >> 16) << 16))[0]


class Random:
    def __init__(self, seed=SEED):
        self.seed = seed

    def next(self):
        self.seed = (self.seed * 1664525 + 1013904223) & 0xFFFFFFFF
        return ((self.seed >> 8) & 0xFFFF) / 32768.0 - 1.0

    def bf16s(self, count):
        return [bf16(self.next()) for _ in range(count)]

    def norm_weights(self, count):
        return [bf16(1.0 + 0.2 * self.next()) for _ in range(count)]


def rope_rotate(values, position, theta):
    half = len(values) // 2
    out = list(values)
    for index in range(half):
        angle = position * theta ** (-2.0 * index / len(values))
        c, s = math.cos(angle), math.sin(angle)
        low, high = index * 2, index * 2 + 1
        a, b = values[low], values[high]
        out[low] = bf16(a * c - b * s)
        out[high] = bf16(a * s + b * c)
    return out


def rms_norm(row, weight, eps):
    scale = 1.0 / math.sqrt(sum(x * x for x in row) / len(row) + eps)
    return [bf16(x * scale * w) for x, w in zip(row, weight)]


def compare(name, want, got, failures, tolerance):
    worst = max(abs(w - g) / max(abs(w), 1.0) for w, g in zip(want, got))
    if worst > tolerance:
        print(f"  FAIL {name}: relative error {worst:.3e} over {tolerance:.0e}")
        return failures + 1
    return failures


def exact(name, want, got, failures):
    if list(want) != list(got):
        print(f"  FAIL {name}: {list(want)[:8]}... != {list(got)[:8]}...")
        return failures + 1
    return failures


def main():
    outputs = []
    for codec in CODECS:
        build = subprocess.run(
            [host_cuda_cxx(), "-std=c++17", "-O1",
             f"-DEXPERT_CODEC={codec}",
             # the firmware header (the split-partials sizing) names the codec
             # in its variant module id; the string is unused identity here.
             '-DGLM52_EXPERT_CODEC_NAME="int8"', *INCLUDES,
             "-x", "c++", str(SOURCE), "-o", str(BINARY)],
            capture_output=True, text=True)
        if build.returncode != 0:
            errors = [line for line in build.stderr.split("\n")
                      if "error" in line]
            print(f"FAIL {codec} host build:",
                  (errors or [build.stderr])[0][:240])
            return 1
        run = subprocess.run([str(BINARY)], capture_output=True, text=True)
        if run.returncode != 0 or "done" not in run.stdout:
            print(f"FAIL {codec} layer faulted (returncode {run.returncode})")
            print(run.stdout[-400:])
            return 1
        outputs.append(run.stdout)
    if any(output != outputs[0] for output in outputs[1:]):
        print("FAIL expert codec specializations changed the model execution contract")
        return 1
    output = outputs[0]

    series = {}
    gemms = []
    scalars = {}
    poisons = []
    for line in output.split("\n"):
        match = re.match(r"gemm (\w+) (\d+) K (\d+) N (\d+) rows (\d+) "
                         r"grouped (\d+) indirect (\d+) act (\w+) w (\w+) dst (\w+)", line)
        if match:
            gemms.append((match.group(1), int(match.group(2)),
                          int(match.group(3)), int(match.group(4)),
                          int(match.group(5)), int(match.group(6)),
                          int(match.group(7)), match.group(8),
                          match.group(9), match.group(10)))
            continue
        match = re.match(r"status (\w+) (\S+)$", line)
        if match:
            scalars[f"status {match.group(1)}"] = match.group(2)
            continue
        match = re.match(r"poison (\d+)$", line)
        if match:
            poisons.append(int(match.group(1)))
            continue
        match = re.match(r"(qkscale|theta|routedscale|eps) (\S+)$", line)
        if match:
            scalars[match.group(1)] = float(match.group(2))
            continue
        match = re.match(r"(\w+) ([\d.eE+\-]+)$", line)
        if match:
            series.setdefault(match.group(1), []).append(float(match.group(2)))

    failures = 0
    for phase in ("attention", "moe", "densefused", "densetwo", "head"):
        if scalars.get(f"status {phase}") != "0":
            print(f"  FAIL {phase} returned {scalars.get(f'status {phase}')}")
            failures += 1
    failures = exact("gate/up poison", [0, 0], poisons, failures)

    # The gemm log IS the launch sequence: shapes, grouping, routing of every
    # weight and destination.
    want_gemms = [
        ("attention", 1, HIDDEN, 2048, ROWS, 0, 0,
         "normed", "w_q_a", "q_compressed"),
        ("attention", 2, 2048, HEADS * (192 + ROPE), ROWS, 0, 0,
         "q_compressed", "w_q_b", "q_b"),
        ("attention", 3, HIDDEN, LATENT_ROW, ROWS, 0, 0,
         "normed", "w_kv_a", "kv_slot"),
        ("attention", 4, HEADS * 256, HIDDEN, ROWS, 0, 0,
         "attention_value", "w_o", "attention_out"),
        ("moe", 5, HIDDEN, EXPERTS, ROWS, 0, 0,
         "normed", "w_router", "router_logits"),
        ("moe", 6, HIDDEN, GATE_UP, PACKED, 1, 1,
         "normed", "w_e1", "gate_up"),
        ("moe", 7, INTER, HIDDEN, PACKED, 1, 0,
         "intermediate", "w_e2", "expert_out"),
        ("moe", 8, HIDDEN, GATE_UP, ROWS, 0, 0,
         "normed", "w_shared_gate_up", "gate_up"),
        ("moe", 9, INTER, HIDDEN, ROWS, 0, 0,
         "intermediate", "w_shared_down", "shared_out"),
        ("densefused", 10, HIDDEN, 2 * DENSE_INTER, ROWS, 0, 0,
         "normed", "w_q_a", "gate_up"),
        ("densefused", 11, DENSE_INTER, HIDDEN, ROWS, 0, 0,
         "intermediate", "w_down", "hidden"),
        ("densetwo", 12, HIDDEN, DENSE_INTER, ROWS, 0, 0,
         "normed", "w_q_a", "gate_up"),
        ("densetwo", 13, HIDDEN, DENSE_INTER, ROWS, 0, 0,
         "normed", "w_q_b", "gate_up"),
        ("densetwo", 14, DENSE_INTER, HIDDEN, ROWS, 0, 0,
         "intermediate", "w_down", "hidden"),
    ]
    failures = exact("gemm sequence", want_gemms, gemms, failures)

    # Replay the inputs the harness generated.
    rng = Random()
    hidden = rng.bf16s(ROWS * HIDDEN)
    residual = rng.bf16s(ROWS * HIDDEN)
    attn_w = rng.norm_weights(HIDDEN)
    mlp_w = rng.norm_weights(HIDDEN)
    head_w_norm = rng.norm_weights(HIDDEN)
    head_weight = rng.bf16s(HEAD_VOCAB * HIDDEN)
    prefill = [rng.bf16s(LATENT_ROW) for _ in range(2 * POSITION)]

    qk_scale, theta = scalars["qkscale"], scalars["theta"]
    routed_scale, eps = scalars["routedscale"], scalars["eps"]
    value = [0.0] + [0.125 * index for index in range(1, 15)]

    def row_of(values, row, width):
        return values[row * width:(row + 1) * width]

    # Attention: norm, then the four projections land where the log says.
    res1, normed1 = [], []
    for row in range(ROWS):
        stream = [h + r for h, r in zip(row_of(hidden, row, HIDDEN),
                                        row_of(residual, row, HIDDEN))]
        res1.append([bf16(x) for x in stream])
        normed1.append(rms_norm(stream, attn_w, eps))
    failures = compare("normed1", [x for r in normed1 for x in r],
                       series["normed1"], failures, 2e-2)

    kvslot = []
    for row in range(ROWS):
        latent = rms_norm([bf16(value[3])] * LATENT,
                          [bf16(1.0)] * LATENT, eps)
        kvslot.append(latent +
                      rope_rotate([bf16(value[3])] * ROPE, POSITION, theta))
    failures = compare("kv slot layout", [x for r in kvslot for x in r],
                       series["kvslot"], failures, 2e-2)
    # The store must have written exactly that row into the cache: this is the
    # check the retired join used to make necessary, and a projection writing
    # the wrong offset fails here, not on hardware.
    failures = compare("stored slot", [x for r in kvslot for x in r],
                       series["slot"], failures, 1e-6)

    # The layer's own attention output is recorder-erased on the host (the
    # shim floors the decode kernel at 64 threads, where one thread covers a
    # sixty-fourth of the query); the kernel's math is checked at a small
    # geometry where one thread covers everything.
    small_slots = [[bf16(0.1 * (p + 1) + 0.01 * e) for e in range(16)]
                   for p in range(3)]
    want_small = []
    for head in range(2):
        query = ([bf16(0.3 - 0.02 * e) for e in range(8)] +
                 [bf16(-0.2 + 0.03 * e + 0.05 * head) for e in range(8)])
        scores = [0.5 * sum(q * s for q, s in zip(query, slot))
                  for slot in small_slots]
        top = max(scores)
        weights = [math.exp(s - top) for s in scores]
        total = sum(weights)
        want_small.extend(
            sum(w * slot[e] for w, slot in zip(weights, small_slots)) / total
            for e in range(8))
    failures = compare("latent attention math", want_small,
                       series["smallattn"], failures, 5e-2)
    # R3 flash-decode: the split path over the same three positions (the
    # launcher engages 16 partitions, 13 of them empty tails) is the same
    # softmax to rounding, and the bit-level receipts must both hold.
    failures = exact("split receipts", [1, 1], series["splitreceipt"],
                     failures)
    failures = compare("latent attention split math", want_small,
                       series["smallattnsplit"], failures, 5e-2)

    # MoE: the second norm folds the attention output into the stream.
    res2, normed2 = [], []
    for row in range(ROWS):
        stream = [value[4] + r for r in res1[row]]
        res2.append([bf16(x) for x in stream])
        normed2.append(rms_norm(stream, mlp_w, eps))
    failures = compare("normed2", [x for r in normed2 for x in r],
                       series["normed2"], failures, 2e-2)

    # Top-k over constant logits: any eight distinct experts, weights
    # renormalised sigmoids times the routed scale.
    route_expert = [int(x) for x in series["routeexpert"]]
    for row in range(ROWS):
        chosen = route_expert[row * TOP_K:(row + 1) * TOP_K]
        if len(set(chosen)) != TOP_K or max(chosen) >= EXPERTS:
            print(f"  FAIL route experts row {row}: {chosen}")
            failures += 1
    failures = compare("route weights", [routed_scale / TOP_K] * PACKED,
                       series["routeweight"], failures, 1e-3)

    # The counting sort, replayed exactly: sequential atomic order on the host.
    counts = [0] * EXPERTS
    for expert in route_expert:
        counts[expert] += 1
    offsets, total = [], 0
    for expert in range(EXPERTS):
        offsets.append(total)
        total += counts[expert]
    offsets.append(total)
    failures = exact("group offsets", offsets,
                     [int(x) for x in series["groupoffset"]], failures)
    cursors = list(offsets[:-1])
    packed_row, source = [0] * PACKED, [0] * PACKED
    for index, expert in enumerate(route_expert):
        packed_row[index] = cursors[expert]
        source[cursors[expert]] = index // TOP_K
        cursors[expert] += 1
    failures = exact("packed rows", packed_row,
                     [int(x) for x in series["routepacked"]], failures)
    failures = exact("source tokens", source,
                     [int(x) for x in series["routesource"]], failures)
    tile_m = 16
    up_prefix, down_prefix, up, down = [], [], 0, 0
    for expert in range(EXPERTS):
        row_tiles = (counts[expert] + tile_m - 1) // tile_m
        up_prefix.append(up)
        down_prefix.append(down)
        up += row_tiles * (GATE_UP // 128)
        down += row_tiles * (HIDDEN // 128)
    up_prefix.append(up)
    down_prefix.append(down)
    failures = exact("w1 tile prefix", up_prefix,
                     [int(x) for x in series["tileup"]], failures)
    failures = exact("w2 tile prefix", down_prefix,
                     [int(x) for x in series["tiledown"]], failures)

    # Routed W2 consumes all packed SiLU rows, then the shared expert reuses only
    # the first ROWS rows. The untouched tail lets the fixture check both paths.
    routed_gate = value[6]
    routed_silu = bf16((routed_gate / (1.0 + math.exp(-routed_gate))) *
                       routed_gate)
    failures = compare("routed expert silu",
                       [routed_silu] * len(series["interrouted"]),
                       series["interrouted"], failures, 2e-2)
    shared_gate = value[8]
    shared_silu = bf16((shared_gate / (1.0 + math.exp(-shared_gate))) *
                       shared_gate)
    failures = compare("shared expert silu",
                       [shared_silu] * len(series["intershared"]),
                       series["intershared"], failures, 2e-2)
    routed = bf16(sum(routed_scale / TOP_K * value[7]
                      for _ in range(TOP_K)))
    hidden2 = [bf16(routed + value[9])] * (ROWS * HIDDEN)
    failures = compare("finalized hidden", hidden2, series["hidden2"],
                       failures, 1e-3)

    # Dense MLP twice: the fused form's single gemm wrote both halves (the
    # poison fill proves no half was skipped), the two-launch form wrote gate
    # then up. Each phase's norm folds the attention output into the stream
    # again, so the second phase's residual sits on top of the first's.
    res3 = [[bf16(value[4] + r) for r in res2[row]] for row in range(ROWS)]
    res3 = [[bf16(value[4] + r) for r in res3[row]] for row in range(ROWS)]
    half = DENSE_INTER
    for index, sample_value in enumerate(series["gateup"]):
        element = (index % ((half * 2 + 996) // 997)) * 997
        want = value[10]
        if abs(sample_value - want) > 1e-3:
            print(f"  FAIL fused gate/up sample {index}: {sample_value} != {want}")
            failures += 1
            break
    for index, sample_value in enumerate(series["gateup2"]):
        element = (index % ((half * 2 + 996) // 997)) * 997
        want = value[12] if element < half else value[13]
        if abs(sample_value - want) > 1e-3:
            print(f"  FAIL two-launch gate/up at element {element}: "
                  f"{sample_value} != {want}")
            failures += 1
            break

    # The head: final norm over hidden + residual - the fold the layer used to
    # skip - then a real argmax against the replayed head weight.
    for row in range(ROWS):
        stream = [value[14] + r for r in res3[row]]
        normed3 = rms_norm(stream, head_w_norm, eps)
        scores = [sum(n * w for n, w in zip(normed3,
                      head_weight[t * HIDDEN:(t + 1) * HIDDEN]))
                  for t in range(HEAD_VOCAB)]
        want_token = max(range(HEAD_VOCAB), key=lambda t: scores[t])
        got_token = int(series["token"][row])
        if got_token != want_token:
            print(f"  FAIL head token row {row}: {got_token} != {want_token}")
            failures += 1
    failures = compare("head normed", [x for r in range(ROWS)
                       for x in rms_norm([value[14] + rr for rr in res3[r]],
                                          head_w_norm, eps)],
                       series["normed3"], failures, 2e-2)

    if failures:
        print(f"\n{failures} failures")
        return 1
    print(f"\nrows {ROWS}  context {CONTEXT}  gemms {len(gemms)}  "
          f"codecs {len(CODECS)}")
    print("the layer carries the reference stream: slot written directly by "
          "the kv projections, routed W1 is indirect, finalize is consistent, the head "
          "norms hidden + residual")
    return 0


if __name__ == "__main__":
    sys.exit(main())
