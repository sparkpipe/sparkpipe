#!/usr/bin/env python3
import argparse
import json
import math
import sys

HIDDEN = 6144
LAYERS = 78
DENSE_LAYERS = 3
ROUTED_LAYERS = LAYERS - DENSE_LAYERS
EXPERTS = 256
TOP_K = 8
SHARED_EXPERTS = 1
EXPERT_INTERMEDIATE = 2048
DENSE_INTERMEDIATE = 12288
HEADS = 64
QUERY_A = 2048
LATENT = 512
ROPE = 64
QK_NOPE = 192
VALUE = 256
VOCAB = 154880
DSA_SELECTED = 2048
DSA_INDEX_HEADS = 32
DSA_INDEX_DIM = 128
DSA_INDEX_LAYERS = 21
RANKS = 16
PREFILL_DISPATCH_TOKENS = 256
RSAG_MIN_ELEMENTS = 49152
REDUCE_ELEMENT_BYTES = 2

DEFAULTS = {
    "memory_gbps": 243.0,
    "link_gbps_each_way": 25.0,
    "hop_floor_us": 40.0,
    "peak_tflops": 250.0,
    "peak_precision": "fp8 dense tensor-core, spec-derived (1 PFLOP FP4 sparse / 2 dense / 2 per precision step); not measured",
    "spine_bytes_per_rank": 3.28e9,
    "expert_key_bytes_per_rank": 2.43e6,
    "kv_latent_bytes_per_token": LAYERS * (LATENT + ROPE) * 2,
    "kv_index_bytes_per_token": DSA_INDEX_LAYERS * DSA_INDEX_DIM * 2,
    "rounds_per_step": 1 + LAYERS * 3 + 1,
}

SOURCES = {
    "memory_gbps": "GB10 read measured 243 GB/s (tools/hardware/stream_read_probe.cu, lanes/ROOFLINE_REPORTING.md)",
    "link_gbps_each_way": "400 Gbps per node = 25 GB/s each way; send bytes counted against 25 GB/s",
    "hop_floor_us": "~40 us isolated round (lanes/ROOFLINE_REPORTING.md worked example); per-hop floor not yet measured",
    "spine_bytes_per_rank": "bf16 spine read per rank per step, 3.28 GB (glmfull-rounds window report)",
    "expert_key_bytes_per_rank": "fp8 expert slice per (layer, expert) per rank, 2.43 MB (3 x 6144 x 2048 fp8 / 16 + scales)",
}


def attention_linear_params():
    q_a = HIDDEN * QUERY_A
    q_b = QUERY_A * HEADS * (QK_NOPE + ROPE)
    kv_a = HIDDEN * (LATENT + ROPE)
    kv_b = LATENT * HEADS * (QK_NOPE + VALUE)
    o = HEADS * VALUE * HIDDEN
    return q_a + q_b + kv_a + kv_b + o


def index_linear_params():
    return QUERY_A * DSA_INDEX_HEADS * DSA_INDEX_DIM + HIDDEN * DSA_INDEX_DIM + HIDDEN * DSA_INDEX_HEADS


def expert_params():
    return 3 * HIDDEN * EXPERT_INTERMEDIATE


def active_params():
    attention = LAYERS * attention_linear_params() + DSA_INDEX_LAYERS * index_linear_params()
    dense = DENSE_LAYERS * 3 * HIDDEN * DENSE_INTERMEDIATE
    routed = ROUTED_LAYERS * ((TOP_K + SHARED_EXPERTS) * expert_params() + HIDDEN * EXPERTS)
    return attention + dense + routed + HIDDEN * VOCAB


def unique_expert_keys(rows):
    return ROUTED_LAYERS * EXPERTS * (1.0 - (1.0 - TOP_K / EXPERTS) ** rows)


def attention_flops_per_row(context):
    selected = min(context, DSA_SELECTED)
    mla = LAYERS * HEADS * selected * 2 * ((LATENT + ROPE) + LATENT)
    index = DSA_INDEX_LAYERS * DSA_INDEX_HEADS * context * 2 * DSA_INDEX_DIM
    return mla + index


def kv_bytes_per_row(context, p):
    return min(context, DSA_SELECTED) * p["kv_latent_bytes_per_token"] + context * p["kv_index_bytes_per_token"]


def collective_phases(rows):
    return 2 if HIDDEN * rows >= RSAG_MIN_ELEMENTS else 1


def egress_bytes_per_op(rows):
    elements = HIDDEN * rows
    if collective_phases(rows) == 2:
        return 2.0 * (RANKS - 1) / RANKS * elements * REDUCE_ELEMENT_BYTES
    return (RANKS - 1) * elements * REDUCE_ELEMENT_BYTES


def step_model(rows, context, p=None, expert_keys=None, causal=False):
    p = dict(DEFAULTS, **(p or {}))
    keys = unique_expert_keys(rows) if expert_keys is None else expert_keys
    kv_rows = 1 if causal else rows
    attended = max(1, context - rows // 2) if causal else context
    bytes_ = p["spine_bytes_per_rank"] + keys * p["expert_key_bytes_per_rank"] + kv_rows * kv_bytes_per_row(context, p)
    flops = rows * (2.0 * active_params() + attention_flops_per_row(attended)) / RANKS
    ops = p["rounds_per_step"]
    phases = collective_phases(rows)
    egress = ops * egress_bytes_per_op(rows)
    return {"rows": rows, "context": context, "expert_keys": keys, "bytes_per_rank": bytes_, "flops_per_rank": flops,
            "collective_ops": ops, "phases_per_op": phases, "rounds": ops * phases, "egress_bytes_per_rank": egress,
            "t_memory_ms": bytes_ / (p["memory_gbps"] * 1e9) * 1e3,
            "t_compute_ms": flops / (p["peak_tflops"] * 1e12) * 1e3,
            "t_transport_bw_ms": egress / (p["link_gbps_each_way"] * 1e9) * 1e3,
            "t_transport_latency_ms": ops * phases * p["hop_floor_us"] / 1e3}


def prefill_model(prompt_tokens, p=None):
    chunks = max(1, math.ceil(prompt_tokens / PREFILL_DISPATCH_TOKENS))
    total = {"t_memory_ms": 0.0, "t_compute_ms": 0.0, "t_transport_bw_ms": 0.0, "t_transport_latency_ms": 0.0,
             "bytes_per_rank": 0.0, "flops_per_rank": 0.0, "rounds": 0, "egress_bytes_per_rank": 0.0, "expert_keys": 0.0}
    done = 0
    for _ in range(chunks):
        rows = min(PREFILL_DISPATCH_TOKENS, prompt_tokens - done)
        done += rows
        m = step_model(rows, done, p, causal=True)
        for k in total:
            total[k] += m[k]
    total.update({"rows": prompt_tokens, "context": prompt_tokens, "chunks": chunks})
    return total


LABELS = (("t_memory_ms", "memory"), ("t_compute_ms", "compute"), ("t_transport_bw_ms", "transport bandwidth"),
          ("t_transport_latency_ms", "transport latency"))


def evaluate(model, measured_ms, tokens_per_step, p=None):
    p = dict(DEFAULTS, **(p or {}))
    pct = {name: 100.0 * model[key] / measured_ms for key, name in LABELS}
    binding = max(LABELS, key=lambda kv: model[kv[0]])[1]
    floor_max = max(model[key] for key, _ in LABELS)
    floor_serial = model["t_memory_ms"] + model["t_transport_latency_ms"] + model["t_transport_bw_ms"]
    return {"measured_ms": measured_ms, "tokens_per_step": tokens_per_step, "percent": pct, "binding": binding,
            "memory_ceiling_tok_s": tokens_per_step * 1e3 / model["t_memory_ms"],
            "ceiling_overlapped_tok_s": tokens_per_step * 1e3 / floor_max,
            "ceiling_serial_tok_s": tokens_per_step * 1e3 / floor_serial,
            "model": model, "peak_tflops": p["peak_tflops"], "peak_precision": p["peak_precision"],
            "hop_floor_us": p["hop_floor_us"]}


def line(ev, step_label):
    m, pct = ev["model"], ev["percent"]
    return (f"roofline @B={m['rows']} ({step_label}, {ev['measured_ms']:.2f} ms measured): "
            f"memory {pct['memory']:.0f}% (ceiling {ev['memory_ceiling_tok_s']:.1f} tok/s) | "
            f"compute {pct['compute']:.1f}% ({ev['peak_tflops']:.0f} TFLOP/s fp8 peak used) | "
            f"transport bandwidth {pct['transport bandwidth']:.1f}% | "
            f"transport latency {pct['transport latency']:.0f}% ({m['rounds']} rounds x {ev['hop_floor_us']:.0f} us floor) "
            f"- binding: {ev['binding']} "
            f"(inputs per rank: {m['bytes_per_rank'] / 1e9:.2f} GB read, {m['flops_per_rank'] / 1e9:.1f} GFLOP, "
            f"{m['egress_bytes_per_rank'] / 1e6:.1f} MB sent, context {m['context']}, expert keys {m['expert_keys']:.0f}; "
            f"4-roofline ceiling {ev['ceiling_overlapped_tok_s']:.1f} tok/s overlapped, {ev['ceiling_serial_tok_s']:.1f} serial)")


def decode_line(rows, step_ms, context, step_label="end-to-end fleet decode step", p=None):
    ev = evaluate(step_model(rows, context, p), step_ms, rows, p)
    return line(ev, step_label), ev


def prefill_ideal(prompt_tokens, p=None):
    m = step_model(prompt_tokens, prompt_tokens, p, causal=True)
    floor = max(m[key] for key, _ in LABELS)
    return {"single_dispatch_ceiling_tok_s": prompt_tokens * 1e3 / floor,
            "single_dispatch_binding": max(LABELS, key=lambda kv: m[kv[0]])[1]}


def prefill_line(prompt_tokens, ttft_ms, p=None):
    ev = evaluate(prefill_model(prompt_tokens, p), ttft_ms, prompt_tokens, p)
    ev.update(prefill_ideal(prompt_tokens, p))
    text = line(ev, f"end-to-end prefill of {prompt_tokens} tokens in {PREFILL_DISPATCH_TOKENS}-token dispatches")
    return (f"{text}; one-dispatch ideal {ev['single_dispatch_ceiling_tok_s']:.0f} tok/s bound by "
            f"{ev['single_dispatch_binding']}"), ev


def ceilings(rows_list, context, p=None):
    out = []
    for rows in rows_list:
        m = step_model(rows, context, p)
        floor_max = max(m[key] for key, _ in LABELS)
        floor_serial = m["t_memory_ms"] + m["t_transport_latency_ms"] + m["t_transport_bw_ms"]
        out.append({"rows": rows, "memory_ceiling_tok_s": rows * 1e3 / m["t_memory_ms"],
                    "ceiling_overlapped_tok_s": rows * 1e3 / floor_max, "ceiling_serial_tok_s": rows * 1e3 / floor_serial,
                    "binding": max(LABELS, key=lambda kv: m[kv[0]])[1], **m})
    return out


def main(argv):
    ap = argparse.ArgumentParser(description="GLM-5.3 Full TP16 four-roofline model")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("decode")
    d.add_argument("--rows", type=int, required=True)
    d.add_argument("--step-ms", type=float, required=True)
    d.add_argument("--context", type=int, default=512)
    f = sub.add_parser("prefill")
    f.add_argument("--prompt", type=int, required=True)
    f.add_argument("--ttft-ms", type=float, required=True)
    c = sub.add_parser("ceilings")
    c.add_argument("--rows", default="1,8,16,32,64,128,256")
    c.add_argument("--context", type=int, default=512)
    for s in (d, f, c):
        s.add_argument("--peak-tflops", type=float, default=DEFAULTS["peak_tflops"])
        s.add_argument("--hop-floor-us", type=float, default=DEFAULTS["hop_floor_us"])
        s.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    p = {"peak_tflops": a.peak_tflops, "hop_floor_us": a.hop_floor_us}
    if a.cmd == "decode":
        text, ev = decode_line(a.rows, a.step_ms, a.context, p=p)
    elif a.cmd == "prefill":
        text, ev = prefill_line(a.prompt, a.ttft_ms, p=p)
    else:
        ev = ceilings([int(x) for x in a.rows.split(",")], a.context, p)
        text = "\n".join(f"B={e['rows']}: memory ceiling {e['memory_ceiling_tok_s']:.1f} tok/s, 4-roofline ceiling "
                         f"{e['ceiling_overlapped_tok_s']:.1f} overlapped / {e['ceiling_serial_tok_s']:.1f} serial, binding {e['binding']} "
                         f"(read {e['bytes_per_rank'] / 1e9:.2f} GB, {e['rounds']} rounds, {e['egress_bytes_per_rank'] / 1e6:.1f} MB sent, "
                         f"{e['flops_per_rank'] / 1e9:.1f} GFLOP per rank)" for e in ev)
    print(json.dumps(ev, indent=1) if a.json else text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
