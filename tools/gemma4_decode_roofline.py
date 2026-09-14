#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path

DENSE_HEADER = Path(__file__).resolve().parent.parent / (
    "model-families/gemma4/include/sparkpipe/llm_defines.h")
MOE_HEADER = Path(__file__).resolve().parent.parent / (
    "model-families/gemma4/include/sparkpipe/llm_defines.h")

SPEC_BW_GB_S = 273.0
MEASURED_BW_GB_S = 242.1
MEASURED_NVME_GB_S = 4.686
PROBE_SOURCE = "spark0 240.3 / spark1 243.9 GB/s, 1.5 GiB stream read, 2026-09-13"
CLASSIFICATION = "analytical estimate (measured bandwidth input)"


def defines(text, prefix):
    found = {}
    pattern = re.compile(r"^#define\s+(%s[A-Z0-9_]+)\s+([0-9]+)u?\s*$" % prefix, re.M)
    for name, value in pattern.findall(text):
        found[name] = int(value)
    return found


def require(d, name):
    if name not in d:
        sys.exit("missing " + name)
    return d[name]


def build_table(d, context_tokens, tp_degree, moe):
    hidden = require(d, "SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION")
    layers = require(d, "SPARK_GEMMA4_MODEL_LAYER_COUNT")
    vocab = d.get("SPARK_GEMMA4_MODEL_OUTPUT_VOCAB_COUNT",
                  require(d, "SPARK_GEMMA4_MODEL_VOCAB_COUNT"))
    sliding_q_heads = require(d, "SPARK_GEMMA4_MODEL_SLIDING_QUERY_HEAD_COUNT")
    sliding_kv_heads = require(d, "SPARK_GEMMA4_MODEL_SLIDING_KV_HEAD_COUNT")
    sliding_head_dim = require(d, "SPARK_GEMMA4_MODEL_SLIDING_HEAD_DIMENSION")
    full_q_heads = d.get("SPARK_GEMMA4_MODEL_FULL_QUERY_HEAD_COUNT",
                         require(d, "SPARK_GEMMA4_MODEL_SLIDING_QUERY_HEAD_COUNT"))
    full_kv_heads = require(d, "SPARK_GEMMA4_MODEL_FULL_KV_HEAD_COUNT")
    full_head_dim = require(d, "SPARK_GEMMA4_MODEL_FULL_HEAD_DIMENSION")
    window = require(d, "SPARK_GEMMA4_MODEL_SLIDING_WINDOW_TOKENS")
    period = require(d, "SPARK_GEMMA4_MODEL_FULL_LAYER_PERIOD")
    bf16 = require(d, "SPARK_GEMMA4_MODEL_BF16_ELEMENT_BYTES")
    dense_intermediate = require(d, "SPARK_GEMMA4_MODEL_DENSE_INTERMEDIATE_DIMENSION")
    prefix = "SPARK_GEMMA4_MODEL_"
    routed_experts = d.get(prefix + "ROUTED_EXPERT_COUNT", 0)
    experts_per_token = d.get(prefix + "EXPERTS_PER_TOKEN", 0)
    expert_intermediate = d.get(prefix + "EXPERT_INTERMEDIATE_DIMENSION", 0)

    full_layers = layers // period
    sliding_layers = layers - full_layers
    local = lambda heads: (heads + tp_degree - 1) // tp_degree
    rows = []
    sliding_weight = 0
    sliding_weight += local(sliding_q_heads) * sliding_head_dim * hidden * bf16
    sliding_weight += local(2 * sliding_kv_heads * sliding_head_dim) * hidden * bf16
    sliding_weight += local(sliding_q_heads) * sliding_head_dim * hidden * bf16
    sliding_weight += (sliding_q_heads + sliding_kv_heads) * sliding_head_dim * bf16
    sliding_weight += 4 * hidden * bf16
    rows.append(("sliding attention spine (swa layers)", sliding_layers, sliding_weight))
    full_weight = 0
    full_weight += local(full_q_heads) * full_head_dim * hidden * bf16
    full_weight += local(full_kv_heads) * full_head_dim * hidden * bf16
    full_weight += local(full_q_heads) * full_head_dim * hidden * bf16
    full_weight += (full_q_heads + full_kv_heads) * full_head_dim * bf16
    full_weight += 4 * hidden * bf16
    rows.append(("full attention spine (k equals v)", full_layers, full_weight))
    expert_bytes = 0
    if moe:
        expert_bytes = (2 * expert_intermediate * hidden + hidden * expert_intermediate) * bf16
        mlp_weight = 0
    else:
        mlp_weight = 0
        mlp_weight += local(2 * dense_intermediate) * hidden * bf16
        mlp_weight += local(dense_intermediate) * hidden * bf16
    rows.append(("mlp spine", layers, mlp_weight))
    head_weight = local(vocab) * hidden * bf16
    rows.append(("lm head", 1, head_weight))
    rows.append(("embedding gather", 1, hidden * bf16))

    state_rows = []
    sliding_slot_bytes = local(sliding_kv_heads) * 2 * sliding_head_dim * bf16
    full_slot_bytes = local(full_kv_heads) * 2 * full_head_dim * bf16
    sliding_kv = sliding_layers * min(context_tokens, window) * sliding_slot_bytes
    full_kv = full_layers * context_tokens * full_slot_bytes
    state_rows.append(("sliding kv read+write (window %d)" % window, 1, sliding_kv))
    state_rows.append(("full attention kv read+write", 1, full_kv))
    if moe:
        active_experts = experts_per_token * expert_bytes
        local_expert_bytes = (active_experts + tp_degree - 1) // tp_degree
        state_rows.append(("active expert weights (hot resident)", 1, local_expert_bytes))
    else:
        active_experts = 0
        local_expert_bytes = 0
    return rows, state_rows, local_expert_bytes, active_experts


def gigabytes(value):
    return value / 1e9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["31b-dense", "26b-a4b"], default="31b-dense")
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--tp", type=int, default=0)
    parser.add_argument("--bw-gb-s", type=float, default=MEASURED_BW_GB_S)
    parser.add_argument("--receipt-tok-s", type=float, default=0.0)
    args = parser.parse_args()
    moe = args.variant == "26b-a4b"
    tp_degree = args.tp if args.tp else (4 if not moe else 4)
    path = MOE_HEADER if moe else DENSE_HEADER
    prefix = "SPARK_GEMMA4_MOE_" if moe else "SPARK_GEMMA4_MODEL_"
    raw = defines(path.read_text(), prefix)
    d = {}
    for name, value in raw.items():
        d["SPARK_GEMMA4_MODEL_" + name[len(prefix):]] = value
    table, state_rows, local_expert_bytes, active_experts = build_table(
        d, args.context, tp_degree, moe)

    total = sum(count * value for _, count, value in table)
    total += sum(count * value for _, count, value in state_rows)
    print("gemma4 %s decode roofline, B1, TP%d, context %d tokens" % (
        args.variant, tp_degree, args.context))
    print("classification: %s" % CLASSIFICATION)
    print("bandwidth: measured %.1f GB/s of %.0f GB/s spec (%.1f%%) [%s]" % (
        args.bw_gb_s, SPEC_BW_GB_S, 100.0 * args.bw_gb_s / SPEC_BW_GB_S, PROBE_SOURCE))
    print()
    print("%-52s %8s %14s" % ("block", "count", "bytes/token"))
    for name, count, value in table:
        print("%-52s %8d %14s" % (name, count, "%.3f MB" % (gigabytes(count * value) * 1e3)))
    for name, count, value in state_rows:
        print("%-52s %8s %14s" % (name, "-", "%.3f MB" % (gigabytes(count * value) * 1e3)))
    print("%-52s %8s %14s" % ("TOTAL per rank per token", "-", "%.3f GB" % gigabytes(total)))
    print()
    ceiling = args.bw_gb_s * 1e9 / total
    print("ceiling (hot resident weights): %.1f tok/s per rank at %.1f GB/s [ESTIMATE]" % (
        ceiling, args.bw_gb_s))
    if moe:
        cold = active_experts / (MEASURED_NVME_GB_S * 1e9)
        print("active expert set per token: %.1f MB fleet, %.3f MB rank slice" % (
            gigabytes(active_experts) * 1e3, gigabytes(local_expert_bytes) * 1e3))
        print("ceiling (cold experts, NVMe-streamed at %.2f GB/s): %.1f tok/s [ESTIMATE]" % (
            MEASURED_NVME_GB_S, min(ceiling, 1.0 / cold)))
    if args.receipt_tok_s > 0.0:
        print("receipted %.4f tok/s -> %.1f%% of measured roofline" % (
            args.receipt_tok_s, 100.0 * args.receipt_tok_s * total / (args.bw_gb_s * 1e9)))
    else:
        print("receipted tok/s: none (daemon-gated) -> percent of roofline N/A")
    return 0


if __name__ == "__main__":
    sys.exit(main())
