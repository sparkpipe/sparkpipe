#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path

HEADER = Path(__file__).resolve().parent.parent / (
    "model-families/muse_glimmer/include/sparkpipe/spark_muse_glimmer_model.h")

SPEC_BW_GB_S = 273.0
MEASURED_BW_GB_S = 242.1
PROBE_SOURCE = "spark0 240.3 / spark1 243.9 GB/s, 1.5 GiB stream read, 2026-09-13"
CLASSIFICATION = "analytical estimate (measured bandwidth input)"
DENSE = "muse_glimmer is dense: active-expert bytes/token = 0 (no routed experts)"


def defines(text):
    found = {}
    pattern = re.compile(r"^#define\s+(SPARK_MUSE_GLIMMER_MODEL_[A-Z0-9_]+)\s+([0-9]+)u?\s*$", re.M)
    for name, value in pattern.findall(text):
        found[name] = int(value)
    return found


def require(d, name):
    if name not in d:
        sys.exit("missing " + name + " in " + HEADER.name)
    return d[name]


def alias(d, name, source):
    return d.get(name, require(d, source))


def build_table(d, context_tokens, tp_degree):
    hidden = require(d, "SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION")
    layers = require(d, "SPARK_MUSE_GLIMMER_MODEL_LAYER_COUNT")
    vocab = d.get("SPARK_MUSE_GLIMMER_MODEL_OUTPUT_VOCAB_COUNT",
                  require(d, "SPARK_MUSE_GLIMMER_MODEL_VOCAB_COUNT"))
    q_heads = alias(d, "SPARK_MUSE_GLIMMER_MODEL_ATTN_QUERY_HEAD_COUNT",
                    "SPARK_MUSE_GLIMMER_MODEL_ATTENTION_HEAD_COUNT")
    kv_heads = alias(d, "SPARK_MUSE_GLIMMER_MODEL_ATTN_KV_HEAD_COUNT",
                     "SPARK_MUSE_GLIMMER_MODEL_KV_HEAD_COUNT")
    head_dim = alias(d, "SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION",
                     "SPARK_MUSE_GLIMMER_MODEL_HEAD_DIMENSION")
    intermediate = require(d, "SPARK_MUSE_GLIMMER_MODEL_INTERMEDIATE_DIMENSION")
    window = require(d, "SPARK_MUSE_GLIMMER_MODEL_SLIDING_WINDOW")
    full_layers = require(d, "SPARK_MUSE_GLIMMER_MODEL_FULL_ATTENTION_LAYER_COUNT")
    sliding_layers = require(d, "SPARK_MUSE_GLIMMER_MODEL_SLIDING_LAYER_COUNT")
    bf16 = require(d, "SPARK_MUSE_GLIMMER_MODEL_BF16_ELEMENT_BYTES")

    local = lambda value: (value + tp_degree - 1) // tp_degree
    rank_kv_heads = kv_heads // min(tp_degree, kv_heads)
    kv_slot_bytes = rank_kv_heads * 2 * head_dim * bf16

    rows = []
    attention_weight = 0
    attention_weight += local(q_heads + 2 * q_heads + 2 * kv_heads) * head_dim * hidden * bf16
    attention_weight += local(q_heads) * head_dim * hidden * bf16
    attention_weight += (q_heads + kv_heads) * head_dim * bf16
    attention_weight += 3 * hidden * bf16
    rows.append(("attention spine (fused qgkv + output)", layers, attention_weight))
    mlp_weight = 0
    mlp_weight += local(2 * intermediate) * hidden * bf16
    mlp_weight += local(intermediate) * hidden * bf16
    mlp_weight += hidden * bf16
    rows.append(("mlp spine", layers, mlp_weight))
    head_weight = local(vocab) * hidden * bf16
    rows.append(("lm head", 1, head_weight))
    rows.append(("embedding gather", 1, hidden * bf16))

    state_rows = []
    sliding_kv = sliding_layers * min(context_tokens, window) * kv_slot_bytes
    full_kv = full_layers * context_tokens * kv_slot_bytes
    state_rows.append(("sliding kv read+write (window %d)" % window, 1, sliding_kv + sliding_layers * kv_slot_bytes))
    state_rows.append(("full attention kv read+write", 1, full_kv + full_layers * kv_slot_bytes))
    return rows, state_rows, kv_slot_bytes


def gigabytes(value):
    return value / 1e9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--tp", type=int, default=16)
    parser.add_argument("--bw-gb-s", type=float, default=MEASURED_BW_GB_S)
    parser.add_argument("--receipt-tok-s", type=float, default=0.0)
    args = parser.parse_args()

    d = defines(HEADER.read_text())
    table, state_rows, kv_slot_bytes = build_table(d, args.context, args.tp)

    total = sum(count * value for _, count, value in table)
    total += sum(count * value for _, count, value in state_rows)
    print("muse_glimmer decode roofline, B1, TP%d, context %d tokens" % (
        args.tp, args.context))
    print("classification: %s" % CLASSIFICATION)
    print("bandwidth: measured %.1f GB/s of %.0f GB/s spec (%.1f%%) [%s]" % (
        args.bw_gb_s, SPEC_BW_GB_S, 100.0 * args.bw_gb_s / SPEC_BW_GB_S, PROBE_SOURCE))
    print(DENSE)
    print()
    print("%-52s %8s %14s" % ("block", "count", "bytes/token"))
    for name, count, value in table:
        print("%-52s %8d %14s" % (name, count, "%.3f MB" % (gigabytes(count * value) * 1e3)))
    for name, count, value in state_rows:
        print("%-52s %8s %14s" % (name, "-", "%.3f MB" % (gigabytes(count * value) * 1e3)))
    print("%-52s %8s %14s" % ("TOTAL per rank per token", "-", "%.3f GB" % gigabytes(total)))
    print()
    ceiling = args.bw_gb_s * 1e9 / total
    print("ceiling (all weights resident, bf16 pack law: never quantize): "
          "%.1f tok/s per rank at %.1f GB/s [ESTIMATE]" % (ceiling, args.bw_gb_s))
    if args.receipt_tok_s > 0.0:
        print("receipted %.4f tok/s -> %.1f%% of measured roofline" % (
            args.receipt_tok_s, 100.0 * args.receipt_tok_s * total / (args.bw_gb_s * 1e9)))
    else:
        print("receipted tok/s: none (daemon-gated) -> percent of roofline N/A")
    return 0


if __name__ == "__main__":
    sys.exit(main())
