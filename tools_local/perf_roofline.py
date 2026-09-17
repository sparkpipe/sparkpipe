#!/usr/bin/env python3
import sys

MEM_PEAK_GB_S = 273.0
NET_PEAK_GB_S = 25.0
PACK_BYTES = 21706046976
TP_DEGREE = 16
HIDDEN = 4096
LAYERS = 45
OPS_PER_TOKEN = 2 * LAYERS + 16
BF16_PAYLOAD = HIDDEN * 2
TX_BYTES_PER_TOKEN = OPS_PER_TOKEN * 2 * BF16_PAYLOAD
WEIGHT_BYTES_PER_NODE = PACK_BYTES / TP_DEGREE
MEM_ROOFLINE_TOK_S = MEM_PEAK_GB_S * 1e9 / WEIGHT_BYTES_PER_NODE


def roofline(tok_s):
    mem_pct = 100.0 * tok_s * WEIGHT_BYTES_PER_NODE / (MEM_PEAK_GB_S * 1e9)
    net_pct = 100.0 * tok_s * TX_BYTES_PER_TOKEN / (NET_PEAK_GB_S * 1e9)
    print("decode %.2f tok/s | mem %.2f%% of %.0f GB/s | net %.4f%% of %.0f GB/s | mem-roofline %.1f tok/s"
          % (tok_s, mem_pct, MEM_PEAK_GB_S, net_pct, NET_PEAK_GB_S,
             MEM_ROOFLINE_TOK_S))


def prefill(tokens, seconds):
    gbs = WEIGHT_BYTES_PER_NODE / seconds / 1e9
    print("prefill %d tok in %.2fs | %.1f tok/s | weight-stream %.2f GB/s = %.2f%% mem roofline (one shard pass)"
          % (tokens, seconds, tokens / seconds, gbs,
             100.0 * gbs / MEM_PEAK_GB_S))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "prefill":
        prefill(int(sys.argv[2]), float(sys.argv[3]))
    elif len(sys.argv) == 2:
        roofline(float(sys.argv[1]))
    else:
        print("usage: perf_roofline.py <decode tok/s>")
        print("       perf_roofline.py prefill <tokens> <seconds>")
        print("assumes: pack %.2f GB, TP%d, hidden %d, %d allreduce ops/token"
              % (PACK_BYTES / 1e9, TP_DEGREE, HIDDEN, OPS_PER_TOKEN))
