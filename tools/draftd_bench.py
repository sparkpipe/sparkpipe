import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import draftd_mtp_g8 as g8
from draftd_glm53flash_mtp import Glm53FlashMtpDrafter


def read_peak(gib=4, repeats=20):
    buffer = torch.empty(int(gib * (1 << 30)) // 2, dtype=torch.bfloat16, device="cuda").normal_()
    for _ in range(3):
        buffer.sum(dtype=torch.float32)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        buffer.sum(dtype=torch.float32)
    end.record()
    torch.cuda.synchronize()
    seconds = start.elapsed_time(end) / 1000.0 / repeats
    del buffer
    torch.cuda.empty_cache()
    return buffer_bytes(gib) / seconds


def buffer_bytes(gib):
    return int(gib * (1 << 30)) // 2 * 2


def bytes_per_draft_token(d):
    dense = [d.eh_proj, d.q_a, d.q_b, d.kv_a, d.o_proj, d.wk, d.wv, d.router, d.router_bias,
             d.shared_gate, d.shared_up, d.shared_down, d.lm_head]
    total = sum(t.numel() * t.element_size() for t in dense)
    per_expert = sum(c[0].numel() for c in (d.gate_codes, d.up_codes, d.down_codes))
    per_expert += sum(s[0].numel() * 4 for s in (d.gate_scale, d.up_scale, d.down_scale))
    return total + d.topk * per_expert + d.hidden * 2


def percentile(values, q):
    return float(np.percentile(np.asarray(values), q))


def time_graph(drafter, inputs, depth, repeats):
    drafter.capture(depth)
    graph, state = drafter.graphs[depth]
    device_ms, host_ms = [], []
    for i in range(repeats):
        hidden, token = inputs[i % len(inputs)]
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall = time.perf_counter()
        state["hidden"].copy_(hidden, non_blocking=True)
        state["token"].fill_(token)
        start.record()
        graph.replay()
        end.record()
        drafts = state["drafts"].tolist()
        host_ms.append((time.perf_counter() - wall) * 1000.0)
        device_ms.append(start.elapsed_time(end))
    return device_ms, host_ms, drafts


def main(argv=None):
    parser = argparse.ArgumentParser(description="rtx5090 draftd MTP latency, memory and determinism bench")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fixtures", nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=7)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    torch.cuda.reset_peak_memory_stats()
    drafter = Glm53FlashMtpDrafter(args.checkpoint, max_chain=args.max_depth)
    config = drafter.config
    real, _ = g8.fixture_cases(args.fixtures, drafter.layer - 1, int(config["hc_mult"]), drafter.hidden, 1)
    inputs = [(torch.as_tensor(c["hidden"], device="cuda"), c["fed"][0]) for c in real]
    peak = read_peak()
    per_token = bytes_per_draft_token(drafter)
    report = {"read_peak_gbs": peak / 1e9, "bytes_per_draft_token": per_token,
              "device_bytes": drafter.device_bytes(), "depths": {}}
    for depth in range(1, args.max_depth + 1):
        device_ms, host_ms, _ = time_graph(drafter, inputs, depth, args.repeats)
        p50 = percentile(device_ms, 50)
        report["depths"][depth] = {"device_p50_ms": p50, "device_p99_ms": percentile(device_ms, 99),
                                   "host_p50_ms": percentile(host_ms, 50), "host_p99_ms": percentile(host_ms, 99),
                                   "per_token_ms": p50 / depth,
                                   "memory_pct": 100.0 * per_token * depth / (p50 / 1000.0) / peak}
    eager = []
    for i in range(20):
        hidden, token = inputs[i % len(inputs)]
        torch.cuda.synchronize()
        wall = time.perf_counter()
        drafter.chain_eager(hidden, token, args.max_depth).tolist()
        eager.append((time.perf_counter() - wall) * 1000.0)
    report["eager_depth_ms_p50"] = percentile(eager, 50)
    identical = True
    graph_equals_eager = True
    for hidden, token in inputs:
        a = drafter.chain(hidden, token, args.max_depth).tolist()
        b = drafter.chain(hidden, token, args.max_depth).tolist()
        c = drafter.chain_eager(hidden, token, args.max_depth).tolist()
        identical &= a == b
        graph_equals_eager &= a == c
    report["graph_run_to_run_identical"] = identical
    report["graph_equals_eager"] = graph_equals_eager
    report["max_memory_allocated"] = torch.cuda.max_memory_allocated()
    json.dump(report, open(args.output, "w"), indent=1)
    for depth, row in report["depths"].items():
        print(f"DRAFTD-MTP depth={depth} device_p50={row['device_p50_ms']:.3f}ms p99={row['device_p99_ms']:.3f}ms "
              f"host_p50={row['host_p50_ms']:.3f}ms per_token={row['per_token_ms']:.3f}ms "
              f"memory={row['memory_pct']:.1f}%")
    print(f"DRAFTD-MTP read_peak={report['read_peak_gbs']:.0f}GB/s bytes_per_token={per_token} "
          f"device_bytes={report['device_bytes']} max_allocated={report['max_memory_allocated']} "
          f"eager_depth{args.max_depth}_p50={report['eager_depth_ms_p50']:.2f}ms "
          f"run_to_run_identical={identical} graph_equals_eager={graph_equals_eager}")
    return 0 if identical and graph_equals_eager else 1


if __name__ == "__main__":
    sys.exit(main())
