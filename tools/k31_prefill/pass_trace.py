import re
import sys
from collections import defaultdict

STAGES = [
    ("collective", ("SparkTpMesh", "SparkMesh", "SparkTp")),
    ("experts", ("LmCellMma",)),
    ("gemm", ("LmStreamGemm", "LmGemm", "LmSkinny", "LmDense")),
    ("attnres", ("LmAttnRes",)),
    ("mla_attention", ("LmLatentShard", "LmPerHeadProjectSplit", "LmLatentExpand", "LmLatentRope", "LmAttention")),
    ("kda", ("LmDeltaRule", "LmCausalConv", "LmL2NormalisePerHead", "LmBoundedDecay", "K3SplitFusedProjections", "LmSigmoidRows",
             "K3StreamedState", "K3StreamedConv", "K3StreamedPadRows")),
    ("index", ("LmWeightedSparseScore", "LmIndexSelect", "LmIndexShard", "LmLayerNorm", "LmTopkExact", "LmPassKeys")),
    ("moe_route", ("LmMoeFinalize", "LmRouteBuild", "LmSituMul", "K3RunnerLatentLogits", "LmTopk", "K3LatentSlice")),
    ("rowwise", ("LmAddRows", "K3RunnerFusedPack", "LmFusedResidualRmsNorm", "LmCopyRows", "LmOutputGate", "LmKvShardStore", "K3Runner",
                 "LmStage", "StageRunner", "K3Streamed")),
    ("head", ("SparkLmHead",)),
]

line_re = re.compile(r'^(CONCURRENT_KERNEL|KERNEL|MEMCPY|MEMSET)\s*(?:"([^"]*)")?\s*\[\s*(\d+),\s*(\d+)\s*\][^"]*(?:"([^"]*)")?')


def stage_of(kind, name):
    if kind in ("MEMCPY", "MEMSET"):
        return "copies"
    for stage, prefixes in STAGES:
        if name.startswith(prefixes):
            return stage
    return "other"


def merged(intervals):
    total = 0
    last_end = None
    for start, end in sorted(intervals):
        if last_end is None or start > last_end:
            total += end - start
            last_end = end
        elif end > last_end:
            total += end - last_end
            last_end = end
    return total


def main(path, gap_split_s=5.0):
    events = []
    with open(path, errors="replace") as handle:
        for line in handle:
            if not line.startswith(("CONCURRENT_KERNEL", "KERNEL", "MEMCPY", "MEMSET")):
                continue
            match = line_re.match(line)
            if not match:
                continue
            kind, subkind, start, end, name = match.groups()
            name = name if kind in ("CONCURRENT_KERNEL", "KERNEL") else (subkind or kind)
            events.append((int(start), int(end), kind, name or ""))
    events.sort()
    segments = []
    begin = 0
    running_end = events[0][1]
    for index in range(1, len(events)):
        if events[index][0] - running_end > gap_split_s * 1e9:
            segments.append((begin, index))
            begin = index
            running_end = events[index][1]
        else:
            running_end = max(running_end, events[index][1])
    segments.append((begin, len(events)))
    best = max(segments, key=lambda s: events[s[1] - 1][1] - events[s[0]][0])
    window = events[best[0]:best[1]]
    t0, t1 = window[0][0], max(e[1] for e in window)
    span = (t1 - t0) / 1e9
    by_stage = defaultdict(list)
    by_name = defaultdict(lambda: [0, 0])
    compute = []
    for start, end, kind, name in window:
        stage = stage_of(kind, name)
        by_stage[stage].append((start, end))
        if kind in ("CONCURRENT_KERNEL", "KERNEL"):
            by_name[name][0] += end - start
            by_name[name][1] += 1
        if stage not in ("collective", "copies"):
            compute.append((start, end))
    busy = merged([(s, e) for s, e, _, _ in window])
    print(f"window {span:.1f} s, {len(window)} records; any GPU activity {busy / 1e9:.1f} s, idle {span - busy / 1e9:.1f} s; compute (not mesh, not copies) {merged(compute) / 1e9:.1f} s")
    for stage in sorted(by_stage, key=lambda s: -merged(by_stage[s])):
        print(f"stage\t{stage}\t{merged(by_stage[stage]) / 1e9:.2f}\t{len(by_stage[stage])}")
    gaps = []
    last_end = t0
    for start, end, _, _ in window:
        if start > last_end:
            gaps.append(start - last_end)
        last_end = max(last_end, end)
    for limit in (1e5, 1e6, 1e7, 1e8):
        chosen = [g for g in gaps if g >= limit]
        print(f"gaps >= {limit / 1e6:g} ms: {len(chosen)}, {sum(chosen) / 1e9:.2f} s")
    print("seconds\tlaunches\tkernel")
    for name, (total, count) in sorted(by_name.items(), key=lambda item: -item[1][0])[:45]:
        print(f"{total / 1e9:.3f}\t{count}\t{name[:150]}")


if __name__ == "__main__":
    main(sys.argv[1])
