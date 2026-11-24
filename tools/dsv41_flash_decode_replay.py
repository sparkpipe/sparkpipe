#!/usr/bin/env python3
"""Summarise build/dsv41_flash_decode_replay runs traced by the CUPTI injection.

The replay prints one REPLAY line per run and launches ReplayMarkKernel before
every graph launch; the trace (tools/tp_cupti_trace_build.py) records every
kernel. Kernels are split into steps at the marks, the warmup steps of each run
are dropped, and each kernel position gets its median duration and the median
gap since the previous kernel ended. Sequences are matched by kernel names in
order, as in the four-node traces the reference file was taken from.

    python3 -I tools/dsv41_flash_decode_replay.py --trace TRACE --stdout OUT \\
        [--warmup 5] [--reference tools/dsv41_flash_decode_reference.json] [--json OUT.json]
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

RECORD = re.compile(r'^(CONCURRENT_KERNEL|KERNEL) \[ (\d+), (\d+) \] duration (\d+), "([^"]*)"')
RUN = re.compile(r'^REPLAY (\S+) rows=(\d+) steps=(\d+)')
CHAIN = re.compile(r'^REPLAY boundary chain=(\d+) name=(\S+) links=(\d+) rounds=(\d+)')
MARK = 'ReplayMarkKernel'
SIDE = ('ReplayPrefetchKernel',)
SKINNY = ('LmSkinnyFrameKernel', 'LmSkinnyKernel', 'LmSkinnyMultiKernel')
COLLECTIVE = ('ReplayArrivalKernel', 'ReplayCopyKernel', 'SparkTpMesh')
SEQUENCES = {
    'attention_output': ['D41MergeSinkKernel', 'D41RopeRowsKernel', 'D41ColumnsKernel', 'SparkDsv41FlashFp8QdqKernel', SKINNY, 'D41ColumnsKernel',
                         'D41ColumnsKernel', 'SparkDsv41FlashFp8QdqKernel', SKINNY, 'D41ColumnsKernel', 'SparkDsv41FlashFp8QdqKernel', SKINNY],
    'attention_core': [('D41InitRecordsKernel', 'LmLatentShardPrefillKernel'), 'D41WindowFoldWholeKernel', 'D41WindowRingStoreKernel'],
    'attention_input': [SKINNY, 'D41RmsNormQdqKernel', 'D41KvNormRopeQdqKernel', SKINNY, 'D41RopeRowsKernel'],
    'shared_expert': [SKINNY, 'D41ExpertActKernel', 'D41SwigluRowsKernel', 'SparkDsv41FlashFp8QdqKernel', SKINNY],
    'mhc': ['LmHcSinkhornPartialKernel', 'LmHcSinkhornKernel', 'D41RmsNormQdqKernel'],
    'mhc_fused': ['D41FusedMixNormKernel'],
    'attention_output_fused': ['D41FusedOutputKernel', SKINNY, SKINNY, 'SparkDsv41FlashFp8QdqKernel', SKINNY],
    'router_fused': [SKINNY, 'D41FusedRouteKernel'],
    'router': [SKINNY, 'LmTopkWarpKernel', 'D41RouteLocalKernel', 'LmRouteBuildKernel', 'LmPackedRouteWeightKernel'],
}


def stem(name):
    if name.startswith('_Z'):
        rest = name[2:]
        if rest.startswith('N'):
            rest = rest[1:]
        match = re.match(r'(\d+)', rest)
        if match is not None:
            start = len(match.group(1))
            return rest[start:start + int(match.group(1))]
    match = re.search(r'([A-Za-z_]\w*)\s*(?:<|\()', name)
    return match.group(1) if match is not None else name


def kernels(path):
    found = []
    with Path(path).open(errors='replace') as trace:
        for line in trace:
            match = RECORD.match(line)
            if match is not None:
                found.append((int(match.group(2)), int(match.group(3)), stem(match.group(5))))
    found.sort()
    return found


def segments(found):
    steps, current = [], None
    for start, end, name in found:
        if name == MARK:
            if current is not None:
                steps.append(current)
            current = []
        elif current is not None:
            current.append((start, end, name))
    if current is not None:
        steps.append(current)
    return steps


def medians(steps):
    main = [[item for item in step if item[2] not in SIDE] for step in steps]
    side = [[item for item in step if item[2] in SIDE] for step in steps]
    count = statistics.mode(len(step) for step in main)
    kept = [(step, extra) for step, extra in zip(main, side) if len(step) == count]
    names = [item[2] for item in kept[0][0]]
    if any([item[2] for item in step] != names for step, _ in kept):
        raise ValueError('the kept steps do not launch the same kernels in the same order')
    duration = [statistics.median(step[index][1] - step[index][0] for step, _ in kept) / 1e3 for index in range(count)]
    gap = [0.0] + [statistics.median(max(0, step[index][0] - step[index - 1][1]) for step, _ in kept) / 1e3 for index in range(1, count)]
    span = statistics.median((step[-1][1] - step[0][0]) / 1e3 for step, _ in kept)
    prefetch = statistics.median(sum(item[1] - item[0] for item in extra) / 1e3 for _, extra in kept)
    return names, duration, gap, span, len(kept), prefetch


def matches(names, index, pattern):
    if index + len(pattern) > len(names):
        return False
    for offset, want in enumerate(pattern):
        have = names[index + offset]
        if (isinstance(want, tuple) and have not in want) or (isinstance(want, str) and have != want):
            return False
    return True


def profile(names, duration, gap):
    result = {'kernels': len(names), 'busy_us': sum(duration), 'gaps_us': sum(gap)}
    result['collective_us'] = sum(d + g for n, d, g in zip(names, duration, gap) if n.startswith(COLLECTIVE))
    result['compute_us'] = result['busy_us'] + result['gaps_us'] - result['collective_us']
    sequences = {}
    for key, pattern in SEQUENCES.items():
        found = [index for index in range(len(names)) if matches(names, index, pattern)]
        totals = [sum(duration[index:index + len(pattern)]) + sum(gap[index + 1:index + len(pattern)]) for index in found]
        gemv = [sum(duration[index + offset] for offset, want in enumerate(pattern) if want == SKINNY) for index in found]
        sequences[key] = {'instances': len(found), 'us_per_instance': statistics.mean(totals) if totals else 0.0,
                          'gemv_us_per_instance': statistics.mean(gemv) if gemv else 0.0, 'us_per_step': sum(totals)}
        if key == 'attention_core':
            fold = [duration[index + 1] for index in found]
            sequences[key]['fold_us_per_instance'] = statistics.mean(fold) if fold else 0.0
    result['sequences'] = sequences
    families = {}
    for name, d, g in zip(names, duration, gap):
        entry = families.setdefault(name, {'calls': 0, 'us': 0.0})
        entry['calls'] += 1
        entry['us'] += d + g
    result['families'] = families
    return result


def boundary(steps):
    names, duration, gap, _, kept, _ = medians(steps)
    second = [index for index in range(1, len(names), 2)]
    return {'steps': kept, 'first': names[0], 'second': names[1],
            'gap_before_second_us': statistics.median(gap[index] for index in second),
            'second_us': statistics.median(duration[index] for index in second),
            'first_us': statistics.median(duration[index] for index in range(0, len(names), 2))}


def compare(result, reference, rows):
    table = reference.get('rows', {}).get(str(1 if rows <= 2 else 8))
    if table is None:
        return {}
    report = {}
    for key, want in table['sequences'].items():
        have = result['sequences'].get(key, {}).get('us_per_instance', 0.0)
        report[key] = {'replay_us': round(have, 2), 'trace_us': want['us_per_instance'],
                       'ratio': round(have / want['us_per_instance'], 3) if want['us_per_instance'] else None}
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', required=True)
    parser.add_argument('--stdout', required=True)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--reference', default=str(Path(__file__).with_name('dsv41_flash_decode_reference.json')))
    parser.add_argument('--json')
    args = parser.parse_args()
    lines = Path(args.stdout).read_text().splitlines()
    runs = [(m.group(1), int(m.group(2)), int(m.group(3))) for m in map(RUN.match, lines) if m is not None and m.group(1) != 'boundary']
    chains = [(int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))) for m in map(CHAIN.match, lines) if m is not None]
    steps = segments(kernels(args.trace))
    reference = json.loads(Path(args.reference).read_text()) if Path(args.reference).is_file() else {}
    output = {'runs': {}, 'boundaries': {}}
    cursor = 0
    for label, rows, count in runs:
        run = steps[cursor + args.warmup:cursor + args.warmup + count]
        cursor += args.warmup + count
        names, duration, gap, span, kept, prefetch = medians(run)
        result = profile(names, duration, gap)
        result.update({'rows': rows, 'steps_kept': kept, 'span_us': span, 'side_prefetch_us': prefetch, 'against_trace': compare(result, reference, rows)})
        output['runs'][label] = result
        print(f'{label}: rows {rows}, {kept} steps, span {span / 1e3:.2f} ms, {result["kernels"]} kernels, compute {result["compute_us"] / 1e3:.2f} ms, '
              f'collective stand-ins {result["collective_us"] / 1e3:.2f} ms')
        for key, entry in result['sequences'].items():
            against = result['against_trace'].get(key, {})
            print(f'  {key:18s} {entry["instances"]:3d} x {entry["us_per_instance"]:7.1f} us (GEMV {entry["gemv_us_per_instance"]:6.1f})'
                  f'  trace {against.get("trace_us", 0):7.1f} us  ratio {against.get("ratio")}')
    for chain, name, links, rounds in chains:
        run = steps[cursor + args.warmup:cursor + rounds]
        cursor += rounds
        result = boundary(run)
        output['boundaries'][name] = result
        print(f'boundary {name}: {result["first"]} {result["first_us"]:.2f} us, gap {result["gap_before_second_us"]:.2f} us, '
              f'{result["second"]} {result["second_us"]:.2f} us')
    if args.json:
        Path(args.json).write_text(json.dumps(output, indent=1, sort_keys=True) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
