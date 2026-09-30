#!/usr/bin/env python3
import argparse
import collections
import os
import re
import subprocess
import sys

SKIP_MODULES = ('libcuda.so', 'libcupti.so', 'libspark_gpu_bytes_trace.so', 'libcudart.so', 'libc.so', 'libstdc++')
SKIP_FUNCTIONS = ('cuda', '__cudart', 'libcudart', '__device_stub', '__wrapper', 'cudart::')
RECORD = re.compile(r'(\w+)=(\S+)')


def parse(path):
    records = []
    samples = []
    with open(path) as handle:
        for line in handle:
            if line.startswith('GPUB-SAMPLE'):
                samples.append(dict(RECORD.findall(line)))
            elif line.startswith('GPUB '):
                fields = dict(RECORD.findall(line))
                fields['frames'] = fields.get('stack', '').split(',') if fields.get('stack') else []
                records.append(fields)
    return records, samples


def resolve(frames, prefix_map):
    by_module = collections.defaultdict(set)
    for frame in frames:
        module, _, offset = frame.rpartition('+')
        if module and not any(skip in module for skip in SKIP_MODULES):
            by_module[module].add(offset)
    names = {}
    for module, offsets in by_module.items():
        path = module
        for source, target in prefix_map:
            if path.startswith(source):
                path = target + path[len(source):]
        ordered = sorted(offsets)
        if not os.path.exists(path):
            continue
        try:
            output = subprocess.run(['addr2line', '-f', '-C', '-e', path] + ['0x%x' % (int(o, 16) - 1) for o in ordered],
                                    capture_output=True, text=True, check=True).stdout.splitlines()
        except (OSError, subprocess.CalledProcessError):
            continue
        for index, offset in enumerate(ordered):
            function = output[2 * index] if 2 * index < len(output) else '??'
            names[module + '+' + offset] = '%s:%s' % (os.path.basename(module), function)
    return names


def site(record, names, depth):
    chain = []
    for frame in record['frames']:
        module = frame.rpartition('+')[0]
        if any(skip in module for skip in SKIP_MODULES):
            continue
        name = names.get(frame, os.path.basename(module) + ':' + frame.rpartition('+')[2])
        function = name.split(':', 1)[1]
        if function.startswith(SKIP_FUNCTIONS):
            continue
        chain.append(name)
        if len(chain) == depth:
            break
    return ' <- '.join(chain) if chain else '(driver internal)'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('trace')
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--map', action='append', default=[], help='SOURCE_PREFIX=TARGET_PREFIX for binary paths')
    arguments = parser.parse_args()
    prefix_map = [tuple(item.split('=', 1)) for item in arguments.map]
    records, samples = parse(arguments.trace)
    frames = [frame for record in records for frame in record['frames']]
    names = resolve(frames, prefix_map)
    live = {}
    peak = {}
    running = 0
    peak_bytes = -1
    groups = collections.OrderedDict()
    deltas = []
    for record in records:
        kind = record.get('kind')
        key = record.get('key')
        if record.get('release') == '1':
            if key in live:
                running -= live.pop(key)[1]
            continue
        if kind == 'delta':
            if int(record.get('bytes', '0')) != 0:
                deltas.append((record['api'], int(record['bytes']), int(record.get('nodes', '0')), site(record, names, arguments.depth)))
            continue
        if kind in ('event', 'error'):
            continue
        live[key] = (kind, int(record.get('bytes', '0')), site(record, names, arguments.depth))
        running += live[key][1]
        if running > peak_bytes:
            peak_bytes = running
            peak = dict(live)
    for kind, size, where in peak.values():
        entry = groups.setdefault((kind, where), [0, 0])
        entry[0] += size
        entry[1] += 1
    total = collections.Counter()
    print('| kind | bytes live at peak | MiB | count | call site |')
    print('|---|---:|---:|---:|---|')
    for (kind, where), (size, count) in sorted(groups.items(), key=lambda item: -item[1][0]):
        total[kind] += size
        print('| %s | %d | %.1f | %d | %s |' % (kind, size, size / 1048576.0, count, where))
    print()
    for kind, size in total.items():
        print('TOTAL %s %d bytes %.1f MiB' % (kind, size, size / 1048576.0))
    print()
    print('| driver-internal step (NVML process delta) | bytes | MiB | graph nodes | call site |')
    print('|---|---:|---:|---:|---|')
    for api, size, nodes, where in deltas:
        print('| %s | %d | %.1f | %d | %s |' % (api, size, size / 1048576.0, nodes, where))
    if samples:
        last = samples[-1]
        print()
        print('LAST-SAMPLE nvml=%s (%.1f MiB) rss=%s (%.1f MiB)' % (last['nvml'], int(last['nvml']) / 1048576.0, last['rss'], int(last['rss']) / 1048576.0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
