import argparse
import collections
import json
import pathlib
import re


RECORD = re.compile(r'^(CONCURRENT_KERNEL|KERNEL|MEMCPY|MEMSET)(?: "([^"]*)")? \[ (\d+), (\d+) \] duration (\d+)(?:, "([^"]*)")?')


def merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def coverage(intervals):
    return sum(end - start for start, end in merge(intervals))


def fields(line):
    return dict(re.findall(r'(\w+)=([^\s]+)', line))


def report(path, clock='cupti', start=None, end=None):
    records = []
    anchor = None
    exit_record = None
    dropped = buffers = invalid = 0
    with path.open() as trace:
        for line in trace:
            if line.startswith('TRACE_ANCHOR '):
                if anchor is not None:
                    raise ValueError('Multiple clock anchors: traces from different processes cannot be combined')
                anchor = {key: int(value) for key, value in fields(line).items()}
            elif line.startswith('TRACE_EXIT '):
                exit_record = {key: int(value) for key, value in fields(line).items()}
            elif line.startswith('TRACE_BUFFER '):
                buffers += 1
                dropped += int(fields(line)['dropped_records'])
            elif line.startswith(('CONCURRENT_KERNEL ', 'KERNEL ', 'MEMCPY ', 'MEMSET ')):
                match = RECORD.match(line)
                if match is None:
                    invalid += 1
                    continue
                kind, copy_kind, first, last, duration, name = match.groups()
                first, last, duration = int(first), int(last), int(duration)
                if first == 0 or last <= first or duration != last - first:
                    invalid += 1
                    continue
                category = ('host_memcpy' if copy_kind == 'HtoH' else 'memcpy') if kind == 'MEMCPY' else 'memset' if kind == 'MEMSET' else 'mesh' if 'SparkGlm5NextMesh' in (name or '') else 'compute'
                if kind in ('KERNEL', 'CONCURRENT_KERNEL') and not name:
                    invalid += 1
                    continue
                records.append((first, last, category, name or copy_kind or kind))
    gpu_records = [item for item in records if item[2] != 'host_memcpy']
    if not gpu_records:
        raise ValueError('No completed GPU kernel/copy/set activity records')
    offsets = {'cupti': 0}
    if anchor is not None:
        offsets.update({key: anchor[f'{key}_ns'] - anchor['cupti_ns'] for key in ('monotonic', 'realtime')})
    if clock not in offsets:
        raise ValueError(f'{clock} selection requires TRACE_ANCHOR')
    if (start is None) != (end is None):
        raise ValueError('Specify both start and end')
    low = min(item[0] for item in gpu_records) if start is None else start - offsets[clock]
    high = max(item[1] for item in gpu_records) if end is None else end - offsets[clock]
    if high <= low:
        raise ValueError('Selected interval must have positive duration')
    categories = collections.defaultdict(list)
    names = collections.defaultdict(list)
    clipped = 0
    for first, last, category, name in records:
        if first < high and last > low:
            interval = [max(first, low), min(last, high)]
            clipped += interval != [first, last]
            categories[category].append(interval)
            names[(category, name)].append(interval)
    active = merge([interval for category, group in categories.items() if category != 'host_memcpy' for interval in group])
    gaps = []
    cursor = low
    for first, last in active:
        if first > cursor:
            gaps.append([cursor, first])
        cursor = last
    if cursor < high:
        gaps.append([cursor, high])
    gaps.sort(key=lambda interval: interval[1] - interval[0], reverse=True)
    issues = []
    if anchor is None:
        issues.append('missing_clock_anchor')
    if exit_record is None:
        issues.append('missing_normal_exit_flush')
    if buffers == 0:
        issues.append('missing_buffer_loss_accounting')
    if dropped:
        issues.append('dropped_activity_records')
    if invalid:
        issues.append('invalid_or_incomplete_gpu_records')
    return {
        'trace': str(path.resolve()), 'status': 'PARTIAL' if issues else 'COMPLETE', 'issues': issues,
        'anchor': anchor, 'exit': exit_record, 'returned_buffers': buffers,
        'dropped_records': dropped, 'invalid_gpu_records': invalid,
        'selection_ns': {key: [low + offset, high + offset] for key, offset in offsets.items()},
        'window_ns': high - low, 'gpu_covered_ns': coverage(active),
        'gpu_uncovered_ns': high - low - coverage(active), 'clipped_record_count': clipped,
        'categories': {key: {'records': len(value), 'covered_ns': coverage(value),
                             'sum_record_ns': sum(last - first for first, last in value)}
                       for key, value in sorted(categories.items())},
        'top_names': [{'category': key[0], 'name': key[1], 'records': len(value),
                       'covered_ns': coverage(value), 'sum_record_ns': sum(last - first for first, last in value)}
                      for key, value in sorted(names.items(), key=lambda item: coverage(item[1]), reverse=True)[:40]],
        'largest_uncovered_intervals': [{'duration_ns': last - first,
                                         **{f'{key}_ns': [first + offset, last + offset] for key, offset in offsets.items()}}
                                        for first, last in gaps[:20]],
        'interpretation': 'Coverage is the union of GPU intervals, excluding host-to-host copies. Category coverage may overlap and is not additive. Gaps are unobserved GPU activity, not automatically network or CPU time. Clock anchors are nearby samples, not cross-host synchronization.'
    }


def main():
    parser = argparse.ArgumentParser(description='Summarize a bounded CUPTI trace with overlap-safe GPU coverage and explicit loss accounting.')
    parser.add_argument('trace', type=pathlib.Path)
    parser.add_argument('--clock', choices=('cupti', 'monotonic', 'realtime'), default='cupti')
    parser.add_argument('--start-ns', type=int)
    parser.add_argument('--end-ns', type=int)
    parser.add_argument('--output', type=pathlib.Path)
    args = parser.parse_args()
    try:
        result = report(args.trace, args.clock, args.start_ns, args.end_ns)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end='')
    return 0 if result['status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
