#!/usr/bin/env python3
"""Print the lines of a build or audit log that say what failed.

    python3 -I tools/build_log_errors.py [--also REGEX] [--limit 20] [--context 6] [--tail 40] [--width 300] LOG

A parallel build's log is thousands of lines of warnings, each followed by
indented source text, and the command that failed may sit far from the end. A
line that says what failed starts in the first column: a compiler, linker or
toolkit error, a make failure, a Python exception, or a line a tool marks
REFUSED or FAIL. Each such line is printed once with its line number, how often
it repeats, and the indented lines under its first occurrence; make's lines
follow, since they name the targets. --also names further lines to print, such
as an audit's summary. A log with no such line prints its last lines instead.
"""
import argparse
import re
import sys
from pathlib import Path

FAILURE = re.compile(
    r'^\S.*?(?:: (?:fatal |catastrophic )?error\b|undefined reference to|multiple definition of|cannot find -l|ld returned \d+ exit status'
    r'|errors? detected in the compilation of)'
    r'|^(?:ptxas|nvlink|nvcc|cicc|cudafe\+\+)(?: |: )(?:fatal|error)\b|^ptxas .*; (?:fatal|error)\s*:'
    r'|^(?:REFUSED|FAIL)\b|^Traceback \(most recent call last\)|^[A-Za-z_][\w.]*(?:Error|Exception)(?::|$)')
MAKE = re.compile(r'^(?:make(?:\[\d+\])?|[^\s:]+:\d+): \*\*\* (?!Waiting for unfinished jobs)')


def failures(lines, also, context):
    found, targets, order = {}, {}, []
    for index, line in enumerate(lines):
        kind = targets if MAKE.match(line) else found if FAILURE.match(line) or (also is not None and also.search(line)) else None
        if kind is None:
            continue
        if line in kind:
            kind[line]['count'] += 1
            continue
        under = []
        for following in lines[index + 1:index + 1 + context]:
            if not following[:1].isspace() or not following.strip():
                break
            under.append(following)
        kind[line] = {'number': index + 1, 'count': 1, 'under': under if kind is found else []}
        order.append((kind is targets, line))
    return [(line, found[line]) for make, line in order if not make], [(line, targets[line]) for make, line in order if make]


def report(path, also=None, limit=20, context=6, tail=40, width=300):
    lines = Path(path).read_text(errors='replace').splitlines()
    found, targets = failures(lines, re.compile(also) if also else None, context)
    output = []
    for line, entry in found[:limit] + targets[:limit]:
        repeats = f' ({entry["count"]} times)' if entry['count'] > 1 else ''
        output.append(f'{entry["number"]}: {line[:width]}{repeats}')
        output.extend(under[:width] for under in entry['under'])
    left = max(0, len(found) - limit) + max(0, len(targets) - limit)
    if left:
        output.append(f'... and {left} more distinct failing lines of {len(lines)} in {path}')
    if not output:
        output.append(f'{path} names no failing line; its last {min(tail, len(lines))} of {len(lines)} lines:')
        output.extend(line[:width] for line in lines[-tail:])
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log')
    parser.add_argument('--also')
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--context', type=int, default=6)
    parser.add_argument('--tail', type=int, default=40)
    parser.add_argument('--width', type=int, default=300)
    args = parser.parse_args()
    try:
        output = report(args.log, args.also, args.limit, args.context, args.tail, args.width)
    except OSError as error:
        print(f'build_log_errors: cannot read {args.log}: {error}', file=sys.stderr)
        return 1
    print('\n'.join(output))
    return 0


if __name__ == '__main__':
    sys.exit(main())
