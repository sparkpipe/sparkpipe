#!/usr/bin/env python3
"""Refuse kernels that spill registers to local memory or keep a stack frame.

    python3 -I tools/kernel_spill_audit.py [--cuobjdump PATH] --kernel NAME ARTIFACT...

For every function of the artifacts (cubins, modules or programs) whose name
contains one of the --kernel names, cuobjdump's resource report must show no
local memory (registers spilled by ptxas) and no stack frame. A kernel that
spills stores and reloads registers through local memory on every pass of its
loop, and that can make a memory-bound GEMV a hundred times slower. The exit
status is 0 when nothing is refused, 1 when something is, and 2 when an
artifact cannot be read.
"""
import argparse
import re
import subprocess
import sys

FUNCTION = re.compile(r'^\s*Function (\S+):')
RESOURCE = re.compile(r'^\s*REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)')


def resources(listing):
    found, name = [], None
    for line in listing.splitlines():
        match = FUNCTION.match(line)
        if match is not None:
            name = match.group(1)
            continue
        match = RESOURCE.match(line)
        if match is not None and name is not None:
            found.append((name, int(match.group(1)), int(match.group(2)), int(match.group(4))))
            name = None
    return found


def refusals(found, kernels):
    refused, checked = [], 0
    for name, registers, stack, local in found:
        if not any(kernel in name for kernel in kernels):
            continue
        checked += 1
        if stack != 0 or local != 0:
            refused.append((name, registers, stack, local))
    return refused, checked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('artifacts', nargs='+')
    parser.add_argument('--kernel', action='append', required=True)
    parser.add_argument('--cuobjdump', default='cuobjdump')
    args = parser.parse_args()
    refused, checked = [], 0
    for artifact in args.artifacts:
        result = subprocess.run([args.cuobjdump, '-res-usage', artifact], capture_output=True, text=True)
        if result.returncode != 0 and 'does not contain device code' in result.stderr:
            continue
        if result.returncode != 0:
            print(f'kernel_spill_audit: cuobjdump cannot read {artifact}: {result.stderr.strip()}', file=sys.stderr)
            return 2
        found, count = refusals(resources(result.stdout), args.kernel)
        refused += [(artifact,) + item for item in found]
        checked += count
    for artifact, name, registers, stack, local in refused:
        print(f'REFUSED {artifact}: {name} uses {registers} registers and spills: stack {stack} bytes, local {local} bytes')
    print(f'kernel spill audit: {checked} kernels named {", ".join(args.kernel)} in {len(args.artifacts)} artifacts, {len(refused)} spill')
    return 1 if refused else 0


if __name__ == '__main__':
    sys.exit(main())
