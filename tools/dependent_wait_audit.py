#!/usr/bin/env python3
"""Check that no kernel writes global memory before it waits for the kernel it depends on.

    python3 -I tools/dependent_wait_audit.py [--cuobjdump PATH] ARTIFACT...

A kernel launched as a programmatic dependent may start while the kernel before
it still runs: once that kernel has released its dependents, everything the
dependent does before griddepcontrol.wait runs beside it. Before the wait a
kernel may therefore only read what no kernel of the step writes (weights and
their scales) and prefetch it. It must not store to global memory or touch it
with an atomic or a reduction, or it could overwrite what the running kernel
still reads.

The audit reads every function's SASS from the artifacts (cubins, modules or
programs) with cuobjdump. In every function that waits (ACQBULK, the form of
griddepcontrol.wait on sm_90 and later), it refuses a global store, atomic or
reduction that comes before the first wait. ALLOWED names the kernels whose
stores before the wait are safe for a stated reason. The exit status is 0 when
nothing is refused, 1 when something is, and 2 when an artifact cannot be read.
"""
import argparse
import re
import subprocess
import sys

FUNCTION = re.compile(r'^\s*Function : (\S+)')
INSTRUCTION = re.compile(r'^\s*(?:/\*[0-9a-f]+\*/)?\s*(?:@!?U?P[0-9T]+\s+)?([A-Z][A-Z0-9_]*)((?:\.[A-Z0-9_]+)*)\b')
WAIT = 'ACQBULK'
WRITES = {'STG', 'ST', 'ATOMG', 'ATOM', 'RED', 'REDG', 'SUST', 'SURED', 'SUATOM', 'UTMASTG', 'UTMAREDG', 'UBLKCP', 'UBLKRED'}
ALLOWED = {
    'SparkTpMeshHardwareDirectKernel': 'its only dependent predecessor is the mesh arrival kernel, which touches only the band and its control words',
    'SparkTpMeshHardwareDirectWideKernel': 'its only dependent predecessor is the mesh arrival kernel, which touches only the band and its control words',
}


def functions(listing):
    current, found = None, {}
    for line in listing.splitlines():
        match = FUNCTION.match(line)
        if match is not None:
            current = match.group(1)
            found[current] = []
            continue
        if current is None:
            continue
        match = INSTRUCTION.match(line)
        if match is not None:
            found[current].append((match.group(1), match.group(1) + match.group(2), line.strip()))
    return found


def stem(name):
    if not name.startswith('_Z'):
        return name
    rest = name[3:] if name.startswith('_ZN') else name[2:]
    match = re.match(r'(\d+)', rest)
    if match is None:
        return name
    start = match.end()
    return rest[start:start + int(match.group(1))]


def refusals(found):
    refused, checked = [], 0
    for name, instructions in found.items():
        mnemonics = [mnemonic for mnemonic, _, _ in instructions]
        if WAIT not in mnemonics:
            continue
        checked += 1
        if stem(name) in ALLOWED:
            continue
        for mnemonic, full, text in instructions[:mnemonics.index(WAIT)]:
            if mnemonic in WRITES and not full.startswith(('UBLKCP.S.G', 'UBLKCP.S.S')):
                refused.append((name, text))
    return refused, checked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('artifacts', nargs='+')
    parser.add_argument('--cuobjdump', default='cuobjdump')
    args = parser.parse_args()
    refused, checked = [], 0
    for artifact in args.artifacts:
        result = subprocess.run([args.cuobjdump, '-sass', artifact], capture_output=True, text=True)
        if result.returncode != 0 and 'does not contain device code' in result.stderr:
            continue
        if result.returncode != 0:
            print(f'dependent_wait_audit: cuobjdump cannot read {artifact}: {result.stderr.strip()}', file=sys.stderr)
            return 2
        found, count = refusals(functions(result.stdout))
        refused += [(artifact, name, text) for name, text in found]
        checked += count
    for artifact, name, text in refused:
        print(f'REFUSED {artifact}: {name} writes global memory before it waits: {text}')
    print(f'dependent wait audit: {checked} waiting kernels in {len(args.artifacts)} artifacts, {len(refused)} writes before a wait')
    return 1 if refused else 0


if __name__ == '__main__':
    sys.exit(main())
