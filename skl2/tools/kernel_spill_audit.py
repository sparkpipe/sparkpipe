#!/usr/bin/env python3
"""Report every kernel that keeps a stack frame or local memory, and refuse one
in a hot-path family.

    python3 -I skl2/tools/kernel_spill_audit.py [--cuobjdump PATH] [--print-pending] ARTIFACT...
    python3 -I skl2/tools/kernel_spill_audit.py [--cuobjdump PATH] --kernel NAME ARTIFACT...

cuobjdump's resource report gives every function of the artifacts (serving
adapters, CUDA programs) its registers, stack frame and local memory. A frame
holds registers that ptxas spilled or an array the kernel indexes at run time;
either way the kernel moves data through local memory, and in a loop that can
make it several times slower. The audit lists every function with a frame,
grouped by kernel family. The skinny GEMV, cell, stream GEMM and attention
families run on every decode or prefill step, so a frame in any of them is
refused, except the attention kernels in PENDING, which record the frames they
carry today. PENDING must match the artifacts exactly, so it can only shrink: a
new frame or a larger one is refused, a smaller one fails until its entry is
lowered, and an entry whose kernel has no frame or is gone fails until it is
removed. Frames in other families are reported, not refused. --print-pending
prints the attention frames the artifacts carry as a PENDING literal. With
--kernel NAME (repeatable) the audit instead refuses a frame in every function
whose name contains one of the names, whatever its family; an empty name
matches every function, which is how a module's build rule refuses any frame
in it. The exit status is 0 when nothing is refused, 1 when something is, and
2 when an artifact cannot be read.
"""
import argparse
import re
import shutil
import subprocess
import sys

FUNCTION = re.compile(r'^\s*Function (\S+):')
RESOURCE = re.compile(r'^\s*REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)')
FAMILIES = (
    ('skinny', ('LmSkinny',)),
    ('cell', ('LmCell',)),
    ('stream GEMM', ('LmStreamGemm',)),
    ('attention', ('Attention', 'Attend', 'LmGqaShard', 'LmLatentShard', 'LmLatentWarpSplit', 'WindowFold', 'MergeSink')),
    ('tensor-map GEMM', ('LmGemm', 'LmMxGemm')),
    ('collective', ('SparkTp', 'StageRunner')),
)
REFUSED = ('skinny', 'cell', 'stream GEMM', 'attention')
PENDING_FAMILY = 'attention'
PENDING = {
    '_Z15D41AttendKernelI12LmKvGeometryILj1280ELj128ELb1EELj512ELj2ELb1EEv13LmKvShardViewjPKtmPKjS6_S6_S6_jS4_S4_mjS6_S6_S6_S6_S6_PKffPtmS8_PK6float2jjPhmjjj': 16,
    '_Z15D41AttendKernelI12LmKvGeometryILj1280ELj256ELb1EELj512ELj2ELb1EEv13LmKvShardViewjPKtmPKjS6_S6_S6_jS4_S4_mjS6_S6_S6_S6_S6_PKffPtmS8_PK6float2jjPhmjjj': 16,
}


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


def base_name(mangled):
    if not mangled.startswith('_Z'):
        return mangled
    index = 2
    if mangled[index:index + 1] == 'L':
        index += 1
    nested = mangled[index:index + 1] == 'N'
    if nested:
        index += 1
    name = None
    while index < len(mangled) and mangled[index].isdigit():
        end = index
        while end < len(mangled) and mangled[end].isdigit():
            end += 1
        length = int(mangled[index:end])
        name = mangled[end:end + length]
        index = end + length
        if not nested:
            break
    return name or mangled


def family(mangled):
    name = base_name(mangled)
    for label, tokens in FAMILIES:
        if any(token in name for token in tokens):
            return label
    return 'other'


def frames(found):
    kernels = {}
    for artifact, name, registers, stack, local in found:
        entry = kernels.setdefault(name, {'family': family(name), 'registers': 0, 'stack': 0, 'local': 0, 'artifacts': []})
        entry['registers'] = max(entry['registers'], registers)
        entry['stack'] = max(entry['stack'], stack)
        entry['local'] = max(entry['local'], local)
        if stack != 0 or local != 0:
            entry['artifacts'].append(artifact)
    return kernels


def judge(kernels, pending):
    refused, reported = [], []
    for name in sorted(kernels):
        entry = kernels[name]
        size = entry['stack'] + entry['local']
        if size == 0:
            continue
        if entry['family'] not in REFUSED:
            reported.append((name, f'{entry["family"]}: a frame of {size} bytes, reported'))
        elif entry['family'] == PENDING_FAMILY and name in pending:
            want = pending[name]
            if size > want:
                refused.append((name, f'{entry["family"]}: the frame grew from {want} to {size} bytes'))
            elif size < want:
                refused.append((name, f'{entry["family"]}: the frame fell from {want} to {size} bytes; lower its PENDING entry'))
            else:
                reported.append((name, f'{entry["family"]}: a frame of {size} bytes, pending'))
        else:
            refused.append((name, f'{entry["family"]}: a frame of {size} bytes in a hot-path family'))
    for name in sorted(pending):
        if name not in kernels:
            refused.append((name, f'{PENDING_FAMILY}: the PENDING entry is in no artifact; remove it'))
        elif kernels[name]['stack'] + kernels[name]['local'] == 0:
            refused.append((name, f'{PENDING_FAMILY}: the PENDING entry keeps no frame now; remove it'))
    return refused, reported


def demangled(names, filter_binary):
    if filter_binary is None or not names:
        return {name: name for name in names}
    result = subprocess.run([filter_binary], input='\n'.join(names), capture_output=True, text=True)
    lines = result.stdout.splitlines()
    if result.returncode != 0 or len(lines) != len(names):
        return {name: name for name in names}
    return dict(zip(names, lines))


def placement(entry):
    if entry is None:
        return ''
    return (f' uses {entry["registers"]} registers, stack {entry["stack"]} bytes, local {entry["local"]} bytes,'
            f' in {len(entry["artifacts"])} artifacts ({", ".join(sorted(entry["artifacts"]))})')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('artifacts', nargs='+')
    parser.add_argument('--cuobjdump', default='cuobjdump')
    parser.add_argument('--print-pending', action='store_true')
    parser.add_argument('--kernel', action='append')
    args = parser.parse_args()
    found = []
    for artifact in args.artifacts:
        result = subprocess.run([args.cuobjdump, '-res-usage', artifact], capture_output=True, text=True)
        if result.returncode != 0 and 'does not contain device code' in result.stderr:
            continue
        if result.returncode != 0:
            print(f'kernel_spill_audit: cuobjdump cannot read {artifact}: {result.stderr.strip()}', file=sys.stderr)
            return 2
        found += [(artifact,) + item for item in resources(result.stdout)]
    if args.kernel is not None:
        named, checked = [], 0
        for artifact in args.artifacts:
            listed, count = refusals([item[1:] for item in found if item[0] == artifact], args.kernel)
            named += [(artifact,) + item for item in listed]
            checked += count
        for artifact, name, registers, stack, local in named:
            print(f'REFUSED {artifact}: {name} uses {registers} registers and spills: stack {stack} bytes, local {local} bytes')
        print(f'kernel spill audit: {checked} kernels named {", ".join(args.kernel)} in {len(args.artifacts)} artifacts, {len(named)} spill')
        return 1 if named else 0
    kernels = frames(found)
    if args.print_pending:
        print('PENDING = {')
        for name in sorted(kernels):
            entry = kernels[name]
            if entry['family'] == PENDING_FAMILY and entry['stack'] + entry['local'] != 0:
                print(f"    '{name}': {entry['stack'] + entry['local']},")
        print('}')
        return 0
    refused, reported = judge(kernels, PENDING)
    readable = demangled(sorted({name for name, _ in refused + reported}), shutil.which('c++filt'))
    for name, verdict in reported:
        print(f'FRAME {verdict}: {readable[name]}{placement(kernels[name])}')
    for name, verdict in refused:
        print(f'REFUSED {verdict}: {readable[name]}{placement(kernels.get(name))}')
    counts = {}
    for name, _ in reported:
        counts[kernels[name]['family']] = counts.get(kernels[name]['family'], 0) + 1
    summary = ', '.join(f'{label} {count}' for label, count in sorted(counts.items())) or 'none'
    print(f'kernel spill audit: {len(kernels)} kernels in {len(args.artifacts)} artifacts; frames kept: {summary};'
          f' {len(PENDING)} pending; {len(refused)} refused')
    return 1 if refused else 0


if __name__ == '__main__':
    sys.exit(main())
