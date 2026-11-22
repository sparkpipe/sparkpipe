#!/usr/bin/env python3
"""Host check that every CUDA binary linking the runtime library is linked by the sm_121a compile gate.

libsparkpipe_runtime.a holds weightd's code, which reaches the device layer (SparkDeviceMemory*, SparkDeviceInitialize
and the rest) that lives in libsparkpipe_device_cuda.so. A binary that pulls those objects in and does not link the
device library builds every object and then fails to link, and the GPU tests are built only on a GB10, so such a rule
can break unseen. The rule here: every Makefile rule whose recipe links $(RUNTIME_LIBRARY) through $(NVCC) is listed in
STAGE_RUNNER_BINARIES (directly, through a later `+=`, or through a variable it names, such as $(DRAFTER_MODULES)),
which depend on the device library, and the CUDA sm_121a compile gate, which CI runs, builds and links every one of
them; the verify-frames tests, which pull weightd in, link $(DEVICE_CUDA_NVCC_LINK); and every rule that links the CUDA
device library depends on it, directly or through STAGE_RUNNER_BINARIES, so a target built on its own or early in a
parallel build finds the library.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEAD = re.compile(r'^([^\t#=][^=]*?):(?!=)')


def rules(makefile):
    found, current = [], None
    for line in makefile.split('\n'):
        if line.startswith('\t'):
            if current is not None:
                current[1].append(line)
            continue
        match = HEAD.match(line)
        if match is None or re.match(r'^[A-Za-z0-9_]+\s*[:?+]?=', line) or line.split()[0] in ('define', 'endef', 'ifeq', 'ifneq', 'else', 'endif'):
            current = None if line.strip() else current
            continue
        current = (match.group(1).strip(), [], line[match.end():])
        found.append(current)
    return found


def expanded(makefile, recipe):
    text = ' '.join(recipe)
    for name, body in re.findall(r'^define (\S+)\n(.*?)\nendef', makefile, re.M | re.S):
        if '$(call ' + name + ',' in text or '$(' + name + ')' in text:
            text += ' ' + body
    return text


def definitions(makefile, variable):
    values = []
    for match in re.finditer(r'^' + re.escape(variable) + r'\s*(?::|\+|\?)?=[ \t]*((?:.*\\\n)*.*)$', makefile, re.M):
        values += match.group(1).replace('\\\n', ' ').split()
    return values


def listed(makefile, variable):
    found, pending = set(), definitions(makefile, variable)
    while pending:
        token = pending.pop()
        if token in found:
            continue
        found.add(token)
        reference = re.fullmatch(r'\$\(([A-Za-z0-9_]+)\)', token)
        if reference is not None:
            pending += definitions(makefile, reference.group(1))
    return found


def main():
    failures = []
    makefile = (ROOT / 'Makefile').read_text()
    stage_runner = listed(makefile, 'STAGE_RUNNER_BINARIES')
    if not stage_runner:
        failures.append('the Makefile defines no STAGE_RUNNER_BINARIES')
    if '$(STAGE_RUNNER_BINARIES): $(DEVICE_CUDA_LIBRARY)' not in makefile:
        failures.append('STAGE_RUNNER_BINARIES no longer depend on the CUDA device library')
    linked = 0
    for target, recipe, prerequisites in rules(makefile):
        text = expanded(makefile, recipe)
        if ('$(DEVICE_CUDA_NVCC_LINK)' in text or '-lsparkpipe_device_cuda' in text) and '$(DEVICE_CUDA_LIBRARY)' not in prerequisites \
                and target not in stage_runner:
            failures.append(f'{target} links the CUDA device library but does not depend on $(DEVICE_CUDA_LIBRARY)')
        if '$(NVCC)' not in text or '$(RUNTIME_LIBRARY)' not in text:
            continue
        linked += 1
        if target not in stage_runner:
            failures.append(f'{target} links $(RUNTIME_LIBRARY) through $(NVCC) but is not in STAGE_RUNNER_BINARIES, so the sm_121a gate never links it')
        if target.endswith('_verify_frames') or target.endswith('_verify_frames_bf16'):
            if '$(DEVICE_CUDA_NVCC_LINK)' not in text:
                failures.append(f'{target} pulls weightd in from the runtime library but does not link $(DEVICE_CUDA_NVCC_LINK)')
    if linked < 10:
        failures.append(f'only {linked} rules link $(RUNTIME_LIBRARY) through $(NVCC); the rule parser no longer reads the Makefile')
    gate = (ROOT / 'tools' / 'cuda13_sm121a_compile_gate.sh').read_text()
    if 'make_variable STAGE_RUNNER_BINARIES' not in gate or '"${stage_runner_binaries[@]}"' not in gate:
        failures.append('the sm_121a compile gate does not build STAGE_RUNNER_BINARIES')
    for failure in failures:
        print('FAIL ' + failure)
    if not failures:
        print(f'PASS stage-runner link rules: {linked} CUDA rules link the runtime library, all of them are linked by the sm_121a gate')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
