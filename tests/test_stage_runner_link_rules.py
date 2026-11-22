#!/usr/bin/env python3
"""Host check that every CUDA binary linking the runtime library is linked by a CI build.

libsparkpipe_runtime.a holds weightd's code, which reaches the device layer (SparkDeviceMemory*, SparkDeviceInitialize
and the rest) that lives in libsparkpipe_device_cuda.so. A binary that pulls those objects in and does not link the
device library builds every object and then fails to link, and the GPU tests are built only on a GB10, so such a rule
can break unseen. A recipe links the runtime library when it names $(RUNTIME_LIBRARY) or any variable one of whose
definitions reaches it, such as $(ADAPTER_LIBRARIES), followed to a fixed point. A rule head that names several targets
is checked one target at a time. The rules here:
- every CUDA target whose nvcc recipe links the runtime library is in STAGE_RUNNER_BINARIES (directly, through a later
  `+=`, or through a variable it names, such as $(DRAFTER_MODULES)), which depend on the device library, and the CUDA
  sm_121a compile gate, which CI runs, builds and links every one of them; and every entry of the list is such a
  target, so a parser that stops seeing a rule fails instead of passing with fewer rules;
- every nvrm twin in a rule head ($(call NVRM_TWIN,...)) is in NVRM_SERVING_ADAPTERS. A twin of an adapter in
  NVRM_SERVING_ADAPTER_SOURCES is built and linked by the hosted release gate (--complete) in the same CI job; the
  others are the proof twins in PROOF_TWINS, which `make nvrm-adapters` builds for the nvrm proofs on a GB10;
- the verify-frames tests, which pull weightd in, link $(DEVICE_CUDA_NVCC_LINK); and every rule that links the CUDA
  device library depends on it, directly or through STAGE_RUNNER_BINARIES, so a target built on its own or early in a
  parallel build finds the library.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEAD = re.compile(r'^([^\t#=][^=]*?):(?!=)')
DEFINITION = re.compile(r'^(?:[^\t#\n=]*?:[ \t]*)?([A-Za-z0-9_]+)\s*(?::|\+|\?)?=[ \t]*((?:.*\\\n)*.*)$', re.M)
TWIN = re.compile(r'^\$\(call NVRM_TWIN,(.+)\)$')
PROOF_TWINS = {
    'build/libqwen38_27b_bf16_tp1_serving_adapter.$(SHARED_LIBRARY_EXT)': 'the Qwen3.8 27B BF16 TP1 nvrm proof, built by make nvrm-adapters',
    'build/libdsv41_flash_tp4_serving_adapter.$(SHARED_LIBRARY_EXT)': 'the DeepSeek V4.1 Flash TP4 nvrm proof, built by make nvrm-adapters',
}


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


def reaching(makefile, root):
    names, changed = {root}, True
    while changed:
        changed = False
        for match in DEFINITION.finditer(makefile):
            name, value = match.group(1), match.group(2)
            if name not in names and any('$(' + known + ')' in value for known in names):
                names.add(name)
                changed = True
    return names


def targets(head):
    found, depth, current = [], 0, ''
    for character in head:
        if character == '(':
            depth += 1
        elif character == ')':
            depth -= 1
        if character.isspace() and depth == 0:
            if current:
                found.append(current)
            current = ''
            continue
        current += character
    if current:
        found.append(current)
    return found


def flattened(makefile, variable):
    entries, pending = [], definitions(makefile, variable)
    while pending:
        token = pending.pop(0)
        reference = re.fullmatch(r'\$\(([A-Za-z0-9_]+)\)', token)
        values = definitions(makefile, reference.group(1)) if reference is not None else []
        if len(values) > 1:
            pending = values + pending
        else:
            entries.append(token)
    return entries


def twinned(makefile):
    match = re.search(r'^NVRM_SERVING_ADAPTERS := \$\(call NVRM_TWIN,(.*)\)$', makefile, re.M)
    if match is None:
        return set()
    originals = set()
    for token in match.group(1).split():
        reference = re.fullmatch(r'\$\(([A-Za-z0-9_]+)\)', token)
        originals |= {token} | (listed(makefile, reference.group(1)) if reference is not None else set())
    return originals


def main():
    failures = []
    makefile = (ROOT / 'Makefile').read_text()
    stage_runner = listed(makefile, 'STAGE_RUNNER_BINARIES')
    if not stage_runner:
        failures.append('the Makefile defines no STAGE_RUNNER_BINARIES')
    if '$(STAGE_RUNNER_BINARIES): $(DEVICE_CUDA_LIBRARY)' not in makefile:
        failures.append('STAGE_RUNNER_BINARIES no longer depend on the CUDA device library')
    linking = reaching(makefile, 'RUNTIME_LIBRARY')
    hosted = listed(makefile, 'NVRM_SERVING_ADAPTER_SOURCES')
    twin_sources = twinned(makefile)
    if 'ADAPTER_LIBRARIES' not in linking:
        failures.append('$(ADAPTER_LIBRARIES) no longer reaches $(RUNTIME_LIBRARY); the adapter rules are not followed')
    linked, twins = set(), set()
    for head, recipe, prerequisites in rules(makefile):
        text = expanded(makefile, recipe)
        links = '$(NVCC)' in text and any('$(' + name + ')' in text for name in linking)
        for target in targets(head):
            twin = TWIN.match(target)
            if twin is not None:
                if links:
                    twins.add(twin.group(1))
                continue
            if ('$(DEVICE_CUDA_NVCC_LINK)' in text or '-lsparkpipe_device_cuda' in text) and '$(DEVICE_CUDA_LIBRARY)' not in prerequisites \
                    and target not in stage_runner:
                failures.append(f'{target} links the CUDA device library but does not depend on $(DEVICE_CUDA_LIBRARY)')
            if not links:
                continue
            linked.add(target)
            if target not in stage_runner:
                failures.append(f'{target} links $(RUNTIME_LIBRARY) through $(NVCC) but is not in STAGE_RUNNER_BINARIES, so the sm_121a gate never links it')
            if target.endswith('_verify_frames') or target.endswith('_verify_frames_bf16'):
                if '$(DEVICE_CUDA_NVCC_LINK)' not in text:
                    failures.append(f'{target} pulls weightd in from the runtime library but does not link $(DEVICE_CUDA_NVCC_LINK)')
    for entry in flattened(makefile, 'STAGE_RUNNER_BINARIES'):
        if entry not in linked:
            failures.append(f'STAGE_RUNNER_BINARIES lists {entry}, but no nvcc rule linking the runtime library builds it; the rule parser has lost it')
    for original in sorted(twins):
        if original not in twin_sources:
            failures.append(f'$(call NVRM_TWIN,{original}) has a rule but is not in NVRM_SERVING_ADAPTERS')
        elif original not in hosted and original not in PROOF_TWINS:
            failures.append(f'$(call NVRM_TWIN,{original}) is built by neither the hosted release gate nor a stated proof')
    for original in sorted(PROOF_TWINS):
        if original not in twins or original in hosted:
            failures.append(f'PROOF_TWINS names {original}, which is not a twin outside the hosted release gate; remove it')
    gate = (ROOT / 'tools' / 'cuda13_sm121a_compile_gate.sh').read_text()
    if 'make_variable STAGE_RUNNER_BINARIES' not in gate or '"${stage_runner_binaries[@]}"' not in gate:
        failures.append('the sm_121a compile gate does not build STAGE_RUNNER_BINARIES')
    for failure in failures:
        print('FAIL ' + failure)
    if not failures:
        print(f'PASS stage-runner link rules: {len(linked)} CUDA targets link the runtime library, all of them are linked by the sm_121a gate; '
              f'{len(twins)} nvrm twins, {len(twins & hosted)} built by the hosted release gate and {len(twins - hosted)} stated proofs')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
