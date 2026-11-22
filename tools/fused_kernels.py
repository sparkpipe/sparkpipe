#!/usr/bin/env python3
"""Generate fused kernels, their launchers, their exactness test and their manifest from a fusion plan.

    python3 -I tools/fused_kernels.py PLAN [--check]

A plan (model-families/<model>/fusion/<step>.json) lists fused kernels. Each
kernel is a set of roles: every role gives each row of the step a fixed number
of thread blocks that run one op of the op catalogue below, and may end in a
tail that the row's last arriving block runs after LmFusedLastArrival. Ops are
the bodies of the engine's own kernels (device functions shared with them), so
a fused kernel computes every value as the kernels it replaces do; the
generated test compares the two byte for byte. No op waits for a block that may
not have started. --check regenerates every output in memory and fails when a
committed file differs.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAME = re.compile(r'^[A-Za-z_][A-Za-z0-9_]{0,63}$')
FILLS = ('bf16', 'f32', 'zero')
SHARED_ROOTS = ('include/', 'runtime/', 'inference/', 'node/', 'cache/', 'serving/', 'src/', 'common/', 'model-families/common/')
OPS = {
    'hc_sinkhorn_partials': {
        'body': 'LmHcSinkhornPartialTask',
        'source': 'inference/kernels/hc.cuh',
        'replaces': 'LmHcSinkhornPartialKernel',
        'parameters': ('streams_count', 'streams', 'fn', 'partials', 'width'),
        'call': 'LmHcSinkhornPartialTask<{streams_count}>({streams}, {fn}, {partials}, {width}, item, row);',
        'lanes': 0,
    },
    'hc_sinkhorn_row': {
        'body': 'LmHcSinkhornRow',
        'source': 'inference/kernels/hc.cuh',
        'replaces': 'LmHcSinkhornKernel',
        'parameters': ('streams_count', 'partials', 'scale', 'base', 'pre', 'post', 'comb', 'width', 'rms_epsilon', 'iterations', 'epsilon'),
        'call': ('LmHcSinkhornRow<{streams_count}>({partials}, {scale}, {base}, {pre}, {post}, {comb}, {width}, {rms_epsilon}, {iterations}, '
                 '{epsilon}, row, threadIdx.x, {shared});'),
        'lanes': 32,
    },
    'hc_collapse_norm': {
        'body': 'LmHcCollapseNormRow',
        'source': 'inference/kernels/fused/collapse_norm.cuh',
        'replaces': 'a row norm of collapsed streams',
        'parameters': ('streams_count', 'lane', 'streams', 'pre', 'weight', 'output', 'quantized', 'width', 'epsilon'),
        'call': ('LmHcCollapseNormRow<{threads}, {streams_count}, {lane}>((const uint16_t *)0, {streams}, {pre}, {weight}, {output}, {quantized}, '
                 'row, {width}, item, {blocks_per_row}, {epsilon}, {shared});'),
        'lanes': 0,
    },
}


class PlanError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise PlanError(message)


def check_name(value, what):
    require(isinstance(value, str) and NAME.match(value) is not None, f'{what} {value!r} is not a C identifier of at most 64 characters')
    return value


def check_output(path, what):
    require(isinstance(path, str) and not path.startswith('/') and '..' not in Path(path).parts, f'{what} {path!r} is not a repository path')
    require(not path.startswith(SHARED_ROOTS), f'{what} {path} is in shared code, where a model plan may not write')
    require(path.startswith(('model-families/', 'tests/')), f'{what} {path} is outside the model families and the tests')
    require(len(Path(path).name) <= 64, f'{what} {path} has a file name over 64 characters')
    return path


def load(path):
    plan = json.loads(Path(path).read_text())
    require(isinstance(plan, dict), 'a plan is a JSON object')
    for key in ('header', 'test', 'manifest'):
        check_output(plan.get(key), key)
    require(isinstance(plan.get('includes'), list) and all(isinstance(item, str) for item in plan['includes']), 'includes is a list of paths')
    require(isinstance(plan.get('test_includes'), list) and all(isinstance(item, str) for item in plan['test_includes']), 'test_includes is a list of paths')
    require(isinstance(plan.get('kernels'), list) and plan['kernels'], 'a plan has at least one kernel')
    seen = set()
    for kernel in plan['kernels']:
        for key in ('name', 'launcher', 'arguments_type', 'rows_constant'):
            check_name(kernel.get(key), key)
            require(kernel[key] not in seen, f'{kernel[key]} is defined twice')
            seen.add(kernel[key])
        require(isinstance(kernel.get('threads'), int) and kernel['threads'] % 32 == 0 and 32 <= kernel['threads'] <= 1024,
                f'{kernel["name"]}: threads is a multiple of 32 from 32 to 1024')
        require(isinstance(kernel.get('rows_max'), int) and 1 <= kernel['rows_max'] <= 16, f'{kernel["name"]}: rows_max is 1 to 16')
        require(isinstance(kernel.get('shared_bytes'), str), f'{kernel["name"]}: shared_bytes is a C expression')
        names = set()
        for argument in kernel.get('arguments', []):
            require(isinstance(argument, dict), f'{kernel["name"]}: an argument is an object')
            check_name(argument.get('name'), 'argument')
            require(argument['name'] not in names, f'{kernel["name"]}: argument {argument["name"]} twice')
            names.add(argument['name'])
            require(argument.get('type') in ('const uint16_t *', 'uint16_t *', 'const float *', 'float *', 'uint32_t *'),
                    f'{kernel["name"]}: argument {argument["name"]} has an unknown type')
            require(argument.get('fill') in FILLS, f'{kernel["name"]}: argument {argument["name"]} fill is one of {FILLS}')
            require(isinstance(argument.get('elements'), str), f'{kernel["name"]}: argument {argument["name"]} needs an element count expression')
        require(names, f'{kernel["name"]}: no arguments')
        counters = next((argument for argument in kernel['arguments'] if argument['name'] == 'counters'), None)
        require(counters is None or (counters['type'] == 'uint32_t *' and counters['fill'] == 'zero'),
                f'{kernel["name"]}: counters are uint32_t * and start at zero')
        require(isinstance(kernel.get('roles'), list) and kernel['roles'], f'{kernel["name"]}: at least one role')
        for role in kernel['roles']:
            check_op(kernel, role, names, False)
            if 'tail' in role:
                require('counters' in names, f'{kernel["name"]}: a role with a tail needs the counters argument')
                require(role['tail'].get('scope') == 'row', f'{kernel["name"]}: a tail runs for its row')
                check_op(kernel, role['tail'], names, True)
        require(isinstance(kernel.get('perturb'), str) and kernel['perturb'] in names, f'{kernel["name"]}: perturb names an input argument')
        require(isinstance(kernel.get('reference'), list) and kernel['reference'], f'{kernel["name"]}: the reference calls are listed')
        require(isinstance(kernel.get('outputs'), list) and set(kernel['outputs']) <= names and kernel['outputs'],
                f'{kernel["name"]}: outputs name arguments')
        require(isinstance(kernel.get('replaces'), list) and kernel['replaces'], f'{kernel["name"]}: replaces lists the kernels it removes')
    return plan


def check_op(kernel, role, names, tail):
    op = OPS.get(role.get('op'))
    require(op is not None, f'{kernel["name"]}: unknown op {role.get("op")!r}')
    bind = role.get('bind', {})
    require(isinstance(bind, dict), f'{kernel["name"]}: bind is an object')
    missing = [name for name in op['parameters'] if name not in bind]
    require(not missing, f'{kernel["name"]}: op {role["op"]} needs {", ".join(missing)}')
    extra = [name for name in bind if name not in op['parameters']]
    require(not extra, f'{kernel["name"]}: op {role["op"]} takes no {", ".join(extra)}')
    require(tail or isinstance(role.get('blocks_per_row'), str), f'{kernel["name"]}: role {role["op"]} gives its blocks per row as a C expression')
    for value in bind.values():
        require(isinstance(value, str), f'{kernel["name"]}: bind values are C expressions')
        for token in re.findall(r'\barguments\.(\w+)', value):
            require(token in names, f'{kernel["name"]}: {token} is not an argument')


def render(template, bind, shared):
    values = dict(bind)
    values['shared'] = shared
    return template.format(**values)


def emit_header(plan):
    lines = ['#pragma once', '']
    lines += [f'#include "{path}"' for path in plan['includes']]
    lines.append('')
    for kernel in plan['kernels']:
        name, threads, shared = kernel['name'], kernel['threads'], f'{kernel["name"]}Shared'
        lines.append(f'#define {kernel["rows_constant"]} {kernel["rows_max"]}u')
        lines.append('')
        lines.append(f'typedef struct {kernel["arguments_type"]}')
        lines.append('{')
        for argument in kernel['arguments']:
            lines.append(f'\t{argument["type"]}{argument["name"]};')
        lines.append(f'}} {kernel["arguments_type"]};')
        lines.append('')
        lines.append(f'__global__ __launch_bounds__({threads}) void {name}(const __grid_constant__ {kernel["arguments_type"]} arguments, uint32_t rows)')
        lines.append('{')
        lines.append(f'\textern __shared__ float {shared}[];')
        lines.append('\tuint32_t block = blockIdx.x;')
        for role in kernel['roles']:
            per_row = role['blocks_per_row']
            lines.append(f'\tif ( block < rows * ({per_row}) )')
            lines.append('\t{')
            lines.append(f'\t\tconst uint32_t row = block / ({per_row}), item = block % ({per_row});')
            lines.append('\t\t' + render(OPS[role['op']]['call'], dict(role['bind'], threads=f'{threads}u', blocks_per_row=f'({per_row})'), shared))
            if 'tail' in role:
                tail = role['tail']
                lanes = OPS[tail['op']]['lanes']
                guard = f' && threadIdx.x < {lanes}u' if lanes else ''
                lines.append(f'\t\tif ( LmFusedLastArrival(arguments.counters + row, {per_row}){guard} )')
                lines.append('\t\t\t' + render(OPS[tail['op']]['call'], dict(tail['bind'], threads=f'{threads}u', blocks_per_row=f'({per_row})'), shared))
            lines.append('\t\treturn;')
            lines.append('\t}')
            if role is not kernel['roles'][-1]:
                lines.append(f'\tblock -= rows * ({per_row});')
        lines.append('}')
        lines.append('')
        total = ' + '.join(f'({role["blocks_per_row"]})' for role in kernel['roles'])
        lines.append(f'static inline int32_t {kernel["launcher"]}(const {kernel["arguments_type"]} *arguments, uint32_t rows, SparkGpuStream stream)')
        lines.append('{')
        lines.append(f'\tif ( arguments == 0 || rows == 0u || rows > {kernel["rows_constant"]} )')
        lines.append('\t\treturn LM_LAUNCH_ERR_SHAPE;')
        lines.append(f'\tLM_LAUNCH(({name}), rows * ({total}), {threads}u, {kernel["shared_bytes"]}, stream, *arguments, rows);')
        lines.append('\treturn SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;')
        lines.append('}')
        lines.append('')
    return '\n'.join(lines).rstrip('\n') + '\n'


def emit_test(plan):
    lines = ['#include <cuda_runtime.h>', '#include <stdint.h>', '#include <stdio.h>', '#include <stdlib.h>', '#include <string.h>', '#include <vector>', '']
    lines += [f'#include "{path}"' for path in plan['test_includes']]
    lines += ['', '#define FUSED_REQUIRE(condition) do { if ( !(condition) ) { fprintf(stderr, "FAIL %s:%d %s\\n", __FILE__, __LINE__, #condition); exit(1); } } while ( 0 )',
              '#define FUSED_CUDA(call) FUSED_REQUIRE((call) == cudaSuccess)', '',
              'static __device__ __forceinline__ uint64_t FusedMix(uint64_t value)', '{', '\tvalue += 0x9e3779b97f4a7c15ull;',
              '\tvalue = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;', '\tvalue = (value ^ (value >> 27)) * 0x94d049bb133111ebull;',
              '\treturn value ^ (value >> 31);', '}', '',
              'static __global__ void FusedFillKernel(uint8_t *data, uint64_t bytes, uint64_t seed, uint32_t kind, uint32_t edges)', '{',
              '\tfor ( uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < bytes; index += (uint64_t)gridDim.x * blockDim.x )',
              '\t{', '\t\tuint8_t byte = 0u;', '\t\tif ( kind == 1u )', '\t\t{',
              '\t\t\tconst uint64_t element = index >> 1, random = FusedMix(seed ^ element);',
              '\t\t\tuint32_t value = (uint32_t)(random & 0x8000u) | ((118u + (uint32_t)((random >> 16) % 12u)) << 7) | (uint32_t)((random >> 24) & 0x7fu);',
              '\t\t\tif ( edges != 0u && element % 97u == 0u )', '\t\t\t\tvalue = (uint32_t)(random & 0x8000u) | 0x46eau;',
              '\t\t\tif ( edges != 0u && element % 89u == 0u )', '\t\t\t\tvalue = 0u;',
              '\t\t\tbyte = (uint8_t)(value >> (8u * (uint32_t)(index & 1u)));', '\t\t}', '\t\telse if ( kind == 2u )', '\t\t{',
              '\t\t\tconst uint64_t random = FusedMix(seed ^ (index >> 2));',
              '\t\t\tconst uint32_t value = (uint32_t)(random & 0x80000000u) | ((116u + (uint32_t)((random >> 32) % 12u)) << 23) | (uint32_t)((random >> 8) & 0x7fffffu);',
              '\t\t\tbyte = (uint8_t)(value >> (8u * (uint32_t)(index & 3u)));', '\t\t}', '\t\tdata[index] = byte;', '\t}', '}', '']
    for kernel in plan['kernels']:
        name, rows_max = kernel['name'], kernel['rows_max']
        args = kernel['arguments']
        lines.append(f'static void {name}Buffers({kernel["arguments_type"]} *set, uint64_t seed, uint32_t edges)')
        lines.append('{')
        lines.append(f'\tconst uint32_t rows = {rows_max}u;')
        for index, argument in enumerate(args):
            base = argument['type'].replace('const ', '').replace(' *', '')
            lines.append(f'\t{{')
            lines.append(f'\t\tconst uint64_t bytes = (uint64_t)({argument["elements"]}) * sizeof({base});')
            lines.append(f'\t\tvoid *pointer = 0;')
            lines.append(f'\t\tFUSED_CUDA(cudaMalloc(&pointer, bytes));')
            if argument['fill'] == 'zero':
                lines.append(f'\t\tFUSED_CUDA(cudaMemset(pointer, 0, bytes));')
            else:
                kind = 1 if argument['fill'] == 'bf16' else 2
                lines.append(f'\t\tFusedFillKernel<<<256, 256>>>((uint8_t *)pointer, bytes, seed + {index}u, {kind}u, edges);')
            lines.append(f'\t\tset->{argument["name"]} = ({argument["type"]})pointer;')
            lines.append('\t\t(void)rows;')
            lines.append(f'\t}}')
        lines.append('\tFUSED_CUDA(cudaDeviceSynchronize());')
        lines.append('}')
        lines.append('')
        lines.append(f'static void {name}Copy({kernel["arguments_type"]} *to, const {kernel["arguments_type"]} *from)')
        lines.append('{')
        lines.append(f'\tconst uint32_t rows = {rows_max}u;')
        for argument in args:
            base = argument['type'].replace('const ', '').replace(' *', '')
            lines.append(f'\tFUSED_CUDA(cudaMemcpy((void *)to->{argument["name"]}, from->{argument["name"]}, (uint64_t)({argument["elements"]}) * sizeof({base}), '
                         'cudaMemcpyDeviceToDevice));')
        lines.append('\tFUSED_CUDA(cudaDeviceSynchronize());')
        lines.append('\t(void)rows;')
        lines.append('}')
        lines.append('')
        lines.append(f'static void {name}Reference(const {kernel["arguments_type"]} *set, uint32_t rows, cudaStream_t stream)')
        lines.append('{')
        for argument in args:
            lines.append(f'\t{argument["type"]}{argument["name"]} = set->{argument["name"]};')
        for argument in args:
            lines.append(f'\t(void){argument["name"]};')
        for call in kernel['reference']:
            lines.append(f'\t{call}')
        lines.append('\tFUSED_CUDA(cudaGetLastError());')
        lines.append('}')
        lines.append('')
        lines.append(f'static uint64_t {name}Differ(const {kernel["arguments_type"]} *one, const {kernel["arguments_type"]} *two, uint32_t rows, uint32_t first_row_only, '
                     'uint32_t report)')
        lines.append('{')
        lines.append('\tuint64_t differ = 0u;')
        for output in kernel['outputs']:
            argument = next(item for item in args if item['name'] == output)
            base = argument['type'].replace('const ', '').replace(' *', '')
            lines.append('\t{')
            lines.append(f'\t\tconst uint64_t bytes = first_row_only != 0u ? (uint64_t)({argument["elements"]}) * sizeof({base}) / rows : '
                         f'(uint64_t)({argument["elements"]}) * sizeof({base});')
            lines.append('\t\tstd::vector<uint8_t> left(bytes), right(bytes);')
            lines.append(f'\t\tFUSED_CUDA(cudaMemcpy(left.data(), one->{output}, bytes, cudaMemcpyDeviceToHost));')
            lines.append(f'\t\tFUSED_CUDA(cudaMemcpy(right.data(), two->{output}, bytes, cudaMemcpyDeviceToHost));')
            lines.append('\t\tuint64_t count = 0u;')
            lines.append('\t\tfor ( uint64_t index = 0u; index < bytes; ++index )')
            lines.append('\t\t\tcount += left[index] != right[index] ? 1u : 0u;')
            lines.append('\t\tif ( report != 0u && count != 0u )')
            lines.append(f'\t\t\tfprintf(stderr, "{output}: %llu of %llu bytes differ\\n", (unsigned long long)count, (unsigned long long)bytes);')
            lines.append('\t\tdiffer += count;')
            lines.append('\t}')
        lines.append('\treturn differ;')
        lines.append('}')
        lines.append('')
        lines.append(f'static void {name}Poison(const {kernel["arguments_type"]} *set, cudaStream_t stream)')
        lines.append('{')
        lines.append(f'\tconst uint32_t rows = {rows_max}u;')
        for output in kernel['outputs']:
            argument = next(item for item in args if item['name'] == output)
            base = argument['type'].replace('const ', '').replace(' *', '')
            lines.append(f'\tFUSED_CUDA(cudaMemsetAsync((void *)set->{output}, 0xff, (uint64_t)({argument["elements"]}) * sizeof({base}), stream));')
        lines.append('\t(void)rows;')
        lines.append('}')
        lines.append('')
        lines.append(f'static uint32_t {name}Exact(void)')
        lines.append('{')
        lines.append(f'\t{kernel["arguments_type"]} reference, fused, origin;')
        lines.append('\tcudaStream_t stream = 0;')
        lines.append('\tuint32_t checks = 0u;')
        lines.append('\tFUSED_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));')
        lines.append('\tfor ( uint32_t edges = 0u; edges < 2u; ++edges )')
        lines.append('\t{')
        lines.append(f'\t\t{name}Buffers(&origin, 0x5000u + 0x100u * edges, edges);')
        lines.append(f'\t\t{name}Buffers(&reference, 0x7000u, 0u);')
        lines.append(f'\t\t{name}Buffers(&fused, 0x7000u, 0u);')
        lines.append(f'\t\tfor ( uint32_t rows = 1u; rows <= {rows_max}u; ++rows )')
        lines.append('\t\t{')
        lines.append(f'\t\t\t{name}Copy(&reference, &origin);')
        lines.append(f'\t\t\t{name}Copy(&fused, &origin);')
        lines.append(f'\t\t\t{name}Reference(&reference, rows, stream);')
        lines.append(f'\t\t\tFUSED_REQUIRE({kernel["launcher"]}(&fused, rows, stream) == LM_LAUNCH_OK);')
        lines.append('\t\t\tFUSED_CUDA(cudaStreamSynchronize(stream));')
        lines.append(f'\t\t\tconst uint64_t differ = {name}Differ(&reference, &fused, rows, 0u, 1u);')
        lines.append(f'\t\t\tif ( differ != 0u )')
        lines.append(f'\t\t\t\t{{ fprintf(stderr, "FAIL {name} rows=%u edges=%u: %llu bytes differ\\n", rows, edges, (unsigned long long)differ); exit(1); }}')
        lines.append('\t\t\t++checks;')
        lines.append('\t\t}')
        lines.append('\t}')
        lines.append(f'\t{name}Copy(&reference, &origin);')
        lines.append(f'\t{name}Copy(&fused, &origin);')
        lines.append(f'\tFUSED_REQUIRE({kernel["launcher"]}(&reference, 1u, stream) == LM_LAUNCH_OK);')
        lines.append(f'\tFUSED_REQUIRE({kernel["launcher"]}(&fused, {rows_max}u, stream) == LM_LAUNCH_OK);')
        lines.append('\tFUSED_CUDA(cudaStreamSynchronize(stream));')
        lines.append(f'\tFUSED_REQUIRE({name}Differ(&reference, &fused, 1u, 1u, 1u) == 0u);')
        lines.append('\t++checks;')
        lines.append(f'\t{name}Copy(&reference, &origin);')
        lines.append(f'\t{name}Copy(&fused, &origin);')
        lines.append(f'\t{{')
        lines.append(f'\t\tconst uint16_t bump = 0x3f80u;')
        lines.append(f'\t\tFUSED_CUDA(cudaMemcpy((void *)fused.{kernel["perturb"]}, &bump, sizeof(bump), cudaMemcpyHostToDevice));')
        lines.append(f'\t\tFUSED_CUDA(cudaDeviceSynchronize());')
        lines.append(f'\t}}')
        lines.append(f'\tFUSED_REQUIRE({kernel["launcher"]}(&reference, {rows_max}u, stream) == LM_LAUNCH_OK);')
        lines.append(f'\tFUSED_REQUIRE({kernel["launcher"]}(&fused, {rows_max}u, stream) == LM_LAUNCH_OK);')
        lines.append('\tFUSED_CUDA(cudaStreamSynchronize(stream));')
        lines.append(f'\tFUSED_REQUIRE({name}Differ(&reference, &fused, {rows_max}u, 0u, 0u) != 0u);')
        lines.append('\t++checks;')
        lines.append(f'\t{name}Copy(&reference, &origin);')
        lines.append(f'\t{name}Copy(&fused, &origin);')
        lines.append('\t{')
        lines.append('\t\tcudaGraph_t graph = 0;')
        lines.append('\t\tcudaGraphExec_t exec = 0;')
        lines.append(f'\t\tFUSED_REQUIRE({kernel["launcher"]}(&reference, {rows_max}u, stream) == LM_LAUNCH_OK);')
        lines.append('\t\tFUSED_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));')
        lines.append(f'\t\tFUSED_REQUIRE({kernel["launcher"]}(&fused, {rows_max}u, stream) == LM_LAUNCH_OK);')
        lines.append('\t\tFUSED_CUDA(cudaStreamEndCapture(stream, &graph));')
        lines.append('\t\tFUSED_CUDA(cudaGraphInstantiate(&exec, graph, 0));')
        lines.append('\t\tfor ( uint32_t round = 0u; round < 3u; ++round )')
        lines.append('\t\t{')
        lines.append(f'\t\t\t{name}Poison(&fused, stream);')
        lines.append('\t\t\tFUSED_CUDA(cudaGraphLaunch(exec, stream));')
        lines.append('\t\t}')
        lines.append('\t\tFUSED_CUDA(cudaStreamSynchronize(stream));')
        lines.append(f'\t\tFUSED_REQUIRE({name}Differ(&reference, &fused, {rows_max}u, 0u, 1u) == 0u);')
        if any('tail' in role for role in kernel['roles']):
            lines.append('\t\t{')
            lines.append(f'\t\t\tconst uint32_t rows = {rows_max}u;')
            lines.append('\t\t\tstd::vector<uint32_t> counters((uint64_t)(' + next(item for item in args if item['name'] == 'counters')['elements'] + '));')
            lines.append('\t\t\tFUSED_CUDA(cudaMemcpy(counters.data(), fused.counters, counters.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));')
            lines.append('\t\t\tfor ( uint32_t counter : counters )')
            lines.append('\t\t\t\tFUSED_REQUIRE(counter == 0u);')
            lines.append('\t\t}')
        lines.append('\t\tFUSED_CUDA(cudaGraphExecDestroy(exec));')
        lines.append('\t\tFUSED_CUDA(cudaGraphDestroy(graph));')
        lines.append('\t}')
        lines.append('\t++checks;')
        lines.append(f'\tprintf("PASS {name}: %u comparisons\\n", checks);')
        lines.append('\treturn checks;')
        lines.append('}')
        lines.append('')
    lines.append('int main(void)')
    lines.append('{')
    lines.append('\tFUSED_CUDA(cudaSetDevice(0));')
    for kernel in plan['kernels']:
        lines.append(f'\t(void){kernel["name"]}Exact();')
    lines.append('\tprintf("PASS fused kernels\\n");')
    lines.append('\treturn 0;')
    lines.append('}')
    return '\n'.join(lines) + '\n'


def emit_manifest(plan, plan_text):
    sources = sorted({OPS[role['op']]['source'] for kernel in plan['kernels'] for role in kernel['roles']} |
                     {OPS[role['tail']['op']]['source'] for kernel in plan['kernels'] for role in kernel['roles'] if 'tail' in role})
    manifest = {'plan_sha256': hashlib.sha256(plan_text.encode()).hexdigest(), 'header': plan['header'], 'test': plan['test'], 'kernels': []}
    for kernel in plan['kernels']:
        roles = []
        for role in kernel['roles']:
            entry = {'op': role['op'], 'body': OPS[role['op']]['body'], 'replaces': OPS[role['op']]['replaces'], 'blocks_per_row': role['blocks_per_row']}
            if 'tail' in role:
                entry['tail'] = {'op': role['tail']['op'], 'body': OPS[role['tail']['op']]['body'], 'replaces': OPS[role['tail']['op']]['replaces'],
                                 'sync': 'last arrival of the row'}
            roles.append(entry)
        manifest['kernels'].append({'name': kernel['name'], 'launcher': kernel['launcher'], 'rows': [1, kernel['rows_max']], 'threads': kernel['threads'],
                                    'replaces': kernel['replaces'], 'roles': roles, 'outputs': kernel['outputs'],
                                    'patchable': {'rows': 'parameter 1, and the grid width rows * (sum of blocks per row)'}})
    manifest['op_sources'] = sources
    return json.dumps(manifest, indent=1, sort_keys=True) + '\n'


def outputs(plan_path):
    plan_text = Path(plan_path).read_text()
    plan = load(plan_path)
    return {plan['header']: emit_header(plan), plan['test']: emit_test(plan), plan['manifest']: emit_manifest(plan, plan_text)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('plan')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    try:
        produced = outputs(args.plan)
    except (PlanError, json.JSONDecodeError, KeyError, OSError) as error:
        print(f'fused_kernels: {error}', file=sys.stderr)
        return 2
    stale = []
    for relative, text in produced.items():
        path = ROOT / relative
        if args.check:
            if not path.is_file() or path.read_text() != text:
                stale.append(relative)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    if stale:
        print('fused_kernels: generated files differ from the plan: ' + ', '.join(stale), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
