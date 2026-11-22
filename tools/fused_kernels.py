#!/usr/bin/env python3
"""Generate fused kernels, their launchers, their exactness test and their manifest from a fusion plan.

    python3 -I tools/fused_kernels.py PLAN [--check]

A plan (model-families/<model>/fusion/<step>.json) lists fused kernels. Each
kernel is a set of roles: every role gives each row of the step a fixed number
of thread blocks that run one op, and may end in a tail that the row's last
arriving block runs after LmFusedLastArrival. Ops are the bodies of the
engine's own kernels (device functions shared with them): the shared ones are
in the catalogue below, and a plan declares its own model's ops under "ops". A
fused kernel therefore computes every value as the kernels it replaces do; the
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
POINTERS = ('const uint16_t *', 'uint16_t *', 'const float *', 'float *', 'const uint32_t *', 'uint32_t *')
SCALARS = ('uint32_t', 'int32_t', 'uint64_t', 'float')
FILLS = ('bf16', 'f32', 'zero', 'index')
BUMPS = {'uint16_t': '0x3f80u', 'float': '1.5f', 'uint32_t': '1u'}
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


def pointer(argument):
    return argument['type'] in POINTERS


def writable(argument):
    return pointer(argument) and not argument['type'].startswith('const ')


def base_type(argument):
    return argument['type'].replace('const ', '').replace(' *', '')


def declaration(argument):
    return f'{argument["type"]}{argument["name"]}' if pointer(argument) else f'{argument["type"]} {argument["name"]}'


def catalogue(plan):
    ops = dict(OPS)
    declared = plan.get('ops', {})
    require(isinstance(declared, dict), 'ops is an object of the model\'s own ops')
    for name, op in declared.items():
        check_name(name, 'op')
        require(name not in OPS, f'op {name} is in the shared catalogue already')
        require(isinstance(op, dict), f'op {name} is an object')
        check_name(op.get('body'), f'op {name} body')
        source = op.get('source')
        require(isinstance(source, str) and source.startswith('model-families/') and not source.startswith(SHARED_ROOTS),
                f'op {name}: a plan declares only its model\'s own ops, with their bodies under model-families/<model>/')
        require(isinstance(op.get('replaces'), str), f'op {name}: replaces names the kernels whose body it is')
        require(isinstance(op.get('parameters'), list) and all(isinstance(item, str) and NAME.match(item) for item in op['parameters']),
                f'op {name}: parameters are C identifiers')
        require(isinstance(op.get('call'), str), f'op {name}: call is a C statement template')
        require(isinstance(op.get('lanes'), int) and 0 <= op['lanes'] <= 1024, f'op {name}: lanes is 0 to 1024')
        ops[name] = dict(op, parameters=tuple(op['parameters']))
    return ops


def check_argument(kernel, argument, names):
    require(isinstance(argument, dict), f'{kernel["name"]}: an argument is an object')
    check_name(argument.get('name'), 'argument')
    require(argument['name'] not in names, f'{kernel["name"]}: argument {argument["name"]} twice')
    names.add(argument['name'])
    require(argument.get('type') in POINTERS + SCALARS, f'{kernel["name"]}: argument {argument["name"]} has an unknown type')
    if not pointer(argument):
        require(isinstance(argument.get('value'), str) and 'elements' not in argument and 'fill' not in argument,
                f'{kernel["name"]}: scalar argument {argument["name"]} takes a value expression only')
        return
    require(argument.get('fill') in FILLS, f'{kernel["name"]}: argument {argument["name"]} fill is one of {FILLS}')
    require(isinstance(argument.get('elements'), str), f'{kernel["name"]}: argument {argument["name"]} needs an element count expression')
    require(argument['fill'] != 'index' or isinstance(argument.get('expression'), str),
            f'{kernel["name"]}: argument {argument["name"]} is filled from an expression of its index')
    require(isinstance(argument.get('partial', False), bool), f'{kernel["name"]}: partial of {argument["name"]} is true or false')
    regions = argument.get('row_zero', [])
    require(isinstance(regions, list) and all(isinstance(region, list) and len(region) == 2 and all(isinstance(item, str) for item in region)
                                              for region in regions),
            f'{kernel["name"]}: row_zero of {argument["name"]} lists [offset, count] expressions')


def load(path):
    plan = json.loads(Path(path).read_text())
    require(isinstance(plan, dict), 'a plan is a JSON object')
    for key in ('header', 'test', 'manifest'):
        check_output(plan.get(key), key)
    require(isinstance(plan.get('includes'), list) and all(isinstance(item, str) for item in plan['includes']), 'includes is a list of paths')
    require(isinstance(plan.get('test_includes'), list) and all(isinstance(item, str) for item in plan['test_includes']), 'test_includes is a list of paths')
    require(isinstance(plan.get('kernels'), list) and plan['kernels'], 'a plan has at least one kernel')
    ops = catalogue(plan)
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
            check_argument(kernel, argument, names)
        require(names, f'{kernel["name"]}: no arguments')
        arguments = {argument['name']: argument for argument in kernel['arguments']}
        counters = arguments.get('counters')
        require(counters is None or (counters['type'] == 'uint32_t *' and counters['fill'] == 'zero'),
                f'{kernel["name"]}: counters are uint32_t * and start at zero')
        for buffer in kernel.get('scratch', []):
            require(isinstance(buffer, dict), f'{kernel["name"]}: a scratch buffer is an object')
            check_name(buffer.get('name'), 'scratch')
            require(buffer['name'] not in names, f'{kernel["name"]}: scratch {buffer["name"]} twice')
            names.add(buffer['name'])
            require(buffer.get('type') in POINTERS and not buffer['type'].startswith('const ') and isinstance(buffer.get('elements'), str),
                    f'{kernel["name"]}: scratch {buffer["name"]} is a writable pointer with an element count')
        require(isinstance(kernel.get('roles'), list) and kernel['roles'], f'{kernel["name"]}: at least one role')
        for role in kernel['roles']:
            check_op(ops, kernel, role, arguments, False)
            if 'tail' in role:
                require(counters is not None, f'{kernel["name"]}: a role with a tail needs the counters argument')
                require(role['tail'].get('scope') == 'row', f'{kernel["name"]}: a tail runs for its row')
                check_op(ops, kernel, role['tail'], arguments, True)
        require(isinstance(kernel.get('perturb'), str) and kernel['perturb'] in arguments and pointer(arguments[kernel['perturb']]),
                f'{kernel["name"]}: perturb names an input argument')
        require(isinstance(kernel.get('reference'), list) and kernel['reference'], f'{kernel["name"]}: the reference calls are listed')
        require(isinstance(kernel.get('outputs'), list) and kernel['outputs'] and
                all(output in arguments and writable(arguments[output]) for output in kernel['outputs']),
                f'{kernel["name"]}: outputs name arguments')
        require(isinstance(kernel.get('replaces'), list) and kernel['replaces'], f'{kernel["name"]}: replaces lists the kernels it removes')
    plan['catalogue'] = ops
    return plan


def check_op(ops, kernel, role, arguments, tail):
    op = ops.get(role.get('op'))
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
            require(token in arguments, f'{kernel["name"]}: {token} is not an argument')


def render(template, bind, shared):
    values = dict(bind)
    values['shared'] = shared
    return template.format(**values)


def emit_header(plan):
    ops = plan['catalogue']
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
            lines.append(f'\t{declaration(argument)};')
        lines.append(f'}} {kernel["arguments_type"]};')
        lines.append('')
        lines.append(f'__global__ __launch_bounds__({threads}) void {name}(const __grid_constant__ {kernel["arguments_type"]} arguments, uint32_t rows)')
        lines.append('{')
        lines.append(f'\textern __shared__ float {shared}[];')
        lines.append('\tuint32_t block = blockIdx.x;')
        for role in kernel['roles']:
            per_row = role['blocks_per_row']
            values = dict(role['bind'], threads=f'{threads}u', blocks_per_row=f'({per_row})')
            call = render(ops[role['op']]['call'], values, shared)
            tail = role.get('tail')
            tail_call = render(ops[tail['op']]['call'], dict(tail['bind'], threads=f'{threads}u', blocks_per_row=f'({per_row})'), shared) if tail else ''
            lines.append(f'\tif ( block < rows * ({per_row}) )')
            lines.append('\t{')
            if re.search(r'\bitem\b', call + tail_call):
                lines.append(f'\t\tconst uint32_t row = block / ({per_row}), item = block % ({per_row});')
            else:
                lines.append(f'\t\tconst uint32_t row = block / ({per_row});')
            lines.append('\t\t' + call)
            if tail:
                lanes = ops[tail['op']]['lanes']
                guard = f' && threadIdx.x < {lanes}u' if lanes else ''
                lines.append(f'\t\tif ( LmFusedLastArrival(arguments.counters + row, {per_row}){guard} )')
                lines.append('\t\t\t' + tail_call)
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


TEST_PRELUDE = [
    '#define FUSED_REQUIRE(condition) do { if ( !(condition) ) { fprintf(stderr, "FAIL %s:%d %s\\n", __FILE__, __LINE__, #condition); exit(1); } } while ( 0 )',
    '#define FUSED_CUDA(call) FUSED_REQUIRE((call) == cudaSuccess)', '',
    'static __host__ __device__ __forceinline__ uint64_t FusedMix(uint64_t value)', '{', '\tvalue += 0x9e3779b97f4a7c15ull;',
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


def emit_buffers(kernel):
    name, args = kernel['name'], kernel['arguments']
    lines = [f'static void {name}Buffers({kernel["arguments_type"]} *set, uint64_t seed, uint32_t edges)', '{', f'\tconst uint32_t rows = {kernel["rows_max"]}u;']
    for index, argument in enumerate(args):
        if not pointer(argument):
            lines.append(f'\tset->{argument["name"]} = ({argument["type"]})({argument["value"]});')
            continue
        base = base_type(argument)
        lines += ['\t{', f'\t\tconst uint64_t count = (uint64_t)({argument["elements"]}), bytes = count * sizeof({base});', '\t\tvoid *pointer = 0;',
                  '\t\tFUSED_CUDA(cudaMalloc(&pointer, bytes));']
        if argument['fill'] == 'zero':
            lines.append('\t\tFUSED_CUDA(cudaMemset(pointer, 0, bytes));')
        elif argument['fill'] == 'index':
            lines += [f'\t\tstd::vector<{base}> host(count);', '\t\tfor ( uint64_t index = 0u; index < count; ++index )',
                      f'\t\t\thost[index] = ({base})({argument["expression"]});',
                      '\t\tFUSED_CUDA(cudaMemcpy(pointer, host.data(), bytes, cudaMemcpyHostToDevice));']
        else:
            kind = 1 if argument['fill'] == 'bf16' else 2
            lines.append(f'\t\tFusedFillKernel<<<256, 256>>>((uint8_t *)pointer, bytes, seed + {index}u, {kind}u, edges);')
        lines += [f'\t\tset->{argument["name"]} = ({argument["type"]})pointer;', '\t}']
    lines += ['\t(void)rows;', '\t(void)seed;', '\t(void)edges;', '\tFUSED_CUDA(cudaDeviceSynchronize());', '}', '']
    lines += [f'static void {name}Copy({kernel["arguments_type"]} *to, const {kernel["arguments_type"]} *from)', '{',
              f'\tconst uint32_t rows = {kernel["rows_max"]}u;']
    for argument in args:
        if pointer(argument):
            lines.append(f'\tFUSED_CUDA(cudaMemcpy((void *)to->{argument["name"]}, from->{argument["name"]}, (uint64_t)({argument["elements"]}) * '
                         f'sizeof({base_type(argument)}), cudaMemcpyDeviceToDevice));')
        else:
            lines.append(f'\tto->{argument["name"]} = from->{argument["name"]};')
    lines += ['\tFUSED_CUDA(cudaDeviceSynchronize());', '\t(void)rows;', '}', '']
    return lines


def emit_reference(kernel):
    name, args, scratch = kernel['name'], kernel['arguments'], kernel.get('scratch', [])
    lines = []
    if scratch:
        lines += ['static struct', '{'] + [f'\t{declaration(buffer)};' for buffer in scratch] + [f'}} {name}Scratch;', '']
    lines += [f'static void {name}Reference(const {kernel["arguments_type"]} *set, uint32_t rows, cudaStream_t stream)', '{']
    lines += [f'\t{declaration(argument)} = set->{argument["name"]};' for argument in args]
    lines += [f'\t{declaration(buffer)} = {name}Scratch.{buffer["name"]};' for buffer in scratch]
    lines += [f'\t(void){item["name"]};' for item in args + scratch]
    lines += [f'\t{call}' for call in kernel['reference']]
    lines += ['\tFUSED_CUDA(cudaGetLastError());', '}', '']
    return lines


def emit_compare(kernel):
    name, arguments = kernel['name'], {argument['name']: argument for argument in kernel['arguments']}
    kind = kernel['arguments_type']
    lines = [f'static uint64_t {name}Differ(const {kind} *one, const {kind} *two, uint32_t rows, uint32_t report)', '{', '\tuint64_t differ = 0u;']
    for output in kernel['outputs']:
        argument = arguments[output]
        lines += ['\t{', f'\t\tconst uint64_t bytes = (uint64_t)({argument["elements"]}) * sizeof({base_type(argument)});',
                  '\t\tstd::vector<uint8_t> left(bytes), right(bytes);', '\t\tuint64_t count = 0u;',
                  f'\t\tFUSED_CUDA(cudaMemcpy(left.data(), one->{output}, bytes, cudaMemcpyDeviceToHost));',
                  f'\t\tFUSED_CUDA(cudaMemcpy(right.data(), two->{output}, bytes, cudaMemcpyDeviceToHost));',
                  '\t\tfor ( uint64_t index = 0u; index < bytes; ++index )', '\t\t\tcount += left[index] != right[index] ? 1u : 0u;',
                  '\t\tif ( report != 0u && count != 0u )',
                  f'\t\t\tfprintf(stderr, "{output}: %llu of %llu bytes differ\\n", (unsigned long long)count, (unsigned long long)bytes);',
                  '\t\tdiffer += count;', '\t}']
    lines += ['\treturn differ;', '}', '']
    lines += [f'static uint64_t {name}RowZero(const {kind} *one, uint32_t rows_one, const {kind} *two, uint32_t rows_two)', '{', '\tuint64_t differ = 0u;']
    for output in kernel['outputs']:
        argument = arguments[output]
        base = base_type(argument)
        for region in argument.get('row_zero', []) or [None]:
            lines += ['\t{', '\t\tuint64_t offset[2], count[2];']
            if region is None:
                lines.append(f'\t\t{{ const uint32_t rows = 1u; offset[0] = offset[1] = 0u; count[0] = count[1] = (uint64_t)({argument["elements"]}); }}')
            else:
                for side, rows in ((0, 'rows_one'), (1, 'rows_two')):
                    lines.append(f'\t\t{{ const uint32_t rows = {rows}; offset[{side}] = (uint64_t)({region[0]}); count[{side}] = (uint64_t)({region[1]}); '
                                 '(void)rows; }')
            lines += ['\t\tFUSED_REQUIRE(count[0] == count[1]);', f'\t\tstd::vector<{base}> left(count[0]), right(count[1]);',
                      f'\t\tFUSED_CUDA(cudaMemcpy(left.data(), one->{output} + offset[0], count[0] * sizeof({base}), cudaMemcpyDeviceToHost));',
                      f'\t\tFUSED_CUDA(cudaMemcpy(right.data(), two->{output} + offset[1], count[1] * sizeof({base}), cudaMemcpyDeviceToHost));',
                      f'\t\tif ( memcmp(left.data(), right.data(), count[0] * sizeof({base})) != 0 )', '\t\t{',
                      f'\t\t\tfprintf(stderr, "{output}: row 0 differs between %u and %u rows\\n", rows_one, rows_two);', '\t\t\t++differ;', '\t\t}', '\t}']
    lines += ['\treturn differ;', '}', '']
    lines += [f'static void {name}Poison(const {kind} *set, const {kind} *origin, cudaStream_t stream)', '{',
              f'\tconst uint32_t rows = {kernel["rows_max"]}u;']
    for output in kernel['outputs']:
        argument = arguments[output]
        bytes = f'(uint64_t)({argument["elements"]}) * sizeof({base_type(argument)})'
        if argument.get('partial', False):
            lines.append(f'\tFUSED_CUDA(cudaMemcpyAsync((void *)set->{output}, origin->{output}, {bytes}, cudaMemcpyDeviceToDevice, stream));')
        else:
            lines.append(f'\tFUSED_CUDA(cudaMemsetAsync((void *)set->{output}, 0xff, {bytes}, stream));')
    lines += ['\t(void)origin;', '\t(void)rows;', '}', '']
    return lines


def emit_exact(kernel):
    name, rows_max, launcher = kernel['name'], kernel['rows_max'], kernel['launcher']
    arguments = {argument['name']: argument for argument in kernel['arguments']}
    perturbed = arguments[kernel['perturb']]
    lines = [f'static uint32_t {name}Exact(void)', '{', f'\t{kernel["arguments_type"]} reference, fused, origin;', '\tcudaStream_t stream = 0;',
             '\tuint32_t checks = 0u;', '\tFUSED_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));']
    for buffer in kernel.get('scratch', []):
        lines += ['\t{', f'\t\tconst uint32_t rows = {rows_max}u;',
                  f'\t\tFUSED_CUDA(cudaMalloc((void **)&{name}Scratch.{buffer["name"]}, (uint64_t)({buffer["elements"]}) * sizeof({base_type(buffer)})));',
                  '\t\t(void)rows;', '\t}']
    lines += ['\tfor ( uint32_t edges = 0u; edges < 2u; ++edges )', '\t{', f'\t\t{name}Buffers(&origin, 0x5000u + 0x100u * edges, edges);',
              f'\t\t{name}Buffers(&reference, 0x7000u, 0u);', f'\t\t{name}Buffers(&fused, 0x7000u, 0u);',
              f'\t\tfor ( uint32_t rows = 1u; rows <= {rows_max}u; ++rows )', '\t\t{', f'\t\t\t{name}Copy(&reference, &origin);',
              f'\t\t\t{name}Copy(&fused, &origin);', f'\t\t\t{name}Reference(&reference, rows, stream);',
              f'\t\t\tFUSED_REQUIRE({launcher}(&fused, rows, stream) == LM_LAUNCH_OK);', '\t\t\tFUSED_CUDA(cudaStreamSynchronize(stream));',
              f'\t\t\tconst uint64_t differ = {name}Differ(&reference, &fused, rows, 1u);', '\t\t\tif ( differ != 0u )',
              f'\t\t\t\t{{ fprintf(stderr, "FAIL {name} rows=%u edges=%u: %llu bytes differ\\n", rows, edges, (unsigned long long)differ); exit(1); }}',
              '\t\t\t++checks;', '\t\t}', '\t}']
    lines += [f'\t{name}Copy(&reference, &origin);', f'\t{name}Copy(&fused, &origin);',
              f'\tFUSED_REQUIRE({launcher}(&reference, 1u, stream) == LM_LAUNCH_OK);', f'\tFUSED_REQUIRE({launcher}(&fused, {rows_max}u, stream) == LM_LAUNCH_OK);',
              '\tFUSED_CUDA(cudaStreamSynchronize(stream));', f'\tFUSED_REQUIRE({name}RowZero(&reference, 1u, &fused, {rows_max}u) == 0u);', '\t++checks;']
    lines += [f'\t{name}Copy(&reference, &origin);', f'\t{name}Copy(&fused, &origin);', '\t{',
              f'\t\tconst {base_type(perturbed)} bump = {BUMPS[base_type(perturbed)]};',
              f'\t\tFUSED_CUDA(cudaMemcpy((void *)fused.{kernel["perturb"]}, &bump, sizeof(bump), cudaMemcpyHostToDevice));',
              '\t\tFUSED_CUDA(cudaDeviceSynchronize());', '\t}', f'\tFUSED_REQUIRE({launcher}(&reference, {rows_max}u, stream) == LM_LAUNCH_OK);',
              f'\tFUSED_REQUIRE({launcher}(&fused, {rows_max}u, stream) == LM_LAUNCH_OK);', '\tFUSED_CUDA(cudaStreamSynchronize(stream));',
              f'\tFUSED_REQUIRE({name}Differ(&reference, &fused, {rows_max}u, 0u) != 0u);', '\t++checks;']
    lines += [f'\t{name}Copy(&reference, &origin);', f'\t{name}Copy(&fused, &origin);', '\t{', '\t\tcudaGraph_t graph = 0;', '\t\tcudaGraphExec_t exec = 0;',
              f'\t\tFUSED_REQUIRE({launcher}(&reference, {rows_max}u, stream) == LM_LAUNCH_OK);',
              '\t\tFUSED_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));',
              f'\t\tFUSED_REQUIRE({launcher}(&fused, {rows_max}u, stream) == LM_LAUNCH_OK);', '\t\tFUSED_CUDA(cudaStreamEndCapture(stream, &graph));',
              '\t\tFUSED_CUDA(cudaGraphInstantiate(&exec, graph, 0));', '\t\tfor ( uint32_t round = 0u; round < 3u; ++round )', '\t\t{',
              f'\t\t\t{name}Poison(&fused, &origin, stream);', '\t\t\tFUSED_CUDA(cudaGraphLaunch(exec, stream));', '\t\t}',
              '\t\tFUSED_CUDA(cudaStreamSynchronize(stream));', f'\t\tFUSED_REQUIRE({name}Differ(&reference, &fused, {rows_max}u, 1u) == 0u);']
    if any('tail' in role for role in kernel['roles']):
        lines += ['\t\t{', f'\t\t\tconst uint32_t rows = {rows_max}u;', f'\t\t\tstd::vector<uint32_t> counters((uint64_t)({arguments["counters"]["elements"]}));',
                  '\t\t\tFUSED_CUDA(cudaMemcpy(counters.data(), fused.counters, counters.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));',
                  '\t\t\tfor ( uint32_t counter : counters )', '\t\t\t\tFUSED_REQUIRE(counter == 0u);', '\t\t}']
    lines += ['\t\tFUSED_CUDA(cudaGraphExecDestroy(exec));', '\t\tFUSED_CUDA(cudaGraphDestroy(graph));', '\t}', '\t++checks;',
              f'\tprintf("PASS {name}: %u comparisons\\n", checks);', '\treturn checks;', '}', '']
    return lines


def emit_test(plan):
    lines = ['#include <cuda_runtime.h>', '#include <stdint.h>', '#include <stdio.h>', '#include <stdlib.h>', '#include <string.h>', '#include <vector>', '']
    lines += [f'#include "{path}"' for path in plan['test_includes']]
    lines += [''] + TEST_PRELUDE
    for kernel in plan['kernels']:
        lines += emit_buffers(kernel) + emit_reference(kernel) + emit_compare(kernel) + emit_exact(kernel)
    lines += ['int main(void)', '{', '\tFUSED_CUDA(cudaSetDevice(0));']
    lines += [f'\t(void){kernel["name"]}Exact();' for kernel in plan['kernels']]
    lines += ['\tprintf("PASS fused kernels\\n");', '\treturn 0;', '}']
    return '\n'.join(lines) + '\n'


def emit_manifest(plan, plan_text):
    ops = plan['catalogue']
    used = [role['op'] for kernel in plan['kernels'] for role in kernel['roles']]
    used += [role['tail']['op'] for kernel in plan['kernels'] for role in kernel['roles'] if 'tail' in role]
    manifest = {'plan_sha256': hashlib.sha256(plan_text.encode()).hexdigest(), 'header': plan['header'], 'test': plan['test'], 'kernels': []}
    for kernel in plan['kernels']:
        roles = []
        for role in kernel['roles']:
            entry = {'op': role['op'], 'body': ops[role['op']]['body'], 'replaces': ops[role['op']]['replaces'], 'blocks_per_row': role['blocks_per_row']}
            if 'tail' in role:
                entry['tail'] = {'op': role['tail']['op'], 'body': ops[role['tail']['op']]['body'], 'replaces': ops[role['tail']['op']]['replaces'],
                                 'sync': 'last arrival of the row'}
            roles.append(entry)
        manifest['kernels'].append({'name': kernel['name'], 'launcher': kernel['launcher'], 'rows': [1, kernel['rows_max']], 'threads': kernel['threads'],
                                    'replaces': kernel['replaces'], 'roles': roles, 'outputs': kernel['outputs'],
                                    'patchable': {'rows': 'parameter 1, and the grid width rows * (sum of blocks per row)'}})
    manifest['op_sources'] = sorted({ops[op]['source'] for op in used})
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
