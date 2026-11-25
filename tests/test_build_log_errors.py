#!/usr/bin/env python3
import importlib.util
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOOL = ROOT / 'tools' / 'build_log_errors.py'
GATE = ROOT / 'tools' / 'cuda13_sm121a_compile_gate.sh'
SPEC = importlib.util.spec_from_file_location('build_log_errors', TOOL)
ERRORS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ERRORS)

SITE = ('     do { spark_last_error_site.code = (int32_t)((-42)); spark_last_error_site.file = "./inference/kernels/stream_gemm.cuh"; '
        'spark_last_error_site.line = (uint32_t)799; } while ( 0 );')
WARNING = ['./inference/kernels/stream_gemm.cuh(799): warning #177-D: variable "error" was declared but never referenced', SITE, '                 ^', '',
           './inference/kernels/weight_codec.cuh:34:51: warning: comparison between \'enum SparkWeightScaleEncoding\' and \'enum LmScaleEncoding\' [-Wenum-compare]',
           '   34 | static_assert(SPARK_WEIGHT_SCALE_ENCODING_NONE == LM_SCALE_ENCODING_NONE, "error: the encodings differ");', '']
COMMAND = '/usr/local/cuda/bin/nvcc -std=c++17 -O3 -Werror=return-type -DSPARK_ERROR_SITE=1 tools/dsv41_flash_decode_replay.cu -o build/dsv41_flash_decode_replay'
UNDEFINED = 'tools/dsv41_flash_decode_replay.cu(342): error: identifier "D41KvNormRopeQdqKernel" is undefined'
EXPRESSION = 'tools/dsv41_flash_decode_replay.cu(342): error: expected an expression'
SOURCE = '   D41KvNormRopeQdqKernel<256u><<<(rows), (256u), ((512u + 8u) * sizeof(float)), (stream)>>>(state->kv, layer->kv_norm)'
DETECTED = '2 errors detected in the compilation of "tools/dsv41_flash_decode_replay.cu".'
TARGET = 'make: *** [Makefile:2441: build/dsv41_flash_decode_replay] Error 2'
WAITING = 'make: *** Waiting for unfinished jobs....'
REFUSED = ('REFUSED build/modules/dsv41_dspark_draft_stage.so.tmp: _Z14LmSkinnyKernelI7LmMxfp8Lj8ELj1ELj2EEv17LmSkinnyArguments uses 64 registers and '
           'spills: stack 8 bytes, local 0 bytes')
MODULE = 'make: *** [Makefile:2690: build/modules/dsv41_dspark_draft_stage.so] Error 1'
GCC = ['runtime/model_batch_engine.c:4919:17: error: implicit declaration of function \'SparkModelBatchTraceStep\' [-Wimplicit-function-declaration]',
       ' 4919 |                 SparkModelBatchTraceStep(engine, lane);', '      |                 ^~~~~~~~~~~~~~~~~~~~~~~~']
LINK = ['/usr/bin/ld: build/objects/stage_runner.o: in function `StageRunnerSubmit\':',
        'stage_runner.cu:(.text+0x1c4): undefined reference to `SparkTpLaunchMeshHardwareFast\'', 'collect2: error: ld returned 1 exit status']
PYTHON = ['Traceback (most recent call last):', '  File "tools/k3_pack.py", line 12, in <module>', '    main()', 'KeyError: \'k_tiles\'']


def text(lines):
    return '\n'.join(lines) + '\n'


def report(lines, *options):
    with tempfile.TemporaryDirectory() as directory:
        log = pathlib.Path(directory) / 'build.txt'
        log.write_text(text(lines))
        result = subprocess.run([sys.executable, '-I', str(TOOL), *options, str(log)], capture_output=True, text=True)
    return result.returncode, result.stdout.splitlines()


def gate_fail(lines, root, *arguments):
    function = re.search(r'^gate_fail\(\)\n\{\n.*?\n\}\n', GATE.read_text(), re.M | re.S)
    with tempfile.TemporaryDirectory() as directory:
        logs = pathlib.Path(directory) / 'logs'
        logs.mkdir()
        (logs / 'phase.txt').write_text(text(lines))
        script = f'set -euo pipefail\nrepository_root="$1"\noutput_directory="$2"\nshift 2\n{function.group(0)}gate_fail "$@"\necho reached\n'
        result = subprocess.run(['bash', '-c', script, 'gate', str(root), directory, *arguments], capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr.splitlines()


def check(failures):
    buried = WARNING * 400 + [COMMAND, UNDEFINED, SOURCE, '   ^', '', EXPRESSION, SOURCE, '                               ^', '', DETECTED, TARGET, WAITING] + WARNING * 40
    first = 7 * 400 + 2
    code, lines = report(buried)
    if code != 0 or lines != [f'{first}: {UNDEFINED}', SOURCE, '   ^', f'{first + 4}: {EXPRESSION}', SOURCE, '                               ^',
                              f'{first + 8}: {DETECTED}', f'{first + 9}: {TARGET}']:
        failures.append('a compiler error among thousands of warning lines is not printed with its line number, its source lines and make\'s target, alone')
    code, lines = report([COMMAND] + WARNING * 3 + [REFUSED, REFUSED.replace('Lj8E', 'Lj16E'), '', MODULE, WAITING] + WARNING)
    if lines != [f'23: {REFUSED}', '24: ' + REFUSED.replace('Lj8E', 'Lj16E'), f'26: {MODULE}']:
        failures.append('lines a tool marks REFUSED under make are not printed with the target that failed')
    code, lines = report(WARNING + GCC + ['runtime/model_batch_engine.c:4919:17: note: each undeclared identifier is reported once'] + LINK + PYTHON)
    if lines != [f'8: {GCC[0]}', GCC[1], GCC[2], f'13: {LINK[1]}', f'14: {LINK[2]}', f'15: {PYTHON[0]}', PYTHON[1], PYTHON[2], f'18: {PYTHON[3]}']:
        failures.append('a gcc error, a linker error and a Python exception are not each printed with the indented lines under them')
    code, lines = report(([UNDEFINED, SOURCE, '   ^', ''] + WARNING) * 27 + [TARGET] * 3)
    if lines != [f'1: {UNDEFINED} (27 times)', SOURCE, '   ^', f'298: {TARGET} (3 times)']:
        failures.append('a line that repeats is not printed once with its count')
    many = [f'model-families/k3/source/slice.cuh({index}): error: identifier "K{index}" is undefined' for index in range(1, 31)]
    code, lines = report(many + [TARGET], '--limit', '5')
    if lines[:5] != [f'{index}: {many[index - 1]}' for index in range(1, 6)] or lines[5] != f'31: {TARGET}' or not lines[6].startswith('... and 25 more '):
        failures.append('--limit does not keep the first failing lines and make\'s target and say how many it left out')
    code, lines = report(WARNING * 20, '--tail', '3')
    if code != 0 or len(lines) != 4 or 'names no failing line; its last 3 of 140 lines' not in lines[0] or lines[1:] != WARNING[-3:]:
        failures.append('a log with no failing line does not print its last lines')
    audit = ['FRAME kept: LmAttendKernel', 'dependent wait audit: 412 waiting kernels in 31 artifacts, 0 writes before a wait',
             'dependent_wait_audit: cuobjdump cannot read build/x.so: no such file']
    code, lines = report(audit, '--also', '^dependent')
    if lines != [f'2: {audit[1]}', f'3: {audit[2]}'] or report(audit)[1][0].startswith('2: '):
        failures.append('--also does not add an audit\'s own lines, or they are printed without it')
    code, lines = report([UNDEFINED, '   ' + 'x' * 900, TARGET], '--width', '80')
    if lines != [f'1: {UNDEFINED[:80]}', '   ' + 'x' * 77, f'3: {TARGET}']:
        failures.append('--width does not bound a printed line')
    if subprocess.run([sys.executable, '-I', str(TOOL), str(ROOT / 'no-such-log.txt')], capture_output=True, text=True).returncode != 1:
        failures.append('a log that cannot be read does not fail the tool')
    for quiet in WARNING + [COMMAND, WAITING, SITE, 'make: Leaving directory \'/__w/sparkpipe/sparkpipe\'', 'gate phase: adapters and CUDA programs: 936 s wall',
                            'ptxas info    : 0 bytes gmem', 'kernel spill audit: 5120 kernels in 31 artifacts; frames kept: none']:
        if ERRORS.FAILURE.match(quiet) or ERRORS.MAKE.match(quiet):
            failures.append(f'a line that reports no failure is taken for one: {quiet[:80]}')
    for loud in [UNDEFINED, DETECTED, GCC[0], LINK[1], LINK[2], REFUSED, PYTHON[0], PYTHON[3], 'FAIL nvrm twin of libk3_tp16_serving_adapter.so is missing',
                 'nvcc fatal   : Value \'sm_121x\' is not defined for option \'gpu-architecture\'', 'ptxas fatal   : Unresolved extern function \'_Z3Foov\'',
                 'ptxas /tmp/tmpxft_1.ptx, line 91; error   : Illegal operand type to instruction \'ld\'', 'g++: fatal error: Killed signal terminated program cc1plus',
                 '/usr/bin/ld: cannot find -lsparkpipe_device_cuda: No such file or directory', 'x.cu(7): catastrophic error: cannot open source file "y.cuh"']:
        if not ERRORS.FAILURE.match(loud):
            failures.append(f'a line that reports a failure is not printed: {loud[:80]}')
    for loud in [TARGET, MODULE, 'make[1]: *** No rule to make target \'build/x\', needed by \'all\'.  Stop.', 'Makefile:2441: *** missing separator.  Stop.']:
        if not ERRORS.MAKE.match(loud):
            failures.append(f'a make failure is not printed: {loud[:80]}')
    code, out, lines = gate_fail(buried, ROOT, '11', 'phase.txt', 'a stage-runner binary does not build; see logs/phase.txt')
    if code != 11 or out or lines[0] != f'{first}: {UNDEFINED}' or lines[-1] != 'a stage-runner binary does not build; see logs/phase.txt' or SITE in lines:
        failures.append('the gate\'s gate_fail does not print the failing lines, then its message, and exit with the phase\'s code')
    code, out, lines = gate_fail(audit, ROOT, '9', 'phase.txt', 'refused', '--also', '^dependent', '--limit', '200')
    if code != 9 or lines != [f'2: {audit[1]}', f'3: {audit[2]}', 'refused']:
        failures.append('the gate\'s gate_fail does not pass its options to the excerpt')
    with tempfile.TemporaryDirectory() as empty:
        code, out, lines = gate_fail(buried, empty, '5', 'phase.txt', 'adapters do not build')
    if code != 5 or lines[-1] != 'adapters do not build' or lines[-41:-1] != buried[-40:]:
        failures.append('the gate\'s gate_fail does not fall back to the log\'s last lines when the excerpt cannot run')
    gate = GATE.read_text()
    captured = re.findall(r'logs/([\w${}.]+)"(?: 2>&1; then| \|\|)\n(?:\t+(?!gate_fail)[^\n]*\n)*?\t+gate_fail (\d+) ("?[\w${}.]+"?) ', gate)
    tested = re.findall(r'logs/([\w${}.]+)"(?: 2>&1; then| \|\|)\n', gate)
    if len(captured) != 7 or tested != [name for name, _, _ in captured] or any(name != reported.strip('"') for name, _, reported in captured) or \
            sorted(int(code) for _, code, _ in captured) != [5, 6, 8, 9, 10, 11, 12]:
        failures.append('a phase of the sm_121a compile gate captures its output in a log that it does not report through gate_fail under its own exit code')


def main():
    failures = []
    check(failures)
    for failure in failures:
        print('FAIL ' + failure)
    if not failures:
        print('PASS build log errors: compiler, linker, toolkit, make, Python and REFUSED or FAIL lines are printed once with their line numbers and '
              'the source under them, warnings and their source text are not, a log without one prints its tail, and every phase of the sm_121a '
              'compile gate that captures its output reports it through gate_fail')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
