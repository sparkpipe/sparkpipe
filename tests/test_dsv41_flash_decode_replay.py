import importlib.util
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('dsv41_flash_decode_replay', ROOT / 'tools/dsv41_flash_decode_replay.py')
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


def kernel_names():
    names = set()
    for folder in ('inference', 'model-families', 'modules', 'ring', 'runtime', 'tools'):
        for path in sorted((ROOT / folder).rglob('*.cu*')):
            for match in re.finditer(r'__global__([^;{}]*)', path.read_text(errors='replace')):
                header = re.sub(r'__(?:launch_bounds|cluster_dims)__\s*\((?:[^()]|\([^()]*\))*\)', ' ', match.group(1))
                name = re.search(r'([A-Za-z_]\w*)\s*\(', header)
                if name is not None:
                    names.add(name.group(1))
    return names


def record(start, end, name):
    return f'CONCURRENT_KERNEL [ {start}, {end} ] duration {end - start}, "{name}", correlationId 1\n'


def step(base, fold, prefetch=False):
    lines = [record(base, base + 1000, '_Z16ReplayMarkKerneljPj')]
    t = base + 2000
    for name, length in (('_Z25LmHcSinkhornPartialKernelILj4EEvPKtPKfPfj', 16000), ('_Z18LmHcSinkhornKernelILj4EEvPKfS1_S1_PfS2_S2_jfjf', 4000),
                         ('void D41RmsNormQdqKernel<256u>(unsigned short const*, unsigned short const*)', 4000),
                         ('D41InitRecordsKernel', 1000), ('_Z24D41WindowFoldWholeKernelPfjPKtS1_S1_', fold), ('D41WindowRingStoreKernel', 2000)):
        lines.append(record(t, t + length, name))
        if prefetch and name == 'D41InitRecordsKernel':
            lines.append(record(t + length + 100, t + length + 40000, '_Z20ReplayPrefetchKernel10ReplaySpans'))
        t += length + 500
    return lines


class DecodeReplayTest(unittest.TestCase):
    def test_names(self):
        self.assertEqual(REPLAY.stem('_Z25LmHcSinkhornPartialKernelILj4EEvPKtPKfPfj'), 'LmHcSinkhornPartialKernel')
        self.assertEqual(REPLAY.stem('void D41RmsNormQdqKernel<256u>(unsigned short const*)'), 'D41RmsNormQdqKernel')
        self.assertEqual(REPLAY.stem('LmSkinnyFrameKernel<LmMxfp8, 32u, 4u, 2u, true>'), 'LmSkinnyFrameKernel')
        self.assertEqual(REPLAY.stem('D41MergeSinkKernel'), 'D41MergeSinkKernel')

    def test_runs_steps_sequences_and_reference(self):
        trace = ''.join(step(0, 60000) + step(200000, 62000) + step(400000, 90000, True) + step(600000, 92000, True))
        stdout = 'REPLAY plain rows=8 steps=2 step_ms_median=1.0 step_ms_min=1.0\nREPLAY fetched rows=8 steps=2 step_ms_median=1.0 step_ms_min=1.0\n'
        reference = {'rows': {'8': {'sequences': {'mhc': {'us_per_instance': 25.0}, 'attention_core': {'us_per_instance': 100.0}}}}}
        with tempfile.TemporaryDirectory() as directory:
            folder = pathlib.Path(directory)
            (folder / 'trace.log').write_text(trace)
            (folder / 'out.txt').write_text(stdout)
            (folder / 'reference.json').write_text(json.dumps(reference))
            result = subprocess.run([sys.executable, '-I', str(ROOT / 'tools/dsv41_flash_decode_replay.py'), '--trace', str(folder / 'trace.log'),
                                     '--stdout', str(folder / 'out.txt'), '--warmup', '0', '--reference', str(folder / 'reference.json'),
                                     '--json', str(folder / 'result.json')], capture_output=True, text=True, check=True)
            output = json.loads((folder / 'result.json').read_text())
        plain, fetched = output['runs']['plain'], output['runs']['fetched']
        self.assertIn('mhc', result.stdout)
        self.assertEqual(plain['steps_kept'], 2)
        self.assertEqual(plain['kernels'], 6)
        self.assertEqual(fetched['kernels'], 6)
        self.assertEqual(plain['sequences']['mhc']['instances'], 1)
        self.assertAlmostEqual(plain['sequences']['mhc']['us_per_instance'], 25.0)
        self.assertAlmostEqual(plain['sequences']['attention_core']['fold_us_per_instance'], 61.0)
        self.assertAlmostEqual(plain['sequences']['attention_core']['us_per_instance'], 65.0)
        self.assertAlmostEqual(fetched['sequences']['attention_core']['fold_us_per_instance'], 91.0)
        self.assertAlmostEqual(fetched['side_prefetch_us'], 39.9)
        self.assertEqual(plain['side_prefetch_us'], 0.0)
        self.assertAlmostEqual(plain['against_trace']['mhc']['ratio'], 1.0)
        self.assertAlmostEqual(plain['against_trace']['attention_core']['ratio'], 0.65)

    def test_every_name_the_summary_matches_is_a_kernel_in_the_tree(self):
        defined = kernel_names()
        wanted = {REPLAY.MARK, *REPLAY.SIDE}
        for pattern in REPLAY.SEQUENCES.values():
            for want in pattern:
                wanted.update(want if isinstance(want, tuple) else (want,))
        self.assertEqual(sorted(wanted - defined), [])
        for prefix in REPLAY.COLLECTIVE:
            self.assertTrue(any(name.startswith(prefix) for name in defined), prefix)

    def test_steps_that_differ_are_refused(self):
        names = ['A', 'B']
        steps = [[(0, 10, 'A'), (20, 30, 'B')], [(0, 10, 'A'), (20, 30, 'C')], [(0, 10, 'A'), (20, 30, 'B')]]
        with self.assertRaises(ValueError):
            REPLAY.medians(steps)
        self.assertEqual(REPLAY.medians([steps[0], steps[2]])[0], names)


if __name__ == '__main__':
    unittest.main()
