import importlib.util
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('tp_cupti_trace_report', ROOT / 'tools/tp_cupti_trace_report.py')
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)
TRACE = '''TRACE_ANCHOR pid=123 cupti_ns=100 monotonic_ns=1100 realtime_ns=2100 flush_period_ms=1000
CONCURRENT_KERNEL [ 10, 50 ] duration 40, "matrix_product", correlationId 1
CONCURRENT_KERNEL [ 20, 30 ] duration 10, "SparkGlm5NextMeshHardwarePublishKernel", correlationId 2
MEMCPY "HtoD" [ 40, 60 ] duration 20, size 1024, correlationId 3
MEMSET [ 80, 90 ] duration 10, value 0, size 1024, correlationId 4
RUNTIME [ 1, 1000 ] duration 999, ignored_host_call
TRACE_BUFFER context=0x1 stream=2 valid_bytes=1024 capacity=8192 dropped_records=0
TRACE_EXIT buffers_requested=1 buffers_completed=1
'''


class TraceReportTest(unittest.TestCase):
    def analyze(self, text=TRACE, **options):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'trace.log'
            path.write_text(text)
            return REPORT.report(path, **options)

    def test_overlap_and_clock_anchor(self):
        result = self.analyze()
        self.assertEqual(result['status'], 'COMPLETE')
        self.assertEqual(result['window_ns'], 80)
        self.assertEqual(result['gpu_covered_ns'], 60)
        self.assertEqual(result['gpu_uncovered_ns'], 20)
        self.assertEqual(result['selection_ns']['monotonic'], [1010, 1090])
        self.assertEqual(result['selection_ns']['realtime'], [2010, 2090])
        self.assertEqual(result['categories']['compute']['covered_ns'], 40)
        self.assertEqual(result['categories']['mesh']['covered_ns'], 10)
        self.assertEqual(result['largest_uncovered_intervals'][0]['cupti_ns'], [60, 80])

    def test_clipped_host_window(self):
        result = self.analyze(clock='monotonic', start=1025, end=1085)
        self.assertEqual(result['window_ns'], 60)
        self.assertEqual(result['gpu_covered_ns'], 40)
        self.assertEqual(result['gpu_uncovered_ns'], 20)
        self.assertEqual(result['clipped_record_count'], 3)
        self.assertEqual(result['categories']['compute']['covered_ns'], 25)

    def test_host_copy_is_not_gpu_activity(self):
        result = self.analyze(TRACE + 'MEMCPY "HtoH" [ 5, 500 ] duration 495, size 1024\n')
        self.assertEqual(result['window_ns'], 80)
        self.assertEqual(result['gpu_covered_ns'], 60)
        self.assertEqual(result['gpu_uncovered_ns'], 20)
        self.assertEqual(result['categories']['host_memcpy']['covered_ns'], 80)

    def test_loss_incomplete_and_malformed_are_partial(self):
        result = self.analyze(TRACE.replace('dropped_records=0', 'dropped_records=7').replace('TRACE_EXIT', 'UNFINISHED') +
                              'KERNEL [ 0, 100 ] duration 100, "incomplete"\nMEMCPY unreadable\n')
        self.assertEqual(result['status'], 'PARTIAL')
        self.assertEqual(result['dropped_records'], 7)
        self.assertEqual(result['invalid_gpu_records'], 2)
        self.assertIn('missing_normal_exit_flush', result['issues'])

    def test_no_records_and_multiple_anchors_reject(self):
        with self.assertRaisesRegex(ValueError, 'No completed GPU'):
            self.analyze('TRACE_EXIT buffers_completed=0\n')
        with self.assertRaisesRegex(ValueError, 'Multiple clock anchors'):
            self.analyze(TRACE + TRACE.splitlines()[0] + '\n')
        with self.assertRaisesRegex(ValueError, 'both start and end'):
            self.analyze(start=100)


if __name__ == '__main__':
    unittest.main()
