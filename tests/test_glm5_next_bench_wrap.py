import importlib.util
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bench", ROOT / "tools/glm5_next_bench_wrap.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def event(kind, request, index=0):
    return {"event": kind, "request_id": request, "sequence_id": request,
            "token_index": index, "token_id": request * 10 + index, "status": 0}


class BenchTests(unittest.TestCase):
    def test_decode_window_excludes_remaining_prefill_and_completion_tail(self):
        events = [(1., event("token", 1, 0)), (2., event("token", 1, 1)),
                  (10., event("token", 2, 0)), (11., event("token", 1, 2)),
                  (12., event("token", 2, 1)), (13., event("token", 1, 3)),
                  (14., event("token", 2, 2)), (15., event("completed", 1)),
                  (16., event("completed", 2))]
        result = bench.summarize(events, 0)
        window = result["all_sequences_decode_window"]
        self.assertTrue(window["valid"])
        self.assertEqual(window["all_prefixes_ready_seconds"], 10.)
        self.assertEqual(window["elapsed_seconds"], 3.)
        self.assertEqual(window["token_count"], 3)
        self.assertEqual(window["aggregate_tokens_per_second"], 1.)
        self.assertEqual(result["sequences"][0]["token_ids"], [10, 11, 12, 13])

    def test_interleaved_stream_preserves_arrival_order(self):
        events = [(1., event("token", 1, 0)), (2., event("token", 1, 1)),
                  (3., event("token", 2, 0)), (4., event("token", 2, 1)),
                  (5., event("completed", 1)), (6., event("completed", 2))]
        result = bench.summarize(events, 0)
        self.assertTrue(result["valid"])
        self.assertEqual(result["token_ids"], (10, 11, 20, 21))
        self.assertEqual(result["decode_tokens_per_second"], 1.)
        self.assertNotIn("inter_token_median_seconds", result)
        self.assertFalse(result["all_sequences_decode_window"]["valid"])
        self.assertNotIn("aggregate_tokens_per_second", result["all_sequences_decode_window"])

    def test_failure_and_incomplete_streams_have_no_rate(self):
        streams = [([], 0), ([(1., event("token", 1))], 0),
                   ([(1., event("token", 1)), (2., event("token", 1)),
                     (3., event("completed", 1))], 0),
                   ([(1., event("token", 1)), (2., event("completed", 1))], 7)]
        for events, status in streams:
            result = bench.summarize(events, status)
            self.assertFalse(result["valid"])
            self.assertNotIn("decode_tokens_per_second", result)

    def test_large_stderr_does_not_deadlock(self):
        script = ('import json,sys; sys.stderr.write("x"*200000); '
                  f'print(json.dumps({event("token", 1)!r})); '
                  f'print(json.dumps({event("completed", 1)!r}))')
        result = bench.measure([sys.executable, "-c", script], 5)
        self.assertTrue(result["valid"])
        self.assertEqual(result["stderr_tail"], "x" * 500)

    def test_timeout_reaps_child_and_rejects_receipt(self):
        result = bench.measure([sys.executable, "-c", "import time; time.sleep(30)"], .1)
        self.assertFalse(result["valid"])
        self.assertIn("benchmark deadline exceeded", result["errors"])
        self.assertIsNotNone(result["process_status"])


if __name__ == "__main__":
    unittest.main()
