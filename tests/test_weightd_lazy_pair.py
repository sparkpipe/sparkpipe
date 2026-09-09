import importlib.util
import os
import pathlib
import time
import types
import unittest

spec = importlib.util.spec_from_file_location("lazy_pair", pathlib.Path(__file__).parents[1] / "tools/weightd_lazy_pair.py")
pair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pair)


class ReadinessTests(unittest.TestCase):
    def check_pipe(self, data, error=None, close=False):
        reader, writer = os.pipe()
        with os.fdopen(reader, "rb", buffering=0) as stream:
            try:
                os.write(writer, data)
                if close:
                    os.close(writer)
                    writer = None
                client = types.SimpleNamespace(stdout=stream)
                start = time.monotonic()
                if error:
                    with self.assertRaisesRegex(RuntimeError, error):
                        pair.wait_ready([client], timeout=0.05)
                else:
                    pair.wait_ready([client], timeout=0.05)
                self.assertLess(time.monotonic() - start, 1)
            finally:
                if writer is not None:
                    os.close(writer)

    def test_complete(self):
        self.check_pipe(b"READY\n")

    def test_partial_live_writer_times_out(self):
        self.check_pipe(b"REA", "deadline")

    def test_closed_partial(self):
        self.check_pipe(b"REA", "closed", close=True)

    def test_unbounded_output(self):
        self.check_pipe(b"x" * 65, "limit")

    def test_wrong_message(self):
        self.check_pipe(b"PASS\n", "invalid")


if __name__ == "__main__":
    unittest.main()
