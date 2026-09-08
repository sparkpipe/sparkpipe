#!/usr/bin/env python3
"""Exercise the residentd startup gate against real Unix sockets and sidecars."""
import ctypes
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SupervisedWeightd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory(prefix="weightd-gate-")
        library = Path(cls.build.name) / "gate.so"
        subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror",
                        "-shared", "-fPIC", str(ROOT / "node/weightd_spawn.c"),
                        "-o", str(library)], check=True)
        cls.library = ctypes.CDLL(str(library))
        cls.gate = cls.library.SparkModelResidentdPrepareWeightd
        cls.gate.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        cls.gate.restype = ctypes.c_int32
        cls.libc = ctypes.CDLL(None)
        cls.libc.getenv.argtypes = [ctypes.c_char_p]
        cls.libc.getenv.restype = ctypes.c_char_p

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="wd-", dir="/tmp")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "packs").mkdir()
        self.sidecar = self.root / "packs/rank.sp.sha256"
        self.sidecar.write_text("a" * 64 + "  rank.sp\n")
        self.path = str(self.root / "wd.sock")
        self.server = socket.socket(socket.AF_UNIX)
        self.server.bind(self.path)
        self.server.listen(8)
        self.addCleanup(self.server.close)
        for name in ("SPARK_WEIGHTD_ATTACH", "SPARK_WEIGHTD_SOCKET",
                     "SPARK_WEIGHTD_PACK_SHA256"):
            previous = self.libc.getenv(name.encode())
            self.addCleanup(self.restore, name, previous)
            os.environ.pop(name, None)
            os.unsetenv(name)

    @staticmethod
    def restore(name, previous):
        os.environ.pop(name, None)
        if previous is None:
            os.unsetenv(name)
        else:
            os.environ[name] = previous.decode()

    def call(self):
        return self.gate(str(self.root).encode(), self.path.encode())

    def test_ready_sets_exact_identity(self):
        self.assertEqual(self.call(), 0)
        self.assertEqual(self.libc.getenv(b"SPARK_WEIGHTD_SOCKET"), self.path.encode())
        self.assertEqual(self.libc.getenv(b"SPARK_WEIGHTD_PACK_SHA256"), b"a" * 64)

    def test_missing_and_ambiguous_digest(self):
        self.sidecar.unlink()
        self.assertEqual(self.call(), -9)
        self.sidecar.write_text("b" * 64)
        (self.root / "packs/other.sha256").write_text("c" * 64)
        self.assertEqual(self.call(), -5)
        self.assertIsNone(self.libc.getenv(b"SPARK_WEIGHTD_PACK_SHA256"))

    def test_bad_digest(self):
        for text, expected in (("a" * 63, -1), ("z" * 64, -2),
                               ("a" * 65, -3)):
            self.sidecar.write_text(text)
            self.assertEqual(self.call(), expected)

    def test_absent_daemon_does_not_spawn(self):
        self.server.close()
        self.assertEqual(self.call(), -16)
        self.assertIsNone(self.libc.getenv(b"SPARK_WEIGHTD_SOCKET"))

    def test_explicit_off_cannot_override_deployment(self):
        os.environ["SPARK_WEIGHTD_ATTACH"] = "0"
        self.assertEqual(self.call(), -14)

    def test_invalid_arguments(self):
        self.assertEqual(self.gate(None, b"socket"), -12)
        self.assertEqual(self.gate(b"root", b"x" * 200), -13)


if __name__ == "__main__":
    unittest.main()
