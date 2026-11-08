#!/usr/bin/env python3
"""ck128, the 128-bit checksum of src/spark_ck128.c that weightd and the .experts manifests use, computed by
that C implementation through build/libsparkpipe_ck128 (make build/libsparkpipe_ck128.so, or .dylib on
macOS). A pack tool that checksums tens of gigabytes needs the C speed, so a missing library is an error,
not a slower path.

  spark_ck128.py FILE...    print each file's ck128 hex digest
"""
import ctypes
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("libsparkpipe_ck128.so", "libsparkpipe_ck128.dylib")
READ_CHUNK = 64 * 1024 * 1024


class _Context(ctypes.Structure):
    _fields_ = [("h1", ctypes.c_uint64), ("h2", ctypes.c_uint64), ("total_bytes", ctypes.c_uint64),
                ("tail", ctypes.c_uint8 * 16), ("tail_bytes", ctypes.c_uint32)]


_LIBRARY = None


def library():
    global _LIBRARY
    if _LIBRARY is None:
        path = next((ROOT / "build" / name for name in NAMES if (ROOT / "build" / name).is_file()), None)
        if path is None:
            raise SystemExit(f"spark_ck128: {ROOT / 'build'} has no {' or '.join(NAMES)}; build it with make build/libsparkpipe_ck128.so (.dylib on macOS)")
        loaded = ctypes.CDLL(str(path))
        loaded.SparkCk128Initialize.argtypes = [ctypes.POINTER(_Context)]
        loaded.SparkCk128Update.argtypes = [ctypes.POINTER(_Context), ctypes.c_char_p, ctypes.c_size_t]
        loaded.SparkCk128Finalize.argtypes = [ctypes.POINTER(_Context), ctypes.c_uint8 * 16]
        for name in ("SparkCk128Initialize", "SparkCk128Update", "SparkCk128Finalize"):
            getattr(loaded, name).restype = None
        _LIBRARY = loaded
    return _LIBRARY


class Ck128:
    def __init__(self):
        self.context = _Context()
        library().SparkCk128Initialize(ctypes.byref(self.context))

    def update(self, data) -> None:
        raw = data if isinstance(data, bytes) else bytes(memoryview(data).cast("B"))
        if raw:
            library().SparkCk128Update(ctypes.byref(self.context), raw, len(raw))

    def digest(self) -> bytes:
        out = (ctypes.c_uint8 * 16)()
        library().SparkCk128Finalize(ctypes.byref(self.context), out)
        return bytes(out)


def ck128(data) -> bytes:
    state = Ck128()
    state.update(data)
    return state.digest()


def ck128_file(path) -> bytes:
    state = Ck128()
    with open(path, "rb") as handle:
        while True:
            piece = handle.read(READ_CHUNK)
            if not piece:
                return state.digest()
            state.update(piece)


if __name__ == "__main__":
    for name in sys.argv[1:]:
        print(f"{ck128_file(name).hex()}  {name}")
