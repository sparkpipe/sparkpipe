import hashlib
import json
import os
import re
import struct
import zlib
from collections import OrderedDict

import numpy as np

T1R_MAGIC = b"T1R1"


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_u16(x):
    u = x.astype(np.float32).view(np.uint32).copy()
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return rounded.astype(np.uint16)


def bf16_round_f32(x):
    return bf16_to_f32(f32_to_bf16_u16(x))


_E4M3_LUT = np.zeros(256, dtype=np.float32)
for _i in range(256):
    _s = -1.0 if _i & 0x80 else 1.0
    _e = (_i >> 3) & 0xF
    _m = _i & 0x7
    if _e == 0:
        _v = _m / 8.0 * 2.0 ** -6
    elif _e == 15 and _m == 7:
        _v = np.nan
    else:
        _v = (1.0 + _m / 8.0) * 2.0 ** (_e - 7)
    _E4M3_LUT[_i] = _s * _v

_E8M0_LUT = (2.0 ** np.arange(-127, 128, dtype=np.float64)).astype(np.float32)

_E2M1_LUT = np.zeros(16, dtype=np.float32)
for _i in range(16):
    _s = -1.0 if _i & 0x8 else 1.0
    _e = (_i >> 1) & 0x3
    _m = _i & 0x1
    _E2M1_LUT[_i] = _s * ((1.0 + _m / 2.0) * 2.0 ** _e if _e else _m / 2.0)


def fp8_block_to_bf16(payload_u8, scale_inv, out_dim, in_dim):
    w = _E4M3_LUT[payload_u8.reshape(out_dim, in_dim)].astype(np.float32)
    s = np.repeat(np.repeat(scale_inv, 128, axis=0), 128, axis=1)
    s = s[:out_dim, :in_dim]
    return f32_to_bf16_u16(w * s)


def nvfp4_to_f32(payload_u8, scale_e4m3, out_dim, in_dim):
    if in_dim % 16 != 0:
        raise ValueError(f"nvfp4 input dim {in_dim} not a multiple of 16")
    groups = in_dim // 16
    scales = _E4M3_LUT[scale_e4m3.reshape(out_dim, groups)].astype(np.float32)
    w = np.empty((out_dim, groups, 16), dtype=np.float32)
    w[:, :, 0::2] = _E2M1_LUT[(payload_u8 & 0xF).reshape(out_dim, groups, 8)]
    w[:, :, 1::2] = _E2M1_LUT[(payload_u8 >> 4).reshape(out_dim, groups, 8)]
    return (w * scales[:, :, None]).reshape(out_dim, in_dim)


def mxfp4_to_bf16(payload_u8, scale_e8m0, out_dim, in_dim):
    if in_dim % 32 != 0:
        raise ValueError(f"mxfp4 input dim {in_dim} not a multiple of 32")
    lo = (payload_u8 & 0xF).astype(np.int32)
    hi = (payload_u8 >> 4).astype(np.int32)
    nib = np.empty((out_dim, in_dim), dtype=np.int32)
    nib[:, 0::2] = lo
    nib[:, 1::2] = hi
    w = _E2M1_LUT[nib].reshape(out_dim, in_dim // 32, 32).astype(np.float32)
    s = _E8M0_LUT[scale_e8m0.reshape(out_dim, in_dim // 32).astype(np.int32) + 127]
    return f32_to_bf16_u16((w * s[:, :, None]).reshape(out_dim, in_dim))


class Safetensors:
    def __init__(self, root):
        self.root = root
        index_path = os.path.join(root, "model.safetensors.index.json")
        if os.path.exists(index_path):
            self.map = json.load(open(index_path))["weight_map"]
        else:
            self.map = {}
        self.headers = {}
        self.fds = {}
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.cache_limit = int(os.environ.get("T1_REF_CACHE_BYTES",
                                              80 * (1 << 30)))

    def _open(self, fname):
        if fname not in self.fds:
            fh = open(os.path.join(self.root, fname), "rb")
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
            self.headers[fname] = (hdr, 8 + n)
            self.fds[fname] = fh
        return self.headers[fname]

    def _entry(self, name):
        fname = self.map.get(name, "model.safetensors")
        hdr, base = self._open(fname)
        if name not in hdr:
            raise KeyError(f"tensor {name} absent from checkpoint {self.root}")
        return fname, hdr[name], base

    @staticmethod
    def _np(dt):
        return {"BF16": np.uint16, "F32": np.float32, "F16": np.float16,
                "U8": np.uint8, "F8_E4M3": np.uint8, "I64": np.int64}[dt]

    def entry(self, name):
        _, e, _ = self._entry(name)
        return e

    def raw(self, name):
        if name in self.cache:
            self.cache.move_to_end(name)
            return self.cache[name]
        fname, e, base = self._entry(name)
        fh = self.fds[fname]
        fh.seek(base + e["data_offsets"][0])
        data = fh.read(e["data_offsets"][1] - e["data_offsets"][0])
        array = np.frombuffer(data, dtype=self._np(e["dtype"])).reshape(e["shape"])
        self.cache[name] = array
        self.cache_bytes += array.nbytes
        while self.cache_bytes > self.cache_limit and len(self.cache) > 1:
            _, evicted = self.cache.popitem(last=False)
            self.cache_bytes -= evicted.nbytes
        return array

    def raw_rows(self, name, first, count):
        fname, e, base = self._entry(name)
        shape = e["shape"]
        if len(shape) != 2 or first < 0 or count <= 0 or first + count > shape[0]:
            raise ValueError(f"row range {first}+{count} outside {name} {shape}")
        dtype = np.dtype(self._np(e["dtype"]))
        stride = shape[1] * dtype.itemsize
        if e["data_offsets"][1] - e["data_offsets"][0] != shape[0] * stride:
            raise ValueError(f"extent disagrees with shape for {name}")
        fh = self.fds[fname]
        fh.seek(base + e["data_offsets"][0] + first * stride)
        data = fh.read(count * stride)
        if len(data) != count * stride:
            raise ValueError(f"truncated row range for {name}")
        return np.frombuffer(data, dtype=dtype).reshape(count, shape[1])

    def raw_slab(self, name, first, count):
        fname, e, base = self._entry(name)
        shape = e["shape"]
        if len(shape) < 2 or first < 0 or count <= 0 \
                or first + count > shape[0]:
            raise ValueError(f"slab range {first}+{count} outside {name} "
                             f"{shape}")
        dtype = np.dtype(self._np(e["dtype"]))
        slab = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
        if e["data_offsets"][1] - e["data_offsets"][0] != shape[0] * slab:
            raise ValueError(f"extent disagrees with shape for {name}")
        fh = self.fds[fname]
        fh.seek(base + e["data_offsets"][0] + first * slab)
        data = fh.read(count * slab)
        if len(data) != count * slab:
            raise ValueError(f"truncated slab range for {name}")
        return np.frombuffer(data, dtype=dtype).reshape([count] + shape[1:])


DEFINE_RE = re.compile(r"^#define\s+SPARK_LLM_([A-Z0-9_]+)\s+(.+?)[ \t]*$", re.M)
ANY_DEFINE_RE = re.compile(r"^#define\s+([A-Z][A-Z0-9_]+)\s+(.+?)[ \t]*$", re.M)


def parse_llm_defines(path):
    text = open(path).read()
    graph = {}
    for name, value in ANY_DEFINE_RE.findall(text):
        graph[name] = value.strip()
    defines = {}
    for name, value in DEFINE_RE.findall(text):
        value = value.strip()
        if value.startswith("SET_ME_"):
            raise ValueError(f"{path}: SPARK_LLM_{name} is unset ({value})")
        resolved = value
        for _ in range(8):
            if resolved not in graph:
                break
            resolved = graph[resolved]
        else:
            raise ValueError(f"{path}: SPARK_LLM_{name} indirection deeper than 8")
        if resolved.startswith("SET_ME_"):
            raise ValueError(f"{path}: SPARK_LLM_{name} is unset ({resolved})")
        defines[name] = resolved
    required = ["FAMILY_TAG", "HIDDEN_DIMENSION", "LAYER_COUNT", "OUTPUT_VOCAB_COUNT",
                "RMS_NORM_EPSILON"]
    for name in required:
        if name not in defines:
            raise ValueError(f"{path}: SPARK_LLM_{name} missing")
    return defines


def _resolve_define(defines, name):
    v = defines[name]
    for _ in range(8):
        if not v.startswith("SPARK_LLM_"):
            return v
        v = defines[v[len("SPARK_LLM_"):]]
    raise ValueError(f"define {name} indirection deeper than 8")


def define_uint(defines, name):
    v = _resolve_define(defines, name)
    return int(v[:-1] if v.endswith("u") else v, 0)


def define_float(defines, name):
    v = _resolve_define(defines, name)
    if v.endswith("f"):
        v = v[:-1]
    return float(v)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_fixture(path, arrays):
    meta = {"format": "T1R1", "arrays": []}
    blobs = []
    for name in sorted(arrays):
        a = arrays[name]
        if a.dtype == np.float32:
            dt = "F32"
        elif a.dtype == np.uint16:
            dt = "BF16"
        elif a.dtype == np.int32:
            dt = "I32"
        elif a.dtype == np.uint32:
            dt = "U32"
        else:
            raise ValueError(f"unsupported fixture dtype for {name}: {a.dtype}")
        raw = a.astype(a.dtype).tobytes()
        comp = zlib.compress(raw, 6)
        meta["arrays"].append({"name": name, "dtype": dt, "shape": list(a.shape),
                               "bytes": len(raw), "compressed": len(comp),
                               "sha256": hashlib.sha256(raw).hexdigest()})
        blobs.append(comp)
    meta_bytes = json.dumps(meta, sort_keys=True).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(T1R_MAGIC)
        fh.write(struct.pack("<Q", len(meta_bytes)))
        fh.write(meta_bytes)
        for comp in blobs:
            fh.write(struct.pack("<Q", len(comp)))
            fh.write(comp)


def read_fixture(path):
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:4] != T1R_MAGIC:
        raise ValueError(f"{path}: not a T1R1 fixture")
    n = struct.unpack("<Q", data[4:12])[0]
    meta = json.loads(data[12:12 + n])
    off = 12 + n
    arrays = {}
    for entry in meta["arrays"]:
        n = struct.unpack("<Q", data[off:off + 8])[0]
        comp = data[off + 8:off + 8 + n]
        off += 8 + n
        raw = zlib.decompress(comp)
        if len(raw) != entry["bytes"]:
            raise ValueError(f"{path}: {entry['name']} extent mismatch")
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError(f"{path}: {entry['name']} sha256 mismatch")
        dt = {"F32": np.float32, "BF16": np.uint16, "I32": np.int32,
              "U32": np.uint32}[entry["dtype"]]
        arrays[entry["name"]] = np.frombuffer(raw, dtype=dt).reshape(entry["shape"])
    return meta, arrays


def write_manifest(path, document):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(document, indent=2, sort_keys=True) + "\n")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def rmsnorm(x, weight, epsilon):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + epsilon) * weight
