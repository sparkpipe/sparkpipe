#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

FORMAT = "sparkpipe-score-merged-v1"
EXACT_FORMAT = "sparkpipe-score-exact-kl-v1"
REQUIRED = {
    "doc": (np.uint32, 1), "pos": (np.uint32, 1), "target": (np.uint32, 1), "lse": (np.float64, 1),
    "target_logit": (np.float32, 1), "top_ids": (np.uint32, 2), "top_logits": (np.float32, 2),
    "probe_logits": (np.float32, 2),
}
OPTIONAL = {"select_hash": (np.uint64, 2)}


class DumpError(ValueError):
    pass


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_sha256(top_ids: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(top_ids, dtype="<u4").tobytes()).hexdigest()


def _write_npz(path, payload: dict) -> str:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(payload):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            member = io.BytesIO()
            np.lib.format.write_array(member, np.asarray(payload[name]), allow_pickle=False, version=(1, 0))
            archive.writestr(info, member.getvalue())
    Path(path).write_bytes(buffer.getvalue())
    return sha256_file(path)


def write(path, header: dict, arrays: dict) -> str:
    header = dict(header)
    header["format"] = FORMAT
    header["rows"] = int(len(arrays["doc"]))
    header["probe_k"] = int(arrays["top_ids"].shape[1])
    payload = {"header": np.array(canonical_json(header))}
    for name, (dtype, _) in {**REQUIRED, **OPTIONAL}.items():
        if name in arrays:
            payload[name] = np.ascontiguousarray(arrays[name], dtype=dtype)
    return _write_npz(path, payload)


def read(path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        if "header" not in archive.files:
            raise DumpError(f"{path}: no header array")
        header = json.loads(str(archive["header"]))
        if header.get("format") != FORMAT:
            raise DumpError(f"{path}: format {header.get('format')!r} is not {FORMAT}")
        out = {"header": header, "path": str(path), "sha256": sha256_file(path)}
        for name, (dtype, ndim) in REQUIRED.items():
            if name not in archive.files:
                raise DumpError(f"{path}: missing array {name}")
            out[name] = archive[name]
        for name in OPTIONAL:
            if name in archive.files:
                out[name] = archive[name]
    rows = header.get("rows")
    for name, (dtype, ndim) in {**REQUIRED, **OPTIONAL}.items():
        if name not in out:
            continue
        array = out[name]
        if array.dtype != np.dtype(dtype) or array.ndim != ndim or array.shape[0] != rows:
            raise DumpError(f"{path}: array {name} is {array.dtype}{array.shape}, expected {np.dtype(dtype)} with {rows} rows and {ndim} dims")
    k = header.get("probe_k")
    for name in ("top_ids", "top_logits", "probe_logits"):
        if out[name].shape[1] != k:
            raise DumpError(f"{path}: {name} has {out[name].shape[1]} columns, header probe_k={k}")
    return out


def write_exact(path, header: dict, rows: np.ndarray, kl: np.ndarray) -> str:
    header = dict(header)
    header["format"] = EXACT_FORMAT
    return _write_npz(path, {"header": np.array(canonical_json(header)), "rows": np.asarray(rows, dtype=np.uint32),
                             "kl": np.asarray(kl, dtype=np.float64)})


def read_exact(path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        header = json.loads(str(archive["header"]))
        if header.get("format") != EXACT_FORMAT:
            raise DumpError(f"{path}: format {header.get('format')!r} is not {EXACT_FORMAT}")
        return {"header": header, "rows": archive["rows"], "kl": archive["kl"], "sha256": sha256_file(path)}


def top_k(logits: np.ndarray, k: int) -> tuple:
    order = np.lexsort((np.broadcast_to(np.arange(logits.shape[1]), logits.shape), -logits), axis=1)[:, :k]
    return order.astype(np.uint32), np.take_along_axis(logits, order, axis=1)


def from_full_logits(logits: np.ndarray, doc: np.ndarray, pos: np.ndarray, target: np.ndarray,
                     probe_ids: np.ndarray | None, k: int) -> dict:
    logits32 = np.asarray(logits, dtype=np.float32)
    wide = logits32.astype(np.float64)
    top = wide.max(axis=1)
    lse = top + np.log(np.exp(wide - top[:, None]).sum(axis=1))
    ids, top_logits = top_k(logits32, k)
    probes = ids if probe_ids is None else probe_ids
    return {
        "doc": doc, "pos": pos, "target": target, "lse": lse,
        "target_logit": logits32[np.arange(len(target)), target],
        "top_ids": ids, "top_logits": top_logits,
        "probe_logits": np.take_along_axis(logits32, probes.astype(np.int64), axis=1),
    }
