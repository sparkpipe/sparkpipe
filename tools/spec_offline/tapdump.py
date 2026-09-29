from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .streams import Stream, read_u32, write_u32

FORMAT = "spark-tapdump-1"


@dataclass(frozen=True)
class DumpStream:
    stream: Stream
    content_class: str
    taps: np.ndarray | None


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    return (raw.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounding = ((bits >> 16) & 1) + 0x7FFF
    return ((bits + rounding) >> 16).astype(np.uint16)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_dump(directory: Path, model: str, firmware: str, tap_name: str, tap_layer: int, hidden: int,
               streams: list[tuple[str, str, list[int], list[int], np.ndarray | None]], notes: dict | None = None) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, content_class, prompt, output, taps in streams:
        token_path = directory / f"{name}.u32"
        write_u32(token_path, prompt, output)
        entry = {"name": name, "class": content_class, "tokens": token_path.name,
                 "prompt_tokens": len(prompt), "total_tokens": len(prompt) + len(output)}
        if taps is not None:
            if taps.shape != (len(prompt) + len(output), hidden):
                raise ValueError(f"{name}: taps shape {taps.shape} must be (total_tokens, hidden)")
            tap_path = directory / f"{name}.{tap_name}.bf16"
            f32_to_bf16(taps).tofile(tap_path)
            entry["taps"] = tap_path.name
        entries.append(entry)
    manifest = {"format": FORMAT, "model": model, "firmware": firmware, "hidden_dimension": hidden,
                "taps": [{"name": tap_name, "layer": tap_layer, "dtype": "bf16", "row_semantics": "row p is the tap after token p was committed"}],
                "streams": entries, "notes": notes or {}}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=1))
    lines = []
    for path in sorted(directory.iterdir()):
        if path.name != "SHA256SUMS":
            lines.append(f"{file_sha256(path)}  {path.name}")
    (directory / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    return manifest


def verify_sums(directory: Path) -> None:
    sums = directory / "SHA256SUMS"
    if not sums.exists():
        raise ValueError(f"{directory}: SHA256SUMS missing")
    for line in sums.read_text().splitlines():
        digest, name = line.split("  ", 1)
        path = directory / name
        if not path.exists():
            raise ValueError(f"{directory}: {name} listed in SHA256SUMS is missing")
        if file_sha256(path) != digest:
            raise ValueError(f"{directory}: {name} does not match SHA256SUMS")


def read_dump(directory: Path, want_taps: bool = True) -> tuple[dict, list[DumpStream]]:
    verify_sums(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError(f"{directory}: format {manifest.get('format')!r} is not {FORMAT}")
    hidden = int(manifest["hidden_dimension"])
    streams = []
    for entry in manifest["streams"]:
        stream = read_u32(directory / entry["tokens"], entry["name"])
        if stream.prompt != entry["prompt_tokens"] or stream.length != entry["total_tokens"]:
            raise ValueError(f"{entry['name']}: token file header disagrees with the manifest")
        taps = None
        if want_taps and "taps" in entry:
            raw = np.fromfile(directory / entry["taps"], dtype=np.uint16)
            if raw.size != stream.length * hidden:
                raise ValueError(f"{entry['name']}: tap file holds {raw.size} values, expected {stream.length * hidden}")
            taps = bf16_to_f32(raw.reshape(stream.length, hidden))
        streams.append(DumpStream(stream, entry["class"], taps))
    return manifest, streams
