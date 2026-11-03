"""Named-tensor stage pack container shared by every model packer.

Layout: a 16-byte header (u32 magic, u32 version, u64 manifest bytes), the
JSON manifest {"format": ..., "config": ..., "tensors": {name: {offset,
bytes, align, kind, shape, ...}}}, zero padding to the alignment, then the
payload. Tensor offsets are relative to the payload base. The runtime reader
is runtime/named_pack.c.

The writer streams payload to <out>.payload with a journal whose first line
names the source the pack is cut from: every entry is appended to
<out>.payload.journal only after its bytes are fsynced, and the journal record
is fsynced too, so a killed run resumes by re-walking the journal and
truncating to the last complete tensor. A resume refuses another source and a
payload shorter than its journal. Emission order must be deterministic.
"""

import hashlib
import json
import os
import struct
from pathlib import Path

NAMED_PACK_MAGIC = 0x504E5053
NAMED_PACK_VERSION = 1
NAMED_PACK_ALIGN = 128


def file_sha256(path):
    path = Path(str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def source_identity(source, *parts, files=()):
    source = Path(str(source)).resolve()
    return json.dumps({"source": str(source), "config_sha256": file_sha256(source / "config.json" if source.is_dir() else source),
                       "parts": [str(part) for part in parts], "files": [file_sha256(path) for path in files]}, sort_keys=True)


class PayloadWriter:
    def __init__(self, out_path, identity, resume=False, align=NAMED_PACK_ALIGN):
        out_path = Path(str(out_path)).resolve()
        if ".." in out_path.parts:
            raise ValueError(f"rejecting path with ..: {out_path}")
        if not identity:
            raise ValueError("a pack payload needs the identity of the source it is cut from")
        self.align = align
        self.journal_path = str(out_path.parent / (out_path.name + ".journal"))
        self.manifest = {}
        self.offset = 0
        if resume and os.path.exists(self.journal_path):
            with open(self.journal_path, "r", encoding="utf-8") as journal:
                lines = [line for line in journal if line.strip()]
            if not lines or json.loads(lines[0]).get("identity") != identity:
                raise ValueError(f"{self.journal_path} was written from another source; refusing to resume it from {identity}")
            for line in lines[1:]:
                record = json.loads(line)
                self.manifest[record["name"]] = record["entry"]
                self.offset = record["end"]
            if os.path.getsize(out_path) < self.offset:
                raise ValueError(f"{out_path} holds {os.path.getsize(out_path)} bytes, fewer than the {self.offset} its journal records; refusing to resume")
            self.handle = out_path.open("r+b")
            self.handle.truncate(self.offset)
            self.handle.seek(0, 2)
        else:
            self.handle = out_path.open("wb")
            Path(self.journal_path).write_text(json.dumps({"identity": identity}) + "\n", encoding="utf-8")
        self.journal = Path(self.journal_path).open("a", encoding="utf-8")

    def commit(self, name, entry):
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.journal.write(json.dumps({"name": name, "entry": entry, "end": self.offset}, separators=(",", ":")) + "\n")
        self.journal.flush()
        os.fsync(self.journal.fileno())

    def add(self, name, payload, kind, shape, extra=None):
        if name in self.manifest:
            return
        pad = (-self.offset) % self.align
        if pad:
            self.handle.write(b"\0" * pad)
            self.offset += pad
        raw = bytes(payload)
        entry = {"offset": self.offset, "bytes": len(raw), "align": self.align,
                 "kind": kind, "shape": list(shape)}
        if extra:
            entry.update(extra)
        self.manifest[name] = entry
        self.handle.write(raw)
        self.offset += len(raw)
        self.commit(name, entry)

    def add_parts(self, name, parts, kind, shape, extra=None):
        if name in self.manifest:
            return
        pad = (-self.offset) % self.align
        if pad:
            self.handle.write(b"\0" * pad)
            self.offset += pad
        start = self.offset
        for part in parts:
            view = memoryview(part).cast("B")
            self.handle.write(view)
            self.offset += len(view)
        entry = {"offset": start, "bytes": self.offset - start, "align": self.align,
                 "kind": kind, "shape": list(shape)}
        if extra:
            entry.update(extra)
        self.manifest[name] = entry
        self.commit(name, entry)

    def close(self):
        self.journal.close()
        self.handle.close()
        os.unlink(self.journal_path)


def assemble(out_path, payload_path, magic, version, align, fmt, config, tensors):
    manifest = json.dumps({"format": fmt, "config": config, "tensors": tensors},
                          separators=(",", ":")).encode()
    out_final = Path(str(out_path)).resolve()
    if ".." in out_final.parts:
        raise ValueError(f"rejecting path with ..: {out_path}")
    with out_final.open("wb") as out:
        out.write(struct.pack("<IIQ", magic, version, len(manifest)))
        out.write(manifest)
        out.write(b"\0" * ((-out.tell()) % align))
        with Path(str(payload_path)).open("rb") as body:
            while True:
                chunk = body.read(1 << 24)
                if not chunk:
                    break
                out.write(chunk)
    Path(str(payload_path)).unlink()
