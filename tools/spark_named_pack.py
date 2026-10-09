"""Named-tensor stage pack container shared by every model packer.

Layout: a 16-byte header (u32 magic, u32 version, u64 manifest bytes), the
JSON manifest {"format": ..., "config": ..., "tensors": {name: {offset,
bytes, align, kind, shape, ...}}}, zero padding to the alignment, then the
payload. Tensor offsets are relative to the payload base. The runtime reader
is runtime/named_pack.c.

The writer streams payload to <out>.payload with a journal: every entry is
appended to <out>.payload.journal only after its bytes are on disk, so a
killed run resumes by re-walking the journal and truncating to the last
complete tensor. Emission order must be deterministic for that to hold.
"""

import json
import os
import struct
from pathlib import Path

NAMED_PACK_MAGIC = 0x504E5053
NAMED_PACK_VERSION = 1
NAMED_PACK_ALIGN = 128


class PayloadWriter:
    def __init__(self, out_path, resume=False, align=NAMED_PACK_ALIGN):
        out_path = Path(str(out_path)).resolve()
        if ".." in out_path.parts:
            raise ValueError(f"rejecting path with ..: {out_path}")
        self.align = align
        self.journal_path = str(out_path.parent / (out_path.name + ".journal"))
        self.manifest = {}
        self.offset = 0
        if resume and os.path.exists(self.journal_path):
            with open(self.journal_path, "r", encoding="utf-8") as journal:
                for line in journal:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    self.manifest[record["name"]] = record["entry"]
                    self.offset = record["end"]
            self.handle = out_path.open("r+b")
            self.handle.truncate(self.offset)
            self.handle.seek(0, 2)
        else:
            self.handle = out_path.open("wb")
            Path(self.journal_path).write_text("", encoding="utf-8")
        self.journal = Path(self.journal_path).open("a", encoding="utf-8")

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
        self.journal.write(json.dumps({"name": name, "entry": entry, "end": self.offset},
                                      separators=(",", ":")) + "\n")
        self.journal.flush()

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
