from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Stream:
    name: str
    tokens: tuple[int, ...]
    prompt: int

    @property
    def length(self) -> int:
        return len(self.tokens)

    @property
    def generated(self) -> tuple[int, ...]:
        return self.tokens[self.prompt:]

    def sha256(self) -> str:
        return sha256_ids(self.tokens)


def sha256_ids(ids) -> str:
    return hashlib.sha256(struct.pack(f"<{len(ids)}I", *ids)).hexdigest()


def write_u32(path: Path, prompt: list[int], output: list[int]) -> None:
    tokens = list(prompt) + list(output)
    path.write_bytes(struct.pack(f"<II{len(tokens)}I", len(prompt), len(tokens), *tokens))


def read_u32(path: Path, name: str | None = None) -> Stream:
    data = path.read_bytes()
    if len(data) < 8 or len(data) % 4 != 0:
        raise ValueError(f"{path}: not a u32 stream file")
    prompt, total = struct.unpack("<II", data[:8])
    tokens = struct.unpack(f"<{(len(data) - 8) // 4}I", data[8:])
    if prompt == 0 or total <= prompt or total != len(tokens):
        raise ValueError(f"{path}: header prompt={prompt} total={total} does not match {len(tokens)} ids")
    return Stream(name or path.stem, tokens, prompt)


def class_of(name: str) -> str:
    return name.rsplit("_", 1)[0]
