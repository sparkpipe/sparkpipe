#!/usr/bin/env python3
"""CPU verifier: glm53full BF16 TP16 stage packs vs the promoted source.

Answers the operator question "is the stagepack accurate?" without a GPU:
the bf16 arm's contract is BYTE EXACTNESS (no quantization anywhere —
publisher BF16 is the payload), so every check is a byte comparison.

  A. expert verbatim   - sampled (layer, rank, expert, projection) payload
                         blocks equal the source safetensors slice, for
                         up/gate (row-sharded) and down (col-sharded)
  B. full-expert sweep - ONE (layer, rank) walks ALL 256 experts of both
                         expert kinds byte-for-byte
  C. spine exactness   - sampled spine kinds (embedding/lm_head row
                         shards, q_a, o_proj col shard, router, norms)
                         equal the source tensor slice
  D. replicated identity - replicated kinds byte-identical across all 16
                         ranks (router, attn_norm, q_a)
  E. tp16 partition    - the 16 rank slices of one expert tensor
                         reassemble the full source tensor exactly

Source layout: model.safetensors.index.json + shards; experts are
individual 2-D BF16 tensors
model.layers.{L}.mlp.experts.{e}.{up,gate,down}_proj.weight (the packer's
add_experts is the layout authority: up rows stacked above gate rows,
expert-major payload order, rank r takes rows [r*R/16,(r+1)*R/16) of
up/gate and cols [r*C/16,(r+1)*C/16) of down).

Exit 0 = all checks PASS; any FAIL prints the offending tensor and exits 1.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC, FORMAT_VERSION, HEADER_BYTES, ENTRY_BYTES = 0x32534C47, 3, 264, 64
GLOBAL_LAYER = 0xFFFFFFFF
PAYLOAD_BF16, PAYLOAD_F32 = 1, 2

K_EMBEDDING, K_LM_HEAD, K_ATTN_NORM, K_Q_A, K_KV_A, K_ATTN_OUTPUT = 0, 2, 3, 4, 7, 11
K_POST_ATTN_NORM, K_ROUTER, K_ROUTER_CORR = 12, 20, 21
K_EXPERT_UP_GATE, K_EXPERT_DOWN, K_SHARED_GATE_UP, K_SHARED_DOWN = 22, 23, 24, 25

KIND_NAME = {0: "EMBEDDING", 2: "LM_HEAD", 3: "ATTN_NORM", 4: "Q_A",
             7: "KV_A", 11: "ATTN_OUTPUT", 12: "POST_ATTN_NORM",
             20: "ROUTER", 21: "ROUTER_CORRECTION", 22: "EXPERT_UP_GATE",
             23: "EXPERT_DOWN", 24: "SHARED_GATE_UP", 25: "SHARED_DOWN"}

HIDDEN, VOCAB, MOE_INT, EXPERTS = 6144, 154880, 2048, 256
FIRST_ROUTED = 3


def load_source(root: Path):
    index = json.loads((root / "model.safetensors.index.json").read_text())
    shard_cache: dict[str, tuple[np.memmap, dict]] = {}
    tensors = {}
    for name, rel in index["weight_map"].items():
        if rel not in shard_cache:
            path = root / rel
            with open(path, "rb") as handle:
                header_len = struct.unpack("<Q", handle.read(8))[0]
                header = json.loads(handle.read(header_len))
            shard_cache[rel] = (np.memmap(path, dtype=np.uint8, mode="r"),
                                header, 8 + header_len)
        _base, header, data_off = shard_cache[rel]
        meta = header[name]
        tensors[name] = (meta["dtype"], tuple(meta["shape"]),
                         _base, data_off + meta["data_offsets"][0])
    return tensors


class PackReader:
    def __init__(self, path: Path):
        self.path = path
        with open(path, "rb") as handle:
            head = handle.read(HEADER_BYTES)
        vals = struct.unpack_from("<20I", head, 0)
        assert vals[0] == MAGIC and vals[1] == FORMAT_VERSION, "bad pack magic/version"
        self.tensor_count = vals[6]
        self.expert_codec = vals[16]
        self.tp_degree, self.tp_rank = vals[18], vals[19]
        dir_off, self.file_bytes = struct.unpack_from("<2Q", head, 80)
        self.entries = []
        with open(path, "rb") as handle:
            handle.seek(dir_off)
            raw = handle.read(self.tensor_count * ENTRY_BYTES)
        for i in range(self.tensor_count):
            e = struct.unpack_from("<8I4Q", raw, i * ENTRY_BYTES)
            kind, layer = e[0], e[1]
            rows, cols, group_count = e[6], e[7], e[5]
            payload_offset, payload_bytes = e[8], e[9]
            self.entries.append((kind, layer, group_count, rows, cols,
                                 payload_offset, payload_bytes))
        self.by_key = {(e[0], e[1]): e for e in self.entries}
        self.mmap = np.memmap(path, dtype=np.uint8, mode="r")

    def payload(self, kind: int, layer: int) -> np.ndarray:
        ent = self.by_key[(kind, layer)]
        return self.mmap[ent[5]:ent[5] + ent[6]]


def expert_block(source, name, r0, r1, c0, c1) -> np.ndarray:
    dtype, shape, base, off = source[name]
    assert dtype == "BF16" and len(shape) == 2, f"{name}: {dtype} {shape}"
    matrix = base[off:off + shape[0] * shape[1] * 2].view(np.uint16).reshape(shape)
    return matrix[r0:r1, c0:c1]


def check_expert_verbatim(pack, source, layer, rank, experts, verbose):
    failures = 0
    tp = pack.tp_degree
    up_rows = MOE_INT // tp            # 128 at tp16
    down_cols = MOE_INT // tp
    for kind, projs, shard in ((K_EXPERT_UP_GATE, ("up_proj", "gate_proj"), "rows"),
                               (K_EXPERT_DOWN, ("down_proj",), "cols")):
        ent = pack.by_key[(kind, layer)]
        payload = pack.payload(kind, layer)
        # entry rows = len(projs) * shard_rows; per-expert payload splits
        # evenly across projections, each one contiguous:
        per_expert_bytes = ent[6] // EXPERTS
        proj_bytes = per_expert_bytes // len(projs)
        for e in experts:
            for pi, proj in enumerate(projs):
                name = f"model.layers.{layer}.mlp.experts.{e}.{proj}.weight"
                if shard == "rows":
                    block = expert_block(source, name, rank * up_rows, (rank + 1) * up_rows, 0, HIDDEN)
                else:
                    block = expert_block(source, name, 0, HIDDEN, rank * down_cols, (rank + 1) * down_cols)
                off = e * per_expert_bytes + pi * proj_bytes
                got = payload[off:off + proj_bytes].view(np.uint16).reshape(block.shape)
                if not np.array_equal(got, block):
                    where = np.argwhere(got != block)[:3]
                    print(f"  FAIL {KIND_NAME[kind]} L{layer} rank{rank} "
                          f"expert{e}.{proj}: first diffs at {where.tolist()}")
                    failures += 1
        if verbose:
            print(f"  pass {KIND_NAME[kind]} L{layer} rank{rank}: "
                  f"{len(experts)} experts byte-exact")
    return failures


def check_spine(pack, source, kind, layer, name, shard, verbose):
    ent = pack.by_key[(kind, layer)]
    payload = pack.payload(kind, layer)
    dtype, shape, base, off = source[name]
    assert dtype == "BF16", f"{name}: {dtype}"
    matrix = base[off:off + shape[0] * shape[1] * 2].view(np.uint16).reshape(shape) \
        if len(shape) == 2 else base[off:off + shape[0] * 2].view(np.uint16)
    tp, rank = pack.tp_degree, pack.tp_rank
    if shard == "rows" and len(shape) == 2:
        s = rank * (shape[0] // tp)
        want = matrix[s:s + shape[0] // tp, :]
    elif shard == "cols" and len(shape) == 2:
        s = rank * (shape[1] // tp)
        want = matrix[:, s:s + shape[1] // tp]
    elif len(shape) == 1:
        want = matrix.reshape(1, -1)
    else:
        want = matrix
    got = payload[:want.size * 2].view(np.uint16).reshape(want.shape)
    if np.array_equal(got, want):
        if verbose:
            print(f"  pass {KIND_NAME[kind]} L{layer} {name}: "
                  f"{want.shape} byte-exact (shard={shard or 'none'})")
        return 0
    where = np.argwhere(got != want)[:3]
    print(f"  FAIL {KIND_NAME[kind]} L{layer} {name}: diffs at {where.tolist()} "
          f"(pack {got.shape} vs source slice {want.shape})")
    return 1


def check_router_f32(pack, source, layer, name, verbose):
    ent = pack.by_key[(K_ROUTER_CORR, layer)]
    payload = pack.payload(K_ROUTER_CORR, layer)
    dtype, shape, base, off = source[name]
    want = base[off:off + shape[0] * 4].view(np.float32)
    got = payload[:shape[0] * 4].view(np.float32)
    if np.array_equal(got, want):
        if verbose:
            print(f"  pass ROUTER_CORR L{layer} {name}: f32 byte-exact")
        return 0
    print(f"  FAIL ROUTER_CORR L{layer} {name}")
    return 1


def check_replicated(packs, kinds_layers, verbose):
    failures = 0
    for kind, layer in kinds_layers:
        reference = packs[0].payload(kind, layer)
        for pack in packs[1:]:
            if not np.array_equal(reference, pack.payload(kind, layer)):
                print(f"  FAIL replicated {KIND_NAME[kind]} L{layer}: "
                      f"rank{pack.tp_rank} differs from rank0")
                failures += 1
        if verbose:
            print(f"  pass replicated {KIND_NAME[kind]} L{layer}: "
                  f"16/16 byte-identical")
    return failures


def check_tp16_partition(packs, source, layer, expert, proj, verbose):
    full = expert_block(source,
                        f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight",
                        0, MOE_INT, 0, HIDDEN)
    parts = []
    for pack in packs:
        rank = pack.tp_rank
        ent = pack.by_key[(K_EXPERT_UP_GATE, layer)]
        payload = pack.payload(K_EXPERT_UP_GATE, layer)
        per_expert_bytes = ent[6] // EXPERTS
        proj_bytes = per_expert_bytes // 2
        pi = 0 if proj == "up_proj" else 1
        off = expert * per_expert_bytes + pi * proj_bytes
        parts.append(payload[off:off + proj_bytes].view(np.uint16)
                     .reshape(MOE_INT // 16, HIDDEN))
    reassembled = np.concatenate(parts, axis=0)
    if np.array_equal(reassembled, full):
        if verbose:
            print(f"  pass tp16 partition L{layer} expert{expert}.{proj}: "
                  f"16 rank slices reassemble the source exactly")
        return 0
    where = np.argwhere(reassembled != full)[:3]
    print(f"  FAIL tp16 partition L{layer} expert{expert}.{proj}: {where.tolist()}")
    return 1


def rank_filename(packs_dir: Path, rank: int) -> Path:
    hex_name = packs_dir / f"glm53full.bf16.tp16-rank{rank:x}.glm52sp"
    dec_name = packs_dir / f"glm53full.bf16.tp16-rank{rank}.glm52sp"
    return hex_name if hex_name.exists() else dec_name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="/mnt/model-warm/glm-5.3-bf16")
    parser.add_argument("--packs-dir", default=str(
        Path.home() / "sparkdata/glm53full.bf16.tp16/packs"))
    parser.add_argument("--expert-layers", default="3,40,77")
    parser.add_argument("--full-sweep-rank", type=int, default=0)
    args = parser.parse_args()

    source = load_source(Path(args.source))
    packs = [PackReader(rank_filename(Path(args.packs_dir), r)) for r in range(16)]
    tp_set = {p.tp_degree for p in packs}
    ranks_set = {p.tp_rank for p in packs}
    assert tp_set == {16}, f"tp degrees {tp_set}"
    assert ranks_set == set(range(16)), f"ranks {sorted(ranks_set)}"
    assert packs[0].expert_codec == 1, "not a bf16 (codec 1) pack set"
    print(f"packs: 16/16 opened, tp16, expert codec 1 (bf16)")

    failures = 0
    expert_layers = [int(x) for x in args.expert_layers.split(",")]

    # A: sampled experts on all ranks, every sampled layer
    sample = [0, 1, 7, 63, 128, 200, 255]
    for layer in expert_layers:
        for pack in packs:
            failures += check_expert_verbatim(pack, source, layer,
                                              pack.tp_rank, sample, True)
    # B: full 256-expert sweep on one (layer, rank)
    failures += check_expert_verbatim(packs[args.full_sweep_rank], source,
                                      expert_layers[0],
                                      args.full_sweep_rank,
                                      list(range(EXPERTS)), True)
    # C: spine kinds (rows shard, cols shard, unsharded, replicated, f32)
    p0 = packs[0]
    failures += check_spine(p0, source, K_EMBEDDING, GLOBAL_LAYER,
                            "model.embed_tokens.weight", "rows", True)
    failures += check_spine(p0, source, K_LM_HEAD, GLOBAL_LAYER,
                            "lm_head.weight", "rows", True)
    failures += check_spine(p0, source, K_Q_A, 40,
                            "model.layers.40.self_attn.q_a_proj.weight", "", True)
    failures += check_spine(p0, source, K_ATTN_OUTPUT, 40,
                            "model.layers.40.self_attn.o_proj.weight", "cols", True)
    failures += check_spine(p0, source, K_ROUTER, 40,
                            "model.layers.40.mlp.gate.weight", "", True)
    failures += check_router_f32(p0, source, 40,
                                 "model.layers.40.mlp.gate.e_score_correction_bias", True)
    # D: replicated identity across ranks
    failures += check_replicated(packs, [(K_ROUTER, 40), (K_ATTN_NORM, 40),
                                         (K_Q_A, 40)], True)
    # E: full TP16 partition of one expert tensor
    failures += check_tp16_partition(packs, source, expert_layers[0], 5,
                                     "up_proj", True)

    if failures:
        print(f"RESULT: FAIL — {failures} check(s) failed")
        return 1
    print("RESULT: PASS — bf16 TP16 packs are byte-exact against the source")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
