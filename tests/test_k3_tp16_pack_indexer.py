#!/usr/bin/env python3
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import k3_tp16_pack as pack
from spark_pack_common import PackFailure

DSA = {"index_topk": 2048, "index_num_heads": 32, "index_head_dim": 128, "index_rope_dim": 0, "index_topk_freq": 1, "index_k_norm_type": "layernorm"}
MLA_LAYERS = [layer for layer in range(93) if layer % 4 == 3 or layer == 92]


class Tensors:
    def __init__(self, tensors):
        self.tensors = tensors

    def bf16(self, name, shape=None):
        if name not in self.tensors:
            raise PackFailure(f"missing tensor: {name}")
        values = self.tensors[name]
        if shape is not None and tuple(values.shape) != tuple(shape):
            raise PackFailure(f"{name} is {values.shape}, not {shape}")
        return values.tobytes()


class Reader:
    def __init__(self, tensors):
        self.st = Tensors(tensors)
        self.names = set(tensors)


class Packs:
    def __init__(self, ranks):
        self.ranks = ranks
        self.added = {rank: {} for rank in ranks}

    def replicated(self, name, array, kind):
        for rank in self.ranks:
            self.added[rank][name] = (np.ascontiguousarray(array).tobytes(), kind, tuple(array.shape))


def indexer_names(layers):
    return [f"{pack.S}layers.{layer}.self_attn.indexer.{source}" for layer in layers for _, source, _ in pack.INDEXER]


def refused(call):
    try:
        call()
    except PackFailure as error:
        return str(error)
    return None


def main():
    failures = []
    if pack.indexed({}) or pack.indexed({"dsa_config": None}):
        failures.append("a checkpoint without dsa_config (Kimi K3) must pack without an indexer")
    if not pack.indexed({"dsa_config": dict(DSA)}):
        failures.append("Kimi K3.1 Pro's dsa_config must pack the indexer")
    for change in ({"index_topk": 1024}, {"index_num_heads": 64}, {"index_head_dim": 64}, {"index_rope_dim": 64}, {"index_topk_freq": 4},
                   {"index_k_norm_type": "rmsnorm"}):
        if refused(lambda: pack.indexed({"dsa_config": dict(DSA, **change)})) is None:
            failures.append(f"dsa_config with {change} must be refused, not packed as K3.1's indexer")
    names = indexer_names(MLA_LAYERS)
    if len(names) != 120:
        failures.append(f"K3.1 Pro carries 120 indexer tensors, five on each of 24 MLA layers; the packer expects {len(names)}")
    if refused(lambda: pack.check_indexer_census(set(names) | {"other.weight"}, True)) is not None:
        failures.append("one indexer on every MLA layer must pass the census")
    if refused(lambda: pack.check_indexer_census({"other.weight"}, False)) is not None:
        failures.append("Kimi K3 with no indexer tensors must pass the census")
    for label, census, indexer in (("a missing MLA layer's indexer", set(names[5:]), True),
                                   ("an indexer on a KDA layer", set(names) | set(indexer_names([0])[:1]), True),
                                   ("indexer tensors without dsa_config", set(names), False)):
        if refused(lambda: pack.check_indexer_census(census, indexer)) is None:
            failures.append(f"{label} must be refused")
    rng = np.random.default_rng(31)
    a = f"{pack.S}layers.3.self_attn."
    tensors = {a + "indexer." + source: rng.integers(0, 1 << 16, size=shape, dtype=np.uint16) for _, source, shape in pack.INDEXER}
    packs = Packs([0, 3, 7])
    pack.pack_indexer(Reader(tensors), packs, a, "layers.3.")
    for rank in packs.ranks:
        for target, source, shape in pack.INDEXER:
            entry = packs.added[rank].get("layers.3." + target)
            if entry is None:
                failures.append(f"rank {rank} lacks layers.3.{target}")
                continue
            raw, kind, packed_shape = entry
            if kind != "bf16" or packed_shape != shape or raw != tensors[a + "indexer." + source].tobytes():
                failures.append(f"rank {rank}'s layers.3.{target} must be the checkpoint's {source} bit for bit as bf16 {shape}")
    short = dict(tensors)
    short[a + "indexer.wk.weight"] = short[a + "indexer.wk.weight"][:, :64]
    if refused(lambda: pack.pack_indexer(Reader(short), Packs([0]), a, "layers.3.")) is None:
        failures.append("an indexer tensor of the wrong shape must be refused")
    if pack.MODELS[False][:2] != ("kimi-k3", "k3") or pack.MODELS[True][:2] != ("kimi-k3.1-pro", "k31"):
        failures.append("K3 packs keep their model and file names; K3.1 Pro packs are kimi-k3.1-pro in k31 files")
    for failure in failures:
        print("FAIL", failure)
    if failures:
        return 1
    print("PASS k3 tp pack indexer: K3.1 Pro's dsa_config packs one replicated, bit-preserved bf16 indexer per MLA layer; other indexers, "
          "missing or stray indexer tensors and wrong shapes are refused; K3 packs without one")
    return 0


if __name__ == "__main__":
    sys.exit(main())
