#!/usr/bin/env python3
"""Verify a K3 V2 pack (stage or TP rank) - the P2 exit gate.

Structural pass (every pack):
  header magic/version, manifest parses, payload base 128-aligned, every
  tensor offset 128-aligned and non-overlapping inside the file, byte count
  consistent with kind+shape (bf16/f32 element size; expert tensors priced by
  interleave_geometry), fused sections tile their rows, config echo carries
  the shard geometry k3_shard.py requires, and every tensor's field lands in
  exactly one shard class (the sharder's "refuse to guess" table).

Cross pass (rank pack vs its stage pack, --stage-pack):
  the rank manifest must equal the shard-class split of the stage manifest:
  replicated tensors byte-identical (full sha256), row/section splits
  reconstructed from the stage payload and compared byte-for-byte, column
  splits rebuilt with the sharder's own interleave rule, expert tensors
  sampled at random (expert, k-tile) cells - a full compare of ~89 GB x 4
  ranks is NVMe-hours; 64 sampled cells per pack catches a mis-mapped tile.

Exit 0 = PASS, 1 = FAIL, with a one-line PASS/FAIL verdict naming counts.

usage:
  k3_verify_pack.py PACK [--stage-pack PATH] [--expect-tp-degree 4]
      [--expect-rank R] [--expect-first N] [--expect-layers N]
      [--expert-cells 64] [--seed 7] [--quick]
"""
import argparse
import hashlib
import json
import mmap
import random
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import k3_shard  # noqa: E402  the shard-class tables are the split contract
import k3_pack  # noqa: E402  interleave_geometry prices the expert tensors

MAGIC = 0x4B33504B
ALIGN = 128


class VerifyFailure(RuntimeError):
    pass


class PackView:
    """mmap'd V2 pack: manifest, config, byte-range access."""

    def __init__(self, path):
        self.path = Path(path)
        self.handle = open(path, "rb")
        self.raw = mmap.mmap(self.handle.fileno(), 0, access=mmap.ACCESS_READ)
        magic, version, length = struct.unpack_from("<IIQ", self.raw, 0)
        if magic != MAGIC:
            raise VerifyFailure(f"{path}: magic {magic:#x} is not a K3 pack")
        if version != 2:
            raise VerifyFailure(f"{path}: format version {version}, expected 2")
        self.manifest = json.loads(self.raw[16:16 + length])
        self.base = 16 + length
        self.base += (-self.base) % ALIGN
        self.config = self.manifest["config"]
        self.file_bytes = len(self.raw)

    def bytes_of(self, name):
        entry = self.manifest["tensors"][name]
        return self.raw[self.base + entry["offset"]:
                        self.base + entry["offset"] + entry["bytes"]]

    def close(self):
        self.raw.close()
        self.handle.close()


def shard_class_of(name):
    field = name.split(".")[-1]
    if name in k3_shard.MODEL_REPLICATED:
        return "model_replicated"
    if name in ("model.embed_tokens.weight", "lm_head.weight"):
        return "vocab_rows"
    for table, kind in ((k3_shard.REPLICATED, "replicated"),
                        ("kda_qkv_beta_weight", "fused_qkvb"),
                        (k3_shard.OUTPUT_HEADS, "output_heads"),
                        (k3_shard.INPUT_HEADS, "input_heads"),
                        (k3_shard.OUTPUT_DIM, "output_dim"),
                        (k3_shard.INPUT_DIM, "input_dim"),
                        (k3_shard.CONCAT_OUTPUT, "concat_output"),
                        (k3_shard.INPUT_DIM_PLAIN, "input_dim_plain"),
                        (k3_shard.EXPERT_CONCAT, "expert_w1"),
                        (k3_shard.EXPERT_INPUT, "expert_w2")):
        if field == table or field in table:
            return kind
    return None


def structural(view, failures, checked):
    cfg = view.config
    for key in ("hidden", "layers", "first_layer", "total_layers", "experts",
                "top_k", "latent", "intermediate", "vocab", "kda_heads",
                "kda_head", "heads", "kv_lora", "rope", "v_head", "nope",
                "shared", "q_lora"):
        if key not in cfg:
            failures.append(f"config echo lacks '{key}' (k3_shard needs it)")
    if view.base % ALIGN:
        failures.append(f"payload base {view.base} not {ALIGN}-aligned")
    ordered = sorted(view.manifest["tensors"],
                     key=lambda n: view.manifest["tensors"][n]["offset"])
    cursor = view.base
    for name in ordered:
        entry = view.manifest["tensors"][name]
        if entry["offset"] % ALIGN:
            failures.append(f"{name}: offset not {ALIGN}-aligned")
        start = view.base + entry["offset"]
        if start < cursor:
            failures.append(f"{name}: overlaps the previous tensor")
        cursor = start + entry["bytes"]
        if cursor > view.file_bytes:
            failures.append(f"{name}: payload runs past EOF")
        kind = entry.get("kind")
        shape = entry.get("shape")
        if kind == "mxfp4_ws_interleaved_v1":
            geom = entry["interleave"]
            priced = k3_pack.interleave_geometry(
                geom["out_dim"], geom["k_dim"], geom["experts"],
                tile_k=geom["tile_k"])
            if priced["tensor_bytes"] != entry["bytes"]:
                failures.append(
                    f"{name}: {entry['bytes']} bytes, geometry prices "
                    f"{priced['tensor_bytes']}")
        elif kind in ("bf16", "f32"):
            esize = 2 if kind == "bf16" else 4
            count = 1
            for dim in shape:
                count *= dim
            if count * esize != entry["bytes"]:
                failures.append(f"{name}: {entry['bytes']} bytes, shape "
                                f"{shape} x {esize} = {count * esize}")
        else:
            failures.append(f"{name}: unknown kind {kind!r}")
        if "sections" in entry:
            rows = 0
            for section in entry["sections"]:
                if section["row_offset"] != rows:
                    failures.append(
                        f"{name}: section {section['name']} does not tile")
                rows += section["rows"]
            if rows != shape[0]:
                failures.append(
                    f"{name}: sections cover {rows} rows, shape {shape[0]}")
        if shard_class_of(name) is None:
            failures.append(f"{name}: field outside every shard class")
        checked["tensors"] += 1
    return ordered


def expected_layer_names(cfg):
    """The tensor-name sets the packer emits (tools/k3_pack.py pack_model).
    A layer = common norms/folds, PLUS exactly one attention kind
    (kda_* / mla_*), PLUS exactly one MLP part (the routed MoE block, or the
    dense pair where the checkpoint has no block_sparse_moe gate)."""
    common = {"attn_norm_weight", "attnres_attn_weight", "mlp_norm_weight",
              "attnres_mlp_weight"}
    moe = {"router_weight", "router_bias", "expert_w1_weight",
           "expert_w2_weight", "shared_w1_weight", "shared_w2_weight",
           "routed_down_weight", "routed_up_weight", "routed_norm_weight"}
    dense = {"dense_gate_up_weight", "dense_down_weight"}
    kda = {"kda_qkv_beta_weight", "kda_q_conv_weight", "kda_k_conv_weight",
           "kda_v_conv_weight", "kda_decay_down_weight", "kda_decay_up_weight",
           "kda_decay_bias", "kda_head_log_scale", "kda_gate_weight",
           "kda_out_norm_weight", "kda_out_weight"}
    mla = {"mla_q_down_weight", "mla_q_norm_weight", "mla_kv_a_weight",
           "mla_kv_a_norm_weight", "mla_q_up_weight", "mla_kv_b_value_weight",
           "mla_gate_weight", "mla_out_weight"}
    return common, moe, dense, kda, mla


def layer_set_check(view, failures, checked):
    cfg = view.config
    first, count = cfg["first_layer"], cfg["layers"]
    common, moe, dense, kda, mla = expected_layer_names(cfg)
    by_layer = {}
    for name in view.manifest["tensors"]:
        parts = name.split(".")
        if len(parts) >= 4 and parts[0] == "model" and parts[1] == "layers":
            layer = int(parts[2])
            by_layer.setdefault(layer, set()).add(".".join(parts[3:]))
        elif name in ("model.norm.weight", "model.attnres_out_weight",
                      "model.embed_tokens.weight", "lm_head.weight"):
            pass
        else:
            failures.append(f"{name}: tensor outside layer/model namespaces")
    for layer, names in sorted(by_layer.items()):
        attn = (names & kda) | (names & mla)
        if len(names & kda) and len(names & mla):
            failures.append(f"layer {layer}: mixes kda and mla attention")
        want = common | attn | (moe if names & moe else dense)
        missing = want - names
        extra = names - want
        if missing:
            failures.append(f"layer {layer}: missing {sorted(missing)[:3]}"
                            f" ({len(missing)} total)")
        if extra:
            failures.append(f"layer {layer}: unexpected {sorted(extra)[:3]}"
                            f" ({len(extra)} total)")
        checked["layers"] += 1
    covered = set(by_layer)
    want_layers = set(range(first, first + count))
    if covered != want_layers:
        failures.append(f"layer coverage {min(covered)}..{max(covered)} != "
                        f"first_layer {first} + {count}")


def cross_verify(stage, rank, failures, checked, expert_cells, seed):
    degree = rank.config["tp_degree"]
    rank_idx = rank.config["tp_rank"]
    cfg = rank.config
    stage_cfg = stage.config
    for key in ("hidden", "vocab", "kda_heads", "kda_head", "heads",
                "kv_lora", "rope", "v_head"):
        if cfg.get(key) != stage_cfg.get(key):
            failures.append(f"config {key}: rank {cfg.get(key)}, stage "
                            f"{stage_cfg.get(key)}")
    if set(rank.manifest["tensors"]) != set(stage.manifest["tensors"]):
        failures.append("rank tensor-name set differs from stage")
    geo = {k: cfg[k] for k in ("kda_heads", "kda_head", "heads", "kv_lora",
                               "rope", "v_head")}

    def stage_block(name, lo_row, hi_row, row_bytes):
        entry = stage.manifest["tensors"][name]
        base = stage.base + entry["offset"]
        return stage.raw[base + lo_row * row_bytes:base + hi_row * row_bytes]

    rng = random.Random(seed)
    for name, entry in rank.manifest["tensors"].items():
        s_entry = stage.manifest["tensors"][name]
        cls = shard_class_of(name)
        checked["cross"] += 1
        if cls in ("replicated", "model_replicated"):
            if (entry["bytes"] != s_entry["bytes"]
                    or hashlib.sha256(rank.bytes_of(name)).digest()
                    != hashlib.sha256(
                        stage.raw[stage.base + s_entry["offset"]:
                                  stage.base + s_entry["offset"]
                                  + s_entry["bytes"]]).digest()):
                failures.append(f"{name}: replicated tensor differs from stage")
            continue
        if cls == "vocab_rows":
            rows = stage_cfg["vocab"]
            row_bytes = cfg["hidden"] * 2
            lo = (rows // degree) * rank_idx
            hi = lo + rows // degree
            if rank.bytes_of(name) != stage_block(name, lo, hi, row_bytes):
                failures.append(f"{name}: row block [{lo},{hi}) differs")
            continue
        if cls == "fused_qkvb":
            heads = geo["kda_heads"]
            h0 = (heads // degree) * rank_idx
            h1 = h0 + heads // degree
            row_bytes = cfg["hidden"] * 2
            expect = bytearray()
            for section in s_entry["sections"]:
                lo = (section["row_offset"] + h0 * section["rows_per_head"]) \
                    * row_bytes
                hi = (section["row_offset"] + h1 * section["rows_per_head"]) \
                    * row_bytes
                expect += stage.raw[stage.base + s_entry["offset"] + lo:
                                    stage.base + s_entry["offset"] + hi]
            if rank.bytes_of(name) != bytes(expect):
                failures.append(f"{name}: fused section split differs")
            continue
        if cls == "output_heads":
            field = name.split(".")[-1]
            kind = k3_shard.OUTPUT_HEADS[field][0]
            heads = k3_shard.head_count(kind, geo)
            block = k3_shard.head_block(kind, geo)
            rows = heads * block
            row_bytes = max(1, s_entry["bytes"] // rows)
            if rank.bytes_of(name) != stage_block(
                    name, (rows // degree) * rank_idx,
                    (rows // degree) * (rank_idx + 1), row_bytes):
                failures.append(f"{name}: head-block row split differs")
            continue
        if cls == "input_heads":
            field = name.split(".")[-1]
            kind = k3_shard.INPUT_HEADS[field][0]
            heads = k3_shard.head_count(kind, geo)
            block = k3_shard.head_block(kind, geo)
            in_bytes = heads * block * 2
            rows = s_entry["bytes"] // in_bytes
            got = k3_shard.slice_cols(stage.bytes_of(name), rows, in_bytes,
                                      (in_bytes // degree) * rank_idx,
                                      (in_bytes // degree) * (rank_idx + 1))
            if rank.bytes_of(name) != got:
                failures.append(f"{name}: input-head column split differs")
            continue
        if cls == "output_dim":
            row_bytes = cfg["hidden"] * 2
            rows = s_entry["bytes"] // row_bytes
            if rank.bytes_of(name) != stage_block(
                    name, (rows // degree) * rank_idx,
                    (rows // degree) * (rank_idx + 1), row_bytes):
                failures.append(f"{name}: output-dim row split differs")
            continue
        if cls == "input_dim":
            in_bytes = cfg["latent"] * 2
            rows = s_entry["bytes"] // in_bytes
            got = k3_shard.slice_cols(stage.bytes_of(name), rows, in_bytes,
                                      (in_bytes // degree) * rank_idx,
                                      (in_bytes // degree) * (rank_idx + 1))
            if rank.bytes_of(name) != got:
                failures.append(f"{name}: input-dim column split differs")
            continue
        if cls == "concat_output":
            half_rows = (s_entry["bytes"] // 2) // (cfg["hidden"] * 2)
            half = s_entry["bytes"] // 2
            base = stage.base + s_entry["offset"]
            lo = (half_rows // degree) * rank_idx
            span = (half_rows // degree) * cfg["hidden"] * 2
            expect = stage.raw[base + lo * cfg["hidden"] * 2:
                               base + lo * cfg["hidden"] * 2 + span] \
                + stage.raw[base + half + lo * cfg["hidden"] * 2:
                            base + half + lo * cfg["hidden"] * 2 + span]
            if rank.bytes_of(name) != bytes(expect):
                failures.append(f"{name}: gate|up half split differs")
            continue
        if cls == "input_dim_plain":
            rows = cfg["hidden"]
            in_bytes = s_entry["bytes"] // rows
            got = k3_shard.slice_cols(stage.bytes_of(name), rows, in_bytes,
                                      (in_bytes // degree) * rank_idx,
                                      (in_bytes // degree) * (rank_idx + 1))
            if rank.bytes_of(name) != got:
                failures.append(f"{name}: plain column split differs")
            continue
        if cls in ("expert_w1", "expert_w2"):
            geom = s_entry["interleave"]
            experts, cells = geom["experts"], geom["cells"]
            k_tiles, cell_rows = geom["k_tiles"], geom["cell_rows"]
            row_bytes = geom["row_bytes"]
            rpe = geom["rows_per_expert"]  # ROWS per expert, incl. cell_rows
            stage_base = stage.base + s_entry["offset"]
            rank_raw = rank.bytes_of(name)
            rank_rpe = (rank.manifest["tensors"][name]
                        ["interleave"]["rows_per_expert"])
            if cls == "expert_w1":
                take = k_tiles // degree
                t0 = rank_idx * take
                chunk_rows = cells * cell_rows
            else:
                take_out = cells // degree
                c0 = rank_idx * take_out
                chunk_rows = take_out * cell_rows
            picks = min(expert_cells, experts)
            for e in rng.sample(range(experts), picks):
                if cls == "expert_w1":
                    t = rng.randrange(take)
                    s_row = e * rpe + (t0 + t) * chunk_rows
                    r_row = e * rank_rpe + t * chunk_rows
                else:
                    t = rng.randrange(k_tiles)
                    s_row = e * rpe + (t * cells + c0) * cell_rows
                    r_row = e * rank_rpe + t * take_out * cell_rows
                s_chunk = stage.raw[stage_base + s_row * row_bytes:
                                    stage_base + (s_row + chunk_rows)
                                    * row_bytes]
                r_chunk = rank_raw[r_row * row_bytes:
                                   (r_row + chunk_rows) * row_bytes]
                if not r_chunk or len(s_chunk) != len(r_chunk) \
                        or s_chunk != r_chunk:
                    failures.append(
                        f"{name}: expert {e} sample differs from stage")
                    break
            # the E8M0 scale lanes of the sampled rank cells: no 0xff
            continue
        failures.append(f"{name}: unhandled class {cls} in cross pass")
    # scale-lane sanity on the rank's expert tensors: no E8M0 0xff in the
    # sampled cells' 17th rows
    for name in k3_shard.EXPERT_CONCAT | k3_shard.EXPERT_INPUT:
        if name not in rank.manifest["tensors"]:
            continue
        entry = rank.manifest["tensors"][name]
        geom = entry["interleave"]
        cell_rows, row_bytes = geom["cell_rows"], geom["row_bytes"]
        rpe = geom["rows_per_expert"]
        rank_base = rank.base + entry["offset"]
        for _ in range(min(32, geom["experts"])):
            e = rng.randrange(geom["experts"])
            cell = rng.randrange(geom["cells"] * geom["k_tiles"])
            row = e * rpe + cell
            scale_row = rank.raw[rank_base + (row * cell_rows
                                              + cell_rows - 1) * row_bytes:
                                 rank_base + (row * cell_rows + cell_rows)
                                 * row_bytes]
            if 0xFF in tuple(scale_row):
                failures.append(f"{name}: E8M0 0xff in a sampled scale row")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pack", type=Path)
    parser.add_argument("--stage-pack", type=Path)
    parser.add_argument("--expect-tp-degree", type=int)
    parser.add_argument("--expect-rank", type=int)
    parser.add_argument("--expect-first", type=int)
    parser.add_argument("--expect-layers", type=int)
    parser.add_argument("--expert-cells", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quick", action="store_true",
                        help="structural pass only, no stage cross-compare")
    args = parser.parse_args()

    failures = []
    checked = {"tensors": 0, "layers": 0, "cross": 0}
    try:
        view = PackView(args.pack)
    except VerifyFailure as failure:
        print(f"K3 PACK VERIFY FAIL: {failure}")
        return 1
    cfg = view.config
    for what, got, want in (
            ("tp_degree", args.expect_tp_degree, cfg.get("tp_degree")),
            ("tp_rank", args.expect_rank, cfg.get("tp_rank")),
            ("first_layer", args.expect_first, cfg.get("first_layer")),
            ("layers", args.expect_layers, cfg.get("layers"))):
        if want is not None and got is not None and got != want:
            failures.append(f"config {what}: expected {got}, pack says {want}")
    structural(view, failures, checked)
    layer_set_check(view, failures, checked)

    if args.stage_pack and not args.quick:
        stage = PackView(args.stage_pack)
        cross_verify(stage, view, failures, checked,
                     args.expert_cells, args.seed)
        stage.close()

    view.close()
    if failures:
        for failure in failures[:10]:
            print(f"  {failure}")
        print(f"K3 PACK VERIFY FAIL: {args.pack} - {len(failures)} failures, "
              f"{checked['tensors']} tensors, {checked['layers']} layers")
        return 1
    print(f"K3 PACK VERIFY PASS: {args.pack} - {checked['tensors']} tensors, "
          f"{checked['layers']} layers, {checked['cross']} cross-checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
