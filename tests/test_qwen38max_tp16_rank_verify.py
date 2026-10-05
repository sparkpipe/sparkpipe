"""The qwen38_max rank verifier accepts only the current-packer form.

The current-packer form carries the codec-8 nvfp4 experts with per-expert
f32 tails and a plan-shaped directory. The placed tp4pp4 form stamps codec 4
and repeats the last inventory ref's packed shape in every directory entry;
it is refused. A named receipt must exist and carry every recomputed field.
"""

import hashlib
import json
import struct
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import qwen38max_tp16_rank_verify as V

HEADER_BYTES = 128
ENTRY_BYTES = 56

MINI_TABLES_SOURCE = '''
MAGIC = 0x51384154
FORMAT2_VERSION = 2
HEADER2_BYTES = 128
ENTRY_BYTES = 56
PAYLOAD_ALIGNMENT = 256
HIDDEN = 8
LAYER_COUNT = 2
ATTENTION_PERIOD = 4
FULL_PHASE = 3
GDN_KEY_HEADS = 1
GDN_VALUE_HEADS = 2
GDN_HEAD_KEY_DIM = 2
GDN_HEAD_VALUE_DIM = 2
GDN_CONV_KERNEL = 4
ATTN_QUERY_HEADS = 2
ATTN_KV_HEADS = 1
ATTN_HEAD_DIM = 4
ATTN_ROPE_DIM = 2
EXPERT_COUNT = 4
EXPERTS_PER_TOKEN = 2
EXPERT_INTERMEDIATE = 4
VOCAB = 16
MXFP4_GROUP = 32
MTP_LAYERS = 1
STRIP_MTP = False
EXPERT_CODEC = "nvfp4"
WEIGHT_BF16 = 0
WEIGHT_F32 = 1
WEIGHT_FP8_F32B128 = 4
WEIGHT_NVFP4_PACKED = 8
KIND_NORM, KIND_FINAL_NORM, KIND_LM_HEAD, KIND_MOE_GATE, KIND_MOE_W1, KIND_MOE_W3, KIND_MOE_DOWN = range(7)
GLOBAL_LAYER = 0xFFFFFFFF
MTP_LAYER = 0xFFFFFFFE

class PackFailure(Exception):
    pass

class Ref:
    def __init__(self, kind, layer, name, rows, columns, weight_format):
        self.kind = kind
        self.layer = layer
        self.name = name
        self.rows = rows
        self.columns = columns
        self.weight_format = weight_format

class SafetensorsSource:
    def __init__(self, *a, **k):
        raise RuntimeError("content pass not under test")

def ref_weight_format(ref):
    if ref.kind in (KIND_MOE_W1, KIND_MOE_W3, KIND_MOE_DOWN):
        return WEIGHT_BF16 if ref.layer == MTP_LAYER else WEIGHT_NVFP4_PACKED
    return ref.weight_format

def build_inventory(first_layer, layer_count):
    assert (first_layer, layer_count) == (0, 2), "mini inventory covers the full stack"
    return [
        Ref(KIND_LM_HEAD, GLOBAL_LAYER, "head", VOCAB, HIDDEN, WEIGHT_BF16),
        Ref(KIND_MOE_GATE, 0, "gate", EXPERT_COUNT, HIDDEN, WEIGHT_BF16),
        Ref(KIND_MOE_W1, 0, "w1", EXPERT_COUNT * EXPERT_INTERMEDIATE, HIDDEN,
            WEIGHT_FP8_F32B128),
        Ref(KIND_NORM, 0, "norm", 1, GDN_HEAD_VALUE_DIM, WEIGHT_BF16),
        Ref(KIND_MOE_GATE, 1, "gate", EXPERT_COUNT, HIDDEN, WEIGHT_BF16),
        Ref(KIND_MOE_W1, 1, "w1", EXPERT_COUNT * EXPERT_INTERMEDIATE, HIDDEN,
            WEIGHT_FP8_F32B128),
        Ref(KIND_NORM, 1, "norm", 1, GDN_HEAD_VALUE_DIM, WEIGHT_BF16),
    ]

def build_tp_plan(ref, degree, rank):
    if degree <= 1:
        return None
    if ref.layer == GLOBAL_LAYER:
        if ref.kind == KIND_LM_HEAD:
            return ("rows", rank * (ref.rows // degree), ref.rows // degree)
        return None
    if ref.kind in (KIND_MOE_W1, KIND_MOE_W3, KIND_MOE_DOWN):
        per = ref.rows // degree
        return ("experts", rank * per, per)
    return None

def packed_tp_shape(ref, plan):
    if plan is None:
        return (ref.rows, ref.columns)
    return (plan[2], ref.columns)

def copy_tp_plan(source, ref, plan, out):
    raise PackFailure("content pass not under test")
'''


def mini_tables():
    import types
    module = types.ModuleType("mini_qwen38_max_tables")
    exec(MINI_TABLES_SOURCE, module.__dict__)
    return module


class MiniRef:
    pass


def build_pack(tables, tp_degree: int, tp_rank: int, expert_fmt: int,
               stale_directory: bool, mtp_count: int = 0,
               corrupt: bool = False) -> bytes:
    tables.STRIP_MTP = mtp_count == 0
    refs = tables.build_inventory(0, tables.LAYER_COUNT)
    plans = []
    for ref in refs:
        plan = tables.build_tp_plan(ref, tp_degree, tp_rank)
        rows, cols = tables.packed_tp_shape(ref, plan)
        resident = 0
        if ref.kind in (tables.KIND_MOE_W1, tables.KIND_MOE_W3, tables.KIND_MOE_DOWN):
            resident = rows // (ref.rows // tables.EXPERT_COUNT)
        fmt = tables.ref_weight_format(ref)
        if ref.kind in (tables.KIND_MOE_W1, tables.KIND_MOE_W3, tables.KIND_MOE_DOWN):
            fmt = expert_fmt
        if fmt == tables.WEIGHT_NVFP4_PACKED:
            pbytes = rows * (cols // 2)
            sbytes = rows * (cols // 16) + resident * 8
            group = 16
        elif fmt == tables.WEIGHT_FP8_F32B128:
            pbytes = rows * (cols // 2)
            sbytes = rows * (cols // 16)
            group = 128
        else:
            pbytes = rows * cols * 2
            sbytes = 0
            group = 0
        plans.append((ref, fmt, rows, cols, group, pbytes, sbytes))
    entries = []
    cursor = 0
    align = 256
    for ref, fmt, rows, cols, group, pbytes, sbytes in plans:
        cursor = (cursor + align - 1) & ~(align - 1)
        poff = cursor
        cursor += pbytes
        soff = 0
        if sbytes:
            soff = (cursor + align - 1) & ~(align - 1)
            cursor = soff + sbytes
        stale_rows, stale_cols = plans[-1][2], plans[-1][3]
        wire_rows, wire_cols = (stale_rows, stale_cols) if stale_directory \
            else (rows, cols)
        entries.append(struct.pack("<6I4Q", ref.kind, ref.layer, fmt,
                                   wire_rows, wire_cols, group, poff, pbytes,
                                   soff, sbytes))
    if corrupt and entries:
        fields = list(struct.unpack_from("<6I4Q", entries[0], 0))
        fields[7] += 64
        entries[0] = struct.pack("<6I4Q", *fields)
    directory = b"".join(entries)
    file_bytes = HEADER_BYTES + len(directory) + cursor
    header = struct.pack(
        "<28I2Q", 0x51384154, 2, HEADER_BYTES, ENTRY_BYTES,
        len(entries), 8, 2, 0, 2, 4, 3, 1, 2, 2, 2, 4, 2, 1, 4, 2,
        4, 2, 4, 16, 32, mtp_count, tp_degree, tp_rank,
        HEADER_BYTES, file_bytes)
    return header + directory + b"\0" * cursor


def run_verify(pack: Path, tp_degree: int, tp_rank: int, tables, receipt=None):
    return V.verify(pack, tp_degree, tp_rank, None, receipt, False, tables=tables)


def full_receipt(pack: Path, tp_degree: int, tp_rank: int) -> dict:
    body = pack.read_bytes()
    header = struct.unpack_from("<28I2Q", body, 0)
    return {
        "tensor_count": header[4],
        "bytes": len(body),
        "first_layer_index": header[7],
        "layer_count": header[6],
        "tp_degree": tp_degree,
        "tp_rank": tp_rank,
        "output_sha256": hashlib.sha256(body).hexdigest(),
        "source_index_sha256": "0" * 64,
    }


def test_current_packer_form_passes():
    with tempfile.TemporaryDirectory() as tmp:
        tables = mini_tables()
        pack = Path(tmp) / "current.spstage"
        pack.write_bytes(build_pack(tables, 4, 3, tables.WEIGHT_NVFP4_PACKED, False))
        ok, verdict = run_verify(pack, 4, 3, tables)
        assert ok, verdict["errors"]


def test_placed_legacy_form_refused():
    with tempfile.TemporaryDirectory() as tmp:
        tables = mini_tables()
        pack = Path(tmp) / "placed.spstage"
        pack.write_bytes(build_pack(tables, 4, 3, tables.WEIGHT_FP8_F32B128, True,
                                    mtp_count=1))
        ok, verdict = run_verify(pack, 4, 3, tables)
        assert not ok
        assert verdict["verdict"] == "FAIL"
        assert any("repack with the current packer" in e for e in verdict["errors"])


def test_stale_shape_directory_refused():
    with tempfile.TemporaryDirectory() as tmp:
        tables = mini_tables()
        pack = Path(tmp) / "stale.spstage"
        pack.write_bytes(build_pack(tables, 4, 3, tables.WEIGHT_NVFP4_PACKED, True))
        ok, verdict = run_verify(pack, 4, 3, tables)
        assert not ok
        assert any("expected rank shard" in e for e in verdict["errors"])


def test_one_off_plan_shape_refused():
    with tempfile.TemporaryDirectory() as tmp:
        tables = mini_tables()
        body = bytearray(build_pack(tables, 4, 3, tables.WEIGHT_NVFP4_PACKED, False))
        struct.pack_into("<II", body, HEADER_BYTES + ENTRY_BYTES + 12, 7, 7)
        pack = Path(tmp) / "bad.spstage"
        pack.write_bytes(bytes(body))
        ok, verdict = run_verify(pack, 4, 3, tables)
        assert not ok
        assert any("shape 7x7, expected rank shard" in e for e in verdict["errors"])


def test_receipt_must_exist():
    with tempfile.TemporaryDirectory() as tmp:
        tables = mini_tables()
        pack = Path(tmp) / "current.spstage"
        pack.write_bytes(build_pack(tables, 4, 3, tables.WEIGHT_NVFP4_PACKED, False))
        ok, verdict = run_verify(pack, 4, 3, tables, Path(tmp) / "absent.json")
        assert not ok
        assert any("does not exist" in e for e in verdict["errors"])


def test_full_receipt_passes_and_every_field_is_required():
    with tempfile.TemporaryDirectory() as tmp:
        tables = mini_tables()
        pack = Path(tmp) / "current.spstage"
        pack.write_bytes(build_pack(tables, 4, 3, tables.WEIGHT_NVFP4_PACKED, False))
        receipt_path = Path(tmp) / "receipt.json"
        receipt = full_receipt(pack, 4, 3)
        receipt_path.write_text(json.dumps(receipt))
        ok, verdict = run_verify(pack, 4, 3, tables, receipt_path)
        assert ok, verdict["errors"]
        for name in receipt:
            partial = dict(receipt)
            del partial[name]
            receipt_path.write_text(json.dumps(partial))
            ok, verdict = run_verify(pack, 4, 3, tables, receipt_path)
            assert not ok, name
            assert any(f"receipt is missing {name}" in e for e in verdict["errors"]), name
        for name in ("tensor_count", "bytes", "tp_rank", "output_sha256"):
            wrong = dict(receipt)
            wrong[name] = "f" * 64 if name == "output_sha256" else receipt[name] + 1
            receipt_path.write_text(json.dumps(wrong))
            ok, verdict = run_verify(pack, 4, 3, tables, receipt_path)
            assert not ok, name
            assert any(f"receipt {name}=" in e for e in verdict["errors"]), name


def test_corrupt_byte_math_fails():
    with tempfile.TemporaryDirectory() as tmp:
        tables = mini_tables()
        pack = Path(tmp) / "corrupt.spstage"
        pack.write_bytes(build_pack(tables, 4, 3, tables.WEIGHT_NVFP4_PACKED, False,
                                    corrupt=True))
        ok, verdict = run_verify(pack, 4, 3, tables)
        assert not ok
        assert any("payload_bytes" in e for e in verdict["errors"])


def test_unexpected_mtp_count_fails():
    with tempfile.TemporaryDirectory() as tmp:
        tables = mini_tables()
        pack = Path(tmp) / "mtp.spstage"
        pack.write_bytes(build_pack(tables, 4, 3, tables.WEIGHT_NVFP4_PACKED, False,
                                    mtp_count=0))
        ok, verdict = run_verify(pack, 4, 3, tables)
        assert ok
        struct.pack_into("<I", body := bytearray(pack.read_bytes()), 100, 2)
        pack.write_bytes(bytes(body))
        ok, verdict = run_verify(pack, 4, 3, tables)
        assert not ok
        assert any("mtp_layer_count" in e for e in verdict["errors"])


def test_header_rank_mismatch_fails():
    with tempfile.TemporaryDirectory() as tmp:
        tables = mini_tables()
        pack = Path(tmp) / "rank.spstage"
        pack.write_bytes(build_pack(tables, 4, 3, tables.WEIGHT_NVFP4_PACKED, False))
        ok, verdict = run_verify(pack, 4, 1, tables)
        assert not ok
        assert any("tp_rank" in e for e in verdict["errors"])


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
