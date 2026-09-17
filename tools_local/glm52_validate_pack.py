#!/usr/bin/env python3
"""Validate a GLM52 stage pack against the C module policy (1:1 mirror).

Mirrors modules/glm52_resident_decode_stage/source/spark_glm52_stagepack_format.h
SparkGlm52StagePackExpectedShape + ExpectedPayloadBytes/ExpectedScaleBytes and
the geometry checks in SparkGlm52PackValidateEntryGeometry, plus the inventory
masks (SparkGlm52ExpectedLayerMask / ExpectedGlobalMask).

The expert codec is taken from the pack header (bf16=1, fp8=5 or nvfp4=6)
— the module accepts any compiled expert codec; entry shapes are codec-
independent, scale planes are not (bf16: NONE — native-precision experts;
fp8: F32 per 128-block; nvfp4: UE4M3 per 16-block plus one F32 global per
expert). The module accepts codec 1 (the glm53full bf16 arm landed with
the per-source firmware pins); byte-exactness of codec-1 packs against
the promoted source is proven by tools/glm53full_bf16_tp16_source_verify.py.
"""
import struct
import sys

GLOBAL_LAYER = 0xFFFFFFFF
BF16, FP8, NVFP4, NONE = 1, 5, 6, 0
PAYLOAD_BF16, PAYLOAD_F32, PAYLOAD_U32, PAYLOAD_PACKED = 1, 2, 3, 4
SCALE_NONE, SCALE_F32, SCALE_UE4M3_F32_GLOBAL = 0, 1, 4

HIDDEN = 6144
VOCAB = 154880
HEADS = 64
LATENT = 512
ROPE = 64
QUERY_A = 2048
QK_NOPE = 192
VALUE_HEAD = 256
MOE_EXPERTS = 256
MOE_INT = 2048
DENSE_INT = 12288
FIRST_ROUTED = 3
DSA_HEAD_COUNT = 32
DSA_HEAD_DIM = 128
SHARE = 4

K_EMBEDDING, K_FINAL_NORM, K_LM_HEAD = 0, 1, 2
K_ATTN_NORM, K_Q_A, K_Q_A_NORM, K_Q_B, K_KV_A, K_KV_A_NORM = 3, 4, 5, 6, 7, 8
K_KV_B_KEY_T, K_KV_B_VALUE, K_ATTN_OUTPUT, K_POST_ATTN_NORM = 9, 10, 11, 12
K_INDEX_Q, K_INDEX_K, K_INDEX_HEAD, K_INDEX_NORM_W, K_INDEX_NORM_B = 13, 14, 15, 16, 17
K_DENSE_GATE_UP, K_DENSE_DOWN, K_ROUTER, K_ROUTER_CORR = 18, 19, 20, 21
K_EXPERT_UP_GATE, K_EXPERT_DOWN, K_SHARED_GATE_UP, K_SHARED_DOWN = 22, 23, 24, 25
KIND_COUNT = 26

KIND_NAME = {
    0: "EMBEDDING", 1: "FINAL_NORM", 2: "LM_HEAD", 3: "ATTN_NORM", 4: "Q_A",
    5: "Q_A_NORM", 6: "Q_B", 7: "KV_A", 8: "KV_A_NORM", 9: "KV_B_KEY_T",
    10: "KV_B_VALUE", 11: "ATTN_OUTPUT", 12: "POST_ATTN_NORM", 13: "INDEX_Q",
    14: "INDEX_K", 15: "INDEX_HEAD", 16: "INDEX_NORM_W", 17: "INDEX_NORM_B",
    18: "DENSE_GATE_UP", 19: "DENSE_DOWN", 20: "ROUTER", 21: "ROUTER_CORRECTION",
    22: "EXPERT_UP_GATE", 23: "EXPERT_DOWN", 24: "SHARED_GATE_UP", 25: "SHARED_DOWN",
}

TP_SHARDS_ROWS = {K_EMBEDDING, K_LM_HEAD, K_Q_B, K_DENSE_GATE_UP,
                  K_EXPERT_UP_GATE, K_SHARED_GATE_UP}
TP_SHARDS_COLS = {K_ATTN_OUTPUT, K_DENSE_DOWN, K_EXPERT_DOWN, K_SHARED_DOWN}


def kind_is_global(k):
    return k <= K_LM_HEAD


def kind_is_indexer(k):
    return K_INDEX_Q <= k <= K_INDEX_NORM_B


def kind_is_dense(k):
    return k in (K_DENSE_GATE_UP, K_DENSE_DOWN)


def kind_is_routed(k):
    return K_ROUTER <= k <= K_SHARED_DOWN


def layer_is_dense(l):
    return l < FIRST_ROUTED


def layer_has_full_indexer(l):
    return l < 3 or (l >= 6 and (l - 6) % SHARE == 0)


def shape_bf16(rows, cols, groups=1):
    return (PAYLOAD_BF16, BF16, SCALE_NONE, groups, rows, cols)


def expected_shape(kind, layer, tp, expert_codec=FP8):
    if kind >= KIND_COUNT or tp == 0:
        return None, "kind range"
    g = kind_is_global(kind)
    if (g and layer != GLOBAL_LAYER) or (not g and layer >= 78):
        return None, "global/layer placement"
    if kind_is_indexer(kind) and not layer_has_full_indexer(layer):
        return None, "indexer on shared layer"
    if kind_is_dense(kind) != layer_is_dense(layer) and (kind_is_dense(kind) or kind_is_routed(kind)):
        return None, "dense/routed layer kind mismatch"
    qk_head = QK_NOPE + ROPE
    expert_entry = ((PAYLOAD_BF16, BF16, SCALE_NONE) if expert_codec == BF16
                    else (PAYLOAD_PACKED, expert_codec,
                          SCALE_F32 if expert_codec == FP8
                          else SCALE_UE4M3_F32_GLOBAL))
    table = {
        K_EMBEDDING: shape_bf16(VOCAB, HIDDEN),
        K_LM_HEAD: shape_bf16(VOCAB, HIDDEN),
        K_FINAL_NORM: shape_bf16(1, HIDDEN),
        K_ATTN_NORM: shape_bf16(1, HIDDEN),
        K_POST_ATTN_NORM: shape_bf16(1, HIDDEN),
        K_Q_A: shape_bf16(QUERY_A, HIDDEN),
        K_Q_A_NORM: shape_bf16(1, QUERY_A),
        K_Q_B: shape_bf16(HEADS * qk_head, QUERY_A),
        K_KV_A: shape_bf16(LATENT + ROPE, HIDDEN),
        K_KV_A_NORM: shape_bf16(1, LATENT),
        K_KV_B_KEY_T: shape_bf16(LATENT, QK_NOPE, groups=HEADS),
        K_KV_B_VALUE: shape_bf16(VALUE_HEAD, LATENT, groups=HEADS),
        K_ATTN_OUTPUT: shape_bf16(HIDDEN, HEADS * VALUE_HEAD),
        K_INDEX_Q: shape_bf16(DSA_HEAD_COUNT * DSA_HEAD_DIM, QUERY_A),
        K_INDEX_K: shape_bf16(DSA_HEAD_DIM, HIDDEN),
        K_INDEX_HEAD: shape_bf16(DSA_HEAD_COUNT, HIDDEN),
        K_INDEX_NORM_W: shape_bf16(1, DSA_HEAD_DIM),
        K_INDEX_NORM_B: shape_bf16(1, DSA_HEAD_DIM),
        K_DENSE_GATE_UP: shape_bf16(2 * DENSE_INT, HIDDEN),
        K_DENSE_DOWN: shape_bf16(HIDDEN, DENSE_INT),
        K_ROUTER: shape_bf16(MOE_EXPERTS, HIDDEN),
        K_ROUTER_CORR: (PAYLOAD_F32, NONE, SCALE_NONE, 1, 1, MOE_EXPERTS),
        K_EXPERT_UP_GATE: (*expert_entry, MOE_EXPERTS, 2 * MOE_INT, HIDDEN),
        K_EXPERT_DOWN: (*expert_entry, MOE_EXPERTS, HIDDEN, MOE_INT),
        K_SHARED_GATE_UP: shape_bf16(2 * MOE_INT, HIDDEN),
        K_SHARED_DOWN: shape_bf16(HIDDEN, MOE_INT),
    }
    s = list(table[kind])
    if kind in TP_SHARDS_ROWS:
        if s[4] == 0 or s[4] % tp != 0:
            return None, "rows not divisible by tp"
        s[4] //= tp
    if kind in TP_SHARDS_COLS:
        if s[5] == 0 or s[5] % tp != 0:
            return None, "cols not divisible by tp"
        s[5] //= tp
    return tuple(s), None


def expected_payload_bytes(s):
    pt, wc, se, g, rows, cols = s
    if pt == PAYLOAD_BF16:
        return g * rows * cols * 2
    if pt in (PAYLOAD_F32, PAYLOAD_U32):
        return g * rows * cols * 4
    if pt == PAYLOAD_PACKED:
        bits = {FP8: 8, NVFP4: 4}.get(wc)
        if bits is None:
            return None
        return (bits * g * rows * cols + 7) // 8
    return None


def expected_scale_bytes(s):
    pt, wc, se, g, rows, cols = s
    if pt != PAYLOAD_PACKED:
        return 0
    if wc == NVFP4:
        # UE4M3 per 16-column block + one F32 global per expert group
        # (SparkWeightCodecScaleBytes, SPARK_WEIGHT_CODEC_NVFP4_E2M1).
        return g * rows * ((cols + 15) // 16) + g * 4
    blocks = (cols + 127) // 128
    return g * rows * blocks * 4


def main():
    path = sys.argv[1]
    tp = int(sys.argv[2])
    with open(path, "rb") as f:
        h = f.read(264)
        vals = struct.unpack_from("<20I", h, 0)
        magic, ver, hb, eb, abi, flags, count = vals[0:7]
        lin, expc, kv = vals[15], vals[16], vals[17]
        tp_degree, tp_rank = vals[18], vals[19]
        dir_off, file_bytes = struct.unpack_from("<2Q", h, 80)
    assert magic == 0x32534C47 and ver == 3
    print("header: tensors=%d linear_codec=%d expert_codec=%d kv_codec=%d tp=%d rank=%d" % (count, lin, expc, kv, tp_degree, tp_rank))
    if expc not in (BF16, FP8, NVFP4):
        print("UNSUPPORTED expert codec %d (expected bf16=1, fp8=5 or nvfp4=6)" % expc)
        return 1
    if tp_degree != tp:
        print("WARNING: header tp_degree %d != requested %d" % (tp_degree, tp))
    errors = 0
    with open(path, "rb") as f:
        f.seek(dir_off)
        raw = f.read(count * 64)
    seen_layer = {}
    seen_global = 0
    dir_end = dir_off + count * 64
    for i in range(count):
        e = struct.unpack_from("<8I4Q", raw, i * 64)
        kind, layer, pt, wc, se, g, rows, cols = e[0:8]
        poff, pbytes, soff, sbytes = e[8:12]
        es, err = expected_shape(kind, layer, tp_degree, expc)
        if es is None:
            print("entry %d kind %s layer %s: EXPECTED SHAPE ERROR: %s" % (i, KIND_NAME.get(kind, kind), layer, err))
            errors += 1
            continue
        e_pt, e_wc, e_se, e_g, e_rows, e_cols = es
        if (pt, wc, se, g, rows, cols) != es:
            print("entry %d kind %s layer %s: fields (%d,%d,%d,%d,%d,%d) != expected (%d,%d,%d,%d,%d,%d)" % (
                i, KIND_NAME.get(kind, kind), layer, pt, wc, se, g, rows, cols, e_pt, e_wc, e_se, e_g, e_rows, e_cols))
            errors += 1
        exp_pb = expected_payload_bytes(es)
        exp_sb = expected_scale_bytes(es)
        if exp_pb is not None and pbytes != exp_pb:
            print("entry %d: payload_bytes %d != expected %d" % (i, pbytes, exp_pb))
            errors += 1
        if sbytes != exp_sb:
            print("entry %d: scale_bytes %d != expected %d" % (i, sbytes, exp_sb))
            errors += 1
        if poff % 256 != 0 or poff < dir_end or poff > file_bytes or pbytes > file_bytes - poff:
            print("entry %d: payload range bad (off=%d bytes=%d)" % (i, poff, pbytes))
            errors += 1
        if sbytes == 0:
            if soff != 0:
                print("entry %d: scale_offset %d with zero scale_bytes" % (i, soff))
                errors += 1
        elif soff % 256 != 0 or soff < dir_end or soff > file_bytes or sbytes > file_bytes - soff:
            print("entry %d: scale range bad (off=%d bytes=%d)" % (i, soff, sbytes))
            errors += 1
        if layer == GLOBAL_LAYER:
            seen_global |= 1 << kind
        else:
            seen_layer.setdefault(layer, 0)
            seen_layer[layer] |= 1 << kind
    exp_global = (1 << K_EMBEDDING) | (1 << K_FINAL_NORM) | (1 << K_LM_HEAD)
    if seen_global != exp_global:
        print("global inventory mismatch: seen %x expected %x" % (seen_global, exp_global))
        errors += 1
    for layer in range(78):
        exp = 0
        for kind in range(3, KIND_COUNT):
            es, _err = expected_shape(kind, layer, tp_degree, expc)
            if es is not None:
                exp |= 1 << kind
        got = seen_layer.get(layer, 0)
        if got != exp:
            missing = [KIND_NAME.get(k, k) for k in range(26) if (exp & (1 << k)) and not (got & (1 << k))]
            extra = [KIND_NAME.get(k, k) for k in range(26) if (got & (1 << k)) and not (exp & (1 << k))]
            print("layer %d inventory: got %x expected %x missing=%s extra=%s" % (layer, got, exp, missing, extra))
            errors += 1
    print("errors: %d" % errors)
    return 0


if __name__ == "__main__":
    sys.exit(main())
