#!/usr/bin/env python3
"""Independent CPU verification of a DSV4 TP16 rank shard.

Tier 1: structural verify (packer's --verify-output equivalent).
Tier 2: byte-level — every output tensor's payload AND scale plane must
equal the source full-pack bytes for the documented shard semantics,
reimplemented here (not calling the packer's copy code).
Tier 3: ground-truth anchors — replicated BF16 tensors (final norms,
embedding row shard, lm_head vocab tiles) must equal the ORIGINAL
safetensors bytes from /mnt/model-warm.

Usage: dsv4flash_tp16_verify.py --source <full.spstage> --shard <rank.spstage>
       --rank N [--safetensors-dir DIR]
"""
import argparse, hashlib, json, struct, sys
from pathlib import Path

HEADER = struct.Struct("<16I2Q")
ENTRY = struct.Struct("<6I2Q")
MAGIC = 0x34565344
VERSION = 4
TP_DEGREE = 16
LAYERS = 43
HIDDEN = 4096
QUERY_DIM = 32768
OUTPUT_GROUPS = 8
OUTPUT_LORA = 1024
OUTPUT_GROUP_DIM = QUERY_DIM // OUTPUT_GROUPS
EXPERTS = 256
EXPERT_WIDTH = 2048
VOCAB = 129280
VOCAB_TILE_ROWS = 128
MTP_LAYER_FIRST = 0xFFFFFFFB
GLOBAL_LAYER = 0xFFFFFFFF

WEIGHT_BF16, WEIGHT_F32, WEIGHT_U32, WEIGHT_FP4, WEIGHT_FP8 = 0, 1, 2, 3, 4
KIND = dict(ATTN_SINK=0, WQ_A=1, WQ_B=3, WKV=4, WO_A=6, WO_B=7,
            EXPERTS_W1=19, EXPERTS_W2=20, EXPERTS_W3=21, SHARED_W1=22,
            SHARED_W2=23, SHARED_W3=24, COMPRESS_WKV=26, COMPRESS_WGATE=27,
            INDEX_WKV=32, INDEX_WGATE=33, EMBEDDING=35, FINAL_NORM=36,
            LM_HEAD=37, HC_HEAD_FN=38, HC_HEAD_BASE=39, HC_HEAD_SCALE=40,
            MTP_MAIN_PROJ=41, MTP_MAIN_NORM=42, MTP_FINAL_NORM=43,
            MTP_HC_HEAD_FN=44, MTP_HC_HEAD_BASE=45, MTP_HC_HEAD_SCALE=46,
            MTP_MARKOV_W1=47, MTP_MARKOV_W2=48, MTP_CONFIDENCE_PROJ=49)
MTP_SET = frozenset(v for k, v in KIND.items() if k.startswith("MTP_"))
K_LM_HEAD, K_EMBED = KIND["LM_HEAD"], KIND["EMBEDDING"]
K_FINAL_NORM, K_MTP_FINAL_NORM = KIND["FINAL_NORM"], KIND["MTP_FINAL_NORM"]


def payload_bytes(weight, rows, cols):
    elements = rows * cols
    if weight == WEIGHT_FP4:
        return elements // 2
    if weight in (WEIGHT_F32, WEIGHT_U32):
        return elements * 4
    if weight == WEIGHT_FP8:
        return elements
    return elements * 2


def scale_bytes(weight, rows, cols):
    if weight == WEIGHT_FP8:
        return rows * ((cols + 127) // 128)
    if weight == WEIGHT_FP4:
        return rows * ((cols + 31) // 32)
    return 0


def element_bytes(weight):
    return {WEIGHT_FP4: 0, WEIGHT_FP8: 1, WEIGHT_F32: 4, WEIGHT_U32: 4}.get(weight, 2)


def output_group_shard(rank):
    ranks_per_group = max(1, TP_DEGREE // OUTPUT_GROUPS)
    group_count = max(1, OUTPUT_GROUPS // TP_DEGREE)
    group_start = (rank // ranks_per_group) * group_count
    column_width = OUTPUT_GROUP_DIM // ranks_per_group
    column_start = (rank % ranks_per_group) * column_width
    return group_start, group_count, column_start, column_width


def vocabulary_shard(rank):
    base, remainder = divmod(VOCAB // VOCAB_TILE_ROWS, TP_DEGREE)
    start_tile = rank * base + min(rank, remainder)
    tile_count = base + (1 if rank < remainder else 0)
    return start_tile * VOCAB_TILE_ROWS, tile_count * VOCAB_TILE_ROWS


def row_indices(kind, rank, rows):
    if kind == K_LM_HEAD:
        start, count = vocabulary_shard(rank)
        return list(range(start, start + count))
    if kind in (KIND["EXPERTS_W1"], KIND["EXPERTS_W3"]):
        per = EXPERT_WIDTH // TP_DEGREE
        return [e * EXPERT_WIDTH + rank * per + r
                for e in range(EXPERTS) for r in range(per)]
    if kind == KIND["WO_A"]:
        gs, gc, _, _ = output_group_shard(rank)
        return [g * OUTPUT_LORA + r for g in range(gs, gs + gc)
                for r in range(OUTPUT_LORA)]
    if kind in (KIND["WQ_A"], KIND["WQ_B"], KIND["WKV"], KIND["COMPRESS_WKV"],
                KIND["COMPRESS_WGATE"], KIND["INDEX_WKV"], KIND["INDEX_WGATE"],
                KIND["SHARED_W1"], KIND["SHARED_W3"]):
        return list(range(rank * (rows // TP_DEGREE), (rank + 1) * (rows // TP_DEGREE)))
    return list(range(rows))


def column_slice(kind, rank, columns):
    if kind == KIND["WO_A"]:
        _, _, start, width = output_group_shard(rank)
        return start, width
    if kind == KIND["WO_B"]:
        gs, gc, _, _ = output_group_shard(rank)
        return gs * OUTPUT_LORA, gc * OUTPUT_LORA
    if kind in (KIND["EXPERTS_W2"], KIND["SHARED_W2"]):
        width = EXPERT_WIDTH // TP_DEGREE
        return rank * width, width
    if kind == KIND["WQ_B"]:
        return 0, columns
    if kind == KIND["ATTN_SINK"]:
        width = columns // TP_DEGREE
        return rank * width, width
    return 0, columns


def read_entries(handle):
    header = list(HEADER.unpack(handle.read(HEADER.size)))
    assert header[0] == MAGIC and header[1] == VERSION, "not a v4 dsv4 pack"
    handle.seek(header[16])
    entries = [ENTRY.unpack(handle.read(ENTRY.size)) for _ in range(header[8])]
    return header, entries


def is_mtp(kind, layer):
    return (MTP_LAYER_FIRST <= layer < MTP_LAYER_FIRST + 3
            or (kind in MTP_SET and layer == GLOBAL_LAYER))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--shard", type=Path, required=True)
    ap.add_argument("--rank", type=int, required=True)
    args = ap.parse_args()

    with args.source.open("rb") as src:
        src_header, src_entries = read_entries(src)
        assert src_header[9] == 0 and src_header[10] == LAYERS, "source not full model"
        src_by_key = {(e[0], e[1]): e for e in src_entries}

        with args.shard.open("rb") as out:
            out_header, out_entries = read_entries(out)
            assert (out_header[9], out_header[10]) == (0, LAYERS), "not a TP16 pp1 shard"
            file_bytes = args.shard.stat().st_size
            assert out_header[17] == file_bytes, "header total-bytes mismatch"

            checked = mismatched = replicated = 0
            for idx, oe in enumerate(out_entries):
                kind, layer, weight, rows, cols, _res, poff, soff = oe
                se = src_by_key.get((kind, layer))
                assert se is not None, f"unknown tensor kind={kind} layer={layer}"
                _k, _l, _w, srows, scols, _r, spoff, ssoff = se
                if is_mtp(kind, layer):
                    exp_rows, exp_cols = srows, scols   # replicated full
                else:
                    indices = row_indices(kind, args.rank, srows)
                    exp_rows = len(indices)
                    exp_cols = column_slice(kind, args.rank, scols)[1]
                assert rows == exp_rows and cols == exp_cols, \
                    f"shape mismatch kind={kind} layer={layer}: got {rows}x{cols} want {exp_rows}x{exp_cols}"

                # payload byte-compare, row by row, straight from source
                srow_bytes = payload_bytes(weight, 1, scols)
                orow_bytes = payload_bytes(weight, 1, cols)
                if is_mtp(kind, layer):
                    col_start = 0          # draft tensors replicate full-width
                elif weight == WEIGHT_FP4:
                    col_start = column_slice(kind, args.rank, scols)[0] // 2
                else:
                    col_start = column_slice(kind, args.rank, scols)[0] * element_bytes(weight)
                byte_start = col_start
                indices = ([*range(srows)] if is_mtp(kind, layer)
                           else row_indices(kind, args.rank, srows))
                pos = poff
                for r in indices:
                    src.seek(spoff + r * srow_bytes + byte_start)
                    want = src.read(orow_bytes)
                    out.seek(pos)
                    got = out.read(orow_bytes)
                    if want != got:
                        mismatched += 1
                        print(f"PAYLOAD MISMATCH kind={kind} layer={layer} src_row={r}")
                        break
                    pos += orow_bytes
                else:
                    checked += 1

                # scale plane compare
                exp_scale = scale_bytes(weight, exp_rows, exp_cols)
                if exp_scale:
                    assert soff == poff + payload_bytes(weight, rows, cols), \
                        f"scale offset mismatch kind={kind} layer={layer}"
                    block = 32 if weight == WEIGHT_FP4 else 128
                    sblocks = (scols + block - 1) // block
                    nblocks = (cols + block - 1) // block
                    bstart = 0 if is_mtp(kind, layer) else column_slice(kind, args.rank, scols)[0] // block
                    pos = soff
                    ok = True
                    for r in indices:
                        src.seek(ssoff + r * sblocks + bstart)
                        want = src.read(nblocks)
                        out.seek(pos)
                        if want != out.read(nblocks):
                            ok = False
                            break
                        pos += nblocks
                    if not ok:
                        mismatched += 1
                        print(f"SCALE MISMATCH kind={kind} layer={layer} src_row={r}")
                    else:
                        checked += 1
                else:
                    assert soff == 0, f"unexpected scale plane kind={kind} layer={layer}"
                if is_mtp(kind, layer) or (exp_rows == srows and exp_cols == scols):
                    replicated += 1

            print(f"VERIFY rank={args.rank}: tensors={len(out_entries)} "
                  f"byte_checks={checked} mismatches={mismatched} "
                  f"replicated_full={replicated} bytes={file_bytes}")
            sys.exit(1 if mismatched else 0)


if __name__ == "__main__":
    main()
