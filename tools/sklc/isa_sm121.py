import struct

RZ = 0xFF
URZ = 0xFF
PT = 7

SR_LANEID = 0x00
SR_TID_X = 0x21
SR_TID_Y = 0x22
SR_CTAID_X = 0x25
SR_CTAID_Y = 0x26
SR_CTAID_Z = 0x27

C0_GLOBAL_DESC = 0x358
C0_NTID_X = 0x360
C0_NTID_Y = 0x364
C0_NCTAID_X = 0x370
C0_NCTAID_Y = 0x374
C0_PARAM_BASE = 0x380


def pred_bits(pred=PT, neg=False):
    return ((1 if neg else 0) << 3) | (pred & 0x7)


def ctrl_bits(stall=1, yield_=False, wr=7, rd=7, wait=0, reuse=0):
    if stall == 0 or stall > 11:
        yield_ = True
    return (
        ((stall & 0xF) << 41)
        | ((0 if yield_ else 1) << 45)
        | ((wr & 0x7) << 46)
        | ((rd & 0x7) << 49)
        | ((wait & 0x3F) << 52)
        | ((reuse & 0xF) << 58)
    )


def f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def pack_insn(lo, hi):
    return struct.pack("<QQ", lo & 0xFFFFFFFFFFFFFFFF, hi & 0xFFFFFFFFFFFFFFFF)


def encode_nop(stall=0, yield_=True, wait=0):
    lo = 0x7918
    hi = ctrl_bits(stall=stall, yield_=yield_, wr=7, rd=7, wait=wait)
    return (lo, hi)


def encode_exit(pred=PT, neg=False, stall=5, yield_=False, wait=0):
    lo = 0x94D | (pred_bits(pred, neg) << 12)
    hi = 0x0000000003800000 | ctrl_bits(stall=stall, yield_=yield_, wr=7, rd=7, wait=wait)
    return (lo, hi)


def encode_bra(delta_bytes, pred=PT, neg=False, stall=5, yield_=False, wait=0):
    off4 = delta_bytes >> 2
    lo = (
        0x947
        | (pred_bits(pred, neg) << 12)
        | ((off4 & 0xFF) << 16)
        | (((off4 >> 8) & 0x3FFFFFFF) << 34)
    )
    hi = (
        0x0000000003800000
        | ((off4 >> 38) & 0x3FFFF)
        | ctrl_bits(stall=stall, yield_=yield_, wr=7, rd=7, wait=wait)
    )
    return (lo, hi)


def encode_bar_sync(bar=0, stall=1, wait=0):
    lo = 0x7B1D | ((bar & 0xF) << 24)
    hi = ctrl_bits(stall=stall, yield_=True, wr=7, rd=7, wait=wait)
    return (lo, hi)


def encode_bssy(delta_bytes, bar=0, stall=1, wait=0):
    lo = 0x7945 | ((bar & 0xF) << 16) | ((delta_bytes & 0xFFFFFFFF) << 32)
    hi = 0x0000000003800200 | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    return (lo, hi)


def encode_bsync(bar=0, stall=5, wait=0):
    lo = 0x7941 | ((bar & 0xF) << 16)
    hi = 0x0000000003800200 | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    return (lo, hi)


def encode_acqbulk(stall=1, wait=0):
    lo = 0x782E
    hi = ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    return (lo, hi)


def encode_preexit(stall=8, wait=0):
    lo = 0x782D
    hi = ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    return (lo, hi)


def encode_s2r(rd, sr, wr=0, stall=1, wait=0):
    lo = 0x7919 | ((rd & 0xFF) << 16)
    hi = ((sr & 0xFF) << 8) | ctrl_bits(stall=stall, yield_=False, wr=wr, rd=7, wait=wait)
    return (lo, hi)


def encode_s2ur(urd, sr, wr=0, stall=1, wait=0):
    lo = 0x79C3 | ((urd & 0xFF) << 16)
    hi = ((sr & 0xFF) << 8) | ctrl_bits(stall=stall, yield_=False, wr=wr, rd=7, wait=wait)
    return (lo, hi)


def encode_ldc(rd, offset, width=32, wr=0, stall=1, wait=0):
    sz_bits = {32: 0x0800, 64: 0x0A00}[width]
    off_enc = (offset >> 2) & 0xFFFF
    lo = 0x7B82 | ((rd & 0xFF) << 16) | (0xFF << 24) | (off_enc << 40)
    hi = sz_bits | ctrl_bits(stall=stall, yield_=False, wr=wr, rd=7, wait=wait)
    return (lo, hi)


def encode_ldcu(urd, offset, width=32, wr=0, stall=1, wait=0):
    sz_bits = {32: 0x0800, 64: 0x0A00, 128: 0x0C00}[width]
    off_enc = (offset >> 3) & 0xFFFF
    extra = (offset >> 2) & 1
    lo = 0x77AC | ((urd & 0xFF) << 16) | (0xFF << 24) | (((off_enc << 1) | extra) << 39)
    hi = 0x08000000 | sz_bits | ctrl_bits(stall=stall, yield_=False, wr=wr, rd=7, wait=wait)
    return (lo, hi)


def encode_mov_r(rd, rb, pred=PT, neg=False, stall=1, wait=0):
    lo = 0x202 | (pred_bits(pred, neg) << 12) | ((rd & 0xFF) << 16) | ((rb & 0xFF) << 32)
    hi = 0x0000000000000F00 | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    return (lo, hi)


def encode_mov_i(rd, imm32, pred=PT, neg=False, stall=1, wait=0):
    lo = 0x802 | (pred_bits(pred, neg) << 12) | ((rd & 0xFF) << 16) | ((imm32 & 0xFFFFFFFF) << 32)
    hi = 0x0000000000000F00 | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    return (lo, hi)


def encode_mov_u(rd, urb, pred=PT, neg=False, stall=1, wait=0):
    lo = 0xC02 | (pred_bits(pred, neg) << 12) | ((rd & 0xFF) << 16) | ((urb & 0xFF) << 32)
    hi = 0x0000000008000F00 | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    return (lo, hi)


def encode_umov_i(urd, imm32, stall=1, wait=0):
    lo = 0x7882 | ((urd & 0xFF) << 16) | ((imm32 & 0xFFFFFFFF) << 32)
    hi = ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    return (lo, hi)


def encode_iadd3_r(rd, ra, rb, rc=RZ, neg_a=False, neg_b=False, pred=PT, neg=False, stall=1, wait=0):
    lo = 0x210 | (pred_bits(pred, neg) << 12) | ((rd & 0xFF) << 16) | ((ra & 0xFF) << 24) | ((rb & 0xFF) << 32)
    if neg_b:
        lo |= 1 << 63
    hi = (
        0x07FFE000
        | (rc & 0xFF)
        | ((1 if neg_a else 0) << 8)
        | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    )
    return (lo, hi)


def encode_iadd3_i(rd, ra, imm32, rc=RZ, neg_a=False, pred=PT, neg=False, stall=1, wait=0):
    lo = 0x810 | (pred_bits(pred, neg) << 12) | ((rd & 0xFF) << 16) | ((ra & 0xFF) << 24) | ((imm32 & 0xFFFFFFFF) << 32)
    hi = (
        0x07FFE000
        | (rc & 0xFF)
        | ((1 if neg_a else 0) << 8)
        | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    )
    return (lo, hi)


def encode_iadd64(rd, ra, rb, neg_a=False, neg_b=False, stall=1, wait=0):
    lo = 0x7235 | ((rd & 0xFF) << 16) | ((ra & 0xFF) << 24) | ((rb & 0xFF) << 32)
    if neg_b:
        lo |= 1 << 63
    hi = 0x078E0200 | ((1 if neg_a else 0) << 8) | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    return (lo, hi)


def encode_imad_i(rd, ra, imm32, rc=RZ, stall=1, wait=0):
    lo = 0x7824 | ((rd & 0xFF) << 16) | ((ra & 0xFF) << 24) | ((imm32 & 0xFFFFFFFF) << 32)
    hi = 0x078E0000 | (rc & 0xFF) | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    return (lo, hi)


def encode_imad_r(rd, ra, rb, rc=RZ, stall=1, wait=0):
    lo = 0x7224 | ((rd & 0xFF) << 16) | ((ra & 0xFF) << 24) | ((rb & 0xFF) << 32)
    hi = 0x078E0000 | (rc & 0xFF) | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    return (lo, hi)


def encode_imad_u(rd, ra, urb, rc=RZ, stall=1, wait=0):
    lo = 0x7C24 | ((rd & 0xFF) << 16) | ((ra & 0xFF) << 24) | ((urb & 0xFF) << 32)
    hi = 0x0F8E0200 | (rc & 0xFF) | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    return (lo, hi)


def encode_imad_wide_i(rd, ra, imm32, rc, stall=1, wait=0):
    lo = 0x7825 | ((rd & 0xFF) << 16) | ((ra & 0xFF) << 24) | ((imm32 & 0xFFFFFFFF) << 32)
    hi = 0x078E0000 | (rc & 0xFF) | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    return (lo, hi)


def encode_imad_wide_r(rd, ra, rb, rc, stall=1, wait=0):
    lo = 0x7225 | ((rd & 0xFF) << 16) | ((ra & 0xFF) << 24) | ((rb & 0xFF) << 32)
    hi = 0x078E0000 | (rc & 0xFF) | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    return (lo, hi)


def encode_shf_r_u32_hi_i(rd, shift, rc, stall=1, wait=0):
    lo = 0x7819 | ((rd & 0xFF) << 16) | (0xFF << 24) | ((shift & 0xFFFFFFFF) << 32)
    hi = 0x00011600 | (rc & 0xFF) | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    return (lo, hi)


def encode_shf_r_u32_hi_r(rd, rshift, rc, stall=1, wait=0):
    lo = 0x7219 | ((rd & 0xFF) << 16) | (0xFF << 24) | ((rshift & 0xFF) << 32)
    hi = 0x00011600 | (rc & 0xFF) | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    return (lo, hi)


def encode_shf_l_i(rd, ra, shift, rc=RZ, stall=1, wait=0):
    lo = 0x7819 | ((rd & 0xFF) << 16) | ((ra & 0xFF) << 24) | ((shift & 0xFFFFFFFF) << 32)
    hi = 0x00000600 | (rc & 0xFF) | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    return (lo, hi)


def encode_shf_l_r(rd, ra, rb, rc=RZ, stall=1, wait=0):
    lo = 0x7219 | ((rd & 0xFF) << 16) | ((ra & 0xFF) << 24) | ((rb & 0xFF) << 32)
    hi = 0x00000600 | (rc & 0xFF) | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    return (lo, hi)


def encode_lop3_i(rd, ra, imm32, rc=RZ, lut=0xC0, pd=PT, stall=1, wait=0):
    lo = 0x7812 | ((rd & 0xFF) << 16) | ((ra & 0xFF) << 24) | ((imm32 & 0xFFFFFFFF) << 32)
    hi = (
        0x07800000
        | ((pd & 0x7) << 17)
        | ((lut & 0xFF) << 8)
        | (rc & 0xFF)
        | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    )
    return (lo, hi)


def encode_lop3_r(rd, ra, rb, rc=RZ, lut=0xC0, pd=PT, stall=1, wait=0):
    lo = 0x7212 | ((rd & 0xFF) << 16) | ((ra & 0xFF) << 24) | ((rb & 0xFF) << 32)
    hi = (
        0x07800000
        | ((pd & 0x7) << 17)
        | ((lut & 0xFF) << 8)
        | (rc & 0xFF)
        | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    )
    return (lo, hi)


def encode_prmt_i(rd, ra, sel16, rc=RZ, stall=1, wait=0):
    lo = 0x7816 | ((rd & 0xFF) << 16) | ((ra & 0xFF) << 24) | ((sel16 & 0xFFFF) << 32)
    hi = (rc & 0xFF) | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    return (lo, hi)


CMP_LT = 1
CMP_EQ = 2
CMP_LE = 3
CMP_GT = 4
CMP_NE = 5
CMP_GE = 6


def encode_isetp_r(pd, ra, rb, cmp_op=CMP_GE, unsigned=True, stall=1, wait=0):
    lo = 0x720C | ((ra & 0xFF) << 24) | ((rb & 0xFF) << 32)
    hi = (
        0x03F00070
        | ((pd & 0x7) << 17)
        | ((cmp_op & 0x7) << 12)
        | ((0 if unsigned else 1) << 9)
        | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    )
    return (lo, hi)


def encode_isetp_i(pd, ra, imm32, cmp_op=CMP_GE, unsigned=True, stall=1, wait=0):
    lo = 0x780C | ((ra & 0xFF) << 24) | ((imm32 & 0xFFFFFFFF) << 32)
    hi = (
        0x03F00070
        | ((pd & 0x7) << 17)
        | ((cmp_op & 0x7) << 12)
        | ((0 if unsigned else 1) << 9)
        | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    )
    return (lo, hi)


def encode_isetp_u(pd, ra, urb, cmp_op=CMP_GE, unsigned=True, stall=1, wait=0):
    lo = 0x7C0C | ((ra & 0xFF) << 24) | ((urb & 0xFF) << 32)
    hi = (
        0x0BF00070
        | ((pd & 0x7) << 17)
        | ((cmp_op & 0x7) << 12)
        | ((0 if unsigned else 1) << 9)
        | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    )
    return (lo, hi)


def encode_fsetp_r(pd, ra, rb, cmp_op=CMP_GT, stall=1, wait=0):
    lo = 0x720B | ((ra & 0xFF) << 24) | ((rb & 0xFF) << 32)
    hi = (
        0x03F00000
        | ((pd & 0x7) << 17)
        | ((cmp_op & 0xF) << 12)
        | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    )
    return (lo, hi)


def encode_fsetp_i(pd, ra, fval, cmp_op=CMP_GT, stall=1, wait=0):
    lo = 0x780B | ((ra & 0xFF) << 24) | (f32_bits(fval) << 32)
    hi = (
        0x03F00000
        | ((pd & 0x7) << 17)
        | ((cmp_op & 0xF) << 12)
        | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    )
    return (lo, hi)


def encode_sel_i(rd, ra, imm32, sp, neg_sp=False, stall=1, wait=0):
    lo = 0x7807 | ((rd & 0xFF) << 16) | ((ra & 0xFF) << 24) | ((imm32 & 0xFFFFFFFF) << 32)
    hi = (((1 if neg_sp else 0) << 26) | ((sp & 0x7) << 23)) | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    return (lo, hi)


def encode_sel_r(rd, ra, rb, sp, neg_sp=False, stall=1, wait=0):
    lo = 0x7207 | ((rd & 0xFF) << 16) | ((ra & 0xFF) << 24) | ((rb & 0xFF) << 32)
    hi = (((1 if neg_sp else 0) << 26) | ((sp & 0x7) << 23)) | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    return (lo, hi)


def encode_i2fp_u32(rd, rb, stall=1, wait=0):
    lo = 0x7245 | ((rd & 0xFF) << 16) | ((rb & 0xFF) << 32)
    hi = 0x00201000 | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    return (lo, hi)


def encode_ffma_r(rd, ra, rb, rc, pred=PT, neg=False, stall=1, wait=0, reuse=0):
    lo = 0x223 | (pred_bits(pred, neg) << 12) | ((rd & 0xFF) << 16) | ((ra & 0xFF) << 24) | ((rb & 0xFF) << 32)
    hi = (rc & 0xFF) | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait, reuse=reuse)
    return (lo, hi)


def encode_ffma_i(rd, ra, fval, rc, pred=PT, neg=False, stall=1, wait=0):
    lo = 0x823 | (pred_bits(pred, neg) << 12) | ((rd & 0xFF) << 16) | ((ra & 0xFF) << 24) | (f32_bits(fval) << 32)
    hi = (rc & 0xFF) | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    return (lo, hi)


def encode_ffma_u(rd, ra, urb, rc, pred=PT, neg=False, stall=1, wait=0):
    lo = 0xC23 | (pred_bits(pred, neg) << 12) | ((rd & 0xFF) << 16) | ((ra & 0xFF) << 24) | ((urb & 0xFF) << 32)
    hi = 0x08000000 | (rc & 0xFF) | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    return (lo, hi)


def encode_fadd_r(rd, ra, rb, neg_a=False, neg_b=False, abs_a=False, pred=PT, neg=False, stall=1, wait=0):
    lo = 0x221 | (pred_bits(pred, neg) << 12) | ((rd & 0xFF) << 16) | ((ra & 0xFF) << 24) | ((rb & 0xFF) << 32)
    if neg_b:
        lo |= 1 << 63
    hi = (
        ((1 if neg_a else 0) << 8)
        | ((1 if abs_a else 0) << 9)
        | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    )
    return (lo, hi)


def encode_fadd_i(rd, ra, fval, abs_a=False, pred=PT, neg=False, stall=1, wait=0):
    lo = 0x421 | (pred_bits(pred, neg) << 12) | ((rd & 0xFF) << 16) | ((ra & 0xFF) << 24) | (f32_bits(fval) << 32)
    hi = ((1 if abs_a else 0) << 9) | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    return (lo, hi)


def encode_fmul_r(rd, ra, rb, pred=PT, neg=False, stall=1, wait=0):
    lo = 0x220 | (pred_bits(pred, neg) << 12) | ((rd & 0xFF) << 16) | ((ra & 0xFF) << 24) | ((rb & 0xFF) << 32)
    hi = 0x00400000 | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    return (lo, hi)


def encode_fmul_i(rd, ra, fval, pred=PT, neg=False, stall=1, wait=0):
    lo = 0x820 | (pred_bits(pred, neg) << 12) | ((rd & 0xFF) << 16) | ((ra & 0xFF) << 24) | (f32_bits(fval) << 32)
    hi = 0x00400000 | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    return (lo, hi)


def encode_fmnmx_r(rd, ra, rb, is_max=True, abs_a=False, abs_b=False, stall=1, wait=0):
    lo = 0x7209 | ((rd & 0xFF) << 16) | ((ra & 0xFF) << 24) | ((rb & 0xFF) << 32)
    if abs_b:
        lo |= 1 << 62
    hi = (
        (0x07800000 if is_max else 0x03800000)
        | ((1 if abs_a else 0) << 9)
        | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    )
    return (lo, hi)


MUFU_EX2 = 0x0800
MUFU_RCP = 0x1000
MUFU_RSQ = 0x1400


def encode_mufu(rd, rb, op=MUFU_RCP, wr=0, stall=1, wait=0):
    lo = 0x7308 | ((rd & 0xFF) << 16) | ((rb & 0xFF) << 32)
    hi = op | ctrl_bits(stall=stall, yield_=False, wr=wr, rd=7, wait=wait)
    return (lo, hi)


def encode_f2f_bf16(rd, rb, stall=1, wait=0):
    lo = 0x7304 | ((rd & 0xFF) << 16) | ((rb & 0xFF) << 32)
    hi = 0x00202000 | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    return (lo, hi)


def encode_f2fp_bf16_pack(rd, rhi, rlo, stall=1, wait=0):
    lo = 0x723E | ((rd & 0xFF) << 16) | ((rhi & 0xFF) << 24) | ((rlo & 0xFF) << 32)
    hi = 0x000010FF | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    return (lo, hi)


def encode_f2fp_e4m3_pack(rd, rhi, rlo, stall=1, wait=0):
    lo = 0x723E | ((rd & 0xFF) << 16) | ((rhi & 0xFF) << 24) | ((rlo & 0xFF) << 32)
    hi = 0x048070FF | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    return (lo, hi)


def encode_f2fp_e2m1_unpack(rd, rb, stall=1, wait=0):
    lo = 0x723E | ((rd & 0xFF) << 16) | (0xFF << 24) | ((rb & 0xFF) << 32)
    hi = 0x020402FF | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    return (lo, hi)


def encode_f2fp_e4m3_unpack(rd, rb, hi_half=False, stall=1, wait=0):
    lo = 0x723E | ((rd & 0xFF) << 16) | (0xFF << 24) | ((rb & 0xFF) << 32)
    hi = (0x030006FF if hi_half else 0x020006FF) | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    return (lo, hi)


def encode_hadd2_f32_half(rd, rb, hi_half=False, stall=1, wait=0):
    lo = 0x7230 | ((rd & 0xFF) << 16) | (0xFF << 24) | ((rb & 0xFF) << 32) | ((0x30 if hi_half else 0x20) << 56)
    hi = 0x00004100 | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    return (lo, hi)


def encode_shfl_bfly(rd, ra, lane_xor, clamp=0x1F, wr=0, stall=1, wait=0):
    lo = 0x7F89 | ((rd & 0xFF) << 16) | ((ra & 0xFF) << 24) | ((clamp & 0x1F) << 40) | ((lane_xor & 0x1F) << 53) | (0x3 << 58)
    hi = 0x000E0000 | ctrl_bits(stall=stall, yield_=False, wr=wr, rd=7, wait=wait)
    return (lo, hi)


def encode_shfl_down(rd, ra, delta, clamp=0x1F, wr=0, stall=1, wait=0):
    lo = 0x7F89 | ((rd & 0xFF) << 16) | ((ra & 0xFF) << 24) | ((clamp & 0x1F) << 40) | ((delta & 0x1F) << 53) | (0x2 << 58)
    hi = 0x000E0000 | ctrl_bits(stall=stall, yield_=False, wr=wr, rd=7, wait=wait)
    return (lo, hi)


def encode_shfl_up(rd, ra, delta, clamp=0x00, wr=0, stall=1, wait=0):
    lo = 0x7F89 | ((rd & 0xFF) << 16) | ((ra & 0xFF) << 24) | ((clamp & 0x1F) << 40) | ((delta & 0x1F) << 53) | (0x1 << 58)
    hi = 0x000E0000 | ctrl_bits(stall=stall, yield_=False, wr=wr, rd=7, wait=wait)
    return (lo, hi)


def encode_shfl_idx(rd, ra, lane_reg=RZ, clamp=0x1F, wr=0, stall=1, wait=0):
    lo = 0x7589 | ((rd & 0xFF) << 16) | ((ra & 0xFF) << 24) | ((lane_reg & 0xFF) << 32) | ((clamp & 0x1F) << 40)
    hi = 0x000E0000 | ctrl_bits(stall=stall, yield_=False, wr=wr, rd=7, wait=wait)
    return (lo, hi)


LDG_W_U8 = 0x1100
LDG_W_U16 = 0x1500
LDG_W_32 = 0x1900
LDG_W_64 = 0x1B00
LDG_W_128 = 0x1D00

CACHE_DEFAULT = 0x0C1E0000
CACHE_EF = 0x0C0E0000
CACHE_CONSTANT = 0x0C1E8000


def encode_ldg(rd, raddr, ur_desc, width_bits=LDG_W_128, cache=CACHE_DEFAULT, offset=0, pred=PT, neg=False, wr=0, rd_bar=7, stall=1, wait=0):
    lo = (
        0x981
        | (pred_bits(pred, neg) << 12)
        | ((rd & 0xFF) << 16)
        | ((raddr & 0xFF) << 24)
        | ((ur_desc & 0xFF) << 32)
        | ((offset & 0xFFFFFF) << 40)
    )
    hi = cache | width_bits | ctrl_bits(stall=stall, yield_=False, wr=wr, rd=rd_bar, wait=wait)
    return (lo, hi)


def encode_ldg256(rd_lo, rd_hi, raddr, ur_desc, policy="default", pred=PT, neg=False, wr=0, rd_bar=7, stall=1, wait=0):
    pol_bits = {"ef": 0x08021900, "default": 0x08121900, "el": 0x08221900}[policy]
    lo = (
        0x97E
        | (pred_bits(pred, neg) << 12)
        | ((rd_lo & 0xFF) << 16)
        | ((raddr & 0xFF) << 24)
        | ((ur_desc & 0xFF) << 32)
        | (0xFE << 56)
    )
    hi = pol_bits | (rd_hi & 0xFF) | ctrl_bits(stall=stall, yield_=False, wr=wr, rd=rd_bar, wait=wait)
    return (lo, hi)


STG_W_U8 = 0x1100
STG_W_U16 = 0x1500
STG_W_32 = 0x1900
STG_W_64 = 0x1B00
STG_W_128 = 0x1D00


STG_CACHE_DEFAULT = 0x0C100000
STG_CACHE_EF = 0x0C000000
STG_CACHE_STRONG_SYS = 0x0C114000


def encode_stg(raddr, rval, ur_desc, width_bits=STG_W_128, cache=STG_CACHE_DEFAULT, offset=0, pred=PT, neg=False, rd=7, stall=1, wait=0):
    lo = (
        0x986
        | (pred_bits(pred, neg) << 12)
        | ((raddr & 0xFF) << 24)
        | ((rval & 0xFF) << 32)
        | ((offset & 0xFFFFFF) << 40)
    )
    hi = cache | width_bits | (ur_desc & 0xFF) | ctrl_bits(stall=stall, yield_=False, wr=7, rd=rd, wait=wait)
    return (lo, hi)


def encode_stg256(raddr, rval_lo, rval_hi, ur_desc, pred=PT, neg=False, rd=7, stall=1, wait=0):
    lo = (
        0x97F
        | (pred_bits(pred, neg) << 12)
        | ((rval_hi & 0xFF) << 16)
        | ((raddr & 0xFF) << 24)
        | ((rval_lo & 0xFF) << 32)
        | (0xF8 << 56)
    )
    hi = 0x0F121800 | (ur_desc & 0xFF) | ctrl_bits(stall=stall, yield_=False, wr=7, rd=rd, wait=wait)
    return (lo, hi)


def encode_lds(rd, raddr, ur_base=URZ, offset=0, wr=0, rd_bar=7, stall=1, wait=0):
    lo = (
        0x7984
        | ((rd & 0xFF) << 16)
        | ((raddr & 0xFF) << 24)
        | ((ur_base & 0xFF) << 32)
        | ((offset & 0xFFFFFF) << 40)
    )
    hi = 0x08000800 | ctrl_bits(stall=stall, yield_=False, wr=wr, rd=rd_bar, wait=wait)
    return (lo, hi)


def encode_lds128(rd, raddr, ur_base=URZ, offset=0, wr=0, rd_bar=7, stall=1, wait=0):
    lo = (
        0x7984
        | ((rd & 0xFF) << 16)
        | ((raddr & 0xFF) << 24)
        | ((ur_base & 0xFF) << 32)
        | ((offset & 0xFFFFFF) << 40)
    )
    hi = 0x08000C00 | ctrl_bits(stall=stall, yield_=False, wr=wr, rd=rd_bar, wait=wait)
    return (lo, hi)


def encode_sts(raddr, rval, offset=0, pred=PT, neg=False, rd_bar=7, stall=1, wait=0):
    lo = (
        0x388
        | (pred_bits(pred, neg) << 12)
        | ((raddr & 0xFF) << 24)
        | ((rval & 0xFF) << 32)
        | ((offset & 0xFFFFFF) << 40)
    )
    hi = 0x00000800 | ctrl_bits(stall=stall, yield_=False, wr=7, rd=rd_bar, wait=wait)
    return (lo, hi)


def encode_ublkpf_l2(ur_addr, ur_bytes, stall=1, wait=0):
    lo = 0x73BC | ((ur_addr & 0xFF) << 24)
    hi = 0x08000000 | (ur_bytes & 0xFF) | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    return (lo, hi)


def encode_hmma_16816_f32_bf16(rd, ra, rb, rc=RZ, stall=1, wait=0):
    lo = 0x723C | ((rd & 0xFF) << 16) | ((ra & 0xFF) << 24) | ((rb & 0xFF) << 32)
    hi = 0x00041800 | (rc & 0xFF) | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    return (lo, hi)


def encode_ldgsts128(r_smem, r_gaddr, ur_desc, offset=0, pred=PT, neg=False, rd_bar=7, stall=1, wait=0):
    lo = (
        0x0FAE
        | (pred_bits(pred, neg) << 12)
        | ((r_smem & 0xFF) << 16)
        | ((r_gaddr & 0xFF) << 24)
        | ((offset & 0xFFFFFFFF) << 32)
    )
    hi = 0x0B981800 | (ur_desc & 0xFF) | ctrl_bits(stall=stall, yield_=False, wr=7, rd=rd_bar, wait=wait)
    return (lo, hi)


def encode_ldgdepbar(wr=0, stall=1, wait=0):
    lo = 0x79AF
    hi = ctrl_bits(stall=stall, yield_=False, wr=wr, rd=7, wait=wait)
    return (lo, hi)


def encode_depbar_le(count=0, stall=7, wait=0):
    lo = 0x791A | (0x8000 << 32) | ((count & 0x3F) << 38)
    hi = ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    return (lo, hi)


MEMBAR_SC_SYS = 0x00003000
MEMBAR_ALL_SYS = 0x0000B000


def encode_membar(mode=MEMBAR_SC_SYS, stall=6, wait=0):
    lo = 0x7992
    hi = mode | ctrl_bits(stall=stall, yield_=False, wr=7, rd=7, wait=wait)
    return (lo, hi)
