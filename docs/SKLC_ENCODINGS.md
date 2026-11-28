# SKLC `sm_121` (Blackwell) SASS Instruction Encodings and CUBIN Format

This document is the encoding reference and disassembly verification record for
`skl2/compiler/sklc/isa_sm121.py` and `skl2/compiler/sklc/cubin_elf.py`.

`skl2/compiler/sklc/` is the low-level Blackwell codegen/CUBIN backend under `skl2/compiler/`, tested by
`tests/test_sklc.py` and `tests/test_skl2c.py`. Every instruction encoding below is verified
against NVIDIA's `cuobjdump --dump-sass` and `nvdisasm` (CUDA 13.0) for target
`sm_121a` (Blackwell GB10).

## 1. 128-Bit Instruction Word Layout

Blackwell (`sm_121` / `sm_120`) SASS instructions are fixed 128-bit words
packed little-endian as two 64-bit unsigned integers `(lo, hi)`
(`struct.pack("<QQ", lo, hi)` in `skl2/compiler/sklc/isa_sm121.py`).

### Low 64-Bit Word (`lo`)

| Bits | Width | Field | Notes |
|---|---|---|---|
| `lo[11:0]` | 12 | Opcode & operand-source class | Low 9 bits select the operation; bits `[11:9]` select operand B source class (`0x200` register `Rb`, `0x400`/`0x800` 32-bit immediate, `0xC00` uniform register `URb`, `0x900` global memory / control flow). |
| `lo[14:12]` | 3 | Guard predicate `Pg` | `0..6` = `P0..P6`, `7` = `PT` (always true). |
| `lo[15]` | 1 | Guard predicate invert | `0` = `@Pg`, `1` = `@!Pg`. Unconditional instructions set `lo[15:12] = 0x7`, producing `0x7xxx` in `lo[15:0]`. |
| `lo[23:16]` | 8 | Destination register `Rd` / `URd` | `0xFF` = `RZ` / `URZ`. On `BSSY` / `BSYNC`, bits `[19:16]` hold convergence barrier `B0..B15`. On `BRA`, bits `[23:16]` hold `off4[7:0]`. |
| `lo[31:24]` | 8 | Source register `Ra` / `URa` | `0xFF` = `RZ` / `URZ`. On `BAR.SYNC`, bits `[27:24]` hold barrier index `0..15`. |
| `lo[63:32]` | 32 | Operand B / immediate / address tail | Register `Rb`/`URb` in `lo[39:32]`, or 32-bit integer / IEEE-754 float immediate, or `[ur_desc:8 | offset:24]` for `LDG`/`STG`/`LDS`/`STS`, or `[clamp:5 | lane_xor:5 | mode:2]` for `SHFL`. |
| `lo[62]` | 1 | `|Rb|` modifier | Absolute value on `Rb` (`FMNMX`). |
| `lo[63]` | 1 | `-Rb` modifier | Negate `Rb` (`IADD3`, `IADD.64`, `FADD`). |

### High 64-Bit Word (`hi`)

| Bits | Width | Field | Notes |
|---|---|---|---|
| `hi[7:0]` | 8 | Operand C / aux register | `Rc` (`IADD3`, `IMAD`, `IMAD.WIDE`, `SHF`, `LOP3`, `PRMT`, `FFMA`), `Rd_hi` (`LDG.256`), `ur_desc` (`STG`, `STG.256`), or `ur_bytes` (`UBLKPF.L2`). |
| `hi[40:8]` | 33 | Operation modifiers | Special register ID (`hi[15:8]` on `S2R`/`S2UR`), `-Ra` (`hi[8]`), `|Ra|` (`hi[9]`), `LOP3` 8-bit truth table (`hi[15:8]`), `ISETP`/`FSETP` compare op (`hi[15:12]`) and `Pd` (`hi[19:17]`), `SEL` source predicate (`hi[25:23]`, invert `hi[26]`), `MUFU` op (`hi[13:10]`), uniform-bank flag (`hi[27]`), memory width & cache policy bits. |
| `hi[44:41]` | 4 | Stall count | `0..15` cycles before next warp issue (`ctrl_bits`). |
| `hi[45]` | 1 | Yield flag (inverted) | `0` = yield hint, `1` = no-yield. Forced to yield (`0`) whenever `stall == 0` or `stall > 11`. |
| `hi[48:46]` | 3 | Write dependency barrier `wr` | Scoreboard barrier `0..5` set on variable-latency write; `7` = none. |
| `hi[51:49]` | 3 | Read dependency barrier `rd` | Scoreboard barrier `0..5` set on source read; `7` = none. |
| `hi[57:52]` | 6 | Wait barrier bitmask `wait` | Bit `b` (`0..5`) stalls until scoreboard barrier `b` clears. |
| `hi[61:58]` | 4 | Operand reuse bitmask `reuse` | Register-file cache reuse hints for source operands. |

## 2. Registers, Special Registers, and Constant Bank `c[0x0]`

Constants defined in `skl2/compiler/sklc/isa_sm121.py`:

| Symbol | Value | Meaning |
|---|---|---|
| `RZ` / `URZ` | `0xFF` | Zero register (vector / uniform) |
| `PT` | `7` | Always-true predicate register |
| `SR_LANEID` | `0x00` | Special register `SR_LANEID` |
| `SR_TID_X` / `SR_TID_Y` | `0x21` / `0x22` | `SR_TID.X` / `SR_TID.Y` |
| `SR_CTAID_X` / `Y` / `Z` | `0x25` / `0x26` / `0x27` | `SR_CTAID.X` / `SR_CTAID.Y` / `SR_CTAID.Z` |
| `C0_GLOBAL_DESC` | `0x358` | Constant-bank 0 global memory descriptor (`desc[URx]`) |
| `C0_NTID_X` / `C0_NTID_Y` | `0x360` / `0x364` | CTA dimensions (`blockDim.x`, `blockDim.y`) |
| `C0_NCTAID_X` / `C0_NCTAID_Y` | `0x370` / `0x374` | Grid dimensions (`gridDim.x`, `gridDim.y`) |
| `C0_PARAM_BASE` | `0x380` | Kernel parameter base offset in `.nv.constant0.<kernel>` (Blackwell ABI v8) |

## 3. Complete Instruction Encoding Table

Every encoder function in `skl2/compiler/sklc/isa_sm121.py`, its opcode/modifier bits,
and its verified `sm_121a` SASS disassembly under `cuobjdump --dump-sass` and
`nvdisasm`:

### Control Flow and Synchronization

| Encoder | `lo[15:0]` | Key `lo` / `hi` Fields | Verified `sm_121a` Disassembly |
|---|---|---|---|
| `encode_nop` | `0x7918` | `ctrl_bits` only | `NOP` |
| `encode_exit` | `0x094D \| pred<<12` | `hi \|= 0x03800000` | `[@Pg] EXIT` |
| `encode_bra` | `0x0947 \| pred<<12` | `off4 = delta_bytes >> 2`; `lo[23:16] = off4[7:0]`, `lo[63:34] = off4[37:8]`, `hi[17:0] = off4[55:38]`, `hi \|= 0x03800000` | `[@Pg] BRA target` |
| `encode_bar_sync` | `0x7B1D` | `lo[27:24] = bar`, `yield_=True` | `BAR.SYNC 0x<bar>` |
| `encode_bssy` | `0x7945` | `lo[19:16] = bar`, `lo[63:32] = delta_bytes`, `hi \|= 0x03800200` | `BSSY.RECONVERGENT B<bar>, target` |
| `encode_bsync` | `0x7941` | `lo[19:16] = bar`, `hi \|= 0x03800200` | `BSYNC.RECONVERGENT B<bar>` |
| `encode_acqbulk` | `0x782E` | `ctrl_bits` only | `ACQBULK` |
| `encode_preexit` | `0x782D` | `ctrl_bits` only | `PREEXIT` |

### Special Register and Constant Loads

| Encoder | `lo[15:0]` | Key `lo` / `hi` Fields | Verified `sm_121a` Disassembly |
|---|---|---|---|
| `encode_s2r` | `0x7919` | `lo[23:16] = rd`, `hi[15:8] = sr` | `S2R Rd, SR_*` |
| `encode_s2ur` | `0x79C3` | `lo[23:16] = urd`, `hi[15:8] = sr` | `S2UR URd, SR_*` |
| `encode_ldc` | `0x7B82` | `lo[23:16] = rd`, `lo[31:24] = 0xFF`, `lo[55:40] = offset>>2`, `hi[11:8] = {32: 0x800, 64: 0xA00}` | `LDC[.64] Rd, c[0x0][off]` |
| `encode_ldcu` | `0x77AC` | `lo[23:16] = urd`, `lo[31:24] = 0xFF`, `lo[55:39] = offset>>2`, `hi \|= 0x08000000 \| {32: 0x800, 64: 0xA00, 128: 0xC00}` | `LDCU[.64\|.128] URd, c[0x0][off]` |

Note: `LDC` on `sm_121a` supports 32-bit (`0x0800`) and 64-bit (`0x0A00`)
widths; 128-bit constant loads use `LDCU.128` (`encode_ldcu`). Passing
`0x0C00` to `LDC` disassembles as `LDC.INVALID6`, so `encode_ldc` permits only
`width in {32, 64}`.

### Data Movement

| Encoder | `lo[15:0]` | Key `lo` / `hi` Fields | Verified `sm_121a` Disassembly |
|---|---|---|---|
| `encode_mov_r` | `0x0202 \| pred<<12` | `rd`, `rb` in `lo[39:32]`, `hi \|= 0x0F00` | `MOV Rd, Rb` |
| `encode_mov_i` | `0x0802 \| pred<<12` | `rd`, `imm32` in `lo[63:32]`, `hi \|= 0x0F00` | `MOV Rd, imm32` |
| `encode_mov_u` | `0x0C02 \| pred<<12` | `rd`, `urb` in `lo[39:32]`, `hi \|= 0x08000F00` | `MOV Rd, URb` |
| `encode_umov_i` | `0x7882` | `urd`, `imm32` in `lo[63:32]` | `UMOV URd, imm32` |

### Integer ALU, Bitwise, Shift, Permute, Compare, and Select

| Encoder | `lo[15:0]` | Key `lo` / `hi` Fields | Verified `sm_121a` Disassembly |
|---|---|---|---|
| `encode_iadd3_r` | `0x0210 \| pred<<12` | `rd`, `ra`, `rb`, `lo[63]=neg_b`, `hi \|= 0x07FFE000 \| (neg_a<<8) \| rc` | `IADD3 Rd, PT, PT, [-]Ra, [-]Rb, Rc` |
| `encode_iadd3_i` | `0x0810 \| pred<<12` | `rd`, `ra`, `imm32`, `hi \|= 0x07FFE000 \| (neg_a<<8) \| rc` | `IADD3 Rd, PT, PT, [-]Ra, imm32, Rc` |
| `encode_iadd64` | `0x7235` | `rd`, `ra`, `rb`, `lo[63]=neg_b`, `hi \|= 0x078E0200 \| (neg_a<<8)` | `IADD.64 Rd, [-]Ra, [-]Rb` |
| `encode_imad_i` | `0x7824` | `rd`, `ra`, `imm32`, `hi \|= 0x078E0000 \| rc` | `IMAD.U32 Rd, Ra, imm32, Rc` |
| `encode_imad_r` | `0x7224` | `rd`, `ra`, `rb`, `hi \|= 0x078E0000 \| rc` | `IMAD.U32 Rd, Ra, Rb, Rc` |
| `encode_imad_u` | `0x7C24` | `rd`, `ra`, `urb`, `hi \|= 0x0F8E0200 \| rc` | `IMAD Rd, Ra, URb, Rc` |
| `encode_imad_wide_i` | `0x7825` | `rd`, `ra`, `imm32`, `hi \|= 0x078E0000 \| rc` | `IMAD.WIDE.U32 Rd, Ra, imm32, Rc` |
| `encode_imad_wide_r` | `0x7225` | `rd`, `ra`, `rb`, `hi \|= 0x078E0000 \| rc` | `IMAD.WIDE.U32 Rd, Ra, Rb, Rc` |
| `encode_shf_r_u32_hi_i` | `0x7819` | `rd`, `ra=0xFF`, `shift`, `hi \|= 0x00011600 \| rc` | `SHF.R.U32.HI Rd, RZ, shift, Rc` |
| `encode_shf_r_u32_hi_r` | `0x7219` | `rd`, `ra=0xFF`, `rshift`, `hi \|= 0x00011600 \| rc` | `SHF.R.U32.HI Rd, RZ, Rshift, Rc` |
| `encode_shf_l_i` | `0x7819` | `rd`, `ra`, `shift`, `hi \|= 0x00000600 \| rc` | `SHF.L.U32 Rd, Ra, shift, Rc` |
| `encode_shf_l_r` | `0x7219` | `rd`, `ra`, `rb`, `hi \|= 0x00000600 \| rc` | `SHF.L.U32 Rd, Ra, Rb, Rc` |
| `encode_lop3_i` | `0x7812` | `rd`, `ra`, `imm32`, `hi \|= 0x07800000 \| pd<<17 \| lut<<8 \| rc` | `LOP3.LUT Rd, Ra, imm32, Rc, lut, !PT` |
| `encode_lop3_r` | `0x7212` | `rd`, `ra`, `rb`, `hi \|= 0x07800000 \| pd<<17 \| lut<<8 \| rc` | `LOP3.LUT Rd, Ra, Rb, Rc, lut, !PT` |
| `encode_prmt_i` | `0x7816` | `rd`, `ra`, `sel16` in `lo[47:32]`, `hi[7:0]=rc` | `PRMT Rd, Ra, sel16, Rc` |
| `encode_isetp_r` | `0x720C` | `ra`, `rb`, `hi \|= 0x03F00070 \| pd<<17 \| cmp<<12 \| signed<<9` | `ISETP.<cmp>[.U32].AND Pd, PT, Ra, Rb, PT` |
| `encode_isetp_i` | `0x780C` | `ra`, `imm32`, `hi \|= 0x03F00070 \| pd<<17 \| cmp<<12 \| signed<<9` | `ISETP.<cmp>[.U32].AND Pd, PT, Ra, imm32, PT` |
| `encode_isetp_u` | `0x7C0C` | `ra`, `urb`, `hi \|= 0x0BF00070 \| pd<<17 \| cmp<<12 \| signed<<9` | `ISETP.<cmp>[.U32].AND Pd, PT, Ra, URb, PT` |
| `encode_fsetp_r` | `0x720B` | `ra`, `rb`, `hi \|= 0x03F00000 \| pd<<17 \| cmp<<12` | `FSETP.<cmp>.AND Pd, PT, Ra, Rb, PT` |
| `encode_fsetp_i` | `0x780B` | `ra`, `f32_bits(fval)`, `hi \|= 0x03F00000 \| pd<<17 \| cmp<<12` | `FSETP.<cmp>.AND Pd, PT, Ra, fval, PT` |
| `encode_sel_i` | `0x7807` | `rd`, `ra`, `imm32`, `hi \|= neg_sp<<26 \| sp<<23` | `SEL Rd, Ra, imm32, [!]P<sp>` |
| `encode_sel_r` | `0x7207` | `rd`, `ra`, `rb`, `hi \|= neg_sp<<26 \| sp<<23` | `SEL Rd, Ra, Rb, [!]P<sp>` |
| `encode_i2fp_u32` | `0x7245` | `rd`, `rb`, `hi \|= 0x00201000` | `I2FP.F32.U32 Rd, Rb` |

Compare opcodes (`cmp_op`): `CMP_LT = 1`, `CMP_EQ = 2`, `CMP_LE = 3`,
`CMP_GT = 4`, `CMP_NE = 5`, `CMP_GE = 6`.

### Floating-Point, Special Function, and Precision Conversion

| Encoder | `lo[15:0]` | Key `lo` / `hi` Fields | Verified `sm_121a` Disassembly |
|---|---|---|---|
| `encode_ffma_r` | `0x0223 \| pred<<12` | `rd`, `ra`, `rb`, `hi[7:0]=rc` | `FFMA Rd, Ra, Rb, Rc` |
| `encode_ffma_i` | `0x0823 \| pred<<12` | `rd`, `ra`, `f32_bits(fval)`, `hi[7:0]=rc` | `FFMA Rd, Ra, fval, Rc` |
| `encode_ffma_u` | `0x0C23 \| pred<<12` | `rd`, `ra`, `urb`, `hi \|= 0x08000000 \| rc` | `FFMA Rd, Ra, URb, Rc` |
| `encode_fadd_r` | `0x0221 \| pred<<12` | `rd`, `ra`, `rb`, `lo[63]=neg_b`, `hi[8]=neg_a`, `hi[9]=abs_a` | `FADD Rd, [-][\|]Ra[\|], [-]Rb` |
| `encode_fadd_i` | `0x0421 \| pred<<12` | `rd`, `ra`, `f32_bits(fval)`, `hi[9]=abs_a` | `FADD Rd, [\|]Ra[\|], fval` |
| `encode_fmul_r` | `0x0220 \| pred<<12` | `rd`, `ra`, `rb`, `hi \|= 0x00400000` | `FMUL Rd, Ra, Rb` |
| `encode_fmul_i` | `0x0820 \| pred<<12` | `rd`, `ra`, `f32_bits(fval)`, `hi \|= 0x00400000` | `FMUL Rd, Ra, fval` |
| `encode_fmnmx_r` | `0x7209` | `rd`, `ra`, `rb`, `lo[62]=abs_b`, `hi[9]=abs_a`, `hi \|= 0x07800000` (max) or `0x03800000` (min) | `FMNMX Rd, [\|]Ra[\|], [\|]Rb[\|], [!]PT` |
| `encode_mufu` | `0x7308` | `rd`, `rb`, `hi \|= {EX2: 0x0800, RCP: 0x1000, RSQ: 0x1400}` | `MUFU.{EX2\|RCP\|RSQ} Rd, Rb` |
| `encode_f2f_bf16` | `0x7304` | `rd`, `rb`, `hi \|= 0x00202000` | `F2F.BF16.F32 Rd, Rb` |
| `encode_f2fp_bf16_pack` | `0x723E` | `rd`, `rhi`, `rlo`, `hi \|= 0x000010FF` | `F2FP.BF16.F32.PACK_AB Rd, Rhi, Rlo` |
| `encode_f2fp_e4m3_pack` | `0x723E` | `rd`, `rhi`, `rlo`, `hi \|= 0x048070FF` | `F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C Rd, Rhi, Rlo, RZ` |
| `encode_f2fp_e2m1_unpack` | `0x723E` | `rd`, `ra=0xFF`, `rb`, `hi \|= 0x020402FF` | `F2FP.F16.E2M1.UNPACK_B Rd, Rb` |
| `encode_hadd2_f32_half` | `0x7230` | `rd`, `ra=0xFF`, `rb`, `lo[63:56]=0x20` (`H0_H0`) or `0x30` (`H1_H1`), `hi \|= 0x00004100` | `HADD2.F32 Rd, -RZ, Rb.{H0_H0\|H1_H1}` |

### Warp Shuffle

| Encoder | `lo[15:0]` | Key `lo` / `hi` Fields | Verified `sm_121a` Disassembly |
|---|---|---|---|
| `encode_shfl_bfly` | `0x7F89` | `rd`, `ra`, `lo[44:40]=clamp`, `lo[57:53]=lane_xor`, `lo[59:58]=3`, `hi \|= 0x000E0000` | `SHFL.BFLY PT, Rd, Ra, xor, clamp` |
| `encode_shfl_down` | `0x7F89` | `rd`, `ra`, `lo[44:40]=clamp`, `lo[57:53]=delta`, `lo[59:58]=2`, `hi \|= 0x000E0000` | `SHFL.DOWN PT, Rd, Ra, delta, clamp` |
| `encode_shfl_up` | `0x7F89` | `rd`, `ra`, `lo[44:40]=clamp`, `lo[57:53]=delta`, `lo[59:58]=1`, `hi \|= 0x000E0000` | `SHFL.UP PT, Rd, Ra, delta, clamp` |
| `encode_shfl_idx` | `0x7589` | `rd`, `ra`, `lo[39:32]=lane_reg`, `lo[44:40]=clamp`, `hi \|= 0x000E0000` | `SHFL.IDX PT, Rd, Ra, Rlane, clamp` |

### Global Memory, Shared Memory, and L2 Prefetch

Width bits (`LDG_W_*` / `STG_W_*` in `hi[12:8]`): `U8 = 0x1100`,
`U16 = 0x1500`, `32 = 0x1900`, `64 = 0x1B00`, `128 = 0x1D00`.
Cache policy bits (`hi[31:15]`): `CACHE_DEFAULT = 0x0C1E0000`,
`CACHE_EF = 0x0C0E0000` (evict-first), `CACHE_CONSTANT = 0x0C1E8000`.

| Encoder | `lo[15:0]` | Key `lo` / `hi` Fields | Verified `sm_121a` Disassembly |
|---|---|---|---|
| `encode_ldg` | `0x0981 \| pred<<12` | `rd`, `raddr`, `lo[39:32]=ur_desc`, `lo[63:40]=offset`, `hi \|= cache \| width_bits` | `LDG.E[.<cache>][.<width>] Rd, desc[UR][Raddr.64+off]` |
| `encode_ldg256` | `0x097E \| pred<<12` | `rd_lo`, `raddr`, `ur_desc`, `lo[63:56]=0xFE`, `hi[7:0]=rd_hi`, `hi \|= {ef: 0x08021900, default: 0x08121900, el: 0x08221900}` | `LDG.E[.<pol>].ENL2.256 Rd_hi, Rd_lo, desc[UR][Raddr.64]` |
| `encode_stg` | `0x0986 \| pred<<12` | `raddr`, `rval` in `lo[39:32]`, `lo[63:40]=offset`, `hi \|= 0x0C100000 \| width_bits \| ur_desc` | `STG.E[.<width>] desc[UR][Raddr.64+off], Rval` |
| `encode_stg256` | `0x097F \| pred<<12` | `lo[23:16]=rval_hi`, `raddr`, `lo[39:32]=rval_lo`, `lo[63:56]=0xF8`, `hi \|= 0x0F121800 \| ur_desc` | `STG.E.ENL2.256 desc[UR][Raddr.64], Rval_lo, Rval_hi` |
| `encode_lds` | `0x7984` | `rd`, `raddr`, `lo[39:32]=ur_base`, `lo[63:40]=offset`, `hi \|= 0x08000800` | `LDS Rd, [Raddr+URbase+off]` |
| `encode_lds128` | `0x7984` | `rd`, `raddr`, `lo[39:32]=ur_base`, `lo[63:40]=offset`, `hi \|= 0x08000C00` | `LDS.128 Rd, [Raddr+URbase+off]` |
| `encode_sts` | `0x0388 \| pred<<12` | `raddr`, `rval` in `lo[39:32]`, `lo[63:40]=offset`, `hi \|= 0x00000800` | `STS [Raddr+off], Rval` |
| `encode_ublkpf_l2` | `0x73BC` | `lo[31:24]=ur_addr`, `hi \|= 0x08000000 \| ur_bytes` | `UBLKPF.L2 [URaddr], URbytes` |
| `encode_hmma_16816_f32_bf16` | `0x723C` | `rd`, `ra`, `lo[39:32]=rb`, `hi[7:0]=rc`, `hi \|= 0x00041800` | `HMMA.16816.F32.BF16 Rd, Ra, Rb, Rc` |
| `encode_ldgsts128` | `0x0FAE \| pred<<12` | `lo[23:16]=r_smem`, `lo[31:24]=r_gaddr`, `lo[63:32]=offset`, `hi \|= 0x0B981800 \| ur_desc` | `LDGSTS.E.BYPASS.128 [Rsm], desc[UR][Rgaddr.64+off]` |
| `encode_ldgdepbar` | `0x79AF` | `wr=0` (`SB0`) | `LDGDEPBAR` |
| `encode_depbar_le` | `0x791A` | `lo \|= (0x8000 << 32) \| ((count & 0x3F) << 38)` | `DEPBAR.LE SB0, count` |
| `encode_stg` (`STG_CACHE_STRONG_SYS`) | `0x0986 \| pred<<12` | `cache=0x0C114000`, `width_bits=STG_W_128\|STG_W_32` | `STG.E[.<width>].STRONG.SYS desc[UR][Raddr.64+off], Rval` |
| `encode_membar` | `0x7992` | `hi \|= {SC_SYS: 0x3000, ALL_SYS: 0xB000}` | `MEMBAR.{SC\|ALL}.SYS` |

## 4. CUBIN ELF64 Container Format (`skl2/compiler/sklc/cubin_elf.py`)

`build_cubin` writes an ELF64 executable (`ET_EXEC = 2`, `EM_CUDA = 190`,
`EI_OSABI_CUDA = 0x41`, `EI_ABIVERSION_BLACKWELL = 8`):

- `e_flags`: `0x06007902` for `sm121`/`sm121a` (`EF_CUDA_SM121`), `0x06007802`
  for `sm120`/`sm120a` (`EF_CUDA_SM120`).
- Note sections: `.note.nv.tkinfo` (type `2000`, producer metadata) and
  `.note.nv.cuinfo` (type `1000`, SM number and toolkit version `130`), plus
  `.nv.compat` (`SHT_CUDA_COMPAT = 0x70000086`).
- Per-kernel sections: `.text.<k>` (128-byte aligned; `sh_info` packs
  `func_sym | ((barriers & 0x7F) << 20)`), optional `.nv.shared.<k>`
  (`SHT_NOBITS`), `.nv.constant0.<k>` (`C0_PARAM_BASE + param_bytes` bytes),
  and `.nv.info.<k>` (`SHT_CUDA_INFO = 0x70000000`).
- Global `.nv.info` records register count (`0x2F`), frame size (`0x11`), and
  stack size (`0x12`) per kernel symbol. Per-kernel `.nv.info.<k>` records
  ABI version (`0x37`), parameter descriptors (`0x17`, reverse ordinal order),
  exit instruction offsets (`0x1C`), `MAX_THREADS` (`0x05`), parameter size
  (`0x19`), and constant-bank parameter window (`0x0A` at `C0_PARAM_BASE`).
- Code epilogue: `KernelImage.code_bytes()` appends a self-looping
  `BRA -16` (`stall=0, yield_=True`) and pads to a 128-byte boundary with
  `NOP` (`stall=0, yield_=True`), matching `ptxas` output.

## 5. `cuobjdump` / `nvdisasm` Verification Receipt (`sm_121a`)

Verified on GB10 (`sm_121a`, CUDA 13.0 `cuobjdump` and `nvdisasm`) across all
6 `.skl` source files (13 kernels, 3,113 emitted instructions) and the 84-case
synthetic encoder coverage suite covering every encoder in
`skl2/compiler/sklc/isa_sm121.py`:

| Source | Kernels | Instructions | `cuobjdump --dump-sass` | `nvdisasm` | Unknown / Illegal Opcodes |
|---|---:|---:|---|---|---:|
| `inference/kernels/skl/attn_decode.skl` | 1 | 436 | PASS | PASS | 0 |
| `inference/kernels/skl/moe_route.skl` | 2 | 216 | PASS | PASS | 0 |
| `inference/kernels/skl/mx_gemm.skl` | 1 | 76 | PASS | PASS | 0 |
| `inference/kernels/skl/norm_act.skl` | 3 | 585 | PASS | PASS | 0 |
| `inference/kernels/skl/skinny_decode.skl` | 3 | 1,702 | PASS | PASS | 0 |
| `inference/kernels/skl/tp_mesh_mem.skl` | 3 | 98 | PASS | PASS | 0 |
| Exhaustive `isa_sm121.py` encoder suite | 1 | 84 | PASS | PASS | 0 |
| `skl2/` Slice 1 (`allreduce`, `linear`, `mma_gemm`, `norm_rope`) | 53 | 15,856 | PASS | PASS | 0 |
| `skl2/` Full Slice 2 (all 91 kernels, 0 spills, 100% `.skl` v1 retired) | 91 | 24,520 | PASS | PASS | 0 |
| **Total (active `skl2` + exhaustive encoder suite)** | **92** | **24,604** | **PASS** | **PASS** | **0** |

To reproduce disassembly verification on any host with CUDA 13.0 tools:

```sh
python3 skl2/compiler/skl2c/skl2c.py --arch sm121a skl2/stdlib/allreduce.skl2 skl2/kernels/*.skl2 -o /tmp/skl2_all.cubin
cuobjdump --dump-sass /tmp/skl2_all.cubin | grep -iE 'unknown|illegal|invalid|\?\?\?' && exit 1
nvdisasm /tmp/skl2_all.cubin | grep -iE 'unknown|illegal|invalid|\?\?\?' && exit 1
```
