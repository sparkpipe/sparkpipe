# Lane brief: Qwen Max sharded-pack sprint (TP4×PP4 + MXFP4)

Worktree: /tmp/lane-qwen38max-shard (branch lane/qwen38max-shard)
Build nodes: spark7 (module compile, GPU validation) and spark5 (pack
building — check ceph health first, fallback to spark0)
Deploy: all 16 sparks after validation passes

## Mission
Implement true per-rank sharded packs with MXFP4 expert quantization for
Qwen Max (2.4T FP8 source). This unblocks the model on our 16-spark
fleet: 73.8 GiB payload per rank at MXFP4 (vs 147.6 at FP8, which doesn't
fit 119 GB nodes).

## The three module changes (from the pack agent's F3 analysis)

### 1. Pack wire format v2 (~60-80 lines)
- Add tp_degree + tp_rank to the 120B header (→128B, FORMAT_VERSION 2)
- Files: stagepack_format.h, packer, verifier, synthesizer

### 2. Sharded loader (~150-250 lines)
- Shape table gains TP axis:
  - Expert-sharded: MOE_W1/W3/DOWN rows/4, scale planes slice clean
  - Head-sharded: GDN gate/beta/decay/A_log/dt_bias/norms, ATTN q/k/v
  - Input-dim sharded: GDN_OUTPUT, ATTN_OUTPUT cols/4
  - Composed: GDN_QKV+CONV rows (q512|k512|v4096 per rank)
  - Replicated: router, shared expert, embed, LM head, MTP globals
- ValidateEntry: accept sharded shapes + MXFP4 experts (currently BF16-only)
- Natural-format exception: extend to MXFP4

### 3. Kernel TP-aware indexing (~300-500 lines, one .cu file)
- AttnPrepare/Decode: already TP-aware, drop global head base (~30-60 lines)
- Grouped expert launchers: base-offset 0 + MXFP4 path (SparkLmDotRowMxfp4<32>
  and mma.cuh LM_MMA4_MXFP4_GROUP already in-tree — just ungated from FP8-only)
- GdnStep/GdnChunk head loops + state pool /4 (~100-200 lines)
- Possibly a second all-reduce (~50 lines)

### 4. MXFP4 requantizer (~100-150 lines)
- numpy vectorized FP8(block-128×128 scale_inv) → MXFP4(group-32 E8M0)
- ~1-2h I/O pass for the full model
- Quality gate: kernel-cosine + decode parity vs FP8 before serving

### 5. Validation harness (NEW — qwen38_max has none)
- Port from qwen38_27b's validation/ (the 27b harness is the template)
- Must include: kernel oracle checks, module admit/snapshot, decode-vs-prefill
  bit-exact, determinism, AND the MXFP4-vs-FP8 parity gate
- This is also the prerequisite for ANY future qwen38_max GPU work

## Milestones
S1: Validation harness port from 27b (get `make validate` working)
S2: Wire format v2 (header, packer, verifier, synthesizer)
S3: Loader sharded shape table + MXFP4 codec acceptance
S4: Kernel TP-aware indexing (attn first, then GDN, then experts)
S5: MXFP4 requantizer + pack build from the cold FP8 source
S6: Parity gate: MXFP4 decode vs FP8 reference (cosine > 0.999, bit-exact
    determinism, in-vocab drafts)
S7: Build all 16 rank packs, verify, deploy, smoke test

## Scope
modules/qwen38_max_resident_decode_stage/ (all files), tools/ (packer,
verifier), model-families/qwen38_max/, tests/.
Do NOT touch: other families' modules, runtime/, node/, include/ (shared
headers — use integration requests).

## Quality policy (user-approved)
Variable expert bit widths (4/6/8) are part of the platform plan.
MXFP4 experts at group-32 is the standard industry practice. The parity
gate (S6) is the safeguard — if cosine drops below threshold, we fall
back to 6-bit MXFP6 (1.7 TB / 16 = 106 GB/rank, still fits).
