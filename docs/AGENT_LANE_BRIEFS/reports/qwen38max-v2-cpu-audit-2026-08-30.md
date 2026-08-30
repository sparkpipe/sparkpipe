# Qwen 3.8 Max stagepack v2 — CPU wire-contract audit (2026-08-30)

Lane: qwen38max (model dev). Branch: `lane/qwen38max-v2-audit` (this PR).
Everything below ran on the build host CPU (clang + python3): the stagepack
wire contract is fully checkable without a spark, and a layout drift must be
a red stop BEFORE any 16-rank, ~80 GiB-per-rank build is dispatched.

## Verdict

**INACCURATE — the lane's format-v2 artifacts disagree with each other in
two places.** Both were found by CPU checks, both are proven on a real
17.6 GiB C-written pack, and both are fixed by the alignment this PR pins
(the fix is verified end-to-end on CPU in this report).

## F1 — header field order: the v2 python packer and the (rebased) C header
##       disagree on where the TP fields sit

- Lane python packer/verifier (branch tip `lane/qwen38max-shard`, commit
  ca577c4): `HEADER_STRUCT = struct.Struct("<28I2Q")` — tp_degree@104,
  tp_rank@108, directory_offset@112, file_bytes@120.
- C family header rebased onto main's shared stagepack format (which pins a
  120-byte `SparkStagePackHeaderCommon` prefix via offsetof proofs):
  directory_offset@104, file_bytes@112, tp_degree@120 (tail), tp_rank@124.
- Both claim FORMAT_VERSION 2 and HEADER_BYTES 128.

Receipt (tools/qwen38_stagepack_layout_audit.py, this PR):

```
== AUDIT: rebased lane C header vs lane python packer (as built)
   C format_version=2 fields=30; python layout=v2-as-built fields=30
   VERDICT: INACCURATE -- 4 divergent field(s):
     tp_degree: C C@120 size 4 vs python py@104 size 4
     tp_rank: C C@124 size 4 vs python py@108 size 4
     directory_offset: C C@104 size 8 vs python py@112 size 8
     file_bytes: C C@112 size 8 vs python py@120 size 8
```

Proof on real bytes: the C synthesizer (rebased tree) wrote a 2-layer
tp2/rank1 pack (39 tensors, 17.6 GiB, mxfp4 experts). Decoding its header
both ways:

```
== decoded as v2-as-built (branch python packer)
   magic=1347631185  format_version=2  header_bytes=128  tensor_count=39
   tp_degree=128  tp_rank=0
   directory_offset=18853662720   <- == the FILE SIZE, i.e. garbage
   file_bytes=4294967298          <- garbage
== decoded as v2-tail (rebased C header)
   tp_degree=2  tp_rank=1  directory_offset=128  file_bytes=18853662720  (exact)
```

The insidious part: magic, format_version and header_bytes decode
IDENTICALLY in both layouts — every cheap sanity check passes and the
directory walk then runs off the tp bytes. This is exactly the
silent-format-drift class the shared stagepack format library was created
to prevent ("stated once"), which is why the rebased C side is the one to
keep: the 120-byte common prefix must stay layout-identical to
`SparkStagePackHeaderCommon` or the shared comparison misreads the family
header.

## F2 — weight-format code: MXFP4_E2M1 is 3 in the C firmware ABI, 7 in the
##       lane python

```
C  firmware header: WEIGHT_FORMAT_MXFP4_E2M1 3u   (BF16 0, F32 1, U32 2, FP8 4)
py lane packer:     WEIGHT_MXFP4_E2M1 = 7
shared format lib:  SPARK_STAGEPACK_FORMAT_WEIGHT_I64 7u   <- collision
```

The lane's own 08-28 round trip missed this because it was python-packer →
python-verifier on FP8-source data (code 4, same on both sides); the MXFP4
code never crossed the language line until now. Symptom on the real pack:

```
FAIL q38_rt_l0r1.qwen38sp: entry 4 kind=6 format 3 unexpected (natural 7)
```

Code 7 on the wire would be read by the C loader as the shared table's
WEIGHT_I64. The C firmware value (3) is the module's law; python must use 3.

## The fix, verified end-to-end on CPU

With the two alignments applied to the python pair (HEADER_STRUCT
"<26I2Q2I", pack/unpack tuples reordered, WEIGHT_MXFP4_E2M1 = 3):

```
$ cc -std=c11 -O2 <includes> -o q38_synth <rebased>/tools/qwen38_max_pack_synthesize.c \
      runtime/stagepack_format.c
$ ./q38_synth --output q38_rt_l0r1.qwen38sp --first-layer 0 --layer-count 2 \
              --tp-degree 2 --tp-rank 1
qwen38_pack_synthesize slice=0+2 tensors=39 mxfp4_bytes=13690208256 \
dense_bytes=5163451904 file_bytes=18853662720 file_gib=17.6
$ python3 qwen38_pack_verify.py --pack q38_rt_l0r1.qwen38sp --tp-degree 2 --tp-rank 1
PASS q38_rt_l0r1.qwen38sp: header geometry (tp 2/1), 39 directory entries,
coverage exact [structure-only, no checkpoint]
```

Harness sanity (it detects drift, not phantoms): main's current v1 C header
vs main's v1 python layout — VERDICT: accurate, all 28 fields agree.

## What this PR contains

- `tools/qwen38_stagepack_layout_probe.c` — offsetof table for the family
  header (adapts to FORMAT_VERSION; v2 prints the TP fields wherever the
  family put them).
- `tools/qwen38_stagepack_layout_audit.py` — compiles/compares the probe
  against a python layout; exit 1 with the divergent fields named. The three
  layouts (v1, v2-as-built, v2-tail) live here as the single statement of
  the python-side contract.
- This report.

## Asks (coordinator)

1. **Bless the v2-tail wire order as the family contract** (120-byte common
   prefix + 8-byte tp tail, HEADER_BYTES 128): it is the only order that
   keeps the shared `SparkStagePackHeaderMatches` cast valid. The lane's
   python packer/verifier will carry the fix when lane/qwen38max-shard
   lands (rebased on current main; the S7 16-rank build is HELD until then).
2. **State the MXFP4_E2M1 code in the shared codec table** (value 3 per the
   family firmware ABI; the lane python's 7 collides with WEIGHT_I64 and is
   dead). Same single-source-of-truth rule as the header.
3. **Shared synth core / universal packer TP support**: main's
   spark_pack_synthesize_common.h cannot emit sharded packs (no tp args),
   and the qwen38_max family synthesizer in this lane stays standalone
   until the core grows an optional TP hook (inert for other families).
4. **Architecture decision recorded in passing**: main's qwen38 packer
   states full-width packs per TP rank with runtime slicing ("rank-local
   shards would fail pack_entry_invalid"). At MXFP4 that is ~295 GiB per
   rank for this model (4 x 73.8 GiB) — infeasible on the 119 GB nodes;
   the rank-local sharded path (this lane) is the one that fits. The two
   designs need one owner and one loader.
