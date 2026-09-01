# DSV4 Flash TP16 stagepack verification + placement — 2026-09-01

Lane: `lane/dsv4flash-dev`. Operator directive: verify the TP16 stagepack
for accuracy via CPU first; if inaccurate, PR the problem. Result:
**ACCURATE — 16/16 ranks byte-verified against the source full pack, and
16/16 placed fleet-wide with sha receipts.**

## What was verified

Source: `spark5:/home/spark5/lane-dsv4bisect/packs/dsv4_flash_v4_full.spstage`
(166,918,150,256 B, 1409 tensors, sha256 `56a07b2d92e5e26d…` — re-hashed
2026-08-30, matches the bisect receipt; this is the pack family whose TP4
siblings produced the hardware-canonical O128 exact hash `211462f2…`).

Sharding: `tools/dsv4_tp16_stagepack.py --tp-degree 16` per rank from that
full pack (rank 0 built first, ranks 1–15 in 3-way batches).

Independent verifier: `tools/dsv4flash_tp16_verify.py` (written for this
check; re-implements the documented shard semantics — row/column slicing
per kind, FP8 128-block and FP4 32-block scale planes, replicated
MTP/DSpark draft set — WITHOUT importing the packer's copy code). Every
output tensor's payload AND scale plane is byte-compared row-by-row
against the source.

**Result per rank: 1409 tensors, 1937 byte checks, 0 mismatches.**
(768 tensors replicate full-width: the DSpark/MTP draft stack.)

Two verifier bugs were found and fixed during the sweep — neither was a
pack defect: (1) GLOBAL_LAYER draft kinds were confused with the
`0xFFFFFFFB..FD` draft-layer markers; (2) replicated draft tensors on
ranks ≠ 0 were wrongly column-sliced (their `byte_start` must be 0). The
false mismatches these produced (ranks 1–3, kinds ATTN_SINK/WO_A/WO_B/W2
on draft layers) vanished after the fixes; all 16 ranks re-verified clean.

Built-in structural verify (`--verify-output` semantics) additionally
passed on every rank (header fields, directory order, bounds, key-set
identity vs plan).

## Placement (coordinator convention)

The coordinator collected the 16 shard files into
`/home/<node>/sparkdata/dsv4flash.tp16/packs/dsv4flash.tp16.rank<r>.spstage`
with node i holding rank (i−1) mod 16 (spark0→rank0 … sparkf→rank14,
spark1→rank15). I audited all 16 nodes by sha256: **every placed copy
matches the byte-verified build** (rank0 `6dab3a03fef6…`, rank1
`c4976dd20e15…`, rank2 `f96b91c392b2…`, rank3 `b8ee338db245…`, rank4
`7d1697d232e8…`, rank5 `9e7a96af02b5…`, rank6 `68c55e395c2c…`, rank7
`bb9fe41fec58…`, rank8 `69cf8e03b2a1…`, rank9 `a3b0bef071e7…`, rank10
`60d337a3610a…`, rank11 `13301e4b79dc…`, rank12 `33e4f797ff30…`, rank13
`3fc29f0c9e49…`, rank14 `97f56f6c3e7f…`, rank15 `0ee5f902f436…`).

Sizes: rank0/1 22,107,554,848 B; ranks 2–15 22,106,506,272 B — matches
(167G−11G sharded)/16 + 11G replicated-draft arithmetic.

## Module-side defect found on the way (fixed, PR #756 MERGED)

`sparkpipe_model_compile` failed at link on clean main because the dsv4
module's `MODULE_ADDITIONAL_HOST_SOURCES` never grew the W3 weightd tail
(+status/sha256/admission). Reproduced three times in-queue; the fix
(self-contained module link tail) is PR #756, merged 2026-08-30.

## Cell state (next step)

`dsv4flash-tp16-cell1` queued (priority 5, 16 nodes, ttl 50): publishes
the b1 module (GPU validator on the v4 3-layer slice), compiles the
driver from main tip `6d84440`, stages the NCCL-backed 16-rank runtime
(configs generated to the physical rank→node mapping, nccl listen ports
64640+r, control 18500+r, KV 32/32 pages, graphs=0 for first bring-up),
launches residentd ×16, and runs the O128 B1 cell 3× with the exactness
gate BEFORE timing — expected token vector = the canonical
`211462f2…` stream (re-derived from the in-repo lean receipt, hash
re-confirmed). Mismatch = RED stop. Non-spec decode tok/s is the
deliverable; TP4 baseline 40.19–40.46.
