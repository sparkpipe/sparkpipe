# Pack-emission wave 1 — the 9 unplaced warm quants — 2026-09-19

Mission: close the warm-completeness gap. Nine warm quants had no stagepack
on NVMe. Each arm was inventoried (config + tensor census), probed against
its family packer on main, and — where the packer accepts the source —
emitted, placed, and verified. CPU only; emission read warm
(`/mnt/model-warm`); packs landed on node-local NVMe under
`~/sparkdata/<arm>/packs/`. Placement law: rank r → spark r (hex), replica →
spark r+8; no existing pack was overwritten (staged builds, exclusive
links, `PLACE-SKIP` guards).

Source identity pins (sha256 of `model.safetensors.index.json` / `config.json`
unless noted; measured live on the warm mount this wave):

| warm source | index sha256 | config sha256 |
|---|---|---|
| gemma-4-31b-it | d4aff3b976d69c123a29d1c085d7ba4de1ac3f4ca1726a7f81e1b11462a64ea2 | e967dd38bc5cfd38bd09a995a7bf4a754075df2b46aba68f7fbb5a791e6d8dd1 |
| gemma-4-31b-it-nvfp4 | aff5569bed013db014afe23db3986434729bb33d9f817017cc0617cd479212b9 | aa03a6a490fb743b8186f09c60ec39a30fdbaf7c3a18dfef8e52a62c2beb9ae4 |
| deepseek-v4.1-flash-nvfp4-nvidia | 050c4552828f5a7160823fab529d0fa7a8c02651d1e8c9d85eed20017357faa0 | 6f53b8f185a587f8e4b9ebd859c158ba92459cc119d11243e0c8291a02a0e265 |
| ling-3.0-flash-fp8 | c5b0212d2f4af850363f3b0a1ce76c5b5809c4d4a3a1f9f20db9c57e6e72fea6 | 621969ccadf2cf1f5575f82baeadccc841221922ba0dd265b6da329781738b74 |
| ling-3.0-flash-fin | 19b4f2f2e199c7e34bab049c1fc1d99b1661700a669a9ea7e7fabcb4527c5920 | f3e5b1ad82762c7910a303a3642290f0ac4d3abbc9046bf2eab6b19c0385aa05 |
| laguna-s-2.1-fp8 | aec4ef10244640b4a60b4c74cddc3c08399acef547e1f6f973f6381b4745ebb7 | 876de1e4a6c8baa234e414c4129a197d2b3dfa34476447ceafb266bebd236376 |
| laguna-s-2.1-nvfp4 | d21d362ea5b85abc642a721b208834fe1d66a07dfa1c43a70f28a9932f568902 | 3b3a6e369129eaf628e6b96e417ce06d631d61856ba1d2e8eecdf31dfb8dfe01 |
| qwen3.8-27b-fp8 | f0838c766951bdfe76d6afbdb2771a8f67aaa2231dedb3d33cebd817729843a2 | 74227dd615bf1ea975aa676bdf355a0379858c12f394b5365cd9dfa5fc2c70bc |

ling-3.0-flash-fin's pins match `model-families/ling/name_map_lingfin.json`
exactly (index `19b4f2f2…`, config `f3e5b1ad…`) — the source is the pinned
revision.

## Verdicts (grading is MEASURED)

| # | warm quant | verdict |
|---|---|---|
| 1 | minimax-h3 | NO-PACKER |
| 2 | gemma-4-31b-it (bf16) | EMITTED+PLACED+VERIFIED |
| 3 | gemma-4-31b-it-nvfp4 | SOURCE-MISMATCH |
| 4 | deepseek-v4.1-flash-nvfp4-nvidia | SOURCE-MISMATCH |
| 5 | ling-3.0-flash-fp8 | FAILED (packer is bf16-only) |
| 6 | ling-3.0-flash-fin | EMITTED+PLACED+VERIFIED |
| 7 | laguna-s-2.1-fp8 | SOURCE-MISMATCH |
| 8 | laguna-s-2.1-nvfp4 | SOURCE-MISMATCH |
| 9 | qwen3.8-27b-fp8 | EMITTED+PLACED+VERIFIED |

## Arm 1 — minimax-h3: NO-PACKER

No family packer exists on main: `grep tools/ minimax` returns nothing; no
`model-families/minimax*` tree; the only mentions are design-doc product
targets (`docs/MODEL_SUPPORT.md`, `ARCHITECTURE.md`, `TECHDEBT.md`) and the
DESIGN family-id table. The warm tree is a multimodal monorepo
(`model_index.json`, `audio_scheduler/`, `audio_vae/`, `FL2VA/`, `processor/`)
with zero top-level `.safetensors` — ingest scripts referenced by lane
history are not in the tree. Per wave orders no packer was written.
NOT PLACED (465 GiB warm source).

## Arm 2 — gemma-4-31b-it (bf16): EMITTED+PLACED+VERIFIED

- Packer: `tools/gemma4_stagepack.py --model 31b` (TP16×PP1, the only 31b
  topology the packer defines; per-rank census 723 tensors / 60 layers).
- Arm: `gemma4_31b.bf16.tp16`; file
  `gemma4_31b_tp16_rank<h>_stage0.gemma4sp` + `.sha256` +
  `.receipt.json` + `.verify-receipt.json`.
- Topology/placement: 16 ranks, rank r → spark r (no replicas; tp16 covers
  the fleet).
- Per-rank emit is a two-pass proof (plan sha, then byte-exact verify walk);
  `--verify-existing` re-proved the staged pack before placement
  (`proof=True, tensors=723`). Packs chattr-locked.
- Receipt fills from the placement matrix below.

## Arm 3 — gemma-4-31b-it-nvfp4: SOURCE-MISMATCH

`tools/gemma4_stagepack.py` fails closed on the pre-quantized checkpoint:

```
SPARK_FAIL gemma4_stagepack: model.language_model.layers.0.mlp.gate_proj.weight:
dtype U8, expected BF16 (never quantize)
```

Census: 1728 tensors = 1008 BF16 + 360 F32 + 180 F8_E4M3 + 180 U8 (nvfp4
packed weights + per-16 E4M3 scales + `hf_quant_config.json`). The 31b pack
geometry and the gemma4 module carry BF16 spine only; an nvfp4 arm needs a
packer codec + module wire work, not a flag. NOT PLACED (31 GiB warm source).

## Arm 4 — deepseek-v4.1-flash-nvfp4-nvidia: SOURCE-MISMATCH

`tools/dsv4_stagepack.py` (the dsv4 flash/pro family packer; the placed
`dsv41flash.mxfp4.tp8` arm's family) fails immediately:

```
dsv4_stagepack: reduced source index tensor set does not match the stage
```

The packer's `validate_meta` pins expert stacks to native I8-packed FP4 with
F8_E8M0 scales (`expected I8 <rows, cols/2>` / `expected F8_E8M0`); the
nvidia modelopt release (`producer: modelopt dsv4-nvfp4-experts`,
`quant_algo: NVFP4, group_size 16` on `layers.N.ffn.experts` only) ships U8
packed e2m1 + F8_E4M3 group-16 scales + F8_E8M0 globals — a different codec
and scale layout. Census 188245 tensors / 520.7 GiB. NOT PLACED.

## Arm 5 — ling-3.0-flash-fp8: FAILED (packer is bf16-only)

`tools/ling_stagepack.py` rejects the source at the pinned census:

```
FAIL index census 127115 != pinned 63783; re-pin name_map against this revision
```

The packer is the official-BF16-release tool by contract: it refuses
`quantization_config` sources, accepts only BF16/F32 spine dtypes, and the
expert pass-through is wired for `CODEC_BF16` only ("only the bf16 expert
pass-through arm is wired"). The fp8 release carries 63332 F8_E4M3 expert
tensors + F32 `_scale_inv` planes; the ling module's pack-format table
(`spark_ling_stagepack_format.h`) likewise pins BF16 payloads for every
kind. An fp8 arm is a module+packer codec lane, not a tool flag. NOT PLACED
(120 GiB warm source).

## Arm 6 — ling-3.0-flash-fin: EMITTED+PLACED+VERIFIED

- Packer: `tools/ling_stagepack.py --model lingfin --tp-degree 16`
  (the proven bf16 flow; `name_map_lingfin.json` census 63783 matches the
  warm source bit-for-bit at the pinned index/config shas).
- Arm: `lingfin.bf16.tp16`; file `lingfin.bf16.tp16.rank<h>.sp` +
  `.experts` (weightd manifest, 40960 ranges) + `.sha256` +
  `receipts/rank<N>.json`.
- Per-rank: 730 tensors, 15,725,069,824 B (14.65 GiB), census
  63783 = 62230 packed + 1553 omitted MTP.
- MTP sidecar law: the main packs are MTP-FREE (the packer omits
  `model.layers.42.*`). The warm MTP file
  `model-mtp-00001-of-00001.safetensors` (6,144,582,792 B, sha256
  `e6d6ce5b0cbb785739f6baa6fee414443bcc81eeb2c17e667f2e4b3b98097b83`) is
  placed as a SEPARATE sidecar under
  `~/sparkdata/lingfin.bf16.tp16/mtp/` with its own `.sha256` — never into
  the main pack. (A per-rank MTP sidecar generator remains the separate
  MTP-variant program; the warm file is the family's MTP source artifact.)
- Placement: rank r → spark r, 16/16.

## Arms 7+8 — laguna-s-2.1-fp8 / nvfp4: SOURCE-MISMATCH

`tools/laguna_stagepack.py` fails closed at the census lock (23 locked
patterns / 36769 tensors = the bf16 release):

```
arm 7: FAIL unknown checkpoint tensor (census lock):
       model.layers.1.mlp.experts.0.down_proj.weight_scale_inv
arm 8: FAIL unknown checkpoint tensor (census lock):
       model.layers.1.mlp.experts.0.down_proj.input_global_scale
```

The packer's `SourceReader` already dequantizes fp8 spines and carries
latent nvfp4 helpers, but `add_experts` requires native BF16 expert tensors
and the census is pinned to the bf16 release; the fp8/nvfp4 expert paths
are not wired into the plan. NOT PLACED (123 GiB + 93 GiB warm sources).

## Arm 9 — qwen3.8-27b-fp8: EMITTED+PLACED+VERIFIED

- Packer: `tools/qwen38_27b_stagepack.py` (its official `-fp8` release
  reader: F8_E4M3 ffn weights pass through on the FP8 wire with
  `_scale_inv` [128,128] f32 block scales; projections stay bf16).
  Dry-run: `slice=0+64 tensors=866 file_gib=9.91` at TP4.
- MTP sidecar law: the fresh pack carries the MTP chain mid-file, so each
  rank pack was compact-stripped MTP-free with
  `tools/stagepack_mtp_strip.py --family qwen36sp --compact`
  (10,645,053,184 → 10,524,421,440 B; header `mtp_layer_count=0`,
  `tp_degree=4`, `tp_rank` intact — see tool fix below), then
  sha-sidecarred and placed. Placement re-check asserts the stripped
  header fields.
- Arms/topology (family has TP4 as TPmax; TP4×PP4 second arm per the
  placement law):
  - `qwen27b.fp8.tp4` — rank r → spark r (primary) + replica rank r →
    spark r+8. Files `qwen27b.fp8.tp4.rank<r>.qwen36sp` +
    `.receipt.json` + `.sha256`.
  - `qwen27b.fp8.tp4pp4` — world rank w = node index, pp_stage = w//4,
    tp_rank = w%4, stage slice 16 layers from w//4*16 (the
    `qwen38_27b_tp4pp4_stagepacks.py` plan). Non-head stages carry no MTP
    (strip NORMALIZE zeroed the header field); head stages compact-strip.
    Files `qwen27b.fp8.tp4pp4.rank<w>.spstage`.

## Tool fixes shipped in this wave

1. `tools/stagepack_mtp_strip.py` — `FAMILIES["qwen36sp"]["mtp_index"]`
   was 25; the qwen38_27b header has `mtp_layer_count` at u32[23] with
   tp_degree at 24 and tp_rank at 25 (the qwen4_flash v1 header ends
   `…vocab, mxfp4_group, mtp` at 25 — the two layouts are NOT the same).
   With the old index, the 2026-09-05 16-wide strip of `qwen27b.tp4`
   zeroed tp_rank on every rank ≠ 0 pack and left `mtp_layer_count=1`.
   Measured fleet state this wave: every node holds all four ranks of
   `qwen27b.tp4`, and ALL 48 rank≠0 packs
   (`qwen27b.tp4.rank{1,2,3}.qwen36sp` × 16 nodes) read
   `mtp=1 tpdeg=4 tprank=0` — they declare rank 0 and claim an MTP layer
   they no longer carry. Example (spark1, rank1): header
   `mtp(23)=1 tpdeg(24)=4 tprank(25)=0`; the unstripped
   `qwen38_27b.tp4pp4.rank01.spstage` correctly shows `1,4,1`. Those
   packs were NOT modified by this wave (never-overwrite law) — they need
   a coordinator-approved header repair (`u32[23]=0`, `u32[25]=rank`,
   re-sha, re-receipt). The fix makes every future strip patch the right
   field.
2. `tools/stagepack_mtp_strip.py` — `chattr`/`lsattr` helpers now treat a
   missing binary as "not locked" instead of crashing (hosts without
   chattr took a traceback through the fail-closed path).
3. `tests/test_stagepack_mtp_strip_qwen36sp.py` — pins the qwen36sp family
   map against the qwen38_27b packer's real header layout and exercises a
   full compact strip on a synthetic pack (MTP tail dropped,
   `tensor_count`/`mtp_layer_count` patched, `tp_degree`/`tp_rank`
   byte-identical, receipt + sha rewritten). Catches the corruption class
   end-to-end.

## Placement map (measured after placement)

Filled from the per-node verify sweep — see the matrix below.

## Disk impact

| arm | per-node bytes | nodes |
|---|---|---|
| gemma4_31b.bf16.tp16 | 3,881,009,408 | 16 |
| lingfin.bf16.tp16 | 15,725,069,824 + 1,966,096 | 16 |
| lingfin MTP sidecar | 6,144,582,792 | 16 |
| qwen27b.fp8.tp4 | 10,524,421,440 × 8 packs | 8 |
| qwen27b.fp8.tp4pp4 | ~3,4-4,4 GB per node (stage-dependent) | 16 |

Every heavy write node ran `sync` + `drop_caches` (sudo -n) after
placement; packs are chattr-locked.
