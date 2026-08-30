# qwen38max stagepack CPU accuracy verification — 2026-08-30

Lane: Qwen 3.8 Max (4-bit). Question from operator: verify via CPU
whether the stagepack is accurate; if not, PR the identified problem.

## Setup

- Checkpoint download completed: 213 safetensors, 1.3T, no partials at
  /mnt/model-warm/packbuild/qwen38max/amd-mxfp4 (spark0). No rank packs
  had been built yet — S7 never ran.
- Built a real pack on CPU (packer is pure CPU/IO) from the AMD source:
  `qwen38_stagepack.py --first-layer 0 --layer-count 1 --tp-degree 4
  --tp-rank 1 --source-format quark-mxfp4` → 7.28 GiB, 20 tensors.
  This is the FIRST quark-mxfp4-source pack ever built (the sprint's
  byte-exact round-trip used the vendor FP8 source).

## Verdict: the pack is accurate; the verifier's dequant probe was not

- 20/20 byte-traced samples receipt=verified — every sampled entry
  (embedding, GDN, attention, shared expert, 128-expert MXFP4 payloads,
  MTP) recomputes byte-exactly through the packer's copy plans against
  the checkpoint.
- Initial run FAILED `mxfp4 dequant sanity`: nonzero=1/256 at element 0.
  Investigation: the checkpoint's expert payloads legitimately begin
  with runs of exact-zero E2M1 elements (leading zero bytes measured on
  layer-0: expert0 gate 0, expert128 gate 91, expert200 gate/up 2,722).
  Overall nonzero fraction is healthy (13.6%–57% across sampled
  experts). The probe decoded a single 256-element window at payload
  offset 0 — a false-FAIL on healthy packs.
- Why it matters: tools/qwen38max_build_ranks.sh runs this verifier per
  rank; the false-FAIL would abort the S7 16-rank build.

## Fix (in this PR)

Probe now decodes 8 windows spread across the payload (2048 elements)
and requires >=1/8 nonzero. Re-run on the real pack:
`finite=2048/2048 nonzero=448/2048 → PASS`.

## Reproduce

spark0, repo ~/sparkpipe-lane (branch verify/qwen38max-pack):

```sh
python3 tools/qwen38_stagepack.py --checkpoint /mnt/model-warm/packbuild/qwen38max/amd-mxfp4 \
  --output .../verify_l0_tp4r1.qwen38sp --first-layer 0 --layer-count 1 \
  --tp-degree 4 --tp-rank 1 --source-format quark-mxfp4
python3 tools/qwen38_pack_verify.py --pack .../verify_l0_tp4r1.qwen38sp \
  --checkpoint .../amd-mxfp4 --source-format quark-mxfp4 --tp-degree 4 --tp-rank 1 --sample 20
```

Note: this branch is lane/qwen38max-shard + the probe fix — the v2
packer/verifier (tp_degree/quark-mxfp4) do not exist on main, so the fix
rides the shard sprint PR. S7 (16-rank build) is unblocked by this fix
and by the completed download; queue a window when stagepacks go live.
