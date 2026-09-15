# GOALS.md — Q3F-T1

## Fleet goals G1–G6 (shared, short form)

- G1 fast+correct ≥14 tok/s
- G2 accuracy binding
- G3 production legality
- G4 fleet stability
- G5 multi-dev common fleet
- G6 the architecture runway

## Driver goals

- D1. Driver correctness: T1 gate for qwen3flash — reference-engine port +
  fixtures (first-ever for this driver), then decode both fixture prompts
  through the real driver on the real fp8.tp8 packs via the SHARED weightsd;
  exact token ids + expert routing, in-band streams (defaults; EXACT rates
  reported separately); receipt = raw decode logs + comparison output +
  fixture/node SHAs + engine sha.
- D2. Driver performance: ONE measured B1 tok/s graded MEASURED, placed next
  to the 105.5 tok/s DERIVED ceiling; then STOP (operator directive).
- D3. Driver accuracy: stamp-semantics verification of the freshly placed
  fp8.tp8 packs BEFORE decode (the qwen38max nvfp4-stamped-FP8 defect class);
  on divergence report FIRST DIVERGENCE, no loosening.
- D4. (secondary, standing) Common-code contribution: adopt the donor T1
  pattern for qwen3flash; propose upstream anything generic written
  family-locally.

## Significance vector

| Result | G1 | G2 | G3 | G4 | G5 | D1-D4 | Verdict |
|---|---|---|---|---|---|---|---|

## Significance vector — 09-15 wave (Q3F-T1)

| Result | G1 | G2 | G3 | G4 | G5 | D1-D4 | Verdict |
|---|---|---|---|---|---|---|---|
| Stamp check BEFORE decode: fp8 scale planes corrupt fleet-wide (144/144/rank, 16 ranks + tp4pp4 12-15); verifier coverage gap found | — | ++ | + | + | + | D3 | found defect; fix path = re-emit scale planes; verifier must force fp8 coverage |
| Latent main break: _Static_assert in qwen4_flash CUDA source fails nvcc | — | — | + | + | + | D1 | same class as the #991 qwen38max find; fixed on lane |
| qwen4_flash module missed the #1013 mesh-lanes adoption; full lazy-attach ported (attach + manifest check + expert lease + mesh prepare) | + | ++ | + | + | + | D1 | attach green end-to-end on 8 ranks via shared weightsd |
| Weightsd unit LimitMEMLOCK 8MB -> ibv_reg_mr(4G) ENOMEM; mesh single-shot init without retry | — | — | + | ++ | + | D1 | drop-in + restarts recovered ranks 0-7 (peers=15); PR adds init retry loop |
| First Execute SIGBUS inside the mesh-lane arena mapping at the first routed MoE | — | ++ | + | + | + | D1 | uncharted shared spine-wire defect; receipted; needs owner (coredev/mgr2) |
| Reference engine + tokenizer + fixtures plumbing + wave scripts + stamp-check instrument | + | ++ | + | — | + | D1/D4 | the T1 half that never existed for this driver now exists and builds green |
| B1 tok/s | — | — | — | — | — | D2 | NOT MEASURED — zero tokens; correctly not fabricated |
