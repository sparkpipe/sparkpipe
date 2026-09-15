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
