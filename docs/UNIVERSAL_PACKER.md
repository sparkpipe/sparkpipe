# The Universal Packer — design (operator directive 2026-08-30)

## Why (the incident that justifies it)

Eight packers, ~8,700 lines, one shared job: read safetensors source →
slice by topology → encode by codec → emit family pack + receipts.
Tonight qwen-max-4bit blocked because qwen38_stagepack hard-asserts
F8_E4M3 experts while glm52_stagepack has the complete NVFP4 path
sitting unused. Same class as the validator DRY wave (six copy-pasted
validation scripts → one driver): per-family packers drift; codec and
topology knowledge belongs in ONE place.

## Shape: universal core + family descriptors + thin emitters

```
tools/sparkpipe_stagepack.py          (ONE CLI)
tools/stagepack_core/                 (the universal machinery)
  codecs.py      CODECS table: bf16 | fp8-e4m3-b128 | nvfp4-g16-ue4m3 |
                 mxfp4-e2m1-g32 | fp8-dynamic — payload+scale planes,
                 group shapes, scale encodings (glm CODECS is the seed;
                 it already built glm53full nvfp4)
  source.py      safetensors reading, dtype/shape contracts (the
                 radixark U8+scales layout = a codec-6 source, not a
                 special case)
  topology.py    TP-rank slicing (glm/dsv4 style), full-width-per-stage
                 (qwen_max runtime-slice style), PP stage windows
  receipts.py    sha-256 receipts, manifests, staging, atomic outputs
  emit/          per-family BYTE-COMPATIBLE emitters (glm52sp, spstage,
                 qwen38sp, k3 pack) — the modules' loaders are the
                 contract; formats do NOT change in this phase
family descriptors = model_contracts/*_authoritative.json (geometry,
tensor maps, topology class, codec allowlist) — no family names in
the core, ever (dry-law).
```

## Build order (each step gated)

1. CORE EXTRACTION: codecs + source + receipts from glm52_stagepack
   (the most complete) — standalone, unit-tested against the existing
   pack fixtures.
2. FIRST EMITTER: qwen38 (the motivating case) — universal core +
   qwen38 emitter must byte-reproduce an existing fp8 pack (the
   q27b-style identity proof), THEN add the nvfp4 expert path (glm
   reference) → unblocks qwen-max-4bit as the universal packer's
   first new capability.
3. Remaining emitters ported one per family, each gated on
   byte-identity against its existing packs: glm52 → glm5_next →
   dsv4 → k3 → qwen4_flash → 27B.
4. The 16-way parallel build (tonight's proven pattern) becomes a
   core flag (--fleet-build: stage/rank fan-out via the queue).
5. OLD PACKERS retire behind the drift/verify gates; SHA receipts
   pin the transition.

## What does NOT change

Module loaders, pack formats, validators, the placement tooling —
byte compatibility is the contract. The k3_pack V2 interleave and
family quirks live in emitters, not the core.
