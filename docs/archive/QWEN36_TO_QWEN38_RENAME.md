# Model family naming correction (2026-08-25)

## The problem

Two different models have confusingly wrong names:

| Directory | Hidden | Layers | Actual model | Should be |
|----------|--------|--------|--------------|-----------|
| model-families/qwen38_27b | 5120 | 64 | Qwen 3.8-27B | qwen38_27b |
| model-families/qwen38 | 8192 | 92 | Qwen 3.8 Max | qwen38_max |
| modules/qwen38_27b_resident_decode_stage | | | serves Qwen 3.8-27B | qwen38_27b_resident_decode_stage |
| modules/qwen38_resident_decode_stage | | | serves Qwen 3.8 Max | qwen38_max_resident_decode_stage |

Both firmware headers incorrectly say "Qwen 3.6 27B" — the qwen38 header
was copy-pasted from qwen38_27b without updating the model name.

## The correct naming

- `qwen38_27b` — the 27B dense hybrid (ours, production, DFlash2)
- `qwen38_max` — the larger MoE model (decode-only module)

## Execution plan (1-2 hour dedicated session)

Phase 1: Rename directories (breaks builds, fix immediately)
  git mv model-families/qwen38_27b model-families/qwen38_27b
  git mv model-families/qwen38 model-families/qwen38_max
  git mv modules/qwen38_27b_resident_decode_stage modules/qwen38_27b_resident_decode_stage
  git mv modules/qwen38_resident_decode_stage modules/qwen38_max_resident_decode_stage

Phase 2: Rename files
  cd modules/qwen38_27b_resident_decode_stage
  for f in spark_qwen38_27b_*; do git mv $f ${f/qwen38_27b/qwen38_27b}; done
  (same for qwen38_max, renaming qwen38 → qwen38_max)

Phase 3: Update all content (sed pass)
  find . -type f \( -name "*.c" -o -name "*.h" -o -name "*.cu" -o -name "*.cuh" -o -name "*.py" -o -name "*.sh" -o -name "Makefile" \) -exec sed -i 's/qwen38_27b/qwen38_27b/g; s/QWEN38_27B/QWEN38_27B/g; s/Qwen38_27b/Qwen38_27b/g' {} +
  # then for the max module:
  find modules/qwen38_max_resident_decode_stage -type f -exec sed -i 's/qwen38(?!_27b|_max)/qwen38_max/g' {} +
  # (careful with look-ahead; may need manual pass)

Phase 4: Update Makefile, tests, docs, build scripts

Phase 5: Verify
  - Module validates PASS
  - Full stack deploys
  - O512: 512 tokens, d7f79880, ~21s
  - API: completion works
  - All tests pass

## Risk mitigation

- Do it on a branch
- Commit after each phase
- Full rebuild + test after each phase
- The 27B rename (qwen38_27b→qwen38_27b) is the priority (169 files)
- The Max rename (qwen38→qwen38_max) is smaller (4 files + Makefile)
