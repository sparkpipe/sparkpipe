# GLM-5.2 PP13 stage-slice balancing estimate, 2026-07-01

This note uses the Spark2 PP13 timing data from
`GLM52_PP13_STAGE_RULES_AND_TIMING_20260701.md` and keeps the production
constraint that only 13 sparks are currently available.  Each measured six-layer
slice is converted into uniform per-layer costs inside that slice.  The
final-stage LM-head/restricted-token cost is modeled as an extra cost on the
last stage.

These are planning estimates, not new CUDA hardware measurements.

## Current uniform PP13 baseline

| bucket | baseline slowest stage | estimated throughput |
| --- | ---: | ---: |
| B64 including final | 112.159 ms | 571 tok/s |
| B64 hidden-only | 96.356 ms | 664 tok/s |
| B32 including final | 112.656 ms | 284 tok/s |
| B32 hidden-only | 81.610 ms | 392 tok/s |

## Final-aware measured PP13 plan

| bucket | sparks | slowest modeled stage | estimated throughput | stage ranges |
| --- | ---: | ---: | ---: | --- |
| B64 | 13 | 78.741 ms | 813 tok/s | `0:9,9:8,17:8,25:8,33:8,41:6,47:5,52:5,57:5,62:5,67:4,71:4,75:3` |
| B32 | 13 | 63.119 ms | 507 tok/s | `0:10,10:8,18:8,26:8,34:8,42:6,48:6,54:5,59:5,64:4,68:4,72:5,77:1` |

## Interpretation

The measured profile says the right side of the model is much more expensive
than the left side, and the final-token stage adds another large fixed cost.
Uniform six-layer cuts leave the last stages overloaded.  Final-aware PP13
balancing moves layers leftward and shrinks the final stage.

Expected relative gains from the current uniform PP13 baseline:

| bucket | plan | estimated gain |
| --- | --- | ---: |
| B64 | PP13 measured balanced | 1.42x |
| B32 | PP13 measured balanced | 1.78x |

The first stage is legal even though it is wider than eight total layers,
because the cut rule is eight routed layers plus the three dense-prefix layers.
The stage-slice ABI now permits this dense-prefix slice while keeping routed
slices capped by the GLM-5.2 routed-layer rule.

Bulk-prefill gets the PP13 submit-count win only when prefill frames are routed
through the stage-slice context.  The module now sends a prefill frame for a
slice through one backend completion path and validates every layer's bulk
prefill plan before admission.  A fully fused production bulk-prefill launch
can replace the per-layer launch functions behind that same ABI.
