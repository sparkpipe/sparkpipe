# Qwen 3.8 Max DRY audit against the unified contract

unified.md owns: one DRY tree - model facts in tables, shared machinery
once, no parallel copies of tooling/adapters/kernels. This audit scans the
qwen38 contribution for violations and ranks them. Fixes landed this round
are marked DONE; the rest carry exact consolidation plans.

## Landed this round (DONE)

1. Geometry fingerprints: the per-family FNV-1a copy
   (SparkQwen38ModuleFingerprint) is deleted; the module now calls the
   shared SparkHashBytes from spark_status.h (identical byte loop).
   qwen36 still carries its own copy - migrate it the same way.
2. TP combine kernel: the qwen38 copy of the accumulate-add kernel is
   deleted; SparkQwen38LaunchTpCombineAdd now forwards to the new shared
   SparkLmHostLaunchAccumAddBf16 in spark_lm_kernels.cuh. dsv4, glm52 and
   qwen36 still carry identical copies (each ~20 lines) - migrate them
   next (their callbacks are the same shape).

## Remaining violations, ranked

P1 - spark_qwen38_work_control.c is a verbatim rename of
     spark_qwen36_work_control.c (model-families/qwen36 and /qwen38), and
     tests/test_qwen38_work_control.cpp mirrors test_qwen36_work_control.cpp.
     The building blocks are model-agnostic (keyed GET/PUT batches, lane
     restore/evict plans). Fix: promote to one shared
     model-families/common work_control (or cache/store), keep the two
     headers as thin include aliases if the per-family names are load-
     bearing, point both modules and both tests at it. No behavior change.

P1 - Per-family KV-tier client-open + fingerprint plumbing: qwen38's
     SparkQwen38ModuleOpenKvTier repeats qwen36's SparkQwen36ModuleOpenKvTier
     shape (env reads, client open, record sizing). The fingerprints are
     now shared (item 1); the remaining env/open boilerplate should move to
     a shared SparkStageModuleKvTierOpen in stage_module_common once the
     tier loops themselves converge.

P2 - Router top-k + softmax: the qwen38 module's GateSelectKernel (bitonic
     sort + softmax renormalise) and dsv4's equivalent duplicate the shared
     LmTopkSmallKernel (inference/kernels/topk.cuh) the family drivers use.
     Fix: the module-side router should call the shared kernel (it already
     handles identity/sigmoid score transforms + RENORMALISE); the only
     module-specific piece is the NaN-to-last ranking, which the shared
     kernel should gain as a flag rather than fork.

P2 - SwiGLU elementwise: qwen36 and qwen38 modules each carry a
     SwiGluKernel copy; the shared SparkLmSwish device function exists but
     no shared launcher. Fix: SparkLmHostLaunchSiluMul in
     spark_lm_kernels.cuh; both modules forward to it.

P3 - tools/qwen38_timing_probe.c is qwen38-specific, but the probe pattern
     (PACK BATCH STEPS + debug bisect flags) is model-agnostic. Fix when a
     second family needs it: parameterize the module entry points and
     share one probe tool.

P3 - qwen38_stagepack.py vs qwen36_stagepack.py: per-family packers carry
     shared scaffolding (safetensors source, receipt, alignment). unified
     intends packer consolidation ("parameterize first"); check the
     unified packer state before extending qwen38's further.

## Not violations (accepted divergence)

- Per-family serving adapters + firmware headers: the established pattern
  (descriptor-registered, per-model geometry).
- The KV tier residency loop (SparkQwen38ModuleKvPrepareFrame): qwen38-
  specific window/eviction policy; converges with qwen36's only when the
  tier paging loop is itself promoted.
- docs/QWEN38_MAX_*.md: per-session records, not code.

## Gate status on the unified tree

test_dry_law.py PASS; test_must_work_targets.py PASS; the module archive,
both stage smokes, test_qwen38_math_kernels and test_qwen38_work_control
all PASS after the two fixes above.
