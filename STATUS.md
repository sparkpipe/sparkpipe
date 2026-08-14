# SparkPipe Status — Phase 10 Hardware Handoff

Current live DSV4 decode measurements are tracked in
[`PERFORMANCE_STATUS.md`](PERFORMANCE_STATUS.md).  The Phase 10 flags below
describe the older source-package handoff and are not the current DSV4 runtime
measurement status.

This source tree is an implementation and qualification handoff candidate for:

- Kimi K3: MXFP4 routed-expert weights, BF16 expert activations, BF16 non-expert tensors, FP32 accumulation.
- GLM 5.2: FP8 E4M3 routed-expert weights, BF16 activations and non-expert tensors, FP32 accumulation.
- Qwen 3.6 27B: BF16 weights and activations.
- DeepSeek V4 Flash and DeepSeek V4 Pro: separate checkpoint-derived contracts and execution packages.

Phase 10 adds a fail-closed Spark hardware truth suite. Thirty-three named questions cover GB10 memory and launch behavior, exact production kernels, NVMe, TCP/RDMA, physical ring and one-switch behavior, PP degree, transport window, and stage placement. Every production decision must trace to exact source-bound cells, retained receipts, a generated policy, and a closure report.

The release supports the initial direct single-rail ring and the first one-switch single-rail topology. Dual-switch/dual-rail operation remains disabled until single-rail ownership, ordering, retry, and failure behavior are measured and closed.

The source tree does not certify itself. Build, test, CUDA, network, and production claims are valid only when tied to the exact released archive SHA-256 by an external verification receipt or hardware receipt.

```text
SOURCE_PACKAGE_KIND=sparkpipe_source
HOST_BUILD_STATUS=SEE_EXTERNAL_VERIFICATION_RECEIPT
HOST_TEST_STATUS=SEE_EXTERNAL_VERIFICATION_RECEIPT
ARCHITECTURE_GATE_STATUS=SEE_EXTERNAL_VERIFICATION_RECEIPT
HARDWARE_QUESTION_COUNT=33
CUDA13_SM121A_COMPILE_NOT_RUN=true
BLACKWELL_EXECUTION_NOT_MEASURED=true
PHYSICAL_NETWORK_EXECUTION_NOT_MEASURED=true
PHYSICAL_RING_NOT_MEASURED=true
SINGLE_SWITCH_NOT_MEASURED=true
DUAL_RAIL_ENABLED=false
PRODUCTION_READY=false
```

See:

- `docs/SPARK_HARDWARE_HANDOFF.md`
- `docs/PHASE10_HARDWARE_HANDOFF_RELEASE.md`
- `docs/PHASE10_VALIDATION_STATUS.md`
- `docs/PHASE10_REMAINING_WORK.md`
