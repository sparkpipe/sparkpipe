# Phase 4 Validation Status

Generated at `2026-07-31T19:10:41Z`.

## Result

Phase 4 is a **host-validated source candidate**. The complete Makefile test
inventory was executed from a clean build after the final source changes.

| Validation surface | Result |
|---|---:|
| Strict clean host build | Passed |
| Clean `make all` including study/tools targets | Passed |
| C/C++ test binaries | **42 / 42 passed** |
| Python tests | **49 / 49 passed** |
| Architecture/source gates | **66 passed, 4 CUDA-only skipped, 0 failed** |
| Core-boundary audit | Passed |
| Non-GLM model-driver contracts | Passed |
| Memory-contract audit | Passed |
| CUDA 13 `compute_121a` compilation | **Not run** |
| CUDA 13 `sm_121a` assembly/device link | **Not run** |
| Blackwell execution | **Not measured** |
| Network execution | **Not measured** |
| Production readiness | **False** |

The retained clean command was:

```sh
make clean && make -j4 test
```

The final host suite and aggregate architecture gates were serialized. The full
suite covered the exact Makefile inventories: 42 compiled C/C++ test binaries
and 49 Python tests.

A single transient `test_tp_collective` `IO_ERROR` occurred during an earlier
aggregate run. The same binary passed 30 consecutive standalone repetitions and
the retained complete binary inventory. No product-code fix is claimed for that
observation; it remains a bring-up item for the physical ring and switched
fabric.

## Source identity

The fingerprint includes authored source, tests, configuration, and retained
non-Phase-4 documents. It excludes generated packaging receipts, retained Phase
4 execution logs, and transient build/cache artifacts.

```text
source files:       655
source bytes:       7,670,877
source fingerprint: 74aaacc38be53f1cb9d1b00209033421a2fb82d2048eaf4f48758c216bc4ae1e
```

Excluded exact paths:

```text
PACKAGE_MANIFEST.json
SHA256SUMS
docs/VALIDATION_STATUS.md
docs/VALIDATION_STATUS.json
docs/PHASE4_VALIDATION_STATUS.md
docs/PHASE4_VALIDATION_STATUS.json
docs/PHASE4_CHANGE_MANIFEST.md
```

Excluded prefixes, directories, and artifacts:

```text
qualification/phase4/
.git/
build/
.audit/
__pycache__/
.pytest_cache/
.mypy_cache/
*.pyc
*.pyo
*.o
*.a
*.so
*.dylib
*.dll
*.exe
```

## CUDA boundary

The four skipped architecture gates are:

```text
PTX capability gate          ptxas unavailable
grouped top-k CUDA build     nvcc unavailable
K3 replay-fold CUDA build    nvcc unavailable
complete sm_121a CUDA build  nvcc unavailable
```

The CUDA installer was attempted, but this sandbox could not resolve NVIDIA's
download host. No CUDA compiler, driver, or partial toolkit installation is
claimed. The source tree includes the exact `compute_121a`/`sm_121a` compile
gate for execution on a Linux CUDA 13 host.

Host validation does not establish CUDA compilation, architecture-specific
PTX/SASS acceptance, device linking, register/spill behavior, GPU numerical
correctness, memory/race safety, CUDA Graph behavior, GPUDirect ordering,
network performance, latency, throughput, or production readiness.

## Precision scope validated at source level

- K3: MXFP4 routed-expert weights, BF16 routed activations and BF16 non-expert
  tensors, FP32 accumulation.
- GLM 5.2: FP8 E4M3 routed-expert weights, BF16 routed activations and BF16
  non-expert tensors, FP32 accumulation.
- Qwen 3.6 27B: BF16 weights and activations, FP32 accumulation where required.
- DeepSeek V4 Flash and Pro: separate generated geometry and checkpoint-native
  mixed-precision contracts; complete execution remains unfinished.

## Status labels

```text
HOST_COMPLETE_SUITE_VALIDATED
HOST_ARCHITECTURE_GATES_VALIDATED
CUDA13_SM121A_COMPILE_NOT_RUN
BLACKWELL_EXECUTION_NOT_MEASURED
NETWORK_EXECUTION_NOT_MEASURED
PRODUCTION_READY=false
```
