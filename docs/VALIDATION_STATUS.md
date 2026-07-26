# Validation Status

Generated at `2026-07-25T08:25:34.741764+00:00`.

This is a host-side receipt for the audited source tree. It does not qualify CUDA execution, numerical correctness, latency, throughput, or production readiness.

The harness acquires an exclusive repository lock, fingerprints the source tree, and begins with `make clean`, so a passing receipt cannot be inherited from an incremental or concurrent build. The final fingerprint must match before the receipt is issued.

Input source fingerprint: `40f4913b21d1b943f9cbc9f43dc5a650eb7a4d81aa07d15d98cfed2453346b1f` (5031 files).

Output source fingerprint: `40f4913b21d1b943f9cbc9f43dc5a650eb7a4d81aa07d15d98cfed2453346b1f` (5031 files).

| Step | Category | Status | Return code | Duration | Log |
|---|---|---:|---:|---:|---|
| `clean_build` | preflight | **passed** | 0 | 0.240s | `docs/validation-logs/clean_build.log` |
| `python_tool_syntax` | tooling | **passed** | 0 | 0.179s | `docs/validation-logs/python_tool_syntax.log` |
| `host_build` | build | **passed** | 0 | 7.470s | `docs/validation-logs/host_build.log` |
| `core_boundary_audit` | architecture | **passed** | 0 | 0.235s | `docs/validation-logs/core_boundary_audit.log` |
| `non_glm_model_driver_contracts` | driver-contract | **passed** | 0 | 1.550s | `docs/validation-logs/non_glm_model_driver_contracts.log` |
| `memory_contracts` | contract | **passed** | 0 | 4.504s | `docs/validation-logs/memory_contracts.log` |
| `host_test_suite` | test | **failed** | 2 | 16.222s | `docs/validation-logs/host_test_suite.log` |
| `required_host_targets` | build | **passed** | 0 | 0.071s | `docs/validation-logs/required_host_targets.log` |
| `cuda_node_context_builder` | optional-cuda-build | **skipped** | — | 0.000s | `docs/validation-logs/cuda_node_context_builder.log` |
| `source_tree_stability` | preflight | **passed** | 0 | 1.624s | `docs/validation-logs/source_tree_stability.log` |

The aggregate host suite stopped only at `test_tp_collective`: this Codex
sandbox cannot reserve a loopback TCP port for that test. A follow-up run
excluding only `test_tp_collective` completed every other C/C++ binary and
Python contract, including `test_cuda_performance_contracts.py`; its receipt is
`docs/validation-logs/host_test_suite_without_loopback.log`. This follow-up does
not convert the sandbox-blocked network test into a pass.

## Commands

### `clean_build`

```sh
make clean
```

### `python_tool_syntax`

```sh
/Users/cem/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m py_compile tools/audit_core_boundaries.py tools/generate_proposed_change_manifest.py tools/package_audited_proposal.py tools/run_deep_audit_validation.py tests/test_model_driver_contracts.py
```

### `host_build`

```sh
make -j2 all
```

### `core_boundary_audit`

```sh
make architecture_audit
```

### `non_glm_model_driver_contracts`

```sh
make model_driver_contracts
```

### `memory_contracts`

```sh
/Users/cem/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 tests/test_memory_contracts.py
```

### `host_test_suite`

```sh
make -j2 test
```

### `required_host_targets`

```sh
make -j2 glm52_pp13_service_backend tools
```

### `cuda_node_context_builder`

```sh
make -j2 glm52_pp13_node_context_builder
```

Note: nvcc is unavailable; this is not a pass

### `source_tree_stability`

```sh
internal source-tree-fingerprint
```

Note: input=40f4913b21d1b943f9cbc9f43dc5a650eb7a4d81aa07d15d98cfed2453346b1f output=40f4913b21d1b943f9cbc9f43dc5a650eb7a4d81aa07d15d98cfed2453346b1f input_files=5031 output_files=5031

## Interpretation

- `passed` means only that the named host command returned success.
- `failed` is a release blocker for this proposal.
- `skipped` is not a pass; the required local tool was unavailable or a preflight blocked the step.
- Non-GLM drivers remain `NOT_MEASURED` until exact-hardware GPU receipts exist.
