# SparkPipe audited architecture proposal

This source tree applies the deeper core-boundary and non-GLM driver audit requested for SparkPipe. It is a complete source snapshot, not a Git clone; commit history, branches, and remotes are not included.

## Main changes

### Neutral core boundary

```text
include/sparkpipe/             neutral public ABI
src/                          neutral core/compiler/runtime
core/                         neutral source manifests
model-families/common/        reusable model-runtime facilities
model-families/glm52/         GLM-5.2 host implementation
model-families/dsv4/          DSV4 geometry
model-families/k3/            K3 geometry
model-families/mimo25/        MiMo 2.5 geometry
model-families/qwen36/        Qwen 3.6 host implementation
deployment/                   release and rollout services
modules/                      linkable firmware modules
```

GLM-5.2 is no longer part of the neutral archive or include closure. The build emits separate core, compiler, runtime, shared-model, deployment, GLM, and Qwen archives.

### ABI and validation

- Model-driver ABI 7 with a versioned create request and resident-owned execution stream.
- Firmware-module ABI 4 with sized descriptors and reserved-field validation.
- Module-record schema 4 binds the validator executable SHA-256.
- Generated multi-operation admission and snapshot aggregation is conservative.
- External completion is allowed only when one operation unambiguously owns it.
- Loader descriptor and profile checks fail closed.

### Non-GLM drivers

DSV4, K3, MiMo 2.5, and Qwen 3.6 now share strict resident-stage support and package contracts. They reject generic dispatch tickets as model lanes, guard lane releases by successful ownership, validate stage-specific buffers and model contexts, and remain explicitly `NOT_MEASURED`.

### Shared tensor-parallel transport

The TCP reference collective is now ABI 2 and fail-closed under peer loss or protocol disagreement. Connection, reciprocal handshake, and payload I/O use absolute monotonic deadlines. Handshakes bind degree, rank, and a deployment-supplied collective identifier; operation headers bind group, sequence, step, operation, and element count before payload transfer. A failed collective closes all sockets and cannot be reused.

The loopback test no longer deadlocks when only part of a group initializes. It uses dynamic port ranges and a cross-process test lock, verifies missing-peer timeouts, mismatched element counts, disjoint value/scratch buffers, destroy-safe failed creation, idempotent teardown, sequence advancement across consecutive collectives, repeated runs, and TP degrees through 16.

### Publication

The common non-GLM publication rule requires exact `sm_121a`, a readable stage pack, an executable GPU validator, and a configuration-bound validation identity. CPU references cannot qualify firmware.

## Host validation

Run the retained audit harness:

```sh
python3 tools/run_deep_audit_validation.py --jobs 2
```

The harness takes an exclusive repository lock, fingerprints every retained source input, removes the prior build with `make clean`, and then runs every host gate sequentially. Concurrent validation attempts fail closed instead of sharing build products or overwriting receipts, and the final source fingerprint must match before a passing receipt is issued.

Individual gates:

```sh
make -j2 all
make architecture_audit
make model_driver_contracts
python3 tests/test_memory_contracts.py
make -j2 test
make -j2 glm52_pp13_service_backend tools
```

The harness writes:

```text
docs/VALIDATION_STATUS.json
docs/VALIDATION_STATUS.md
docs/validation-logs/
```

Absence of `nvcc` is recorded as **skipped**, never as passed. Host success does not qualify CUDA execution or model correctness.

## Deterministic source package

Create the source-clean proposal archive with:

```sh
python3 tools/package_audited_proposal.py \
    --repository . \
    --output ../sparkpipe-audited-proposed.tar.gz \
    --root-name sparkpipe-audited-proposed
```

The packager excludes Git metadata, build products, Python caches, platform metadata, and nested archives. It holds the same repository lock used by validation, rejects outputs inside the source tree, fingerprints the complete retained source before and after staging, verifies that the staged tree exactly matches the locked source snapshot, scans retained files for credential and private-key patterns, writes an internal file manifest and `SHA256SUMS`, normalizes metadata for reproducibility, and atomically publishes the archive and checksum sidecar under an exclusive output lock.

## Audit documents

- `docs/CORE_CONTAMINATION_AUDIT.md`
- `docs/MODEL_DRIVER_AUDIT.md`
- `docs/LLM_DEVICE_DRIVER_INTERFACE.md`
- `docs/PROPOSED_CHANGE_MANIFEST.md`
- `docs/VALIDATION_STATUS.md`

## Qualification boundary

The non-GLM modules remain controlled bring-up implementations. Before production, each exact package needs:

- target-GPU compilation and link receipt;
- real-weight pack validation;
- numerical comparison against an authoritative implementation;
- memory-safety and bounds tests on representative shapes;
- stream/transport/completion tests;
- latency and throughput receipts;
- immutable runtime configuration rather than environment-only configuration.

No fallback or production-ready claim has been added by this proposal.
