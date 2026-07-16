# Codex Runbook

Follow this workflow for SparkPipe changes. Do not substitute a Mac CUDA build,
a dirty Spark checkout, a copied shared object, or a compile-only result.

## Goal

The goal is correct, fast GLM-5.2 inference through the public API and all 13
Sparks. A build is not an inference test. A ready health endpoint is not an
inference test. A release is tested only after a real prompt returns correct
tokens through the installed release.

## Fixed Rules

- `main` is the advisor handoff branch.
- Publish repository changes with `updaterepo`; do not manually push.
- Run `updaterepo` from a checkout whose Git common directory is writable by
  the task. Do not use a linked worktree whose `.git` points outside the
  writable workspace.
- Run host tests with `make -j`.
- Build CUDA and releases on spark0 from a clean worktree at merged `main`.
- Keep rank-local FP8 packs at the one configured stable root.
- Never delete or regenerate `.sp*` packs without explicit user approval.
- Never silently fall back to a reference, compatibility, or demo path.
- Deploy one immutable manifest generation to every rank.
- Keep the deployed runtime at the measured B1 shape while correctness or
  deployment is changing. Increase active lanes only through a new manifest
  and a measured inference gate.
- Use `MEASURED`, `OBSERVED`, `NOT_MEASURED`, and `NOT_WORKING` exactly as
  defined in `docs/GLM52_MEASURED_STATUS.md`.
- Code presence, compile success, host tests, capability flags, and health
  readiness are never evidence of accuracy or performance.

## Repository Update

Work in the checkout whose contents are the desired change. Run local tests,
then let the repository tool create, validate, merge, and synchronize the PR:

```sh
make -j test
updaterepo "Short human title"
```

Do not pass paths, validation commands, PATs, branches, or manual GitHub
commands to `updaterepo`. If it fails, fix the failing workflow step rather
than bypassing it.

## Spark0 Build Root

Use spark0 for CUDA and release work. The persistent checkout may be dirty, so
build from a detached clean worktree at the exact merged commit:

```sh
git -C /home/spark0/src/sparkpipe-main-live fetch origin main
git -C /home/spark0/src/sparkpipe-main-live worktree add --detach \
    /tmp/sparkpipe-release-<sha8> origin/main
git -C /tmp/sparkpipe-release-<sha8> status --short
git -C /tmp/sparkpipe-release-<sha8> rev-parse HEAD
```

The status output must be empty and the commit must equal merged `main`.

## Build Gates

Run the host suite, release tools, CUDA resident archive, backend, and
transport from the clean worktree:

```sh
make -C /tmp/sparkpipe-release-<sha8> -j \
    test tools \
    glm52_pp13_service_backend \
    hidden_transport_spark_host_rdma_verbs \
    glm52_pp13_node_context_builder
```

Package the FP8 driver with the exact six-layer PP13 validator, not the legacy
dense-to-layer3 default:

```sh
make -C /tmp/sparkpipe-release-<sha8> -j \
    glm52_resident_decode_stage_firmware_package \
    MAX_STAGE_MICROSECONDS=1000000 \
    GLM52_VALIDATION_MODE=exact_pp13_stage_slice \
    GLM52_VALIDATION_ACTIVE_SEQUENCE_COUNT=1 \
    GLM52_VALIDATION_FIRST_ROUTED_LAYER_INDEX=0 \
    GLM52_VALIDATION_ROUTED_CHAIN_LAYER_COUNT=6 \
    GLM52_PIPELINE_INPUT_HIDDEN_BF16=<nonzero-12288-byte-fixture> \
    GLM52_ENABLE_CUDA_GRAPH_REPLAY=1 \
    GLM52_MODEL_DIR=/home/spark0/models/hf/zai-org/GLM-5.2-FP8 \
    GLM52_STAGE_PACK_DIR=/home/spark0/models/sparkpipe/glm52_fp8_pp13_stage_payload_v1 \
    GLM52_FP8_MOE_PACK_DIR=/home/spark0/models/sparkpipe/glm52_fp8_pp13_stage_payload_v1 \
    GLM52_EXACT_PP13_MODEL_QUANTIZATION=fp8 \
    GLM52_MOE_BACKEND=fp8 \
    GLM52_REQUIRE_B12X_RESIDENT_PACK=0
```

Require both validation lines: one for the archive and one for the linked
`model_driver.so`. Record stage time, graph captures/replays, nonzero output
count, and checksum.

## Assemble A Release

Clone the previous known-good role manifest and replace every rebuilt artifact
with the repo-owned assembler. It refuses unknown paths, recomputes every size
and SHA-256, writes into a temporary directory, and renames atomically:

```sh
python3 tools/sparkpipe_release_assemble.py \
    --template <known-good-release> \
    --output <new-release> \
    --release-id <release-id> \
    --git-commit <full-merged-sha> \
    --max-active 1 \
    --kv-pool-tokens 16384 \
    --kv-logical-blocks 256 \
    --mtp \
    --replace bin/sparkpipe_release_manager=build/sparkpipe_release_manager \
    --replace bin/sparkpipe_glm52_cuda_residentd=build/sparkpipe_glm52_cuda_residentd \
    --replace bin/sparkpipe_glm52_pp13_rank_daemon=build/sparkpipe_glm52_pp13_rank_daemon \
    --replace bin/sparkpipe_glm52_http_gateway=build/sparkpipe_glm52_http_gateway \
    --replace lib/libglm52_pp13_node_context_builder.so=build/libglm52_pp13_node_context_builder.so \
    --replace lib/libglm52_pp13_service_backend.so=build/libglm52_pp13_service_backend.so \
    --replace lib/libhidden_transport_spark_host_rdma_verbs.so=build/libhidden_transport_spark_host_rdma_verbs.so \
    --replace lib/model_driver.so=build/packages/glm52_resident_decode_stage/stages/stage_000/model_driver.so
```

`--mtp` is mandatory for an MTP release. Omit it only for an intentionally
named plain-decode control release. Validate the resulting role commands and
require `--mtp` on both the resident and gateway roles before serving an MTP
release.

Use repeated `--role-env ROLE=NAME=VALUE` arguments for narrowly scoped runtime
experiments. The assembler replaces an existing value for the same role and
name, so profiling releases remain immutable and reproducible. Do not hand-edit
the assembled manifest. For graph-compatible MTP timing, add:

```sh
--role-env pp13_cuda_residentd=SPARKPIPE_MTP_GPU_PROFILE=1
```

MTP is capped at one draft token until all 13 ranks implement a distributed
speculative-KV commit and rollback transaction. Do not raise the draft budget
from measured acceptance alone; multirow verification leaves rejected future
KV rows resident on ranks that never receive the final-rank acceptance count.

Validate before serving:

```sh
<new-release>/bin/sparkpipe_release_manager validate \
    --manifest <new-release>/sparkpipe.json
```

Production releases omit runtime diagnostics by default. Add `--diagnostics`
only when assembling a release for an explicit correctness investigation.
Template history cannot silently retain stage dumps, phase hashes, completion
logs, or PP13 packet tracing. Do not hand-edit the manifest.

The 16,384-token physical pool and 256 logical blocks are the current measured
B1 configuration when rank12 also owns native MTP weights. Increase either
value only after measuring rank12 resident memory headroom; this configuration
is not a long-context or B16 claim.

## Deploy The Ring

Serve the immutable release from spark0. Apply roles in this order:

1. Stop `spark0_gateway` and verify its PID is gone.
2. Stop `pp13_rank_daemon` on ranks 1 through 12 and verify every PID is gone.
3. Apply `pp13_cuda_residentd` on ranks 0 through 12 concurrently.
4. Wait for every resident to report `state=ready` with no work frames received.
5. Apply `pp13_rank_daemon` on ranks 1 through 12 concurrently.
6. Apply `spark0_gateway` on rank 0.

Each role uses the installed release manager:

```sh
/home/<host>/sparkpipe_runtime/bin/sparkpipe_release_manager agent \
    --release-url http://spark0:<release-port>/ \
    --staging-dir /home/<host>/sparkpipe_state/release_staging \
    --install-dir /home/<host>/sparkpipe_runtime \
    --state-dir /home/<host>/sparkpipe_state \
    --host <host> \
    --rank <rank> \
    --role <role> \
    --once
```

Do not launch a second resident beside an existing allocation. The resident
agent stops its old generation, but it does not quiesce the gateway or rank
daemons. Leaving either control role alive can replay buffered work into the
fresh resident and invalidate the test. Compare the installed builder and
driver hashes on all 13 ranks with the manifest.

## Actual Inference Gate

First require local health, then allow one probe request to prove the ring:

```sh
curl -fsS http://spark0:18080/health
```

`local_control_ready=1` proves only the gateway, backend, and rank-0 resident
are attached. It is not a whole-ring readiness claim. Whole-ring readiness is
observed only after a request produces token and done events and drains cleanly.

Then send a streaming, greedy request. Read the API key from the installed
file without printing it:

```sh
curl -sS -N --max-time 120 \
    -H "Authorization: Bearer $(cat /home/spark0/sparkpipe_runtime/API_KEY)" \
    -H "Content-Type: application/json" \
    -d '{"model":"glm-5.2","prompt":"Say OK. OK.","max_tokens":1,"temperature":0,"stream":true}' \
    http://127.0.0.1:18080/v1/completions
```

The correctness receipt is token `10397`, text `" OK"`, followed by a done
event. Also run a distinct factual prompt and an 8-or-more-token decode. A
queued `202` without token and done events is not a passed inference test.

Those prompt checks are smoke observations, not model-accuracy measurements.
Accuracy remains `NOT_MEASURED` until a retained reference or corpus score is
run against the exact deployed release.

## Performance Gate

Use `tools/sparkpipe_api_stress.py` for API timing and retain its JSONL and
summary. Report separately:

- exact six-layer CUDA stage time
- prefill time
- decode-step time
- first-token latency
- steady generated-token rate
- concurrent aggregate token rate
- transport time

Run concurrency 1 first, then 4, 16, and larger only while throughput rises.
If completion times form a staircase, requests are serialized before the GPU
scheduler and larger bucket claims are invalid.

Every performance receipt must include the merged commit, immutable release
identity and generation, all-rank artifact hashes, actual observed lane maxima,
backend and transport identities, raw output, and post-run queue state. A
theoretical ceiling must be labeled theoretical and kept separate from measured
end-to-end throughput.

## Diagnostics

Role logs are under:

```text
/home/<host>/sparkpipe_state/run/<role>.pid.log
```

When accuracy is under investigation, keep the detailed stage dump and hash
instrumentation enabled and compare the ring against the serialized FP8 oracle
at every stage. First divergence wins; do not guess downstream causes.

## Forbidden Evidence

- Mac CUDA output
- a dirty Spark checkout
- an unmerged branch or copied shared object
- `make -j test` without the SM121 archive/package gate
- health without a returned token
- a reference fallback presented as production performance
- a B64/B256/B1024 claim when the live request reaches one lane
