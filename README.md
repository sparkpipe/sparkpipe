# SparkPipe

SparkPipe is a model-aware C/CUDA runtime for distributed inference across a
fabric of NVIDIA DGX Spark nodes. It makes the execution contract explicit:
stage and layer ownership, stage-local KV state, activation payloads,
transport, process boundaries, and readiness evidence.

SparkPipe is infrastructure. It owns model execution and the runtime/API
boundary. Applications can add routing, memory, tool use, policy, and UI above
the OpenAI-compatible endpoint; those concerns are intentionally outside this
repository.

## Current status

The repository is simulator-first. The simulator exercises the boundaries that
the native runtime must preserve, but simulator output is not live-model
readiness.

The current integration target is a 13-stage model-aware pipeline. The stage
count and topology are deployment parameters, not permanent product claims.
The real fast-ring endpoint and service, prepared activation payloads, native
CUDA execution, distributed launch, and the first live multi-Spark model
dry-run each require their own evidence before the runtime can report ready.

Until those gates pass, SparkPipe does not present theoretical throughput or
model-serving claims as measured production results.

## Runtime shape

```text
OpenAI-compatible API
        |
coordinator and scheduler
        |
model plan: layers, stages, memory, KV, transport
        |
stage 0 -> stage 1 -> ... -> stage N
        |       |              |
   local KV  local KV     local KV
        \_______ prepared activations over the fast ring ______/
```

The runtime keeps these boundaries visible:

- Model-family drivers own geometry, layer execution, routing, sampling, and
  model-specific state.
- The runtime owns admission, request lifecycle, stage scheduling, residency,
  activation movement, and operational counters.
- Stage-local KV ownership and prepared activation payloads are explicit
  contracts rather than hidden allocations.
- Native C/CUDA hot paths, transport, and service/process boundaries qualify
  independently so each result has a clear meaning.

## Fast-ring policy

The inference path is fail-closed:

```text
No validated 200 Gbps Spark ring
    -> no ready endpoint
    -> no ready service
    -> no ready stage
    -> no activation route
```

Slow control-channel fallback is not an inference transport. A ready result
requires evidence from the path that will actually serve the model.

## Build and test

```sh
make clean
make -j1 all
make test
make trace
make stagepack
make loadsummary
make processdryrun
make filetransportdryrun
make fastringvalidate
make fastringservicedryrun
make ringtransportdryrun
make sparkdryrunbundle
make readinessbundle
make cuda_dummy
```

The default build is pure C. `make cuda_dummy` is optional and skips cleanly
when `nvcc` is unavailable.

## Readiness evidence

The repository includes fail-closed checkers for transport, service,
StagePack, process, trace, CUDA, and live dry-run evidence. For example:

```sh
make readinessbundle
./build/sparkpipe_readiness_bundle_check \
    --bundle docs/readiness_live_sample --format summary
```

The live package is accepted only when the checker emits `ready=1`. Sample
receipts are useful for exercising the checker; they are not proof of a live
Spark deployment.

## Model support

SparkPipe uses one runtime with model-specific drivers. GLM 5.2 is an active
qualification target. Other model-family contracts may exist in the tree, but
each model needs independent numerical, transport, and live readiness evidence
before it should be described as production-ready.

## Development boundary

The local sandbox can exercise host and simulator paths, but it cannot prove
the physical Spark NICs, the CUDA toolchain, live StagePacks, or distributed
launch behavior. The qualification and deployment handoffs live in:

- [`docs/READINESS_BUNDLE_WORKFLOW.md`](docs/READINESS_BUNDLE_WORKFLOW.md)
- [`docs/CUDA_AND_200G_QUALIFICATION_HANDOFF.md`](docs/CUDA_AND_200G_QUALIFICATION_HANDOFF.md)
- [`docs/CODEX_FIRST_SPARK_DRYRUN_TASKS.md`](docs/CODEX_FIRST_SPARK_DRYRUN_TASKS.md)
- [`docs/SANDBOX_BOUNDARY.md`](docs/SANDBOX_BOUNDARY.md)

See [sparkpipe.ai](https://sparkpipe.ai/) for the public project overview.
