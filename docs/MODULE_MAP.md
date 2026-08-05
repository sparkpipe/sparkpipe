# SparkPipe Module Map

The production stack is model-neutral until the package-selected adapter and
driver boundary. Dependencies point from the generic process toward the model
module, never from common infrastructure back to a model family.

| Module | Contract | Owned Paths |
|---|---|---|
| Core | Status, hashing, filesystem, JSON, module loading | `src/`, selected `runtime/` primitives |
| Model ABI | Resident endpoint, deployment, IPC, pipeline and batch contracts | `include/sparkpipe/spark_model_*`, `runtime/model_*` |
| Transport | RDMA boundary movement and polling | `ring/transport/` |
| Cache | Resident KV, NVMe tier, and Mooncake integration | `cache/`, `modules/kv_mooncake/` |
| Text | Tokenizer and prompt template primitives | `text/` |
| Model Families | Generated model facts and host-side family contracts | `model-families/`, `model_contracts/` |
| Kernels | Shared compile-time CUDA mechanisms and formats | `inference/kernels/` |
| Model Modules | Package-selected adapter, immutable AOT driver, stage lifecycle | `modules/*_resident_decode_stage/` |
| Speculation | Model-owned DSpark or MTP provider modules | `modules/*draft*`, family contracts |
| Node | One resident process and one generic batch client | `node/model_residentd.c`, `node/model_batch.c` |
| Deployment | Immutable release manifest, install, and activation | `deployment/`, `tools/sparkpipe_release_assemble.py` |
| Qualification | Source, host, CUDA, hardware, and evaluation gates | `tests/`, `qualification/`, `tools/hardware/` |

## Hard Boundaries

1. Common runtime code never names a model family or selects a codec.
2. A package binds one adapter, one driver target, one stage pack, and one
   codec tuple. Startup rejects every identity mismatch.
3. Model CUDA specializes the package codec at compile time. There is no
   runtime codec branch or production fallback kernel.
4. The resident owns CUDA allocation, transport, scheduling progress, and
   request state for its rank. There is no second model-specific resident or
   rank daemon.
5. Batch execution enters through `spark_model_batch_engine` and resident IPC.
   Model-specific gateways and schedulers are not part of the runtime.
6. Model-owned packers translate source weights into rank-local immutable
   packs. The common deployment layer treats their contents as opaque.
7. Generated contracts and deployments have one editable source plus a
   byte-exact `--check` gate. Generated outputs are not independent facts.

## Adding A Model Or Codec

A model family provides generated constants, a model description, a serving
adapter, a stage module, a stage packer, and one or more immutable AOT driver
targets. It uses the common resident, deployment, transport, pipeline, batch,
and release interfaces unchanged.

A new codec extends the generic codec ABI and CUDA format traits, adds pack and
decode tests, and is then selected by an explicit model package. It must not
become a common default, environment switch, fallback, or if/else branch in a
shipping AOT module.
