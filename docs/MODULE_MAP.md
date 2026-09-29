# SparkPipe Module Map

The production stack stays model-neutral up to the adapter and driver the
package selects. Dependencies point from the generic process toward the model
module, never from common infrastructure back to a model family.

| Module | Contract | Owned Paths |
|---|---|---|
| Core | Status, hashing, filesystem, JSON, module loading | `src/`, selected `runtime/` primitives |
| Model ABI | Resident endpoint, deployment, IPC, pipeline and batch contracts | `include/sparkpipe/spark_model_*`, `runtime/model_*` |
| Weight daemon and mesh | Weight arenas exported read-only through CUDA VMM handles, lazy expert leases, and the all-to-all RDMA mesh wired at boot (README §weightd, §The mesh) | `node/weightd.c`, `node/weightd_mesh.c`, `node/weightd_spawn.c`, `runtime/spark_weightd*.c`, `include/sparkpipe/spark_weightd*.h` |
| Transport | TP collective engine over the weightd mesh | `ring/transport/` |
| Cache | Resident KV, NVMe tier, and Mooncake integration | `cache/`, `modules/kv_mooncake/` |
| Text | Tokenizer and prompt template primitives | `text/` |
| Model Families | Generated model facts and host-side family contracts | `model-families/`, `model_contracts/` |
| Common modules | Parameterized shared modules, configured by each family's `llm_defines.h` ([COMMON_MODULE_ARCHITECTURE](COMMON_MODULE_ARCHITECTURE.md)) | `common/`, `model-families/common/include/sparkpipe/` |
| Family templates | Code shared across forked model families ([FAMILY_TEMPLATES](FAMILY_TEMPLATES.md)) | `include/sparkpipe/family/` |
| Kernels | Shared compile-time CUDA mechanisms and formats ([KERNEL_PLAYBOOK](KERNEL_PLAYBOOK.md)) | `inference/kernels/` |
| Model Modules | Package-selected adapter, immutable AOT driver, stage lifecycle | `modules/*_resident_decode_stage/` |
| Speculation | Speculation seam, provider, tree and policy, plus model-owned draft modules; the RTX 5090 drafter node | `include/sparkpipe/spark_speculation_*.h`, `src/spark_speculation_*.c`, `runtime/speculation_provider.c`, `modules/glm52_dspark_draft_backend/`, `deployment/rtx5090_speculation/` ([RTX5090_SPECULATION_NODE](RTX5090_SPECULATION_NODE.md)) |
| Node | One resident process per rank, the generic batch client, and the HTTP API process | `node/model_residentd.c`, `node/model_batch.c`, `node/model_api.c` |
| Deployment | Built and published release roots, and node convergence by MANIFEST diff ([FLEET_RELEASE_RUNBOOK](FLEET_RELEASE_RUNBOOK.md)) | `deployment/`, `tools/module_build_release.sh`, `tools/fleet_release_serve.py`, `tools/fleet_node_agent.sh`, `tools/fleet-agent.service` |
| Qualification | Source, host, CUDA, hardware, and evaluation gates | `tests/`, `qualification/`, `tools/hardware/` |

**Host roles.** These come from `docs/FLEET_RELEASE_RUNBOOK.md` §1 and the
lead dev's 2026-09-28 facts:

- **Build host: sparkf.** It builds for aarch64 and sm_121a in
  `~/g5n-rd-build`, and is also rank 15.
- **Hub: rtx5090.** It serves the release roots `~/release/<root>/` over
  HTTP on port 8802 and collects the heartbeats in `~/current/<host>.json`.
- **Every Spark** runs `tools/fleet_node_agent.sh` as the systemd user unit
  `fleet-agent`. weightd and residentd run inside that unit's cgroup, so
  restarting the agent restarts both.
- **No hand deploys.** Releases go through the runbook.

## Hard Boundaries

1. Common runtime code never names a model family or selects a codec.
2. A package binds one adapter, one driver target, one stage pack, and one
   codec tuple. Startup rejects every identity mismatch.
3. Model CUDA specializes the package codec at compile time. There is no
   runtime codec branch or production fallback kernel.
4. **weightd** owns the weight arenas and the RDMA mesh for its node. The
   **resident** attaches to both, and owns its stage's other CUDA
   allocations, scheduling progress and request state for its rank. There
   is no second model-specific resident or rank daemon.
5. Batch execution enters through `spark_model_batch_engine` and resident IPC.
   Model-specific gateways and schedulers are not part of the runtime.
6. Model-owned packers translate source weights into rank-local immutable
   packs. The common deployment layer treats their contents as opaque.
7. Generated contracts and deployments have one editable source plus a
   byte-exact `--check` gate. Generated outputs are not independent facts.

## Adding A Model Or Codec

A new model family provides:

- generated constants and its `llm_defines.h`;
- a model description;
- a serving adapter;
- a stage module;
- a stage packer;
- one or more immutable AOT driver targets.

It uses the common resident, weightd, deployment, transport, pipeline, batch
and release interfaces unchanged.

A new codec extends the generic codec ABI and the CUDA format traits, and
adds pack and decode tests. An explicit model package then selects it. It
must never become a common default, an environment switch, a fallback, or an
if/else branch in a shipping AOT module.
