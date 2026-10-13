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
| Cache | KV page cache, snapshot tier and prefix cache; the stage KV binding attaches the weightd-owned KV pools | `cache/`, `runtime/stage_kv_binding.c` |
| Text | Tokenizer and prompt template primitives | `text/` |
| Model Families | Generated model facts and host-side family contracts | `model-families/`, `model_contracts/` |
| Stage runner | The one TP stage runner every driver runs on (weight attach through weightd, the TP collective, CUDA graphs, KV attach, decode chains, lanes) and the serving adapter that wraps it | `inference/runner/`, `include/sparkpipe/spark_stage_runner*.h`, `include/sparkpipe/spark_stage_serving_adapter.h`, `model-families/common/include/sparkpipe/` |
| Stage models | Each family's math behind the runner's model interface (open, bind, step, KV attach); no system code | `model-families/<family>/source/`, `inference/llms/kimi_k3/` |
| Family templates | Code shared across forked model families ([FAMILY_TEMPLATES](FAMILY_TEMPLATES.md)) | `include/sparkpipe/family/` |
| Kernels | Shared compile-time CUDA mechanisms and formats ([KERNEL_PLAYBOOK](KERNEL_PLAYBOOK.md)) | `inference/kernels/` |
| Model Modules | Per-driver serving glue (KV region layout, adapter descriptor, topology) and the stage-model instantiation, built into one serving adapter library | `modules/*_resident_decode_stage/` |
| Speculation | Speculation seam, draft sources, tree and policy; the RTX 5090 drafter node | `include/sparkpipe/spark_speculation_*.h`, `src/spark_speculation_*.c`, `deployment/rtx5090_speculation/` ([RTX5090_SPECULATION_NODE](RTX5090_SPECULATION_NODE.md)) |
| Node | One resident process per rank, the generic batch client, and the HTTP API process | `node/model_residentd.c`, `node/model_batch.c`, `node/model_api.c` |
| Deployment | Built and published release roots, and node convergence by MANIFEST diff ([FLEET_RELEASE_RUNBOOK](FLEET_RELEASE_RUNBOOK.md)) | `deployment/`, `tools/fleet_release/`, `scripts/release_rollout.sh`, `tools/fleet_release_serve.py`, `tools/fleet_node_agent.sh`, `tools/fleet-agent.service` |
| Chat layer | OpenAI-compatible chat front end over the C API (chat templates, tool and reasoning parsers) and the model swap controller | `tools/serving/` |
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

- a model contract under `model_contracts/` and its generated constants;
- a stage model implementing `include/sparkpipe/spark_stage_runner_model.h`
  under `model-families/<family>/source/`;
- a module directory with its serving glue (KV region layout, adapter
  descriptor) and the stage-model instantiation;
- a stage packer built on `tools/spark_pack_common.py`;
- a `tools/stage_lane.py` lane entry and a chat-layer model entry.

It uses the common stage runner, resident, weightd, deployment, transport,
pipeline, batch and release interfaces unchanged.

A new codec extends the generic codec ABI and the CUDA format traits, and
adds pack and decode tests. An explicit model package then selects it. It
must never become a common default, an environment switch, a fallback, or an
if/else branch in a shipping AOT module.
