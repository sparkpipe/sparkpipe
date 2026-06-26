# SparkPipe firmware architecture contract

This is the authoritative production architecture.

## 1. Input language

One model-description JSON is the sole compile-time authority. It names exact stage targets, externally callable programs, and the ordered firmware modules used by each program.

```json
{
  "schema_version": 1,
  "model": {
    "id": "glm.example",
    "revision": "weights-layout-and-program-revision"
  },
  "metadata": {
    "architecture": "glm",
    "checkpoint": "sha256:..."
  },
  "stages": [
    {
      "name": "node_0_decode",
      "target": "cuda.sm121.gb10",
      "programs": [
        {
          "name": "decode",
          "id": 1,
          "max_inflight": 4,
          "completion": "external",
          "scheduling": {
            "flags": [
              "stream_ordered",
              "driver_owns_kv_cache",
              "jit_kv_cache",
              "zero_copy_node_context",
              "private_queue_pressure",
              "no_host_staging",
              "fixed_firmware"
            ],
            "max_active_slots": 64,
            "max_new_tokens": 3,
            "host_staging_bytes_per_submit_ceiling": 0
          },
          "operations": [
            {
              "name": "resident_decode",
              "module": "glm.node0.decode.sm121.profile_a.v1",
              "configuration": {
                "weight_package": "sha256:...",
                "kv_pages": 8192,
                "stream_priority": 0
              }
            }
          ]
        }
      ]
    }
  ]
}
```

Required invariants:

- model ID and revision are non-empty;
- stage names are unique;
- every stage has one exact target;
- program names and nonzero IDs are unique within a stage;
- every program contains at least one ordered operation;
- operation names are unique within a program;
- every operation names one exact module ID;
- `configuration` is embedded unchanged in generated code;
- completion is either `submit_return` or `external`;
- `scheduling` is an optional neutral program contract for route admission, inflight limits, resident ownership, JIT-KV ownership, expected no-staging behavior, and latency/memcpy ceilings.

The language deliberately sits below a universal tensor graph. One operation may be an entire fused decode stage, a persistent CUDA program, a transport-aware pipeline segment, or a reusable primitive. End-to-end measurement determines the boundary.

The machine-readable syntax is `schema/model_description.schema.json`.

## 2. Firmware link-unit library

A module artifact contract consists of:

```text
module ID
exact target
firmware module ABI version
link-unit kind
exact link-unit SHA-256
validation recipe ID
ordered validator contract arguments
optional initialize symbol
required execute symbol
optional admit symbol
optional snapshot symbol
optional destroy symbol
```

A link unit is one self-contained linker input:

```text
relocatable object   <artifact-sha256>.o
normal static archive <artifact-sha256>.a
```

A static archive may contain any number of host objects, CUDA objects, device-link objects, and private helpers. A thin archive is forbidden because its bytes do not contain its members.

Library storage is:

```text
<library>/link_units/<artifact-sha256>.o|.a
<library>/records/<module-target-key>-<validation-key>.json
<library>/active/<module-target-key>.json
```

Publication behavior is exact:

1. identify the supplied link-unit kind;
2. hash and copy its exact bytes into content-addressed storage;
3. verify the stored bytes and kind;
4. run the validator only when this exact artifact contract, including the ordered validator arguments, has no passing record;
5. reject a validator that changes the stored bytes or kind;
6. make the passing link unit read-only;
7. atomically activate that contract for its module ID and target.

An unchanged artifact contract reuses its passing record. A changed link unit, target, ABI, recipe, ordered validator argument, or symbol contract gets a new identity and is validated once. Model compilation, driver loading, route creation, and request submission never invoke module validation.

Validation records are offline library indexes. They are not runtime readiness authorities.

## 3. Firmware module ABI

Each selected link unit supplies unique C-linkage symbols recorded at publication:

```c
SparkStatus ModuleInitialize(
    const SparkFirmwareModuleConfiguration *configuration,
    const SparkFirmwareModuleHostServices *host_services,
    void **module_state);

SparkStatus ModuleExecute(
    void *module_state,
    SparkModelDriverFrame *frame);

SparkStatus ModuleAdmit(
    void *module_state,
    const SparkModelDriverAdmissionRequest *request,
    SparkModelDriverAdmissionDecision *decision);

SparkStatus ModuleSnapshot(
    void *module_state,
    uint32_t program_id,
    SparkModelDriverRuntimeSnapshot *snapshot);

void ModuleDestroy(void *module_state);
```

Only `ModuleExecute` is required. Initialization runs once per driver instance. Destruction runs once after the instance is quiescent. `ModuleAdmit` and `ModuleSnapshot` are optional direct symbols for firmware that owns private queues, resident KV, CUDA streams, graph instances, or expert scheduling pressure.

The ABI constrains only the scheduler boundary. A module may own resident weights, KV pages, CUDA graphs, streams, events, workspaces, persistent kernels, transport queues, expert queues, and completion production. It may specialize for exact model revision, tensor shapes, quantization, layout, GPU, stage placement, and deployment profile. Model-specific device resources may be bound once through `node_context`; submissions should carry only genuinely dynamic request data. For an LLM decode firmware module, SparkPipe can ask for admission, dispatch-slot choice, dispatch-ticket integrity, zero-copy/no-staging counters, private queue pressure, graph replay/capture counts, stale-admission counts, and a runtime snapshot without learning the KV layout, MoE queue topology, CUDA graph structure, or token-selection internals.

## 4. Offline model compilation

The deployment compiler parses the model description once and builds every stage in one transaction.

For each stage it:

1. resolves every exact module ID against the stage target;
2. rejects absent, unvalidated, corrupted, or ABI-incompatible artifacts;
3. gathers one read-only copy of every unique selected link unit;
4. generates a stage-specific C orchestrator with direct calls in JSON order;
5. links only the generated code and selected link units;
6. emits the stage driver shared object;
7. hashes the generated program and driver.

A normal archive is passed directly to the linker. Only members required by the generated direct references and their dependencies are extracted. No module registry or operation interpreter is generated.

The complete package contains one build receipt embedding the source JSON and exact resolved artifact map. It is not parsed by the runtime. A failed stage invalidates the package transaction rather than leaving mixed old and new firmware.

## 5. Generated execution path

A generated submit function has this shape:

```c
static SparkStatus SparkGeneratedSubmitProgram_0(
    void *driver_instance,
    SparkModelDriverFrame *frame)
{
    SparkGeneratedDriverInstance *instance;
    SparkStatus execution_status;

    instance = (SparkGeneratedDriverInstance *)driver_instance;
    execution_status = SparkExactModuleAExecute(
        instance->operation_0_state,
        frame);
    if (execution_status != SPARK_STATUS_OK)
    {
        return execution_status;
    }
    return SparkExactModuleBExecute(
        instance->operation_1_state,
        frame);
}
```

Production submit functions contain no graph traversal, module lookup, kernel selection, capability negotiation, validation scan, fallback search, or per-operation indirect dispatch.

The shared object exports only `SparkModelDriverGetInterface`. Module and archive-helper symbols remain internal to the driver image.

## 6. Runtime and orchestrator

The serving runtime contains only the driver loader and orchestrator.

One-time binding checks cover mutable deployment facts:

- driver ABI and descriptor size;
- driver target versus node target;
- exact firmware identity across replicas;
- route and endpoint capacity;
- one-time driver initialization success.

The orchestrator resolves a numeric route to cached program pointers and replica instances. Request submission asks candidate replicas for a neutral admission decision and chooses among accepted endpoints by cost, queue delay, private pressure, zero-copy/no-staging counters, and route capacity. If a driver returns an opaque dispatch slot, the orchestrator writes it into the frame and marks that slot valid; it does not interpret the slot.

The orchestrator does not understand attention, MoE, MTP, KV layout, quantization, CUDA graph topology, expert placement, or JIT-KV policy. Those details belong inside model firmware. The shared boundary is limited to route handles, program descriptors, admission decisions, opaque dispatch slots, dispatch generations/cookies, runtime snapshots, request frames, and completions.

## 7. Validation meaning

“Validated” applies to the exact artifact contract, not to a source filename. The validator is responsible for whatever that module requires, normally including:

- numerical comparison against a trusted oracle;
- target architecture compatibility;
- cold-build completeness;
- model-stage benchmark thresholds;
- absence of forbidden allocation, synchronization, or host staging;
- completion and cancellation behavior.

The publication workflow may deliberately rebuild the candidate link unit from an empty module build directory before offering it to the library. That cold build checks source/header/generated-artifact completeness; it is not a service-startup test.

Once that exact artifact contract passes, using it in another model does not retest it. Changing relevant code or build inputs changes the link-unit bytes and therefore the content identity. Changing a validation tolerance or performance ceiling changes the ordered validator arguments and therefore the validation identity.

## 8. Forbidden production behavior

The active architecture must not add:

- automatic host or reference fallback;
- a universal model-graph interpreter;
- per-request module or kernel selection;
- runtime model/package manifest parsing;
- repeated CUDA-module qualification;
- overlapping readiness, gate, blocker, or plan authorities;
- broad synchronization used to rescue a generic path;
- compatibility wrappers that prevent profitable model-specific CUDA fusion;
- artificial LLM-driver compatibility code that forces KV, MoE, CUDA graph, sparse-index, MTP, sampler, or transport internals into the SparkPipe orchestrator;
- compiler or validator machinery linked into the serving process.
