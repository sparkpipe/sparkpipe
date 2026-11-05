# Family schedules

A model family's layer schedule is written once and runs on every backend.
CUDA and Metal differ in how they carry out an operation, never in which
operations a layer runs or in what order.

## The three parts of a family

| Part | Where | Shared by |
| --- | --- | --- |
| Geometry: every number of one checkpoint | the model's module header, `modules/<model>/include/sparkpipe/*_stage_model.h`, as `template<uint32_t TP> struct <Model>Geometry` | CUDA (`<Model>Geometry<16>`) and Metal (`<Model>Geometry<1>`) |
| Layout and schedule: which layers are which, and the order of operations in each | `model-families/<family>/include/sparkpipe/spark_<family>_family.h`, device-neutral C++ | every backend |
| Operations: how one step of the schedule runs | one backend class per family and device | one device |

The schedule is a template on the geometry `G` and a backend `B`, for example
`GemmaSchedule<G, B>::Step` and `QwenHybridSchedule<G, B>::Step`. It calls
`B` for every operation: `B::Begin`, `B::RmsNorm`, `B::Attention`,
`B::AllReduce`, `B::LayerDone` and so on. It holds no device type, no kernel
and no launch; `tests/test_device_neutral.py` keeps device headers out of it.

The backends:

| Family | CUDA backend | Metal backend |
| --- | --- | --- |
| Gemma 4 (Gemma 4 26B and 31B, K-EXAONE 2.0, Laguna S 2.1, Muse Glimmer 30B) | `GemmaModel<G>` in `model-families/gemma4/source/gemma4_stage.cuh` | `GemmaDevice<P>` in `model-families/gemma4/device/gemma4_device_stage.cpp` (Gemma 4 26B and 31B, Muse Glimmer 30B) |
| Qwen 3.8 hybrid (27B, Flash, Max) | `QwenHybridModel<G>` in `model-families/qwen38/source/qwen38_hybrid_stage.cuh` | `QwenDevice<P>` in `model-families/qwen38/device/qwen38_device_stage.cpp` (27B) |
| MiMo 2.6 (Flash, Pro) | `MimoModel<G>` in `model-families/mimo26/source/mimo26_stage.cuh` | not yet |
| Ling 3 (Ling 3 Flash and Fin, Solar Open 2) | `LingModel<G>` in `model-families/ling/source/ling_stage.cuh` | not yet |
| HY4 | `HyModel<G>` in `model-families/hy4/source/hy4_stage.cuh` | not yet |
| GLM 5.3 Flash | `Glm5NextStage<G>` in `model-families/glm5_next/source/glm5_next_stage_model.cuh` | not yet |
| GLM 5.3 Full (GLM 5.3 Full, A.X K2) | `GlmModel<G>` in `model-families/glm52/source/glm52_stage.cuh` | not yet |
| K3 | `K3Cuda<Format, Geometry>` in `inference/llms/kimi_k3/slice.cuh` | not yet |
| DeepSeek V4.1 Flash | `D41Cuda` in `modules/dsv41_flash_resident_decode_stage/source/spark_dsv41_flash_stage_model.cu` | `D41Device<TP>` in `model-families/dsv41_flash/device/dsv41_flash_device_stage.cpp` (dense, below position 512) |

A backend defines `State`, `Frame` (the per-step handles: the CUDA stream and
step, or the Metal queue and step tables), `Layer` (the bound weights of one
layer, with the field names the schedule reads) and one static function per
operation. Operations the backend does not have yet are simply absent: the
schedule reaches them only through `if constexpr` on the geometry, and the
Metal backends `static_assert` the geometry switches they do not cover, so a
model that needs a missing operation does not compile for that device.

## What stays per backend

- The kernels and their launch shapes, as `METAL_BACKEND.md` sets out.
- Memory: arena planning, weight binding (weightd lazy packs on CUDA, the
  mapped rank pack on Metal), KV views, lanes and graphs.
- Collectives. `B::AllReduce` is the mesh round on CUDA and the runner's
  `round` on Metal (`SparkDeviceStageAllReduce`, nothing on one rank); both
  reduce in rank order with one BF16 rounding. The transport between Metal
  ranks is the runner's `SparkDeviceCollective` (in-process today,
  Thunderbolt between Studios). How a backend splits attention over ranks is
  its own choice: CUDA is context-parallel, Metal head-parallel
  (`METAL_BACKEND.md`, "Tensor parallel ranks").

GLM 5.3 Flash's numbers come from the contract-generated
`spark_glm5_next_model.h`, which both backends read. The GLM 5.3 Full and A.X
K2 geometries stay in their `.cu` files until the rope pairing
(`LmRopePairing`, in `attn.cuh`) and the stagepack geometry hooks have a
device-neutral home. The GLM 5.3 Full schedule walks the waves its backend
chooses; wave choice, gather or scatter attention and row sharding are the
CUDA backend's way of running a step.

K3 has three orderings in one `K3Schedule<B>`: `Slice` (one lane), `SlicePair`
(two row halves on two streams, the long-prefill path) and `SliceHalf` (one
layer phase). All three take the layer kind, the dense or routed MLP, the
attention-residual sources and the block boundary from `K3Layout`; the
paired ordering adds fences and collective rounds, which are backend
operations, and keeps the enqueue order the pipelining depends on. A
backend without streams runs `Slice` and never instantiates `SlicePair`.

DeepSeek V4.1 Flash's layer map (compression ratio, KV and index sources,
cache owners, engram layers) lives in its family header; the rows that
later layers skip ("tail" rows) and the candidate-source compress are the
CUDA backend's way of running a step.

Both backends bind the same pack tensor names (`layers.N.attn_qkv`,
`layers.N.ffn_gate_up`, ...), and the Metal packs are the CUDA packers run
with `--tp-degree 1`.

## Adding a model or a device

- A model of an existing family is one geometry struct in its module header
  and one instantiation per device: the CUDA module's `*_stage_model.cu`
  and, for Metal, a placement struct (tag, name, projection family) next to
  the family's Metal backend.
- A Metal port of a model whose family schedule exists needs only the Metal
  operations its geometry switches on; it never copies the schedule.
- A new family adds its `spark_<family>_family.h` first, then one backend per
  device.

## Moving a family onto its schedule

A family moves without changing a computed value on any device:

- The CUDA kernels must keep their machine code. Every serving adapter of
  the family is built before and after on the hub (`sm_121a`), every
  kernel's `.text` section is extracted with `cuobjdump -xelf` and compared
  by demangled name (the geometry's TP template argument is part of the
  mangled name). A kernel that the schedule no longer instantiates for a
  model, because the model never launched it, is listed.
- The Metal stages must produce byte-identical output on the gate prompts:
  teacher-forced logits, greedy tokens, a 1/4/16-row ladder and, for a model
  with recurrent state, speculation.
- The CUDA launch order is the old `Step` moved into the schedule; each
  driver is qualified on the fleet before its release.
