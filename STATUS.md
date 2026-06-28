# SparkPipe status

## Implemented

- One low-level model-description JSON input contract.
- One-shot compilation of every model stage from one JSON parse.
- Exact stage-target and module-ID resolution.
- Content-addressed validated firmware link-unit library.
- Relocatable-object and normal-static-archive module artifacts.
- Explicit rejection of thin archives because they are not self-contained.
- Validator execution only for a new artifact contract; identical publication reuses the passing record.
- Ordered validator contract arguments included in the validation identity, so a changed tolerance or latency ceiling causes one new validation.
- Stored-byte and link-unit-kind verification before and after validation.
- Read-only library artifacts and separate read-only package copies.
- Direct collection of only selected link units.
- Generated C with explicit operation calls and no hot-path operation loop.
- Direct archive linking, including extraction of private helper members required by the firmware entry point.
- One model-driver shared object per stage with one stable scheduler ABI.
- Hidden module and archive-helper symbols; only `SparkModelDriverGetInterface` is public.
- One package receipt embedding the source JSON and complete resolved artifact map.
- Fail-closed package transaction that removes stale or partially rebuilt firmware.
- Separate offline compiler and serving-runtime archives.
- Driver loader with one-time ABI and target binding.
- Heterogeneous-target orchestrator with pre-resolved numeric routes, firmware-identical replicas, driver admission decisions, opaque dispatch-slot handoff, per-program inflight accounting, runtime snapshots, and completion retirement.
- Tests proving compilation, loading, routing, and execution do not rerun validators.
- First exact CUDA firmware source under `modules/glm52_resident_sparse_mla/`.
- Fixed GLM 5.2 BF16 resident sparse-MLA boundary with adjacent-pair query/key RoPE, current-token placement into the final paged KV layout, sparse cached attention, and stream-ordered completion.
- Correctness-first GLM 5.2 resident decode-stage CUDA firmware source under `modules/glm52_resident_decode_stage/`.
- Decode-stage CUDA covers attention RMSNorm, BF16 Q/KV projection kernels, native DSA sparse-token selection, RoPE plus final-layout KV write, sparse MLA, attention output projection, residual, final RMSNorm, restricted-vocabulary logits, MXFP4 E2M1/E8M0 MTP draft logits, verify/commit/rollback counters, phase clock markers, and stream-ordered completion.
- Decode-stage CUDA also has explicit live BF16 layer-0 checkpoint fixture gates for GLM 5.2's first dense layer. The live validator can load layer-0 attention tensors, dense `post_attention_layernorm` / `gate_proj` / `up_proj` / `down_proj` tensors, and restricted `lm_head.weight` rows from the NVFP4 checkpoint, then run them through the resident CUDA stage before final restricted logits.
- The GLM resident decode-stage module owns `artifact_check`, a module-local fail-closed gate for live GLM `config.json`, model-description JSON artifact geometry, raw safetensors tensor contract, resident decode-stage firmware constants, and model-description route agreement. This is intentionally outside the generic SparkPipe tool surface.
- The GLM resident decode-stage `artifact_check --check-body-samples` mode seeks into live safetensors tensor bodies and hashes deterministic tensor samples plus the restricted-vocabulary `lm_head.weight` rows. This proves the accepted artifacts are not just shape/header matches.
- The target-hardware decode-stage validator can run with `GLM52_MODEL_DIR=<artifact>` to load real BF16 restricted `lm_head.weight` rows into the resident CUDA buffer and check every CUDA restricted logit against a CPU reference using the same 256-lane reduction order as the kernel.
- Driver ABI v3 adds optional direct module admission and runtime snapshot symbols, program scheduling profiles, opaque dispatch-slot validity, dispatch generations/cookies, residency tokens, CUDA graph replay counters, stale-admission counters, and no-host-staging/device-memcpy counters without exposing LLM internals to SparkPipe.
- Model-specific node binding for caller-owned streams, resident device buffers, paged KV storage, RoPE tables, CUDA graph slots, and capacities; the sparse-MLA and decode-stage drivers choose opaque dispatch slots through admission functions and report zero memcpy/host-staging counters plus private queue pressure.
- Cold-build module workflow with dependency files and self-contained archive creation.
- Target-hardware sparse-MLA validator source covering nonzero first-block offset, block-table remapping, fused KV write, RoPE, full 64-head output comparison, completion semantics, and a mandatory maximum submission-to-completion wall-clock threshold.
- Target-hardware decode-stage validator source covering full-shape resident device buffer allocation, deterministic nonzero context-length-4 fixture tensors, two-block KV remapping, two checked attention heads/dimensions, cached attention host reference, restricted argmax, MTP accept/reject counters, latency ceiling, and post-package generated-driver/orchestrator submit.
- End-to-end host control-path test proving one-time publication, validation reuse, complete JSON-to-package compilation, hidden archive symbols, direct driver execution, slot ownership, asynchronous completion, and quiescent destruction.
- Remaining iteration-073 CUDA implementation work preserved outside the active build as candidate source.

The active documentation authorities are:

```text
README.md                              use and build flow
SPEC.md                                production architecture contract
STATUS.md                              current implementation boundary
docs/LLM_DEVICE_DRIVER_INTERFACE.md    scheduler/driver handoff contract
docs/CUDA_FIRMWARE_SOTA_METHODS.md     CUDA firmware handoff methods
docs/HANDOFF_RELEASE_079.md            release-specific handoff summary
```

## Removed

- overlapping execution-plan types;
- generic graph and adapter execution;
- readiness bundles and inference gates;
- live blocker registries;
- runtime qualification-file scanning;
- production fallback paths;
- universal kernel registries;
- monolithic linking of every implementation;
- per-iteration architecture and remaining-items paperwork.

## Current boundary

- The sparse-MLA and decode-stage CUDA firmware archives have been compiled with CUDA 13.0 for `sm_121` on a GB10 Spark node, admitted by hardware validators, published into the module library, compiled into generated drivers, and loaded from packaged `model_driver.so` outputs.
- The live GLM 5.2 NVFP4/FP8 artifacts on Spark report `hidden_size=6144`, `num_attention_heads=64`, `kv_lora_rank=512`, `qk_rope_head_dim=64`, and `index_topk=2048`; the active decode-stage firmware identity is therefore `h6144.h64.d512.r64.k2048`, not the earlier incorrect `h8192` shape.
- Live GLM run setup should call `make -C modules/glm52_resident_decode_stage artifact_check GLM52_MODEL_DIR=<GLM-5.2 model dir> ARTIFACT_CHECK_ARGS=--check-body-samples` before compiling or launching a resident decode-stage package. The model-description JSON owns the expected HF artifact geometry and first raw tensor contract; the firmware header owns the compiled CUDA module geometry. A stale model description, wrong checkpoint family, wrong raw tensor layout, unreadable tensor body, or wrong compiled firmware geometry is a hard failure.
- The decode-stage package target additionally runs a generated-driver/orchestrator submit smoke test after package compilation. This proves driver load, route resolution, admission, CUDA backend submit, stream-ordered completion, runtime snapshot counters, and zero host-staging/device-memcpy accounting through the LLM driver boundary while checking deterministic nonzero tensor outputs over a remapped two-block KV layout.
- The sparse-MLA module covers resident sparse MLA plus fused RoPE/current-token KV placement.
- The decode-stage module fills the first CUDA gaps around raw BF16/FP8 projections, native DSA selection, explicit key-nope/value cache writes from `kv_b_proj`, value-head cached attention, real `[6144,16384]` `o_proj`, restricted logits, dense layer-0 BF16 MLP progression, local MoE progression, and MTP draft/verify, but it is still correctness-first code. The current validator proves one deterministic nonzero remapped cached-attention/restricted-logit/MTP path with four checked heads, eight checked key/value dimensions, four checked RoPE dimensions, non-identity RoPE, value-cache-fed attention-output projection, a layer-body output, and the MTP accept/reject contract. With `GLM52_MODEL_DIR` set, it additionally proves the restricted logits CUDA kernel against live checkpoint `lm_head.weight` rows for the NVFP4 and FP8 artifacts. With `GLM52_INPUT_TOKEN_ID=<id>`, `GLM52_LOAD_LAYER0_ATTENTION_BF16=1`, and `GLM52_LOAD_LAYER0_DENSE_BF16=1`, it loads one real BF16 `embed_tokens.weight` input row, 330056704 bytes of real NVFP4 layer-0 BF16 attention weights, 452997120 bytes of real layer-0 BF16 dense MLP weights, and runs the layer-0 BF16 path before restricted logits. It is not full GLM logits equivalence across the complete checkpoint-backed layer stack.
- Restricted logits are treated as an exact model-driver program and final-head optimization. They may avoid full-vocabulary projection, full-vocabulary logits buffers, host sampling, and generic masks, but they do not remove attention, KV, MoE, residual, norm, routing, or other transformer-body work. Thinking-capable generation should use broad-vocabulary programs for reasoning and switch to final restricted or classifier programs only for answer emission.
- The first raw tensor contract passes on the NVFP4 artifacts whose attention and lm_head tensors are BF16. The model description also carries an FP8 E4M3 q/kv/o projection contract for the zAI FP8 artifact; artifact acceptance plus checkpoint-backed restricted-lm-head validation proves shape/dtype/body readability and one real-weight logits kernel, not full decode correctness.
- Resident transport handoff and tensor-core/persistent-kernel optimization remain outside the new decode-stage archive. CUDA graph state and replay hooks are present, but target execution and graph-update debugging remain to be done on hardware.
- Publication is intentionally impossible without a user-supplied maximum full-stage latency and a target-hardware pass; a slow implementation must be optimized, not accepted because it is numerically correct.
- The orchestrator currently manages local driver instances. Remote node agents and wire transport are unfinished.
- The JSON selects exact prebuilt modules. A checkpoint plus deployment-profile importer that emits this low-level language is not implemented.
- Inter-stage deployment topology and fixed buffer contracts are not yet compiled automatically.
- Package signing and immutable deployment activation are not implemented.
- The Mac development environment has neither `nvcc` nor compatible CUDA hardware. CUDA claims must come from the Spark hardware validators.

## Next engineering sequence

1. Add varied sparse-token selection and multi-position KV read/write checksums across several remapped blocks.
2. Add checkpoint-derived CPU reference checks for the real layer-0 BF16 q/kv/o and dense MLP path instead of only checking nonzero progression plus final restricted-logit consistency.
3. Replace seeded prior KV blocks with checkpoint-derived prefill/KV fixtures; the B1 input hidden can now come from a real checkpoint embedding row via `GLM52_INPUT_TOKEN_ID=<id>`.
4. Add real router/top-k and checkpoint expert weights for sparse layers after dense layers 0-2.
5. If the latency ceiling fails with real tensor fixtures, replace the correctness-first projection, DSA, attention, logits, and MTP kernels with measured tiled, tensor-core, persistent, or graph-captured implementations; do not publish the slow artifact.
6. Split GLM driver programs explicitly into broad decode, final restricted decode, and classifier decode profiles so restricted logits stay in the model-driver final-head path rather than becoming a SparkPipe-wide mask.
7. Add graph capture, transport handoff, and checkpoint-derived multi-layer fixtures inside one or a few model-specific GLM firmware archives.
8. Publish only exact archives that pass numerical and model-stage performance qualification, then compile the GLM model JSON into direct-call drivers.
9. Measure end-to-end stage latency and throughput before deciding whether another module split or fusion is profitable.
10. Add the remote node agent and fixed submission/completion wire path without changing the driver ABI.

## Deferred research backlog

- Track DeepSeek DeepSpec/DSpark as a post-basics speculative-decoding item.
  The public drop is a Python/PyTorch training and evaluation framework, not a
  C/CUDA kernel library. It is still useful as an algorithmic reference for a
  future GLM 5.2 block-draft path: selected target-layer hidden-state taps,
  block proposals, Markov bias correction, confidence-gated proposal length,
  and target verification.
- Do not put DSpark adoption on the critical path for first working GLM 5.2
  inference. The current priority remains deterministic nonzero GLM fixtures,
  real checkpoint tensor loading, cached attention/KV correctness, restricted
  logits, MTP verify/commit checks, and a live smoke test.
- Revisit DSpark only after the baseline GLM resident decode stage produces
  checkpoint-backed logits through the driver boundary. At that point, define a
  GLM-specific DSpark-style drafter contract before writing CUDA: target layer
  IDs, hidden-state export layout, per-slot context cache, block size, Markov
  rank, confidence threshold, verification counters, and performance gates.
