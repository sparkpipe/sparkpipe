# GLM52 resident sparse MLA firmware

This directory is one exact firmware link unit, not a generic attention backend.
It compiles the following fixed program into one static archive:

```text
adjacent-pair query RoPE
+ adjacent-pair key RoPE
+ current-token latent/key placement into the final paged BF16 MLA cache
+ 64-head sparse cached attention over 2048 selected tokens
+ one stream-ordered external completion
```

The fixed geometry is 64 heads, 512 latent elements, 64 RoPE elements, 64 tokens per KV block, and 2048 selected tokens. The node binding supplies resident device pointers, streams, capacities, block tables, positions, RoPE tables, CUDA graph slot state, and the model scale once at driver creation. A submission supplies only active sequence count; the admission function chooses the opaque dispatch slot and protects it with a dispatch generation.

The module publishes direct admission and snapshot symbols, reports zero host-staging/device-memcpy counters, and supports optional per-slot CUDA graph capture/replay. It is still a primitive firmware module; the full decode-stage module is the preferred production boundary when projections, logits, MTP, MoE, and transport are fused into the same model-specific archive.

Normal publication validates a new archive exactly once:

```sh
make -C modules/glm52_resident_sparse_mla publish \
    CUDA_ARCH=sm_121 \
    MAX_STAGE_MICROSECONDS=<qualified-limit>
```

An unchanged archive, symbol contract, target, recipe, and latency limit reuse the existing passing record. Driver compilation and service startup do not rerun the validator.
