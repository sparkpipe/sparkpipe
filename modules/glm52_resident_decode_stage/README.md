# GLM52 resident decode-stage firmware

This directory is an exact model-specific CUDA firmware link unit. It is not a generic serving backend and it is not a reusable graph interpreter.

The fixed stage program is specialized for SM121 GLM 5.2 BF16 decode with 6144 hidden elements, 64 MLA heads, 512 latent elements, 64 adjacent-pair RoPE elements, 2048 selected context tokens, 64-token KV blocks, a 256-token restricted vocabulary head, and depth-2 MXFP4 MTP draft verification.

One submission executes this stream-ordered sequence:

```text
attention RMSNorm
BF16 Q latent / Q RoPE / K RoPE / KV latent projections
driver-owned sparse-token selection
RoPE + final-layout current KV write
resident sparse MLA attention
attention output projection + residual
final RMSNorm + restricted-vocabulary logits + argmax
MXFP4 E2M1/E8M0 MTP draft logits + argmax
MTP verify / commit / rollback counters
external completion
```

The node context binds resident weight pointers, paged KV cache, streams, workspaces, RoPE tables, token maps, and output buffers once when the driver instance is created. Per-submission inputs are only dynamic decode facts such as active sequence count, requested token count, sequence identity, deadline, priority, and residency token. The firmware admission function chooses the opaque pipeline slot; SparkPipe does not assign or interpret CUDA stream/KV ownership.

The module also publishes direct admission and snapshot symbols. They expose only neutral scheduling data: accepted/rejected, dispatch slot, dispatch generation/cookies, private queue pressure, resident token capacity, active submissions, CUDA graph capture/replay counts, stale-admission count, and zero memcpy/host-staging counters.

Sparse-token selection has three modes. `preselected` is the intended production path: DSA or another model-specific policy produces sparse rows before graph replay. `copy_context_prefix` is a parallel bring-up path. `debug_serial_topk` is deliberately not a production mode.

Each pipeline slot may hold one captured CUDA graph for its fixed active-sequence shape. Production launch checking should be disabled; peek/sync modes exist for target bring-up. Completion is enqueued after the graph launch so the arithmetic graph stays reusable and the completion mechanism can later move to event polling or a doorbell without changing the captured graph.

Normal publication validates a new archive exactly once:

```sh
make -C modules/glm52_resident_decode_stage publish \
    CUDA_ARCH=sm_121 \
    MAX_STAGE_MICROSECONDS=<qualified-limit>
```

The source is correctness-first until hardware profiling says which fused pieces should be replaced by tensor-core or persistent-kernel implementations. It must not be published unless the hardware validator passes the numerical checks and the maximum full-stage submission-to-completion latency ceiling.
