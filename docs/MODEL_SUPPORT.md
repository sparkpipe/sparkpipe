# Model Support Contract

SparkPipe targets all open-source frontier-level models that can be mapped to a
supported deployment. The runtime is shared; every model has an exact,
checkpoint-derived execution package.

## Product model set

| Family | Variants | Typical placement |
| --- | --- | --- |
| DeepSeek V4 | Flash, Pro | TP4 x PP4 on sixteen Sparks for large resident execution |
| GLM | GLM 5.2 | model-profiled TP/PP placement |
| Kimi | K3 | model-profiled TP/PP placement |
| MiniMax | H3 | model-profiled TP/PP placement |
| Qwen | 3.8 Pro, 3.8 27B | global TP for smaller dense variants or profiled large-model placement |

MiniMax 2.5 is not a support target.

The catalog is not capped at this table. A new open-source frontier model joins
the product set by satisfying the same package and qualification contract; it
does not require a second serving stack.

## Package boundary

Every supported checkpoint binds:

- upstream model ID, exact revision, and source implementation revision;
- tokenizer and prompt-template identities;
- model geometry, layer kinds, routing, attention, recurrent state, and head;
- weight, activation, accumulation, KV, and scale formats;
- rank-local sharding, PP slices, TP communicators, and storage placement;
- exact native CUDA modules and graph geometry;
- speculation provider and verification contract when enabled; and
- numerical, memory, transport, and performance qualification identities.

Common runtime code never infers one model's constants from another model and
never treats a nearby architecture as compatible because tensor names happen to
match.

## Qualification

A model is ready only when the exact checkpoint and deployment have retained:

1. package and source identity;
2. host contract and bounds tests;
3. exact target CUDA compile and link receipts;
4. real-weight pack validation;
5. numerical comparison with the authoritative implementation;
6. physical route and collective evidence;
7. deterministic prompt, streaming, stop, cancellation, and error behavior;
8. latency and throughput measurements under named request shapes; and
9. clean merged-main release and all-rank ready identity.

A model-family directory, simulator, analytical estimate, successful pack, or
compiled kernel is evidence for its own domain only. None is a production-ready
claim by itself.

## Residency and promotion

The endpoint keeps the complete configured catalog addressable. Models may be
resident, warm, promotable, or unavailable. Promotion from the verified model
store to ready state has a maximum target of one minute and includes shard
placement, binding, prewarm, communicator creation, all-rank agreement, and
atomic publication.

Co-resident models share hardware through priority-aware gang scheduling. A
model switch changes the selected execution plan; it does not change the API or
require applications to reconnect.

Open implementation gaps live in [`../TECHDEBT.md`](../TECHDEBT.md). Measured
model results live in [`../PERFORMANCE_STATUS.md`](../PERFORMANCE_STATUS.md).
