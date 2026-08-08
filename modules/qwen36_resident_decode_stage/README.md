# Qwen 3.6 27B Resident Decode Stage

A pipeline-parallel stage driver for Qwen 3.6 27B. One process is one stage
owning a contiguous layer slice; the whole-stack case is a one-stage
pipeline. Every architectural form is pinned: the model header against the
checkpoint config, the GDN and attention math against modeling_qwen3_5, and
the MTP structure against the checkpoint safetensors index (transformers
drops mtp.* on load, so the checkpoint is the only ground truth for it).

Decode microbatches carry one next token for up to 512 distinct lanes. v1
executes the recurrent GDN step, which the carry oracle proves bitwise equal
to any chunked formulation; prefill is the runtime feeding prompt tokens
through the same path. The chunked prefill kernel is a throughput commit,
not a correctness requirement.

## Configuration

Every variable is required. A missing value is a refused Initialize, never a
default.

| Variable | Meaning |
| --- | --- |
| SPARK_QWEN36_STAGE_PACK_PATH | stage pack for exactly this slice |
| SPARK_QWEN36_STAGE_COUNT / SPARK_QWEN36_STAGE_INDEX | pipeline shape and this stage's position |
| SPARK_QWEN36_STAGE_FIRST_LAYER / SPARK_QWEN36_STAGE_LAYER_COUNT | the slice; embedding/head ownership is derived and checked against position |
| SPARK_QWEN36_STAGE_MAX_ACTIVE_SEQUENCES | lane capacity, at most 512 |
| SPARK_QWEN36_STAGE_PIPELINE_SLOTS | concurrent frame scratch sets, at most 4 |
| SPARK_QWEN36_STAGE_KV_BLOCKS | paged KV physical blocks (64 tokens each) |
| SPARK_QWEN36_STAGE_KV_STORE | literal `none`, or the Mooncake provider .so; a path additionally requires SPARK_QWEN36_STAGE_KV_SERVICE, _KV_SOCKET, _KV_POOL_BYTES, _KV_WORKERS |

## Bring-up sequence on a sparkring node

1. `qwen36_pack_synthesize --output /nvme/q36.pack --first-layer F --layer-count N` (add `--bf16` to skip MXFP4; whole stack is 50.9 GiB bf16, 866 tensors).
2. Set the environment for the same slice, `SPARK_QWEN36_STAGE_KV_STORE=none`, and Initialize: the ready line reports the slice, layer split and resident GiB. A wrong pack fails naming the exact geometry field.
3. One decode frame, one row, stage_count 1: token in, argmax token out.
4. 512-row microbatch; then two carried dispatches per lane against one, which must agree bitwise (the oracle proved the math; this proves the plumbing).
5. First on-device oracle: the GDN step kernel against the CPU recurrence in `validation/spark_qwen36_reference.c` at real head dims.
6. Multi-stage: two processes, slices F..k and k..64, hidden transport between them; middle frames must carry both transport flags or they are refused.
7. Enable the Mooncake tier by pointing SPARK_QWEN36_STAGE_KV_STORE at the provider; keys are fingerprinted against the slice geometry and cache layout, so a rebuilt pack cannot read stale KV.

## Gates already run (container, no GPU)

Both translation units compile warning-free (nvcc sm_121a, cc -Werror);
every launcher symbol the module imports resolves against the CUDA object;
the format probe holds 866/67 tensor inventories with layer-class
enforcement; the synthesizer byte accounting closes on the 27.32B parameter
total; all five CPU oracles pass with the carry test bitwise exact; the KV
client probe verifies the disabled tier and the key format; every function
is at or under fifty lines.

## Open by design

The chunked prefill kernel (throughput only); the MTP draft/verify execution
against the tree engine (weights load and verify today); the work-control
residency layer that drives the KV client with the glm52 JIT discipline
(lookahead, pressure limits, packet-zero priority); the tensor-core decode
attention. Each is a bounded follow-up commit on this branch, none blocks
bring-up.
