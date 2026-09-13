# GLM52 Prompt Pipeline Input

`tools/glm52_prompt_pipeline_input.py` is the current prompt-facing bridge into
the Spark2 local GLM52 pipeline.

It supports:

```text
--prompt "..."
--prompt-file path
--token-ids 1,2,3,4
--chat
--run-pipeline
```

The script writes:

```text
prompt.txt
prompt_tokens.txt
prompt_tokens.json
prefill_plan.json
prefill_chunks.jsonl
pipeline_env.sh
```

The script now emits two different things:

```text
1. full-prompt prefill plan artifacts
2. narrow compatibility environment for the current local validation pipeline
```

The prefill plan describes the production-shaped prompt work:

```text
prompt token ids [0..N)
    -> prefill chunks over token ids [0..N)
    -> first decode step begins after the full prompt is resident
```

By default the chunk size is:

```text
256 tokens per prefill chunk
16 tokens per KV/prefix-cache block
```

For a 41-token prompt and `--prefill-chunk-tokens 16`, the emitted chunk
manifest is:

```text
chunk 0: offset 0,  token_count 16
chunk 1: offset 16, token_count 16
chunk 2: offset 32, token_count 9, final_prefill_chunk=1
first decode step: after full prompt prefill
```

That is the shape the C request API and scheduler need for:

```text
arbitrary-length prompt tokens
    -> chunked prefill
    -> KV block table reservation
    -> decode continuation
```

The C-side dry-run checker consumes the token file and proves the request API
turns it into prefill dispatches:

```sh
build/sparkpipe_glm52_prefill_dryrun \
    --tokens build/glm52_prompt_prefill_chunk_smoke/prompt_tokens.txt \
    --max-prefill-tokens 16
```

For a 21-token prompt, the expected sequence is:

```text
0   prefill       offset 0    token_count 16   remaining 5
1   prefill       offset 16   token_count 5    remaining 0
2   decode_ready
```

When `GLM52_PREFILL_TOKEN_IDS_FILE` is present, the Spark2 local pipeline gate
runs that same C checker before the decode/tail-window compatibility pipeline:

```text
tools/glm52_spark2_local_pipeline_gate.sh
    -> build/sparkpipe_glm52_prefill_dryrun
    -> prefill_schedule.tsv
    -> require final row kind=decode_ready
    -> run current CUDA local pipeline
```

The gate exports:

```text
glm52_local_pipeline_prefill_schedule=<path>
glm52_local_pipeline_prefill_steps=<count>
glm52_local_pipeline_prefill_tokens=<count>
```

For prefill schedule testing without launching CUDA stages:

```sh
GLM52_LOCAL_PIPELINE_PREFILL_ONLY=1 \
GLM52_PREFILL_TOKEN_IDS_FILE=build/glm52_prompt_prefill_chunk_smoke/prompt_tokens.txt \
tools/glm52_spark2_local_pipeline_gate.sh
```

This is still not claiming CUDA prefill execution. It is a fail-closed bridge
that makes the local pipeline consume and check the SparkPipe C prefill schedule
instead of ignoring prompt prefill artifacts.

The current local execution bridge is still intentionally narrow:

```text
full prompt text
    -> local HF tokenizer or explicit token ids
    -> persisted full token-id artifact
    -> persisted chunked prefill plan artifact
    -> last four tokens become the current validation context
    -> first three tail tokens feed prefill/KV
    -> final tail token becomes GLM52_LOCAL_PIPELINE_INPUT_TOKEN_ID
    -> existing local pipeline gate runs dense-prefix/routed decode
```

The script refuses to run prompts longer than the four-token validation window
through `--run-pipeline` unless `--allow-tail-window-run` is passed. This keeps
the local compatibility path from reporting a fake full-prompt pass.

This is not yet full arbitrary-length prompt prefill execution. The current
validation context is four tokens, so the compatibility path uses a tail window.
This is separate from the production-shaped prefill plan, which covers the full
prompt:

```text
prompt token ids [0..N)
    -> production prefill chunks [0..N)
    -> compatibility validation context [N-4..N)
```

That is still a real improvement over the previous fake prefill behavior, where
the validator invented prior tokens with:

```text
input_token_id - 3
input_token_id - 2
input_token_id - 1
```

The artifact preserves the full token list so the next production step can wire:

```text
prompt token ids
    -> prefill_plan.json / prefill_chunks.jsonl
    -> arbitrary-length chunked prefill frames
    -> KV block table
    -> stage-slice prefill
    -> decode continuation
```

Python is allowed here only as setup/bringup glue. The production target is a C
tokenizer artifact plus C/CUDA request ingestion:

```text
tokenizer.model/tokenizer.json
    -> generated C tokenization tables
    -> SparkPipe request API prompt-token buffer
    -> prefill/decode scheduler
```

The prompt bridge refuses to report a fake full-prompt pass. It labels the
current mode as:

```text
tail_window_prompt_prefill_validation_context
```

## C/CUDA full-prompt handoff added

The tail-window validation guard remains in the Python compatibility bridge, but
there is now a C/CUDA path that can consume the full scheduled prompt-token
windows instead of reducing them to the last four tokens.

New request-API helpers expose the exact scheduled prefill slice for each
dispatch:

```c
SparkRequestApiDescribePrefillDispatch(...)
SparkRequestApiCopyPrefillDispatchTokenIds(...)
```

A production prompt server should use these after
`SparkRequestApiScheduleNext(...)` for prefill dispatches. The copied token
matrix is rectangular and lane-major, so it can be uploaded once per scheduled
prefill step without Python stepping one token at a time.

The SM121 required CUDA surface now has a prompt-prefill device workspace and
stage-slice launcher:

```c
SparkGlm52Sm121RequiredDecodeStageCalculatePromptPrefillWorkspaceBytes(...)
SparkGlm52Sm121RequiredDecodeStageResolvePromptPrefillWorkspace(...)
SparkGlm52Sm121RequiredDecodeStageUploadPromptTokenIds(...)
SparkGlm52Sm121RequiredDecodeStageLaunchPromptEmbeddingGatherBf16(...)
SparkGlm52Sm121RequiredDecodeStageLaunchPromptPrefillMetadataBuild(...)
SparkGlm52Sm121RequiredDecodeStageBuildPromptPrefillFrameView(...)
SparkGlm52Sm121RequiredDecodeStageLaunchPromptStageSliceBulkPrefillFromTokenIds(...)
SparkGlm52Sm121RequiredDecodeStageLaunchPromptStageSliceBulkPrefillFromHostTokenIds(...)
```

That CUDA path performs:

```text
host token-id window from request API
    -> device token-id matrix
    -> BF16 embedding gather
    -> prompt positions / slot mapping / context lengths / per-lane counts
    -> SparkResidentDecodeStagePrefillFrameView
    -> stage-slice bulk prefill
    -> KV-resident decode continuation
```

Raw prompt text still requires a tokenizer source outside this CUDA module. The
important change is that, once token IDs exist, the production runtime no longer
needs to spoon-feed one decode token through Python. The remaining Spark2 work is
nvcc bringup, wiring the server loop to this handoff, and verifying that decode
uses the committed KV blocks after the final prefill dispatch.

## C tokenizer and prompt dry-run handoff

The repository now has a C Hugging Face byte-level BPE tokenizer path wired into
both a standalone tokenizer CLI and the prefill dry-run scheduler:

```sh
build/sparkpipe_tokenize_prompt \
    --tokenizer-json /model/tokenizer.json \
    --prompt "..." \
    --output build/prompt_tokens.txt

build/sparkpipe_glm52_prefill_dryrun \
    --tokenizer-json /model/tokenizer.json \
    --prompt "..." \
    --max-prefill-tokens 256 \
    --write-tokens build/prompt_tokens.txt
```

The dry-run path performs the production-shaped input sequence in one C process:

```text
raw UTF-8 prompt text
    -> C byte-level BPE tokenizer
    -> prompt token-id buffer
    -> SparkRequestApiSubmit
    -> scheduled chunked prefill dispatches
    -> decode_ready only after the full prompt is scheduled/committed
```

This does not make a CUDA correctness claim by itself. It removes the Python
one-token-at-a-time prompt input dependency and gives the Spark2 server a direct
C entry surface for prompt text or explicit token IDs.

The tokenizer is intentionally CPU-side. Byte-level BPE is branchy, table-heavy,
and tiny compared with a 50+ tok/sec GLM52 prefill/decode pipeline. The GPU
should stay occupied with embedding, layer prefill, KV writes, and decode. The
C tokenizer can run ahead on host threads and fill request token buffers while
GPU work for previous prompt chunks is already in flight.

## Prefill throughput and pipelining target

The current planning target is still measured on Spark2, not asserted here. A
1500 tok/sec prefill target means an effective budget of about 0.667 ms per
prefill token across the slowest end-to-end stage path. With chunked prefill,
the important metric is not a single-token loop but the amortized stage-slice
rate after the PP13 pipeline is full:

```text
steady_state_prefill_tok_per_sec ~=
    active_prompt_tokens_per_chunk /
    max_stage_slice_seconds_for_that_chunk
```

The production scheduler should therefore run prompt input as a producer/consumer
pipeline:

```text
host tokenizer thread(s)
    -> token-id ring buffers
    -> request API prefill dispatches
    -> CUDA prompt-prefill H2D/token upload on stream A
    -> embedding + stage-slice prefill on stream B
    -> KV commit
    -> decode_ready when final prompt chunk commits
```

Parallelism that is safe now:

```text
1. CPU-tokenize request N+1 while GPU prefills request N.
2. Batch multiple request prefill lanes when the scheduler emits PREFILL_BATCH.
3. Pipeline PP13 chunks so later stages consume chunk k while earlier stages
   prepare chunk k+1, subject to KV commit ordering.
4. Interleave decode-ready requests with lower-priority prefill chunks when the
   scheduler policy says decode latency is at risk.
```

Dependency that cannot be skipped:

```text
A request's first decode token must wait until that request's final prompt
prefill chunk has committed all required KV blocks.
```

JIT KV prefetch helps cold prefix/decode residency. It is not a substitute for
full prompt prefill of new uncached tokens.

## Native C tokenizer and prompt pipeline pump update

The production-shaped path now has native C pieces instead of relying on Python token feeding:

```text
spark_tokenizer.h / spark_tokenizer.c
    tokenizer.json BPE load
    UTF-8 prompt -> token IDs

spark_glm52_text_prompt.h / spark_glm52_text_prompt.c
    prompt text -> request API submit

spark_glm52_prompt_pipeline.h / spark_glm52_prompt_pipeline.c
    request API dispatch loop
    prefill token-window copy
    KV block table view build
    callback into CUDA prefill/decode execution
```

The Python script remains useful artifact glue, but it is no longer the intended production control loop. The C prompt pipeline pump is the handoff point for the Sparkring server/direct runner to execute:

```text
full prompt text -> C tokenizer -> request scheduler -> chunked CUDA prefill -> decode
```

See `docs/GLM52_NATIVE_PROMPT_INPUT_AND_PREFILL_PIPELINE_20260704.md` for the speed estimate and pipeline plan.
