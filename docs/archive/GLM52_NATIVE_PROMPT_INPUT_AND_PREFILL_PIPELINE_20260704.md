# GLM52 Native Prompt Input and Prefill Pipeline

This pass removes the runtime dependency on Python token feeding for production-shaped prompt ingestion.

## New native tokenizer path

The runtime now has a C tokenizer surface:

```text
include/sparkpipe/spark_tokenizer.h
src/spark_tokenizer.c
```

It loads a Hugging Face `tokenizer.json` BPE vocabulary and merge table once, then encodes UTF-8 prompt text to token IDs in-process. The command-line tools are:

```text
build/sparkpipe_tokenize_prompt
build/sparkpipe_glm52_tokenize
```

`tools/glm52_prompt_pipeline_input.py` now tries the native C tokenizer first when `tokenizer.json` and the C binary are present, then falls back to Python `transformers` / `tokenizers` only as a bring-up convenience. Production code should call the C tokenizer directly.

The GLM52 chat helper in `sparkpipe_glm52_tokenize` renders the current lightweight GLM prompt control-token envelope before tokenization. It is a native fallback for Spark2 environments that do not have Python tokenizer packages installed. The Sparkring handoff should still validate it against the exact model `apply_chat_template` output before declaring chat-template parity.

## New C prompt pipeline pump

The runtime now has a C prompt pipeline pump:

```text
include/sparkpipe/spark_prompt_pipeline.h
src/spark_glm52_prompt_pipeline.c
```

It owns the scheduler loop shape that should replace Python one-token feeding:

```text
SparkGlm52RequestApiSubmitTextPrompt(...)
SparkPromptPipelineRun(...)
    SparkRequestApiScheduleNext(...)
    if prefill:
        SparkRequestApiDescribePrefillDispatch(...)
        SparkRequestApiCopyPrefillDispatchTokenIds(...)
        SparkRequestApiBuildDispatchKvBlockTableView(...)
        prefill_function(context, prefill_dispatch)
        SparkRequestApiCompleteDispatch(...)
    if decode/verify:
        decode_function(context, dispatch)
        SparkRequestApiCompleteDispatch(...)
```

The callback boundary is intentionally narrow. Sparkring only has to plug the production CUDA callback into `prefill_function` and the existing fast PP13 decode callback into `decode_function`.

The new test proves a 97-token prompt becomes:

```text
prefill dispatch 0: offset 0,  count 64
prefill dispatch 1: offset 64, count 33
decode dispatch: first continuation step
```

without Python control flow.

## Prefill-speed estimate

The old documented B64 PP13 slowest-stage number was:

```text
50.660288 ms for 64 tokens ~= 1263 tok/sec before transport
```

The reported surprise 4-bit doubling changes the working estimate materially. If the same B64 stage shape now takes roughly half that time, the filled-pipeline ceiling is approximately:

```text
64 / 0.02533 ~= 2526 tok/sec before transport
```

A 1500 tok/sec prefill target corresponds to the following maximum slowest-stage times:

```text
64-token chunk:   42.7 ms
128-token chunk:  85.3 ms
256-token chunk: 170.7 ms
```

So 1500 tok/sec is plausible for chunked prefill if the 4-bit improvement applies to the prefill linears/MoE path and the paged/chunked attention path is not falling back to a scalar or tiny-tile path. For long prompts, attention cost scales with the active chunk times the visible prefix, so very long contexts can drop below 1500 tok/sec unless paged FlashAttention-style prefill is used.

The immediate measurement gate should report:

```text
chunk_tokens
visible_prefix_tokens
prefill_stage_slowest_ms
prefill_tokens_per_sec = chunk_tokens / slowest_stage_seconds
```

for at least:

```text
64, 128, 256 token chunks
prefix lengths 0, 256, 1024, 4096
```

## Parallelism and pipelining plan

Do not CUDA-tokenize first. Tokenization should run on CPU and be overlapped with GPU work. The useful pipeline is:

```text
CPU thread A: tokenizer + chat template + prefix hash
CPU thread B: request scheduler + KV block reservation
CUDA stream H2D: token window upload + metadata upload
CUDA stream prefill: embedding gather + stage-slice bulk prefill + KV writes
CUDA stream decode: ready decode batches / speculative verify
KV stream: JIT KV prefetch / external KV residency
```

Use double-buffered prompt token staging:

```text
while GPU prefills chunk N:
    CPU tokenizes or schedules chunk N+1
    H2D stream uploads token ids and metadata for chunk N+1
```

The prompt pipeline pump added here is deliberately callback-based so Sparkring can implement that double buffering without changing the request API.

## Remaining handoff gates

1. Build the new `.cu` prompt-prefill helpers with real `nvcc` on Spark2.
2. Plug `SparkPromptPipelineRun` into the server / direct runner loop.
3. Bind the CUDA prefill callback to the stage-slice bulk prefill launcher.
4. Validate C tokenizer output against the exact GLM-5.2 tokenizer for non-chat and chat prompts.
5. Measure prefill throughput by chunk size and prefix length, then set the production chunk bucket policy.
