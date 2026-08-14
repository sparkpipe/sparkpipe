# GLM-5.2 production serving pipeline handoff

This pass adds a production-shaped serving facade above the lower-level request scheduler and prompt pipeline.  The goal is to make the outside API simple while keeping prefill chunking, KV block-table management, batching, prefix cohorting, and JIT KV prefetching inside Sparkpipe.

## External shape

The intended server integration is now:

```text
SparkServingEngineSubmitText(...)
    or
SparkServingEngineSubmitTokenIds(...)

repeat:
    SparkServingEnginePump(...)
    while SparkServingEnginePopEvent(...) == OK:
        stream accepted / prefill / token / complete / error events
```

Callers do not need to choose prefill windows, KV block-table strides, decode batch membership, prefix cohorts, JIT prefetch plans, or tail-window validation modes.  The engine owns copied prompt token storage before submitting into `SparkRequestApi`, so fire-and-forget prompt buffers can be released or reused immediately after submit returns.

## Production runtime contract

The serving engine has a fail-closed runtime contract.  With default flags, initialization requires a production contract with:

```text
bulk prefill token windows
resident KV writes from prefill
resident KV consumption by decode
decode token IDs returned to the result stream
request-owned KV block tables
internal prefill/decode batching
```

It rejects the old validation bridge flag:

```text
tail-window-validation-only
```

It also checks that the lower-level `SparkRequestApi` configuration actually enables the requested internal batching/JIT features.  A configuration that merely claims batching while disabling request-api prefill/decode batching is rejected.

## Internal pipeline

Each pump iteration calls `SparkRequestApiScheduleNext(...)`.  That lower layer already owns:

```text
queued prefill requests
chunked prefill windows
prefill batching
prefix cohorting
decode batching
speculative-verify dispatches
critical JIT KV prefetch
opportunistic JIT KV prefetch
async JIT KV prefetch polling
queue-aware prefix-cache eviction
resident KV block-table views
```

For prefill dispatches, the serving engine builds:

```text
SparkPromptPipelinePrefillDispatch
    host token matrix
    prefill lane view
    resident KV block-table view
```

and calls the configured production prefill callback.  That callback is where the Spark2 CUDA path should connect:

```text
SparkGlm52Sm121RequiredDecodeStageLaunchPromptStageSliceBulkPrefillFromHostTokenIds(...)
```

For decode and speculative-verify dispatches, the serving engine builds:

```text
SparkServingDecodeDispatch
    request dispatch
    resident KV block-table view
```

and calls the configured decode callback.  That callback should connect to the fast PP13 decode/verify path and fill `SparkServingDecodeResult` with token IDs.  The serving engine then publishes token events, completes the request-api dispatch, handles stop-token/request-finish flags, and releases completed records when auto-release is enabled.

## Event stream

The event ring supports:

```text
request accepted
prefill progress
token
request completed
request cancelled
error
backpressure
```

Telemetry is exposed through `SparkServingStats`, including prompt token count, prefill dispatch count, prefill batch count, decode dispatch count, decoded token count, JIT prefetch counters, async prefetch counters, and prefix-family counters.

## Tests added

`tests/test_glm52_serving_engine.c` validates:

```text
production initialization rejects tail-window validation contracts
production initialization rejects contract/API batching mismatches
submitted token buffers are copied and remain valid after caller mutation
a 97-token prompt becomes two prefill windows, then decode
prefill callback receives bulk token windows and resident KV tables
decode callback receives resident KV tables
request completion events are streamed
prefill batching is internal for two compatible requests
```

## Remaining Spark2 integration work

This pass is the serving/control-plane handoff.  The remaining work is real-device integration:

```text
bind the prefill callback to the CUDA bulk prefill launcher
bind the decode callback to the fast PP13 CUDA decode path
make callback implementations use stream/event synchronization safely
add double/triple-buffered host-token and device-token staging
validate that first decode consumes the full committed prompt KV
run end-to-end prompt text -> tokenizer -> full CUDA prefill -> resident KV -> decode
measure prompt tok/sec by chunk size and prefix length
measure aggregate tok/sec under multiple queued prompts
```

The key guardrail is that a tail-window validation bridge should not be registered with the production serving contract.
