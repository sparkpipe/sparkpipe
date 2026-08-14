# GLM-5.2 PP13 DSA Prefill History Receipt

## Scope

This receipt covers merged main commit
`aeb32970ddfe1f5af935f4ac62dcc89befc8e430`, including PR #485 and PR
#486. It records the live 13-rank FP8 ring result; it is not a corpus accuracy
score.

The deployed immutable release was:

```text
release_id=glm52-fp8-aeb3297-b64-dsa-history-r2
generation=20260718182824
max_active_sequence_count=64
prefill_wave_token_count=64
kv_pool_tokens=16384
kv_logical_blocks=256
model_quantization=fp8
mtp=enabled
diagnostics=disabled
```

All 13 ranks matched these manifest hashes before inference:

```text
libglm52_pp13_node_context_builder.so cfd4bf237f5f863fb711d54744871f0902bde6161dc7a2a8d65964446a145a84
libhidden_transport_tcp_cuda.so       cba5a9e24201847759547bfaf5db084bb8a3a26f85608403d7da487bc35c8a78
model_driver.so                       0e4b86dc7cfc62f9514e07db0b8cda7ad7e08cbca863e349c6ba6ed34af5ac9e
```

## Root Causes

PR #485 fixed two failures in the parallel 64-row prefill path:

1. CUDA graph capture attempted to use the reusable external DSA KV prefetch
   stream and event. Capture now performs that prefetch inline on the capture
   stream; eager execution retains the dedicated stream.
2. DSA candidate progression used the full future prompt length instead of the
   end of the current prefill wave.

PR #486 fixed the remaining history corruption. Full DSA index-share mode
returned context-prefix indices while the candidate count was at most 2,048,
before running the DSA indexer. Tokens 0 through 2,047 therefore never
populated `key_index_cache_bf16`; sparse scoring first consumed that invalid
history at token 2,048. The DSA indexer now runs before the prefix shortcut, so
the cache is populated on every source-layer submission.

## Build Gates

The full host suite passed with `make -j test`. The native SM121 archive and
linked package both passed the exact six-layer FP8 stage-slice validator:

```text
archive timed_us=14126.675 nonzero=6143 checksum64=5958189933213524842 graph_captures=1 graph_replays=2
driver  timed_us=14133.739 nonzero=6143 checksum64=5958189933213524842 graph_captures=1 graph_replays=2
```

## Live Results

The prompts were real C source and technical text, submitted through the public
streaming completions API with `temperature=0` and `max_tokens=1`. Times include
the distributed prefill, pipeline/control overhead, and one output token. They
exclude model initialization and resident loading.

```text
prompt_tokens run total_seconds token_id token_text done
1983          1   15.313494     80780    configured yes
1983          2   15.298303     80780    configured yes
3861          1   28.569240     555      " }\n\n" yes
3861          2   28.362233     555      " }\n\n" yes
7868          1   55.547800     3850     unt        yes
7868          2   55.505562     3850     unt        yes
7868          3   55.542593     3850     unt        yes
```

A least-squares fit over the mean latency at each prompt size is:

```text
total_seconds = 1.917040 + prompt_tokens * 0.006822674
sustainable_prefill_rate = 146.570 prompt_tokens/second
```

Before PR #486, repeated 7,868-token runs returned token IDs 15, 11857, and 15
in 56.62, 59.27, and 53.45 seconds. The fixed release returned token ID 3850
on all three runs. Repeated 3,861-token runs also became identical across the
former 2,048-token activation boundary.

The exact smoke prompt `Say OK. OK.` returned token `10397` (`" OK"`) and a
done event. A separate eight-token greedy decode returned:

```text
 Paris. Distance from Paris to Lyon is
```

Post-run health reported zero live requests, zero queued requests, zero event
backlog, and zero dropped events. Every resident log was clean after its final
`state=ready` marker.

## Status

```text
end_to_end_observation_status=OBSERVED
prefill_determinism_status=MEASURED
sustainable_prefill_performance_status=MEASURED
short_prompt_exact_receipt_status=MEASURED
corpus_accuracy_status=NOT_MEASURED
multi_sequence_prefill_status=NOT_MEASURED
```

The result closes the single-request DSA history correctness failure and
measures the current sustainable rate. It does not claim broad model accuracy,
multi-sequence batching, or acceptable production prefill performance.
