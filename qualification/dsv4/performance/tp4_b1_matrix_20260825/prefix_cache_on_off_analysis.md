# Prefix caching ON vs OFF — dsv4-flash context matrix (ctx512/2048)

Question: delta between prefix caching ENABLED and DISABLED at ctx512 and
ctx2048 on the dsv4-flash TP4 stack.

Answer: **delta = 0 (zero) at both contexts, by construction of this stack
generation** — and the empirical receipts already agree.

## Why it cannot be otherwise at B1 serving settings

1. The client-side prefix cache lives inside ONE sparkpipe_model_batch
   process (runtime/model_batch_engine.c: SparkPrefixCache instance, entries,
   bindings, hash heads are engine fields). It is populated only as prompt
   blocks complete during that process (PublishCompletedBlocks) and every
   lookup happens in-process. There is no persistence knob in the batch JSON
   API; a new client invocation starts cache-cold. Cross-run reuse therefore
   never happens no matter what.
2. Within one process, reuse needs a SECOND request. The B1 serving config
   admits request_capacity=1 / max_input_rows=1 / resident_sequence_capacity=1,
   and the batch client rejects anything larger (status=2, probed). One
   request per process => the lookup set is always empty before the only
   request ships.
3. Enable/disable is not user-facing anyway: the engine arms its cache iff
   adapter_descriptor->cache_block_token_count != 0 (DSV4 flash declares 128),
   and there is no flag/env to turn it off — so "DISABLED" can only be
   emulated by guaranteeing lookups miss (unique sequences + distinct prompts),
   which changes nothing when there is nothing to hit.

## Empirical agreement (existing receipts, this directory)

The ON-max pattern (identical sequence_id AND identical 512/2048-token prompt
resubmitted back-to-back on a warm ring) was already measured three times per
cell: prefill accepted->first-token stayed at 13.56 s @ctx512 and 55.29 s
@ctx2048 across repeats with under 2 percent spread — identical to fresh-prompt
runs within noise. A working prefix path would collapse those numbers toward
~0 for the repeated runs. It did not.

The same structural wall was confirmed on a bucket-8 runtime inspected during
the session (max_active_sequences=1, resident_sequence_capacity=1).

## What WOULD show a delta

- max_active_sequences >= 2 or resident_sequence_capacity >= 2 plus a
  multi-request batch whose later requests share a >=128-token prefix block
  with an earlier completed request in the SAME process, on a driver
  generation whose width!=1 TP graph islands actually capture (today widths
  4/16 fail tp_graph_projection_island; see README.md).
- Or server-side persistent prefix identity across client invocations, which
  the current protocol does not expose to this client.

Until one of those lands: report the prefix-cache delta as 0.0 tok/s and
0.0 s at both contexts — that is a measured-and-proven zero, not a gap.
