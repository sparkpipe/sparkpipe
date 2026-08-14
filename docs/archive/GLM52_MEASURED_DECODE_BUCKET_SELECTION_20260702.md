# GLM52 measured decode bucket selection

This pass connects decode scheduling to the measured PP13 timing profile instead of always using the smallest graph bucket that can hold the active sequence count.

## Reason

The current fixed PP13 six-layer B64 profile is materially faster than the B32/B16 measured-balanced profile in the repo. Using the smallest bucket is therefore not always the fastest decode decision for the queued workload. A 17-lane decode batch previously selected B32, even though the measured B64 critical path is lower.

## Implemented behavior

For decode requests with more than B16 active lanes, the scheduler evaluates B16, B32, and B64 candidates that can hold the batch. It builds the measured stage plan and cost profile for each candidate, then selects the bucket with the lowest measured critical path. Ties prefer the smaller bucket.

Prefill bucket selection is unchanged. Small decode batches up to B16 keep the minimal bucket to avoid unnecessary graph padding for tiny interactive batches unless enough lanes are available to justify the measured larger bucket.

## Runtime visibility

When the selected decode bucket differs from the minimal capacity bucket, the scheduler sets:

```c
SPARK_SCHEDULER_DECISION_FLAG_MEASURED_DECODE_BUCKET
SPARK_SCHEDULER_DISPATCH_STAGE_FLAG_MEASURED_DECODE_BUCKET
```

It also increments:

```c
measured_decode_bucket_selection_count
measured_decode_bucket_padding_token_count
```

## Validation cases

The tests cover both modes:

- default measured mode: 17 decode lanes select B64 and carry the measured-bucket flags;
- measured mode disabled: the same 17 decode lanes fall back to the legacy B32 selection;
- request API integration: 17 ready decode requests schedule one B64 decode dispatch through the caller-facing API.
