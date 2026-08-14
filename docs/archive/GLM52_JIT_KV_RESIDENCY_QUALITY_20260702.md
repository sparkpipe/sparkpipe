# GLM-5.2 PP13 JIT KV Residency Quality Pass

This pass keeps KV-cache residency decisions inside Sparkpipe, where the near-future queue and runtime batch shape are visible.

The request API can now enforce a resident KV budget with `max_resident_kv_block_count`. After synchronous or asynchronous JIT prefetch completion, Sparkpipe trims resident KV blocks back to the configured target while protecting:

- blocks required by the just-selected critical dispatch,
- blocks covered by pending async prefetches,
- blocks owned by running prefill/decode/cohort work,
- blocks needed by the near-future request lookahead window.

The prefix cache also supports a lookahead protection sweep. Before scheduling, Sparkpipe scans the same internal near-future request window and tags reusable prefix-cache entries that are likely to be needed soon. Eviction skips unprotected entries before considering protected ones, so large queued bulk work is less likely to evict hot shared-prefix blocks that Centaur or a nearby minibatch is about to reuse.

This is not caller-managed cache control. External callers still submit prompts, budgets, and priority. Sparkpipe owns queue inspection, prefetch, residency, eviction, and greenlighting.

Validation added:

- prefix-cache eviction skips protected shared-prefix blocks,
- request scheduling refreshes lookahead protection automatically,
- request scheduling trims resident KV to the configured limit without evicting the near-future hot prefix.
