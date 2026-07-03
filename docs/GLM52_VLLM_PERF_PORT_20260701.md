# GLM-5.2 PP13 vLLM v0.24 performance-port notes

Input studied: `vllm-main (16).zip`.

Relevant vLLM mechanisms found:

1. The v1 scheduler treats prefill, decode, prefix cache hits, and speculative decode as one token-budget problem. Requests carry a computed-token count, and each scheduling step chooses only the additional tokens needed to catch up.
2. Long prompts are chunked. A prefill can be admitted as a partial prompt chunk instead of monopolizing the worker for the full prompt.
3. Prefix caching is block aligned. Cache hits are counted only up to a safe prompt prefix; if the entire prompt hits, the final token is still recomputed to obtain logits.
4. CUDA graph replay uses padded/static descriptors. vLLM records the real scheduled token count separately from padded graph capacity and explicitly marks stale padded slots invalid.
5. KV/cache transport paths avoid per-block submission overhead by offering batched copy submission, falling back to scalar copies only when the platform lacks the batch path.

Sparkpipe changes made from those mechanisms:

1. `SparkGlm52SchedulerRequest` now accepts `computed_prompt_token_count`, `cached_prefix_token_count`, and an optional per-request `max_scheduled_prompt_token_count`.
2. `SparkGlm52SchedulerAdmit` now schedules a prefill chunk, not necessarily the whole prompt. Default chunk size is 256 tokens and remains block-aligned to 16 tokens for non-final chunks.
3. Prefix-cache admission is block-aligned and safe for all-hit prompts. For a 100-token prompt with a claimed 100-token cache hit, Sparkpipe admits 96 cached tokens and schedules the final 4 tokens.
4. `SparkGlm52SchedulerDecision` and per-stage dispatch records now expose graph sequence capacity, graph padding count, scheduled prompt offset/count, cached-prefix token count, prefill block count, and total scheduled token count.
5. Scheduler counters now track scheduled decode tokens, scheduled prefill tokens, prefix-cache hit tokens, and chunked prefill admissions.
6. Hidden transport now has optional native batched submission capability with `SparkHiddenTransportSendBatch` and `SparkHiddenTransportPostReceiveBatch`. If a transport backend advertises `SPARK_HIDDEN_TRANSPORT_CAP_BATCHED_SUBMISSION`, Sparkpipe submits the whole packet array through one callback. Otherwise it validates the batch and falls back to scalar sends/receives.

Expected performance impact:

- Uncached long prefill: lower tail latency and better decode/prefill interleaving because a long prompt no longer occupies all 13 sparks as one monolithic admission. Raw single-request prompt throughput still depends on the production bulk-prefill kernel behind the stage-slice plan.
- Cached or shared-prefix traffic: direct work reduction. For example, with a 1024-token prompt and a 768-token block-aligned prefix hit, the scheduler admits only the remaining 256 prompt tokens for compute.
- Sub-bucket batches: the runtime now receives explicit graph padding counts, allowing graph-safe B16/B32/B64 execution while keeping the true active-sequence count visible to kernels that can skip padding work.
- Hidden transport: native batch submission removes per-packet transport callback overhead for multi-packet hidden handoff paths, mirroring vLLM's single-driver-call block transfer approach.

The changes are intentionally scheduler/ABI-level. They do not replace the remaining CUDA work: the high-throughput long-context bulk-prefill kernel, real fused stage-slice launch function, and production 4-bit/8-bit tensor-core plans still determine the absolute ceiling on hardware.
