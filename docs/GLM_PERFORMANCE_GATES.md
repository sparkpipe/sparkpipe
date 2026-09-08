# GLM non-speculative performance gates

Local driver qualification on merged main `80acca1` passes real-pack B1/B3
resident/lazy token comparison, two concurrently started lazy B3 consumers,
and resident B3 CUDA memcheck with zero errors. These are TP16 rank0 runs with
collectives disabled, four fixed-input positions, and short context. They do
not establish full-model accuracy or distributed performance. See PR #856 for
the index-cache corruption that invalidated earlier B3 comparisons.

Before reporting a full-system throughput result:

1. Build all participating drivers from the same clean merged commit. Record
   package, driver and pack hashes, topology, GPU/driver versions, context,
   occupancy, precision and queue job IDs. Use assigned-node queue sync and
   build jobs; reserve the full topology for the actual distributed run.
2. Pass real distributed token/numerical checks, prefill/decode continuity and
   teardown on the topology being measured. TP4xPP4 still needs its serving
   adapter and boundary path qualified independently of TP16.
3. Run memory checking outside the timing cell. Disable diagnostic probes for
   timing, prewarm the intended expert working set within its explicit budget,
   and report cold loading separately. A local token smoke pass is insufficient
   to skip distributed or long-context checks.
4. Measure B1 and arbitrary occupancies such as 3, 7 and 15, then continuous
   arrivals/completions. Report aggregate output tok/sec separately from
   per-sequence latency and TTFT. Repeat comparable unprofiled runs, recording
   variability. Do not infer linear scaling to 100 sequences without memory
   and compute measurements.

Use `tools/glm5_next_bench_wrap.py --timeout-seconds 600 -- COMMAND ...` around
the actual `sparkpipe_model_batch` client. It consumes the client's newline
JSON events, checks per-request token order and terminal completion, drains
stderr to a temporary file, and kills its owned process group at the deadline.
Failed, cancelled, incomplete or malformed token streams do not receive a
throughput field, and the wrapper exits nonzero for invalid receipts.

For batched decode, use `all_sequences_decode_window`: its clock starts
after every sequence has emitted its first token and ends at the earliest
sequence's last token. It excludes remaining prefill and the shrinking-batch
tail. If those bounds do not overlap, no decode-window rate is reported.
Per-sequence token IDs and arrival times are retained for parity checks and
independent timing analysis. The first-token boundary includes scheduling,
prefill and first-token work; it is not a measurement of prefill compute alone.

A cached-prefix benchmark must restore KV, index KV, KDA recurrent state,
convolution windows and sequence positions before its timed decode phase.
GLM does not yet advertise that full prefix-restore contract. Do not label
an uncached run a cache hit or silently recompute a missing benchmark entry.

`decode_tokens_per_second` counts all output tokens after the first global
token divided by the first-to-last arrival interval. Tokens stay in arrival
order; request-local indices never reorder different requests. TTFT is from
client command launch, not from each request's individual admission time.
Single-sequence inter-token statistics are omitted for concurrent streams.
These are client-observed rates, so stdout buffering and client scheduling are
part of the observation; use device/transport profiling separately to explain
the critical path. A single token has no measurable decode rate.
