# GLM52 Tech Debt

This file tracks missing production work. It is not a roadmap for compatibility
fallbacks. Each item should disappear when the production implementation lands.

## External API Surface

Status: partially implemented.

What exists:

```text
C service runtime
C binary frame protocol
OpenAI JSON to service request adapter
Anthropic JSON to service request adapter
LAN HTTP/SSE gateway shell
demo UI served by the gateway
text file upload folding into prompt input
CORS preflight for public JSON endpoints
token-id to text decoder
SSE service event formatter
PP13 rank-plan API
resident decode-stage production runner API
resident PP13 rank daemon
sparkc to spark0 final completion event route
```

Missing:

```text
attach HTTP/SSE gateway to the production C service runtime
authentication and tenant policy above the C API
public schema examples for final OpenAI and Anthropic responses
versioned API compatibility tests for external callers
```

## End-to-end Prompt Inference

Status: not yet final.

Missing:

```text
one command or service path that accepts a real prompt
full prompt prefill through all 78 layers
KV retained across prefill and decode on every rank
PP13 decode loop for generated tokens
token ids decoded back to text
streaming client-visible response
```

## Distributed PP13 Runtime

Status: resident rank daemon exists; full live ring still needs completion.

What exists:

```text
fixed PP13 rank planning
rank-local FP8 pack validation at daemon startup
resident daemon binary per rank
rank-specific input/output hidden transport requirements
sparkc completion event socket back to spark0
fail-closed requirement for production hidden transport shared object
fail-closed requirement for GLM-5.2 model driver shared object
```

Missing:

```text
13-rank service startup and health protocol
real persistent hidden transport backend
GLM-5.2 rank model-driver shared object for the resident daemon
driver node-context binding for resident weights, KV, graph buckets, and packs
rank dispatch loop from Spark0 service requests into stage runner submissions
cross-rank error propagation
ring restart and quiesce path
```

## Hidden Transport

Status: C interface and validation exist.

Missing:

```text
production fabric backend
zero-copy device buffer path
stream-ordered send and receive completion proof
transport latency benchmark in the PP13 ring
transport counters in service stats
```

## FP8 Accuracy

Status: source integrated; final proof still required.

Missing:

```text
FP8 final-token accuracy gate from pulled main
FP8 B12x route-output correctness gate
FP8 DSA selected-token parity gate in the full path
FP8 B64/B128/B256/B512/B1024 timing with expert coverage
FP8 prompt prefill correctness gate
```

## Prefill

Status: scheduler and API shape exist; production proof still required.

Missing:

```text
full-prompt prefill through PP13
paged DSA prefill attention timing
prefill KV write verification across all ranks
realistic prompt corpus benchmark
tokenization timing separated from prefill timing
```

## Tokenizer

Status: C tokenizer exists; production service connection needs proof.

Missing:

```text
service-level tokenizer initialization contract
zero-allocation tokenizer loop proof
large-context tokenizer benchmark from pulled main
OpenAI and Anthropic response formatting over decoded text
```

## Speculative Decode

Status: optional work, not required for baseline inference.

Missing:

```text
MTP one-time artifact generation for chosen buckets
MTP correctness and acceptance-rate gates
DSpark speculator resident loading
DSpark tap wiring into the serving pipeline
DSpark acceptance-rate benchmark
fallback-free disable path when speculation is unavailable
```

## Storage And Pack Placement

Status: stable pack-root policy exists; cleanup still manual.

Missing:

```text
rank-local required-pack manifest
per-rank disk-space preflight
safe cleanup plan for old model trees and caches
no fallback pack search paths
operator-visible pack version and hash report
```

## Performance Qualification

Status: individual benchmarks exist; final production report missing.

Missing:

```text
B1/B4/B16/B64/B128/B256/B512/B1024 decode sweeps
expert coverage in every MoE timing run
prefill timing for realistic prompts
transport timing in the real PP13 ring
graph capture and replay counters
single-request latency
filled-pipeline throughput
```

## Code Cleanup

Status: ongoing.

Missing:

```text
remove reference-only code from production targets
remove validation harness code from production targets
remove compatibility directory chains
remove comments from production code
remove dead demos and obsolete docs after replacement docs land
keep PR rationale in PR text, not source comments
```
