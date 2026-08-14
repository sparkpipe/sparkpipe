# GLM52 Final Expected Condition

This is the pass condition for GLM-5.2 on SparkPipe. Anything short of this is
not final production inference.

## Final User-visible Behavior

A client submits a real prompt through Spark0 and receives a streamed text
response.

Accepted input forms:

```text
token ids
OpenAI-compatible chat JSON
OpenAI-compatible prompt JSON
Anthropic-compatible messages JSON
```

The output must include:

```text
accepted request id
prefill progress
streamed token ids
streamed decoded text
completion or cancellation event
final status
```

## Production Runtime Shape

The final runtime is a 13-rank GLM-5.2 FP8 PP13 pipeline:

```text
spark0  layers 0..5
spark1  layers 6..11
spark2  layers 12..17
spark3  layers 18..23
spark4  layers 24..29
spark5  layers 30..35
spark6  layers 36..41
spark7  layers 42..47
spark8  layers 48..53
spark9  layers 54..59
sparka  layers 60..65
sparkb  layers 66..71
sparkc  layers 72..77
```

Every rank must:

```text
load its layer-specific .sp* packs from local NVMe
keep rank weights resident in GPU memory
own its local KV cache for its assigned layers
open persistent hidden transport with adjacent ranks
execute one stage slice per token or batch bucket
send only hidden state and small metadata across rank boundaries
```

The final stage must additionally:

```text
run final-token selection
return token ids
send final token events back to Spark0
optionally run MTP or DSpark verification when enabled
```

Spark0 must additionally:

```text
accept the public API request
own the C service runtime
own the tokenizer handle
own the scheduler and request API state
own the final-event listener
bridge request token IDs into distributed PP13 prefill/decode driver frames
stream final tokens back to the client
```

Current PP13 service-backend progress:

```text
Spark0 owns the C service runtime, scheduler, request API, prefix cache, KV arena, tokenizer handle, and final-event listener.
Spark0 remains fail-closed until the rank0 token-id input bridge to the distributed PP13 driver is present.
```

## Prompt Inference Pass Condition

Real prompt inference requires all of these to happen in one request:

```text
1. parse client request
2. tokenize or accept caller token ids
3. prefill the full prompt through all 78 layers
4. write resident KV for every owning rank
5. retain KV ownership after prefill
6. decode generated tokens through the exact PP13 pipeline
7. return final-stage token events from sparkc to spark0
8. stream generated token ids
9. decode token ids to text
10. finish or cancel with a service event
```

Last-token embedding validation is not prompt inference.

Single-rank validation is not distributed inference.

Fixture-only token generation is not arbitrary prompt inference.

## Production Performance Pass Condition

Benchmarks must be run from pulled `main` on the target Spark checkout.

The benchmark report must include:

```text
git commit
rank count
stage plan
quantization mode
batch bucket
prompt token count
decode token count
context window
DSA selected-token count
expert coverage
prefill latency
decode stage-slice latency
final-token latency
transport latency
filled-pipeline tok/s
single-request tok/s
accuracy gate result
```

Performance numbers must be separated by path:

```text
prefill
decode
final logits
MTP or DSpark
transport
tokenization
```

Do not collapse these into one number.

## Accuracy Pass Condition

The final run must prove:

```text
tokenizer round-trip for submitted prompt
prefill correctness against the selected reference gate
decode token match or bounded logit error for the chosen quantization
FP8 B12x route-output correctness
DSA selected-token set correctness
KV block-table correctness across prefill and decode
```

For FP8, timing alone is not enough. The FP8 path must pass the same final-token
or logit-error gate used to qualify the production run.

## Forbidden Production Behavior

The production runtime must not use:

```text
Python in the request or decode hot path
bash scripts in the request or decode hot path
file handoff between stages
scp or rsync between stages
Mac-hosted model files
spinning-disk model reads
validation fixtures as request input
reference fallback kernels
compatibility directory search chains
per-request pack generation
per-token cudaMalloc or cudaFree
per-token descriptor rebuilding
```

Setup tools may use Python when generating tokenizer artifacts, packs, or
offline metadata. The running service must be C/CUDA plus generated artifacts.

## Deployment Pass Condition

Deployment is only valid after:

```text
1. PR merged to main
2. every target Spark pulls main
3. every rank validates its local pack files
4. every rank starts the resident GLM-5.2 process
5. every rank opens persistent hidden transport
6. spark0 accepts a real prompt request
7. sparkc emits final tokens
8. the client receives streamed text
```

If any step is missing, the status is fail with a named blocker.
