# GLM52 Final Stage And Runtime Boundary - 2026-06-30

SparkPipe GLM52 runtime code must stay C/CUDA after one-time setup.

Allowed one-time setup tools:

```text
tools/glm52_b12x_aot_compile.py
tools/glm52_b12x_resident_pack.py
tools/glm52_b12x_pack_worker.py
tools/glm52_stage_bucket_sweep.py
```

These tools may use Python, Torch, FlashInfer, CuTe DSL, and filesystem fixtures
because they are setup, qualification, or benchmark harnesses.

Production path rule:

```text
after resident packs and AOT objects exist, prompt/hidden execution is C/CUDA
only
```

The serving path must not import Python, Torch, FlashInfer Python, or CuTe DSL.
It must load resident artifacts, execute the linked C/CUDA module, and fail if a
required module is missing.

## New Final-From-Hidden Mode

Intermediate stage validation uses:

```text
GLM52_VALIDATION_MODE=routed_from_hidden
```

That mode is hidden-only and must not claim final token inference. It sets:

```text
GLM52_CHAIN_ROUTED_FROM_HIDDEN_BF16=1
OUTPUT_HIDDEN_ONLY=1
```

Final stage validation now uses:

```text
GLM52_VALIDATION_MODE=routed_from_hidden_final
```

That mode consumes `GLM52_PIPELINE_INPUT_HIDDEN_BF16`, runs the routed layer
slice, keeps hidden-only enabled for intermediate layers inside the slice, and
clears hidden-only only for the final layer. The final layer must run:

```text
final RMSNorm
restricted logits
restricted argmax
MTP draft
MTP verify/commit
```

The success line is distinct:

```text
routed_pipeline_from_hidden_final=1 final_stage=1
```

It prints real final-stage evidence:

```text
restricted_token
mtp_draft
mtp_reject
real_lm_head
real_lm_head_max_logit_error
launch_chains
graph_captures
graph_replays
```

This is still a validator/package gate, not a production scheduler. It exists so
SparkPipe can prove that a hidden vector from the previous stage can reach the
final C/CUDA token path without Python runtime glue.

## Current End-To-End Status

The repo now has:

```text
intermediate hidden stage mode: yes
final hidden-to-token mode: yes
Python production pipeline runner: no, intentionally
```

Full end-to-end inference is still not PASS until the live run executes:

```text
dense prefix or real prompt input
all routed hidden stages
final routed_from_hidden_final stage
real hidden transport or accepted local transport substitute
```

Do not report `make test` or a Python sweep as final inference. The accepted
signal is a C/CUDA run that reaches `routed_pipeline_from_hidden_final=1` and
emits a final `restricted_token` from the final stage.
