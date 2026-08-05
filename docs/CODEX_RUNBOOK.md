# SparkPipe Development And Release Runbook

This is the only current operational path. Deleted model-specific daemons,
gateways, schedulers, compatibility shims, and fallback drivers are not release
options.

## Runtime Contract

- Every rank runs one `sparkpipe_model_residentd` process.
- Every client uses the generic resident IPC contract. The repository batch
  client is `sparkpipe_model_batch`.
- A model package names one serving adapter, one immutable AOT driver, one
  stage pack, and one exact weight/KV codec tuple.
- Codec selection happens when the package is built. Common runtime code does
  not choose a codec. A driver does not switch codecs at runtime.
- Package, adapter, driver, and stage-pack identity must agree byte-for-byte at
  startup. Any mismatch is fatal.
- GLM 5.2 has six peer expert-package variants: INT6, INT7, INT8, FP8 E4M3,
  NVFP4 E2M1, and MXFP4 E2M1. BF16 nonexpert weights and BF16 KV are part of
  each current GLM package contract.
- DeepSeek V4 GA Flash currently declares FP8 E4M3 nonexpert weights, MXFP4
  E2M1 routed experts, and BF16 KV. That tuple belongs to that package; it is
  not a common runtime default.
- There is no reference, serialized, compatibility, or alternate-kernel
  fallback in a production package.

## Change Cycle

1. Start from current `origin/main` in one clean checkout.
2. Make the change on a `codex/` branch and run source/host tests.
3. Open a PR. Do not deploy files copied from the branch.
4. Qualify the exact PR commit on a GB10 Spark in an isolated clean checkout.
5. Merge only after the Spark CUDA gate and required host/runtime tests pass.
6. Pull merged `main` into a clean Spark checkout, rebuild, assemble one
   immutable release, and install that release on all ranks.
7. Run live correctness before measuring throughput.

The Mac can run parsers, generators, and host-only tests. CUDA compilation,
archive qualification, driver loading, and execution receipts must come from a
Spark.

## Native Spark Qualification

Use the repository branch, not copied source files or a dirty live checkout:

```sh
git clone --single-branch --branch <pr-branch> \
    https://github.com/sparkpipe/sparkpipe /tmp/sparkpipe-pr-<sha8>
cd /tmp/sparkpipe-pr-<sha8>
test "$(git rev-parse HEAD)" = "<full-pr-sha>"
test -z "$(git status --porcelain)"
make -j8 test
env PATH=/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin \
    NVCC=/usr/local/cuda/bin/nvcc CUDA_ARCH=sm_121a \
    tools/cuda13_sm121a_compile_gate.sh
```

The CUDA gate is all-or-nothing. It compiles exact `sm_121a` objects, verifies
their architecture, builds every selectable GLM codec package, builds DSV4,
and writes SHA-256 receipts. Missing CUDA, verbs headers, translation units, or
one codec is a failed gate.

After merge, repeat from a clean checkout of `main`. A premerge receipt proves
the PR; it does not replace the merged-main deployment receipt.

## Stage Packs

Raw model data is staging input, never the inference hot path. Each Spark packs
only the layers its stage owns and serves the resulting rank-local files from
local NVMe.

- `tools/dsv4_stage_source.py` creates DSV4 stage source data.
- `tools/dsv4_stagepack.py` creates the exact DSV4 package stage pack.
- `tools/glm52_stagepack.py` requires `--expert-codec`; it has no default.

Retain each pack receipt with source revision, model contract SHA-256, pack
recipe SHA-256, codec tuple, layer range, byte size, and output SHA-256. A pack
from a different revision or codec is not reusable under a renamed path.

## Deployment Generation

Edit the compact specification, then generate the expanded runtime document:

```sh
python3 tools/generate_model_resident_deployment.py \
    --specification <model>.spec.json \
    --output model_resident.json
python3 tools/generate_model_resident_deployment.py \
    --specification <model>.spec.json \
    --output model_resident.json \
    --check
```

The generator is model-neutral. It expands rank hosts, stage order, paths, and
control endpoints. The model package supplies the opaque `node_target`,
adapter, driver, and stage configuration. Unknown fields, missing fields,
duplicate ranks/stages/endpoints, inconsistent capacities, and unresolved
placeholders are fatal.

## Release Assembly

Every release contains exactly one role named `model_resident` and these
required files:

```text
bin/sparkpipe_model_residentd
lib/model_serving_adapter.so
lib/model_driver.so
lib/hidden_transport.so
config/model_resident.json
```

Use `tools/sparkpipe_release_assemble.py` with the full merged commit and
explicit replacements. The assembler recomputes sizes and SHA-256 values and
publishes atomically. Do not hand-edit the assembled manifest.

Install all ranks from the same release generation. Before launch, verify the
installed manifest, adapter, driver, transport, deployment, and rank-local
stage-pack hashes. Do not start a second resident beside an existing GPU owner.

The release role invokes:

```sh
bin/sparkpipe_model_residentd \
    --deployment config/model_resident.json \
    --rank-index <rank>
```

Readiness requires all ranks to print the same adapter ID, model ID, revision,
stage count, and codec IDs expected by the package. A listening socket alone
is not whole-ring readiness.

## Correctness Gate

Use a strict pretokenized batch document and run:

```sh
bin/sparkpipe_model_batch \
    --deployment config/model_resident.json \
    --runtime-root "$PWD" \
    --batch qualification-batch.json > results.jsonl
```

The client emits a flushed `ready`, `accepted`, per-token, and terminal event
stream. Keep the full JSONL. The `ready` event records the model revision and
numeric codec tuple actually attached to the ring.

Run these gates in order:

1. One deterministic prompt, one output token, all 13 ranks, terminal event.
2. Serial and batched copies of identical rendered prompts; token outputs must
   match exactly.
3. A retained accuracy subset with the model's exact tokenizer, template,
   sampling policy, and output budget.
4. Full accuracy evaluation.
5. Throughput and latency benchmarks only after correctness passes.

A process-ready line, accepted request, compile receipt, or plausible text is
not an accuracy result. Report measured scores and speed separately.

## Failure Policy

- Stop at the first identity, schema, CUDA, transport, or token mismatch.
- Never substitute another package, codec, driver, transport, or test mode.
- Preserve logs and receipts under `/tmp` or the release evidence directory;
  do not commit raw private-network logs.
- Fix source through a PR, merge it, pull clean `main`, rebuild, and rerun the
  failed gate. A live hotpatch is not a validation result.
