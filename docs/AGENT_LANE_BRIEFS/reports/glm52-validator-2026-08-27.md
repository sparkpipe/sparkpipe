# GLM 5.2 validator routed-oracle fix — 2026-08-27

Worktree /tmp/lane-glm52-fix, branch lane/glm52-validator-fix (from
origin/lane/glm52). Scope: the M2 routed-expert validator gate that blocks
`module publish` (pack lane's P4). Node: spark8 (GPU GB10, driver 580.159.03,
CUDA 13.0). Inputs: the synthetic validator fixture + the deployed rank00
pack `/home/spark8/sparkdata/glm52.tp8.fp8/packs/glm52_tp8_rank00.fp8.glms52sp`
(102,835,957,760 bytes). The validator never reads the pack (synthetic
weights); it gates the publish receipt.

## F1 Reproduce — exact failure captured

Command (spark8, workspace /home/spark8/glm52_fix_repo, this branch):

```
$ make -C modules/glm52_resident_decode_stage validate EXPERT_CODEC=fp8 \
    MODEL_REVISION=b4734de4facf877f85769a911abafc5283eab3d9 \
    CONTRACT_SHA256=ec5afd74d2ba0c913474f30d78209b4a49fa87667845f2968039fddad0fead7a \
    NVCC=/usr/local/cuda/bin/nvcc CUDA_ARCH=sm_121a \
    STAGE_PACK_PATH=/home/spark8/sparkdata/glm52.tp8.fp8/packs/glm52_tp8_rank00.fp8.glms52sp
```

Raw output (validator stdout; full log /home/spark8/glm52_fix_f1_reproduce.log):

```
glm52_validation check=layer_forward_hidden elements=6144 relative_l2=0.00603604963 cosine=0.999981849 max_abs=0.001953125
glm52_validation check=layer_forward_residual elements=6144 relative_l2=0.0035535041 cosine=0.999993692 max_abs=0.001953125
glm52_validation check=determinism elements=12288 bit_exact=1
glm52_validation tier=dense PASS
glm52_validation check=routed_selection token=0 experts=5,1,0,3,7,6,4,2
glm52_validation check=routed_selection token=1 experts=1,7,5,3,4,0,6,2
glm52_validation check=routed_selection_set elements=16 exact=1 max_weight_delta=6.24359e-05
glm52_validation check=routed_expert_forward elements=6144 relative_l2=0.335913807 cosine=0.942554536 max_abs=882688
glm52_validation failure=routed_expert_forward detail=relative_l2
make: *** [../resident_decode_stage_rules.mk:200: validate] Error 1
```

Identical numbers to the pack lane's report and commit 9f56300 — fully
deterministic on the synthetic fixture (pre-dates the real packs, as reported).

## F2 Root cause — the oracle's payload read drops the expert dimension

The coordinator's working hypothesis (fixture allocates only 8 compact
payload slabs while the kernel indexes payload expert-major across 256
experts) is HALF right about the layouts but wrong about the disagreement
side. Byte-level audit of all three parties:

1. DEVICE (kernel): `LmGemmEncodeWeightMap` (runtime/gemm.cuh) encodes the
   expert payload as `rows = per-expert rows, columns = input dim,
   groups = group_count` — one TMA group per expert, so expert e's slab sits
   at `payload + e * rows*cols*bits/8`, expert-major across ALL 256 experts.
   Scales: `LmWeightCodecScaleTensor<ExpertCodec>(scale, GLM52_EXPERTS=256,
   rows, hidden)` + `LmScaleTensorIndex` = `group*group_stride +
   row*row_stride + k/k_group` — expert-major across 256 groups,
   row_group_size=1. This matches the real-pack layout the pack lane froze:
   payload fp8 codes verbatim expert-major, scales F32 scale_inv
   row-expanded per 128-block.
2. FIXTURE (SparkGlm52ValFixtureSetup): payload slabs for lanes 0..7 at
   `lane * payload_bytes` — for the pinned selection {0..7} this IS the
   expert-major prefix, inside an 8-slab allocation the kernel only ever
   addresses at offsets 0..7 (selection is pinned by the +4/-4 correction
   bias and `routed_selection_set exact=1` proves experts {0..7} ran).
   Scale buffer priced for all 256 experts (`ScaleBufferBytes(codec,256,...)`),
   planes 0..7 filled at their 256-expert offsets. THE FIXTURE IS CORRECT —
   it already mirrors the real pack layout verbatim. No out-of-bounds or
   zero reads exist: the device read real, distinct slabs.
3. ORACLE (the bug): `SparkGlm52ValDequantWeight` applies the expert
   dimension to the SCALE index only:

   ```c
   uint64_t scale_index = SparkGlm52ValScaleIndex(codec,rows,columns,expert,row,column);
   ...
   raw = SparkGlm52ValE4m3Decode((uint8_t)SparkGlm52ValReadCode(payload,codec,row,columns,column));
   ```

   `SparkGlm52ValReadCode` indexes `payload + row*row_bytes` with NO expert
   offset — every expert's forward reads slab 0's codes. Scales are uniform
   powers of two in the fixture, so the wrong slab is the ONLY error: for
   expert 0 the oracle is right; for experts 1..7 it computes expert 0's
   weights with expert e's (identical) scale.

Why the numbers fit exactly: route 0 of token 0 selected expert 5 (raw
output above: `experts=5,1,0,...`), so `routed_expert_forward` compares the
device's slab-5 forward against the oracle's slab-0 forward. The fixture's
fp8 code grid {0x08,0x2c,0x30,0x34,0x38,0x3c,0x40,0x44,0x88,0xb0} has a
positive mean (~0.76), so two independent slabs' forwards are strongly
correlated through the common-mean component: host simulation of two
independent slabs through the fixture's exact expert MLP gives
cosine 0.994 / rel_l2 0.11 at unit activation scale (see scratch
reproduction below) — the same regime as the observed cosine 0.9426 /
rel_l2 0.336 at the fixture's real activation scale. The alternatives are
excluded: identical slabs would give cosine 1.0; zeros/garbage (the
coordinator's missing-mixture theory) would give cosine ~0 and rel_l2 ~1.4.
Equal-norm check: rel_l2 = sqrt(2*(1-cosine)) = 0.339 at cosine 0.9426 —
the observed pair (0.3359, 0.94255) is self-consistent with a
partially-correlated wrong-slab read.

Scratch simulation (controller mac, python3, fixture grid + expert MLP):

```
$ python3 - <<'EOF'
grid = [0x08,0x2c,0x30,0x34,0x38,0x3c,0x40,0x44,0x88,0xb0]
... e4m3 decode: [0.015625 0.375 0.5 0.75 1. 1.5 2. 3. -0.015625 -0.5]
two independent slabs: cosine=0.993975 rel_l2=0.109633
EOF
```

The oracle selftest (`SPARK_GLM52_VALIDATOR_ORACLE_SELFTEST`) cannot catch
this: it fills ONE slab at offset 0 of a single-slab buffer and reads it
back with `expert=1` — it inherited the same wrong convention (the read at
expert 1 aliasing slab 0 was invisible), and it asserts only finiteness.
The referenced harness tests/test_glm52_cuda_validator_tier2_oracle.py did
not exist.

## F3 Fix — TWO defects found, both on the oracle/fixture side; kernel correct

Fix 1 (the blocker): mirror the kernel's expert-major payload addressing in
the oracle. `SparkGlm52ValDequantWeight` now offsets the payload base by
`expert * payload_bytes_per_expert` (new helper `SparkGlm52ValPayloadExpertOffset`,
the mirror of `SparkGlm52ValScaleExpertOffset`), matching the weight tensor
map's group stride. Callers are unchanged. The selftest now fills TWO
distinguishable slabs and asserts per-expert reads differ (payload plane)
and that a power-of-two scale patch of expert 1's plane rescales its weights
bit-exactly (scale plane); against the old code it fails with
`codec=2 expert payload planes alias: 0/256 positions differ`. Result:
`routed_expert_forward` 0.3359/0.9426 -> 0.00497/0.9999961.

Fix 2 (unmasked by fix 1): the fixture's fp8/nvfp4/mxfp4 code grids were
positive-mean (fp8: mean +0.76 vs std ~1.0), making the expert forward
QUADRATIC in the activation's ones-component (gate/up ~ mean*S,
silu(gate)*up ~ (mean*S)^2 with S = sum(normed)). Evidence trail:
after fix 1, `routed_layer_forward` failed at 0.0356 while every component
matched — per-expert rows 0.3-0.6%, shared 0.47%, weights 6.2e-5, and the
decomposition probe proved the device hidden bit-exactly equals the weighted
sum of its own captured rows (`hidden_vs_own_rows rel_l2=0 cosine=1`), with
normed/attention_out/residual all ~0.35% (dense-tier normal; the dense tier
itself passes at 0.6%). A 4-token walk showed the amplification is
token-dependent (token 1 rows 2-7% off, token 3 rows 0.6-1% off at identical
chain quality) - the signature of mean-amplification through S^2, not a
layout bug. All six codec grids are now exactly zero-mean sign-symmetric
sets (e.g. fp8 {0,±0.015625,±0.375,±0.75,±1.5,±3.0}); `routed_layer_forward`
dropped to 0.0068. This also retroactively explains commit 9f56300's
fp8-vs-int8 split: int8's near-zero-mean grid gave sqrt(2) (uncorrelated
slabs) while fp8's mean-heavy grid gave 0.336 (correlated slabs).

New host gate: tests/test_glm52_cuda_validator_tier2_oracle.py compiles the
validator TU's SPARK_GLM52_VALIDATOR_ORACLE_SELFTEST entry against
tests/cuda_stub (tests/glm52_validator_oracle_selftest_stubs.cpp closes the
module launcher symbols) and runs it on any host.

F3 gate output (spark8, make ... validate, full log
/home/spark8/glm52_fix_f3_clean.log):

```
glm52_validation check=layer_forward_hidden elements=6144 relative_l2=0.00603604963 cosine=0.999981849
glm52_validation check=layer_forward_residual elements=6144 relative_l2=0.0035535041 cosine=0.999993692
glm52_validation check=determinism elements=12288 bit_exact=1
glm52_validation tier=dense PASS
glm52_validation check=routed_selection token=0 experts=5,1,0,3,7,6,4,2
glm52_validation check=routed_selection token=1 experts=1,7,5,3,4,0,6,2
glm52_validation check=routed_selection_set elements=16 exact=1 max_weight_delta=6.24359e-05
glm52_validation check=routed_expert_forward elements=6144 relative_l2=0.00502641659 cosine=0.99998737 max_abs=16384
glm52_validation check=routed_layer_forward elements=6144 relative_l2=0.00683001012 cosine=0.999976681 max_abs=14336
glm52_validation check=routed_layer_forward_residual elements=6144 relative_l2=0.00324159129 cosine=0.999994776
glm52_validation check=routed_determinism routes=16 bit_exact=1
glm52_validation tier=routed PASS
glm52_validation check=dsa_shaping top_min=1.24239874 tail_max=1.32821069e-05 bucket=191
glm52_validation check=dsa_selection elements=2048 exact=1
glm52_validation check=dsa_sparse_attention elements=6144 relative_l2=0.00350011726 cosine=0.999993882
glm52_validation check=dsa_rerun elements=8192 set_and_tolerance_exact=1
glm52_validation tier=dsa PASS
glm52_validation PASS
```

The routed and DSA tiers are green for the first time since the validator
restore. Ratchet 186401 -> 189979 in the same commit (+3492 pre-existing
from the packs lane's head b845f70, verified by running the test on the
unmodified branch; +86 this fix).

## F4 Publish + smoke — publish/compile DONE; TP8 bring-up BLOCKED by sparke wedging

Publish (spark8, the packs lane's P4 command verbatim):

```
glm52_validation PASS
module=spark.glm52.resident_decode_stage.bf16.expert_fp8.h6144.l78.e256.k8.v2
  target=cuda.sm121.glm52.resident_decode_stage.bf16.expert_fp8
  artifact=b7437e4600985506c75f55c82aecdb50681ef53d3404c7f1a0a7cdc62373cb47
  validator=1099fae7971402a73095335b1e8a4e18df615dbd58cfc0b7099f1566933b66e1
  kind=static_archive validation=executed
  record=build/module_library/active/94584cd8702a66cdc27ab68f127304a47ccc51c27b82ffa93c363049546ced3a.json
```

Driver compile: the packs report's command failed at link with undefined
SparkKvPageCache*/SparkKvPageStore* — glm52's module (unlike dsv4's, which
bundles its own paged-cache implementation) calls the shared
cache/kv_page_cache.c that lives in build/libsparkpipe_model_common.a.
Adding `make model_common` + `--cc-arg build/libsparkpipe_model_common.a`
to the compile command links clean:

```
package_manifest=/home/spark8/sparkdata/glm52.tp8.fp8/model_package.json
  stages=1 programs=1 operations=1 collected_link_units=1
  model_sha256=ec5afd74...94095f82165b47
stages/stage_000/model_driver.so (2,035,056 bytes)
```

Adapter: the module Makefile's adapter rule passed the whole
MODULE_BATCH_VARIANT_BUCKETS list into -DSPARK_BATCH_BUCKET= (link error
"cannot find 128/256/..."); fixed to build the b1024 bucket (= the
unflagged archive). Runtime layout staged to all 8 nodes:
lib/{model_driver.so, model_serving_adapter.so, hidden_transport.so} +
bin/{residentd,api,batch}.

### WEDGED NODE REPORT (resolved): sparke (rank 6)

sparke staged artifacts fine, launched rank 6 in the 8-way parallel start,
then degraded within ~5 minutes: first ssh "Connection timed out during
banner exchange", then full packet loss from peers:

```
$ ssh sparkf 'ping -c 2 -W 2 sparke'
2 packets transmitted, 0 received, 100% packet loss
```

Per the lane rules I did NOT reboot it. It recovered BY ITSELF roughly 25
minutes later (`uptime` showed "up 4 min" - the node power-cycled on its
own or via infrastructure watchdog; nothing from this lane touched it).
After it returned (mem 112G free, no processes, artifacts intact on local
disk), the 8-rank launch succeeded.

### TP8 bring-up and B1 decode - three more restore bugs found and fixed

After the band came up, the B1 decode surfaced three additional restore
defects, each fixed and each re-gated through publish -> compile ->
deploy -> relaunch cycles:

1. The admission predicate never decided (module). glm52's
   SparkGlm52AdmissionPredicate performed its KV page-cache mutations but
   returned without touching the decision; the ladder's initializer
   default is REJECTED/UNSUPPORTED_SHAPE and an undecided predicate is
   terminal (spark_admission.h), so EVERY submission died as
   `adapter_submit status=unsupported kind=1` on every rank. qwen38's
   predicate explicitly accepts. Fix: the predicate now accepts on its
   normal path (the shape rules already vetted the request); the
   cache-release branch keeps deciding for itself.
2. The adapter had no KV cache-lane wiring at all (adapter). The module's
   SparkKvPageCache is driven through admission CACHE_PREPARE/COMMIT/ABORT
   and RELEASE frames; the glm52 adapter passed zero cache lanes and zero
   admission flags, implemented no prefetch/resolve hooks, and declared no
   PREFETCH/JIT_KV/RELEASE capabilities - so lanes reached
   SparkKvPageCacheCompleteLane never prepared and every request completed
   INTERNAL_ERROR (`completion driver_status=17 accepted=4`, no chain
   failure, no kv-access record). Fix (mirrors the dsv4 adapter): pending
   and state carry SparkModelDriverCacheLane tables, the descriptor
   declares PREFETCH|JIT_KV|RELEASE|ASYNC_COMPLETION with
   cache_block_token_count=64, SparkGlm52ServingPrefetch prepares lanes,
   SparkGlm52ServingResolvePrefetch commits/aborts them, submit passes the
   lanes, and RELEASE submissions take the CACHE_RELEASE frame path.
3. The deployment generator emitted JIT_KV-incompatible limits (tool).
   With JIT_KV declared, SparkModelServingAdapterValidateRuntimeLimits
   requires kv_logical_page_capacity >= resident capacity and
   kv_physical_page_capacity >= active sequences; the generated configs
   had zeros (and adapter_load initially rejected the descriptor until
   cache_block_token_count matched JIT_KV). Fix:
   tools/glm52_gen_deployment.py now emits 16*ceil(32768/64) = 8192 for
   both page capacities; configs regenerated and redeployed.

The 8-rank band then came up CLEAN on the final (diagnostics-stripped)
build - all eight ready lines captured, e.g.:

```
sparke: model_residentd ready rank=6 stage=6 inflight=4 active=16 rows=16 resident=16 adapter=spark.glm52.serving-adapter.tp8.expert_fp8.v1 model=zai-org/GLM-5.2 revision=b4734de4... tcp=sparke:19486
```

B1 decode (spark8 coordinator, ranks 0-7 = spark8..sparkf, deployed rank
packs, real GLM-5.2 weights; captured on the build whose only delta from
the final tree is six fprintf diagnostics since stripped - the token
stream below is from that run):

```
$ bin/sparkpipe_model_batch --deployment config/model_resident.json \
    --runtime-root /home/spark8/sparkdata/glm52.tp8.fp8 --batch /tmp/glm52_b1_decode.json
{"event":"ready","adapter_id":"spark.glm52.serving-adapter.tp8.expert_fp8.v1","model_id":"zai-org/GLM-5.2","model_revision":"b4734de4...","stage_count":8,"linear_weight_codec":1,"expert_weight_codec":5,"kv_cache_codec":1}
{"event":"accepted","status":0,"request_id":1,"sequence_id":1,...}
{"event":"token","token_id":98665,"token_index":0,...,"generated_token_count":1,...}
{"event":"token","token_id":98655,"token_index":1,...}
{"event":"token","token_id":98567,"token_index":2,...}
{"event":"token","token_id":3837,"token_index":3,...}
{"event":"token","token_id":110834,"token_index":4,...}
{"event":"token","token_id":123473,"token_index":5,...}
{"event":"token","token_id":98314,"token_index":6,...}
{"event":"token","token_id":102840,"token_index":7,...}
{"event":"completed","status":0,...,"generated_token_count":8,...}
sparkpipe_model_batch_pipeline submitted=9 continued=0 admitted=9 rejected=0 leases=0
sparkpipe_model_batch_status=0 terminal=1 requests=1
```

(prefill of the 7-token prompt in two 4+3 waves + 8 decode steps, all
admitted, zero rejections, terminal completion.)

FLEET RULE COMPLIANCE: after evidence capture the entire band was TERMed
(all 8 nodes verified dark) - a half-lit fleet or a lone rank holding
~114GB is a neighbor-killer (the coordinator TERMed an earlier rank-6 on
sparke for exactly that; sparke is K3's build node). Any future bring-up:
reserve spark8-f in the queue first, then launch all 8 ranks within one
180s window (setsid nohup bin/sparkpipe_model_residentd --deployment
config/model_resident.json --rank-index r on each), and tear down after
the run.

## INTEGRATION REQUESTS

1. tests/test_glm52_cuda_validator_tier2_oracle.py (new, this lane) and
   tests/test_glm52_pack_fp8_source.py (packs lane) are NOT registered in
   the Makefile test list (outside my write set) - please add both to the
   gate.
2. The packs lane's P4 driver-compile command needs
   `make model_common` + `--cc-arg build/libsparkpipe_model_common.a`
   appended (glm52 module depends on the shared kv page cache). Either fix
   the runbook text or teach sparkpipe_model_compile to link
   model_common itself.
3. Regenerated deployment configs (tools/glm52_gen_deployment.py with the
   KV page capacities) are live on all 8 nodes; if the coordinator prefers
   different capacities the generator is the single source.
4. sparke self-recovered after ~25 min of network-level wedge; root cause
   unknown (it power-cycled itself). Worth a look by whoever owns node
   health; nothing in this lane rebooted it.
