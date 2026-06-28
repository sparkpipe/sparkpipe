# GLM 5.2 LLM driver debug log

This log records live Spark/CUDA driver gates that should not be lost in chat
history. A passing gate here is not a full GLM inference claim unless it says
so explicitly.

## Current verified gate

Hardware:

```text
GPU: NVIDIA GB10
CUDA: 13.0
Target: sm_121
```

Live GLM artifact geometry checked on Spark1:

```text
model_dirs=/home/spark1/models/hf/nvidia/GLM-5.2-NVFP4,/home/spark1/models/hf/lukealonso/GLM-5.2-NVFP4,/home/spark1/models/hf/zai-org/GLM-5.2-FP8
hidden_size=6144
num_attention_heads=64
kv_lora_rank=512
qk_rope_head_dim=64
index_topk=2048
num_hidden_layers=78
```

Commands:

```sh
make -C modules/glm52_resident_decode_stage artifact_check GLM52_MODEL_DIR=/home/spark1/models/hf/nvidia/GLM-5.2-NVFP4
make -C modules/glm52_resident_decode_stage artifact_check GLM52_MODEL_DIR=/home/spark1/models/hf/lukealonso/GLM-5.2-NVFP4
make -C modules/glm52_resident_decode_stage artifact_check GLM52_MODEL_DIR=/home/spark1/models/hf/nvidia/GLM-5.2-NVFP4 ARTIFACT_CHECK_ARGS=--check-body-samples
make -C modules/glm52_resident_decode_stage artifact_check GLM52_MODEL_DIR=/home/spark1/models/hf/lukealonso/GLM-5.2-NVFP4 ARTIFACT_CHECK_ARGS=--check-body-samples
make -C modules/glm52_resident_decode_stage artifact_check GLM52_MODEL_DIR=/home/spark1/models/hf/zai-org/GLM-5.2-FP8 ARTIFACT_CHECK_ARGS=--check-body-samples
PATH=/usr/local/cuda-13.0/bin:$PATH make -j1 test
PATH=/usr/local/cuda-13.0/bin:$PATH make glm52_resident_sparse_mla_firmware_package MAX_STAGE_MICROSECONDS=1500
PATH=/usr/local/cuda-13.0/bin:$PATH make glm52_resident_decode_stage_firmware_package MAX_STAGE_MICROSECONDS=10000
GLM52_MODEL_DIR=/home/spark1/models/hf/nvidia/GLM-5.2-NVFP4 PATH=/usr/local/cuda-13.0/bin:$PATH make -C modules/glm52_resident_decode_stage validate MAX_STAGE_MICROSECONDS=10000
```

The artifact gate verifies the live `config.json` against
`metadata.hf_config_geometry`, verifies `metadata.module_geometry` against the
compiled resident decode-stage constants, and verifies the route selects the
same module before any launch claim. With `--model-dir`, it also validates the
first raw checkpoint tensor contract through `model.safetensors.index.json` and
the safetensors shard headers. With `ARTIFACT_CHECK_ARGS=--check-body-samples`,
it also seeks into tensor bodies and hashes deterministic edge samples plus the
restricted-vocabulary `lm_head.weight` rows:

This gate is deliberately module-local. Generic SparkPipe compiler/runtime code
does not parse GLM tensor names, dtypes, or safetensors layouts.

```text
hidden_size=6144
num_attention_heads=64
kv_lora_rank=512
qk_rope_head_dim=64
index_topk=2048
num_hidden_layers=78
vocab_size=154880
n_routed_experts=256
n_shared_experts=1
num_experts_per_tok=8
qk_nope_head_dim=192
qk_head_dim=256
v_head_dim=256
model_revision=bf16-h6144-h64-d512-r64-k2048-b64-rv256-mtp2-v1
module=spark.glm52.resident_decode_stage.bf16.h6144.h64.d512.r64.k2048.b64.rv256.mtp2.v1
tensor_contract_ready=1
tensor_count=9
tensor_bytes=2233222144
tensor_body_sample_ready=1
tensor_body_sample_count=22
tensor_body_sample_bytes=3236864
tensor_body_nonzero_bytes=3229801
tensor_body_sha256=edd995fc32fbe0ac33241a1ff7c17f2549ea71ca5f3ba0911821b91a2972f480
```

The same geometry check against `/home/spark1/models/hf/zai-org/GLM-5.2-FP8`
uses the `hf_tensor_contract_fp8_e4m3` model-description contract. The resident
CUDA path now has raw FP8 q/kv/o projection support for:

```text
q_a_proj.weight F8_E4M3 [2048,6144]
q_a_proj.weight_scale_inv F32 [16,48]
q_b_proj.weight F8_E4M3 [16384,2048]
q_b_proj.weight_scale_inv F32 [128,16]
kv_a_proj_with_mqa.weight F8_E4M3 [576,6144]
kv_a_proj_with_mqa.weight_scale_inv F32 [5,48]
kv_b_proj.weight F8_E4M3 [28672,512]
kv_b_proj.weight_scale_inv F32 [224,4]
o_proj.weight F8_E4M3 [6144,16384]
o_proj.weight_scale_inv F32 [48,128]
```

Do not claim full FP8 resident decode-stage readiness from the artifact check
alone. The resident CUDA path now writes explicit key-nope and value caches from
`kv_b_proj`, emits value-head attention output with shape `64 x 256`, and feeds
the real `[6144,16384]` `o_proj` path. Full readiness still requires hardware
validator success and checkpoint-derived logits equivalence across layer
progression. Restricted-vocabulary logits now have a narrower checkpoint-backed
CUDA validator: it loads real BF16 `lm_head.weight` restricted rows and checks
the resident CUDA logits against a CPU reference reduction.

Latest checkpoint-backed artifact/body and restricted-logits evidence from
`spark1` at commit `a5809ed`:

```text
NVIDIA NVFP4 artifact body gate:
tensor_contract_ready=1
tensor_count=9
tensor_bytes=2233222144
tensor_body_sample_ready=1
tensor_body_sample_count=22
tensor_body_sample_bytes=3236864
tensor_body_nonzero_bytes=3229801
tensor_body_sha256=edd995fc32fbe0ac33241a1ff7c17f2549ea71ca5f3ba0911821b91a2972f480

Luke NVFP4 artifact body gate:
tensor_contract_ready=1
tensor_count=9
tensor_bytes=2233222144
tensor_body_sample_ready=1
tensor_body_sample_count=22
tensor_body_sample_bytes=3236864
tensor_body_nonzero_bytes=3229801
tensor_body_sha256=edd995fc32fbe0ac33241a1ff7c17f2549ea71ca5f3ba0911821b91a2972f480

ZAI FP8 artifact body gate:
tensor_contract_ready=1
tensor_count=14
tensor_bytes=2068242880
tensor_body_sample_ready=1
tensor_body_sample_count=29
tensor_body_sample_bytes=3264960
tensor_body_nonzero_bytes=3256531
tensor_body_sha256=9e58397cf3e1785ce7a4b22616d1f0d9e730fed6c4e1059dbf9b972d5e21ce62

Synthetic CUDA decode-stage gate:
average_us=4523.755
maximum_us=4529.280
limit_us=10000.000
restricted_token=1009
real_lm_head=0

NVIDIA NVFP4 real-lm-head CUDA gate:
real_lm_head_fixture_ready=1
bytes=3145728
average_us=5301.888
maximum_us=5477.664
limit_us=10000.000
restricted_token=1028
real_lm_head=1
real_lm_head_max_logit_error=0.00000000

Luke NVFP4 real-lm-head CUDA gate:
real_lm_head_fixture_ready=1
bytes=3145728
average_us=4677.696
maximum_us=4953.600
limit_us=10000.000
restricted_token=1028
real_lm_head=1
real_lm_head_max_logit_error=0.00000000

ZAI FP8 real-lm-head CUDA gate:
real_lm_head_fixture_ready=1
bytes=3145728
average_us=4682.368
maximum_us=5482.176
limit_us=10000.000
restricted_token=1028
real_lm_head=1
real_lm_head_max_logit_error=0.00000000
```

Latest value-cache / `o_proj` evidence from `spark1` at commit `d9ee92a`:

```text
NVFP4 artifact gate:
ready=1
tensor_contract_ready=1
tensor_count=9
tensor_bytes=2233222144

FP8 artifact gate:
ready=1
tensor_contract_ready=1
tensor_count=14
tensor_bytes=2068242880

CUDA package gate:
average_us=5151.669
maximum_us=5480.000
limit_us=10000.000
restricted_token=1009
mtp_draft=1011
mtp_reject=1003

Generated-driver/orchestrator gate:
elapsed_us=5251.264
limit_us=10000.000
restricted_token=1009
mtp_draft=1011
mtp_reject=1003
```

The sparse-MLA package gate verifies:

```text
nvcc archive build
hardware sparse-MLA validator execution
module-library publication
model package compilation
generated driver loadability
```

Latest observed sparse-MLA timing:

```text
average_us=579.980
maximum_us=1229.486
limit_us=1500.000
```

This pass is valid, but the sparse-MLA maximum widened relative to the earlier
~414 us observation. Treat that as a performance-variance/regression signal to
retest before tightening the ceiling.

The decode-stage package gate verifies:

```text
nvcc archive build
hardware backend-submit validator execution
deterministic nonzero GLM-shaped smoke tensors
KV current-token write into a block-table-remapped cache slot
cached attention over context length 4 crossing two physical KV blocks
eight checked latent dimensions across four checked heads
four checked RoPE dimensions with non-identity rotation
attention output projection feeding restricted logits
restricted-vocabulary argmax
MTP draft accept/reject/commit counters
module-library publication
model package compilation
generated driver loadability
generated driver route resolution
orchestrator admission
orchestrator submit into CUDA backend
stream-ordered completion
runtime snapshot counters
zero host-staging bytes
zero device-memcpy bytes
```

Latest observed decode-stage timings:

```text
backend validator:
    fixture=remapped_nonzero_context4_h4_d8_r4
    average_us=5861.643
    maximum_us=6801.984
    limit_us=10000.000

generated-driver/orchestrator validator:
    fixture=remapped_nonzero_context4_h4_d8_r4
    elapsed_us=6499.168
    limit_us=10000.000
```

The latest decode-stage gate includes resident local MoE layer progression:

```text
post-attention RMSNorm
preselected top8 local expert routes
BF16 gate/up projections
SiLU gate
BF16 down projection
top8 weighted combine into layer_output_hidden_bf16
final norm/logits consume layer_output_hidden_bf16
```

This proves the resident driver can execute local expert math and continue the
layer progression on device. It does not prove production NVFP4 expert
throughput; the production next step is to replace the BF16 expert fixture with
pre-bound NVFP4 expert weights and grouped/persistent expert GEMM.

Latest layer-0 dense BF16 MLP evidence from `spark1` at commit `78a29ae`:

```text
command:
GLM52_MODEL_DIR=/home/spark1/models/hf/nvidia/GLM-5.2-NVFP4 \
GLM52_LOAD_LAYER0_DENSE_BF16=1 \
PATH=/usr/local/cuda-13.0/bin:$PATH \
make -C modules/glm52_resident_decode_stage validate MAX_STAGE_MICROSECONDS=10000

layer0_dense_bf16_fixture_ready=1
layer0_dense_bf16_bytes=452997120
real_lm_head_fixture_ready=1
real_lm_head_bytes=3145728
average_us=4029.398
maximum_us=4216.256
limit_us=10000.000
restricted_token=1104
mtp_draft=1011
mtp_reject=1003
real_lm_head=1
real_lm_head_max_logit_error=0.00000000
launch_chains=4
```

Package/generated-driver gate from the same branch:

```text
command:
GLM52_MODEL_DIR=/home/spark1/models/hf/nvidia/GLM-5.2-NVFP4 \
GLM52_LOAD_LAYER0_DENSE_BF16=1 \
PATH=/usr/local/cuda-13.0/bin:$PATH \
make -C modules/glm52_resident_decode_stage package MAX_STAGE_MICROSECONDS=10000

validation_recipe=glm52.resident_decode_stage.sm_121.layer0_dense_bf16.max_us_10000.v1
module_artifact=53677f6c3a80e8ed8be7f40563d2038394156090241832916fc2f9f0a94133f4
package_manifest_sha256=99aa2d3664af1b6720b06f94388aa4ffc8469b5f72e9218081f2756ec263749c
backend_average_us=3660.939
backend_maximum_us=3675.392
orchestrator_elapsed_us=4810.208
limit_us=10000.000
restricted_token=1104
real_lm_head=1
layer0_dense_bf16=1
layer0_dense_bf16_bytes=452997120
real_lm_head_max_logit_error=0.00000000
launch_chains=1
```

This removes the false assumption that GLM layer 0 can be represented by the
local-MoE path. GLM 5.2 has dense MLP in the first layers, and the resident
stage now has a distinct dense BF16 MLP progression mode. This is still not a
full token-equivalence pass because q/kv/o projection fixtures and attention
references are not yet loaded from the checkpoint in the same run.

Latest combined layer-0 BF16 attention+dense evidence from `spark1` at commit
`4ce5b4c`:

```text
command:
GLM52_MODEL_DIR=/home/spark1/models/hf/nvidia/GLM-5.2-NVFP4 \
PATH=/usr/local/cuda-13.0/bin:$PATH \
make -C modules/glm52_resident_decode_stage package_layer0_bf16 MAX_STAGE_MICROSECONDS=10000

validation_recipe=glm52.resident_decode_stage.sm_121.layer0_bf16.max_us_10000.v1
layer0_attention_bf16_fixture_ready=1
layer0_attention_bf16_bytes=330056704
layer0_dense_bf16_fixture_ready=1
layer0_dense_bf16_bytes=452997120
real_lm_head_fixture_ready=1
real_lm_head_bytes=3145728
backend_average_us=3910.037
backend_maximum_us=3919.136
orchestrator_elapsed_us=5336.672
limit_us=10000.000
restricted_token=1104
mtp_draft=1011
mtp_reject=1003
real_lm_head=1
real_lm_head_max_logit_error=0.00000000
module_artifact=7ca69b58e49fb20e4929cffb30555d57cdac6b7d204230215664292f28628db5
package_manifest_sha256=af75ba5fcacea09358d580cf32d0ce274707a0ab3171afffa16b35c42eb68551
```

This removes another synthetic-weight gap: the resident validator can now run
real layer-0 BF16 attention projection weights and real layer-0 dense MLP
weights in the same package/generated-driver path. It still does not prove
full token equivalence because the input hidden state and previous KV blocks are
synthetic, and the real q/kv/o + dense path is not yet compared to a full CPU or
external reference activation.

Latest input-embedding-backed layer-0 BF16 evidence from `spark1` at commit
`99a8908328327cbf5c1fa964fb9963f0dc0b47fb`:

```text
command:
GLM52_MODEL_DIR=/home/spark1/models/hf/nvidia/GLM-5.2-NVFP4 \
GLM52_INPUT_TOKEN_ID=1000 \
PATH=/usr/local/cuda-13.0/bin:$PATH \
make -C modules/glm52_resident_decode_stage package_layer0_embedding_bf16 MAX_STAGE_MICROSECONDS=10000

validation_recipe=glm52.resident_decode_stage.sm_121.layer0_embedding_bf16.max_us_10000.v1
input_embedding_bf16_fixture_ready=1
input_embedding_token=1000
input_embedding_bf16_bytes=12288
layer0_attention_bf16_fixture_ready=1
layer0_attention_bf16_bytes=330056704
layer0_dense_bf16_fixture_ready=1
layer0_dense_bf16_bytes=452997120
real_lm_head_fixture_ready=1
real_lm_head_bytes=3145728
backend_average_us=4168.555
backend_maximum_us=4187.136
orchestrator_elapsed_us=4954.816
limit_us=10000.000
restricted_token=1021
real_lm_head=1
real_lm_head_max_logit_error=0.00000000
backend_launch_chains=4
orchestrator_launch_chains=1
module_artifact=055c65a895624801cf591484c838a6c1e75a9be90a44077ee85af1b8dbef28db
package_manifest_sha256=a5eb3b5d56a82f2fe06c29f60762fea5b736a81acf3591a5a369d3b879861b65
```

This removes the synthetic input-hidden gap for the current B1 layer-0 gate:
the submitted hidden state can now be copied from a real checkpoint
`embed_tokens.weight` row. The previous KV cache is still seeded by the
validator, so the next inference-correctness gap is checkpoint-derived
prefill/KV plus an external reference activation comparison.

Latest checkpoint-prefilled layer-0 BF16 evidence from `spark1` at commit
`58e703e188f5adbedb265d24d03290ec16e64d60`:

```text
command:
GLM52_MODEL_DIR=/home/spark1/models/hf/nvidia/GLM-5.2-NVFP4 \
GLM52_INPUT_TOKEN_ID=1000 \
PATH=/usr/local/cuda-13.0/bin:$PATH \
make -C modules/glm52_resident_decode_stage package_layer0_prefill_bf16 MAX_STAGE_MICROSECONDS=10000

validation_recipe=glm52.resident_decode_stage.sm_121.layer0_prefill_bf16.max_us_10000.v1
input_embedding_bf16_fixture_ready=1
input_embedding_token=1000
input_embedding_bf16_bytes=12288
prefill_kv_bf16_fixture_ready=1
prefill_first_token=997
prefill_token_count=3
prefill_kv_bf16_bytes=49152
layer0_attention_bf16_fixture_ready=1
layer0_attention_bf16_bytes=330056704
layer0_dense_bf16_fixture_ready=1
layer0_dense_bf16_bytes=452997120
real_lm_head_fixture_ready=1
real_lm_head_bytes=3145728
backend_average_us=4486.293
backend_maximum_us=4591.136
orchestrator_elapsed_us=5371.968
limit_us=10000.000
restricted_token=1021
real_lm_head=1
real_lm_head_max_logit_error=0.00000000
backend_launch_chains=7
orchestrator_launch_chains=4
module_artifact=2f1a1728352f95d6221e87ca238ea2b16426cda8a31b590e262fe7cff7307669
package_manifest_sha256=5b1df9c60fb66e55877021d556a3b44df93b63338b55bb5e269f17cdae08cbbf
```

This removes the seeded prior-KV assumption for the checked B1 context-4
window. The validator now writes prior cache rows by running token IDs
997-999 through the resident CUDA stage, then the final token 1000 attends over
those actual cache rows. This is still not a full GLM inference pass because
the final post-attention and dense layer outputs are not yet compared against
an external checkpoint-derived reference activation.

Negative result from the first sampled-reference attempt at commit `974746d`:

```text
attention_norm sampled reference mismatch index=0
observed=1.24218750
expected=0.02941895
```

That did not indicate the layer-0 attention norm kernel was wrong. The failed
assumption was that `normalized_hidden_bf16` still held the attention-norm
output after a full stage run. It does not; the resident stage reuses the same
buffer for final RMSNorm before restricted logits. The reference gate was fixed
to reconstruct the attention-norm vector on the CPU from the final input
embedding row and checkpoint norm weights, then use that reconstructed BF16
vector as the q/kv projection reference input.

Latest sampled-reference layer-0 BF16 evidence from `spark1` at commit
`3fe97034e0deab3fc0c4b3aeab8cbf95f2fa30cb`:

```text
command:
GLM52_MODEL_DIR=/home/spark1/models/hf/nvidia/GLM-5.2-NVFP4 \
GLM52_INPUT_TOKEN_ID=1000 \
PATH=/usr/local/cuda-13.0/bin:$PATH \
make -C modules/glm52_resident_decode_stage package_layer0_reference_bf16 MAX_STAGE_MICROSECONDS=10000

validation_recipe=glm52.resident_decode_stage.sm_121.layer0_reference_bf16.max_us_10000.v1
input_embedding_bf16_fixture_ready=1
input_embedding_token=1000
input_embedding_bf16_bytes=12288
prefill_kv_bf16_fixture_ready=1
prefill_first_token=997
prefill_token_count=3
prefill_kv_bf16_bytes=49152
layer0_reference_sampled=1
layer0_attention_bf16_fixture_ready=1
layer0_attention_bf16_bytes=330056704
layer0_dense_bf16_fixture_ready=1
layer0_dense_bf16_bytes=452997120
real_lm_head_fixture_ready=1
real_lm_head_bytes=3145728
backend_average_us=4844.341
backend_maximum_us=4961.600
orchestrator_elapsed_us=5478.560
limit_us=10000.000
restricted_token=1021
real_lm_head=1
real_lm_head_max_logit_error=0.00000000
backend_launch_chains=7
orchestrator_launch_chains=4
module_artifact=8e4ff2ebf555bde8eee60a798ad3600785cd80c7322fd5d3bdc1d33d4e33232e
package_manifest_sha256=dd0108eda40dc1d3f3c27151c1257a92fc1265c733c44991cc2675674c3fc7c1
```

This adds sampled CPU-reference agreement for the live checkpoint-backed
layer-0 q/kv projections, q/kv RMSNorms, `o_proj`, residual, post-attention
RMSNorm, dense gate/up/SwiGLU/down, and dense residual boundaries. The next
correctness step is full-vector checksum comparison or an external reference
activation for the same layer, then multi-layer progression.

Latest full-reference layer-0 BF16 evidence from `spark1` at commit
`722dd307fc0f88ab9d7921b72203e7ed8ffc2b05`:

```text
command:
GLM52_MODEL_DIR=/home/spark1/models/hf/nvidia/GLM-5.2-NVFP4 \
GLM52_INPUT_TOKEN_ID=1000 \
PATH=/usr/local/cuda-13.0/bin:$PATH \
make -C modules/glm52_resident_decode_stage package_layer0_full_reference_bf16 MAX_STAGE_MICROSECONDS=10000

validation_recipe=glm52.resident_decode_stage.sm_121.layer0_full_reference_bf16.max_us_10000.v1
input_embedding_bf16_fixture_ready=1
input_embedding_token=1000
input_embedding_bf16_bytes=12288
prefill_kv_bf16_fixture_ready=1
prefill_first_token=997
prefill_token_count=3
prefill_kv_bf16_bytes=49152
layer0_reference_sampled=1
layer0_reference_full=1
layer0_reference_full_max_error=0.00195312
layer0_attention_bf16_fixture_ready=1
layer0_attention_bf16_bytes=330056704
layer0_dense_bf16_fixture_ready=1
layer0_dense_bf16_bytes=452997120
real_lm_head_fixture_ready=1
real_lm_head_bytes=3145728
backend_average_us=5087.840
backend_maximum_us=5274.624
orchestrator_elapsed_us=5368.064
limit_us=10000.000
restricted_token=1021
real_lm_head=1
real_lm_head_max_logit_error=0.00000000
backend_launch_chains=7
orchestrator_launch_chains=4
module_artifact=762b7d2125110a690a77168be8bcb2fd2aa784abb8d4b5a003b74db716c79ef3
package_manifest_sha256=6c5aeae90d675f5fbafe53d75be31bb2711236f514923ceee3c35b8de781d173
```

This converts the prior "sampled only" layer-0 gate into a full output-side
BF16 reference check for `o_proj`, attention residual, post-attention RMSNorm,
SwiGLU activation, and dense-down residual. It still deliberately does not
teach the generic SparkPipe runtime about GLM tensor names or reference math.

## What this proves

The generated GLM 5.2 decode-stage `model_driver.so` can be loaded by the
SparkPipe runtime, attached to an orchestrator route, admitted through the
driver scheduler, and submitted into the real CUDA backend on SM121 hardware.

It also proves the current resident decode-stage control path does not use
host-staged frame buffers or serving-path device copies for the submitted
frame. The counters are checked by the validator, not just printed.

## New nonzero fixture

The decode-stage validator now seeds deterministic nonzero tensors directly
into the resident device buffers. The pattern is intentionally small enough for
a host reference but still uses full-shape resident allocations and the real
CUDA chain.

The checked invariants are:

```text
logical context tokens 61..64 resolve through physical block table [1,0]
cache slot 0 receives the current KV latent value
cache slot 0 receives non-identity-rotated current key RoPE values
query latent is nonzero for four checked heads and eight checked dimensions
rotated query RoPE is nonzero for four checked heads and four checked dimensions
attention outputs match a host softmax reference over remapped slots 125,126,127,0
restricted logits depend on attention output projection, not only input hidden
local MoE route output and layer output become nonzero
restricted argmax selects token 1009
MTP drafts token 1011 twice
MTP accepts the first draft and rejects the second against token 1003
MTP event counters are accepted=1 rejected=1 committed=2 rollback=1 cancelled=0
phase completion marker is written
driver/orchestrator counters remain clean
```

## What this does not prove yet

This is not a full GLM 5.2 inference pass. The decode-stage validator now uses
real checkpoint `lm_head.weight` rows for restricted logits when
`GLM52_MODEL_DIR` is set, but the rest of the layer path still uses
deterministic nonzero smoke tensors. It does not yet load all checkpoint
projection, attention, MoE, MTP, and norm weights or compare final logits
against a known GLM artifact.

The next correctness gate must replace smoke tensors with deterministic
nonzero GLM fixtures and check:

```text
multiple positions
larger nonzero attention dimension/head coverage
checkpoint-derived cached attention and MoE references
checkpoint-derived dense MLP references for the first dense layers
checkpoint-derived attention q/kv/o references for layer 0
MTP draft and verify/commit behavior with varied target patterns
runtime snapshot counters after real tensor work
```

## Next experiment hypothesis

Previous hypothesis:

```text
If deterministic nonzero GLM-shaped tensors are used, the current resident
decode-stage CUDA chain will preserve the same driver/orchestrator invariants
but expose numerical gaps in KV layout, sparse-index semantics, restricted
logits, or MTP verification before it exposes transport-level failures.
```

Result:

```text
The B1 context-length-4 remapped fixture passed on GB10. KV write layout,
two-block cached attention over slots 125,126,127,0, restricted argmax, and MTP
accept/reject matched the host reference through both direct backend submit and
generated-driver/orchestrator submit. The fixture was then broadened to four
heads, eight latent dimensions, four RoPE dimensions, non-identity key/query
RoPE, and attention-output-projection-fed restricted logits; that also passed
on Spark1 / GB10 under the 10000 us package ceiling.
```

Next hypothesis:

```text
The next likely gap is not the driver boundary or the small remapped-cache
layout. It is real GLM tensor semantics: varied positions, varied sparse-token
selection, multi-block KV checksums, and checkpoint quantization layout.

The raw tensor gate narrowed that: NVFP4/BF16 attention tensors match the first
resident checkpoint contract, while FP8 needs a separate quantized-weight
contract and lowering/module path.
```

If it fails:

```text
Do not tune constants.
Classify the failure as a layout, arithmetic, scheduling, or ownership bug.
Update this log with the failed assumption, then fix the model before rerunning.
```
