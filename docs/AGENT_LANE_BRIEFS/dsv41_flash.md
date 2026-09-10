# Lane brief: DeepSeek-V4.1-Flash (dsv5-flash) — NEW ARCHITECTURE

Worktree: /Users/mac/lane-dsv5flash
Branch: lane/dsv5-flash-dev (reset onto origin/main 50bd0d3, the E2E-proven
weightd-mesh dataflow; the stale-main incident base 14df85a was discarded)
Agent: A-dsv5 (respawn of a rate-limit casualty)
Nodes: spark6 (build/publish) + spark2 (cells, shared with hy4 — queue-arbitrate)
Node split exclusive: sparks 0-8 only. Queue v2 is the only cross-node path.

## Mission

Port the DeepSeek-V4.1-Flash resident decode stage onto the converged stack.
This is NOT a dsv4 re-parameterization: V4.1-Flash is a new architecture
(CED + CSA2 + Single-Pass mHC + FP4 KV + Engram + DSpark) that obsoletes the
dsv4 flash/pro module line. modules/dsv4_resident_decode_stage is a CONVENTIONS
donor only (adapter shape, pack layout, stagepack usage); every geometry fact
comes from the pinned HF sources below.

## Source of truth (weights still downloading — HF is the only complete source)

Repo: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
Revision: dba1be0a40aa45a94ad051997016db3960a90277
48 shards, 510,296,708,312 file bytes, 96,085 tensors,
510,286,023,000 tensor bytes. Largest two shards (~101.5 GiB each) are the
two Engram embedding tables. Contract: model_contracts/dsv41_flash_authoritative.json.

## Geometry facts table (fact → pinned source)

| Fact | Value | Source |
|---|---|---|
| Architecture / model_type | DeepseekV41ForCausalLM / deepseek_v41 (text deepseek_v41_text) | config.json |
| Layers / hidden / vocab | 40 / 5120 / 129280 | config.json text_config |
| Heads / head_dim / rope dim | 64 / 512 / 64 (yarn factor 16, theta 10000, orig 65536) | config.json + tech report §4.2.1 |
| q_lora / q_norm | 1280 (wq_a [1280,5120], wq_b [32768,1280] F8_E4M3), q_norm BF16[1280] | shard-3 safetensors header |
| KV latent | 512 (wkv [512,5120]), kv_norm BF16[512] | shard-3 header |
| Output projection | o_lora_rank 1024, o_groups 8: wo_a [8192,4096], wo_b [5120,8192]; value head dim 64 | config.json + shard-3 header |
| Attention sink | attn_sink F32 [64] per head | shard-3 header |
| SWA window | 128, FP8 SWA KV | tech report §2.4.4, §4.2.1 |
| CSA2 modes / layers | Full 2,8,14,20; Reindex 24,28,32,36; Reuse the rest of 2..37; SWA-only 0,1,38,39 | config.json kv/index/candidate source ids + tech report §2.3.1, Fig. 3 |
| Compression ratios | m=2 encoder (2..19), m=1 decoder (20..37), 0 on 0,1,38,39 + 3 dspark blocks | config.json compress_ratios |
| Hierarchical indexer | 32 heads x 128 dim, top-512; candidate pool 2048 blocks x 8 = 16384 positions, built by layer 20 | config.json + tech report §2.3.2 |
| Compressor | BF16 wkv/wgate [512,5120] + norm [512] on layers 2,8,14,20 only | shard headers + index census |
| MoE | 384 routed top-6 + 1 shared, moe_int 2304, noaux_tc, sqrtsoftplus, scale 1.5 | config.json + tech report §4.2.1 |
| Router | gate.weight BF16 [384,5120], gate.bias F32 [384], gate.bias_vl F32 [384] | shard-3 header |
| Expert codec | packed FP4 as I8 (2/byte), UE8M0 scale per 32 input elems: w1/w3 [2304,2560] s[2304,160], w2 [5120,1152] s[5120,72] | shard-3 header + quantization_config expert_dtype fp4 |
| Shared expert | F8_E4M3 w1/w3 [2304,5120], w2 [5120,2304], scales [72,160] | shard-3 header |
| Dense codec | FP8 E4M3 + UE8M0 scale per 32x32 block | config.json quantization_config |
| mHC (hyper-connections) | Single-Pass mHC, mult 4: base F32[24], fn F32[24,20480], scale F32[3], attn+ffn | config.json hc_mult + tech report §2.4.1 |
| Global KV cache | FP4 E2M1, one E4M3 scale per 16 channels, no global second scale, quantized after rope, 890 B/token | tech report §2.4.4 |
| Engram | layers 1,14; 2 modules; ~16M-entry x 8-head tables; embed F8_E4M3 [384006168,256]; HOST-RESIDENT (RDMA prefetch), not GPU pack | config.json + tech report §2.4.2 |
| DSpark | 3 blocks (SWA-128), draft 5, markov rank 256, 128 routed experts top-3; markov/confidence heads only on mtp.2; main_proj F8 [5120,15360] | config.json + shard-44/47 headers + tech report §2.4.3 |
| Expert bytes | 17,694,720 B/expert packed; 271.8 GB all routed; TP16 → 16.99 GB/rank | derived from shard headers |
| Manifest capacity | 40x384x3x2 = 92,160 v2 records < 131,072 SPARK_WEIGHTD_RANGE_COUNT_MAX; per-expert bytes < 64 MiB cap | include/sparkpipe/spark_weightd.h |

Open facts (resolve at contract freeze): o_groups=8 does not divide TP16
(wo_a/wo_b handling needs exemption or split); exact wq_b rope split layout
(32768 rows = 64 heads x (448 nope + 64 rope)); compressor gate semantics at
m=1 vs m=2; engram addressing keys at decode time.

## Donor rules

- glm5_next_resident_decode_stage = readiness template (glm53flash converged
  stack E2E): weightd lazy attach (SparkWeightdAttachRequested fail-closed,
  SparkWeightdLazyPackCreateChecked, SparkWeightdMapAcquire leases,
  SparkWeightdLazyPackSlice), fopen of pack ONLY for header/directory
  metadata via SparkStageModulePackRead, seen-bit inventory validation.
- qwen4_flash shows the hc (hyper-connection) weight plumbing pattern.
- modules/dsv4_resident_decode_stage = DeepSeek-family conventions donor;
  its geometry is obsolete. Do NOT copy structures from it.

## Design constraints from line one

- NO credit-binding anything: no CreditBinding* symbols, no binding tables,
  no binding allocs. The new engine has no binding population; qwen4_flash's
  tp_credit_bindings must NOT be imitated.
- Weightd-only lazy attach via the shared seam (stage_module_common +
  spark_weightd*). Fail-closed SPARK_FAIL on absent weightd. Never a direct
  pack read/mmap/open for weights in driver code (audit A-0023 is the
  cautionary example).

## Milestone ladder (each rung = verifiable exit, commit + report)

M1 Contract freeze: model_contracts/dsv41_flash_authoritative.json (this PR,
WIP) — pinned HF revision + digests; warm dd-probe + hash-verify against the
pinned LFS oids closes it.
M2 Geometry header: model-families/dsv41_flash/include/sparkpipe/
spark_dsv41_flash_model.h generated from the contract; conformance test binds
header constants to config.json values.
M3 Stagepack contract: source/spark_dsv41_flash_stagepack_format.h — pack
header (mirror glm5_next layout), tensor kinds for the V4.1 fact table above,
variants: base / dense_probe / dspark_sidecar.
M4 Packer + manifest: tools/dsv41_flash_pack_synthesize.c (synthetic) then
the warm-source packer; .experts v2 manifest producer (48B records:
layer, expert, kind, zero, offset, bytes, ck128) mirroring
tools/glm5_next_experts_manifest.c; SPINE vs EXPERT bytes reported
separately in receipts.
M5 Validator: independent expectations (own math from pinned HF sources, NO
driver imports) prove packer==checkpoint; verifier proves pack==packer;
boundary ranks checked.
M6 Module skeleton: modules/dsv41_flash_resident_decode_stage — driver ABI v4
entry points, weightd lazy attach, host compile on spark6, then CUDA cell
gates on spark2 via queue v2 (cpu-class gates on the spark, never the mac).
M7 Hill climb: measure each stage; where 10-100x slower than theoretical, ask
why it exists, DELETE it. DSpark sidecar LAST (separate sysadmin sidecar
files, MTP-sidecar law).

## Ports / topology

TP16 preferred (384 experts → 24/rank; 64 heads → 4/rank; 32 index heads →
2/rank), TP4xPP4 standard fallback, otherwise documented exemption (o_groups
8 is the known TP16 exemption candidate). Port block 27648 requested from
PORT_LEDGER spares — env-parameterized until mgr1 ratifies; collective
listens + control OUT of the matrix zone.

## Gates

Weights gate BEFORE any pack build: warm download complete + dd-probe +
hash-verify vs pinned LFS oids. Pack cells under sparkcap (sudo -n
systemd-run --scope -q -p MemoryMax=4096M -p MemoryHigh=2900M). Manifest +
sha256sums regenerated LAST. Push flake → escalate once → Git Data API.
