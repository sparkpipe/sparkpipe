# Survey: public inference performance for Qwen 3.8 Max (Qwen3.8-2.4T-A95B)

Owner: qwen38-max MODEL agent. Public-source survey (GitHub, Hugging Face, NVIDIA
forums, vLLM/Artificial-Analysis/vendor blogs). No commits/pushes. Method:
web_search + direct page fetch (curl). All sources are dated 2026; the model
open-sourced ~Aug 12-13, 2026, so the serving literature is young.

**Headline caveat (the user's own):** almost every published throughput number is
for a *quantized* checkpoint (FP8/NVFP4/MXFP4/Int4) on B300/GB300-class hardware.
**No public source publishes a B1/B8/B16 decode breakdown for the 2.4T model**, and
none publishes BF16 serving (the 4.45 TiB BF16 checkpoint is not a serving target -
vLLM's own recipe starts at FP8). The honest anchors are therefore: (a) the vLLM
"output tok/s/**user**" low-latency numbers, and (b) single-stream (B1) results from
our *sibling models on GB10* (sm_121) - the exact SparkPipe hardware.

---

## 1. Results table

| Ref | Source (URL) + date | Hardware | TP | Quantization | Small-batch decode | Key techniques |
|---|---|---|---|---|---|---|
| Qwen3.8-2.4T vLLM day-0 | vllm.ai/blog/2026-08-12-qwen3.8 · Aug 12 | 16xB300/GB300 | TP16 | FP8 (E4M3 b128) | **130 tok/s/user** (no MTP) | kv-cache fp8, MTP-3 |
| same | same | 8xB300/GB300 | TP8 | NVFP4 (1.32 TiB) | **133 to 304 tok/s/user** (MTP-3) | flashinfer_cutedsl, MTP-3 |
| Qwen3.8-2.4T vLLM recipe | dev.to (nick_k_gpus_market) · Aug 15-16 | 16xB300/GB300 | TP16 | FP8 | 130 to **307 tok/s/user** (MTP-3) | MTP-1 = 64.8% accept; MTP-3 much better |
| same | same | 8xB300/GB300 (NVL4) | TP8 | NVFP4 | 133 to 304 tok/s/user | TP divides 64 heads; TP4xPP3 verified on 12xGB300 |
| Qwen3.8-2.4T quality | vLLM blog / AMD Quark · Aug | 8xMI355X | TP8 | MXFP4 vs FP8 | - (no throughput) | GSM8K 97.49 = 97.49 (SGLang) |
| Qwen3.5-397B-A17B (same GDN family) | vllm.ai/blog/2026-08-06-qwen35-25k-tps · Aug 6 | GB200 NVL72 (P/D) | DEP | NVFP4 | not measured below c=32 | Blackwell GDN prefill 1.02-5.78x; async sched; HMA GDN-state transfer |
| GLM-5.2-NVFP4 (744B/40B) | vllm.ai/blog/2026-07-23-glm-5.2-nvfp4-b300-pd · Jul 23 | 24xB300 (4P1D) | TP1+EP8 | NVFP4 | 17 ms TPOT (~59 tok/s/user) | spec padding 40->22ms, MRV2 11%, MTP-3, CUDA-graph FULL_DECODE_ONLY |
| DeepSeek-V3.2/R1 | vllm.ai/blog/2026-02-13-gb300-deepseek · Feb 13 | 2xGB300 | TP2 | NVFP4 | MTP accept >80% at c<=256 | FP4 MoE kernel; EP2 vs TP2 (TP decode 50%-2x TPOT) |
| Qwen3.8-27B (sibling, GB10) | forums.developer.nvidia.com/t/...380258 · Aug 15 | 1x DGX Spark | TP1 | FP8 (official) | 16-conc: 65.6 / 99.5 / 104.4 tok/s; TPOT 148-160 ms | - |
| same | same | 1x DGX Spark | TP1 | NVFP4 (Unsloth) | 16-conc: 87.9 / 132.1 / 134.4 tok/s (+29-34%); TPOT 115-121 ms | 4-bit MLP + 8-bit attn + FP8 KV |
| Qwen3.8-27B (sibling, GB10, B1) | github.com/XtraSaltyDev/qwen3.8-27b-nvfp4-dgx-spark-speed · Aug 15 | 1x DGX Spark | TP1 | NVFP4 | **15.6 to 23.4 tok/s single-stream** (+49.9%) | MTP layer->per-channel FP8; 1-then-5 draft; async drain |
| DSV4-Flash (sibling, GB10) | github.com/joesinvestments/...TP4-4x-DGX-Spark · Jun | 4x DGX Spark | TP4 | FP8/NVFP4 | **123.1 tok/s single-stream** (104.2 pub); 57.9 at 150K ctx | DSpark k=7, probabilistic draft 34.3% accept, warmup depth |
| GLM-5.2 Int4/Int8 (GB10) | forums...376831 · Jul 14 | 8x DGX Spark | TP8 | Int4/Int8 | **33-55 tok/s single-stream**; peak 54-66 | W4A8 MoE, SM120 sparse MLA, draft_tp_size=1, NCCL 16MB |

Notes: "tok/s/user" in vLLM's low-latency suite is a *per-request* rate (low
concurrency, ~B1-B16 shape) - the closest public thing to a small-batch number for
the 2.4T model. The Qwen3.8-27B and DSV4-Flash rows are the same *model family* /
*sibling flagship* on the **same GB10 (sm_121) silicon** SparkPipe targets.

---

## 2. Best reported result PER quantization level (matched-precision comparison)

| Quantization | Best reported | Hardware | Comparable to our BF16/FP8 baseline? |
|---|---|---|---|
| BF16 (4.45 TiB) | none (not a serving target) | - | YES our exact precision; no public number exists |
| FP8 (official, 2.27 TiB) | **130 tok/s/user** (no MTP) -> **307** (MTP-3) | 16xB300/GB300 TP16 | YES directly comparable - our pack is BF16 spine + FP8_E4M3 experts |
| NVFP4 (1.32 TiB) | 133 -> 304 tok/s/user (MTP-3) | 8xB300/GB300 TP8 | NO - 4-bit weights, ~2x less memory/bandwidth |
| MXFP4 (routed experts) | quality-parity only (GSM8K 97.49=97.49); no throughput | 8xMI355X | NO (but == our documented MXFP4 option in TOP10 #7) |
| Int4/Int8 (community GB10) | 33-55 tok/s single-stream (GLM-5.2, different arch) | 8xGB10 | NO - heavily quantized AND DSA (not GDN) AND different model |
| FP8 (GB10, same family) | Qwen3.8-27B: 104.4 tok/s @16-conc; ~15-23 tok/s single-stream | 1xGB10 | partial - same precision & hardware, but a 27B model, not the 95B-active MoE |

### The precision-matched number we should target

- Precision-matched (FP8) SOTA is vLLM's 130 tok/s/user on 16xB300/GB300. That is
  *not* attainable on GB10: B300/GB300 have ~8x HBM3e bandwidth vs GB10's LPDDR5X
  (~250 GB/s in our audit, docs/QWEN38_MAX_AUDIT.md section 3). Use it as the
  *B300-class* ceiling, not the GB10 bar.
- The honest GB10 bar at matched precision is our sibling fleet: **DSV4-Flash
  (700B-MoE) 123.1 tok/s single-stream on 4xGB10 (TP4, DSpark k=7)**, and
  **Qwen3.8-27B ~104 tok/s @16-conc / ~23 tok/s single-stream on 1xGB10**. Qwen3.8-Max
  activates 95B/token (~1.7x DSV4-Flash's active work), so the realistic GB10 FP8
  target is **~50-60 tok/s single-stream with MTP** (123 x 57/95), scaling to
  ~130 tok/s only past B300-class bandwidth.

---

## 3. Ranked experiment ideas (survey-informed)

Each: expected gain, code-size delta (vs our clone), owner. All fit the landed
TOP10_QWEN38-MAX.md; the survey validates their priority and adds the MTP-depth +
drafter-sharding specifics.

1. Serve the packed MTP with depth-3 + draft-layer FP8. The single biggest public
   Qwen3.8 lever: MTP-3 takes FP8 from 130 to 307 tok/s/user (2.36x) and NVFP4
   133 to 304; MTP-1 is only 64.8% acceptance ("much less convincing"). XtraSaltyDev:
   quantizing the MTP proposal layer to per-channel FP8 + 1-then-5 draft bought +49.9%
   B1 on GB10. Gain: **2-2.4x per-request decode** (accurate-slow -> match/exceed SOTA).
   Code: **+~500 lines** (draft forward reusing target kernels + shared V-01 verify).
   Owner: qwen38-max + speculation.
2. Drafter-unsharded + local-argmax reduction for MTP. GLM-5.2 GB10 uses
   draft_tensor_parallel_size=1 (drafter pays no TP8 collectives per draft step) and
   vLLM PR #46448 replaces full-vocab AllGather with a 2xTP-size reduction for MTP
   argmax. Direct fit for our TP16 head-parallel + head vocab-sharding.
   Gain: **10-25%** once MTP lands. Code: **+~60-120 lines**. Owner: qwen38-max.
3. Uniform batch shapes + CUDA-graph FULL_DECODE_ONLY (kill the mixed-batch slow path).
   GLM-5.2's single biggest win was speculative-padding P/D handoff rows (40->22 ms
   TPOT); our module has the same disease (single-slot, per-frame cudaStreamSynchronize,
   mixed prefill/decode shapes - TOP10 #6). Gain: **~40% TPOT** (toward match SOTA).
   Code: **+~80-150 lines** (multi-slot stream-ordered dispatch + graph capture).
   Owner: qwen38-max.
4. Blackwell/sm121 GDN prefill + post-conv MTP decode kernel (common). vLLM's
   FlashInfer GDN kernel = 1.02-5.78x on Qwen3.5, and PR #51674 adds a fused post-conv
   MTP decode kernel for Qwen3.5 GDN - exactly our GDN-decode hot path (the ~330
   per-(row,head) blocks / token). Gain: **up to ~2-5x GDN layer** (exceed SOTA on
   GDN layers). Code: **+~200-300 lines** common kernel. Owner: CUDA-KERNELS.
5. GDN state paging via the tier (validate against public HMA/NIXL GDN-state
   transfer). vLLM's hybrid SSM-FA disaggregation (#36687, #41869, conv-state 3-read
   transfer #37635) is the public proof that GDN state pages correctly across workers
   - the exact thing our 4096-byte placeholder record (module.c:35,663) fails to do.
   Gain: **resident capacity** (enables the concurrency every level needs).
   Code: **+~40 lines**. Owner: qwen38-max (TOP10 #3).
6. Async scheduling / sync-free dispatch. vLLM credits --async-scheduling for
   crossing 25K TPS/GPU (after two race fixes); our adapter's per-frame
   cudaStreamSynchronize is the same stall. Gain: **pipeline overlap, +20-40%
   throughput at B>=8**. Code: **+~80-120 lines**. Owner: qwen38-max (TOP10 #6).
7. NVFP4/MXFP4 requantize routed experts. vLLM: NVFP4 = 2x memory cut (8 GPUs vs
   16), quality >= FP8 (GSM8K 90.37 vs 89.61; AIME 92.22 vs 87.78); AMD MXFP4 = GSM8K
   97.49 parity. Unlocks sm121 B1 expert kernels (our TOP10 #7-8). Gain: **fits 148 to
   74 GB/rank + 5-10x B1** (match/exceed SOTA). Code: **+~30-60 lines** packer.
   Owner: qwen38-max + coordinator (quality decision).
8. Expert-parallel vs TP for decode (re-validate T2/T3). DeepSeek-R1: TP2 decode
   shows 50%-2x TPOT vs EP2 (all-to-all "large packet, low frequency" wins); our
   T1-T4 residual all-reduce schedule (TOP10 #10) should A/B against an EP all-to-all
   now that expert-sharded MoE is landed. Gain: **up to ~2x TPOT at small batch**.
   Code: **~0-80 lines** (config/backends). Owner: qwen38-max + transport.
9. fastsafetensors lazy-load + persistent model store. vLLM: 545->306 s load; cheap
   operational win, no decode gain. Code: **~0** (recipe). Owner: qwen38-max.
10. Prefix-cache-aware scheduling for agentic workloads. DSV4-Flash's 123 tok/s
    recipe is built on 96-98% prefix-cache hits (agent sessions re-send trunks);
    our adapter has no prefix-cache path. Gain: **effective per-request 2-10x on
    re-sent context**. Code: **+~150-250 lines**. Owner: qwen38-max + kv-cache.

---

## 4. Summary line

**At matched FP8 precision, SparkPipe on GB10 should match/beat ~50-60 tok/s
single-stream decode (with MTP-3) - the DSV4-Flash 123 tok/s (4xGB10, TP4, DSpark)
sibling bar scaled to Qwen3.8-Max's 95B active per token - and treat vLLM's
FP8 130 to 307 tok/s/user (16xB300/GB300) as the B300-class ceiling, not a GB10 target.**
