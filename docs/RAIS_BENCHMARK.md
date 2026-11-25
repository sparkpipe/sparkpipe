# RAIS v1.1 — Roofline-Anchored Inference Score

**Version:** 1.1  
**Reference scorer:** [`tools/rais_score.py`](../tools/rais_score.py)  
**Reference runner:** [`tools/rais_bench.py`](../tools/rais_bench.py)  
**Fleet lane authority:** [`deployment/lanes.json`](../deployment/lanes.json)

RAIS (Roofline-Anchored Inference Score) is an engine-agnostic, hardware-comparable inference benchmark specification. Rather than reporting raw tokens per second — which cannot be compared across model sizes, quantization formats, or hardware clusters — RAIS anchors every workload to the exact physical lower bound ($T_{\text{floor}}$) imposed by compute, memory bandwidth, network collectives, storage bandwidth, and usable memory capacity on the target hardware, and pairs roofline efficiency with speculation quality, operational systems physics, and hard binary correctness gates.

An engine running any checkpoint on any supported hardware profile receives a standalone score in $[0, 100]$, enabling direct apples-to-apples comparisons (`SparkPipe X` vs `vLLM / SGLang / TensorRT-LLM / spark-ds41f Y` on identical hardware) and deterministic ROI ranking for optimization work. Optional **Stack Badges** (e.g. `Zero-NVIDIA NVRM + SKL`, `CUDA 13.0`, `Apple Metal 4.0`, `vLLM + TileLang`) record runtime provenance as display-only metadata with zero impact on the numeric score.

---

## 1. The Three Physical Hardware Rooflines

For a model $M$, hardware profile $H$ with $N$ accelerator ranks, batch size $B$, context length $C$, and forward-step row count $R$ per sequence ($T_{\text{rows}} = B \cdot R$; $R = 1$ for plain decode, $R = K + 1$ for speculative verify, $R = L$ for prefill of $L$ tokens), any lossless inference engine must satisfy three independent physical lower bounds.

### 1.1 Compute Roofline ($T_{\text{compute}}$)

Every token row requires $2 P_{\text{active}}$ multiply-add FLOPs across non-expert linear projections ($P_{\text{non\_expert}}$) and active routed MoE experts ($P_{\text{active\_expert}}$), plus causal attention FLOPs ($F_{\text{attn}}$) over the attended context $C_{\text{eff}}$:

$$F_{\text{linear}} = 2 \cdot P_{\text{non\_expert}} \cdot T_{\text{rows}} + 2 \cdot P_{\text{active\_expert}} \cdot T_{\text{rows}}$$

$$F_{\text{attn}} = T_{\text{rows}} \cdot C_{\text{eff}} \cdot \text{FLOPs}_{\text{attn/tok/ctx}}$$

where $C_{\text{eff}} = \min(C, C_{\text{sparse}})$ for sparse-attention layers (DSA / CSA2) and $C_{\text{eff}} = C$ for dense full-attention layers (using average causal context $C_{\text{prefix}} + L/2$ during prefill). Matching each term to the accelerator's precision-specific peak throughput across $N$ ranks:

$$T_{\text{compute}} = \frac{2 P_{\text{non\_expert}} T_{\text{rows}}}{N \cdot \text{TFLOPS}_{\text{non\_expert}}} + \frac{2 P_{\text{active\_expert}} T_{\text{rows}}}{N \cdot \text{TFLOPS}_{\text{expert}}} + \frac{F_{\text{attn}}}{N \cdot \text{TFLOPS}_{\text{bf16}}}$$

### 1.2 Memory Bandwidth Roofline ($T_{\text{memory}}$)

A forward step must stream each touched weight byte once across the $N$-rank fleet, plus KV cache, sparse indexer keys, sliding-window KV, and recurrent state (GDN / KDA):

1. **Non-expert spine weights** (read once per step, amortized over all $T_{\text{rows}}$):
   $$W_{\text{spine}} = P_{\text{non\_expert}} \cdot b_{\text{non\_expert}}$$
2. **Routed MoE experts** (coupon-collector touched-expert union law for $E$ experts, top-$K$ routing):
   $$u(T_{\text{rows}}, E, K) = 1 - \left(1 - \frac{K}{E}\right)^{T_{\text{rows}}}$$
   $$W_{\text{experts}}(T_{\text{rows}}) = \left(P_{\text{active\_expert}} \cdot \frac{E}{K} \cdot b_{\text{expert}}\right) \cdot u(T_{\text{rows}}, E, K)$$
   At $T_{\text{rows}} = 1$, $u(1, E, K) = K / E$, recovering exact single-token active expert weight bytes. As $T_{\text{rows}}$ grows, $u \to 1$ and all $E$ experts are read once per step.
3. **KV cache, sparse indexer scan, sliding-window KV, and recurrent state:**
   - **Decode ($B$ sequences at context $C$):**
     $$W_{\text{kv}} = B \cdot \left(C_{\text{eff}} \cdot b_{\text{kv/tok}} + C \cdot b_{\text{idx/tok}} + \min(C, W_{\text{swa}}) \cdot b_{\text{swa/tok}} + W_{\text{state/seq}}\right)$$
   - **Prefill ($L$ new tokens behind cached prefix $C_{\text{prefix}}$):**
     $$W_{\text{kv}} = B \cdot \left(C_{\text{prefix,eff}} \cdot b_{\text{kv/tok}} + (C_{\text{prefix}} + L) \cdot b_{\text{idx/tok}} + \min(C_{\text{prefix}} + L, W_{\text{swa}}) \cdot b_{\text{swa/tok}} + W_{\text{state/seq}}\right) + 2 T_{\text{rows}} \cdot b_{\text{kv/tok}}$$

With total memory traffic $B_{\text{mem}} = W_{\text{spine}} + W_{\text{experts}}(T_{\text{rows}}) + W_{\text{kv}}$:

$$T_{\text{memory}} = \frac{B_{\text{mem}}}{N \cdot BW_{\text{mem,rank}}}$$

### 1.3 Network Collective Roofline ($T_{\text{network}}^{\min}$)

For single-rank hardware ($N = 1$), $T_{\text{network}}^{\min} = 0$. For multi-rank tensor-parallel hardware ($N > 1$), a model with $L_{\text{layers}}$ layers performs at least $R_{\text{ops}} = 2 L_{\text{layers}}$ full-hidden all-reduce collectives per step on BF16 payloads of size $M_{\text{payload}} = T_{\text{rows}} \cdot d_{\text{hidden}} \cdot 2\text{ bytes}$. By the ring / reduce-scatter+all-gather lower bound, each rank moves at least $\frac{2(N-1)}{N} M_{\text{payload}}$ wire bytes per collective and pays the fabric hop latency floor $\tau_{\text{hop}}$ per round:

$$T_{\text{network}}^{\min} = R_{\text{ops}} \cdot \tau_{\text{hop}} + \frac{R_{\text{ops}} \cdot \frac{2(N-1)}{N} \cdot M_{\text{payload}}}{BW_{\text{net,rank}}}$$

### 1.4 Overlapped Floor, Serialized Floor, Overlap Gain, and Standalone Score

- **True Overlapped Physical Floor:**
  $$T_{\text{floor}} = \max\!\left(T_{\text{compute}},\; T_{\text{memory}},\; T_{\text{network}}^{\min}\right)$$
- **Binding Bottleneck:** $\arg\max_{r \in \{\text{compute},\, \text{memory},\, \text{network}\}} T_r$.
- **Serialized Floor (zero overlap):**
  $$T_{\text{serial}} = T_{\text{compute}} + T_{\text{memory}} + T_{\text{network}}^{\min}$$
- **Maximum Possible Overlap Gain:**
  $$G_{\text{overlap}}^{\max} = \frac{T_{\text{serial}}}{T_{\text{floor}}} \in [1, 3]$$
- **Sub-Metric Roofline Efficiency Score ($S_k$):**
  $$S_k = \min\!\left(100,\; 100 \times \frac{T_{\text{floor},k}}{T_{\text{actual},k}}\right) = \min\!\left(100,\; 100 \times \frac{\text{Rate}_{\text{actual},k}}{\text{Rate}_{\text{floor},k}}\right)$$

Because $T_{\text{floor},k}$ is a pure function of `(model geometry, hardware profile, workload shape)` and never references engine internals, any two engines evaluated on the same `(model, hardware)` share the exact same denominator. An engine that overlaps collectives with GEMMs can surpass $T_{\text{serial}}$ and approach $S_k = 100$, while an unoverlapped engine hits a ceiling at $100 / G_{\text{overlap}}^{\max}$.

---

## 2. Reference Hardware Profiles

| Profile ID | Ranks ($N$) | Mem BW / Rank | Aggregate Mem BW | BF16 / FP8 / FP4 TFLOPS / Rank | Net BW / Rank | Hop Latency ($\tau_{\text{hop}}$) | NVMe BW / Rank | RAM / OS Reserve (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `16x-gb10-1x100g` | 16 | 273 GB/s | 4,368 GB/s | 90 / 210 / 420 | 12.5 GB/s | 16 us | 5.0 GB/s | 128 / 12 GiB |
| `16x-gb10-2x100g` | 16 | 273 GB/s | 4,368 GB/s | 90 / 210 / 420 | 25.0 GB/s | 16 us | 5.0 GB/s | 128 / 12 GiB |
| `4x-gb10-1x100g` | 4 | 273 GB/s | 1,092 GB/s | 90 / 210 / 420 | 12.5 GB/s | 12 us | 5.0 GB/s | 128 / 12 GiB |
| `4x-gb10-2x100g` | 4 | 273 GB/s | 1,092 GB/s | 90 / 210 / 420 | 25.0 GB/s | 12 us | 5.0 GB/s | 128 / 12 GiB |
| `1x-gb10` | 1 | 273 GB/s | 273 GB/s | 90 / 210 / 420 | 0.0 GB/s | 0 us | 5.0 GB/s | 128 / 12 GiB |
| `1x-m5-ultra` | 1 | 1,144 GB/s | 1,144 GB/s | 95 / 190 / 190 | 0.0 GB/s | 0 us | 7.0 GB/s | 256 / 32 GiB |
| `8x-h200-nvlink` | 8 | 4,800 GB/s | 38,400 GB/s | 989 / 1,979 / 1,979 | 450.0 GB/s | 2 us | 7.0 GB/s | 141 / 4 GiB |

---

## 3. Five Pillars and Fifteen Orthogonal Sub-Metrics

RAIS v1.1 distributes 90% of its weight equally across four 22.5% core runtime pillars (`P1–P4`) and 10% to **Pillar 5 — Operational & Systems Physics** (`P5`), decomposed into 15 orthogonal sub-metrics (3 per pillar):

| Pillar (Weight) | ID | Sub-Metric | Weight | Workload & Scoring Rule |
| --- | --- | --- | ---: | --- |
| **P1 Single-Stream Decode (22.5%)** | `1a` | Short-ctx B1 (`C=256`) | **10.8%** | Plain decode (`spec=OFF`), $B=1, C=256$. $S_{1a} = 100 \cdot T_{\text{floor}} / T_{\text{actual}}$. |
| | `1b` | Long-ctx B1 (`C=64K–128K`) | **8.1%** | Plain decode (`spec=OFF`), $B=1$ at 64K–128K (capped to served context). $S_{1b} = 100 \cdot T_{\text{floor}} / T_{\text{actual}}$. |
| | `1c` | Tail Jitter (`p50/p99` ITL) | **3.6%** | Inter-token latency stability on B1 stream: $S_{1c} = 100 \cdot \min(1, \text{ITL}_{p50} / \text{ITL}_{p99})$. |
| **P2 Speculation (22.5%)** | `2a` | Acceptance Quality ($\alpha$) | **10.8%** | Mean committed tokens/round $\alpha = \mathbb{E}[A] + 1$ across `prose`, `code`, `edit`, normalized against $\alpha_{\text{ref}} = 4.0$: $S_{2a} = 100 \cdot \min\!\left(1, \frac{\max(0, \alpha - 1)}{\alpha_{\text{ref}} - 1}\right)$. |
| | `2b-i` | B1 Verify & Draft Efficiency ($1/V$) | **7.2%** | Verify overhead factor $V_{B1} = \frac{\alpha}{\text{TPS}_{\text{spec}} / \text{TPS}_{\text{nospec}}}$ at $K \in [4, 16]$. $S_{2b\text{-}i} = 100 \cdot \min(1, 1/V_{B1})$ when B1 spec is active, else $0$. |
| | `2b-ii` | B8 Batched Verify Efficiency ($1/V$) | **4.5%** | Batched speculative verify at $B=8$: $S_{2b\text{-}ii} = 100 \cdot \min(1, 1/V_{B8})$ when B8 spec is active, else $0$. Combined `2b` = **11.7%**. |
| **P3 Batch & Concurrency Scaling (22.5%)** | `3a` | Interactive Batch (`B=4, 8, 16`) | **9.0%** | Equal average over $B \in \{4, 8, 16\}$ against $T_{\text{floor}}(B, C=256)$ with per-stream SLA $\ge 0.25 \cdot \text{TPS}_{B1}$. Violations scale score by $\text{TPS}_{\text{stream}} / (0.25 \cdot \text{TPS}_{B1})$. |
| | `3b` | High-Concurrency Saturation (`B=32, 64`) | **8.1%** | Equal average over $B \in \{32, 64\}$ against $T_{\text{floor}}(B, C=256)$. Unsupported or capacity-capped $B$ is **scored flat** at the highest working smaller $B$'s aggregate tok/s against $B$'s roofline. |
| | `3c` | Long-Context Batch (`B=8 @ 32K–64K`) | **5.4%** | Concurrent $B=8$ decode at 32K–64K context (capped to served context) against $T_{\text{floor}}(8, C)$. |
| **P4 Prefill & Multi-Turn Agent Loops (22.5%)** | `4a` | Interactive Cold Prefill (`8K`) | **7.2%** | Cold uncached 8,192-token prompt TTFT against $T_{\text{floor,prefill}}(8192)$. |
| | `4b` | Long Cold Prefill (`32K / 128K+`) | **8.1%** | Cold uncached 32K (40%) + 128K (60%, capped to served context) prefill against $T_{\text{floor,prefill}}(L)$. |
| | `4c` | Multi-Turn Prefix & State Reuse TTFT | **7.2%** | Warm follow-up turn `8K + 128` (40%) and `100K + 128` (60%) scored against delta prefill floor $T_{\text{floor,warm}}(C_{\text{prefix}}, 128)$. Cold recomputation collapses score near 0. |
| **P5 Operational & Systems Physics (10.0%)** | `5a` | Cold Boot / Model Swap Efficiency | **5.0%** | Anchored to NVMe sequential read floor $T_{\text{boot}}^{\text{floor}} = \max\!\left(1.0,\; \frac{W_{\text{total}}/N}{BW_{\text{NVMe}} \times 10^9}\right)$. $S_{5a} = \min(100, 100 \cdot T_{\text{boot}}^{\text{floor}} / T_{\text{cold\_boot}})$, reading `cold_boot_s` or `swap_s`. Unmeasured = `0.0`. |
| | `5b` | Usable KV Memory Efficiency | **3.0%** | Anchored to free RAM floor $M_{\text{free\_floor}} = \max(1.0,\; M_{\text{ram/rank}} - M_{\text{weights/rank}} - M_{\text{os\_reserve/rank}})$. $S_{5b} = \min(100, 100 \cdot M_{\text{usable\_kv/rank}} / M_{\text{free\_floor}})$, reading `usable_kv_gib_per_rank` or `kv_cache_bytes_per_rank / 1024^3`. Unmeasured = `0.0`. |
| | `5c` | Batch-Size Bit-Invariance | **2.0%** | Deterministic `B=1` short prompt (`temperature=0`, `spec=OFF`) run as lane 0 of a mixed `B=8` concurrent batch (lanes 1..7 distinct prompts). `100.0` if `batch_invariant_exact == true`, else `0.0`. Separate from `INVALID` gates. |

**Weight sum check:** $(10.8 + 8.1 + 3.6) + (10.8 + 7.2 + 4.5) + (9.0 + 8.1 + 5.4) + (7.2 + 8.1 + 7.2) + (5.0 + 3.0 + 2.0) = 22.5 \times 4 + 10.0 = 100.0\%$.

### 3.1 Composite Metrics, Stack Badges, and ROI Planning

- **Standalone RAIS Model Score:** $\text{RAIS}_m = \sum_{k=1}^{15} w_k S_{m,k} \in [0, 100]$.
- **Base Roofline Score ($R$):** Pure non-speculative kernel efficiency across P1, P3, and P4 (unchanged from v1.0 for transparent cross-version kernel comparisons):
  $$R_m = \frac{P_{1,m} + P_{3,m} + P_{4,m}}{3} \in [0, 100]$$
- **Speculation Summary:** Reports $(\alpha,\; 1/V_{B1},\; 1/V_{B8},\; P_2)$ separately so drafter quality ($\alpha$) and kernel/overlap cost ($V$) remain directly attributable.
- **Stack Badges (Display Metadata Only):** Optional `stack_badge` string (e.g. `Zero-NVIDIA NVRM + SKL`, `CUDA 13.0`, `Apple Metal 4.0`, `vLLM + TileLang`) carried on receipts and rendered in CLI and scorecard UI with **zero impact** on any numeric score.
- **Fleet Aggregate RAIS:** Weighted across the 19 fleet models with Tier 1/2/3 weights $W_m$ ($\sum_{m=1}^{19} W_m = 1.0$):
  $$\text{RAIS}_{\text{fleet,covered}} = \frac{\sum_{m \in \mathcal{M}} W_m \cdot \text{RAIS}_m}{\sum_{m \in \mathcal{M}} W_m}, \qquad \text{RAIS}_{\text{fleet,strict}} = \sum_{m \in \mathcal{M}} W_m \cdot \text{RAIS}_m$$
- **ROI Priority Table:** Ranks every `(model, sub-metric)` by remaining headroom:
  $$\Delta\text{RAIS}_{m,k}^{\max} = w_k (100 - S_{m,k}), \qquad \Delta\text{RAIS}_{\text{fleet},m,k}^{\max} = W_m \cdot w_k (100 - S_{m,k})$$
  paired with its binding hardware bottleneck (`memory`, `network`, `compute`, `speculation`, `jitter`, `capacity_cap`, `storage`, `memory_footprint`, or `determinism`).

---

## 4. Hard Binary Correctness Gates

Performance without numerical and serving fidelity is invalid. Every RAIS run records four hard binary gates:

| Gate | Condition | Failure Behavior |
| --- | --- | --- |
| `prose_exact` | Greedy (`temperature=0`) output with speculation ON matches speculation OFF bit-for-bit on the prose prompt. | Score marked **`INVALID`** |
| `code_exact` | Greedy output with speculation ON matches speculation OFF bit-for-bit on the code and edit prompts. | Score marked **`INVALID`** |
| `prefix_exact` (Gate GX) | Greedy output on a warm cached multi-turn prefix (`8K + 128`) matches a cold uncached run of the exact same prompt bit-for-bit. | Score marked **`INVALID`** |
| `errors == 0` | Zero HTTP 5xx failures, transport drops, OOMs, or GPU Xids across all supported workloads (capacity rejections on `B > max_sequences` are classified as `unsupported`, triggering flat-cap scoring rather than invalidation). | Score marked **`INVALID`** |

- All four gates pass $\implies$ `status = "VALID"`, `certified_rais = raw_rais`.
- Any gate fails $\implies$ `status = "INVALID"`, `certified_rais = null` (`raw_rais` retained only for debugging).
- Legacy imports lacking gate records $\implies$ `status = "UNVERIFIED"` (never certified).

---

## 5. Compatible Execution Tiers

Both tiers emit the exact same RAIS v1.1 JSONL schema (backwards compatible with v1.0 receipts; unmeasured P5 sub-metrics score `0.0`) and are scored identically by `tools/rais_score.py`:

| Parameter | Fast Hill-Climb Tier (`--tier fast`) | Full Certification Tier (`--tier full`) |
| --- | --- | --- |
| Runtime per model | **~3–4 minutes** | **~15–25 minutes** |
| Intended use | Per-PR hill-climbing, rapid A/B verification | Release qualification, cross-engine publication |
| `1a` / `1c` B1 short | `O=128`, 2 repeats | `O=256`, 3 repeats |
| `1b` B1 long | `C=32K` (capped), `O=64` | `C=64K–128K` (capped), `O=128` |
| `2a` / `2b-i` B1 spec | `prose` + `code` + `edit`, `O=128` | `prose` + `code` + `edit`, `O=256` |
| `2b-ii` B8 spec | Measured at `B=8`, `O=96` (toggleable) | Measured at `B=8`, `O=256` |
| `3a` / `3b` Batch sweep | `B=4, 8, 16, 32, 64`, `O=96` | `B=4, 8, 16, 32, 64`, `O=256` |
| `3c` Long-ctx batch | `B=8 @ 32K`, `O=64` | `B=8 @ 64K` (capped), `O=128` |
| `4a` / `4b` Cold prefill | `8K` + `32K` | `8K` + `32K` + `128K` (capped) |
| `4c` Warm multi-turn + GX | `8K + 128` + Gate GX (`O=32`) | `8K + 128` + `100K + 128` + Gate GX (`O=64`) |
| `5a` / `5b` / `5c` Systems Physics | `--cold-boot-s` (or `--swap-url`), `--usable-kv-gib`, live `B=1 vs B=8` lane-0 invariance | Same (`O=256` lane-0 invariance) |

---

## 6. Fleet Model Catalog (19 Models / 21 Lanes)

All 21 served lanes from [`deployment/lanes.json`](../deployment/lanes.json) map to 19 distinct fleet models grouped into three fleet weight tiers: **Tier 1 = 55%** (5 flagship models), **Tier 2 = 25%** (7 secondary frontier and regional models), and **Tier 3 = 20%** (7 fast MoE, dense, and specialist models). Two variant lanes (`qwen3.8-27b-fp8` and `ling-3.0-flash-fin`) score standalone with their exact geometry and fold into their canonical model without double-counting in the 19-model fleet aggregate.

| # | Lane (`deployment/lanes.json`) | Tier | Fleet Wt | Active Params (Non-Exp / Exp) | MoE ($E$ / $K$) | Dtype | Hidden | Layers | KV B/tok | Served Ctx |
| ---: | --- | :---: | ---: | ---: | :---: | --- | ---: | ---: | ---: | ---: |
| 1 | `kimi-k3.1-pro` | T1 | 13.0% | 104.2B (55.6B / 48.6B) | 896 / 16 | BF16+MXFP4 | 7168 | 93 | 27,648 | 1,048,576 |
| 2 | `glm-5.3` | T1 | 12.0% | 40.2B (17.5B / 22.7B) | 256 / 8 | FP8 | 6144 | 78 | 89,856 (+5,376 idx) | 1,048,576 |
| 3 | `glm-5.3-flash` | T1 | 11.0% | 18.0B (7.0B / 11.0B) | 288 / 8 | FP8 | 4096 | 45 | 11,264 (+2,816 idx) | 131,072 |
| 4 | `deepseek-v4.1-flash` | T1 | 10.0% | 14.7B (6.2B / 8.5B) | 384 / 6 | FP8+MXFP4 | 5120 | 40 | 3,200 (+1,024 idx) | 1,048,576 |
| 5 | `kimi-k3` | T1 | 9.0% | 104.2B (55.6B / 48.6B) | 896 / 16 | BF16+MXFP4 | 7168 | 93 | 27,648 | 1,048,576 |
| 6 | `qwen3.8-27b` | T2 | 5.0% | 27.0B (27.0B / 0B) | Dense | BF16 | 5120 | 64 | 65,536 | 262,144 |
| — | `qwen3.8-27b-fp8` | T2 | variant | 27.0B (27.0B / 0B) | Dense | FP8 | 5120 | 64 | 65,536 | 262,144 |
| 7 | `qwen3.8-max` | T2 | 4.0% | 95.0B (48.7B / 46.3B) | 512 / 10 | BF16+NVFP4 | 8192 | 92 | 94,208 | 32,768 |
| 8 | `mimo-v2.6-pro` | T2 | 3.5% | 40.0B (19.2B / 20.8B) | 384 / 8 | FP8+MXFP4 | 6144 | 70 | 51,200 | 32,768 |
| 9 | `hy4-preview` | T2 | 3.5% | 49.0B (24.0B / 25.0B) | 256 / 8 | MXFP8 | 6144 | 78 | 89,856 (+5,376 idx) | 32,768 |
| 10 | `k-exaone-2.0` | T2 | 3.0% | 37.5B (14.5B / 23.0B) | 256 / 8 | BF16 | 6144 | 78 | 81,920 | 262,144 |
| 11 | `a.x-k2` | T2 | 3.0% | 37.1B (16.0B / 21.1B) | 256 / 8 | FP8 | 7168 | 61 | 70,272 (+15,616 idx) | 262,144 |
| 12 | `solar-open-2` | T2 | 3.0% | 12.5B (6.5B / 6.0B) | 320 / 8 | BF16 | 4096 | 48 | 49,152 | 262,144 |
| 13 | `mimo-v2.6-flash` | T3 | 3.5% | 15.0B (5.5B / 9.5B) | 256 / 8 | FP8+MXFP4 | 4096 | 48 | 23,040 | 32,768 |
| 14 | `gemma-4-31b` | T3 | 3.0% | 30.7B (30.7B / 0B) | Dense | BF16 | 5376 | 60 | 81,920 | 32,768 |
| 15 | `gemma-4-26b-a4b` | T3 | 3.0% | 3.8B (1.9B / 1.9B) | 128 / 8 | BF16 | 2816 | 30 | 20,480 | 262,144 |
| 16 | `ling-3.0-flash` | T3 | 3.0% | 5.1B (2.6B / 2.5B) | 512 / 8 | BF16 | 2560 | 42 | 8,064 | 262,144 |
| — | `ling-3.0-flash-fin` | T3 | variant | 5.1B (2.6B / 2.5B) | 512 / 8 | BF16 | 2560 | 42 | 8,064 | 262,144 |
| 17 | `muse-glimmer-30b` | T3 | 2.5% | 29.2B (29.2B / 0B) | Dense | BF16 | 6656 | 52 | 13,312 | 131,072 |
| 18 | `laguna-s-2.1` | T3 | 2.5% | 7.5B (3.1B / 4.4B) | 256 / 10 | BF16 | 3072 | 48 | 49,152 | 32,768 |
| 19 | `qwen3.8-flash` | T3 | 2.5% | 5.4B (3.0B / 2.4B) | 512 / 10 | BF16+FP8 | 2560 | 48 | 24,576 (+1,536 idx) | 262,144 |

**Tier sum check:** Tier 1 = 55.0%, Tier 2 = 25.0%, Tier 3 = 20.0%, Total = **100.0%**.

---

## 7. CLI Reference and Standardized JSONL Schema

### 7.1 Running the Benchmark (`tools/rais_bench.py`)

```sh
python3 tools/rais_bench.py \
  --base-url http://127.0.0.1:8000 \
  --model glm-5.3-flash \
  --tier fast \
  --hardware 16x-gb10-1x100g \
  --engine sparkpipe \
  --stack-badge "CUDA 13.0" \
  --cold-boot-s 8.4 \
  --usable-kv-gib 72.0 \
  --spec-toggle speculation \
  --output rais_results.jsonl \
  --score
```

For multi-model sweeps behind a swap controller:

```sh
python3 tools/rais_bench.py \
  --base-url http://127.0.0.1:8000 \
  --swap-url http://127.0.0.1:8001/swap \
  --swap-token "$SWAP_TOKEN" \
  --model kimi-k3.1-pro \
  --tier full \
  --stack-badge "Zero-NVIDIA NVRM + SKL" \
  --output fleet_cert.jsonl
```

### 7.2 Scoring, Comparing, and Importing (`tools/rais_score.py`)

```sh
python3 tools/rais_score.py score rais_results.jsonl --hardware 16x-gb10-1x100g --format markdown
python3 tools/rais_score.py score rais_results.jsonl --hardware 16x-gb10-1x100g --format json
python3 tools/rais_score.py score rais_results.jsonl --hardware 16x-gb10-1x100g --stack-badge "Zero-NVIDIA NVRM + SKL" --format html-data
python3 tools/rais_score.py compare baseline.jsonl after.jsonl --hardware 16x-gb10-1x100g --label-a SparkPipe --label-b vLLM --stack-badge-a "Zero-NVIDIA NVRM + SKL" --stack-badge-b "vLLM + TileLang"
python3 tools/rais_score.py reconstitute overlap.jsonl --hardware 16x-gb10-1x100g --new-weights basket_v2.json --output basket_v2_divisor.json
python3 tools/rais_score.py import-hillclimb qualification/batch-sweep-20261018/glm-5.3-flash.jsonl --score
```

## 8. Basket Reconstitution & S&P / Dow-Jones Continuity Divisor

Every model in RAIS keeps its standalone `0–100` score on a given hardware profile (`1x100g` single-switch or `2x100g` dual-switch, `4x` or `16x`), so external head-to-head comparisons (`SparkPipe` vs `vLLM`, `SGLang`, or tuned forks) remain direct and timeless.

When the fleet basket evolves—retiring superseded models (for example rotating out `kimi-k3` once `kimi-k3.1-pro` replaces it) and admitting newer frontier releases—the composite **RAIS Fleet Index** stays continuous across generations using an S&P 500 / Dow-Jones continuity divisor $D_k$:

$$\text{RAIS\_Index}(t) = \frac{\sum_{i \in B_k} w_i \cdot S_i(t)}{D_k}, \qquad D_k = D_{k-1} \cdot \frac{\sum_{i \in B_k} w_i^{\text{new}} \cdot S_i(t_k)}{\sum_{j \in B_{k-1}} w_j^{\text{old}} \cdot S_j(t_k)}$$

At reconstitution snapshot $t_k$, an overlap receipt runs all models in $B_{k-1} \cup B_k$ on the same build and hardware profile. `tools/rais_score.py reconstitute` solves for $D_k$ so `index_before == index_after` (zero discontinuity), while future improvements on $B_k$ move `fleet_rais_index` proportionally. Both single-switch (`16x-gb10-1x100g`, `4x-gb10-1x100g`) and dual-switch (`16x-gb10-2x100g`, `4x-gb10-2x100g`) hardware profiles are scored under identical formulas.

### 7.3 Standardized RAIS JSONL Record

```json
{
  "rais_version": "1.1",
  "utc": "2026-11-25T12:00:00Z",
  "model": "glm-5.3-flash",
  "engine": "sparkpipe",
  "stack_badge": "Zero-NVIDIA NVRM + SKL",
  "tier": "fast",
  "hardware": "16x-gb10-1x100g",
  "metrics": {
    "b1_short_tok_s": 55.0,
    "b1_short_ctx": 256,
    "b1_long_tok_s": 46.0,
    "b1_long_ctx": 65536,
    "itl_p50_ms": 18.1,
    "itl_p99_ms": 20.5,
    "speculation": {
      "alpha_prose": 2.4,
      "alpha_code": 3.5,
      "alpha_edit": 4.2,
      "verify_v_b1": 1.18,
      "verify_b1_enabled": true,
      "verify_v_b8": 1.35,
      "verify_b8_enabled": true
    },
    "batch": {
      "4": {"aggregate_tok_s": 132.0},
      "8": {"aggregate_tok_s": 175.0},
      "16": {"aggregate_tok_s": 227.0},
      "32": {"unsupported": true, "aggregate_tok_s": 0.0},
      "64": {"unsupported": true, "aggregate_tok_s": 0.0}
    },
    "b8_long_tok_s": 129.8,
    "b8_long_ctx": 32768,
    "prefill_8k_tokens": 8192,
    "prefill_8k_tok_s": 1162.0,
    "prefill_32k_tokens": 32768,
    "prefill_32k_tok_s": 1353.0,
    "warm_8k_prefix_tokens": 8192,
    "warm_8k_new_tokens": 128,
    "warm_8k_ttft_ms": 210.0,
    "cold_boot_s": 8.4,
    "usable_kv_gib_per_rank": 72.0,
    "batch_invariant_exact": true
  },
  "gates": {
    "prose_exact": true,
    "code_exact": true,
    "prefix_exact": true,
    "errors": 0
  }
}
```
