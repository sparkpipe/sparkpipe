# SparkPipe Compute Marketplace — Business Plan Draft

Status: DRAFT, 2026-08-28 (ratified into the repo 2026-08-29); fee model, revenue basis and network design revised 2026-09-27
Scope: business model, competitive pricing, anti-cheating architecture, unit economics.
The system it describes is the provider network in `README.md`; its gaps are in `TECHDEBT.md`.

---

## 1. Model

A two-sided marketplace for LLM inference:

- **Supply side**: owners of compute (initially NVIDIA DGX Spark / GB10-class nodes) register their installations and serve open-weight models. Providers **choose the models, set their own per-token prices** and limits, and **switch reselling on or off** at will. Registration joins the installation to the SparkPipe **tailnet**, one WireGuard mesh of providers, router and reference nodes, so no provider opens a port to the internet.
- **Owner first**: the hardware stays the owner's. Network traffic runs only on capacity the owner's own traffic leaves idle and yields at the next frame boundary when the owner needs it, so a provider rents out exactly the time they are not using.
- **Demand side**: customers call sparkpipe.ai's OpenAI-compatible API, a **LiteLLM proxy** that is also the router. Every offer is one LiteLLM deployment at the provider's tailnet address, priced at the provider's price.
- **Interchangeable providers**: every provider is validated (§4), so all offers of the same model and quantization are fungible. The router spreads one buyer's requests over as many providers as it needs, by price, latency, load and verification record, and keeps requests that share a prompt prefix on one provider for its KV cache. A buyer can therefore use more capacity than any single installation has.
- **Platform fee**: providers set their prices; the platform keeps **15% of the provider's price** on every token sold, in cash, and providers keep **85%**. Buyers pay the serving provider's price with nothing added. Verification (about 2% of served tokens re-executed, §4) is paid out of the 15%, so the platform clears about 12.5% (§5). This replaces the earlier draft's 10% taken in compute, which needed compute-credit accounting.
- **Supply-side requirement**: providers must run **SparkPipe** firmware, and **every provider is validated**. This is both the quality floor (deterministic, qualified, exact-token serving) and the foundation of the anti-cheating system (§4).

Why providers join: monetize idle hardware at prices they set, with zero billing/gateway/demand-generation work, and take it back whenever they need it. Why customers come: verified-exact open-model inference behind one API, with the capacity of many installations at once.

## 2. What the industry charges

### Marketplace / router take rates

| Platform | Model | Take rate |
| --- | --- | --- |
| **OpenRouter** | Router over inference providers; no token markup, fee on credits | 5.5% on credit purchases ($0.80 min); 5% on BYOK usage above $25k/mo |
| **Vast.ai** | GPU rental marketplace (supply = individuals' machines) | ~10–15% provider-side fee |
| **Salad** | Distributed consumer-GPU cloud | ~20–25% provider-side fee |
| **Akash / io.net** | DePIN container/GPU marketplaces | Pricing 50–70% below AWS; fees embedded in crypto settlement |

Key distinction: **OpenRouter is not a supply-side marketplace.** It resells existing providers (Together, Fireworks, DeepInfra…) who carry their own full serving margins underneath. The comparable set for *registering your own compute* is Vast/Salad/Akash/io.net — where 10–25% is the norm. A **15% take sits inside the 10–25% norm** for compute aggregation, and far below the all-in margin customers pay through the router+provider stack.

### Per-token price context (open models, 2026)

- DeepSeek-V4-Flash: ~$0.09/M input on DeepInfra
- DeepSeek-V4-Pro: $1.30/$2.60 per M in/out on DeepInfra vs $1.74/$3.48 Fireworks, $2.10/$4.40 Together
- DeepInfra per-token range across catalog: $0.02–$2.85/M

Inference on open weights is commoditizing fast; margins accrue to whoever aggregates demand and whoever serves cheapest per token. SparkPipe's measured advantage (e.g. dsv4-flash 40 tok/s B1 TP4 no-spec, beating the retained vLLM reference on identical hardware) is directly a cost advantage that providers can price into their own rates while staying cheapest.

## 3. Why "must run SparkPipe" — and its limit

Requiring SparkPipe gives:

- A **deterministic serving stack**: same driver hash + same weights + same request + same seed ⇒ bit-identical token stream (greedy-only today; see the determinism appendix). This is the repo's qualification discipline, and it is the property that makes cheap, objective auditing possible (§4). Commodity stacks (vLLM et al.) cannot offer this.
- Standardized receipts: content-addressed link units, pinned checkpoint hashes, exact-token gates.
- A raising of the casual-cheating bar: a provider must modify *our* code to lie.

The limit, stated plainly: **nothing a provider's machine reports about itself can be trusted.** Root access beats software attestation; GB10-class hardware has no usable TEE/confidential-compute mode. Any check whose verdict is computed on the provider's host — including "my drafter accepted X%" — is forgeable. And output-based drafter checks fail on their own: verified speculative decoding is by construction output-equivalent to the target model, so the drafter leaves no trace in tokens; acceptance rate is a single scalar a blended drafter can be tuned to match.

**Design rule: only trust artifacts that the real weights must produce, verified off-node.**

## 4. Anti-cheating architecture ("proof of honest serving")

Layered, in order of strength:

1. **Sampled re-execution (fraud proofs).** The platform operates trusted reference nodes (seeded from the audit take). A secret random sample (~0.5–2%) of *real, completed customer requests* is re-executed on trusted nodes with the identical pinned driver hash. Because SparkPipe serving is deterministic per (driver, weights, seed), comparison is exact token-stream equality, not statistics. Quantized or substituted weights diverge in greedy decode within tens of tokens. Real traffic is undetectable as an audit — there is no probe to dodge. Exact comparison needs batch-invariant numerics: today batched decode is not bitwise equal to B1 (`TECHDEBT.md`, dynamic batching), so until the batch kernels agree bitwise, replays of batched requests compare logprobs within a tolerance instead.
2. **Logprob audits.** Providers must return logprobs; the platform recomputes them on the reference model for sampled tokens. Logprobs are a high-dimensional side channel a cheaper model cannot produce — catches quantization specifically, where token streams may coincidentally agree. **Logprobs are temperature-independent evidence** — the primary audit signal for temp>0 traffic.
3. **Weight fingerprints.** Checkpoints distributed for serving carry a secret set of trigger→response pairs. Only the exact weights reproduce them; designed to break under quantization. Checked during audits at near-zero cost.
4. **Performance contracts (coarse filter).** SparkPipe has qualified kernel-level perf envelopes; a node consistently *faster* than qualified suggests quantization, consistently slower is a service-quality issue. Weak, nearly free.

**Enforcement economics**: providers post a bond; payouts settle after a challenge window; a mismatch triggers one disputable third-party re-execution, then slashing. Deterrence condition: `P(detection) × penalty > margin gained by cheating`. Cheating saves the cost delta between the real model and a quantized/smaller one — a modest multiple of serving cost — so even 1% sampling with meaningful slashing makes honesty the cheapest strategy. Optimistic-rollup logic, applied to tokens.

## 5. Unit economics of the fee

Goal: **keep 15% of token revenue in cash, pay verification out of it.**

- **Audit cost at sampling rate *s*.** Re-execution costs about as much compute as the original request, plus logprob and fingerprint overhead: call it 1.2× per sampled request. At s = 2% of tokens, verification costs about 2.4% of the compute sold.
- **The platform clears about 12.5%** of token revenue after verification, if reference compute costs about what providers charge. First-party reference nodes cost less than that.
- **Sampling can float with risk.** New providers start at high coverage, up to 5%, and graduate down as their record grows; bonded, long-standing providers get less. This also caps the worst-case audit spend.

Illustrative: a provider selling $100k/yr of tokens keeps $85k. The platform keeps $15k, of which about $2.4k funds that provider's audit coverage.

## 5a. Revenue per provider (revised 2026-09-27)

Revenue is throughput × time sold × the provider's price. Resold traffic is batched by nature, so the throughput that counts is the installation's aggregate at large batch, not one stream's speed.

| Operating point | Aggregate tok/s | Basis | Price out ($/M) | Buyers pay per month, 50% sold | Provider keeps (85%) | Fee (15%) | 16-Spark clusters for $1M/yr of fees |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| GLM 5.3 Flash, 8 streams, 16 Sparks | 129.9 | measured, iteration 17 (2026-09-26) | 0.50 | $84 | $72 | $13 | 6,600 |
| GLM 5.3 Flash, 8 streams, 16 Sparks | ~300 | M1 target | 0.50 | $194 | $165 | $29 | 2,860 |
| GLM 5.3 Flash, 64 streams, 16 Sparks | ~750 | M2 target | 0.50 | $486 | $413 | $73 | 1,140 |
| GLM 5.3 Flash, 256 streams, 16 Sparks | ~2,200 | M2 target | 0.50 | $1,426 | $1,212 | $214 | 390 |
| GLM 5.3 Flash, 256 streams, 16 Sparks | ~1,650 | projected ([`PERFORMANCE_STATUS.md`](../PERFORMANCE_STATUS.md#glm-53-flash-tp16-decode)); the site's figure | 0.50 | $1,069 | $909 | $160 | 520 |
| GLM 5.2, 16 streams, 8 Sparks (TP8) | 75.55 | measured (2026-08-16) | 1.75–4.40 | $171–$431 | $146–$366 | $26–$65 | 645–1,620 |
| GLM 5.3, 256 streams, 16 Sparks | ~1,900 | 80% of a weight-read bound, derived | 4.40 | ~$11,000 | ~$9,300 | ~$1,650 | ~51 |

- A month is 30 days. Only output tokens are counted; input tokens are billed too, at the provider's input price, and add revenue in proportion to prompt length.
- 50% sold is an assumption. The owner's own use and the network's demand both limit it.
- Prices are the market references of 2026-09-26: GLM 5.3 Flash about $0.15 in and $0.50 out ([pricepertoken](https://pricepertoken.com/pricing-page/model/z-ai-glm-5.3-flash)), GLM 5.2 $1.75–$4.40 out ([pricepertoken](https://pricepertoken.com/pricing-page/model/z-ai-glm-5.2)), GLM 5.3 $4.40 out ([glm5.app](https://glm5.app/blog/glm-5-3-pricing)). Providers set their own.
- The M1 and M2 targets are 80% of the batch roofline at 1K context ([`GLM5_NEXT_ROOFLINE.md`](GLM5_NEXT_ROOFLINE.md)): about 2,800 tok/s is the memory-bound ceiling at 256 streams. Each stream then sees about 8.6 tok/s at the target and 6.5 at the projection, so that capacity sells to batch and agent traffic more than to chat.
- The GLM 5.2 cluster figure counts two TP8 instances per sixteen Sparks.
- The GLM 5.3 row is not a roadmap target. Its 465 GB NVFP4 checkpoint spread over sixteen Sparks is about 29 GB per rank; at 256 streams every expert is read once per step, which takes 106 ms at 273 GB/s and bounds throughput near 2,400 tok/s. It leaves out collectives, which at that batch cost about as much again unless they overlap compute (M2), and GLM 5.3 is still onboarding on the GLM 5.2 driver.

What this says:

- **Today, reselling Flash barely covers electricity.** A sixteen-Spark cluster at an assumed 200 W per Spark and $0.15/kWh costs about $350 a month to run.
- **At M2's batch throughput it pays.** A Flash cluster sold half the time earns its owner about $900 a month at the projection and $1,200 at the target, around the original pitch's $1k per installation, and 390 to 520 such clusters make $1M a year in fees.
- **Large models are worth more per Spark, today and at scale.** GLM 5.2's measured rate already earns 4–10× Flash's per Spark. At large batch a step reads every expert once, so throughput follows the checkpoint's bytes per rank, not its active parameters: GLM 5.3 in NVFP4 is about 29 GB per rank against Flash's 25 GB, at up to 9× the price. Few installations can hold such models (GLM 5.2 is 756 GB, eight Sparks in practice), so their providers compete with the hosted APIs' prices more than with each other. The network should lead with large models.
- **Supply growth.** Mac Studio support (M8) widens the pool of installations. Each hardware type needs its own reference nodes, because different kernels give different bits.

## 5b. Decisions (2026-09-27)

- **Price and fee.** Providers set their own price per model. sparkpipe.ai keeps 15% of that price on every token sold, and buyers pay the provider's price with nothing added. A split of the fee between buyers and providers was considered and dropped.
- **Validation for all.** Every provider is validated the same way (§4). There is no unverified tier and no paid verified tier.
- **No license fee.** A business can run vLLM for free, so a per-node fee for private use would send it there. SparkPipe stays free to run, and the network fee is the revenue.
- **Interchangeable providers on one fabric.** Validation makes offers of the same model and quantization fungible, so one buyer's traffic spreads over many providers. A tailnet connects providers, the router and the reference nodes; the router is the LiteLLM proxy.

## 6. Risks and open questions

- **Two-sided cold start.** No demand without supply, no supply without demand. Mitigation: seed with first-party fleet (the existing 16-node cluster), open supply registration once demand routing through liteLLM is live.
- **Determinism boundary.** Exact-token replay audits only work while all providers run pinned driver builds for a model version. Driver updates need coordinated flag days per model, or per-driver-version audit cohorts. Seed logging is mandatory (see appendix).
- **Replay cost at scale.** Long-context replays cost more than short ones; sampling should be weighted by request cost, not count, with caps (e.g. replay first N tokens + sampled continuation windows instead of full regenerations).
- **Legal/settlement.** Payout and challenge windows belong in the provider agreement. Still to decide: provider identity checks, tax reporting, and fiat vs crypto settlement.
- **Provider margin reality check.** If a provider can earn more per GPU-hour on Vast/Salad than by serving tokens here net of 15%, supply won't come. At today's B8 throughput and Flash prices, reselling barely covers electricity (§5a). The network should open once M2's batch throughput lands, and lead with large models.
- **Prompt privacy.** A provider sees the prompts it serves, as with any hosted API, and GB10 has no confidential-computing mode. Validation proves which model served a request, not who read it.
- **Tailnet scale.** Hundreds of providers on one tailnet need a control server that fits the node count and cost: Tailscale, or a self-hosted Headscale. Access rules must let only the router and the reference nodes reach a provider's model API, and let providers reach nothing of each other.
- **Sophisticated adversaries.** Someone can serve honestly except when they suspect an audit — which is why audits are replays of real traffic, never synthetic probes. Accept that this is economics, not cryptography: make cheating EV-negative and detection evidence objective.

## 7. Near-term build hooks

1. liteLLM front end in front of the existing gateway (never expose `node/model_api.c` directly). **[STATUS: merged, docs/LITELLM_FRONTEND.md; the config routes to the rtx5090 GLM API, but no completion through the door has a receipt yet]**
2. Request-logging pipeline keyed by (driver hash, model contract hash, request) — the audit substrate.
3. Audit service: sampler → replay scheduler → comparator → challenge/slash state machine.
4. Provider onboarding: `sparkpipe provider register`, joining the tailnet with a tagged key under access rules, bond, pinned driver distribution, fingerprinted checkpoints.
5. Receipt ledger: signed per-completion receipts, per-provider sales, the 85/15 split, payouts after the challenge window.
6. Owner-first scheduling: a preemptible network class in the batch engine that yields at frame boundaries, and hand-off of preempted network requests to the router.
7. The router on LiteLLM: one deployment per offer at the provider's tailnet address with the provider's price as its per-token cost, routing by price, latency, load and verification record, prefix affinity, and virtual keys with spend logs (Postgres) for buyer billing. A logging callback feeds the audit sampler.
8. sparkpipe.ai: the landing page, the provider page and the playground are in `site/`; the catalog, buyer console and provider dashboard come with the router.

---

## Appendix A: Determinism status (2026-09-27)

- **Greedy decoding** is bitwise reproducible run to run, and every exact-token receipt is a temp=0 artifact. That is what replay audits need for greedy traffic.
- **Sampling** exists for glm5_next, temperature only: Gumbel-max on the vocab-parallel argmax. The noise comes from a counter-based RNG keyed by (request seed, position, global token id), so it does not depend on the rank or batch row, and a seeded request replays like a greedy one given identical logits.
- **Logits-side determinism** is the hard half, and the bar rises with sampling: temp>0 amplifies near-ties that argmax absorbs.
- **Batch invariance is not there yet.** Batched decode and prefill are not bitwise equal to B1 (`TECHDEBT.md`, dynamic batching). Until they are, exact replay covers only requests whose batch composition is reproduced, and logprob tolerance covers the rest.

**Provider-contract consequences**: temp=0 traffic is where token comparison is sharpest. Providers must honor and log per-request seeds, and may not force temp>0 or refuse seed logging, since that is the obvious loophole around replay audits. Logprob audits do not depend on temperature and are the primary signal for temp>0 traffic; the API returns them for GLM Full, GLM Flash and K3 only (`TECHDEBT.md`, serving API).

*Pricing data in §2 as of 2026-08 and in §5a as of 2026-09-26; re-verify before external use.*
