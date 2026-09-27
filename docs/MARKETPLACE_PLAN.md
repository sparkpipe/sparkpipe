# SparkPipe Compute Marketplace — Business Plan Draft

Status: DRAFT, 2026-08-28 (ratified into the repo 2026-08-29); fee model revised 2026-09-27
Scope: business model, competitive pricing, anti-cheating architecture, unit economics.
The system it describes is the provider network in `README.md`; its gaps are in `TECHDEBT.md`.

---

## 1. Model

A two-sided marketplace for LLM inference:

- **Supply side**: owners of compute (initially NVIDIA DGX Spark / GB10-class nodes) register their installations and serve open-weight models. Providers **choose the models, set their own per-token prices** and limits, and **switch reselling on or off** at will.
- **Owner first**: the hardware stays the owner's. Network traffic runs only on capacity the owner's own traffic leaves idle and yields at the next frame boundary when the owner needs it, so a provider rents out exactly the time they are not using.
- **Demand side**: customers reach the fleet through sparkpipe.ai's OpenAI-compatible API, behind the **liteLLM front end**, routing across registered providers.
- **Platform fee**: the platform keeps **15% of token revenue, in cash**; providers keep **85%**. Verification (about 2% of served tokens re-executed, §4) is paid out of the 15%, so the platform clears about 12.5% (§5). This replaces the earlier draft's 10% taken in compute, which needed compute-credit accounting.
- **Supply-side requirement**: providers must run **SparkPipe** firmware. This is both the quality floor (deterministic, qualified, exact-token serving) and the foundation of the anti-cheating system (§4).

Why providers join: monetize idle hardware at prices they set, with zero billing/gateway/demand-generation work, and take it back whenever they need it. Why customers come: verified-exact open-model inference, behind one API.

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

## 5a. Revenue per provider (2026-09)

The fee is only as large as what providers sell, and that is set by throughput and token prices.

- **Throughput.** The fleet's GLM 5.3 Flash service measured 129.9 tok/s aggregate at 8 streams on sixteen Sparks (iteration 17, 2026-09-26).
- **Price.** GLM 5.3 Flash sells for about $0.15 in and $0.50 out per million tokens at most providers ([pricepertoken](https://pricepertoken.com/pricing-page/model/z-ai-glm-5.3-flash), 2026-09-26).
- **Revenue.** A fully sold sixteen-Spark cluster makes about 337M output tokens a month, or about $170/month from output tokens, plus input tokens in proportion to prompt length. The 15% fee on that is on the order of $50/month.
- **Cost.** At an assumed 200 W per Spark and $0.15/kWh, the same cluster's electricity is about $350/month. At today's speed, reselling a Flash model roughly breaks even for the provider.

Three things move this:

- **Batched throughput.** M1's ~300 tok/s at B8 more than doubles revenue per cluster. M2's batch targets (~750 and ~2200 tok/s at B64 and B256) multiply it by 6 to 17.
- **Model choice.** Large models sell for much more per token: GLM 5.2 at $1.75–$4.40 out ([pricepertoken](https://pricepertoken.com/pricing-page/model/z-ai-glm-5.2)), GLM 5.3 at $4.40 out ([glm5.app](https://glm5.app/blog/glm-5-3-pricing)). Unified memory makes Sparks a natural home for them, so the network should lead with large models.
- **Supply growth.** Mac Studio support (M8) widens the pool of installations. Each hardware type needs its own reference nodes, because different kernels give different bits.

The target in the original pitch was about $1k/month of sales per installation, which is $150/month of fee, so 500 installations make about $0.9M a year. At Flash prices that needs about 3–4× today's throughput, depending on prompt length. With large models it needs less.

## 5b. Other revenue options under consideration

- **A license for private commercial use.** Free for personal use and for installations that sell through the network; a per-node monthly license for businesses running SparkPipe privately. This charges for the engine itself, whatever tokens sell for. At Flash prices, $20/node/month on sixteen nodes ($320) is about 6× the network fee from the same cluster fully sold.
- **Split the fee across both sides.** For example, 5% on buyer credits (OpenRouter-style) plus 10% from providers: the same total take, but a smaller number on each side.
- **A verified tier at a premium.** Replay-audited, pinned-weight inference sold at a premium to buyers who need the guarantee. Verification is what other decentralized networks cannot easily match.

## 6. Risks and open questions

- **Two-sided cold start.** No demand without supply, no supply without demand. Mitigation: seed with first-party fleet (the existing 16-node cluster), open supply registration once demand routing through liteLLM is live.
- **Determinism boundary.** Exact-token replay audits only work while all providers run pinned driver builds for a model version. Driver updates need coordinated flag days per model, or per-driver-version audit cohorts. Seed logging is mandatory (see appendix).
- **Replay cost at scale.** Long-context replays cost more than short ones; sampling should be weighted by request cost, not count, with caps (e.g. replay first N tokens + sampled continuation windows instead of full regenerations).
- **Legal/settlement.** Payout and challenge windows belong in the provider agreement. Still to decide: provider identity checks, tax reporting, and fiat vs crypto settlement.
- **Provider margin reality check.** If a provider can earn more per GPU-hour on Vast/Salad than by serving tokens here net of 15%, supply won't come. At today's throughput and Flash prices, reselling barely covers electricity (§5a). The answer is throughput, large models and demand density: a sold-out token market beats an idle rental listing.
- **Sophisticated adversaries.** Someone can serve honestly except when they suspect an audit — which is why audits are replays of real traffic, never synthetic probes. Accept that this is economics, not cryptography: make cheating EV-negative and detection evidence objective.

## 7. Near-term build hooks

1. liteLLM front end in front of the existing gateway (never expose `node/model_api.c` directly). **[STATUS: DONE — merged, docs/LITELLM_FRONTEND.md]**
2. Request-logging pipeline keyed by (driver hash, model contract hash, request) — the audit substrate.
3. Audit service: sampler → replay scheduler → comparator → challenge/slash state machine.
4. Provider onboarding: `sparkpipe provider register`, the provider agent's outbound connection, bond, pinned driver distribution, fingerprinted checkpoints.
5. Receipt ledger: signed per-completion receipts, per-provider sales, the 85/15 split, payouts after the challenge window.
6. Owner-first scheduling: a preemptible network class in the batch engine that yields at frame boundaries, and hand-off of preempted network requests to the router.

---

## Appendix A: Determinism status (2026-09-27)

- **Greedy decoding** is bitwise reproducible run to run, and every exact-token receipt is a temp=0 artifact. That is what replay audits need for greedy traffic.
- **Sampling** exists for glm5_next, temperature only: Gumbel-max on the vocab-parallel argmax. The noise comes from a counter-based RNG keyed by (request seed, position, global token id), so it does not depend on the rank or batch row, and a seeded request replays like a greedy one given identical logits.
- **Logits-side determinism** is the hard half, and the bar rises with sampling: temp>0 amplifies near-ties that argmax absorbs.
- **Batch invariance is not there yet.** Batched decode and prefill are not bitwise equal to B1 (`TECHDEBT.md`, dynamic batching). Until they are, exact replay covers only requests whose batch composition is reproduced, and logprob tolerance covers the rest.

**Provider-contract consequences**: temp=0 traffic is where token comparison is sharpest. Providers must honor and log per-request seeds, and may not force temp>0 or refuse seed logging, since that is the obvious loophole around replay audits. Logprob audits do not depend on temperature and are the primary signal for temp>0 traffic; logprobs themselves do not exist yet (`TECHDEBT.md`, serving API).

*Pricing data in §2 as of 2026-08 and in §5a as of 2026-09-26; re-verify before external use.*
