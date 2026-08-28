# SparkPipe Compute Marketplace — Business Plan Draft

Status: DRAFT, 2026-08-28 (ratified into the repo 2026-08-29)
Scope: business model, competitive pricing, anti-cheating architecture, unit economics.

---

## 1. Model

A two-sided marketplace for LLM inference:

- **Supply side**: owners of compute (initially NVIDIA DGX Spark / GB10-class nodes) register their units and serve open-weight models. Providers **set their own per-token prices**.
- **Demand side**: customers reach the fleet through a **liteLLM front end** — one OpenAI-compatible API, routing across registered providers.
- **Platform fee**: the platform takes **10% in compute**, not cash. When a provider's capacity is fully sold, the platform controls 10% of their serving capacity and monetizes or allocates it directly. Audit/verification costs are **on top of** the 10% — they are funded from an additional small take, so the platform clears a net 10% (see §5).
- **Supply-side requirement**: providers must run **SparkPipe** firmware. This is both the quality floor (deterministic, qualified, exact-token serving) and the foundation of the anti-cheating system (§4).

Why providers join: monetize idle hardware at prices they set, with zero billing/gateway/demand-generation work. Why customers come: the cheapest verified-exact open-model inference, behind one API.

## 2. What the industry charges

### Marketplace / router take rates

| Platform | Model | Take rate |
| --- | --- | --- |
| **OpenRouter** | Router over inference providers; no token markup, fee on credits | 5.5% on credit purchases ($0.80 min); 5% on BYOK usage above $25k/mo |
| **Vast.ai** | GPU rental marketplace (supply = individuals' machines) | ~10–15% provider-side fee |
| **Salad** | Distributed consumer-GPU cloud | ~20–25% provider-side fee |
| **Akash / io.net** | DePIN container/GPU marketplaces | Pricing 50–70% below AWS; fees embedded in crypto settlement |

Key distinction: **OpenRouter is not a supply-side marketplace.** It resells existing providers (Together, Fireworks, DeepInfra…) who carry their own full serving margins underneath. The comparable set for *registering your own compute* is Vast/Salad/Akash/io.net — where 10–25% is the norm. A **net-10% take is at or below the industry floor** for compute aggregation, and dramatically below the all-in margin customers pay through the router+provider stack.

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

1. **Sampled re-execution (fraud proofs).** The platform operates trusted reference nodes (seeded from the audit take). A secret random sample (~0.5–2%) of *real, completed customer requests* is re-executed on trusted nodes with the identical pinned driver hash. Because SparkPipe serving is deterministic per (driver, weights, seed), comparison is exact token-stream equality, not statistics. Quantized or substituted weights diverge in greedy decode within tens of tokens. Real traffic is undetectable as an audit — there is no probe to dodge.
2. **Logprob audits.** Providers must return logprobs; the platform recomputes them on the reference model for sampled tokens. Logprobs are a high-dimensional side channel a cheaper model cannot produce — catches quantization specifically, where token streams may coincidentally agree. **Logprobs are temperature-independent evidence** — the primary audit signal for temp>0 traffic.
3. **Weight fingerprints.** Checkpoints distributed for serving carry a secret set of trigger→response pairs. Only the exact weights reproduce them; designed to break under quantization. Checked during audits at near-zero cost.
4. **Performance contracts (coarse filter).** SparkPipe has qualified kernel-level perf envelopes; a node consistently *faster* than qualified suggests quantization, consistently slower is a service-quality issue. Weak, nearly free.

**Enforcement economics**: providers post a bond; payouts settle after a challenge window; a mismatch triggers one disputable third-party re-execution, then slashing. Deterrence condition: `P(detection) × penalty > margin gained by cheating`. Cheating saves the cost delta between the real model and a quantized/smaller one — a modest multiple of serving cost — so even 1% sampling with meaningful slashing makes honesty the cheapest strategy. Optimistic-rollup logic, applied to tokens.

## 5. Unit economics of the fee

Goal: **clear 10% in compute, audit costs on top.**

- Audit cost at sampling rate *s*: re-execution is ~1:1 compute, plus logprob/fingerprint overhead — call it ~1.2× per sampled request. At s = 1%, audit compute ≈ 1.2% of GMV compute.
- Therefore gross compute take ≈ **11.5–12%** to clear a net 10% after verification. Alternatively: keep the take at 10%+ε and let sampling rate float with risk (lower for bonded, long-tenured providers; higher for new ones).
- Dynamic sampling also caps worst-case spend: new providers start at high coverage (up to 5%, paid by their bond tier), graduating down as reputation accrues.
- The platform's held compute is itself revenue-grade: sell it as platform-first-party capacity, use it as failover/reserve, or run the audit fleet from it. Verification literally pays for itself out of the same resource.

Illustrative: a provider selling $100k/yr of tokens yields ~$11.5k of compute to the platform; ~$1.4k funds their audit coverage; ~$10k clears.

## 6. Risks and open questions

- **Two-sided cold start.** No demand without supply, no supply without demand. Mitigation: seed with first-party fleet (the existing 16-node cluster), open supply registration once demand routing through liteLLM is live.
- **Determinism boundary.** Exact-token replay audits only work while all providers run pinned driver builds for a model version. Driver updates need coordinated flag days per model, or per-driver-version audit cohorts. Seed logging is mandatory (see appendix).
- **Replay cost at scale.** Long-context replays cost more than short ones; sampling should be weighted by request cost, not count, with caps (e.g. replay first N tokens + sampled continuation windows instead of full regenerations).
- **Legal/settlement.** In-compute fee needs clean accounting (compute credits as consideration), payout/challenge windows in the provider agreement, and a decision on fiat vs crypto settlement.
- **Provider margin reality check.** If a provider can earn more per GPU-hour on Vast/Salad than serving tokens here net of 11.5%, supply won't come. The answer must be demand density: a sold-out token market beats an idle rental listing.
- **Sophisticated adversaries.** Someone can serve honestly except when they suspect an audit — which is why audits are replays of real traffic, never synthetic probes. Accept that this is economics, not cryptography: make cheating EV-negative and detection evidence objective.

## 7. Near-term build hooks

1. liteLLM front end in front of the existing gateway (never expose `node/model_api.c` directly). **[STATUS: DONE — merged, docs/LITELLM_FRONTEND.md]**
2. Request-logging pipeline keyed by (driver hash, model contract hash, request) — the audit substrate.
3. Audit service: sampler → replay scheduler → comparator → challenge/slash state machine.
4. Provider onboarding: bond, pinned driver distribution, fingerprinted checkpoints.
5. Fee metering: compute-credit ledger tracking the platform's in-kind take per provider.

---

## Appendix A: Determinism status (2026-08-29)

SparkPipe today is **greedy-only** — the entire LM-head path is argmax
(`SparkLmHeadArgmaxKernel` and the screened/FP8-candidate head;
warp-then-block reduce, ties to lower index). No temperature/top-p/
multinomial sampler exists in the tree. **All exact-token-hash
receipts to date are temp=0 artifacts** — still valid as greedy-decode
determinism proofs, and that is exactly what replay audits need for
temp=0 traffic.

If/when real sampling is added, determinism at temp>0 has two halves:

- **Sampling-side (easy).** Counter-based RNG (philox-style) keyed on
  (request_seed, sequence_id, token_position). Same seed ⇒ same token,
  given identical logits. Providers must honor and log a per-request
  seed (the OpenAI API already has the field); audit replays reuse it.
  The RNG must never depend on runtime-varying state (clock, block
  scheduling, batch slot index).
- **Logits-side (hard — already our discipline).** Bit-identical
  logits across runs and batch compositions: fixed reduction orders,
  no atomics into fp32 accumulators, batch-invariant kernels — the
  receipts culture already gates this. Note the bar RISES with
  sampling: temp>0 amplifies near-ties that argmax silently absorbs
  (a one-ulp logit flip can change a sampled token where it would
  never flip an argmax).

**Provider-contract consequences**: temp=0 traffic is the canary-rich
environment (token comparison sharpest); providers must NOT be allowed
to force temp>0 or refuse seed logging — that is the obvious loophole
around replay audits. Logprob audits are temperature-independent and
are the primary signal for temp>0 traffic.

*Pricing data as of 2026-08; re-verify before external use.*
