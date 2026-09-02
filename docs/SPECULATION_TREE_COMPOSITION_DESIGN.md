# Speculation TREE COMPOSITION — multiple speculators, one tree (2026-09-03)

Operator retasking (2026-09-01 night, queue note tasking-update): refactor
the speculation code for TREES composed from MULTIPLE speculators — the
5090 direction: a shared drafter drafting for ALL models, sparks verify.
Owner: kimi-k3 lane. Design-first; implementation is sequenced behind
accuracy + non-spec speed to SOTA (operator sequencing, unchanged).

## What exists (surveyed 09-03)

- `include/sparkpipe/spark_speculation_tree.h`: MODEL-NEUTRAL tree
  machinery — node topology rows, resolve, accepted-count, fallback rows,
  topology validity. `model-families/glm52/.../spark_glm52_mtp_tree.h`
  pins ONE static shape (5 candidates, 6 verifier rows, 3 execution
  steps) and keeps the legacy names as aliases.
  tests/test_speculation_tree_pin.c pins the glm52 constants.
- `include/sparkpipe/spark_speculation_provider.h`: the provider slot —
  lifecycle + contract only, hot loop provider-owned. ONE provider per
  adapter; the draft contract is LINEAR-CHAIN shaped (a flat id array;
  VerifyContract.chain_width + accepted_token_count).
- Drafter families: MTP (dsv4/glm52/qwen-flash), DFlash/2, DSpark/2
  (glm52's separate backend module is the provider-as-unit precedent).

## The gap

Nothing composes. One adapter carries one provider; a provider drafts a
chain, not a subtree; verify accounting assumes one chain. The 5090
shared drafter needs to CONTRIBUTE nodes into the target model's tree
alongside the model's own speculator(s), and the merged tree must verify
in one pass with one accept decision.

## Design: composition at the node level, one verifier

1. THE SHARED TREE stays the model-neutral machinery: verifier rows are
   the target model's unit of work; each row = one draft node with
   (parent, depth, position-offset). What changes is who fills it.
2. PROVIDER DELTA — Draft grows structure:
       SparkSpeculationDraftNode { uint32_t parent_row; uint32_t depth;
           uint32_t token; float score; uint16_t provider_slot; }
       SparkSpeculationDraft { nodes; count; /* plus the linear form for
           single-chain providers, flagged in the descriptor */ }
   A provider descriptor gains `contributes_subtree` + a priority (the
   merge policy: depth-budget and row-budget split across providers by
   priority; the verifier row count is the HARD budget — composition
   never grows the verify cost).
3. THE COMPOSITOR (host, common code, model-neutral — the tree
   machinery's natural extension): takes N providers' node sets, merges
   by (priority, score) into the verifier-row budget, validates the
   topology (SparkSpeculationTreeTopologyIsValid generalized to
   multi-root forests rooted at the anchor), emits the verifier row
   plan. THE COMPOSITOR OWNS NOTHING HOT: providers still draft into
   their own buffers; the merge is a host-side ordering pass.
4. VERIFY ACCOUNTING: the existing SparkSpeculationTreeResolve already
   returns the accepted resolution (root-to-node path). The adapter's
   ONE verify_account gains the resolution → per-provider attribution
   (which provider's nodes were accepted) so per-speculator acceptance
   rates are measurable (the tuning signal for priority/budget).
5. THE 5090 SHARED DRAFTER = a remote provider with the SAME ops table:
   draft_begin/next ride the sparkf uplink (request: committed context +
   tree budget; response: node set). The ops table is the only seam, so
   the remote form is a transport detail — the adapter cannot tell a
   local MTP head from the workstation drafter. Latency rule: a remote
   provider drafts ONE ROUND AHEAD (its nodes ride the NEXT verify, not
   the current one), so uplink RTT hides behind the target model's
   forward; a missed round simply contributes zero nodes that round
   (the tree degrades to the local speculators, never stalls).
6. KV CONTRACTS compose per-provider unchanged: each provider keeps its
   SparkSpeculationKvContract (scratch/tail frames, block history); the
   compositor intersects them into the wave's frame plan, and a provider
   whose KV shape cannot fit the merged tree is refused at composition
   with the reason (the supports()->WHY pattern).

## Sequencing (behind SOTA, per operator)

1. ABI delta + compositor + host pins (the glm52 tree re-expressed as a
   composition of ONE provider — bit-identical resolutions, the
   regression gate).
2. Two-local-provider composition on ONE family (cheapest real cell:
   glm52 MTP head + a second drafter; acceptance-rate receipt).
3. The 5090 remote provider (sparkf uplink) — last, per the operator's
   sequencing; the interface work in 1-2 is exactly what makes it a
   transport detail.

Every property lands as a host pin (the test_speculation_tree_pin
pattern) BEFORE any cell runs; the merged-verify equivalence (composed
tree resolve == the same tree verified as one provider's output) is the
S1-style host gate for this program.

## Step 1 in implementation terms (added 09-03 after the API survey)

The blocker surfaced by reading `include/sparkpipe/spark_speculation_tree.h`:
the machinery is COMPILE-TIME-shaped — the topology is a static table
behind `#define SPARK_SPECULATION_TREE_*` constants, and resolve walks it
directly. Composition needs a RUNTIME plan (the compositor's whole output
is a plan). Step 1 therefore lands as a refactor + a wrapper, not new
semantics:

1. `SparkSpeculationTreePlan` (runtime): row count, the node table
   (`SparkSpeculationTreeNode rows[MAX_VERIFIER_ROWS]`), candidate count,
   vocab bound, max committed, context extension — the five macros become
   fields; `MAX_VERIFIER_ROWS` bounds composition (64 initial).
2. The existing functions gain a `const SparkSpeculationTreePlan *`
   first parameter (`SparkSpeculationTreeNodeAt(plan,row)` and friends);
   the macro form becomes a static inline wrapper that builds a
   `static const Plan` from the includer's constants — glm52's include
   surface and all call sites compile UNCHANGED.
3. The step-1 gate is the bit-identity claim: a host pin runs the glm52
   topology through BOTH forms (macro-static and wrapper-built plan)
   across exhaustive verifier-token vectors and requires byte-identical
   resolutions. The pin lands with the refactor in one commit.
4. THEN the compositor (still step 1): `SparkSpeculationTreeCompose(`
   `const SparkSpeculationDraftNode *sets[], const uint32_t counts[],`
   `const uint32_t priorities[], uint32_t set_count,`
   `SparkSpeculationTreePlan *out)` — greedy by (priority, score) under
   the row budget, reusing TopologyIsValid generalized to a plan. The
   one-provider glm52 composition exercises it before any second
   provider exists.
