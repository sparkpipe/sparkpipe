# DFlash2 at TP>1: Analysis

## Current State
- qwen36 adapter has NO explicit TP>1 restriction on speculation (architecturally permissive)
- glm52 adapter REFUSES speculation when tp_degree != 1 ("draft_transport_not_wired_for_fanout")
- No driver has TESTED speculation at TP>1

## Why It Should Work
In a TP setup:
- Embedding output before the first sharded layer is REPLICATED across ranks (not sharded)
- Each rank can run the same drafter model independently on the same input
- Same drafter weights (replicated) + same input = same draft tokens on every rank
- The verify pass already goes through existing TP collectives (each rank processes its shard)
- Acceptance decision needs cross-rank consistency → use the existing all-reduce/all-gather

## What Needs To Change (GLM52)
1. Remove the tp_degree==1 gate in SparkGlm52ServingArmSpeculation (~line 592)
2. Add a post-draft consistency check (allreduce or trust-rank0 pattern)
3. Ensure the drafter pack is deployed as replicated (not sharded) to each rank
4. Test with the 8-rank TP8 deployment

## Risk Assessment
- LOW risk: if drafter inputs are truly replicated, drafts are deterministic and identical across ranks
- MEDIUM risk: floating-point non-determinism from different GPU instances could cause rare divergence
- MITIGATION: after drafting, do a cheap allreduce on draft token ids; if any rank disagrees, fall back to plain decode for that round

## Estimated Effort
- Code change: ~50 lines (remove gate + add consistency check)
- Testing: requires 8-rank hardware (available now)
- Risk: low — worst case falls back to plain decode
