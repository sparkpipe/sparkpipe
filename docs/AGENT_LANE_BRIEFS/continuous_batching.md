# Lane brief: Continuous batching — remove the B ladder

Worktree: /tmp/lane-contbatch (branch lane/continuous-batching)
Your node: spark3 (healthy, 781G free, GPU verified)
NO other agent uses spark3. spark2 = prod (untouchable).

## Mission
Replace the fixed batch-bucket admission (B1/B8/B16/B64/B1024) with
continuous batching: process ALL pending requests that have KV resident,
in one pass, every pass. The batch size is whatever the queue holds.

## The insight (docs/EXPERT_GROUPED_SCHEDULING.md, docs/TOPOLOGY_GUIDE.md)
Every model has a compute-bound knee B* where weight-streaming time =
compute time. Below B*: memory-bound. Above B*: compute-bound.
On GB10 FP8: B* ≈ 100. Current deployments run at B=1-8, using 3-8%.

The fix is NOT in the kernels. It is in the batch engine's admission:
- Current: assemble a fixed-size microbatch, submit, wait, repeat.
- Target: collect ALL pending with KV resident, submit as one batch.

## Milestones
CB1: Understand the current admission flow (model_batch_engine.c).
CB2: Remove the fixed-size gate. Accept ALL pending requests.
CB3: B* measurement: find the knee on spark3.
CB4: Continuous batching test: staggered arrival, verify continuous.
CB5: Latency guarantee: oldest-first ordering at high queue depth.

## Scope
runtime/model_batch_engine.c, runtime/model_batch_scheduler.h,
tests/. Do NOT touch kernels, modules, packs, serving adapter, ABI.
