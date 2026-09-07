# Tap extraction seam: glm5_next -> DFlash2 draft service (5090)

Date: 2026-09-03. For: glm flash dev (owner of modules/glm5_next_resident_decode_stage).
From: sparkdev lane/glm5next-spec. This is a seam SPEC, not a code change request on your lane —
the only files touched live in your module, so you own the implementation. Everything below is
measured or read from the drafter contract, not assumed.

## The contract (from /srv/drafters/glm-5.3-flash-dflash2/config.json)

- Tap layers: [5, 14, 24, 33, 42] (`dflash_config.target_layer_ids`).
- Tap vector per layer: 4096-wide bf16 (the residual stream after that layer's output,
  post-all-reduce — every TP rank has it replicated, no extra collective).
- Sliding window: 2048 positions. The drafter consumes context rows [A-2048, A) contiguously.
- Position relation (the trap): anchor token t_A at position A pairs with tap g_{A-1} as the
  LAST context row. A token sampled from a row's logits has NOT been through the target, so its
  tap does not exist. Never ship a tap for the anchor position itself.

## What the module must do

1. After the all-reduce at each tap layer, copy the 4096-bf16 hidden for each processed row into
   a per-lane ring keyed by ABSOLUTE position, capacity 2048 (2048 x 5 x 4096 x 2B = 80 MiB/lane;
   host-pinned is fine — async D2H of 40 KiB per tapped layer on a side stream).
2. On commit of N tokens (multi-token accept included), the N new tap rows are the positions the
   target actually executed as inputs — never the terminal sampled token.
3. Ship them out via the draft bridge (already merged on this lane: ring/transport/draft_bridge.c,
   header include/sparkpipe/spark_draft_bridge.h, 8 loopback gates + live 5090 smoke PASS).
   DFT3 rules the server enforces loudly: first request for (sequence, generation) must cover the
   whole window; continuation rows may overlap the watermark (re-projection path); any coverage
   gap -> STATUS_BAD_SEQUENCE and the lane must resync with a fresh generation.

## Interaction caution (your lane's open bug)

Rank-0 D2H per layer is the exact pattern that MASKS the multi-row prefill NCCL race. The tap
copies add five small async D2Hs per step on the coordinator rank. Keep them on a side stream,
never add a device sync to the hot path, and re-run the multi-row validator with tap extraction
ON before declaring the race fixed — the timing shift can unmask or re-mask it.

## Failure posture

Tap extraction is optional per lane: if the bridge is unreachable or the server rejects, the lane
degrades to local MTP (byte-exact, already gated) and counts the degradation. Target correctness
never depends on remote draft state.
