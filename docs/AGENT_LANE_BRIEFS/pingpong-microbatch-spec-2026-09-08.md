# Microbatch ping-pong: hiding the TP16 collective behind compute at B>=2

Date: 2026-09-08. Depends on: the deterministic-TP-ordinal fix
(lane/tp-ordinal-determinism) — concurrent frame chains are unsafe until it lands.

## The measured basis (lane/allreduce-takeover, PR #816)

- Per-op collective latency is fixed-overhead-bound, ~147us at rows 1-4
  (tree) / ~78us (d2a), 219us at rows=8.
- Per-layer compute at B1 is ~700us (73ms/token over 45 layers x ~2 ARs).
- Serving today: one execution stream; all frames' GPU work serializes;
  each layer's allreduce waits for the stream to drain (D2A-TIMING:
  post=330-820us is stream-drain behind layer compute).

## The mechanism

Split each decode wave's rows into two halves on TWO execution streams.
While half A's layer collective is in flight (network + progress thread),
half B's layer computes on its stream, and vice versa. The collective
progress thread is CPU-side and already independent; the fold kernels
ride the op's own stream. Requirements:

1. Deterministic ordinals across concurrent chains (the dependency).
2. The adapter owns N=2 execution streams and assigns frames by slot
   parity (module ExecuteBatch already claims pipeline slots; the frame
   carries execution_stream).
3. The batch engine submits decode as two half-batch submissions (rows
   split by lane parity). Engine work: none — admission already packs
   ready lanes; the adapter splits the frame rows.
4. The collective engine already supports concurrent ops (credits + ack
   flow control, measured clean at 1024 async ops).

## Expected effect (from the measured numbers, not a promise)

B2 (2x1-row halves): each half's ~78-147us collective hides under the
other half's ~700us compute -> collective off the critical path. B4:
2x2 rows, same hiding. B8+: same shape; per-op latency at rows=4 is
still ~148us (measured), still hidden. The residual serial cost is the
non-overlappable tail (head maxloc + hc ops, chain begin/end).

## Honest caveats

- Ping-pong doubles the per-step op count (two half-width ops per layer
  instead of one full-width); at rows>=8 per half the fixed overhead is
  amortized identically, so no loss — but the B2 shape pays 2x fixed
  overhead vs one 2-row op, recovered by the hiding.
- The KDA/conv state advances are per-lane and independent — no
  cross-half hazard. The head maxloc runs per half; the emitted tokens
  per lane are independent — no merge semantics change.
- Fail loudly if the row split is odd (one half gets the extra row —
  deterministic assignment, no padding).
