# NCCL 16-wide on the sparks — WORKING (2026-08-31 ~17:45 KRT)

## The working recipe (after 6 failure layers peeled):
1. NCCL_SOCKET_IFNAME=enp1s0f1np1 + NCCL_IB_HCA=rocep1s0f1 + NCCL_IB_GID_INDEX=3
   (the flash dev's pins — NCCL unpinned picks the wrong RoCE port 10.10.200.x
   and dies post-bootstrap with "remote process exited")
2. rank0 = spark5, launched FRESH first, NCCL_DEBUG=INFO on (INFO isn't cosmetic:
   the successful 2-node runs had it on — repeatable)
3. rank0 prints IDHEX <256 hex> to its line-buffered log; the fanout greps it and
   passes it INLINE as argv[4] (node /tmp AND $HOME are swept by agent cleanup
   loops — id files vanish between commands; the dev's file-based flow was the
   16-wide flakiness all along)
4. rank map: r = ((hex(host) - 5 + 16) % 16), spark5 = rank0
5. 3s settle after IDHEX before peers launch
6. libnccl FULL lib set on every node (libnccl.so + .so.2 + .so.2.28.9 +
   static + pkgconfig) at ~/nccl-src/build/lib — sparke had only the .so.2 and
   its rank died at dlopen while 15 ranks waited at RAS

## 16-wide receipts (all 16 ranks, 1000 iters, verified sum=136):
- sum 8KB bf16:  105.9 µs/op   (vs ~820 µs current serving hop = 7.7x)
- sum 14KB bf16: 103.3 µs/op   (the fused-collective serving size)
- sum 40KB bf16: 188.8 µs/op
- sum 80KB bf16: 217.7 µs/op
- max 8KB bf16:   94.4 µs/op
- f32 8KB:        96.7 µs/op
- 2-node for contrast: 8KB = 20.3 µs (the dev's 21.4 receipt reproduced)

## Serving arithmetic (glm5_next TP16, ~95 collectives/token):
- current: 95 x ~205 µs serialized host-tier ≈ 19.5 ms/token
- NCCL ring @14KB: 95 x 103.3 µs ≈ 9.8 ms/token  (plus: NCCL ops are
  stream-ordered — they pipeline with compute instead of serializing)
- expected decode gain: ≥10 ms/token of the 78.7 ms step ≈ 12%+ from the
  collective swap alone, compounding with the landed head/split wins
