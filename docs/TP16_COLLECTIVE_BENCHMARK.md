# TP16 collective benchmark

This benchmark exercises the generic TP16 BF16 boundary collective on all 16
ranks. It is deliberately separate from a DSV4 model launch: the current DSV4
stage module is pipeline-slice based and refuses a tensor-parallel pack, so a
recipe alone must not be reported as TP inference.

Each operation reduces one DSV4 boundary token per request. The payload is
`B * 4 * 4096` BF16 elements, matching the four hyper-connection streams at a
stage boundary. The benchmark fills each rank with its rank value, checks the
exact BF16 result (`sum(1..16)`), and reports the collective-only token
equivalent rate.

Build it with:

```sh
make build/spark_tp_collective_benchmark
```

Run one process on every rank, using the same host CSV, port, and identifier;
only `--rank` changes. The peer list is ordered by rank and must contain all
16 hosts:

```sh
HOSTS=10.10.100.10,10.10.100.11,10.10.100.12,10.10.100.13,10.10.100.14,10.10.100.15,10.10.100.16,10.10.100.17,10.10.100.18,10.10.100.19,10.10.100.20,10.10.100.21,10.10.100.22,10.10.100.23,10.10.100.24,10.10.100.25
build/spark_tp_collective_benchmark --rank 0 --hosts "$HOSTS" \
  --batch 1 --operations 100 --warmup 5 --port 29160 --identifier 1
```

Use `--batch 8` for the B8 case. The collective uses BF16 wire/staging data
and F32 accumulation per recursive-doubling step. The transport progress loop
drains send and receive concurrently; the protocol headers remain ordered and
validated before the payload exchange.

These numbers qualify the transport boundary only. Full DSV4 TP16 inference
still requires rank-sharded stage-pack geometry, shard-aware DSV4 kernels, and
a distributed final-head argmax; those are separate implementation gates.
