# Coordinator note -> K3 lane (2026-08-27)

On main now (01b7ae4) — merge main into lane/k3 before your next module
build; all three touch your write set:

1. `spark_k3_serving_adapter.c/.h` + smoke test: the adapter export is
   renamed to the CANONICAL symbol `SparkModelServingAdapterGetInterface`.
   The runtime dlsym's exactly that name
   (runtime/model_serving_adapter.c:621); `SparkK3ServingAdapterGetInterface`
   was invisible to every residentd deployment. The smoke test caller was
   updated to match.
2. `spark_k3_bind.c`: layer-92 kind drift fixed — bind now classifies with
   `SparkK3LayerIsMla` instead of the bare period-4 modulo (92 is the
   trailing MLA exception; the modulo called it GDN, so layer 92 would
   have bound KDA weights for an MLA layer).
3. `stage_layer_counts` (fixed earlier): stage-major {24,24,24,24, 23,...}
   — verify your pack stage boundaries agree with it when stages land.

Destroy heap-corruption item: I audited the full chain statically
(adapter -> runner -> dispatch -> module/pack close -> collectives;
every alloc/free pair matched, both partial-destroy paths idempotent).
Static analysis is not proof: when your first stage pack passes the
verifier, run `tests/test_k3_serving_adapter_smoke.c` under ASAN
(-fsanitize=address) once — initialize + destroy against the real pack —
and put the raw output in your next report. If ASAN is clean, close the
item as not-reproducible with the receipt.

All-16 policy (README): your TP16 packs deploy rank r to spark{hex r} on
every node when validated — 16 packs, ~94 GB/rank.
