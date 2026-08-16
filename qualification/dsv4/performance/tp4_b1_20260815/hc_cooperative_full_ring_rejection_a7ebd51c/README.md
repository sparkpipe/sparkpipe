# DSV4 TP4 B1 cooperative Hc rejection

The cooperative Hc experiment fused the B1 finalize and pre-reduce work into
one cooperative grid while keeping the existing general path for larger row
counts. It changed no target, spine, activation, or KV precision.

The isolated actual-shape probe was exact in 93/93 pairs and improved the Hc
component by 13.1800%, saving 0.160555 ms per model step. The real cached-prefill
TP4 O128 test nevertheless regressed in every alternating pair:

| Pair | Lean control tok/s | Hc candidate tok/s | Delta |
| ---: | ---: | ---: | ---: |
| 1 | 40.4994 | 40.3316 | -0.4142% |
| 2 | 40.4917 | 40.3548 | -0.3379% |
| 3 | 40.4347 | 40.3342 | -0.2486% |
| Mean | **40.4753** | **40.3402** | **-0.3336%** |

All six O128 token streams and the O24 gate were bit-exact. The end-to-end
result therefore rejects the production change strictly on performance. The
lean source was restored exactly; the rejected patch is retained here only as
diagnostic evidence.

See [summary.json](summary.json) for artifact identities and raw receipt hashes,
the isolated_poc directory for the component probe, and the integration
directory for the rejected integration snapshot.
