# Lazy expert acquisition cost, GLM-5.3 Full fp8 TP16 (weightd lane 6)

Build a477cfa + drivers/glmfull-w7 (lease timing under SPARK_GLM52_T1=1),
expert pool 8 GiB per rank (per-chunk lazy: pool < arena), rank 1 (spark1).
`us` is the wall time of SparkGlm52LazyExperts per routed layer: route
event wait + weightd acquire + begin use + expert launch.

| run | calls | mean | p50 | p10 | p90 |
| --- | --- | --- | --- | --- | --- |
| b1_o16 story prompt (62 waves) | 4650 | 38.2 ms | 33.7 ms | - | 63.4 ms |
| capital prompt, 3rd identical repeat (experts already pooled) | 300 | 27.9 ms | 26.5 ms | 20.1 ms | - |

Split (capital repeat): the route event wait is 15-73 us on average; the
rest is SparkWeightdMapAcquire/BeginUse. Without a premapped pool the map
imports, maps and sets access for every chunk of every lease and unmaps
them on release, so even pooled experts pay about 3 ms per key. 75 routed
layers x ~25-38 ms = 2-2.9 s per token, which is the whole B1 cost.

## Chain stages on rank 1 (SPARK_GLM52_T1=1, reference prompts, t_us trace)

| span | n | mean | p50 |
| --- | --- | --- | --- |
| attention launch -> attention reduce done, layer 0 | 21 | 5.9 ms | 0.67 ms |
| same, layers 1-77 | ~1600 | 3.8-4.1 ms | 3.6-3.8 ms |
| dense MLP launch -> reduce done (layers 0-2) | 63 | 0.28 ms | 0.25 ms |
| routed MLP stage -> reduce done (includes the lease) | ~1575 | 32.6 ms | 31.0 ms |
| lease alone | 1575 | 19.1 ms | 18.8 ms |

Layers 1-77 include rank 0's T1 stream tap (a synchronous copy and a
6144-value print per layer) because every rank waits for rank 0 at the next
reduce; layer 0 of each wave has no tap in front of it, so about 0.6 ms is
the real attention + reduce + host callback cost. With experts pinned the
host-driven chain is therefore about 0.6 ms (attention) + ~0.3-0.5 ms (MLP)
per layer, about 85 ms per token, i.e. roughly 10-12 tok/s at B1 before any
graph or device-driven chain work.
