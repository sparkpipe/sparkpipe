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
