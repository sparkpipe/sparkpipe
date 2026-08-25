# STEP 5 — 8-rank TP8 bring-up PASS — 2026-08-25

Ring: spark8(r0) spark9(r1) sparka(r2) sparkb(r3) spark1(r4) sparkd(r5)
sparke(r6) sparkf(r7) — spark1 substitutes unreachable sparkc per staged
configs (peers list index 4, control 19484).

Deployment: fixed driver dd2ba28c7d936a48ac15dd95b2d235acd85521600bdd1c669d3d1ddaed27e13f
+ traced adapter (358568 B) fleet-wide; backups kept as lib/*.bringup-bak.
Launches staggered 6 s, logs /tmp/glm52-tp8-rank<N>.log per host.

Result (all 8/8):
```
hidden_spark_rdma_fabric_ready ... device=rocep1s0f1
GLM52MODTRACE init tp_collective rc=0
GLM52MODTRACE init speculator rc=0
model_residentd ready rank=N stage=N inflight=4 active=16 rows=16 resident=16
  adapter=spark.glm52.serving-adapter.tp8.expert_fp8.v1
  model=zai-org/GLM-5.2 revision=b4734de4facf877f85769a911abafc5283eab3d9
  tcp=<host>:1948<N>
```
RDMA fabric up on every host; TP collective formed across all peers incl.
spark1; speculation armed-refused as designed for fanout (speculate=0).
First-ever full GLM 5.2 TP8 resident ring on this host set.
