# GLM52 PP13 Rank Daemon

`sparkpipe_glm52_pp13_rank_daemon` is the resident per-rank process for the
fixed 13-rank GLM-5.2 FP8 pipeline.

It is intentionally fail-closed. It does not create a simulation transport, does
not search fallback pack directories, and does not run without a production
model driver shared object.

## Runtime Contract

Each rank process:

```text
validates its exact PP13 rank plan
validates its local FP8 .sp* files
loads the required hidden-transport shared object
opens neighbor hidden-transport sessions
loads the GLM-5.2 model-driver shared object
creates the driver instance so rank weights stay resident
initializes the production stage runner
stays resident until SIGINT or SIGTERM
```

`spark0` additionally opens the final-event listener.

`sparkc` additionally connects to the final-event listener and sends completion
events back to `spark0`.

## Launch Shape

Example for rank 0:

```sh
/home/spark0/sparkpipe_runtime/bin/sparkpipe_glm52_pp13_rank_daemon \
    --rank 0 \
    --fp8-pack-root /home/spark0/models/sparkpipe/glm52_fp8_pp13_stage_payload_v1 \
    --transport-so /home/spark0/sparkpipe_runtime/lib/libsparkpipe_hidden_transport_production.so \
    --driver-so /home/spark0/sparkpipe_runtime/lib/libglm52_pp13_rank_driver.so \
    --node-target cuda.sm121.glm52.pp13.fp8 \
    --final-event-bind 10.20.0.10
```

Example for rank 12:

```sh
/home/sparkc/sparkpipe_runtime/bin/sparkpipe_glm52_pp13_rank_daemon \
    --rank 12 \
    --fp8-pack-root /home/sparkc/models/sparkpipe/glm52_fp8_pp13_stage_payload_v1 \
    --transport-so /home/sparkc/sparkpipe_runtime/lib/libsparkpipe_hidden_transport_production.so \
    --driver-so /home/sparkc/sparkpipe_runtime/lib/libglm52_pp13_rank_driver.so \
    --node-target cuda.sm121.glm52.pp13.fp8 \
    --final-event-return-host spark0
```

All other ranks use the same command with their own `--rank`.

## Final Event Route

The final-event route is:

```text
source: sparkc
sink:   spark0
port:   PP13 port base + 200
name:   sparkc_to_spark0_final_events
```

The daemon currently returns driver completion metadata:

```text
request id
sequence id
sequence position
program id
driver dispatch slot
accepted token count
status
service time
```

Token-id and text streaming remain a gateway/service integration step above
this device-driver event route.

## Required Missing Pieces

The daemon will still fail to start a real 13-rank serving ring until these
production artifacts exist on every rank:

```text
production hidden-transport shared object
rank-local GLM-5.2 model-driver shared object
driver node-context binding for resident weights, KV, graph buckets, and packs
dispatch loop that maps Spark0 service requests into rank runner submissions
Spark0 gateway wiring from final events to SSE token streams
```

Those are missing components, not places to add fallbacks.
