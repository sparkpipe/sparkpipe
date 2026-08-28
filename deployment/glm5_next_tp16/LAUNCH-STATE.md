# glm5_next TP16 fleet launch state (lane-glm53)

Handoff notes for the coordinator: every step's state lives ON THE NODES.

## Layout (all 16 nodes, identical)
- Runtime root: `~/sparkdata/glm5_next.tp16/`
  - `bin/sparkpipe_model_residentd`, `bin/sparkpipe_model_api`
  - `lib/model_driver.so` (sha a40e9ec5-identity, artifact e9ba95ab),
    `lib/model_serving_adapter.so`, `lib/hidden_transport.so`
  - `config/stage.json` (per-rank, adapter-exact nine members)
  - `model_resident.json` (schema v2, 16 nodes, host-rdma, control 19560+rank)
  - `packs/glm5_next_stage.tp16.rank<R>.g5nsp` -> symlink to
    `~/glm53_packs/...` (21.7 GB, header contract bytes = a40e9ec5...)
- KV backing dir: `~/kvcache/glm5_next.tp16/`
- Logs: `/tmp/g5n_residentd.log` per node

## Node -> rank map (hex)
spark0=0 spark1=1 ... spark9=9 sparka=10 sparkb=11 sparkc=12 sparkd=13
sparke=14 sparkf=15

## Launch command (per node; run ALL 16 within the 180s collective window)
    cd ~/sparkdata/glm5_next.tp16 && \
    nohup bin/sparkpipe_model_residentd --deployment model_resident.json \
        --rank-index <R> > /tmp/g5n_residentd.log 2>&1 &

Simultaneous launch from the controller mac (all at once, not sequential —
a staggered launch outlives the 180s hidden-transport connect timeout and
early ranks exit with `initialize=busy`, the failure already seen once):

    for h in spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 \
             spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf; do
      r=$((0x$(echo $h | sed s/spark//)))
      ssh $h "cd ~/sparkdata/glm5_next.tp16 && nohup bin/sparkpipe_model_residentd \
        --deployment model_resident.json --rank-index $r \
        > /tmp/g5n_residentd.log 2>&1 < /dev/null &" &
    done; wait

## Ready line to wait for (per node log)
`model_residentd ready` — then the API on the head (spark0):
    cd ~/sparkdata/glm5_next.tp16 && nohup bin/sparkpipe_model_api \
        --deployment model_resident.json \
        --runtime-root ~/sparkdata/glm5_next.tp16 --port 8433 \
        > /tmp/g5n_api.log 2>&1 < /dev/null &

## Known-good evidence so far
- Single-rank smoke on spark9 passed the full identity chain; the ONLY
  failure was the collective peer wait (expected alone).
- First fleet attempt (staggered): early ranks hit `initialize=busy`
  (collective timeout) — retried with the simultaneous launcher above.

## GPU hygiene
- spark2's 27B dev daemon was TERM-stopped per lane rules (documented in
  the lane report). Nothing else holds GPUs.
