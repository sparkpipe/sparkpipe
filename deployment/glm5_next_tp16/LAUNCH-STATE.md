# glm5_next TP16 LAUNCH-STATE (glm5-kda lane, 2026-08-29)

Runtime roots: /home/<host>/sparkdata/glm5_next.tp16 (host = rank index
in hex: spark0..sparkf = ranks 0..15). Deployed binary set (sha-verified
16/16): lib/model_driver.so 0292a3e55f45b2fd (branch lane/glm5-kda
dace693 - the reduce-then-place HC fix; FIRST REAL TOKENS, see
docs/AGENT_LANE_BRIEFS/reports/glm5-kda-2026-08-29.md), serving adapter
+ api + residentd unchanged from the realtokens2 wave. Probes: the
current standing wave runs WITHOUT SPARK_GLM5_NEXT_PROBE (the diag
ladder is in the driver but env-gated; arming it stalls the chain by
design). Known: generation is degenerate (packer fused-shard defect,
fix committed 1dac68b, needs the 16-rank REPACK - see the lane report's
handoff); the KDA state slot carries across requests on a resident slot
(first-request outputs are canonical).

STOP (cwd-scoped TERM - never -f, never other cwds; spark5 runs the
dry-template2 lane's processes under /tmp/dry2-* and /home/spark5/sparkpipe):

    for h in spark0 spark1 ... sparkf; do ssh $h "rr=/home/$h/sparkdata/glm5_next.tp16; \
      for p in \$(pgrep -x sparkpipe_model); do c=\$(readlink /proc/\$p/cwd 2>/dev/null); \
      [ \"\$c\" = \"\$rr\" ] && kill -TERM \$p; done; exit 0"; done
    # wait for zero glm5 procs (collective teardown can take ~60 s), then 45 s settle

START (all 16 in the same second; DECIMAL rank indices - the hex digit
launches died with deployment=invalid_argument; setsid + session hold is
REQUIRED):

    for i in $(seq 0 15); do h=$(printf "spark%x" $i); rr="/home/$h/sparkdata/glm5_next.tp16"
      ssh "$h" "cd $rr && rm -f residentd.log && setsid nohup env \
        LD_LIBRARY_PATH=$rr/lib ./bin/sparkpipe_model_residentd --deployment model_resident.json \
        --rank-index $i > residentd.log 2>&1 < /dev/null & sleep 1" &
    done; wait

READY: `grep -c "model_residentd ready" residentd.log` = 1 on all 16
(180s transport connect window; the launch ssh fan-out itself may hold
the caller past 2 min - verify readiness in a fresh loop).

API (after 16/16 ready): on spark0 only -
`cd ~/sparkdata/glm5_next.tp16 && setsid nohup ./bin/sparkpipe_model_api
--deployment model_resident.json --runtime-root . --port 8433 >
api.log 2>&1 < /dev/null &` then GET /health (poll until ready; the api
takes ~10 s to hold the residentd's client slot).

Known state: status-0 REAL tokens (first-request receipt
[116315,41267x15] on the probe build; values identical without probes).
Degenerate repetition + a low ~0.5-2 head-score spread remain: the
16-rank REPACK with the fixed packer is the next gate.
