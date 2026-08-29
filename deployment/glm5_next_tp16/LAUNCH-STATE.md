# glm5_next TP16 LAUNCH-STATE (real-tokens2 lane, 2026-08-29)

Runtime roots: /home/<host>/sparkdata/glm5_next.tp16 (host = rank index
in hex: spark0..sparkf = ranks 0..15). Deployed binary set (sha-verified
16/16): lib/model_driver.so 4f3b8f33..., lib/model_serving_adapter.so
88b58b0f..., lib/hidden_transport.so (coordinator diag build, kept),
bin/sparkpipe_model_api fa0891dc..., bin/sparkpipe_model_residentd
b5762b47.... Pre-deploy binaries: *.pre-rt2 alongside.

STOP (cwd-scoped TERM — never -f, never other cwds; spark5 runs the
dry-template2 lane's processes under /tmp/dry2-* and /home/spark5/sparkpipe):

    for h in spark0 spark1 ... sparkf; do ssh $h "rr=/home/$h/sparkdata/glm5_next.tp16; \
      for p in \$(pgrep -x sparkpipe_model); do c=\$(readlink /proc/\$p/cwd 2>/dev/null); \
      [ \"\$c\" = \"\$rr\" ] && kill -TERM \$p; done; exit 0"; done
    # wait for zero glm5 procs, then 45s TIME_WAIT settle

START (all 16 in the same second; setsid + 2s session hold is REQUIRED —
a bare `ssh host "... nohup ... &"` loses the child to the sshd HUP
before exec; committed tools/glm5_next_wave.sh has this latent bug):

    for spec in "spark0 0" "spark1 1" ... "sparkf 15"; do
      set -- $spec; h=$1; i=$2; rr="/home/$h/sparkdata/glm5_next.tp16"
      ssh "$h" "cd $rr && rm -f residentd.log && setsid nohup env \
        SPARK_GLM5_NEXT_PROBE=1 LD_LIBRARY_PATH=$rr/lib \
        ./bin/sparkpipe_model_residentd --deployment model_resident.json \
        --rank-index $i > residentd.log 2>&1 < /dev/null & sleep 2" &
    done; wait
    # SPARK_GLM5_NEXT_PROBE=1 arms the G5N-PROBE dumps (diag only)

READY: `grep -c "model_residentd ready" residentd.log` = 1 on all 16
(180s transport connect window; staggered launches get late joiners
refused).

API (after 16/16 ready): on spark0 only —
`cd ~/sparkdata/glm5_next.tp16 && setsid nohup ./bin/sparkpipe_model_api
--deployment model_resident.json --runtime-root . --port 8433 >
api.log 2>&1 < /dev/null &` then GET /health.

Known state: serving is healthy end to end; completions return
status-0 all-zero tokens (see
docs/AGENT_LANE_BRIEFS/reports/glm5-realtokens2-2026-08-29.md —
layer-34 attention zeroing; attention partials dead by layer 33).
