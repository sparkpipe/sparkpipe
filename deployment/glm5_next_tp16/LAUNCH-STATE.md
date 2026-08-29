# glm5_next TP16 LAUNCH-STATE (glm5-closeout lane, 2026-08-29)

Runtime roots: /home/<host>/sparkdata/glm5_next.tp16 (host = rank index
in hex: spark0..sparkf = ranks 0..15). Deployed binary set 16/16:
lib/model_driver.so sha256-prefix 6ca5f16b8b4259c5 (2,290,016 B, the
coordinator's 2026-08-29 12:50 UTC deploy; the kda receipt
0292a3e55f45b2fd is STALE), serving adapter + api + residentd unchanged.
NOTE: this driver returns BUSY from create() when SPARK_GLM5_NEXT_PROBE
is armed — the per-layer diag ladder does NOT arm on this build
(initialize=busy rc=15 fleet-wide; see
docs/AGENT_LANE_BRIEFS/reports/glm5-closeout-2026-08-29.md).

PACKS (the FIXED repack, LANDED 2026-08-29): packs/<name>.g5nsp is a
SYMLINK -> ~/glm53_packs_fixed/glm5_next_stage.tp16.rank<r>.g5nsp on each
node (the pre-repack packs remain at ~/glm53_packs/ for rollback; the
swap left .pre-closeout-bak symlinks in packs/). Every pack: 1160 tensors,
21,706,046,976 bytes, plan-diff + round-trip verified 16/16 (directory
sha256 421ef0989c054f67 on all ranks; rank 4 was rebuilt on spark9 and
delivered sha-identical 4a36178167143d4d).

REQUIRED AFTER ANY REPACK: header provenance patch — the packer emits
zeros for contract/config/recipe and the module rejects the pack
(hash_mismatch at adapter_initialize, rc=7). Per node:

    python3 tools/glm5_next_pack_header_patch.py \
      --pack <deployed pack symlink path> \
      --reference ~/glm53_packs/<same pack name> \
      --expect-contract-hex a40e9ec5fbfb0c1a180162c9d82915c887e8549fbd779c9f5dacb780a1498db4

(branch lane/glm5-closeout; ship via git bundle -> `git fetch` ->
`git archive` to a scratch dir — see the closeout report for the flow.)

STATUS: 16/16 residentd ready on the fixed packs, api healthy on
spark0:8433. GENERATION STILL DEGENERATE on a served=0 first request
([66188 x21, ...] on the standing prompt); the same prompt yields
DIFFERENT repeats across requests — recurrence state persists across
client sessions (api restarts do NOT reset it; only a residentd wave
does). COMPSEC-17 and the M5 exact-32K cell remain BLOCKED on coherence.

STOP (cwd-scoped TERM — never -f, never other cwds; spark5 runs the
dry-template2 lane's processes under /tmp/dry2-* and /home/spark5/sparkpipe,
sparke the K3 build):

    for h in spark0 spark1 ... sparkf; do ssh $h "rr=/home/$h/sparkdata/glm5_next.tp16; \
      for p in \$(pgrep -x sparkpipe_model); do c=\$(readlink /proc/\$p/cwd 2>/dev/null); \
      [ \"\$c\" = \"\$rr\" ] && kill -TERM \$p; done; exit 0"; done
    # wait for zero glm5 procs (collective teardown can take ~60 s), then 45 s settle

START (STAGGER 2 s per rank — two 16-parallel fan-outs tripped the
controller ssh proxy with "No route to host"; the 32 s spread is well
inside the 180 s transport window; DECIMAL rank indices; setsid REQUIRED):

    for i in $(seq 0 15); do h=$(printf "spark%x" $i); rr="/home/$h/sparkdata/glm5_next.tp16"
      ssh "$h" "cd $rr && rm -f residentd.log && setsid nohup env \
        LD_LIBRARY_PATH=$rr/lib ./bin/sparkpipe_model_residentd --deployment model_resident.json \
        --rank-index $i > residentd.log 2>&1 < /dev/null & sleep 1"
      sleep 2
    done

READY: `grep -c "model_residentd ready" residentd.log` = 1 on all 16
(verify readiness in a fresh loop).

API (after 16/16 ready): on spark0 only -
`cd ~/sparkdata/glm5_next.tp16 && setsid nohup ./bin/sparkpipe_model_api
--deployment model_resident.json --runtime-root . --port 8433 >
api.log 2>&1 < /dev/null &` then GET /health — CHECK "served":0 for a
canonical first-request reading (recurrence state survives api restarts;
only a residentd wave resets slots).

Known state: fixed packs changed the output distribution (cold first
token 116315 on the OLD packs -> 66188 on the FIXED packs) but generation
still collapses into a repeat attractor. Ranked candidates + evidence in
the closeout report.
