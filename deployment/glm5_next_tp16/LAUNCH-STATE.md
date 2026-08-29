# glm5_next TP16 LAUNCH-STATE (probe-fix lane, 2026-08-29/30)

Runtime roots: /home/<host>/sparkdata/glm5_next.tp16 (host = rank index
in hex: spark0..sparkf = ranks 0..15). Current deployed driver set 16/16:
lib/model_driver.so sha256-prefix 43a5fda4df5b889b (the DIAG build: probe
ladder + cross-rank checksum, SPARK_GLM5_NEXT_PROBE-gated; non-probe twin
of the same source is 980fd0fcb42b39cb). Module artifact 01b22fb6…,
`glm5_next validator: PASS (0 failures)` against the CURRENT rank-0 pack.
Branch lane/probe-fix tip d1235a9.

PACKS (the COLS-SHARD o_proj repack, LANDED 2026-08-30):
packs/<name>.g5nsp is a SYMLINK ->
~/glm53_packs_fixed2/glm5_next_stage.tp16.rank<r>.g5nsp (the previous
row-sharded packs remain at ~/glm53_packs_fixed/, rollback = rename the
.pre-probefix2-bak symlinks back; the ORIGINAL packs remain at
~/glm53_packs/). Every pack: 1160 tensors, 21,706,046,976 bytes,
directory sha256 54e2474be88a2544 uniform on all ranks; verifier PASS
16/16; deep spot round-trips PASS on ranks 0/1/8/15. Rank 8 and rank 11
builders hit the slow-ceph path and were requeued solo on spark0/spark9,
shipped, sha-verified.

WHY THE REPACK: the attention OUT projections (ATTN_OUTPUT/KDA_OUT,
checkpoint [hidden, heads*dim]) were row-sharded (pack [hidden/tp, width])
while the out-GEMM consumes the col-shard TP partial [hidden, width/tp] —
a silently transposed o_proj on every rank, the cold-first-request
degeneration. Fixed on lane/probe-fix (2cc9de3): module policy col-shards
them, packer `shard="cols"`, all packs rebuilt. Generation changed but
STILL degenerates (see
docs/AGENT_LANE_BRIEFS/reports/probe-fix-2026-08-29.md) — remaining
defect is rank-invariant (post-reduce checksums bit-identical 16/16);
next instrument = reference semantic diff, per the report.

REQUIRED AFTER ANY REPACK: header provenance patch — the packer emits
zeros for contract/config/recipe and the module rejects the pack
(hash_mismatch at adapter_initialize, rc=7). Per node:

    python3 tools/glm5_next_pack_header_patch.py \
      --pack <deployed pack symlink path> \
      --reference ~/glm53_packs/<same pack name> \
      --expect-contract-hex a40e9ec5fbfb0c1a180162c9d82915c887e8549fbd779c9f5dacb780a1498db4

NOTE: run it only AFTER the builder writes its final receipt line (the
tool patches a mid-build file happily; verify size = 21706046976 first).
Then verify: tools/glm5_next_pack_verify.py --pack <p> --source
/mnt/model-warm/glm-5.3-flash --tp-rank <r> --tp-degree 16 [--skip-spot].

STATUS: 16/16 residentd ready on the fixed2 packs + DIAG driver, api
healthy on spark0:8433. Generation still degenerate (repeat attractors).
COMPSEC-17 and the M5 exact-32K cell remain BLOCKED on coherence. For
gate runs, relaunch the wave WITHOUT --probe (the diag ladder stalls the
chain by design; the same source builds the non-probe driver
980fd0fcb42b39cb).

STOP (cwd-scoped TERM — never -f, never other cwds; spark5 runs the
dry-template2 lane's processes, sparke the K3 build):

    for h in spark0 spark1 ... sparkf; do ssh $h "rr=/home/$h/sparkdata/glm5_next.tp16; \
      for p in \$(pgrep -f 'bin/sparkpipe_model_[r]esidentd'); do c=\$(readlink /proc/\$p/cwd 2>/dev/null); \
      [ \"\$c\" = \"\$rr\" ] && kill -TERM \$p; done; \
      for p in \$(pgrep -f 'bin/sparkpipe_model_[a]pi'); do c=\$(readlink /proc/\$p/cwd 2>/dev/null); \
      [ \"\$c\" = \"\$rr\" ] && kill -TERM \$p; done; exit 0"; done
    # wait for zero glm5 procs (collective teardown can take ~60 s), then 45 s settle

START (RELIABLE FORM — run tools/glm5_next_wave.sh FROM spark0 so the
fan-out is node-to-node; controller-side 16-parallel fan-outs randomly
hang/lose ranks):

    ssh spark0 "cd ~/g5rt2-src && bash tools/glm5_next_wave.sh [--probe] full"
    # --probe arms the G5N-PROBE diag ladder + cross-rank checksums and
    # waits 780s for ready (rank 0's L0 ladder is slow BY DESIGN); the
    # probe build scales the TP connect (x4) and operation (x8) windows.

READY: `grep -c "model_residentd ready" residentd.log` = 1 on all 16.

API (after 16/16 ready): on spark0 only -
`cd ~/sparkdata/glm5_next.tp16 && setsid nohup ./bin/sparkpipe_model_api
--deployment model_resident.json --runtime-root . --port 8433 >
api.log 2>&1 < /dev/null &` then GET /health — CHECK "served":0 for a
canonical first-request reading (recurrence state survives api restarts;
only a residentd wave resets slots).

Known state: cold first request still degenerate; same prompt repeats
differently across waves (weights changed the attractor:
66188-row-sharded → 113235/13765/1617/80346 col-sharded). Cross-rank
reduce checksums bit-identical; retention advances; layout audits clean.
