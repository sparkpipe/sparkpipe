# GLM 5.2 TP8 bring-up plan — band {spark1, spark8, spark9, sparka, sparkb, sparkd, sparke, sparkf}

Date: 2026-08-25 · Author: bring-up recon agent · Status: READY PENDING ONE PACK SHIP
Basis: Aug-16 merged-main recipe (PERFORMANCE_STATUS.md "GLM 5.2 TP8 B1" /
"TP8 B16", fixes #663/#665) — 6.91 tok/s B1, 43.46 tok/s B8-agg, 75.55 tok/s
B16-agg were measured once on the old band (spark8..sparkf incl. sparkc).
This plan re-stages that exact recipe on the NEW band with spark1 replacing
the now-dead sparkc at rank index 4. No daemons were started and no code was
modified during recon; everything below is read-only findings plus the exact
steps for the operator.

## 1. Host readiness matrix (probed 2026-08-25, read-only SSH BatchMode)

| Host | Role | GPU | Disk free (/tmp and ~/sparkdata are one root fs) | TP8 pack present | Residentd | Notes |
|---|---|---|---|---|---|---|
| spark1 | **rank 4** (new member) | GB10 ✓ | 1.5 T | ✗ NONE — needs full runtime tree + rank04 pack | none | `~/kvcache` exists ✓ |
| spark8 | rank 0 (coordinator) | GB10 ✓ | 2.1 T | ✓ rank00 (96 GB) | none | full tree: bin/, lib/, config/, model_resident.json, bench_b1.json |
| spark9 | rank 1 | GB10 ✓ | 2.1 T | ✓ rank01 | none | config staged |
| sparka | rank 2 | GB10 ✓ | 2.0 T | ✓ rank02 (102,835,957,760 B) | none | config staged |
| sparkb | rank 3 | GB10 ✓ | 832 G | ✓ rank03 | none | config staged |
| sparkd | rank 5 | GB10 ✓ | 2.1 T | ✓ rank05 | none | config staged |
| sparke | rank 6 (+ build host) | GB10 ✓ | 747 G | ✓ rank06 | none | holds master pack set `~/srcdata/glm52_tp8_packs/` incl. **rank04** |
| sparkf | rank 7 | GB10 ✓ | 2.4 T | ✓ rank07 | none | config staged |

- **sparkc is NOT in the band** and is currently UNREACHABLE ("Connection
  timed out during banner exchange") — consistent with its removal.
- Reachability probed: spark1↔spark8, spark1↔sparkf, spark8↔spark9 all pass
  ICMP (RTT 0.42–1.6 ms). Full pairwise fabric was gated 2026-08-13
  (PERFORMANCE_STATUS.md, ≥80 Gb/s/direction on all eight pairs).
- `pgrep -af residentd` hits on every host are the probe's own bash command
  line (false positive). No real residentd (`bin/sparkpipe_model_residentd`)
  is running anywhere in the band → clean measurement state.
- Port block (registry + staged configs): control 19480–19487,
  collective 63620–63627, transport-control base 60700.

## 2. Where the 8 rank packs live

Pack identity: `glm52_tp8_rankNN.fp8.glms52sp`, 102,835,957,760 bytes each
(~96 GiB), packed 2026-08-15 ~17:22 from spine zai-org/GLM-5.2 BF16 master
`b4734de4facf877f85769a911abafc5283eab3d9` + FP8 experts (two-source build on
sparke, tools/devcycle/README-glm52.md).

| Rank | Host | In-band location | Master copy (sparke) |
|---|---|---|---|
| 00 | spark8 | `~/sparkdata/glm52.tp8.fp8/packs/` ✓ | `~/srcdata/glm52_tp8_packs/glm52_tp8_rank00.fp8.glms52sp` ✓ |
| 01 | spark9 | ✓ | rank01 ✓ |
| 02 | sparka | ✓ | rank02 ✓ |
| 03 | sparkb | ✓ | rank03 ✓ |
| **04** | **spark1** | **✗ MISSING — ship this** | `~/srcdata/glm52_tp8_packs/glm52_tp8_rank04.fp8.glms52sp` ✓ (verified on sparke 2026-08-25) |
| 05 | sparkd | ✓ | rank05 ✓ |
| 06 | sparke | ✓ (hardlink n=2 → same inode as srcdata copy) | rank06 ✓ |
| 07 | sparkf | ✓ | rank07 ✓ |

**No repacking is required.** The only data movement is one 96 GiB file:
sparke → spark1. At the measured ≥80 Gb/s useful fabric this is roughly
3–4 h wall clock.

Naming caveat: on-disk packs and the *staged* configs both use the transposed
suffix `.glms52sp`; `tools/glm52_gen_deployment.py` emits `.glm52sp`. The
staged configs are authoritative — see §3.

## 3. Deployment spec / config files — USE THE STAGED ONES, DO NOT REGENERATE

Every live host already carries a complete, mutually consistent config set
under `~/sparkdata/glm52.tp8.fp8/`:

- `config/glm52_stage.json` — schema_version **4**, model_revision
  `b4734de4facf877f85769a911abafc5283eab3d9`, expert_weight_codec fp8,
  tp_degree 8, tp_rank = host's index, max_sequence_positions 4096,
  execution_row_capacity 16, speculation_enabled false, draft_count 6;
  `tp_collective`: hidden_transport, collective_identifier
  8811223344556678, listen_port 63620+rank, algorithms
  ["recursive_doubling"], peer_hosts **already the new band**
  `[spark8,spark9,sparka,sparkb,spark1,sparkd,sparke,sparkf]`
  (spot-verified on spark8 and sparkf), rail_peer_hosts identical ×2,
  step_rail_indices [0,0,1].
- `model_resident.json` — schema_version 2, coordinator_rank_index 0,
  adapter lib/model_serving_adapter.so, driver lib/model_driver.so
  (program resident_decode), transport host-rdma control_port_base 60700,
  runtime_limits (max_inflight_submissions 4, max_active_sequences 16,
  max_input_rows 16, resident_sequence_capacity 16, kv_*_page_capacity 0),
  8 nodes: rank i → host_i, runtime_root `/home/<host>/sparkdata/glm52.tp8.fp8`,
  node_target `cuda.sm121.glm52.resident_decode_stage.bf16.expert_fp8`,
  adapter_configuration_path config/glm52_stage.json,
  kv_backing_directory `/home/<host>/kvcache/glm52.tp8`,
  kv_backing_maximum_bytes 8589934592 (8 GiB),
  control_endpoint tcp <host>:19480+i. **Rank 4 node already says spark1.**
- `bench_b1.json` — the retained B1 input (11-token prompt ids, budget 128,
  stop token 154820, request_capacity 16).

Why NOT regenerate with `tools/glm52_gen_deployment.py`: it still hard-codes
HOSTS including sparkc (line 33), writes `.glm52sp` pack paths (mismatch with
on-disk `.glms52sp` names), schema_version 3, and draft_count 7 — three
drifts behind the deployed reality. If regeneration ever becomes necessary,
first update HOSTS/suffix/schema in that tool and re-verify against §3 above.

What spark1 is missing (create before launch):

```
/home/spark1/sparkdata/glm52.tp8.fp8/
├── bin/{sparkpipe_model_residentd,sparkpipe_model_batch}   # copy from any rank (e.g. spark8)
├── lib/{hidden_transport.so,model_driver.so,model_serving_adapter.so}
├── packs/glm52_tp8_rank04.fp8.glms52sp                     # 96 GiB from sparke master set
├── config/glm52_stage.json                                 # clone sibling config, patch 3 fields (below)
├── model_resident.json                                     # byte-identical shared view; copy from spark8
└── (bench_b1.json only needed on coordinator spark8)
```

The rank-4 stage config does not exist anywhere reachable (it lived on
sparkc); derive it by cloning e.g. sparkd's (rank 5) config and patching:

| Field | Value for spark1 |
|---|---|
| `tp_rank` | `4` |
| `stage_pack_path` | `packs/glm52_tp8_rank04.fp8.glms52sp` |
| `listen_port` (tp_collective) | `63624` |

(`peer_hosts`, `peer_ports` 63620–63627, rail lists, collective id, ports
19480+r are already correct in every staged copy.)

Also update bookkeeping afterwards (not code, registry/doc truth):
`tools/devcycle/fleet_registry.json` glm52 entry still lists sparkc in hosts
and notes "rank 7 = sparkf"; it should read the new band with spark1 at rank 4.
`tools/devcycle/README-glm52.md` port table stays valid (slot A unchanged).

## 4. Bring-up command sequence (adapted Aug-16 recipe + devcycle patterns)

Operator-run, in order. All ssh uses `-o BatchMode=yes`.

### Step 0 — pre-flight (clean state)

```sh
BAND="spark8 spark9 sparka sparkb spark1 sparkd sparke sparkf"
for h in $BAND; do
  ssh -o BatchMode=yes "$h" '
    nvidia-smi -L || exit 1                       # GB10 present
    pgrep -af "bin/sparkpipe_model_residentd" && exit 1   # nothing holding ports
    ss -ltnH | grep -E ":(1948[0-7]|6362[0-7]|607[0-9][0-9]) " && exit 1  # ports free
    df --output=avail -BG "$HOME/sparkdata" | tail -1'
done
# expect: GPU line, no pgrep hit, no listening ports, ≥110G avail everywhere
```

### Step 1 — seed spark1 runtime tree (bins/libs/configs, small)

```sh
ssh -o BatchMode=yes spark1 'mkdir -p ~/sparkdata/glm52.tp8.fp8/{bin,lib,config,packs} ~/kvcache/glm52.tp8'
for d in bin lib; do
  ssh -o BatchMode=yes spark8 "tar -C ~/sparkdata/glm52.tp8.fp8 -cf - $d" \
    | ssh -o BatchMode=yes spark1 "tar -C ~/sparkdata/glm52.tp8.fp8 -xf -"
done
ssh -o BatchMode=yes spark8 'cat ~/sparkdata/glm52.tp8.fp8/model_resident.json' \
  | ssh -o BatchMode=yes spark1 'cat > ~/sparkdata/glm52.tp8.fp8/model_resident.json'
ssh -o BatchMode=yes sparkd 'cat ~/sparkdata/glm52.tp8.fp8/config/glm52_stage.json' \
  | ssh -o BatchMode=yes spark1 \
    "jq '.tp_rank=4 | .stage_pack_path=\"packs/glm52_tp8_rank04.fp8.glms52sp\" | .tp_collective.listen_port=63624' \
       > ~/sparkdata/glm52.tp8.fp8/config/glm52_stage.json"
ssh -o BatchMode=yes spark1 'jq .tp_rank,.stage_pack_path,.tp_collective.listen_port ~/sparkdata/glm52.tp8.fp8/config/glm52_stage.json'
# expect: 4, "packs/glm52_tp8_rank04.fp8.glms52sp", 63624
```

### Step 2 — ship the rank04 pack (sparke → spark1, ~96 GiB)

```sh
ssh -o BatchMode=yes sparke \
  'sha256sum srcdata/glm52_tp8_packs/glm52_tp8_rank04.fp8.glms52sp'          # record SRC sha
time ssh -o BatchMode=yes sparke \
  "dd if=$HOME/srcdata/glm52_tp8_packs/glm52_tp8_rank04.fp8.glms52sp bs=64M" \
  | ssh -o BatchMode=yes spark1 \
    "dd of=$HOME/sparkdata/glm52.tp8.fp8/packs/glm52_tp8_rank04.fp8.glms52sp bs=64M"
ssh -o BatchMode=yes spark1 \
  'sha256sum ~/sparkdata/glm52.tp8.fp8/packs/glm52_tp8_rank04.fp8.glms52sp'  # must match SRC
```

(Alternative if agent-forwarded rsync is available on sparke:
`ssh sparke 'rsync -a --info=progress2 srcdata/glm52_tp8_packs/glm52_tp8_rank04.fp8.glms52sp spark1:sparkdata/glm52.tp8.fp8/packs/'`.)

### Step 3 — cross-check fleet consistency

```sh
for h in $BAND; do
  ssh -o BatchMode=yes "$h" \
    'grep -o "\"spark[0-9a-f]*\"" ~/sparkdata/glm52.tp8.fp8/config/glm52_stage.json | sort -u | tr "\n" " "; echo' 
done
# expect the same 8-host peer list on every rank, spark1 present, no sparkc
```

### Step 4 — launch residentd ranks (devcycle pattern, cf. bbench.sh:56-98)

```sh
i=0
for h in spark8 spark9 sparka sparkb spark1 sparkd sparke sparkf; do
  ssh -o BatchMode=yes "$h" "
    cd ~/sparkdata/glm52.tp8.fp8 &&
    export LD_LIBRARY_PATH=\$PWD/lib:\$LD_LIBRARY_PATH \
           SPARKPIPE_RELEASE_GENERATION=<gen> SPARKPIPE_RELEASE_GIT_COMMIT=<commit> \
           SPARKPIPE_RELEASE_ID='glm52-tp8-bringup'
    setsid -f bin/sparkpipe_model_residentd \
        --deployment config/model_resident.json --rank-index $i \
        >/tmp/glm52-tp8-rank$i.log 2>&1 </dev/null"
  i=$((i+1)); sleep 6
done
# readiness: all eight listening on 19480..19487
for a in $(seq 1 60); do r=0
  for h in $BAND; do ssh -o BatchMode=yes "$h" 'ss -ltnH | grep -q ":1948"' && r=$((r+1)); done
  [ "$r" = 8 ] && break; sleep 3
done; echo "ready=$r/8"
```

### Step 5 — B1 smoke (coordinator = spark8, rank 0)

```sh
ssh -o BatchMode=yes spark8 "
  cd ~/sparkdata/glm52.tp8.fp8 &&
  bin/sparkpipe_model_batch \
    --deployment \$PWD/model_resident.json \
    --runtime-root \$PWD \
    --batch \$PWD/bench_b1.json"
```

Gate: single token stream completes; compare against the retained Aug-16
receipts (`qualification/glm52/performance/tp8_b8_20260816/glm52-toks.txt`,
sha bc9edad1…) for token parity. Then optionally repeat the B8/B16 batches
from the same receipts directory for aggregate numbers. Per README-glm52
measurement rules the B1 window is exclusive on the band — no other residentd
traffic during timed runs.

### Step 6 — record receipts

Freeze stdout/stderr + token stream under
`qualification/glm52/performance/tp8_<workload>_<date>/` with SHA-256s, and
update PERFORMANCE_STATUS.md, mirroring the Aug-16 receipt discipline.

## 5. Known blockers and risks

1. **HARD BLOCKER (resolved by Step 2): rank04 pack not in-band.** It lived
   only on sparkc, which is unreachable. The verified master copy on sparke
   makes this a shipping problem, not a repacking problem. Do not start
   ranks before the sha256 gate passes — a short/corrupt rank04 pack will
   fail mid-collective and can wedge all eight processes.
2. **DFlash2 speculation is out of scope for THIS bring-up (by design).**
   Draft weights are untrained — "the base checkpoint ships none"
   (PERFORMANCE_STATUS.md GLM 5.2 section; SURVEY_glm52.md idea #4: DSpark
   backend landed but inert, expected 2× B1 once trained drafts exist).
   Additionally modules/glm52_resident_decode_stage/README.md: speculation
   arms ONLY on single-rank builds (TP degree 1); a fanout deployment is
   refused loudly absent a draft transport. Groundwork implication: the
   future DFlash2 path is either the tp1 shape (tools/glm52_gen_deployment.py
   --pipeline tp1, which needs a trained drafter pack
   `packs/glm52_dspark_drafter`) or a new draft-transport design for fanout.
3. **Generator drift** (§3): repo tool would emit wrong hosts/pack-suffix/
   schema/draft-count. Use staged configs; fix the tool separately.
4. **Performance context, not blockers:** B1 6.91 tok/s sits 2.3–4× below
   matched-precision community SOTA (~16–27 tok/s FP8-class on GB10);
   levers per SURVEY_glm52.md: MTP layer-78 self-spec, collective fusion
   (158 reduces/token ≈ 31 ms), NVFP4 KV. Single-stream floor is bandwidth:
   ~21.7 GB/token over 273 GB/s LPDDR5x ≈ 80 ms.
5. **Disk headroom watch:** sparkb (832 G) and sparke (747 G) are the
   tightest; each rank pack is 96 GiB and KV backing adds up to 8 GiB/node.
   Fine today; re-check before adding any second pack generation.
6. **Registry staleness:** fleet_registry.json still names sparkc; update as
   part of Step 6 so fleet_status/fleet_swap tooling sees the true band.
7. **sparkc itself:** treat as decommissioned until someone investigates the
   SSH banner timeout; do not schedule around it.

## 6. Rollback / cleanup procedure

Stop (any time, safe even mid-run):

```sh
for h in $BAND; do
  ssh -o BatchMode=yes "$h" "pkill -9 -f '^bin/sparkpipe_model_residentd'; sleep 1" || true
done
# verify quiet: no listener, no process
for h in $BAND; do
  ssh -o BatchMode=yes "$h" 'pgrep -af "bin/sparkpipe_model_residentd" || ss -ltnH | grep -E ":(1948[0-7]|6362[0-7]) " || echo clean'
done
```

Remove transient artifacts:

```sh
for h in $BAND; do
  ssh -o BatchMode=yes "$h" 'rm -rf ~/kvcache/glm52.tp8/* /tmp/glm52-tp8-rank*.log'
done
```

Full de-stage of spark1 (only if reverting the whole expansion; the other
seven hosts keep their pre-existing trees untouched):

```sh
ssh -o BatchMode=yes spark1 'rm -rf ~/sparkdata/glm52.tp8.fp8'
# NOTE: frees ~96 GiB but re-triggers Step 1+2 on next bring-up.
# Safer default: keep the tree in place (idle trees cost nothing but disk).
```

Restore prior state guarantees: no always-on slot (qwen27b spark0-3, dsv4
flash spark4-7) is touched by anything above; the GLM slot-A port block
(19480/63620/60700) is exclusive to this band; sparkc was already absent, so
rollback returns the fleet exactly to the pre-bring-up posture observed
2026-08-25.
