# K3 TP16 — pack production + end-to-end run plan

Status: FINALIZED for the Tuesday window (2026-08-23, unified `229526d`).
Every step is scripted or gated; what remains is hardware time plus the two
node/GPU gates in §T1. This closes PR #667's last two open items ("TP16 pack
production", "end-to-end run") under the fleet rule that a measured window
is only requested once everything before it is provably done. Gate-by-gate
status at finalization: see "Gate status" below; execution order for the
day: see "Tuesday run sheet".

Classification discipline (same as `docs/K3_PERF.md`): every number below is
**measured** or **analytical**
(`tools/k3_tp4pp4_perf_estimate.py`, whose projections come in two
bandwidth models — a calibrated one anchored to measured points: K3's 55.5 ms
single-spark B1 stage step, the sparkb stream probe 250.7 GB/s, and the
qwen36 band's achieved 162.7 GB/s; and the historical 177.45 GB/s convention
roofline). Only Phase 5's receipts make a fleet row; `PERFORMANCE_STATUS.md`
stays "RETRACTED (not hardware-measured)" until then.

---

## Gate status at finalization (2026-08-23, unified `229526d`)

| gate | status | evidence |
| --- | --- | --- |
| production pipeline dry run (synthetic TP16-closed ckpt → pack → shard ×16 → `SHA256SUMS.tp16`) | **DONE, measured** | `/tmp/k3_dryrun_r3`: 66 tensors @ tile_k 32 (28,311,264 payload B), 16/16 rank packs independently re-verified (`shasum -a 256 -c`), zero guard warnings; disk-guard negative control fired as designed; `tests/test_k3_pack.py` + `tests/test_k3_shard.py` exit 0 at `229526d` (`.agents/coord/logs/k3.log` 2026-08-23T05:05Z) |
| deploy transfer+verify logic (`k3_deploy_tp16.sh`) | **DONE, measured** | stub-transport control run over the dry-run rank packs: happy path deploys spark0..sparkf (hex 0-9,a-f) and prints `16/16 rank packs verified`, exit 0; corrupted-destination negative control aborts with `SHA MISMATCH ... (want ... got ...)` exit 1. Only the real network hop remains untested (first contact on the node) |
| 0.1 bd34381 classification port | **DONE on unified** | `tools/k3_shard.py` slices the three axis-riding 1-D tensors per rank (`HEAD_1D`: `kda_decay_bias`, `kda_head_log_scale`; `LATENT_1D`: `routed_norm_weight`) and reprices rank shapes; reassembly coverage lives in `tests/test_k3_shard.py` (rank shards concatenate to the pack; replicated controls stay equal). The side branch's content is superseded — do NOT merge `origin/k3-tp4-layer0`; it stays historical |
| 0.2 offline equivalence gate on the fixed sharder | PENDING — needs node checkpoint | §T1 below |
| 0.3 capture fidelity re-proof | PENDING — needs sparka GPU | §T1 below |
| 0.4 PR #667 body update | PENDING | after 0.2/0.3 receipts land |

Open-question disposition:

1. **+43% payload receipt-vs-inventory gap** — NOT settled from this
   workstation (no real stage-pack manifest locally). Decode conclusions are
   unaffected: they anchor to the measured single-spark A1 step (55.5 ms),
   not the byte model. Large-B prefill numbers stay quoted as bands until
   §T0's two-minute reconciliation reads an existing stage-pack manifest's
   tensor-bytes total on the node.
2. **bd34381 fold-in vs separate land** — RESOLVED: fold-in; the content
   already landed in `unified`'s sharder/tests (gate table above).
3. **deploy rsync path untested** — DOWNGRADED: transfer/checksum/abort
   logic now control-tested end to end (row above); only the physical
   network leg awaits first contact.

## Phase 0 — pre-hardware gates (workstation / sparka, no ring needed)

These block the run, not the packing. Do them while boxes are away.

0.1 **~~Port the `bd34381` tensor-classification fix onto `unified`.~~ DONE**
(see gate table above; closed on `unified`, reassembly-covered by
`tests/test_k3_shard.py`). Historical context: the side branch
`origin/k3-tp4-layer0` sliced `kda_decay_bias`, `kda_head_log_scale`,
`routed_norm_weight` per rank while the then-current `tools/k3_shard.py`
classified them REPLICATED against rank-locally-indexing kernels — ranks
1..3 consumed rank 0's segments on all 69 KDA layers and every MoE layer.
The per-rank-slice-at-shard-time mechanism is the one that landed.

0.2 **Re-run the offline equivalence gate on the fixed sharder.**
`tools/k3_tp4_equivalence_check.py`, full vs 4-rank packs of the layers-0-3
slice (~65 GB, registers under the 48 GB chunking from `1febd9c`). Receipt =
max relative deviation line. This was failing *because of* the GEMM bugs
(`5385a63`, `1f8b190`) and the classification defect; it must PASS on the
path that produces the production packs.

0.3 **Re-prove capture fidelity** (gate-rework item): no-capture direct
step-2 vs graph replay, fresh-run determinism. Was red under the old bugs;
must be green on the fixed path before any fleet window is spent on it.

0.4 **Update PR #667's body**: items 1a/1b root-caused and closed on
`unified`; cite the receipts from 0.2/0.3. Close the PR when Phases 1-5
land their receipts. Bisect artifacts remain at `spark1:/tmp/k3h4.tar`.

## Phase 1 — TP16 pack production (one node, no ring)

Prereq: one node holding all 96 checkpoint shards (~1.6 TB,
`moonshotai/Kimi-K3-MXFP4`; the PP13-era fetch left shards on every node,
so this can be any of them). Disk: ≥ 850 GB free (script enforces).

```sh
nohup bash tools/k3_tp16_pack_production.sh \
    /home/<node>/srcdata/kimi_k3.mxfp4.pp13 /home/<node>/k3tp16prod \
    > /home/<node>/k3tp16prod/run.log 2>&1 &
```

What it does: full-model pack at `expert_tile_k 32` → 16-way shard via
`tools/k3_shard.py` → per-rank size sanity (1/16 ±10%: the guard for the
bd34381 dropped/duplicated-tensor class) → `SHA256SUMS.tp16`.

- Why tile_k 32 everywhere: at TP16 the w1 diagonal is 224 k-tiles x 384
  cells and w2 is 192 — neither divides by 128 or 64; 32 divides both
  (`docs/K3_TP16_REPACK.md`). The manifest carries tile_k per tensor, so
  the existing tile_k-128 TP4xPP4 packs stay valid alongside.
- ETA: multi-hour (Python expert interleave dominates, same profile as the
  ~350 GB stage packs). One full pack, not four stages: TP16 is PP1, every
  rank carries every layer, so per-stage slicing would need cross-stage
  concatenation the pack format does not do.
- Receipt: `run.log` tail + `SHA256SUMS.tp16`.

## Phase 2 — deploy to the 16 ranks (still no ring window consumed)

```sh
bash tools/k3_deploy_tp16.sh /home/<node>/k3tp16prod     # rsync --append-verify + sha256 per rank
tools/k3_gen_adapter_configs.sh OUT_DIR 16               # device-direct NCCL, degree 16
tools/k3_gen_deployment.sh model_resident.json 16        # stage_index 0 x16, tp16 runtime roots
```

Lands each verified rank pack at
`spark<i>:/home/spark<i>/sparkdata/k3.mxfp4.tp16/packs/` and emits the
config/deployment set. The serving adapter already fail-closes TP16 without
a `device_collective` block (`spark_k3_serving_adapter.c`: degree >
SPARK_TP_COLLECTIVE_MAX_STEPS must carry the device tier), and the pack
loader reads the manifest's per-tensor `interleave.tile_k`
(`SparkK3PackLoadInterleaveTileK`) — no driver rebuild expected. If the
compile gate is touched at all: `tools/k3_sm121a_compile_gate.sh` on sparka.

## Phase 3 — the window request

K3 sits behind DSV4 Pro and Qwen 3.8 Max in `COORDINATION.md`'s queue. When
the slot opens:

1. `tools/devcycle/fleet_status.sh` probe output saved WITH the receipt
   (fleet-rule requirement).
2. `tools/fleet_swap.sh k3` — evicts everything, starts K3 on all 16.
   Exclusive whole-fleet window; promotion < 60 s.

Note for the coordinator: `COORDINATION.md`'s K3 line ("chunked
registration is still being fixed") is stale — `1febd9c` landed the 48 GB
chunked registration that lets the full ~400 GB pack register.

## Phase 4 — end-to-end run (the actual #667 item)

Ordered so each step can abort cheaply:

1. **Boot + register.** residentd up on 16 ranks, pack registers chunked,
   bind table clean. Abort signal: any rank missing tensors (Phase 1 guard
   should have caught it offline).
2. **Single-token smoke, no capture.** One token through the live ring;
   compare the step-2 hidden against the sparka direct-step golden within
   the gate tolerance (fresh-run determinism held 4 ULP after the o_proj
   fix).
3. **Graph capture + replay determinism.** Bit-identical replay (the B1
   anchor property), capture fidelity green.
4. **B1 decode measurement, ≥ 10 min warm.** Analytical expectation
   (TP16, PP1, no pipeline bubbles): **17.8 tok/s (56.1 ms/token)** under
   the calibrated bandwidth model — the one anchored to measured points
   (`tools/k3_tp4pp4_perf_estimate.py`: K3's own 55.5 ms stage step, the
   sparkb stream probe 250.7 GB/s, and the qwen36 band's achieved
   162.7 GB/s, mutually within 4.5%) — or 20.2 tok/s under the older
   177.45 GB/s convention roofline. The single-spark anchor measured
   55.5 ms (18.0 tok/s). Accept anything in 17-21; investigate below 14.
5. **Prefill rungs B = 8, 32, 128, 1024.** Analytical steady state,
   calibrated -> convention: **81 / 315 / 1346** -> 92 / 359 / 1537 tok/s;
   TP16 prefill is parity with PP4 minus fill (single-prompt latency 0.77 s
   at B1024 calibrated). The expert stream saturates at B=56; beyond it the
   fp32 KDA state term dominates (52% of B1024 bytes) — record the curve
   even where it disappoints, it prices the BF16-state lever.
6. **Receipts:** retained under `qualification/` + a new
   `PERFORMANCE_STATUS.md`/dashboard row. THIS row retires the retraction:
   first hardware-measured K3 numbers. Record accepted/credited separately
   if speculation is probed; lossless-before-speed holds here too.

## Phase 5 — close #667

With Phases 0-4 receipts attached, update the body one final time and close
the PR. Residual known debt rides separately (dead decay|gate-fusion tail,
`test_k3_pack_layout.py` silently-green fixture, `k3_pack.py` off
`spark_pack_common`) — none blocks closure; they are queued quality-law
items.

## Tuesday run sheet (execution order for the day)

Times relative to window start W. Nothing consumes ring time before the
Phase 3 swap; everything up to it is offline and cheap to abort.

- **T0 — node, any free box, W−1h:**
  1. `git -C <checkout> rev-parse --short HEAD` → record in the receipt
     (`229526d` or later).
  2. **+43% reconciliation (two minutes, non-blocking):** read any existing
     stage-pack manifest's tensor-bytes total (or diff a fresh 4-layer slice
     pack's payload bytes against its manifest sum). Record
     receipt-vs-inventory ratio in the run log; quote prefill as bands
     either way.
  3. Start Phase 1 pack production (nohup command above). Multi-hour; this
     is the long pole — start it first.
- **T1 — sparka, parallel with the pack:** run gates 0.2 and 0.3 on the
  fixed path:
  `tools/k3_tp4_slice.sh <layers-0-3 stage pack> <prefix>` then
  `tools/k3_tp4_equivalence_check.py FULL R0 R1 R2 R3` (receipt = max
  relative deviation line), plus the capture-fidelity legs (no-capture
  direct step-2 vs graph replay; fresh-run determinism). **RED on either =
  STOP**: do not request the fleet window; debug offline. A red gate burns
  zero ring time by construction.
- **T2 — when the pack finishes:** Phase 2 deploy.
  `tools/k3_deploy_tp16.sh /home/<node>/k3tp16prod` prints one verified
  line per rank and finishes `16/16 rank packs verified`; a `SHA MISMATCH`
  aborts nonzero and a rerun resumes via `rsync --append-verify`. Then
  `tools/k3_gen_adapter_configs.sh OUT_DIR 16` +
  `tools/k3_gen_deployment.sh model_resident.json 16`.
- **W — Phase 3 window request:** save the `fleet_status.sh` probe output
  WITH the receipt, then `tools/fleet_swap.sh k3`.
- **In-window (Phase 4 order):** boot+register → single-token smoke vs the
  sparka golden → graph-capture replay bit-identical → B1 decode ≥10 min
  warm (accept 17–21 tok/s calibrated-band; investigate below 14) → prefill
  B = 8/32/128/1024 (bands per the +43% caveat) → receipts under
  `qualification/`.

## Rollback / abort posture

`fleet_swap.sh` restores the previous holder (DSV4 Flash + Qwen 27B
residentds) in under 60 s at any point; rank packs are inert files and the
TP4xPP4 deployment set is untouched by Phase 2 (separate runtime roots,
separate config dirs), so a failed window costs only the hour.
