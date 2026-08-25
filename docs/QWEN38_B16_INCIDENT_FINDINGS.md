# "qwen27B crashes" incident — findings (2026-08-23)

Reproduced, decomposed, and root-caused on spark2 against main `98b4668`
(the release bits). Four distinct issues were mixed together in the
report; only one is a serving-engine bug, and it is NOT in the model path.

## TL;DR

1. **The daemon SEGV is a startup crash from concurrent instances** — the
   fleet watchdog restarts `model_residentd` while the previous process
   still holds the GPU; the new instance dies inside pack loading
   (`SparkQwen38_27bModuleLoadPack → SparkStageModuleLoadDeviceRegion → jump
   to 0x2480`, a corrupted call into libcudart under the instance
   conflict). Backtrace: `/var/crash/core.sparkpipe_model.2684848.*`
   (mine, readable) and two root-owned cores from 19:47/19:49 (the
   debugger agent's instances — same era).
2. **B1/B4 cells work** when run cleanly: their exact `ctx512_b1` cell
   completes 128/128 tokens — 117.2 s as-generated, **20.2 s** once
   `max_prefill_rows_per_submission` is 8 (their files set 1 — the
   un-batched prefill is ~16× slower).
3. The 84-second prefills from (2) blew their gateway's ~90 s deadlines →
   retries → retried/duplicated submissions correctly rejected → the
   "decode-step INVALID_ARGUMENT" family.
4. The early `status=6 SCHEMA_ERROR` bench runs (2 ms failures) were the
   bench reading **half-written JSON** — their cell generator racing the
   bench loop. The same files parse and run cleanly now.

## The daemon crash (the actual SEGV)

Backtrace (gdb on the core, spark2-owned):

```
#0  0x0000000000002480 in ?? ()
#1  SparkStageModuleLoadDeviceRegion () from lib/model_driver.so
#2  SparkQwen38_27bModuleLoadPack ()           from lib/model_driver.so
#3  SparkQwen38_27bResidentDecodeStageInitialize ()
#4  SparkGeneratedDriverCreate ()
#5  SparkQwen38_27bServingLoadDriver ()        (adapter, pack load at init)
#6..#11 SparkQwen38_27bServingInitialize → SparkModelResidentdInitialize → main
```

`LoadDeviceRegion` makes no indirect calls itself — the jump to `0x2480`
is a call into libcudart (`cudaMemcpy`/`cudaMalloc` path) with corrupted
state. The trigger reproduced deterministically: a second daemon
instance starting while another holds the ~78 GiB of device memory. The
fleet-swap watchdog logged **34 restarts**; when a restart's `kill` races
the old process's GPU release, the new instance crashes here and the
watchdog sees "the model crashed".

**Fix direction (watchdog, not engine):** before starting a new instance,
kill the old one AND wait for GPU release (`nvidia-smi
--query-compute-apps=pid` empty) — the runbook §2 kill sequence does
this correctly. Defensive engine hardening (fail cleanly on OOM at pack
load instead of faulting in libcudart) is optional follow-up.

## The B16 client cascade (secondary, real but different)

With a healthy exclusive daemon, the 16-request cell runs but: 8-row
prefill submissions across 16 lanes carry ~1 row per lane → each lane's
rows run as separate single-row frames (each a full weight pass) →
submissions average ~1 s (one 11 s frame observed; phase counters frozen
= time outside the measured kernels). The run eventually ends with
`submitted=146 admitted=144 rejected=2` and ALL requests failed
`status=4 NOT_FOUND` — one bad message trips the client's global fail
path (`FailTransactions`), which fails every idle request. Also observed:
the daemon process dying at the end of such a run = issue (1) triggered
by the watchdog reacting to the slow run.

**Notes for the B>1 work:** multi-lane prefill wants lane-major chunking
(finish lanes' rows into full 8-row frames instead of round-robin
splits); the client's global-fail semantics turn one rejected submission
into 16 request failures; and the daemon is single-client
(`last_submission_id` is global per daemon) — two concurrent client
processes interleave submission ids and the second client's ids look
stale → instant INVALID_ARGUMENT. One benchmark client at a time.

## Reproduction commands (all on spark2)

```bash
# clean B1 run (works): 20.2s with the fixed config
python3 - <<'EOF'
import json
b=json.load(open("/tmp/sp2_cells/ctx512_b1.sparkbatch.json"))
b["max_prefill_rows_per_submission"]=8
b["maximum_new_submissions_per_progress"]=2
json.dump(b,open("/tmp/ctx512_fixed.json","w"))
EOF
cd /home/spark2/sparkdata/qwen38.fp8.tp1 && export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
./bin/sparkpipe_model_batch --deployment config/model_resident.json \
  --runtime-root $PWD --batch /tmp/ctx512_fixed.json

# the startup SEGV: start a second daemon while one is serving
# (core lands in /var/crash/ if the dir is writable by the user)
```

## What is NOT broken

The release path itself: O512 bit-identical (`d7f798801a6e43a6`), prefix
borrow bit-identical, B1 and B4 requests complete, 20.2 s walls with the
correct prefill config. The 24.5 tok/s cold / 26.1 cache-hit / ~28
decode-only numbers stand.
