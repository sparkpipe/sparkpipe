# Tool reference

This file describes command-line tools under `tools/`, the host turn bench
under `tests/`, and the libraries they link. For each tool it gives the purpose, the arguments as the tool's
`main()` parses them, and the meaning of each exit code. Behaviour notes
describe what the code does, including known gaps.

## Other tools

### roofline_meter.py (tools/)

Says where a lane's frame time goes and how far its decode is from the memory
roofline, from lines the lane already writes. It runs nothing on the lane.

```sh
tools/roofline_meter.py --api-log API.log [--rank-log RANK.log ...]
    [--node-gb-per-token GB] [--read-gb-s 243] [--rounds-per-step N] [--label TEXT] [--json]
```

- From the API log it reads every `batch_step` line (the engine prints one for
  each frame that carries one sequence): per kind (`prefill`, `single`,
  `chain`, `verify`) the frames, tokens, tokens per second, rows, service
  time, host time (wall minus service), the gap since the request's previous
  frame and the draft wait.
- From each rank log it reads `STAGE-STEP-TIMING` (the step and its time in
  collectives: peer wait, source wait, copy and combine), `STAGE-CHAIN-HOST`
  and `STAGE-WAVE-HOST` (the rank's host time around a chain and a prefill
  wave), `WD-SPINE-PRIVATE` (the rank's weight bytes) and `STAGE-STEP-ARRIVAL`
  (when each peer's data lands after the rank starts to wait).
- With more than one rank log it lists every rank's weight bytes, own work a
  step (the step without its time in collectives) and peer wait, and names
  the rank with the most own work: the collectives hold every rank to that
  rank's pace, so the others' extra wait is its extra work. Give it every
  rank's log to see an uneven split.
- It states, per round with `--rounds-per-step` and per step without, when
  the first, the median and the on-average slowest peer land and when all
  have, as a range over the rank logs.
- `--rounds-per-step` is the number of collective rounds one decode step
  issues, from the model's schedule (two all-reduces a layer for most
  families); with it the report states the peer wait a round. The timing
  line's own round counts include prefill waves, so the meter does not use
  them.
- `--node-gb-per-token` is what the heaviest node reads for one decode pass:
  a dense rank's weights without the embedding table (the `WD-SPINE-PRIVATE
  bytes=` line of its log is an upper bound), or the spine plus the active
  experts.
  The floor is that over `--read-gb-s`, which defaults to the measured read
  rate in `tools/spec_offline/models.json`.

For `single`, `chain` and `verify` it splits a pass into the memory floor,
collectives (the share of its step that the rank with the most own work
spends in collectives, applied to the frame's service time), the rank's
remainder, host and wire, and the gap, and
states the pass's share of the one-token roofline. When the rank's own work
is shorter than its floor, its waits prefetched part of the weights, and the
remainder reads `less N of the floor read during the waits`. A chain counts each step as
a pass; a verify frame is one pass that yields its kept tokens, so its share
can exceed 100%. A frame whose gap is over half its service time waited behind
other work and is left out of the split and counted. The rank windows mix
every kind of step, so read the split from a log of one kind of traffic when
the kinds differ much.

Exit codes: 0; 1 when the API log has no `batch_step` line; 2 on a usage
error. `tests/test_roofline_meter.py` covers it.

### arrival_probe.py (tools/)

Measures how long a short request waits beside a long stream, against an API
on this host. It is the probe behind the drafter split's numbers
(`qualification/fleet-gate-20261111/`).

```sh
tools/arrival_probe.py PORT TAG ROUNDS TOKENS
```

It runs one greedy stream of `TOKENS` tokens twice (with
`draft_replay_server.py` behind the API, the first run teaches the drafter
the stream and the second is drafted), then times a one-token request with a
prompt of its own thirty times alone, and then sends such requests back to
back while the stream runs, for `ROUNDS` streams. The arrivals turn
speculation off for themselves, so they never ask the drafter.

Lines, each starting `probe TAG`: `stream_learn` and `stream_drafted` with the
drafted run's tok/s; `arrival_alone` and `arrival_during_stream` with n,
median, mean, 90th percentile and maximum in ms; `stream_with_arrivals` with
each stream's tokens, seconds and tok/s.

### draft_replay_server.py (tools/)

A DFT3 draft server with no drafter model and no GPU, for gating the batch
engine's model drafts on a lane that has no drafter. Every draft request
carries the request's committed tokens. The server keeps, per sampling rule
(seed, inverse temperature, nucleus, top-k), the longest token stream it has
been shown. A request whose committed tokens begin a kept stream gets the
stream's next tokens as one draft chain, at most the request's depth and node
capacity; any other request gets the anchor alone, which the engine declines
without backing the request off. A deterministic stream therefore drafts
perfectly the second time it runs.

```sh
tools/draft_replay_server.py --port PORT [--host 127.0.0.1] [--delay-ms MS] [--confidence 0.9]
    [--wrong-every N] [--silent-every N] [--streams 256] [--report-every 64]
```

Point the API's `draft_server` block at the port (`depth` and `wait_ms` as the
lane's drafter would have them), run the same requests twice and compare both
runs with a run that has no draft server: the tokens must be equal, and the
second run's `batch_speculation` line shows model frames and kept drafts.

- `--delay-ms` holds every reply, to stand in for the drafter's latency.
- `--wrong-every N` flips the low bit of every Nth draft token, so frames
  reject drafts.
- `--silent-every N` leaves every Nth request unanswered until the client
  closes the connection, so the engine's time limit (the lane's `wait_ms`
  plus 100 ms) is reached.
- `--port 0` takes a free port; the first output line,
  `DRAFT-REPLAY-READY host:port ...`, names it.

It serves one connection at a time, as the drafters do, and prints
`DRAFT-REPLAY requests= drafted= draft_tokens= unknown= bad= silent= streams=`
every `--report-every` requests and when a connection closes. A request with
another magic closes its connection; one whose anchor is not its last
committed token, or that names no source, is answered with status 1.
`tests/test_draft_replay_server.py` covers it.

### host_turn_bench.c (tests/)

Measures what the host spends per frame on the serving path, with no GPU. The
batch engine, the pipeline client and the sixteen resident clients run on the
bench's thread and drive sixteen `sparkpipe_model_residentd` ranks over
loopback TCP. Every rank loads `tests/fixtures/host_turn_bench_adapter.c`, a
parallel-fanout adapter that completes each submission at once, or after
`FRAME_US` microseconds of sleep, and returns on the last rank the tokens the
frame asks for: one per lane for a prefill, and a decode chain of up to 8 per
lane for a decode frame, so a decode frame is not a token. The engine is
configured as `node/model_api.c` configures it (128 requests, 8 messages per
rank per progress unless MESSAGES_PER_PROGRESS says otherwise, at most 4 new
submissions per progress, the default inflight budget). The deployment allows 4 submissions in flight, 16 active
sequences, 1,024 input rows and a 131,072-token context.

```sh
make build/host_turn_bench
build/host_turn_bench decode LANES PROMPT_TOKENS FRAMES FRAME_US [MESSAGES_PER_PROGRESS]
build/host_turn_bench prefill REQUESTS PROMPT_TOKENS ROUNDS FRAME_US [MESSAGES_PER_PROGRESS]
```

Run it from the repository root. It writes the deployment, copies of the two
modules and one log per rank into a new `build/host-turn-bench-XXXXXX` and
removes that directory when the run succeeds. While the ranks start, the
engine retries its connection every 20 ms for up to 300 attempts, and each
refused attempt logs `client_connect_fail` on stderr.

- `decode`: LANES requests (1 to 16), each with its own prompt and an output
  budget that reaches the context. Once every lane has 64 tokens and 64 more
  frames have completed, it measures the next FRAMES frames: the decode
  continuations and the cache publications between them. A lane that finishes
  inside the window fails the run.
- `prefill`: ROUNDS rounds of REQUESTS requests (1 to 128), each with its own
  prompt and one output token, measured from the first submission to the last
  completion: the prefill waves and the release frames. A round's next
  submissions go out as soon as its last request completes.
- PROMPT_TOKENS is 1 to 65,536, FRAME_US 0 to 1,000,000 and
  MESSAGES_PER_PROGRESS 1 to 64; 1 is the limit
  `SparkModelPipelineClientRecover` reads with, which leaves messages
  buffered between progress calls.

Output: one line per run,
`<scenario> frame_us= frames= prefill= decode= release= publish= continued= tokens= coordinator_us_per_frame= rank_us_per_frame_mean= rank_us_per_frame_max= wall_us_per_frame=`.
The coordinator figure is the CPU time of the bench's thread per completed
frame; the rank figures are each residentd process's CPU time (all of its
threads) per frame, as the mean over the ranks and for the busiest rank; wall
is the elapsed time per frame. One line per frame kind follows,
`kind= frames= coordinator_us_per_frame= wall_us_per_frame= bubble_us_per_frame=`.
It charges the time between two frames finishing on all sixteen ranks to the
kind of the frame that finished, and the bubble subtracts the slowest rank's
reported service time. The kind lines are exact only while one frame is in
flight; when frames overlap, read the totals.

These are host numbers. The ranks share the machine with the coordinator and
talk over loopback, so their syscalls, copies and wakeups are real and the
network hop is not. On macOS, timer slack and wakeups inflate the `FRAME_US`
runs. To compare two revisions, build the bench in each tree, interleave their
runs and take the median of at least three.

Exit codes: 0 on success; 2 on a usage error, a setup failure (work
directory, deployment, ranks, connection) or a request that fails or is
cancelled, which keeps the work directory and prints its path. A run still
going after 300 seconds is killed by `SIGALRM`, together with its ranks.

### hy4_wo_fleet_apply.c

Writes a repair payload into byte regions of one deployed hy4 rank GGUF, then
refreshes that pack's digest records. The regions are meant to be the
`blk.N.attn_output.weight` tensors; the tool does not check tensor names.

```sh
hy4_wo_fleet_apply <rank.gguf> <blob.bin> <offsets.txt>
```

- `offsets.txt`: whitespace-separated decimal pairs `<file offset> <length>`,
  read until the input stops parsing as integer pairs.
- `blob.bin`: the new bytes of every region, concatenated in `offsets.txt`
  order. Its size must equal the sum of the lengths.
- A region whose current bytes already equal its slice of the blob is
  skipped. Every other region is overwritten in place.
- The tool then computes SHA-256 over the whole GGUF, writes
  `<rank.gguf>.sha256` as `<hex>  <basename>`, and overwrites the 64 hex
  characters after the first `"gguf_sha256": "` in `manifest.json` in the
  GGUF's directory. The key must match that spelling, with one space after
  the colon.
- The GGUF is fsynced. The sidecar and manifest are rewritten in place without
  fsync.

Output: `applied N bytes, skipped M bytes, sha256 <hex>`.

Exit codes: 0 on success; 2 on usage error; 1 on any other failure: no
regions, an offsets file, blob, GGUF or manifest that cannot be opened or
read, a blob size that does not match, a failed region read or write, a
digest pass that reads fewer bytes than the file size, a sidecar or manifest
that cannot be opened for writing, or a manifest key that is missing or
followed by fewer than 64 characters.

### mesh_register_attach_repro.c

Reproduces the resident's host registration of the weightd mesh region. It
connects to a running weightd, lazily attaches a pack, then calls
`cudaHostRegister` on the mesh send buffer that the attach mapped.

```sh
make build/mesh_register_attach_repro
build/mesh_register_attach_repro SOCKET PACK SHA256 REVISION TOPOLOGY [EXPERT_POOL_CAP_BYTES]
```

The make target links the real CUDA runtime only when
`$(CUDA_HOME)/include/cuda_runtime.h` exists; otherwise it links
`tests/cuda_stub/cuda_runtime_stub.c` and the registrations never reach CUDA.

Arguments:

- `SOCKET`: weightd socket path.
- `PACK`: a regular, non-empty file. `<PACK>.experts` must load with
  `SparkWeightdManifestLoad`.
- `SHA256`: 64 characters, the pack identity digest.
- `REVISION`: the identity revision.
- `TOPOLOGY`: 1 to 65535.
- `EXPERT_POOL_CAP_BYTES`: optional, 1 to 2^40-1, the explicit expert pool
  cap. The request's `expert_pool_bytes` is `SparkWeightdManifestPoolPlan` of
  the loaded manifest and that cap (see Pool sizing in
  [WEIGHTD_DESIGN.md](WEIGHTD_DESIGN.md#pool-sizing)); a cap the plan refuses
  exits 2.

The identity model is fixed to `glm5_next_stage`. `arena_bytes` is the pack
file size and `abi_version` is `SPARK_WEIGHTD_IPC_ABI_VERSION`.

Sequence:

1. `SparkWeightdClientConnect`, then `SparkWeightdClientAttachLazy` with a
   120 s timeout. `tools/weightd_warm.c` makes the same call. The client maps
   the daemon's mesh memfd at a 64 KiB-aligned fixed address
   (`SparkWeightdMapSharedFd`).
2. The tool requires a non-null `mesh_send_buffer_addr` and
   `mesh_send_buffer_bytes == SPARK_WEIGHTD_MESH_REGION_BYTES`, then creates
   the CUDA context.
3. It calls `cudaHostRegister` on the whole region three times: with
   `cudaHostRegisterPortable | cudaHostRegisterMapped` (the flags
   `SparkTpDeviceCollectivePrepareReceiveBf16` uses), with
   `cudaHostRegisterPortable`, and with no flags. After each success it prints
   the `cudaHostGetDevicePointer` result and unregisters.

Exit codes: 0 when all three registrations succeed; 1 on attach failure, an
unusable mesh region, or any failed registration; 2 on usage error, invalid
pack or identity, or manifest load failure.

How the transport handles the same call: `SparkTpDeviceCollectivePrepareReceiveBf16`
in `ring/transport/tp_device_collective.c` registers the region with
`cudaHostRegister(..., SPARK_WEIGHTD_MESH_REGION_BYTES, PORTABLE | MAPPED)`,
unless another collective in the process already prepared the same buffer.
If the call returns `cudaErrorInvalidValue` and the device reports pageable
memory access through host page tables, it logs `MESH-REGISTER-SKIP` and
continues unregistered. Otherwise a failure logs `MESH-REGISTER-FAIL` and
returns `SPARK_STATUS_IO_ERROR`. `TECHDEBT.md` (Mesh collectives) tracks the
open question of why CUDA refuses to host-register a shared weightd's
RDMA-registered pages.

### mesh_register_repro.c

Control probe without weightd. It creates a memfd (`memfd_create`, then
`ftruncate` to `REGION_BYTES`) and runs six attempts. Each attempt maps the
memfd `MAP_SHARED | MAP_FIXED` at a 64 KiB- or 2 MiB-aligned address, writes
the first 4096 bytes, and calls `cudaHostRegister` with
`cudaHostRegisterPortable | cudaHostRegisterMapped`,
`cudaHostRegisterPortable` or no flags.

```sh
bash tools/mesh_register_repro_run.sh
```

Run the script from the repository root: it compiles the tool with `cc`
against `/usr/local/cuda` and runs it. The tool takes no arguments.

Output: one line per attempt,
`<label> ptr=... aligned=... flags=... -> <cuda error> (<code>)`, then
`failures=N`.

Exit codes: 0 when every attempt succeeds, 1 when any fails, 2 when the memfd
setup fails.

`REGION_BYTES` is hard-coded to 134283264. That is not
`SPARK_WEIGHTD_MESH_REGION_BYTES`, which `include/sparkpipe/spark_weightd.h`
defines as 1073938432, so the probe registers a smaller region than the
transport does.
