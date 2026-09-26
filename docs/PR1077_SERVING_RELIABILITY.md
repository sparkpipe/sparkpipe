# PR1077 serving reliability

This change stacks on PR #1077, `hillclimb/graph-pacing` at
`89330dcf8a804c29ff1a20350fadec77321275bf`. It fixes the failure-reporting,
resource-lifetime and reconnect defects found in the September 22 review.
It does not establish TP16 throughput or qualify a production deployment.
The parent incorporated the initial collective/session repairs; the remaining
stacked change completes teardown and recorded-working-set hardening.

## Runtime contract

- A failed CUDA graph collective keeps its error until the owner consumes it.
  Poisoned maxloc output stays invalid. Completion checks failure before cache
  commit or token publication. A graph failure fences the engine; it cannot
  silently switch execution mode or retry as successful inference. Graph replay
  uses the configured collective operation timeout; cancellation drain remains
  bounded and cannot release resources while the stream is still running.
- A synchronous collective rejection returns an error and queues no callback.
  Accepted work has one terminal callback. This prevents freeing a chain on
  rejection and then calling its queued callback through the freed pointer.
- Control socket reconnection aborts the previous session. It does not resume
  partially sent frames. Prepared work is aborted, submitted work retains its
  route and slot claims until terminal completion, and reset completes before
  the new HELLO_ACK enables admission. A hard reset error fences the engine.
- A transport failure resets the whole pipeline session. Its fingerprint stays
  unchanged until every rank is ready. Transactions terminate through their
  callbacks instead of disappearing from bookkeeping. BUSY is backpressure.
- A response that has already emitted tokens ends with an error after session
  loss. It is never silently replayed from token zero. A fresh request can use
  the recovered session. Surviving queued requests also reset their prefix
  digest and cache lookup epoch before rebuilding the prompt.
- Loss of either the lane client or the weight-map client fences the GLM
  engine. Residentd exits through its existing quiesce path so supervision can
  reconstruct the process, graphs, maps, spine and collectives together. It
  never destroys a collective from its own callback. If device work has not
  terminated, resource ownership is retained until process exit.
- Cold arena operations use the existing context-bound weightd worker, with
  one active arena operation. Accepts, HELLO and mesh control progress continue
  on the control thread. Connection cleanup waits for worker completion.
  Interrupted IPC exchanges poison the client socket, preventing stale replies
  from being mistaken for the next request.
- The supervisor retains a live daemon after a failed readiness probe. An
  installed update requires dependent engines to drain before replacement.
  Readiness failure also gates dependent root, API and warm-up startup.
  TCP bind owns the singleton latch; a duplicate launch fails without killing
  or declaring an unverified holder ready.

## Configuration and warm-up

Set `SPARK_GLM5_NEXT_GRAPH_PATH` explicitly to `0` or `1` for every engine
(`G5_GRAPH_PATH` when launching through the fleet agent).
Set `SPARK_WEIGHTD_EXPERT_POOL_BYTES` to the same explicit finite budget in
all clients attaching the same arena. The daemon rejects incompatible budgets,
overflow and allocations exceeding its device ceiling. A configured pool or
preload allocation failure is terminal, not a switch to another allocation
policy. Existing fleet configuration uses 32 GiB; choose the deployment budget
from its actual weight and memory plan, not from that example.

Remove `SPARK_GLM5_NEXT_PREFETCH=1`. The unowned background prefetch thread and
its leaked leases have been removed. Demand loading remains available; the
separate `build/weightd_warm` tool can warm the daemon before starting serving.
It reads the real `.experts` manifest, excludes nonexistent dense-layer experts,
checks both acquire and release, and exits nonzero at the first required failure.
A successful layer print is emitted only after release succeeds. A warm pass
is not a permanent pin: an undersized pool can evict earlier layers.

The upstream `--wset FILE [TIMEOUT_S]` option remains available for one selected
working set. It validates the file and manifest keys and checks acquire/release
before reporting success. A selected working set qualifies only those keys,
not every expert in the model. Recording loads after the pack path is set,
rejects malformed existing files, publishes with checked atomic rename, and
rolls back keys and the acquired lease if publication fails. The pack directory
must be writable; recording failures are visible errors.

Cold span reads accumulate short reads before checksum and device copy. A
short read can no longer validate or publish an incomplete span.

The shared memory-buffer free helper resets its descriptor before freeing the
allocation. Serving adapters may store that descriptor inside the allocation;
writing it afterward caused a real Linux teardown SIGSEGV. Pinned allocations
also use their matching CUDA host-free operation.

TCP keepalive and no-SIGPIPE setup are checked on Linux and macOS. An option
failure rejects the socket. Platform-specific option names implement the same
contract; they do not select a degraded mode.

## Timing interpretation

`collective_host_submit_ms` and `collective_host_submissions` describe host
submission work, including capture. They are not graph replay allreduce timing.
`path` names how the chain ran: `graph`, `linear`, or `eager` for the chain
state machine. A linear chain checks its rounds once at the end, so its
submission time is launch work only.
`GRAPH-REPLAY-TIME wall_ns` includes compute and waits; `stream_status` is the
CUDA query result, followed by the separate sticky collective-error check.
`elapsed_since_previous_wait_end_us` includes intervening computation.

Ninety 60-microsecond reductions account for about 5.4 ms per token. These
figures cannot explain a seconds-per-token serving cadence. The revised error
path must be qualified before comparing NCCL and mesh with identical model,
codec, prompt/context, batch geometry, build, and diagnostics settings.

## Focused validation

The new checks are registered in the existing Makefile test lists. The
`tools/test_serving_reliability_host.sh` runner executes 21 focused host
regression targets on Linux in a fresh checkout, independently of CUDA compilation. GitHub rejected adding
the prepared Actions workflow because the configured PAT lacks `workflow`
scope; CI wiring is pending that permission.

| Check | Result and boundary |
| --- | --- |
| Graph wait/guard/unpack, completion and daemon-loss harness | PASS under ASan/UBSan; executes production function bodies with CUDA/fabric boundaries mocked |
| Allreduce transport fuzzer | PASS, 274 checks; host CUDA and shipper stubs |
| GLM stage context, embedding collective and lazy dispatch | PASS; host module integration and opaque-generation checks |
| Memory buffer | PASS on Linux/macOS and ASan; self-owned descriptors and matching deallocation, negative control fails |
| Resident session/deadline/IPC | PASS; real partial socket writes, reset BUSY/failure, preserved route claims, no-SIGPIPE and terminal progress |
| Resident reconnect | PASS; same live daemon disconnect and separate process restart |
| Pipeline integration and mock | PASS; malformed completion fails exactly once, same daemon PIDs survive and accept later work |
| Batch engine mock | PASS, 40 checks; interrupted partial response errors once, fresh request succeeds |
| Weightd fd/progress | PASS under ASan/UBSan and TSan; blocked real acquire still permits 144 sequential HELLO/close probes and mesh RPCs, timeout poisons socket and fresh client recovers |
| Weightd working set, experts, churn, stress, attach, worker | PASS; finite budgets, configured allocation failure rollback, short reads and lease cleanup |
| Warmer/supervision tests | PASS, 15 tests; actual warmer and latch bodies, injected client failures and supervisor syscall fixtures |

The existing `test_weightd_map` fails its 2-MiB chunk-size assertion against
both the baseline and this branch: cold chunks use a 64-MiB minimum. This
assertion was not weakened. Broad macOS `make all` also has setup failures in
unrelated verbs declarations and missing DSV4 mesh shim symbols. No full-suite
pass is claimed.

Linux/Spark0 validation uses an isolated source archive, separate host-stub
and real-CUDA build directories, and no running service changes. CUDA 13.0.88
successfully compiles the GLM FP8 translation unit; `cuobjdump` identifies
`sm_121a`. The runtime, residentd, weightd, warmer, and GLM module/adapter also
compile against real CUDA headers/libraries. Module identity macros are labeled
`compile-only`; these objects are not deployable model artifacts.

## Deployment qualification still required

Use one immutable build and configuration across all 16 ranks. Verify a real
GPU graph missing-peer/cancellation failure emits no token, then rebuild the
engine and compare successful output with the numerical reference. Exercise
control disconnect during partial output, daemon replacement, and subsequent
requests. Run a sustained serving soak checking lost/duplicate completions,
lease counts and memory growth. Only then compare unprofiled NCCL/mesh cold
load, prefill, TTFT and warm decode. Host mocks and CUDA compilation do not
qualify these hardware behaviors.

## Latest parent integration

PR1081 includes PR1077 head `1a62b723a12701b2ac092f77cf66cea6ad0a0b60`
without conflicts. The broader campaign and its remaining gates are recorded
in `SERVING_FUZZ_COVERAGE.md`. In addition to the earlier teardown repairs,
the change removes deadline-based theft of executing cache lanes, reserves
collective completion capacity before accepting work, preserves prepare/abort
ordering after partial admission, and waits for every rank before final callbacks.

Malformed or nonregular pack digest sidecars fail at their actual read error;
they are not skipped in favor of another identity. Explicit incompatible
`SPARK_WEIGHTD_ATTACH` values fail deployment startup. A runtime bundle must
contain one valid intended pack digest, without unrelated sidecar debris.
