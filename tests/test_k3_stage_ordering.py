#!/usr/bin/env python3
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODULE = ROOT / "modules" / "k3_resident_decode_stage" / "source"
RUNNER = MODULE / "spark_k3_resident_decode_stage_runner.cu"
ADAPTER = MODULE / "spark_k3_serving_adapter.c"
TRANSPORT = ROOT / "ring" / "transport" / "host_staged_tcp.c"


def body(text, name):
    match = re.search(r"^[A-Za-z][^\n;]*\b" + name + r"\([^;{]*\)\s*\{", text, re.M)
    if match is None:
        return ""
    depth, cursor = 1, match.end()
    while cursor < len(text) and depth:
        if text[cursor] == "{":
            depth += 1
        elif text[cursor] == "}":
            depth -= 1
        cursor += 1
    return text[match.end():cursor]


def statements_calling(text, call):
    return [m.group(0) for m in re.finditer(r"(?m)^\s*" + call + r"\(", text)]


def check_runner(failures):
    text = RUNNER.read_text()
    copy = body(text, "K3RunnerCopy")
    if not re.search(r"cudaMemcpyAsync\(destination,\s*source,[^;]*stream\)", copy) or \
            not re.search(r"return cudaStreamSynchronize\(stream\);", copy):
        failures.append("K3RunnerCopy is not an execution-stream copy followed by a "
                        "sync of that stream; residentd's stream is non-blocking, so a "
                        "legacy-stream copy races the step kernels")
    acquire = body(text, "SparkK3RunnerLazyAcquire")
    outside = text.replace(acquire, "")
    if re.search(r"\bcudaMemcpy\(", outside):
        failures.append("the runner names a legacy-stream cudaMemcpy outside the lazy "
                        "route readback; stage inputs, index buffers, collective "
                        "staging and outputs go through K3RunnerCopy")
    sync = acquire.find("cudaStreamSynchronize(state->stream)")
    readback = acquire.find("cudaMemcpy(")
    if readback >= 0 and (sync < 0 or sync > readback):
        failures.append("the lazy route readback does not follow a sync of the "
                        "execution stream, so it can read stale group offsets")
    if statements_calling(text, "SparkTpCollectiveAllReduceSumBf16"):
        failures.append("a host all-reduce result is discarded; a failed reduce "
                        "must fail the step instead of decoding partial sums")
    if statements_calling(text, "SparkTpDeviceCollectiveEnqueue"):
        failures.append("a device-collective enqueue result is discarded")
    completion = body(text, "K3RunnerTpCompletion")
    if not completion or statements_calling(completion, "cudaMemcpyAsync"):
        failures.append("the device-collective completion discards a copy result; "
                        "a failed gate_up or shared copy must fail the step")
    if completion.count("copy_failed = 1u") < 2:
        failures.append("the device-collective completion does not record a failed copy")
    if not re.search(r"completion->status != SPARK_STATUS_OK\s*\)\s*tp->owner->tp_collective_failed = 1u", completion):
        failures.append("the device-collective completion ignores a failed collective status")
    reduce_helper = body(text, "K3RunnerHostAllReduce")
    if "tp_collective_failed = 1u" not in reduce_helper:
        failures.append("K3RunnerHostAllReduce does not record a failed reduce")
    submit = body(text, "SparkK3StageRunnerSubmit")
    begin = submit.find("K3RunnerChainBegin(state, dispatch->request_id)")
    first_work = min(i for i in (submit.find("K3Embedding("), submit.find("K3RunnerCopy(")) if i >= 0)
    if begin < 0 or begin > first_work:
        failures.append("the step does not key its device-collective chain before "
                        "its first collective; an unkeyed chain exhausts the "
                        "rebase budget after a few steps")
    if "K3RunnerChainEnd(state, stream)" not in submit:
        failures.append("the step does not end its device-collective chain")
    take = submit.find("K3RunnerTakeFailure(state)")
    end = submit.find("K3RunnerChainEnd(state, stream)")
    if take < 0 or end < 0 or take > end:
        failures.append("the step does not report a failed copy or collective "
                        "before it completes")
    chain = body(text, "K3RunnerChainBegin")
    for band in ("device_collective", "device_collective_wide"):
        if f"SparkTpDeviceCollectiveChainKey(&state->{band}, key)" not in chain:
            failures.append(f"K3RunnerChainBegin does not key the {band} band")
    half = body(text, "SparkK3StageRunnerStepHalf")
    if "K3RunnerTakeFailure(state)" not in half:
        failures.append("the half step does not report a failed copy or collective")
    seed = body(text, "K3RunnerSeedIndices")
    if not re.search(r"resident_sequence_capacity > configuration->max_active_sequence_count\s*\)"
                     r"[^}]*return SPARK_STATUS_INVALID_ARGUMENT", seed, re.S):
        failures.append("the runner accepts more resident sequences than it has "
                        "KDA state slots")


def check_adapter(failures):
    submit = body(ADAPTER.read_text(), "K3ServingSubmit")
    copies = re.findall(r"SparkMemoryBufferCopy\(&state->(\w+),[^;]*?,\s*([^,;]+)\)\s*!=", submit)
    device = [(name, stream) for name, stream in copies if name.endswith("_device")]
    if not device:
        failures.append("the adapter's device index copies are not checked")
    for name, stream in device:
        if stream.strip() != "state->runner_config.execution_stream":
            failures.append(f"the adapter copies {name} on {stream.strip()}, not the "
                            f"execution stream the step kernels run on")
    if re.search(r"\(void\)SparkMemoryBufferCopy\(", submit):
        failures.append("the adapter discards a copy result")


def check_transport(failures):
    match = body(TRANSPORT.read_text(), "host_staged_match")
    async_copy = match.find("cudaMemcpyAsync((void *)packet->hidden_bf16")
    sync = match.find("cudaStreamSynchronize((cudaStream_t)packet->cuda_stream)")
    complete = match.find("host_staged_complete(state,packet,status)")
    if re.search(r"\bcudaMemcpy\(", match):
        failures.append("the transport receive copies a frame on the legacy stream")
    if async_copy < 0 or sync < 0 or complete < 0 or not async_copy < sync < complete:
        failures.append("the transport completes a received packet before its "
                        "stream-ordered copy has landed")


def main():
    failures = []
    check_runner(failures)
    check_adapter(failures)
    check_transport(failures)
    for failure in failures:
        print(f"  FAIL {failure}")
    if failures:
        print(f"\nFAIL ({len(failures)})")
        return 1
    print("K3 stage copies run on the execution stream and sync before use; a "
          "failed copy or collective fails the step; each step keys and ends its "
          "device-collective chain; the runner refuses more resident sequences than KDA state slots")
    return 0


if __name__ == "__main__":
    sys.exit(main())
