#!/usr/bin/env python3
"""K3 ride-along contracts: F3/F4/F5/F7 fixes, completion stream-ordering,
and D2/D3 single-sourcing (slice geometry, gate|up sizing). Source pins so
the fixed defect classes cannot silently return."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
K3 = ROOT / "modules" / "k3_resident_decode_stage" / "source"
RUNNER = K3 / "spark_k3_resident_decode_stage_runner.cu"
CUDA = K3 / "spark_k3_resident_decode_stage_cuda.cu"
ADAPTER = K3 / "spark_k3_serving_adapter.c"
MODULE = K3 / "spark_k3_resident_decode_stage_module.c"


def strip_line_comments(text):
    return chr(10).join(line.split("//")[0] for line in text.splitlines())


def span(text, start_marker, end_marker):
    begin = text.find(start_marker)
    if begin < 0:
        return None
    if end_marker is None:
        return text[begin:]
    finish = text.find(end_marker, begin + len(start_marker))
    if finish < 0:
        return text[begin:]
    return text[begin:finish]


def fail(msg):
    print("  FAIL " + msg)


def main():
    failures = 0
    runner = strip_line_comments(RUNNER.read_text())
    adapter = strip_line_comments(ADAPTER.read_text())
    cuda_src = strip_line_comments(CUDA.read_text())
    module_src = strip_line_comments(MODULE.read_text())

    # -- F3 -------------------------------------------------------------
    combine = span(runner, "static SparkStatus K3RunnerCombineBf16(",
                   "static SparkStatus K3RunnerCombineTp4Bf16(")
    if combine is None:
        fail("K3RunnerCombineBf16 not found")
        failures += 1
    elif "K3RunnerCombinePairKernel" not in combine:
        fail("two-rank combine must launch the lane-exact pair kernel")
        failures += 1
    elif "pair[4]" in combine:
        fail("two-rank combine still builds the NULL-padded four-lane array")
        failures += 1
    if "K3RunnerCombinePairKernel" not in runner:
        fail("the lane-exact pair kernel is missing")
        failures += 1

    # -- F4 -------------------------------------------------------------
    macros_ok = "#define K3_INIT_FAIL" in runner and "#define K3_CUDA_ALLOC_OR_FAIL" in runner
    if not macros_ok:
        fail("unified init-failure macros gone; unchecked cudaMallocs race boot under memory pressure")
        failures += 1
    bare = runner.count("cudaMalloc(&state->")
    if bare:
        fail(str(bare) + " unchecked state-> cudaMalloc(s) remain in the runner")
        failures += 1
    if "SparkK3StageRunnerDestroy(runner); return" not in runner:
        fail("init failures no longer funnel through the one Destroy")
        failures += 1
    if "head_slots_device" in runner:
        fail("the dead device slot mirror survived")
        failures += 1
    destroy = span(module_src, "void SparkK3ModuleDestroy(", None)
    if destroy is None or "pack.fd = -1;" not in destroy:
        fail("module close lost the fd sentinel; second destroy closes stdin")
        failures += 1
    scratch_ok = "K3ServingFreeScratch" in adapter and adapter.count("K3ServingFreeScratch(state)") >= 2
    if not scratch_ok:
        fail("adapter scratch release helper missing or unused on failure paths")
        failures += 1

    # -- F5 -------------------------------------------------------------
    bind_weights = span(cuda_src, "int32_t SparkK3DispatchBindWeights(",
                        "int32_t SparkK3DispatchStep(")
    if bind_weights is None:
        fail("SparkK3DispatchBindWeights not found")
        failures += 1
    else:
        has_kind = "SPARK_K3_PACK_KIND_MXFP4_WS_INTERLEAVED_V1" in bind_weights
        stale_rule = "layer_is_dense ? 0u : 1u" in bind_weights
        if not has_kind or stale_rule:
            fail("expert_interleave must derive from the manifest tensor kind only")
            failures += 1

    # -- F7 --------------------------------------------------------------
    for name, src in (("runner", runner), ("adapter", adapter)):
        if "completion.accepted_token_count = copied;" not in src:
            fail(name + " completion reports rows instead of carried tokens")
            failures += 1

    # -- TS: publish only after the stream drains -------------------------
    submit_at = adapter.find("SparkK3StageRunnerSubmit(&state->runner")
    sink_at = adapter.find("state->common.sink.function(")
    sync_at = adapter.find("cudaStreamSynchronize")
    ordered = submit_at >= 0 and sink_at >= 0 and sync_at >= 0 and submit_at < sync_at < sink_at
    if not ordered:
        fail("post-submit cudaStreamSynchronize lost or misordered; enqueue-only submit raced its completion")
        failures += 1
    else:
        gate_span = adapter[submit_at:sink_at]
        if "return SPARK_STATUS_INTERNAL_ERROR;" not in gate_span:
            fail("failed synchronization does not abort; OK after a broken sync is the same defect")
            failures += 1

    # -- D5: the bind tables and the dispatch requirement sets agree --------
    bind_src = strip_line_comments((ROOT / "modules" / "k3_resident_decode_stage" / "source" / "spark_k3_bind.c").read_text())
    def names_in(text):
        import re as _re
        pat = chr(34) + "([a-z0-9_]+)" + chr(34)
        return set(_re.findall(pat, text))
    bind_names = names_in(span(bind_src, "static SparkStatus SparkK3BindEveryLayer",
                              "SparkStatus SparkK3BindLayer("))
    cuda_names = names_in(span(cuda_src, "static const char *const k3_required_every",
                               "static int32_t k3_require("))
    missing = sorted(bind_names - cuda_names)
    if missing:
        fail("bind.c resolves names the dispatch never requires: " + ", ".join(missing))
        failures += 1
    if "router_bias" not in cuda_names:
        fail("router_bias not required on routed layers; a pack without it would silently decode biasless")
        failures += 1

    # -- hot path: no per-submit heap traffic in the adapter -----------------
    submit_body = span(adapter, "SparkK3StageRunnerSubmit(&state->runner", None)
    if submit_body is None or "malloc(" in adapter[adapter.find("static SparkStatus K3ServingSubmit"):sink_at]:
        fail("per-submit malloc survived in K3ServingSubmit; staging buffers are preallocated at max_rows")
        failures += 1

    # -- D2/D3: single-sourced geometry ------------------------------------
    for stale in ("{0u, 24u, 47u, 70u}", "{24u, 23u, 23u, 23u}"):
        if stale in runner:
            fail("private PP4 stage table survived in the runner; one spelling lives in spark_k3_pool_sizing.h")
            failures += 1
    if "SPARK_K3_TOTAL_LAYERS" not in module_src:
        fail("module still pins the backbone length as a bare literal")
        failures += 1
    sites = runner.count("K3_EXPERT_INTERMEDIATE")
    if sites != 2:
        fail("gate|up payload factor appears " + str(sites) + " times in the runner; want exactly 2 (the sizing helpers)")
        failures += 1

    if failures:
        print("FAIL (" + str(failures) + ")")
        return 1
    print("pair combine lane-exact; allocations checked with one teardown; interleave follows kind; counts match tokens; completion publishes post-drain; geometry single-sourced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
