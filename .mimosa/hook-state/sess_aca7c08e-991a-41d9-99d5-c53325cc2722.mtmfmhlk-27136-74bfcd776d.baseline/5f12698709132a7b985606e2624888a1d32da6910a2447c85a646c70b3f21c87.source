#!/bin/sh
# Every code file, with what it does. One line each.
#
# Exists because a breakdown by directory says how much and not what, and the
# question "would we lose anything by deleting this" needs the second.
cd "$(dirname "$0")/.." || exit 1
describe() {
	case "$1" in
	*/kernels/mma.cuh) echo "tensor core atoms, fragment layouts verified against CUTLASS" ;;
	*/kernels/tma.cuh) echo "async tile staging, mbarrier primitives" ;;
	*/kernels/tile.cuh) echo "the staged pipeline; two stages, lookahead one" ;;
	*/kernels/layout.cuh) echo "tile and swizzle geometry, host-computable, no CUDA" ;;
	*/kernels/dtype.cuh) echo "element conversions" ;;
	*/kernels/gemm.cuh) echo "ONE GEMM: grouped, dense is one group, format is a trait" ;;
	*/kernels/kv.cuh) echo "paged KV: opaque bytes, growing or recurrent" ;;
	*/kernels/norm.cuh) echo "rms norm, silu-mul, quantise, fused norm+quantise, moe finalize" ;;
	*/kernels/attn.cuh) echo "rope, yarn, latent attention, sparse scoring, hierarchical selection" ;;
	*/kernels/project.cuh) echo "low-rank projection, absorbed projection, fused QKV split" ;;
	*/kernels/topk.cuh) echo "top-k both shapes: bitonic small, radix large" ;;
	*/kernels/head.cuh) echo "sampling head: candidates, commit, softmax" ;;
	*/kernels/speculate.cuh) echo "speculative verify and accept, greedy and sampled" ;;
	*/kernels/linear_attn.cuh) echo "delta rule decode, causal conv; GDN and KDA" ;;
	*/kernels/graph.cuh) echo "CUDA graph capture keyed by shape" ;;
	*/kernels/tensor_map.cuh) echo "TMA descriptor geometry" ;;
	*/kernels/formats/*) echo "weight format trait: stored width, mma, fragment decode" ;;
	modules/*/source/cuda/config.h) echo "package-specialized model shapes and constants" ;;
	modules/*/source/cuda/unity.cu) echo "package-specialized CUDA entry points" ;;
	modules/*/source/cuda/layer.cuh) echo "model layer launch graph" ;;
	modules/*/source/cuda/api.h) echo "model module CUDA ABI" ;;
	modules/*/source/*module.c) echo "immutable model-driver module" ;;
	modules/*/source/*serving_adapter.c) echo "model-owned serving adapter" ;;
	node/model_residentd.c) echo "generic persistent model and transport owner" ;;
	runtime/model_batch_engine.c) echo "generic request admission, batching, prefill and decode" ;;
	runtime/model_pipeline_client.c) echo "generic multi-rank resident IPC client" ;;
	runtime/model_resident_client.c) echo "one resident IPC connection" ;;
	runtime/model_resident_deployment.c) echo "strict deployment parser and topology contract" ;;
	runtime/model_resident_ipc.c) echo "resident wire protocol" ;;
	runtime/model_serving_adapter.c) echo "model adapter loader and ABI validation" ;;
	runtime/pipeline_runtime.c) echo "model-neutral rank and boundary plan" ;;
	text/tokenizer.c) echo "generic compiled byte-level BPE tokenizer" ;;
	modules/glm52_dspark_draft_backend/source/spark_glm52_dspark_dispatch_policy.c) echo "DSpark dispatch policy" ;;
	cache/cache.h) echo "THE CACHE: arena, content-addressed sharing, JIT reserve" ;;
	cache/store/*) echo "KV block store and client" ;;
	ring/sideband.h) echo "cross-rank payloads: index share, hidden tap, prefix indices" ;;
	ring/transport/hidden_transport.*) echo "hidden state between ranks" ;;
	ring/transport/tp_collective.*) echo "tensor-parallel collectives" ;;
	ring/transport/memlink.*) echo "shared memory link" ;;
	ring/transport/rdma.cu) echo "RDMA backend" ;;
	ring/transport/tcp.cu) echo "TCP backend" ;;
	runtime/launch.h) echo "launch planning: tile height, shared bytes, grid" ;;
	runtime/gemm.cuh) echo "the four CUDA calls a GEMM needs" ;;
	runtime/workspace.h) echo "workspace pool layout" ;;
	runtime/tensor_map.h) echo "cuTensorMapEncodeTiled" ;;
	runtime/linear_plan.cu) echo "linear plan binding" ;;
	runtime/json.c) echo "JSON parser" ;;
	runtime/filesystem.c) echo "filesystem wrapper" ;;
	compiler/*) echo "AOT model-driver compiler and loader support" ;;
	deployment/*) echo "immutable release manifest, sync and process manager" ;;
	*) echo "" ;;
	esac
}
for f in $(git ls-files | grep -E '\.(c|cu|cuh|h)$' | grep -vE '^(tests|tools|docs)/')
do
	d=$(describe "$f")
	[ -z "$d" ] && continue
	printf "  %-52s %5s  %s\n" "$f" "$(wc -l < "$f")" "$d"
done
