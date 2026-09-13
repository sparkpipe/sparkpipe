#!/usr/bin/env bash
# K3 F1 owed-evidence closer: THE LIVE MULTI-RANK NCCL EXCHANGE GATE (T1).
#
# Builds tests/test_k3_nccl_multirank_proof.cu against the PRODUCTION NCCL
# transport (ring/transport/tp_device_collective.c + tp_device_collective_nccl.c
# + hidden_transport.c - the dispatcher's SparkHiddenTransport* call sites make
# hidden_transport.o a hard link dependency even on the NCCL-only path; the same
# SparkTpDeviceCollective* the K3 runner calls) and runs WORLD real processes on
# one node: each process bootstraps ncclCommInitRank through
# SparkTpDeviceCollectiveCreate and drives the runner's two device-tier
# exchanges live - SubmitBf16 hidden sums at the K3 7168 width (bit-exact vs
# host expectation + narrowed-override tail integrity) and SubmitU64Max head
# argmax over engineered cross-rank cases vs the packed-key oracle (+ the
# host first-max rule on finite rows) - plus a back-to-back ordinal-chain leg.
#
# PASS here closes the DISPOSITION "OWED to the ring: a live multi-rank NCCL
# exchange proof". It proves what no single-rank gate can: real cross-process
# NCCL connectivity, real on-wire reduction outcomes, strict multi-op ordering.
#
# Usage:
#   bash tools/k3_nccl_multirank_gate.sh [NVCC] [WORLDS]
#     NVCC    default: $NVCC or nvcc on PATH
#     WORLDS  comma list in {2,4,8,16}; default "2,4"
# Env:
#   SPARK_NCCL_MODULE       path to libnccl.so.2 (else searched; SKIP if
#                           absent - never fails a box that cannot run it)
#   SPARK_K3_NVCC_ARCH      gencode pair (default compute_121a,sm_121a GB10)
#   SPARK_K3_MULTIRANK_TIMEOUT  per-world wall cap seconds (default 240)
#   NCCL_DEBUG / other NCCL_* pass through untouched
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
NVCC_ARG="${1:-}"
WORLDS="${2:-${SPARK_K3_MULTIRANK_WORLDS:-2,4}}"
ARCH_PAIR="${SPARK_K3_NVCC_ARCH:-compute_121a,sm_121a}"
WORLD_TIMEOUT="${SPARK_K3_MULTIRANK_TIMEOUT:-240}"
if [ -n "$NVCC_ARG" ]; then NVCC="$NVCC_ARG"; else NVCC="${NVCC:-nvcc}"; fi
ARCH_ARCH="${ARCH_PAIR%%,*}"
ARCH_CODE="${ARCH_PAIR##*,}"
command -v "$NVCC" >/dev/null 2>&1 || { echo "SKIP k3_nccl_multirank: no nvcc ($NVCC) on this host"; exit 0; }
command -v timeout >/dev/null 2>&1 || { echo "SKIP k3_nccl_multirank: no coreutils timeout"; exit 0; }
NCCL_MODULE="${SPARK_NCCL_MODULE:-}"
if [ -z "$NCCL_MODULE" ]; then
	NCCL_MODULE="$(ldconfig -p 2>/dev/null | awk '/libnccl[.]so[.]2[ .]/{print $NF; exit}' || true)"
	if [ -z "$NCCL_MODULE" ]; then
		for cand in /usr/lib/aarch64-linux-gnu/libnccl.so.2 \
			/usr/lib/x86_64-linux-gnu/libnccl.so.2 \
			/usr/local/cuda/lib64/libnccl.so.2 \
			/usr/lib/libnccl.so.2; do
			[ -e "$cand" ] && { NCCL_MODULE="$cand"; break; }
		done
	fi
fi
if [ -z "$NCCL_MODULE" ] || [ ! -e "$NCCL_MODULE" ]; then
	echo "SKIP k3_nccl_multirank: libnccl.so.2 not found (set SPARK_NCCL_MODULE)"
	exit 0
fi
echo "k3_nccl_multirank: nvcc=$NVCC arch=$ARCH_PAIR nccl=$NCCL_MODULE worlds=$WORLDS"
STAMP="$(date +%Y%m%dT%H%M%SZ)"
LOGDIR="$ROOT/tmp/gate_tmp/k3_nccl_multirank_$STAMP"
mkdir -p "$LOGDIR"
BIN="/tmp/k3_nccl_multirank_proof"
"$NVCC" -std=c++17 -O3 --expt-relaxed-constexpr -lineinfo \
	-gencode "arch=$ARCH_ARCH,code=$ARCH_CODE" \
	-Iinclude \
	tests/test_k3_nccl_multirank_proof.cu \
	ring/transport/tp_device_collective.c \
	ring/transport/tp_device_collective_nccl.c \
	ring/transport/hidden_transport.c \
	src/spark_status.c \
	-Xcompiler -fPIC -lpthread -ldl -lcudart -o "$BIN"
pick_port() {
	if command -v python3 >/dev/null 2>&1; then
		python3 - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
	else
		echo $((20000 + RANDOM % 20000))
	fi
}
overall=0
IFS=',' read -r -a WORLD_LIST <<< "$WORLDS"
for W in "${WORLD_LIST[@]}"; do
	W="$(echo "$W" | tr -d '[:space:]')"
	case "$W" in
		2|4|8|16) ;; # the backend's supported degrees only
		*) echo "FAIL k3_nccl_multirank: unsupported world $W (need 2|4|8|16)"; overall=1; continue ;;
	esac
	PORT="$(pick_port)"
	WDIR="$LOGDIR/world$W"
	mkdir -p "$WDIR"
	echo "---- world=$W port=$PORT -> $WDIR"
	pids=()
	for R in $(seq 0 $((W - 1))); do
		timeout "$WORLD_TIMEOUT" "$BIN" --rank "$R" --world "$W" \
			--port "$PORT" --module "$NCCL_MODULE" \
			>"$WDIR/rank$R.log" 2>&1 &
		pids+=("$!")
	done
	rc_all=0
	for P in "${pids[@]}"; do
		wait "$P" || rc_all=1
	done
	results="$(grep -h 'K3_NCCL_MULTIRANK_RANK_RESULT' "$WDIR"/rank*.log 2>/dev/null | sort || true)"
	clean="$(echo "$results" | grep -c 'failures=0' || true)"
	fails="$(grep -h '^FAIL' "$WDIR"/rank*.log 2>/dev/null | wc -l || true)"
	ranks_seen="$(echo "$results" | grep -c 'K3_NCCL_MULTIRANK_RANK_RESULT' || true)"
	if [ "$rc_all" -eq 0 ] && [ "$clean" -eq "$W" ] && [ "$fails" -eq 0 ] && [ "$ranks_seen" -eq "$W" ]; then
		echo "PASS k3_nccl_multirank world=$W ($W/$W ranks clean: live bf16-sum + u64-max + ordinal chain)"
	else
		echo "FAIL k3_nccl_multirank world=$W (rc_all=$rc_all clean=$clean/$W fail_lines=$fails results=$ranks_seen)"
		grep -h '^FAIL' "$WDIR"/rank*.log 2>/dev/null | sort -u | head -20 || true
		overall=1
	fi
done
echo "k3_nccl_multirank: logs under $LOGDIR"
if [ "$overall" -ne 0 ]; then
	echo "FAIL k3_nccl_multirank overall"
	exit 1
fi
echo "PASS k3_nccl_multirank overall (worlds: $WORLDS)"
