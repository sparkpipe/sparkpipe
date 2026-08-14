# Spark Hardware Qualification Handoff

This handoff is designed to answer every hardware-dependent SparkPipe decision with retained evidence from the exact source package, exact production kernels, exact production transports, and the selected physical topology.

The source package intentionally contains no synthetic production provider. A model, transport, or topology cell is accepted only when its provider proves the exact production artifact identity and the required integrity/numerical contract.

## Qualification order

Run the qualification in this order:

1. Verify and extract the exact source archive.
2. Run the complete host build and host test inventory.
3. Compile every owned CUDA translation unit with CUDA 13 for `compute_121a` and exact `sm_121a`.
4. Build the generic hardware probes.
5. Build provider shared objects from the exact production model, transport, and topology implementations.
6. Generate a qualification plan bound to the source archive SHA-256.
7. Generate per-node runner configurations.
8. Run the direct single-rail ring plan first.
9. Aggregate receipts, compile policy, and prove question closure.
10. Fix any failed or unanswered question before interpreting performance.
11. Repeat with the one-switch single-rail topology.
12. Compare the two closed policies; do not enable dual rail yet.

## Verify and extract the release

```sh
archive=sparkpipe-phase10-hardware-handoff-2026-08-02.tar.gz
sha256sum -c "${archive}.sha256"
mkdir -p /opt/sparkpipe-phase10

tar -xzf "${archive}" -C /opt/sparkpipe-phase10 --strip-components=1
cd /opt/sparkpipe-phase10

python3 tools/verify_package_manifest.py
make clean
make -j2 all
make -j2 test
sh tools/gates.sh
```

The source tree does not certify itself. The archive SHA-256 and external verification receipt are the qualification identity.

## CUDA 13 exact-target compilation

A GPU is not required for this step. CUDA 13 is required.

```sh
export NVCC=/usr/local/cuda/bin/nvcc
export CUDA_ARCH=sm_121a
export SPARK_CUDA_GATE_SCOPE=complete

sh tools/cuda13_sm121a_compile_gate.sh
```

The gate retains:

- `nvcc --version`;
- exact command configuration;
- `compute_121a` PTX;
- exact `sm_121a` objects;
- `ptxas` resource reports;
- `cuobjdump` resource and architecture listings;
- SHA-256 for every retained artifact.

A passing compile gate does not imply numerical correctness, race freedom, CUDA Graph correctness, network correctness, or performance.

## Build hardware probes

```sh
make -j2 hardware_tools
make hardware_cuda_tools NVCC="$NVCC"
make hardware_handoff
```

The generic tools are:

```text
build/spark_cuda_characterize
build/spark_nvme_characterize
build/spark_model_kernel_characterize
build/spark_transport_characterize
build/spark_topology_characterize
build/spark_pmtu_characterize
```

## Required production providers

Copy `qualification/spark/provider_map.example.json` and replace every provider path with the exact built artifact.

Required model providers:

```text
Kimi K3 MXFP4 experts / BF16 rest
GLM 5.2 FP8 E4M3 experts / BF16 rest
Qwen 3.6 27B BF16
DeepSeek V4 Flash checkpoint-native mixed precision
DeepSeek V4 Pro checkpoint-native mixed precision
```

Required transport providers:

```text
TCP production path
mapped-host RDMA production path
GPUDirect RDMA production path
```

Required topology provider:

```text
actual end-to-end pipeline implementation
```

The wrappers reject providers that cannot prove:

- immutable production artifact SHA-256;
- exact production implementation selection;
- an independent numerical reference for model kernels;
- a bound `ptxas` resource receipt for model kernels;
- real socket or verbs execution for transports;
- remote completion and source/destination fingerprint equality;
- matching local and remote transport artifact identities;
- no hidden CPU staging for mapped-host or GPUDirect candidates.

## Source-package identity

Use the archive SHA-256, not a Git commit and not the extracted directory hash.

```sh
archive_sha256=$(awk '{print $1}' \
    /path/to/sparkpipe-phase10-hardware-handoff-2026-08-02.tar.gz.sha256)
```

## Generate the ring plan

```sh
run_id="ring-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p qualification/runs/"$run_id"

python3 tools/hardware/spark_qualification_plan.py \
    --topology qualification/spark/topologies/ring_13node_bringup.json \
    --source-package-sha256 "$archive_sha256" \
    --output qualification/runs/"$run_id"/plan.json

python3 tools/spark_hardware_qualify.py \
    validate-plan qualification/runs/"$run_id"/plan.json
```

The ring plan assumes physical pipeline order follows physical ring adjacency. It does not attempt arbitrary multi-hop routing.

## Generate exact per-node configurations

```sh
cp qualification/spark/provider_map.example.json \
    qualification/runs/"$run_id"/provider_map.json
# Edit provider_map.json to point at the exact production providers.

python3 tools/hardware/generate_runner_configs.py \
    --topology qualification/spark/topologies/ring_13node_bringup.json \
    --provider-map qualification/runs/"$run_id"/provider_map.json \
    --output-directory qualification/runs/"$run_id"/configs \
    --run-id "$run_id" \
    --receipt-root qualification/receipts \
    --nvme-file /mnt/nvme/sparkpipe-hardware-probe.bin \
    --nvme-file-bytes 8589934592
```

The checked-in `qualification/spark/configs/*_handoff` directories are topology/path templates. Regenerate them for the actual run ID, provider locations, receipt root, and NVMe path.

## Preflight every node

Run this on each Spark before starting expensive cells:

```sh
node=$(hostname -s)
python3 tools/hardware/spark_handoff_preflight.py \
    --plan qualification/runs/"$run_id"/plan.json \
    --config qualification/runs/"$run_id"/configs/"$node".json \
    --output qualification/runs/"$run_id"/preflight-"$node".json
```

Preflight verifies:

- runner configuration SHA-256;
- topology and rank membership;
- every exact plan cell maps to a real command;
- every required executable and provider exists;
- executable and provider hashes;
- no provider resolves under tests or uses a synthetic name;
- every peer address required by the node is present;
- the receipt directory is writable;
- the prepared NVMe corpus exists at the required size.

## Execute the plan

One process may run the cells assigned to its node:

```sh
python3 tools/hardware/run_probe_job.py \
    --plan qualification/runs/"$run_id"/plan.json \
    --config qualification/runs/"$run_id"/configs/"$(hostname -s)".json \
    --resume
```

For parallel execution within a node, use disjoint deterministic shards:

```sh
shard_count=8
for shard_index in $(seq 0 $((shard_count - 1))); do
    python3 tools/hardware/run_probe_job.py \
        --plan qualification/runs/"$run_id"/plan.json \
        --config qualification/runs/"$run_id"/configs/"$(hostname -s)".json \
        --shard-index "$shard_index" \
        --shard-count "$shard_count" \
        --resume &
done
wait
```

Per-cell locks prevent duplicate ownership. Existing valid receipts are not overwritten under `--resume`. Timeouts produce failed receipts rather than hanging indefinitely.

## Aggregate, compile policy, and prove closure

Collect all cell receipts under one directory, then run:

```sh
python3 tools/spark_hardware_qualify.py aggregate \
    --plan qualification/runs/"$run_id"/plan.json \
    --receipts qualification/receipts \
    --output qualification/runs/"$run_id"/aggregate.json

python3 tools/spark_hardware_qualify.py validate-aggregate \
    --plan qualification/runs/"$run_id"/plan.json \
    qualification/runs/"$run_id"/aggregate.json

python3 tools/hardware/spark_policy.py \
    --plan qualification/runs/"$run_id"/plan.json \
    --aggregate qualification/runs/"$run_id"/aggregate.json \
    --output qualification/runs/"$run_id"/policy.json

python3 tools/hardware/spark_question_closure.py \
    --plan qualification/runs/"$run_id"/plan.json \
    --aggregate qualification/runs/"$run_id"/aggregate.json \
    --policy qualification/runs/"$run_id"/policy.json \
    --output qualification/runs/"$run_id"/closure.json
```

A production policy is valid only when closure reports all required questions closed.

The generated policy includes per-node and per-peer decisions. It also derives:

- mapped-host versus explicit-copy transfer choice for each node and payload;
- measured CPU-read and CPU-write contention effects relative to GPU-only bandwidth.

It does not collapse node or peer differences into one best-node result.

## One-switch qualification

After the ring closes without correctness or recovery failures, repeat the same process with:

```text
qualification/spark/topologies/single_switch_16node.json
```

Use the actual node count if fewer than 16 Sparks are connected. Regenerate the topology rather than leaving absent ranks in the plan.

The one-switch run must answer the same 33 questions independently. Do not copy ring policy values into the switch policy.

## What the policy answers

The closed policy directly answers:

- sustainable GB10 bandwidth by working-set size;
- cache/reuse benefit and sparse pointer-chase cost;
- CPU/GPU memory contention;
- mapped-host versus explicit-copy crossover by payload;
- launch, graph, event, callback, and synchronization crossover;
- useful copy/compute concurrency;
- accepted dynamic shared memory and resulting active blocks per SM;
- atomic strategy;
- sustained thermal derating;
- registers, spills, local memory, shared memory, and exact production-kernel throughput;
- native block-scaled MMA crossover;
- grouped/split-key GQA strategy by model, batch, and context;
- token-centric versus grouped/weight-stationary MoE strategy;
- KDA token-step versus replay/chunk strategy;
- NVMe block size, queue depth, worker count, and GPU-pipeline behavior;
- TCP and RDMA latency/throughput tradeoffs;
- mapped-host versus GPUDirect RDMA;
- lane count, window depth, CQ batch, MR-cache size, and progress mode;
- PMTU;
- ring and one-switch end-to-end behavior;
- PP degree, transport window, and stage placement by model, batch, and context.

## Evidence separation

The source archive excludes raw receipts and qualification logs. Preserve them as a separate immutable evidence package containing:

```text
source archive SHA-256
plan and plan ID
runner configurations and hashes
preflight reports
all cell receipts
aggregate
compiled policy
closure report
CUDA compile artifacts and resource reports
network topology and switch configuration
software, firmware, CUDA driver, NIC, and kernel identities
```

Do not edit a receipt in place. A rerun must use a new run ID and new receipt directory.
