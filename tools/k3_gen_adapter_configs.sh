#!/usr/bin/env bash
# k3_gen_adapter_configs.sh OUT_DIR — emit the 16 per-rank serving adapter
# configs (rank <host>.json) with the pack paths and the TP4 collective
# topology. The ring's static IPs are 10.20.0.(10+i) for spark index i
# (0..9, a..f).
set -euo pipefail
OUT="${1:?usage: k3_gen_adapter_configs.sh OUT_DIR}"
mkdir -p "$OUT"
for i in $(seq 0 15); do
  hex=$(printf '%x' "$i")
  host="spark$hex"
  stage=$((i / 4))
  rank=$((i % 4))
  pack="/home/$host/sparkdata/k3.mxfp4.tp4pp4/packs/k3.stage${stage}.rank0${rank}.pack"
  {
    echo "{"
    echo "  \"stage_pack_path\": \"$pack\","
    echo "  \"tp_degree\": 4,"
    echo "  \"tp_rank\": $rank,"
    echo "  \"max_sequences\": 16,"
    echo "  \"max_rows\": 16,"
    echo "  \"resident_capacity\": 16,"
    echo "  \"kv_pages\": 2,"
    echo "  \"tp_collective\": {"
    echo "    \"listen_port\": $((65620 + rank)),"
    echo "    \"connect_timeout_milli\": 5000,"
    echo "    \"operation_timeout_milli\": 30000,"
    echo "    \"collective_identifier\": 1,"
    echo "    \"peers\": ["
    for r in 0 1 2 3; do
      p=$((stage * 4 + r))
      ph=$(printf '%x' "$p")
      ip="10.20.0.$((10 + p))"
      comma=","
      [ "$r" = "3" ] && comma=""
      echo "      \"$ip:$((65620 + r))\"$comma"
    done
    echo "    ]"
    echo "  },"
    echo "  \"hidden\": 7168,"
    echo "  \"device_collective\": {"
    echo "    \"backend\": \"nccl\","
    echo "    \"backend_module_path\": \"/usr/lib/aarch64-linux-gnu/libnccl.so\","
    echo "    \"local_host\": \"$host\","
    echo "    \"collective_identifier\": 1,"
    echo "    \"listen_port\": 64620,"
    echo "    \"connect_timeout_milli\": 5000,"
    echo "    \"operation_timeout_milli\": 30000,"
    echo "    \"peer_hosts\": ["
    for r in 0 1 2 3; do
      p=$((stage * 4 + r))
      ph=$(printf '%x' "$p")
      ip="10.20.0.$((10 + p))"
      comma=","
      [ "$r" = "3" ] && comma=""
      echo "      \"$ip\"$comma"
    done
    echo "    ]"
    echo "  }"
    echo "}"
  } > "$OUT/$host.json"
  echo "wrote $OUT/$host.json"
done
