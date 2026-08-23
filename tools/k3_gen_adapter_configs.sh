#!/usr/bin/env bash
# k3_gen_adapter_configs.sh OUT_DIR [TP_DEGREE] — emit the per-rank serving
# adapter configs (rank <host>.json) with the pack paths, the world size, the
# optional host TCP TP collective (TP4) and the device-direct collective.
# TP_DEGREE 4 = TP4xPP4 (four PP stages of four ranks), 16 = TP16 (one PP
# stage of sixteen ranks). The ring's static IPs are 10.20.0.(10+i) for spark
# index i (0..9, a..f). Every rank shares one NCCL control port: the fetchers
# dial rank 0's host on their OWN control_port_base value.
set -euo pipefail
OUT="${1:?usage: k3_gen_adapter_configs.sh OUT_DIR [TP_DEGREE]}"
TP="${2:-4}"
case "$TP" in
  4) WORLD=16 ;;
  16) WORLD=16 ;;
  *) echo "unsupported tp degree $TP (4 or 16)" >&2; exit 1 ;;
esac
STAGES=$((WORLD / TP))
RT_ROOT=sparkdata/k3.mxfp4.tp4pp4
[ "$TP" = "16" ] && RT_ROOT=sparkdata/k3.mxfp4.tp16
mkdir -p "$OUT"
for i in $(seq 0 15); do
  hex=$(printf '%x' "$i")
  host="spark$hex"
  stage=$((i / TP))
  rank=$((i % TP))
  if [ "$TP" = "4" ]; then
    pack="/home/$host/sparkdata/k3.mxfp4.tp4pp4/packs/k3.stage${stage}.rank0${rank}.pack"
  else
    # Pack names follow tools/k3_deploy_tp16.sh / k3_tp16_pack_production.sh
    # (k3.tp16.rankNN.pack), NOT the TP4 stage naming.
    pack="/home/$host/sparkdata/k3.mxfp4.tp16/packs/k3.tp16.rank$(printf '%02d' "$rank").pack"
  fi
  {
    echo "{"
    echo "  \"stage_pack_path\": \"$pack\","
    echo "  \"tp_degree\": $TP,"
    echo "  \"tp_rank\": $rank,"
    echo "  \"world_size\": $WORLD,"
    echo "  \"max_sequences\": 16,"
    echo "  \"max_rows\": 16,"
    echo "  \"resident_capacity\": 16,"
    echo "  \"kv_pages\": 2,"
    echo "  \"capture_graphs\": 1,"
    echo "  \"hidden\": 7168,"
    echo "  \"device_collective\": {"
    echo "    \"backend\": \"nccl\","
    echo "    \"backend_module_path\": \"lib/runtime_libs/libnccl.so.2\","
    echo "    \"local_host\": \"$host\","
    echo "    \"collective_identifier\": 1,"
    echo "    \"listen_port\": 64620,"
    # 45 s: ranks reach collective-create at very different times after
    # loading multi-GB packs (measured 2026-08-23: 8 s expired -> INIT FAIL 4
    # IO_ERROR on the TP4 slice gate; TP16 rank packs are ~25 GB each).
    echo "    \"connect_timeout_milli\": 45000,"
    echo "    \"operation_timeout_milli\": 30000,"
    echo "    \"peer_hosts\": ["
    first=1
    for r in $(seq 0 $((TP - 1))); do
      p=$((stage * TP + r))
      ph=$(printf '%x' "$p")
      ip="10.20.0.$((10 + p))"
      comma=","
      [ "$r" = "$((TP - 1))" ] && comma=""
      echo "      \"$ip\"$comma"
    done
    echo "    ]"
    echo "  }"
    if [ "$TP" = "4" ]; then
      echo "  ,"
      echo "  \"tp_collective\": {"
      echo "    \"listen_port\": $((61620 + rank)),"
      echo "    \"connect_timeout_milli\": 5000,"
      echo "    \"operation_timeout_milli\": 30000,"
      echo "    \"collective_identifier\": 1,"
      echo "    \"peers\": ["
      # STEP-ordered: peers[s] pairs with tp_rank ^ (1 << s) - the
      # collective dials its step partners by array index, not rank
      for s in 0 1; do
        partner=$((rank ^ (1 << s)))
        p=$((stage * 4 + partner))
        ip="10.20.0.$((10 + p))"
        comma=","
        [ "$s" = "1" ] && comma=""
        echo "      \"$ip:$((61620 + partner))\"$comma"
      done
      echo "    ]"
      echo "  }"
    fi
    echo "}"
  } > "$OUT/$host.json"
  echo "wrote $OUT/$host.json"
done