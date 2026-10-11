#!/usr/bin/env bash
set -u
. "$(dirname "$0")/rollover_lib.sh"
TAG="$1"
FWT=$D/fw-2aa246576-warm2c5
rm -rf "$FWT" && cp -R /Users/mac/sparkpipe-coord/lanes/driver-sprint/fw-2aa246576 "$FWT" && cp "$FW/weightd_warm" "$FWT/weightd_warm"
APIT=/home/spec/api-build-2aa246576/build
REF3=qualification/qwen38_27b/reference/reference3_fp8_numpy.json
PARK=qwen3.8-max
case "$TAG" in
  q27) SWAP=qwen3.8-27b; M=qwen38_27b; OLD=qwen27b_lp_2fff.json; REF=$REF3 ;;
  qflash) SWAP=qwen3.8-flash; M=qwen38_flash; OLD=flash_lp_new.json; REF=$REF3 ;;
  qmax) SWAP=qwen3.8-max; M=qwen38_max; OLD=qmax_lp_2fff.json; REF=$REF3; PARK=qwen3.8-flash ;;
  d41) SWAP=deepseek-v4.1-flash; M=dsv41_flash; OLD=d41_lp_2fff.json; REF=qualification/dsv41_flash/reference_tokens.json ;;
  k3) SWAP=kimi-k3; M=k3; OLD=golden_t2fff.json ;;
  gflash) SWAP=glm-5.3-flash; M=glm53_flash; OLDPATH=qualification/glm53_flash/golden/port-8d27916a4-warm.json ;;
  *) echo "usage: rollover_attrib.sh q27|qflash|qmax|d41|k3"; exit 2 ;;
esac
Q="qualification/$M/rollover-$V"
OLDPATH="${OLDPATH:-~/rollover/old/$OLD}"
L="$D/attrib-$TAG"
mkdir -p "$L"
cd "$L"

deploy() {
  if [ "$M" = k3 ]; then
    ( . "$W/scripts/rollover_env_k3.sh"; export K3_FIRMWARE="$1"
      bash "$W/tools/k3_lane.sh" setup > "setup_$3.log" 2>&1 || exit 1
      hub "set -e; cd ~/k3-kv-api; rm -f runtime/prefix_index.spi; install -m 755 $2/sparkpipe_model_api bin/sparkpipe_model_api.new && mv bin/sparkpipe_model_api.new bin/sparkpipe_model_api; install -m 755 $2/sparkpipe_tokenize_prompt bin/sparkpipe_tokenize_prompt.new 2>/dev/null && mv bin/sparkpipe_tokenize_prompt.new bin/sparkpipe_tokenize_prompt || true; install -m 755 $2/libk3_tp16_serving_adapter.so runtime/lib/libk3_serving_adapter.so.new && mv runtime/lib/libk3_serving_adapter.so.new runtime/lib/libk3_serving_adapter.so" )
  else
    ( . "$W/scripts/rollover_env_$M.sh"; export STAGE_FIRMWARE="$1" STAGE_API_FIRMWARE="$2"
      bash "$W/tools/stage_lane.sh" setup > "setup_$3.log" 2>&1 && bash "$W/tools/stage_lane.sh" api-setup > "apisetup_$3.log" 2>&1 && hub "rm -f ~/$M-kv-api/runtime/prefix_index.spi" )
  fi
}

capture() {
  if [ "$M" = k3 ]; then
    hub "cd ~/rollover/tree && timeout 1800 python3 scripts/rollover_k3_golden.py 8459 $Q/golden-$1.json > /dev/null"
  elif [ "$M" = glm53_flash ]; then
    hub "cd ~/rollover/tree && timeout 2400 python3 tools/glm53_flash_golden.py --port 8476 --label $1 --output $Q/golden-$1.json > /dev/null 2>&1"
  else
    PORT=$(python3 -c "import sys; sys.path.insert(0, '$W/tools'); import stage_lane; print(stage_lane.MODELS['$M']['api_port'])")
    hub "cd ~/rollover/tree && timeout 1200 python3 scripts/rollover_lp_capture.py $REF $PORT $Q/golden-$1.json > /dev/null"
  fi
}

compare() { hub "cd ~/rollover/tree && python3 scripts/rollover_golden_compare.py $1 $2 0.5 | tail -1 | sed 's/^/$3: /'"; }

stamp "$SWAP attribution precheck"
[ "$(controller)" = "idle qwen3.8-max None" ] || { echo "PRECHECK-FAILED: $(controller)"; exit 1; }
nodes_up || exit 9
[ "$SWAP" = qwen3.8-max ] && { activate "$PARK" || { restore_max; exit 1; }; }
deploy "$FWT" "$APIT" trunc || { echo "TRUNC-DEPLOY-FAILED"; deploy "$FW" "$APIB" back; restore_max; exit 1; }
if activate "$SWAP"; then
  capture trunc-2aa246576
  compare "$OLDPATH" "$Q/golden-trunc-2aa246576.json" "fw-2aa246576 (truncating SUM) vs old golden"
  compare "$Q/golden-trunc-2aa246576.json" "$Q/golden-new.json" "fw-2c5d5b843 (RNE) vs fw-2aa246576"
else
  echo "TRUNC-ACTIVATE-FAILED"
fi
activate "$PARK"
deploy "$FW" "$APIB" back || { echo "RESTORE-DEPLOY-FAILED: lane left on fw-2aa246576"; restore_max; exit 1; }
if activate "$SWAP"; then
  capture recheck
  compare "$Q/golden-new.json" "$Q/golden-recheck.json" "restored fw-2c5d5b843 vs golden-new"
else
  echo "RESTORE-ACTIVATE-FAILED"
fi
restore_max
if [ "$M" = k3 ]; then
  fan "cd ~/k3-lanes/lane8/root && echo \$(sha256sum lib/libk3_serving_adapter.so | cut -c1-16)" | cut -d' ' -f2 | sort | uniq -c
else
  fan "\$HOME/stage-lanes/$M/lane\$(ls \$HOME/stage-lanes/$M | sed -n 's/^lane\([0-9]*\)$/\1/p' | head -1)/root/lane.sh reclaim > /dev/null 2>&1; sha256sum \$HOME/stage-lanes/$M/lane*/root/lib/lib${M}_tp16_serving_adapter.so | cut -c1-16" | cut -d' ' -f2 | sort | uniq -c
fi
echo "fw adapter: $( [ "$M" = k3 ] && fw_sha libk3_tp16_serving_adapter.so || fw_sha lib${M}_tp16_serving_adapter.so)"
nodes_up || exit 9
stamp "attribution done $SWAP ($(controller))"
