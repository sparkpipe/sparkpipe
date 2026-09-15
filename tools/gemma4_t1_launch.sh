#!/bin/bash
# Launch one gemma4-26b TP4xPP4 T1 lane rank on this node.
# usage: gemma4_t1_launch.sh <rank 0-15>   (rank 0 also starts the api)
set -u
RANK="${1:?rank 0-15}"
ROOT="$HOME/sparkdata/gemma4_26b.tp4pp4.t1"
cd "$ROOT" || exit 2
mkdir -p logs
export SPARK_GEMMA4_T1=1
eval "$(python3 -c 'import json,sys;e=json.load(open("config/env_%02d.json" % int(sys.argv[1])));print("\n".join("export %s=%s"%(k,v) for k,v in e.items()))' "$RANK")"
if [ -f config/env_local_override.sh ]; then
    . ./config/env_local_override.sh
fi
nohup "$ROOT/bin/sparkpipe_model_residentd" \
    --deployment "$ROOT/model_resident.json" \
    --rank-index "$RANK" \
    > "logs/residentd_$RANK.log" 2>&1 &
echo "residentd rank=$RANK pid=$!"
if [ "$RANK" = "0" ]; then
    nohup "$ROOT/bin/sparkpipe_model_api" \
        --deployment "$ROOT/model_resident.json" \
        --runtime-root "$ROOT" \
        --port "${GEMMA4_T1_API_PORT:-16000}" \
        > logs/api.log 2>&1 &
    echo "api pid=$!"
fi
