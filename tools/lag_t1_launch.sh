#!/bin/sh
set -eu

ARM=laguna-s-2.1.bf16.tp8pp2
PROMPT_IDS=$1
NEW_TOKENS=$2
API_PORT=18733

stop_mine() {
	host=$1
	ssh -o BatchMode=yes "$host" "
		set -eu
		for pid in \$(pgrep -x sparkpipe_model_residentd || true; pgrep -x sparkpipe_model_api || true); do
			cwd=\$(readlink /proc/\$pid/cwd 2>/dev/null || true)
			case \"\$cwd\" in
			*/\$HOME/$ARM|*\$HOME/$ARM) kill -TERM \$pid 2>/dev/null || true ;;
			esac
		done
		sleep 2
	"
}

for rank in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
	host="spark$(printf '%x' "$rank")"
	stop_mine "$host"
	ssh -o BatchMode=yes "$host" "
		set -eu
		root=\$HOME/$ARM
		mkdir -p \$root/runs
		cd \$root/config && ln -sf stage_$(printf '%02d' "$rank").json stage.json
		cd \$root
		SPARK_WEIGHTD_SOCKET=/run/sparkpipe-weightsd/weightsd.sock \
		SPARK_LAGUNA_T1=1 \
		nohup bin/sparkpipe_model_residentd --deployment config/model_resident.json --rank-index $rank \
			> \$root/runs/residentd_rank$rank.log 2>&1 &
		echo \$! > \$root/runs/residentd_rank$rank.pid
		echo launched $host
	"
done

sleep 1
stop_mine spark0
ssh -o BatchMode=yes spark0 "
	set -eu
	root=\$HOME/$ARM
	mkdir -p \$root/runs
	cd \$root
	SPARK_WEIGHTD_SOCKET=/run/sparkpipe-weightsd/weightsd.sock \
	nohup bin/sparkpipe_model_api --deployment config/model_resident.json --runtime-root . --port $API_PORT \
		> \$root/runs/api.log 2>&1 &
	echo \$! > \$root/runs/api.pid
	echo api launched on spark0 port $API_PORT
"

echo "== decode =="
curl -s -X POST "http://spark0:$API_PORT/v1/completions" \
	-H 'Content-Type: application/json' \
	-d "{\"prompt_token_ids\": [$PROMPT_IDS], \"max_tokens\": $NEW_TOKENS}" || true
