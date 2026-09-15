#!/usr/bin/env bash
# Place the re-emitted hy4 v2 rank packs per the TP16 replica law:
# rank r -> spark{hex r}:~/sparkdata/hy4.fp8.tp16/packs/rank-XX/.
# Stage -> sha gate -> on-node fingerprint old-vs-new -> atomic swap ->
# v1 trash purge. Run from the emit node (~/hy4-v2-emit).
set -uo pipefail
EMIT=${EMIT_NODE:-spark3}
SRC="$EMIT:hy4-v2-emit/packs"
BASE="sparkdata/hy4.fp8.tp16"
STAGE="staging-v2"
TRASH="v1-trash-$(date -u +%Y%m%dT%H%M%SZ)"
FP_TOOL="tools/hy4_v2_fingerprint.py"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=8"
rank_node() {
	local r=$1
	if [ "$r" -lt 10 ]; then echo "spark$r"; else printf "spark%x" "$r"; fi
}
one() {
	local r=$1
	local xx
	xx=$(printf "%02d" "$r")
	local node
	node=$(rank_node "$r")
	echo "== rank $xx -> $node $(date -u +%H:%M:%SZ)"
	if ! $SSH "$node" "df --output=avail -BG ~ | tail -1 | tr -d G" > /tmp/hy4_free_$$; then
		echo "PLACE-FAIL $xx unreachable"; return 1
	fi
	local free
	free=$(cat /tmp/hy4_free_$$)
	rm -f /tmp/hy4_free_$$
	if [ "$free" -lt 120 ]; then
		echo "PLACE-FAIL $xx disk ${free}G < 120G"; return 1
	fi
	$SSH "$node" "mkdir -p ~/$BASE/$STAGE/rank-$xx"
	for f in model-fp8-tp16-rank-$xx.safetensors \
		model-fp8-tp16-rank-$xx.safetensors.sha256 \
		model-fp8-tp16-rank-$xx.safetensors.experts \
		manifest-rank-$xx.json; do
		scp -q "$SRC/rank-$xx/$f" "$node:$BASE/$STAGE/rank-$xx/$f" || {
			echo "PLACE-FAIL $xx scp $f"; return 1
		}
	done
	$SSH "$node" "cd ~/$BASE/$STAGE/rank-$xx && sha256sum -c model-fp8-tp16-rank-$xx.safetensors.sha256" \
		|| { echo "PLACE-FAIL $xx sha gate"; return 1; }
	scp -q "$FP_TOOL" "$node:$BASE/$STAGE/rank-$xx/fingerprint.py" || return 1
	$SSH "$node" "python3 $BASE/$STAGE/rank-$xx/fingerprint.py --rank $r \
		--old-dir $BASE/packs/rank-$xx --new-dir $BASE/$STAGE/rank-$xx" \
		|| { echo "PLACE-FAIL $xx fingerprint"; return 1; }
	$SSH "$node" "cd ~/$BASE && mkdir -p $TRASH/rank-$xx && \
		mv packs/rank-$xx/model-fp8-tp16-rank-$xx.safetensors \
		packs/rank-$xx/model-fp8-tp16-rank-$xx.safetensors.sha256 \
		packs/rank-$xx/model-fp8-tp16-rank-$xx.safetensors.experts \
		packs/rank-$xx/manifest-rank-$xx.json $TRASH/rank-$xx/ && \
		mv $STAGE/rank-$xx/model-fp8-tp16-rank-$xx.safetensors \
		$STAGE/rank-$xx/model-fp8-tp16-rank-$xx.safetensors.sha256 \
		$STAGE/rank-$xx/model-fp8-tp16-rank-$xx.safetensors.experts \
		$STAGE/rank-$xx/manifest-rank-$xx.json packs/rank-$xx/ && \
		rm -rf $STAGE/rank-$xx && \
		cd packs/rank-$xx && sha256sum -c model-fp8-tp16-rank-$xx.safetensors.sha256" \
		|| { echo "PLACE-FAIL $xx swap"; return 1; }
	echo "PLACED $xx $node $(date -u +%H:%M:%SZ)"
}
ranks="${RANKS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15}"
for r in $ranks; do
	one "$r" || exit 1
done
echo "PLACE-ALL-DONE trash=$TRASH (per-node, delete after the serve gate) $(date -u +%H:%M:%SZ)"
