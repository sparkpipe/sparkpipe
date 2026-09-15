#!/bin/sh
set -eu

ARM=laguna-s-2.1.bf16.tp8pp2
STAGING=/mnt/model-warm/packbuild/laguna/real_packs
STAGE0_SHA=151ced0f6aa76241c6c37cab04f70f6121f02e097f72103740be3f46992ca057
STAGE1_SHA=3eb373d76955248ae17a9438825cdc63923cc3be197ac6df76e278eb51e9b535
RECEIPT=/tmp/lagt1_placement_receipt.jsonl
: > "$RECEIPT"

rank_hex() {
	printf '%x' "$1"
}

place_rank() {
	rank=$1
	stage=$((rank / 8))
	tp_rank=$((rank % 8))
	global=$(rank_hex "$rank")
	host="spark$global"
	staged_src="$STAGING/laguna_stage.tp8.pp2.stage$stage.rank$tp_rank.lgsp"
	placed="laguna_stage.tp8.pp2.stage$stage.rank$rank.lgsp"
	case "$stage" in
	0) expect=$STAGE0_SHA ;;
	1) expect=$STAGE1_SHA ;;
	esac
	if [ "$tp_rank" != 0 ]; then
		expect=""
	fi
	ssh -o BatchMode=yes "$host" "
		set -eu
		root=\$HOME/$ARM/packs
		mkdir -p \"\$root\"
		tmp=\"\$root/$placed.tmp$$\"
		final=\"\$root/$placed\"
		rm -f \"\$tmp\"
		sparkcap cp '$staged_src' \"\$tmp\"
		src_digest=\$(sparkcap sha256sum '$staged_src' | cut -d' ' -f1)
		placed_digest=\$(sparkcap sha256sum \"\$tmp\" | cut -d' ' -f1)
		if [ \"\$src_digest\" != \"\$placed_digest\" ]; then
			echo \"DIGEST MISMATCH $host src=\$src_digest placed=\$placed_digest\" >&2
			exit 1
		fi
		if [ -n '$expect' ] && [ \"\$placed_digest\" != '$expect' ]; then
			echo \"ANCHOR MISMATCH $host got=\$placed_digest want=$expect\" >&2
			exit 1
		fi
		mv \"\$tmp\" \"\$final\"
		printf '%s  %s\n' \"\$placed_digest\" \"$placed\" > \"\$root/$placed.sha256\"
		bytes=\$(stat -c %s \"\$final\")
		printf '{\"host\":\"$host\",\"rank\":$rank,\"stage\":$stage,\"tp_rank\":$tp_rank,\"source\":\"$staged_src\",\"placed\":\"\$HOME/$ARM/packs/$placed\",\"bytes\":%s,\"sha256\":\"%s\",\"sidecar\":\"%s.sha256\"}\n' \"\$bytes\" \"\$placed_digest\" \"$placed\"
	" >> "$RECEIPT"
	echo "rank $rank on $host placed"
}

for rank in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
	place_rank "$rank" &
done
fail=0
for pid in $(jobs -p); do
	wait "$pid" || fail=1
done
if [ "$fail" != 0 ]; then
	echo "PLACEMENT INCOMPLETE" >&2
	exit 1
fi
echo "== receipt =="
cat "$RECEIPT"
wc -l "$RECEIPT"
