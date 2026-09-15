#!/bin/sh
# Rebuild all 16 qwen38max TP16 packs under the CEPH lease, swap them into the
# placed roots, re-verify each. Sequential ranks: the lease serializes ceph.
set -eu
LEASE=/Users/mac/sparkpipe-coord/CEPH_LEASE
HOSTS="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf"

say() {
	printf '[q38rebuild %s] %s\n' "$(date -u +%H:%M:%S)" "$*"
}

rank_host() {
	echo $HOSTS | cut -d' ' -f$(($1 + 1))
}

ceph_claim() {
	grep -qi "release" "$LEASE" || { say "ceph lease not free"; exit 1; }
	echo "holder: T1-QMAX | claimed_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ) | purpose: qwen38max TP16 pack rebuild from warm (fixed nvfp4 codec), sequential rank reads on spark0" >> "$LEASE"
}

ceph_release() {
	echo "released_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ) (T1-QMAX rebuild complete, 16/16 verified)" >> "$LEASE"
}

ceph_claim
for rank in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
	host=$(rank_host "$rank")
	ssh -o BatchMode=yes "$host" "mkdir -p /tmp/t1qmax_stage/tools /tmp/t1qmax_rebuild"
	scp -q tools/qwen38_stagepack.py tools/spark_pack_common.py tools/qwen38max_tp16_rank_verify.py tools/qwen38max_rebuild_rank.sh tools/qwen38max_swap_rank.sh "$host:/tmp/t1qmax_stage/tools/"
	scp -q spark7:/tmp/t1qmax_stage/qwen38max_experts_manifest "$host:/tmp/t1qmax_stage/tools/"
done
for rank in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
	host=$(rank_host "$rank")
	say "rebuild+swap rank $rank on $host"
	ssh -o BatchMode=yes "$host" "sh /tmp/t1qmax_stage/tree/tools/qwen38max_rebuild_rank.sh $rank" || {
		say "REBUILD FAIL rank $rank"
		ceph_release
		exit 1
	}
	ssh -o BatchMode=yes "$host" "sh /tmp/t1qmax_stage/tree/tools/qwen38max_swap_rank.sh $rank" || {
		say "SWAP FAIL rank $rank"
		ceph_release
		exit 1
	}
done
ceph_release
say "16/16 rebuilt, swapped, verified"
