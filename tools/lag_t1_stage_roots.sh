#!/bin/sh
set -eu

ARM=laguna-s-2.1.bf16.tp8pp2
BUILD_NODE=sparkf
BUILD_TREE=$HOME/lagt1-build
OUT_TREE=$HOME/sparkdata/out
STAGE=/tmp/lagt1_stage

ssh "$BUILD_NODE" 'rm -rf /tmp/lagt1_stage && mkdir -p /tmp/lagt1_stage/laguna-s-2.1.bf16.tp8pp2/bin /tmp/lagt1_stage/laguna-s-2.1.bf16.tp8pp2/lib /tmp/lagt1_stage/laguna-s-2.1.bf16.tp8pp2/stages/stage_000 && \
	cp ~/lagt1-build/build/sparkpipe_model_residentd ~/lagt1-build/build/sparkpipe_model_api /tmp/lagt1_stage/laguna-s-2.1.bf16.tp8pp2/bin/ && \
	cp ~/lagt1-build/build/modules/laguna_resident_decode_stage/bf16/liblaguna_serving_adapter_bf16.so /tmp/lagt1_stage/laguna-s-2.1.bf16.tp8pp2/lib/model_serving_adapter.so && \
	cp ~/lagt1-build/build/libhidden_transport_spark_host_rdma_verbs.so /tmp/lagt1_stage/laguna-s-2.1.bf16.tp8pp2/lib/hidden_transport.so && \
	cp ~/sparkdata/out/stages/stage_000/model_driver.so /tmp/lagt1_stage/laguna-s-2.1.bf16.tp8pp2/stages/stage_000/ && \
	chmod 755 /tmp/lagt1_stage/laguna-s-2.1.bf16.tp8pp2/bin/*'

rm -rf /tmp/lagt1_config && mkdir -p /tmp/lagt1_config
cp -R deployment/$ARM/config /tmp/lagt1_config/config
cp deployment/$ARM/model_resident.json /tmp/lagt1_config/
scp -q -r /tmp/lagt1_config/config "$BUILD_NODE:$STAGE/$ARM/"
scp -q /tmp/lagt1_config/model_resident.json "$BUILD_NODE:$STAGE/$ARM/"

ssh "$BUILD_NODE" "tar -C $STAGE -czf /tmp/lagt1_root.tgz $ARM"

for rank in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
	host="spark$(printf '%x' "$rank")"
	ssh -o BatchMode=yes "$host" "
		set -eu
		root=\$HOME/$ARM
		mkdir -p \"\$root\" \$HOME/kvcache/$ARM
		rm -rf \"\$root/bin\" \"\$root/lib\" \"\$root/stages\" \"\$root/config\" \"\$root/model_resident.json\"
	"
	scp -q "$BUILD_NODE:/tmp/lagt1_root.tgz" "$host:/tmp/lagt1_root.tgz"
	ssh -o BatchMode=yes "$host" "tar -xzf /tmp/lagt1_root.tgz -C \$HOME/ && chmod 755 \$HOME/$ARM/bin/* && rm -f /tmp/lagt1_root.tgz && ls \$HOME/$ARM"
	echo "staged $host"
done
