#!/bin/sh
set -e
mk() {
	d=$HOME/build/g31$1
	rm -rf $d; cp -R $3 $d; cd $d
	python3 -I - "$2" <<'PY'
import sys
from pathlib import Path
p = Path("inference/kernels/metal/attention.metal")
s = p.read_text()
which = sys.argv[1]
options = {
    "nsimd": ("	for ( uint s = 0u; s < SPARK_ATTENTION_SIMDS; s++ )\n	{\n		total = SparkMathCompensatedAdd(total, shared_sum[s]);",
              "	for ( uint t = 0u; t < SPARK_ATTENTION_SIMDS; t++ )\n	{\n		const uint s = SPARK_ATTENTION_SIMDS - 1u - t;\n		total = SparkMathCompensatedAdd(total, shared_sum[s]);"),
    "osimd": ("	for ( uint s = 0u; s < SPARK_ATTENTION_SIMDS; s++ )\n	{\n		const float factor",
              "	for ( uint t = 0u; t < SPARK_ATTENTION_SIMDS; t++ )\n	{\n		const uint s = SPARK_ATTENTION_SIMDS - 1u - t;\n		const float factor"),
    "comb": ("		for ( uint part = 0u; part < used; part++ )\n		{\n			device const float *record = records + part * (D + 2u);",
             "		for ( uint back = 0u; back < used; back++ )\n		{\n			device const float *record = records + (used - 1u - back) * (D + 2u);"),
}
old, new = options[which]
assert s.count(old) == 1, which
s = s.replace(old, new)
p.write_text(s)
PY
	touch inference/kernels/metal/attention.metal
	make -s build/metal_stage_run build/logit_compare 2>&1 | grep -i error || true
}
mk nsimd2 nsimd $HOME/build/g31ns
mk ncomb2 comb $HOME/build/g31ns
mk osimd osimd $HOME/build/revs/rev-ff51be851
mk ocomb comb $HOME/build/revs/rev-ff51be851
rm -f $HOME/build/legs_summary.txt
for v in ns nsimd2 ncomb2 osimd ocomb; do
	mkdir -p $HOME/build/g31$v/qualification/metal/gemma4_31b
	rm -rf /tmp/leg_$v
	sh $HOME/build/leg31.sh $HOME/build/g31$v /tmp/leg_$v > /tmp/leg_$v.txt 2>&1 || echo "leg $v failed" >> /tmp/leg_$v.txt
	echo "$v: $(cat /tmp/leg_$v.txt | tr '\n' ' ')" >> $HOME/build/legs_summary.txt
done
echo done >> $HOME/build/legs_summary.txt
