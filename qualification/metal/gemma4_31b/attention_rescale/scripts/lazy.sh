#!/bin/sh
set -e
mk() {
	d=$HOME/build/g31$1
	rm -rf $d; cp -R $HOME/build/revs/rev-ff51be851 $d; cd $d
	python3 -I - $2 <<'PY'
import sys
from pathlib import Path
p = Path("inference/kernels/metal/attention.metal")
s = p.read_text()
def rep(old, new):
    global s
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)
rep("#define SPARK_ATTENTION_BLOCK_SIMDS 4u\n", "#define SPARK_ATTENTION_BLOCK_SIMDS 4u\n#define SPARK_ATTENTION_RESCALE_MARGIN 8.0f\n")
rep("""		const float next_max = max(running_max, score);
		const float correction = exp(running_max - next_max), weight = exp(score - next_max);""",
    """		const float next_max = score > running_max + SPARK_ATTENTION_RESCALE_MARGIN ? score : running_max;
		const float correction = next_max == running_max ? 1.0f : exp(running_max - next_max), weight = exp(score - next_max);""")
rep("""			const float previous = row_max[r], next = max(previous, simd_max(max(low_value, high_value)));
			const float correction = next == -INFINITY ? 1.0f : exp(previous - next);""",
    """			const float previous = row_max[r], tile_max = simd_max(max(low_value, high_value));
			const float next = tile_max > previous + SPARK_ATTENTION_RESCALE_MARGIN ? tile_max : previous;
			const float correction = next == previous ? 1.0f : exp(previous - next);""")
for which in sys.argv[1:]:
    if which == "simd":
        rep("	for ( uint s = 0u; s < SPARK_ATTENTION_SIMDS; s++ )\n	{\n		const float factor",
            "	for ( uint t = 0u; t < SPARK_ATTENTION_SIMDS; t++ )\n	{\n		const uint s = SPARK_ATTENTION_SIMDS - 1u - t;\n		const float factor")
    if which == "comb":
        rep("		for ( uint part = 0u; part < used; part++ )\n		{\n			device const float *record = records + part * (D + 2u);",
            "		for ( uint back = 0u; back < used; back++ )\n		{\n			device const float *record = records + (used - 1u - back) * (D + 2u);")
p.write_text(s)
PY
	rm -f build/metal/inference/kernels/metal/attention.air build/runtime_libs/sparkpipe_kernels.metallib
	make -s build/metal_stage_run build/logit_compare build/test_metal_kernels 2>&1 | grep -i error || true
}
mk lazy plain
mk lsimd simd
mk lcomb comb
cd $HOME/build/g31lazy && ./build/test_metal_kernels > /tmp/tk_lazy.log 2>&1; echo "lazy kernels exit $? pass $(grep -c ^PASS /tmp/tk_lazy.log)" >> $HOME/build/lazy_summary.txt
for v in lazy lsimd lcomb; do
	rm -rf /tmp/leg_$v
	sh $HOME/build/leg31.sh $HOME/build/g31$v /tmp/leg_$v > /tmp/leg_$v.txt 2>&1 || echo "leg $v failed" >> /tmp/leg_$v.txt
	echo "$v: $(cat /tmp/leg_$v.txt | tr '\n' ' ')" >> $HOME/build/lazy_summary.txt
done
echo done >> $HOME/build/lazy_summary.txt
