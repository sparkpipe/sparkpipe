from pathlib import Path
import sys
s = Path("tests/test_metal_kernels.c").read_text()
def rep(old, new):
    global s
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)
s = s.replace("int main(int argc, char **argv)\n{", "int unused_main(int argc, char **argv);\nint unused_main(int argc, char **argv)\n{", 1)
rep("static void AttentionFill(AttentionFixture *f, uint32_t context, uint32_t sequences)\n{",
    "static double ramp = 0.0;\nstatic double exact_value[1024], exact_magnitude[1024];\nstatic void AttentionFill(AttentionFixture *f, uint32_t context, uint32_t sequences)\n{")
rep("""	FillBf16((uint16_t *)Shared(&f->keys,(uint64_t)sequences * context * att.kv * att.dim * 2u,"attention keys"),(uint64_t)sequences * context * att.kv * att.dim,1.0);
""", """	FillBf16((uint16_t *)Shared(&f->keys,(uint64_t)sequences * context * att.kv * att.dim * 2u,"attention keys"),(uint64_t)sequences * context * att.kv * att.dim,1.0);
	{
		uint16_t *k = (uint16_t *)f->keys.host;
		for ( row = 0u; row < sequences * context; row++ )
			for ( s = 0u; s < att.kv; s++ )
				k[((uint64_t)row * att.kv + s) * att.dim] = ToBf16((float)(ramp * (double)(row % context) / context));
	}
""")
rep("""		const double attended = RoundBf16(value), g""", """		exact_value[d] = value;
		exact_magnitude[d] = magnitude;
		const double attended = RoundBf16(value), g""")
s += r'''
static void Measure(uint32_t context, uint32_t rows, uint32_t prefill, double ramp_value, const char *label)
{
	AttentionFixture f;
	SparkAttentionParams params;
	uint16_t *query, *gate;
	uint32_t *sequence, *position, row, head, d, part;
	double ref[ATT_MAX_DIM], tolerance[ATT_MAX_DIM], sq = 0.0, worst = 0.0, signed_sum = 0.0;
	uint64_t count = 0u;
	const uint32_t sequences = prefill != 0u ? 2u : 1u, keys = prefill != 0u ? 1024u : 128u;
	ramp = ramp_value;
	memset(&f,0,sizeof(f));
	AttentionFill(&f,context,sequences);
	query = (uint16_t *)Shared(&f.query,(uint64_t)rows * att.heads * att.dim * 2u,"q");
	gate = (uint16_t *)Shared(&f.gate,(uint64_t)rows * att.qkv * 2u,"g");
	Shared(&f.output,(uint64_t)rows * att.heads * att.dim * 2u,"o");
	Shared(&f.partial,SparkOpsAttentionScratchBytes(rows,att.heads,att.dim,context),"partial");
	FillBf16(query,(uint64_t)rows * att.heads * att.dim,att.scale < 0.5f ? 1.0 : 1.0 / 16.0);
	for ( row = 0u; row < rows; row++ )
		for ( head = 0u; head < att.heads; head++ )
			query[((uint64_t)row * att.heads + head) * att.dim] = ToBf16(1.0f);
	FillBf16(gate,(uint64_t)rows * att.qkv,1.0);
	SparkDeviceFree(&f.sequence);
	SparkDeviceFree(&f.positions);
	sequence = (uint32_t *)Shared(&f.sequence,rows * 4u,"s");
	position = (uint32_t *)Shared(&f.positions,rows * 4u,"p");
	for ( row = 0u; row < rows; row++ )
	{
		sequence[row] = prefill != 0u ? (row < rows / 2u ? 0u : 1u) : 0u;
		position[row] = prefill != 0u ? context - (rows / 2u) + (row % (rows / 2u)) : context - 1u - row;
	}
	f.paged.sequence_of_row = Whole(&f.sequence);
	f.paged.positions = Whole(&f.positions);
	memset(&params,0,sizeof(params));
	params.rows = rows; params.heads = att.heads; params.kv_heads = att.kv; params.head_dimension = att.dim;
	params.page_slots = ATT_SLOTS; params.pages_per_sequence = f.pages; params.pool_pages = sequences * f.pages;
	params.query_stride = att.heads * att.dim; params.gate_stride = att.qkv; params.output_stride = att.heads * att.dim;
	params.scale = att.scale; params.output_gate = 0u; params.window = att.window; params.gate_offset = att.dim; params.gate_head_stride = 2u * att.dim;
	Run(SparkOpsAttention(&ops,queue,&f.paged,Whole(&f.query),Whole(&f.gate),Whole(&f.partial),Whole(&f.output),&params,position,sequence),"attention");
	SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER);
	{
		const float *partial = (const float *)f.partial.host;
		const uint32_t partitions = (context + keys - 1u) / keys;
		const uint32_t saved_gate = att.gate;
		att.gate = 0u;
		for ( row = 0u; row < rows; row++ )
			for ( head = 0u; head < att.heads; head++ )
			{
				const uint32_t used = (position[row] + keys) / keys;
				const float *records = partial + ((uint64_t)row * att.heads + head) * partitions * (att.dim + 2u);
				double top = -INFINITY, den = 0.0;
				AttentionReference(&f,query + (uint64_t)row * att.heads * att.dim,gate + (uint64_t)row * att.qkv,sequence[row],position[row],head,ref,tolerance);
				for ( part = 0u; part < used; part++ )
					if ( records[part * (att.dim + 2u)] > top )
						top = records[part * (att.dim + 2u)];
				for ( part = 0u; part < used; part++ )
					if ( records[part * (att.dim + 2u)] != -INFINITY )
						den += exp((double)records[part * (att.dim + 2u)] - top) * records[part * (att.dim + 2u) + 1u];
				for ( d = 0u; d < att.dim; d++ )
				{
					double num = 0.0;
					for ( part = 0u; part < used; part++ )
						if ( records[part * (att.dim + 2u)] != -INFINITY )
							num += exp((double)records[part * (att.dim + 2u)] - top) * records[part * (att.dim + 2u) + 2u + d];
					const double e = (num / den - exact_value[d]) / exact_magnitude[d] / UNIT_F32;
					sq += e * e;
					signed_sum += e;
					if ( fabs(e) > worst )
						worst = fabs(e);
					count++;
				}
			}
		att.gate = saved_gate;
	}
	printf("%-10s %s rows %3u context %5u %u/%u x %u ramp %4.0f: record error / (2^-24 sum p|v|): rms %.2f max %.1f mean %+.3f\n",label,
		prefill != 0u ? "prefill" : "decode ",rows,context,att.heads,att.kv,att.dim,ramp_value,sqrt(sq / count),worst,signed_sum / count);
	AttentionFree(&f);
	SparkDeviceFree(&f.query);
	SparkDeviceFree(&f.gate);
	SparkDeviceFree(&f.output);
	SparkDeviceFree(&f.partial);
}

int main(int argc, char **argv)
{
	const char *label = argc > 1 ? argv[1] : "build";
	const AttentionCase narrow = { 32u, 16u, 256u, 16384u, 12288u, 0u, 0u, 1.0f }, wide = { 32u, 4u, 512u, 20480u, 18432u, 0u, 0u, 1.0f };
	double ramps[3] = { 0.5, 2.0, 8.0 };
	uint32_t r;
	if ( SparkDeviceInitialize(0) != SPARK_STATUS_OK || SparkDeviceQueueCreate(&queue,"acc") != SPARK_STATUS_OK || SparkDeviceOpsOpen(&ops,SPARK_KERNEL_LIBRARY) != SPARK_STATUS_OK )
		return(1);
	for ( r = 0u; r < 3u; r++ )
	{
		att = narrow;
		Measure(8192u,4u,0u,ramps[r],label);
		Measure(8192u,64u,1u,ramps[r],label);
		att = wide;
		Measure(8192u,4u,0u,ramps[r],label);
		Measure(8192u,32u,1u,ramps[r],label);
	}
	return(0);
}
'''
Path("build/attn_accuracy.c").write_text(s)
