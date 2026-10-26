#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_device.h"
#include "inference/runner/device_ops.h"

#define UNIT_BF16 (1.0 / 256.0)
#define UNIT_F32 (1.0 / 16777216.0)
#define FUNCTION_ERROR (16.0 * UNIT_F32)

static int failures;
static SparkDeviceOps ops;
static SparkDeviceQueue *queue;
static FILE *receipt;
static uint64_t rng_state = 0x9e3779b97f4a7c15ull;

static uint64_t NextRandom(void)
{
	rng_state ^= rng_state << 13;
	rng_state ^= rng_state >> 7;
	rng_state ^= rng_state << 17;
	return(rng_state);
}

static double Uniform(void)
{
	return((double)(NextRandom() >> 11) / 9007199254740992.0);
}

static double Normal(void)
{
	double u = Uniform(), v = Uniform();
	if ( u < 1e-300 )
		u = 1e-300;
	return(sqrt(-2.0 * log(u)) * cos(6.283185307179586 * v));
}

static float Bf16(uint16_t bits)
{
	uint32_t word = (uint32_t)bits << 16;
	float value;
	memcpy(&value,&word,4);
	return(value);
}

static uint16_t ToBf16(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,4);
	if ( (bits & 0x7fffffffu) > 0x7f800000u )
		return((uint16_t)((bits >> 16) | 0x40u));
	return((uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16));
}

static double RoundBf16(double value)
{
	return(Bf16(ToBf16((float)value)));
}

static double Gamma(double terms)
{
	return(terms * UNIT_F32 / (1.0 - terms * UNIT_F32));
}

static double FlipSlack(double exact, double relative)
{
	const double rounded = RoundBf16(exact), low = RoundBf16(exact * (1.0 - relative)), high = RoundBf16(exact * (1.0 + relative));
	return(fmax(fabs(low - rounded),fabs(high - rounded)));
}

static double Silu(double x)
{
	return(x / (1.0 + exp(-x)));
}

static double Sigmoid(double x)
{
	return(1.0 / (1.0 + exp(-x)));
}

static void *Shared(SparkDeviceBuffer *buffer, uint64_t bytes, const char *owner)
{
	if ( SparkDeviceAllocate(buffer,SPARK_DEVICE_SPACE_SHARED_COHERENT,bytes,owner) != SPARK_STATUS_OK )
	{
		printf("FAIL allocate %s (%llu bytes): %s\n",owner,(unsigned long long)bytes,SparkDeviceLastError()->text);
		exit(1);
	}
	return(buffer->host);
}

static SparkTensor Whole(const SparkDeviceBuffer *buffer)
{
	return(SparkTensorAt(buffer,0u,buffer->bytes));
}

static void FillBf16(uint16_t *values, uint64_t count, double scale)
{
	uint64_t index;
	for ( index = 0u; index < count; index++ )
		values[index] = ToBf16((float)(Normal() * scale));
}

static int Run(SparkStatus status, const char *what)
{
	if ( status != SPARK_STATUS_OK )
	{
		printf("FAIL %s: status %d\n",what,(int)status);
		failures++;
		return(0);
	}
	status = SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER);
	if ( status != SPARK_STATUS_OK )
	{
		printf("FAIL %s: wait %d %s\n",what,(int)status,SparkDeviceLastError()->text);
		failures++;
		return(0);
	}
	return(1);
}

typedef struct Check
{
	double worst;
	double worst_error;
	double worst_tolerance;
	uint64_t count;
	uint64_t bad;
} Check;

static void Observe(Check *check, double got, double expected, double tolerance)
{
	const double error = fabs(got - expected);
	const double ratio = tolerance > 0.0 ? error / tolerance : (error > 0.0 ? INFINITY : 0.0);
	if ( !(ratio <= 1.0) )
		check->bad++;
	if ( ratio > check->worst || isnan(ratio) )
	{
		check->worst = isnan(ratio) ? INFINITY : ratio;
		check->worst_error = error;
		check->worst_tolerance = tolerance;
	}
	check->count++;
}

static const char *JsonNumber(double value, char *text, size_t bytes)
{
	if ( isfinite(value) )
		snprintf(text,bytes,"%.6g",value);
	else
		snprintf(text,bytes,"null");
	return(text);
}

static void Report(const char *op, const char *shape, const Check *clean, const Check *corrupted, const char *tolerance)
{
	char ratio[32], control_ratio[32];
	const int clean_pass = clean->bad == 0u && clean->count != 0u;
	const int control_fails = corrupted->bad != 0u;
	printf("%s %-22s %-44s checked %9llu worst error/tolerance %.3f; corrupted input: %llu outside tolerance (%s)\n",
		clean_pass && control_fails ? "PASS" : "FAIL",op,shape,(unsigned long long)clean->count,clean->worst,
		(unsigned long long)corrupted->bad,control_fails ? "control fails as required" : "CONTROL DID NOT FAIL");
	if ( !clean_pass || !control_fails )
		failures++;
	if ( receipt != 0 )
		fprintf(receipt,"%s{\"op\":\"%s\",\"shape\":\"%s\",\"checked\":%llu,\"worst_ratio\":%s,\"worst_error\":%.6g,\"worst_tolerance\":%.6g,"
			"\"outside\":%llu,\"control_outside\":%llu,\"control_worst_ratio\":%s,\"tolerance\":\"%s\",\"pass\":%s}",
			ftell(receipt) > 2 ? ",\n" : "",op,shape,(unsigned long long)clean->count,JsonNumber(clean->worst,ratio,sizeof(ratio)),clean->worst_error,
			clean->worst_tolerance,(unsigned long long)clean->bad,(unsigned long long)corrupted->bad,
			JsonNumber(corrupted->worst,control_ratio,sizeof(control_ratio)),tolerance,clean_pass && control_fails ? "true" : "false");
}

static void TestGather(void)
{
	enum { TABLE = 4096, WIDTH = 5120, ROWS = 7 };
	SparkDeviceBuffer table, ids, out;
	uint16_t *t = (uint16_t *)Shared(&table,(uint64_t)TABLE * WIDTH * 2u,"gather table");
	uint32_t *id = (uint32_t *)Shared(&ids,ROWS * 4u,"gather ids");
	uint16_t *o = (uint16_t *)Shared(&out,(uint64_t)ROWS * WIDTH * 2u,"gather out");
	Check clean, corrupted;
	uint32_t pass, row, i, error;
	FillBf16(t,(uint64_t)TABLE * WIDTH,1.0);
	for ( row = 0u; row < ROWS; row++ )
		id[row] = row == 0u ? TABLE - 1u : (uint32_t)(NextRandom() % TABLE);
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		uint16_t saved = t[(uint64_t)id[3] * WIDTH + 77u];
		memset(check,0,sizeof(*check));
		if ( pass == 1u )
			t[(uint64_t)id[3] * WIDTH + 77u] = ToBf16(Bf16(saved) + 1.0f);
		if ( !Run(SparkOpsGather(&ops,queue,Whole(&table),Whole(&ids),Whole(&out),ROWS,WIDTH,TABLE),"gather") )
			return;
		if ( pass == 1u )
			t[(uint64_t)id[3] * WIDTH + 77u] = saved;
		for ( row = 0u; row < ROWS; row++ )
			for ( i = 0u; i < WIDTH; i++ )
				Observe(check,Bf16(o[(uint64_t)row * WIDTH + i]),Bf16(t[(uint64_t)id[row] * WIDTH + i]),0.0);
	}
	SparkDeviceOpsTakeError(&ops,&error);
	Report("gather_rows","rows 7, width 5120, table 4096",&clean,&corrupted,"exact: a gather copies bytes");
	id[0] = TABLE;
	Run(SparkOpsGather(&ops,queue,Whole(&table),Whole(&ids),Whole(&out),ROWS,WIDTH,TABLE),"gather out of range");
	if ( SparkDeviceOpsTakeError(&ops,&error) != SPARK_STATUS_IO_ERROR || error != 1u )
	{
		printf("FAIL gather_rows: an id outside the table did not raise the device error word\n");
		failures++;
	}
	else
		printf("PASS gather_rows            an id outside the table raises the device error word for row %u\n",error - 1u);
	SparkDeviceFree(&table);
	SparkDeviceFree(&ids);
	SparkDeviceFree(&out);
}

static void TestRmsNormResidual(uint32_t rows)
{
	enum { DIM = 5120 };
	SparkDeviceBuffer hidden, addend, weight, out;
	uint16_t *h = (uint16_t *)Shared(&hidden,(uint64_t)rows * DIM * 2u,"norm hidden");
	uint16_t *a = (uint16_t *)Shared(&addend,(uint64_t)rows * DIM * 2u,"norm addend");
	float *w = (float *)Shared(&weight,DIM * 4u,"norm weight");
	uint16_t *o = (uint16_t *)Shared(&out,(uint64_t)rows * DIM * 2u,"norm out");
	uint16_t *original = (uint16_t *)malloc((size_t)rows * DIM * 2u);
	SparkNormParams params;
	SparkTensor addend_tensor = Whole(&addend);
	Check clean, corrupted, sums;
	char shape[64];
	uint32_t pass, row, i;
	FillBf16(h,(uint64_t)rows * DIM,2.0);
	FillBf16(a,(uint64_t)rows * DIM,1.0);
	for ( i = 0u; i < DIM; i++ )
		w[i] = (float)(1.0 + 0.3 * Normal());
	memcpy(original,h,(size_t)rows * DIM * 2u);
	memset(&params,0,sizeof(params));
	params.rows = rows;
	params.dimension = DIM;
	params.input_stride = params.output_stride = DIM;
	params.has_addend = 1u;
	params.write_sum = 1u;
	params.epsilon = 1e-6f;
	memset(&sums,0,sizeof(sums));
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		memset(check,0,sizeof(*check));
		memcpy(h,original,(size_t)rows * DIM * 2u);
		if ( pass == 1u )
			h[(uint64_t)(rows - 1u) * DIM + 5u] = ToBf16(Bf16(original[(uint64_t)(rows - 1u) * DIM + 5u]) * -4.0f);
		if ( !Run(SparkOpsRmsNormResidual(&ops,queue,Whole(&hidden),&addend_tensor,Whole(&weight),Whole(&out),&params),"rmsnorm_residual") )
			break;
		for ( row = 0u; row < rows; row++ )
		{
			double square = 0.0, x[DIM];
			for ( i = 0u; i < DIM; i++ )
			{
				const uint64_t at = (uint64_t)row * DIM + i;
				x[i] = Bf16(ToBf16(Bf16(original[at]) + Bf16(a[at])));
				square += x[i] * x[i];
				if ( pass == 0u )
					Observe(&sums,Bf16(h[at]),x[i],0.0);
			}
			const double scale = 1.0 / sqrt(square / DIM + 1e-6);
			const double scale_error = Gamma(DIM) + FUNCTION_ERROR;
			for ( i = 0u; i < DIM; i++ )
			{
				const double y = x[i] * scale * w[i];
				Observe(check,Bf16(o[(uint64_t)row * DIM + i]),y,UNIT_BF16 * fabs(y) + (1.0 + UNIT_BF16) * (scale_error + 3.0 * UNIT_F32) * fabs(y));
			}
		}
	}
	snprintf(shape,sizeof(shape),"rows %u, dim 5120, residual written",rows);
	Report("rmsnorm_residual",shape,&clean,&corrupted,"u|y| + (1+u)(gamma_5120 + 16 eps32 + 3 eps32)|y|, u = 2^-8, eps32 = 2^-24");
	if ( sums.bad != 0u )
	{
		printf("FAIL rmsnorm_residual: the written residual is not bf16(hidden + addend) at %llu elements\n",(unsigned long long)sums.bad);
		failures++;
	}
	free(original);
	SparkDeviceFree(&hidden);
	SparkDeviceFree(&addend);
	SparkDeviceFree(&weight);
	SparkDeviceFree(&out);
}

static void TestRmsNormGate(uint32_t rows)
{
	enum { HEADS = 48, HEAD = 128, STRIDE = 6144, GATE_STRIDE = 16384, GATE_OFFSET = 10240 };
	SparkDeviceBuffer input, gate, weight, out;
	uint16_t *x = (uint16_t *)Shared(&input,(uint64_t)rows * STRIDE * 2u,"gate norm input");
	uint16_t *g = (uint16_t *)Shared(&gate,(uint64_t)rows * GATE_STRIDE * 2u,"gate norm gate");
	uint16_t *w = (uint16_t *)Shared(&weight,HEAD * 2u,"gate norm weight");
	uint16_t *o = (uint16_t *)Shared(&out,(uint64_t)rows * STRIDE * 2u,"gate norm out");
	SparkGatedNormParams params;
	Check clean, corrupted;
	char shape[64];
	uint32_t pass, row, head, i;
	FillBf16(x,(uint64_t)rows * STRIDE,0.5);
	FillBf16(g,(uint64_t)rows * GATE_STRIDE,1.5);
	for ( i = 0u; i < HEAD; i++ )
		w[i] = ToBf16((float)(1.0 + 0.2 * Normal()));
	memset(&params,0,sizeof(params));
	params.rows = rows;
	params.heads = HEADS;
	params.head_dimension = HEAD;
	params.input_stride = params.output_stride = STRIDE;
	params.gate_stride = GATE_STRIDE;
	params.gate_offset = GATE_OFFSET;
	params.epsilon = 1e-6f;
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		const uint64_t poke = (uint64_t)(rows / 2u) * STRIDE + 7u * HEAD + 3u;
		const uint16_t saved = x[poke];
		memset(check,0,sizeof(*check));
		if ( pass == 1u )
			x[poke] = ToBf16(Bf16(saved) * 3.0f + 0.5f);
		if ( !Run(SparkOpsRmsNormGate(&ops,queue,Whole(&input),Whole(&gate),Whole(&weight),Whole(&out),&params),"rmsnorm_gate") )
			break;
		x[poke] = saved;
		for ( row = 0u; row < rows; row++ )
			for ( head = 0u; head < HEADS; head++ )
			{
				const uint16_t *v = x + (uint64_t)row * STRIDE + head * HEAD;
				double square = 0.0;
				for ( i = 0u; i < HEAD; i++ )
					square += (double)Bf16(v[i]) * Bf16(v[i]);
				const double scale = 1.0 / sqrt(square / HEAD + 1e-6);
				for ( i = 0u; i < HEAD; i++ )
				{
					const double exact = Bf16(v[i]) * scale, normed = RoundBf16(exact);
					const double first = FlipSlack(exact,Gamma(HEAD) + FUNCTION_ERROR + 2.0 * UNIT_F32);
					const double weighted = RoundBf16(normed * Bf16(w[i]));
					const double second = first != 0.0 ? fabs((double)Bf16(w[i])) * first + 2.0 * UNIT_BF16 * (fabs(weighted) + fabs((double)Bf16(w[i])) * first) : 0.0;
					const double z = Bf16(g[(uint64_t)row * GATE_STRIDE + GATE_OFFSET + head * HEAD + i]);
					const double y = weighted * Silu(z);
					const double tolerance = UNIT_BF16 * fabs(y) + fabs(Silu(z)) * second + (2.0 * FUNCTION_ERROR + 2.0 * UNIT_F32) * fabs(y);
					Observe(check,Bf16(o[(uint64_t)row * STRIDE + head * HEAD + i]),y,tolerance);
				}
			}
	}
	snprintf(shape,sizeof(shape),"rows %u, 48 heads x 128, gate in fused row",rows);
	Report("rmsnorm_gate",shape,&clean,&corrupted,"u|y| + |silu(z)| (flip slack of both bf16 roundings, propagated) + (2*16 + 2) eps32 |y|");
	SparkDeviceFree(&input);
	SparkDeviceFree(&gate);
	SparkDeviceFree(&weight);
	SparkDeviceFree(&out);
}

static void TestHeadRope(uint32_t rows, uint32_t heads, uint32_t offset, uint32_t head_stride, const char *label)
{
	enum { HEAD = 256, STRIDE = 14336, ROPE = 64 };
	SparkDeviceBuffer input, weight, positions, out;
	uint16_t *x = (uint16_t *)Shared(&input,(uint64_t)rows * STRIDE * 2u,"rope input");
	float *w = (float *)Shared(&weight,HEAD * 4u,"rope weight");
	uint32_t *pos = (uint32_t *)Shared(&positions,rows * 4u,"rope positions");
	uint16_t *o = (uint16_t *)Shared(&out,(uint64_t)rows * heads * HEAD * 2u,"rope out");
	SparkHeadRopeParams params;
	Check clean, corrupted;
	char shape[96];
	uint32_t pass, row, head, i;
	FillBf16(x,(uint64_t)rows * STRIDE,1.0);
	for ( i = 0u; i < HEAD; i++ )
		w[i] = (float)(1.0 + 0.25 * Normal());
	for ( row = 0u; row < rows; row++ )
		pos[row] = row == 0u ? 262143u : (uint32_t)(NextRandom() % 8192u);
	memset(&params,0,sizeof(params));
	params.rows = rows;
	params.heads = heads;
	params.head_dimension = HEAD;
	params.input_stride = STRIDE;
	params.input_offset = offset;
	params.input_head_stride = head_stride;
	params.output_stride = heads * HEAD;
	params.rope_dimension = ROPE;
	params.epsilon = 1e-6f;
	for ( i = 0u; i < ROPE / 2u; i++ )
		params.inverse_frequency[i] = (float)(1.0 / pow(10000000.0,(double)(2u * i) / ROPE));
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		const uint64_t poke = (uint64_t)(rows - 1u) * STRIDE + offset + (heads - 1u) * head_stride + 40u;
		const uint16_t saved = x[poke];
		memset(check,0,sizeof(*check));
		if ( pass == 1u )
			x[poke] = ToBf16(Bf16(saved) - 2.0f);
		if ( !Run(SparkOpsHeadRmsNormRope(&ops,queue,Whole(&input),Whole(&weight),Whole(&positions),Whole(&out),&params),"head_rmsnorm_rope") )
			break;
		x[poke] = saved;
		for ( row = 0u; row < rows; row++ )
			for ( head = 0u; head < heads; head++ )
			{
				const uint16_t *v = x + (uint64_t)row * STRIDE + offset + head * head_stride;
				double square = 0.0, normed[HEAD], slack[HEAD];
				for ( i = 0u; i < HEAD; i++ )
					square += (double)Bf16(v[i]) * Bf16(v[i]);
				const double scale = 1.0 / sqrt(square / HEAD + 1e-6);
				for ( i = 0u; i < HEAD; i++ )
				{
					normed[i] = RoundBf16(Bf16(v[i]) * scale * w[i]);
					slack[i] = FlipSlack(Bf16(v[i]) * scale * w[i],Gamma(HEAD) + FUNCTION_ERROR + 3.0 * UNIT_F32);
				}
				for ( i = 0u; i < HEAD; i++ )
				{
					double y = normed[i], tolerance;
					if ( i < ROPE )
					{
						const double angle = (double)(float)((float)pos[row] * params.inverse_frequency[i % (ROPE / 2u)]);
						const uint32_t other = i < ROPE / 2u ? i + ROPE / 2u : i - ROPE / 2u;
						const double partner = i < ROPE / 2u ? -normed[other] : normed[other];
						y = normed[i] * cos(angle) + partner * sin(angle);
						tolerance = UNIT_BF16 * fabs(y) + slack[i] * fabs(cos(angle)) + slack[other] * fabs(sin(angle)) +
							(4.0 * FUNCTION_ERROR + 4.0 * UNIT_F32) * (fabs(normed[i]) + fabs(partner));
					}
					else
						tolerance = slack[i];
					Observe(check,Bf16(o[(uint64_t)row * heads * HEAD + head * HEAD + i]),y,tolerance);
				}
			}
	}
	snprintf(shape,sizeof(shape),"rows %u, %s, rope 64, theta 1e7",rows,label);
	Report("head_rmsnorm_rope",shape,&clean,&corrupted,"rotary span: u|y| + flip slack of the normed bf16 rounding (both elements) + (4*16+4) eps32 (|x|+|partner|); elsewhere the flip slack only");
	SparkDeviceFree(&input);
	SparkDeviceFree(&weight);
	SparkDeviceFree(&positions);
	SparkDeviceFree(&out);
}

static uint32_t SampleIndex(uint64_t index, uint64_t total, uint64_t samples)
{
	if ( samples >= total )
		return((uint32_t)index);
	return((uint32_t)((index * 2654435761ull + 12345u) % total));
}

static void TestLinear(uint32_t rows, uint32_t K, uint32_t N, uint32_t output_f32, uint32_t gated, uint64_t samples)
{
	const uint32_t width = gated != 0u ? 2u * K : K;
	SparkDeviceBuffer x, w, y;
	uint16_t *xv = (uint16_t *)Shared(&x,(uint64_t)rows * width * 2u,"linear x");
	uint16_t *wv = (uint16_t *)Shared(&w,(uint64_t)N * K * 2u,"linear w");
	void *yv = Shared(&y,(uint64_t)rows * N * (output_f32 != 0u ? 4u : 2u),"linear y");
	double *xd = (double *)malloc((size_t)rows * K * sizeof(double)), *xs = (double *)calloc((size_t)rows * K,sizeof(double));
	SparkGemvParams params;
	Check clean, corrupted;
	char shape[96];
	uint64_t total = (uint64_t)rows * N, index;
	uint32_t pass, r, k;
	FillBf16(xv,(uint64_t)rows * width,1.0);
	FillBf16(wv,(uint64_t)N * K,1.0 / sqrt((double)K));
	for ( r = 0u; r < rows; r++ )
		for ( k = 0u; k < K; k++ )
		{
			const double act = Silu(Bf16(xv[(uint64_t)r * width + k])) * Bf16(xv[(uint64_t)r * width + K + k]);
			xd[(uint64_t)r * K + k] = gated != 0u ? RoundBf16(act) : Bf16(xv[(uint64_t)r * width + k]);
			xs[(uint64_t)r * K + k] = gated != 0u ? FlipSlack(act,3.0 * FUNCTION_ERROR + 2.0 * UNIT_F32) : 0.0;
		}
	memset(&params,0,sizeof(params));
	params.rows = rows;
	params.input_dimension = K;
	params.output_dimension = N;
	params.input_stride = width;
	params.output_stride = N;
	params.output_f32 = output_f32;
	params.gated_input = gated;
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		const uint64_t poke = (uint64_t)(rows - 1u) * width + K / 2u, partner = gated != 0u ? poke + K : poke;
		const uint16_t saved = xv[poke], saved_partner = xv[partner];
		memset(check,0,sizeof(*check));
		if ( pass == 1u )
		{
			xv[poke] = ToBf16(Bf16(saved) + (Bf16(saved) >= 0.0f ? -64.0f : 64.0f));
			if ( gated != 0u )
			{
				xv[poke] = ToBf16(3.0f);
				xv[partner] = ToBf16(Bf16(saved_partner) + 8.0f);
			}
		}
		if ( !Run(SparkOpsLinear(&ops,queue,Whole(&x),Whole(&w),Whole(&y),&params),"linear") )
			break;
		xv[poke] = saved;
		xv[partner] = saved_partner;
		for ( index = 0u; index < (samples < total ? samples : total); index++ )
		{
			const uint64_t element = SampleIndex(index,total,samples);
			const uint32_t row = (uint32_t)(element / N), n = (uint32_t)(element % N);
			double sum = 0.0, magnitude = 0.0, slack = 0.0, got;
			for ( k = 0u; k < K; k++ )
			{
				const double term = xd[(uint64_t)row * K + k] * Bf16(wv[(uint64_t)n * K + k]);
				sum += term;
				magnitude += fabs(term);
				slack += xs[(uint64_t)row * K + k] * fabs((double)Bf16(wv[(uint64_t)n * K + k]));
			}
			got = output_f32 != 0u ? ((float *)yv)[element] : Bf16(((uint16_t *)yv)[element]);
			Observe(check,got,sum,(output_f32 != 0u ? 0.0 : UNIT_BF16 * fabs(sum)) + (1.0 + UNIT_BF16) * (Gamma(K) * magnitude + slack));
		}
	}
	snprintf(shape,sizeof(shape),"rows %u, K %u -> N %u%s%s%s",rows,K,N,output_f32 != 0u ? ", f32 out" : "",gated != 0u ? ", silu-gated input" : "",
		samples < total ? ", sampled" : "");
	Report(rows > SPARK_OPS_DECODE_MAX_ROWS ? "gemm" : rows > SPARK_OPS_GEMV_MAX_ROWS ? "gemm rows" : "gemv",shape,&clean,&corrupted,
		output_f32 != 0u ? "gamma_K sum|x w|: fp32 accumulation of exact bf16 products" :
		gated != 0u ? "u|y| + (1+u)(gamma_K sum|x w| + sum flip slack(x)|w|), x = bf16(silu(gate) up)" : "u|y| + (1+u) gamma_K sum|x w|, u = 2^-8");
	free(xd);
	free(xs);
	SparkDeviceFree(&x);
	SparkDeviceFree(&w);
	SparkDeviceFree(&y);
}

static void TestLinearRowIndependence(uint32_t rows, uint32_t other_rows, uint32_t K, uint32_t N, uint32_t output_f32, uint32_t gated)
{
	const uint32_t width = gated != 0u ? 2u * K : K, element = output_f32 != 0u ? (uint32_t)sizeof(float) : (uint32_t)sizeof(uint16_t);
	const uint32_t picks[3] = { 0u, rows / 2u, rows - 1u };
	SparkDeviceBuffer x, w, y, ox, oy;
	uint16_t *xv = (uint16_t *)Shared(&x,(uint64_t)rows * width * sizeof(uint16_t),"row independence x");
	uint16_t *wv = (uint16_t *)Shared(&w,(uint64_t)N * K * sizeof(uint16_t),"row independence w");
	uint8_t *yv = (uint8_t *)Shared(&y,(uint64_t)rows * N * element,"row independence y");
	uint16_t *oxv = (uint16_t *)Shared(&ox,(uint64_t)other_rows * width * sizeof(uint16_t),"row independence other x");
	uint8_t *oyv = (uint8_t *)Shared(&oy,(uint64_t)other_rows * N * element,"row independence other y");
	SparkGemvParams params, other;
	Check clean, corrupted;
	char shape[128];
	uint32_t pick, pass, n;
	FillBf16(xv,(uint64_t)rows * width,1.0);
	FillBf16(wv,(uint64_t)N * K,1.0 / sqrt((double)K));
	memset(&params,0,sizeof(params));
	params.rows = rows;
	params.input_dimension = K;
	params.output_dimension = N;
	params.input_stride = width;
	params.output_stride = N;
	params.output_f32 = output_f32;
	params.gated_input = gated;
	other = params;
	other.rows = other_rows;
	memset(&clean,0,sizeof(clean));
	memset(&corrupted,0,sizeof(corrupted));
	if ( Run(SparkOpsLinear(&ops,queue,Whole(&x),Whole(&w),Whole(&y),&params),"row independence batch") )
		for ( pick = 0u; pick < 3u; pick++ )
			for ( pass = 0u; pass < 2u; pass++ )
			{
				Check *check = pass == 0u ? &clean : &corrupted;
				const uint32_t slot = (picks[pick] * 7u + 3u) % other_rows;
				FillBf16(oxv,(uint64_t)other_rows * width,1.0);
				memcpy(oxv + (uint64_t)slot * width,xv + (uint64_t)picks[pick] * width,(size_t)width * sizeof(uint16_t));
				if ( pass == 1u )
					oxv[(uint64_t)slot * width + K / 3u] = ToBf16(Bf16(oxv[(uint64_t)slot * width + K / 3u]) >= 0.0f ? -2.0f : 2.0f);
				if ( !Run(SparkOpsLinear(&ops,queue,Whole(&ox),Whole(&w),Whole(&oy),&other),"row independence other batch") )
					break;
				for ( n = 0u; n < N; n++ )
					Observe(check,memcmp(yv + ((uint64_t)picks[pick] * N + n) * element,oyv + ((uint64_t)slot * N + n) * element,element) == 0 ? 0.0 : 1.0,
						0.0,0.0);
			}
	snprintf(shape,sizeof(shape),"rows %u vs %u, K %u -> N %u%s%s",rows,other_rows,K,N,output_f32 != 0u ? ", f32 out" : "",
		gated != 0u ? ", silu-gated input" : "");
	Report("linear row independence",shape,&clean,&corrupted,"exact: a row's output bits do not depend on the other rows or the row count");
	SparkDeviceFree(&x);
	SparkDeviceFree(&w);
	SparkDeviceFree(&y);
	SparkDeviceFree(&ox);
	SparkDeviceFree(&oy);
}

static void ConvReference(const uint16_t *input, const uint16_t *weight, double *window, uint32_t channels, uint32_t stride, uint32_t first,
	uint32_t count, double *out)
{
	uint32_t row, c, tap;
	for ( c = 0u; c < channels; c++ )
		for ( row = first; row < first + count; row++ )
		{
			float total = 0.0f;
			for ( tap = 0u; tap + 1u < 4u; tap++ )
				window[(uint64_t)c * 4u + tap] = window[(uint64_t)c * 4u + tap + 1u];
			window[(uint64_t)c * 4u + 3u] = Bf16(input[(uint64_t)row * stride + c]);
			for ( tap = 0u; tap < 4u; tap++ )
				total += (float)(window[(uint64_t)c * 4u + tap] * Bf16(weight[c * 4u + tap]));
			out[(uint64_t)row * channels + c] = Silu(RoundBf16(total));
		}
}

static void TestCausalConv(void)
{
	enum { CHANNELS = 10240, STRIDE = 16384, FIRST = 300, SECOND = 5 };
	SparkDeviceBuffer window, index, begin, input, weight, output;
	uint16_t *win = (uint16_t *)Shared(&window,(uint64_t)2u * CHANNELS * 4u * 2u,"conv window");
	uint32_t *slot = (uint32_t *)Shared(&index,4u,"conv slot");
	uint32_t *rb = (uint32_t *)Shared(&begin,8u,"conv begin");
	uint16_t *in = (uint16_t *)Shared(&input,(uint64_t)(FIRST + SECOND) * STRIDE * 2u,"conv input");
	uint16_t *w = (uint16_t *)Shared(&weight,CHANNELS * 4u * 2u,"conv weight");
	uint16_t *out = (uint16_t *)Shared(&output,(uint64_t)(FIRST + SECOND) * STRIDE * 2u,"conv output");
	double *ref_window = (double *)calloc((size_t)CHANNELS * 4u,sizeof(double));
	double *ref = (double *)malloc((size_t)(FIRST + SECOND) * CHANNELS * sizeof(double));
	uint16_t *original = (uint16_t *)malloc((size_t)(FIRST + SECOND) * STRIDE * 2u);
	SparkOpsSequences sequences;
	SparkConvParams params;
	Check clean, corrupted;
	uint32_t pass, row, c;
	FillBf16(in,(uint64_t)(FIRST + SECOND) * STRIDE,1.0);
	FillBf16(w,CHANNELS * 4u,0.5);
	memcpy(original,in,(size_t)(FIRST + SECOND) * STRIDE * 2u);
	memset(&params,0,sizeof(params));
	params.channels = CHANNELS;
	params.row_stride = STRIDE;
	params.sequences = 1u;
	params.kernel_taps = 4u;
	params.commit = 1u;
	sequences.state_index = Whole(&index);
	sequences.row_begin = Whole(&begin);
	sequences.sequences = 1u;
	sequences.commit = 1u;
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		memset(check,0,sizeof(*check));
		memset(win,0,(size_t)2u * CHANNELS * 4u * 2u);
		memset(ref_window,0,(size_t)CHANNELS * 4u * sizeof(double));
		memcpy(in,original,(size_t)(FIRST + SECOND) * STRIDE * 2u);
		if ( pass == 1u )
			in[(uint64_t)(FIRST - 2u) * STRIDE + 999u] = ToBf16(Bf16(original[(uint64_t)(FIRST - 2u) * STRIDE + 999u]) + 3.0f);
		slot[0] = 1u;
		rb[0] = 0u;
		rb[1] = FIRST;
		if ( !Run(SparkOpsCausalConv(&ops,queue,Whole(&window),&sequences,Whole(&input),Whole(&weight),Whole(&input),&params),"causal_conv first") )
			break;
		rb[0] = FIRST;
		rb[1] = FIRST + SECOND;
		if ( !Run(SparkOpsCausalConv(&ops,queue,Whole(&window),&sequences,Whole(&input),Whole(&weight),Whole(&output),&params),"causal_conv second") )
			break;
		ConvReference(original,w,ref_window,CHANNELS,STRIDE,0u,FIRST + SECOND,ref);
		for ( row = 0u; row < FIRST + SECOND; row++ )
			for ( c = 0u; c < CHANNELS; c++ )
			{
				const double y = ref[(uint64_t)row * CHANNELS + c];
				const double got = Bf16((row < FIRST ? in : out)[(uint64_t)row * STRIDE + c]);
				Observe(check,got,y,UNIT_BF16 * fabs(y) + 2.0 * FUNCTION_ERROR * fabs(y));
			}
	}
	Report("causal_conv_silu","10240 channels, 300 rows in place, then 5 more",&clean,&corrupted,
		"u|y| + 2*16 eps32 |y|: the four-tap fp32 sum of exact products is emulated, rounded to bf16, then silu; the window carries over a committed call");
	free(ref_window);
	free(ref);
	free(original);
	SparkDeviceFree(&window);
	SparkDeviceFree(&index);
	SparkDeviceFree(&begin);
	SparkDeviceFree(&input);
	SparkDeviceFree(&weight);
	SparkDeviceFree(&output);
}

typedef struct DeltaReference
{
	double *state;
	double *bound;
} DeltaReference;

static void DeltaStep(DeltaReference *ref, const uint16_t *mix, const uint16_t *gates, const float *a_log, const float *dt_bias, uint32_t row,
	double *out, double *tolerance)
{
	enum { KH = 16, VH = 48, D = 128, MIX = 16384 };
	const uint16_t *m = mix + (uint64_t)row * MIX, *g = gates + (uint64_t)row * 96u;
	uint32_t head, i, v;
	for ( head = 0u; head < VH; head++ )
	{
		const uint32_t kh = head / 3u;
		double q[D], k[D], qs = 0.0, ks = 0.0;
		for ( i = 0u; i < D; i++ )
		{
			q[i] = Bf16(m[kh * D + i]);
			k[i] = Bf16(m[2048u + kh * D + i]);
			qs += q[i] * q[i];
			ks += k[i] * k[i];
		}
		for ( i = 0u; i < D; i++ )
		{
			q[i] *= 1.0 / sqrt(qs + 1e-6) / sqrt((double)D);
			k[i] *= 1.0 / sqrt(ks + 1e-6);
		}
		const double shifted = Bf16(g[48u + head]) + dt_bias[head];
		const double softplus = shifted > 20.0 ? shifted : log1p(exp(shifted));
		const double decay = exp(-exp((double)a_log[head]) * softplus), beta = Sigmoid(Bf16(g[head]));
		for ( v = 0u; v < D; v++ )
		{
			double *s = ref->state + ((uint64_t)head * D) * D + v;
			double predicted = 0.0, result = 0.0, magnitude = 0.0, qnorm = 0.0;
			const double value = Bf16(m[4096u + head * D + v]);
			for ( i = 0u; i < D; i++ )
			{
				s[(uint64_t)i * D] *= decay;
				predicted += s[(uint64_t)i * D] * k[i];
			}
			for ( i = 0u; i < D; i++ )
				s[(uint64_t)i * D] += beta * (value - predicted) * k[i];
			for ( i = 0u; i < D; i++ )
			{
				result += s[(uint64_t)i * D] * q[i];
				magnitude += fabs(s[(uint64_t)i * D] * q[i]);
				qnorm += fabs(q[i]);
			}
			double *b = ref->bound + (uint64_t)head * D + v, smax = 0.0;
			for ( i = 0u; i < D; i++ )
				if ( fabs(s[(uint64_t)i * D]) > smax )
					smax = fabs(s[(uint64_t)i * D]);
			*b = decay * *b + (4.0 * Gamma(D) + 16.0 * FUNCTION_ERROR) * (smax + fabs(value));
			out[head * D + v] = result;
			tolerance[head * D + v] = UNIT_BF16 * fabs(result) + (1.0 + UNIT_BF16) * (Gamma(D) * magnitude + *b * qnorm + 4.0 * FUNCTION_ERROR * magnitude);
		}
	}
}

static void TestGatedDelta(void)
{
	enum { KH = 16, VH = 48, D = 128, MIX = 16384, FIRST = 200, SECOND = 3, ROWS = FIRST + SECOND };
	SparkDeviceBuffer state, index, begin, mix, gates, alog, bias, scratch, out;
	float *s = (float *)Shared(&state,(uint64_t)2u * VH * D * D * 4u,"delta state");
	uint32_t *slot = (uint32_t *)Shared(&index,4u,"delta slot");
	uint32_t *rb = (uint32_t *)Shared(&begin,8u,"delta begin");
	uint16_t *m = (uint16_t *)Shared(&mix,(uint64_t)ROWS * MIX * 2u,"delta mix");
	uint16_t *g = (uint16_t *)Shared(&gates,(uint64_t)ROWS * 96u * 2u,"delta gates");
	float *al = (float *)Shared(&alog,VH * 4u,"delta a_log");
	float *db = (float *)Shared(&bias,VH * 4u,"delta dt_bias");
	uint16_t *o = (uint16_t *)Shared(&out,(uint64_t)ROWS * VH * D * 2u,"delta out");
	DeltaReference ref;
	double *ref_out = (double *)malloc(VH * D * sizeof(double)), *ref_tol = (double *)malloc(VH * D * sizeof(double));
	SparkOpsSequences sequences;
	SparkDeltaParams params;
	Check clean, corrupted;
	uint32_t pass, row, i;
	(void)Shared(&scratch,SparkOpsDeltaScratchBytes(ROWS,KH,VH,D),"delta scratch");
	FillBf16(m,(uint64_t)ROWS * MIX,1.0);
	FillBf16(g,(uint64_t)ROWS * 96u,2.0);
	for ( i = 0u; i < VH; i++ )
	{
		al[i] = (float)(Normal() * 0.5);
		db[i] = (float)(Normal() * 0.5);
	}
	ref.state = (double *)malloc((size_t)VH * D * D * sizeof(double));
	ref.bound = (double *)malloc((size_t)VH * D * sizeof(double));
	memset(&params,0,sizeof(params));
	params.sequences = 1u;
	params.key_heads = KH;
	params.value_heads = VH;
	params.key_dimension = D;
	params.value_dimension = D;
	params.mix_stride = MIX;
	params.query_offset = 0u;
	params.key_offset = 2048u;
	params.value_offset = 4096u;
	params.gate_stride = 96u;
	params.beta_offset = 0u;
	params.decay_offset = 48u;
	params.output_stride = VH * D;
	params.commit = 1u;
	params.epsilon = 1e-6f;
	sequences.state_index = Whole(&index);
	sequences.row_begin = Whole(&begin);
	sequences.sequences = 1u;
	sequences.commit = 1u;
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		const uint64_t poke = (uint64_t)(FIRST - 1u) * MIX + 4096u + 5u * D + 9u;
		const uint16_t saved = m[poke];
		memset(check,0,sizeof(*check));
		memset(s,0,(size_t)2u * VH * D * D * 4u);
		memset(ref.state,0,(size_t)VH * D * D * sizeof(double));
		memset(ref.bound,0,(size_t)VH * D * sizeof(double));
		slot[0] = 1u;
		if ( pass == 1u )
			m[poke] = ToBf16(Bf16(saved) + 2.0f);
		params.rows = FIRST;
		rb[0] = 0u;
		rb[1] = FIRST;
		if ( !Run(SparkOpsGatedDelta(&ops,queue,Whole(&state),&sequences,Whole(&mix),Whole(&gates),Whole(&alog),Whole(&bias),Whole(&scratch),Whole(&out),
			&params),"gated_delta first") )
			break;
		params.rows = SECOND;
		rb[0] = 0u;
		rb[1] = SECOND;
		if ( !Run(SparkOpsGatedDelta(&ops,queue,Whole(&state),&sequences,SparkTensorSlice(Whole(&mix),(uint64_t)FIRST * MIX * 2u,(uint64_t)SECOND * MIX * 2u),
			SparkTensorSlice(Whole(&gates),(uint64_t)FIRST * 96u * 2u,(uint64_t)SECOND * 96u * 2u),Whole(&alog),Whole(&bias),Whole(&scratch),
			SparkTensorSlice(Whole(&out),(uint64_t)FIRST * VH * D * 2u,(uint64_t)SECOND * VH * D * 2u),&params),"gated_delta second") )
			break;
		m[poke] = saved;
		for ( row = 0u; row < ROWS; row++ )
		{
			DeltaStep(&ref,m,g,al,db,row,ref_out,ref_tol);
			for ( i = 0u; i < VH * D; i++ )
				Observe(check,Bf16(o[(uint64_t)row * VH * D + i]),ref_out[i],ref_tol[i]);
		}
	}
	Report("gated_delta_rule","16 key / 48 value heads x 128, 200 rows then 3",&clean,&corrupted,
		"u|o| + (1+u)(gamma_128 sum|S q| + B_t sum|q| + 4*16 eps32 sum|S q|), B_t = decay*B_{t-1} + (4 gamma_128 + 16*16 eps32)(max|S| + |v|)");
	free(ref.state);
	free(ref.bound);
	free(ref_out);
	free(ref_tol);
	SparkDeviceFree(&state);
	SparkDeviceFree(&index);
	SparkDeviceFree(&begin);
	SparkDeviceFree(&mix);
	SparkDeviceFree(&gates);
	SparkDeviceFree(&alog);
	SparkDeviceFree(&bias);
	SparkDeviceFree(&scratch);
	SparkDeviceFree(&out);
}

typedef struct AttentionFixture
{
	SparkDeviceBuffer pool, table, sequence, positions, keys, values, query, gate, partial, output;
	SparkOpsPaged paged;
	uint32_t context;
	uint32_t pages;
} AttentionFixture;

enum { ATT_HEADS = 24, ATT_KV = 4, ATT_DIM = 256, ATT_SLOTS = 64, ATT_QKV = 14336 };

static void AttentionFill(AttentionFixture *f, uint32_t context, uint32_t sequences)
{
	SparkKvParams params;
	uint32_t *table, *sequence, *position, row, s, p;
	f->context = context;
	f->pages = (context + ATT_SLOTS - 1u) / ATT_SLOTS;
	Shared(&f->pool,(uint64_t)sequences * f->pages * ATT_SLOTS * 2u * ATT_KV * ATT_DIM * 2u,"attention pool");
	table = (uint32_t *)Shared(&f->table,(uint64_t)sequences * f->pages * 4u,"attention table");
	for ( s = 0u; s < sequences; s++ )
		for ( p = 0u; p < f->pages; p++ )
			table[s * f->pages + p] = (sequences * f->pages - 1u) - (s * f->pages + p);
	sequence = (uint32_t *)Shared(&f->sequence,(uint64_t)sequences * context * 4u,"attention fill sequence");
	position = (uint32_t *)Shared(&f->positions,(uint64_t)sequences * context * 4u,"attention fill positions");
	FillBf16((uint16_t *)Shared(&f->keys,(uint64_t)sequences * context * ATT_KV * ATT_DIM * 2u,"attention keys"),(uint64_t)sequences * context * ATT_KV * ATT_DIM,1.0);
	FillBf16((uint16_t *)Shared(&f->values,(uint64_t)sequences * context * ATT_QKV * 2u,"attention values"),(uint64_t)sequences * context * ATT_QKV,1.0);
	for ( row = 0u; row < sequences * context; row++ )
	{
		sequence[row] = row / context;
		position[row] = row % context;
	}
	f->paged.pool = Whole(&f->pool);
	f->paged.page_table = Whole(&f->table);
	f->paged.sequence_of_row = Whole(&f->sequence);
	f->paged.positions = Whole(&f->positions);
	f->paged.page_slots = ATT_SLOTS;
	f->paged.pages_per_sequence = f->pages;
	f->paged.pool_pages = sequences * f->pages;
	f->paged.kv_heads = ATT_KV;
	memset(&params,0,sizeof(params));
	params.rows = sequences * context;
	params.kv_heads = ATT_KV;
	params.head_dimension = ATT_DIM;
	params.page_slots = ATT_SLOTS;
	params.pages_per_sequence = f->pages;
	params.key_stride = ATT_KV * ATT_DIM;
	params.value_stride = ATT_QKV;
	params.value_offset = 13312u;
	params.pool_pages = sequences * f->pages;
	Run(SparkOpsKvStore(&ops,queue,&f->paged,Whole(&f->keys),Whole(&f->values),&params),"kv_store");
}

static void AttentionFree(AttentionFixture *f)
{
	SparkDeviceFree(&f->pool);
	SparkDeviceFree(&f->table);
	SparkDeviceFree(&f->sequence);
	SparkDeviceFree(&f->positions);
	SparkDeviceFree(&f->keys);
	SparkDeviceFree(&f->values);
}

static void AttentionReference(const AttentionFixture *f, const uint16_t *query, const uint16_t *gate, uint32_t sequence, uint32_t position,
	uint32_t head, double *out, double *tolerance)
{
	const uint16_t *keys = (const uint16_t *)f->keys.host, *values = (const uint16_t *)f->values.host;
	const uint32_t kv = head / (ATT_HEADS / ATT_KV), length = position + 1u;
	double *score = (double *)malloc(length * sizeof(double)), *slack = (double *)malloc(length * sizeof(double)), best = -INFINITY, total = 0.0;
	uint32_t j, d;
	for ( j = 0u; j < length; j++ )
	{
		const uint16_t *k = keys + ((uint64_t)sequence * f->context + j) * ATT_KV * ATT_DIM + kv * ATT_DIM;
		double dot = 0.0, magnitude = 0.0;
		for ( d = 0u; d < ATT_DIM; d++ )
		{
			dot += (double)Bf16(query[head * ATT_DIM + d]) * Bf16(k[d]);
			magnitude += fabs((double)Bf16(query[head * ATT_DIM + d]) * Bf16(k[d]));
		}
		score[j] = dot / 16.0;
		slack[j] = (Gamma(ATT_DIM + 2.0) * magnitude / 16.0 + 2.0 * FUNCTION_ERROR * (1.0 + fabs(score[j])));
		if ( score[j] > best )
			best = score[j];
	}
	for ( j = 0u; j < length; j++ )
	{
		score[j] = exp(score[j] - best);
		total += score[j];
	}
	for ( d = 0u; d < ATT_DIM; d++ )
	{
		double value = 0.0, spread = 0.0, magnitude = 0.0;
		for ( j = 0u; j < length; j++ )
		{
			const double v = Bf16(values[((uint64_t)sequence * f->context + j) * ATT_QKV + 13312u + kv * ATT_DIM + d]);
			value += score[j] / total * v;
			magnitude += score[j] / total * fabs(v);
			spread += score[j] / total * fabs(v) * 2.0 * slack[j];
		}
		const double attended = RoundBf16(value), g = Bf16(gate[head * 512u + 256u + d]);
		const double absolute = 2.0 * spread + Gamma(length + 8.0) * magnitude + 4.0 * FUNCTION_ERROR * fabs(value);
		const double slack = fabs(RoundBf16(value - absolute) - attended) > fabs(RoundBf16(value + absolute) - attended) ?
			fabs(RoundBf16(value - absolute) - attended) : fabs(RoundBf16(value + absolute) - attended);
		out[d] = attended * Sigmoid(g);
		tolerance[d] = UNIT_BF16 * fabs(out[d]) + Sigmoid(g) * slack + 4.0 * FUNCTION_ERROR * fabs(out[d]);
	}
	free(score);
	free(slack);
}

static void TestAttention(uint32_t context, uint32_t rows, uint32_t prefill)
{
	AttentionFixture f;
	SparkAttentionParams params;
	uint16_t *query, *gate, *output;
	uint32_t *sequence, *position, row, head, d, pass;
	double ref[ATT_DIM], tolerance[ATT_DIM];
	Check clean, corrupted;
	char shape[96];
	const uint32_t sequences = prefill != 0u ? 2u : 1u;
	memset(&f,0,sizeof(f));
	AttentionFill(&f,context,sequences);
	query = (uint16_t *)Shared(&f.query,(uint64_t)rows * ATT_HEADS * ATT_DIM * 2u,"attention query");
	gate = (uint16_t *)Shared(&f.gate,(uint64_t)rows * ATT_QKV * 2u,"attention gate");
	output = (uint16_t *)Shared(&f.output,(uint64_t)rows * ATT_HEADS * ATT_DIM * 2u,"attention output");
	Shared(&f.partial,SparkOpsAttentionScratchBytes(rows,ATT_HEADS,context),"attention partial");
	FillBf16(query,(uint64_t)rows * ATT_HEADS * ATT_DIM,1.0);
	FillBf16(gate,(uint64_t)rows * ATT_QKV,1.0);
	SparkDeviceFree(&f.sequence);
	SparkDeviceFree(&f.positions);
	sequence = (uint32_t *)Shared(&f.sequence,rows * 4u,"attention sequence");
	position = (uint32_t *)Shared(&f.positions,rows * 4u,"attention positions");
	for ( row = 0u; row < rows; row++ )
	{
		sequence[row] = prefill != 0u ? (row < rows / 2u ? 0u : 1u) : 0u;
		position[row] = prefill != 0u ? context - (rows / 2u) + (row % (rows / 2u)) : (row == 0u ? context - 1u : (uint32_t)(NextRandom() % context));
	}
	f.paged.sequence_of_row = Whole(&f.sequence);
	f.paged.positions = Whole(&f.positions);
	memset(&params,0,sizeof(params));
	params.rows = rows;
	params.heads = ATT_HEADS;
	params.kv_heads = ATT_KV;
	params.head_dimension = ATT_DIM;
	params.page_slots = ATT_SLOTS;
	params.pages_per_sequence = f.pages;
	params.pool_pages = sequences * f.pages;
	params.query_stride = ATT_HEADS * ATT_DIM;
	params.gate_stride = ATT_QKV;
	params.gate_offset = 256u;
	params.gate_head_stride = 512u;
	params.output_stride = ATT_HEADS * ATT_DIM;
	params.scale = 1.0f / 16.0f;
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		uint16_t *keys = (uint16_t *)f.keys.host;
		const uint64_t poke = ((uint64_t)(sequences - 1u) * f.context + (context - 1u) / 2u) * ATT_KV * ATT_DIM + 2u * ATT_DIM + 17u;
		const uint16_t saved = keys[poke];
		SparkKvParams store;
		memset(check,0,sizeof(*check));
		uint16_t *values = (uint16_t *)f.values.host;
		const uint64_t value_poke = (poke / (ATT_KV * ATT_DIM)) * ATT_QKV + 13312u + 2u * ATT_DIM + 17u;
		const uint16_t value_saved = values[value_poke];
		if ( pass == 1u )
		{
			keys[poke] = ToBf16(Bf16(saved) + 64.0f);
			values[value_poke] = ToBf16(Bf16(value_saved) + 4.0f);
			memset(&store,0,sizeof(store));
			store.rows = 1u;
			store.kv_heads = ATT_KV;
			store.head_dimension = ATT_DIM;
			store.page_slots = ATT_SLOTS;
			store.pages_per_sequence = f.pages;
			store.key_stride = ATT_KV * ATT_DIM;
			store.value_stride = ATT_QKV;
			store.value_offset = 13312u;
			store.pool_pages = sequences * f.pages;
			{
				SparkDeviceBuffer one_sequence, one_position;
				SparkOpsPaged single = f.paged;
				*(uint32_t *)Shared(&one_sequence,4u,"poke sequence") = sequences - 1u;
				*(uint32_t *)Shared(&one_position,4u,"poke position") = (context - 1u) / 2u;
				single.sequence_of_row = Whole(&one_sequence);
				single.positions = Whole(&one_position);
				Run(SparkOpsKvStore(&ops,queue,&single,SparkTensorSlice(Whole(&f.keys),(poke / (ATT_KV * ATT_DIM)) * ATT_KV * ATT_DIM * 2u,ATT_KV * ATT_DIM * 2u),
					SparkTensorSlice(Whole(&f.values),(poke / (ATT_KV * ATT_DIM)) * ATT_QKV * 2u,ATT_QKV * 2u),&store),"kv_store poke");
				SparkDeviceFree(&one_sequence);
				SparkDeviceFree(&one_position);
			}
			keys[poke] = saved;
			values[value_poke] = value_saved;
		}
		if ( !Run(SparkOpsAttention(&ops,queue,&f.paged,Whole(&f.query),Whole(&f.gate),Whole(&f.partial),Whole(&f.output),&params,position,sequence),
			"attention") )
			break;
		for ( row = 0u; row < rows; row++ )
			for ( head = 0u; head < ATT_HEADS; head++ )
			{
				AttentionReference(&f,query + (uint64_t)row * ATT_HEADS * ATT_DIM,gate + (uint64_t)row * ATT_QKV,sequence[row],position[row],head,ref,tolerance);
				for ( d = 0u; d < ATT_DIM; d++ )
					Observe(check,Bf16(output[((uint64_t)row * ATT_HEADS + head) * ATT_DIM + d]),ref[d],tolerance[d]);
			}
	}
	snprintf(shape,sizeof(shape),"%s rows %u, context %u, 24 q / 4 kv heads x 256",prefill != 0u ? "prefill" : "decode",rows,context);
	Report(prefill != 0u ? "attention_prefill" : "attention_decode",shape,&clean,&corrupted,
		"u|y| + sigmoid(g) bf16 flip slack of o within e + 4*16 eps32 |y|; e = 2 sum p|v| 2 ds + gamma_L sum p|v|, ds = gamma_258 sum|q k|/16 + 2*16 eps32 (1+|s|)");
	AttentionFree(&f);
	SparkDeviceFree(&f.query);
	SparkDeviceFree(&f.gate);
	SparkDeviceFree(&f.partial);
	SparkDeviceFree(&f.output);
}

static void TestSiluMul(uint32_t rows)
{
	enum { DIM = 17408 };
	SparkDeviceBuffer input, output;
	uint16_t *in = (uint16_t *)Shared(&input,(uint64_t)rows * 2u * DIM * 2u,"silu input");
	uint16_t *out = (uint16_t *)Shared(&output,(uint64_t)rows * DIM * 2u,"silu output");
	SparkSiluMulParams params;
	Check clean, corrupted;
	char shape[64];
	uint32_t pass, row, i;
	FillBf16(in,(uint64_t)rows * 2u * DIM,2.0);
	params.rows = rows;
	params.dimension = DIM;
	params.input_stride = 2u * DIM;
	params.output_stride = DIM;
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		const uint64_t poke = (uint64_t)(rows - 1u) * 2u * DIM + DIM + 4321u;
		const uint16_t saved = in[poke];
		memset(check,0,sizeof(*check));
		if ( pass == 1u )
			in[poke] = ToBf16(Bf16(saved) * -1.0f + 1.0f);
		if ( !Run(SparkOpsSiluMul(&ops,queue,Whole(&input),Whole(&output),&params),"silu_mul") )
			break;
		in[poke] = saved;
		for ( row = 0u; row < rows; row++ )
			for ( i = 0u; i < DIM; i++ )
			{
				const double y = Silu(Bf16(in[(uint64_t)row * 2u * DIM + i])) * Bf16(in[(uint64_t)row * 2u * DIM + DIM + i]);
				Observe(check,Bf16(out[(uint64_t)row * DIM + i]),y,UNIT_BF16 * fabs(y) + 3.0 * FUNCTION_ERROR * fabs(y));
			}
	}
	snprintf(shape,sizeof(shape),"rows %u, 17408 gate | 17408 up",rows);
	Report("silu_mul",shape,&clean,&corrupted,"u|y| + 3*16 eps32 |y|");
	SparkDeviceFree(&input);
	SparkDeviceFree(&output);
}

static void TestAddRows(uint32_t rows)
{
	enum { DIM = 5120 };
	SparkDeviceBuffer hidden, addend;
	uint16_t *h = (uint16_t *)Shared(&hidden,(uint64_t)rows * DIM * 2u,"add hidden");
	uint16_t *a = (uint16_t *)Shared(&addend,(uint64_t)rows * DIM * 2u,"add addend");
	uint16_t *original = (uint16_t *)malloc((size_t)rows * DIM * 2u);
	Check clean, corrupted;
	char shape[64];
	uint32_t pass, i;
	FillBf16(h,(uint64_t)rows * DIM,4.0);
	FillBf16(a,(uint64_t)rows * DIM,1.0);
	memcpy(original,h,(size_t)rows * DIM * 2u);
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		memset(check,0,sizeof(*check));
		memcpy(h,original,(size_t)rows * DIM * 2u);
		if ( pass == 1u )
			a[(uint64_t)(rows - 1u) * DIM + 11u] = ToBf16(Bf16(a[(uint64_t)(rows - 1u) * DIM + 11u]) + 2.0f);
		if ( !Run(SparkOpsAddRows(&ops,queue,Whole(&hidden),Whole(&addend),rows,DIM,DIM,DIM),"add_rows") )
			break;
		if ( pass == 1u )
			a[(uint64_t)(rows - 1u) * DIM + 11u] = ToBf16(Bf16(a[(uint64_t)(rows - 1u) * DIM + 11u]) - 2.0f);
		for ( i = 0u; i < rows * DIM; i++ )
			Observe(check,Bf16(h[i]),Bf16(ToBf16(Bf16(original[i]) + Bf16(a[i]))),0.0);
	}
	snprintf(shape,sizeof(shape),"rows %u, dim 5120",rows);
	Report("add_rows",shape,&clean,&corrupted,"exact: bf16(fp32 sum of two bf16 values) is emulated on the host");
	free(original);
	SparkDeviceFree(&hidden);
	SparkDeviceFree(&addend);
}

static void TestLogprob(uint32_t rows)
{
	enum { VOCAB = 248320 };
	SparkDeviceBuffer logits, targets, result, argmax;
	float *l = (float *)Shared(&logits,(uint64_t)rows * VOCAB * 4u,"logprob logits");
	uint32_t *t = (uint32_t *)Shared(&targets,rows * 4u,"logprob targets");
	float *r = (float *)Shared(&result,rows * 16u,"logprob result");
	uint32_t *a = (uint32_t *)Shared(&argmax,rows * 4u,"logprob argmax");
	SparkLogprobParams params;
	Check clean, corrupted;
	char shape[64];
	uint32_t pass, row, i;
	for ( i = 0u; i < rows * VOCAB; i++ )
		l[i] = (float)(Normal() * 3.0);
	for ( row = 0u; row < rows; row++ )
	{
		t[row] = (uint32_t)(NextRandom() % VOCAB);
		l[(uint64_t)row * VOCAB + 1000u + row] = 30.0f;
		l[(uint64_t)row * VOCAB + 2000u + row] = 30.0f;
	}
	memset(&params,0,sizeof(params));
	params.rows = rows;
	params.vocabulary = VOCAB;
	params.logit_stride = VOCAB;
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		const uint64_t poke = (uint64_t)(rows - 1u) * VOCAB + 77777u;
		const float saved = l[poke];
		memset(check,0,sizeof(*check));
		if ( pass == 1u )
			l[poke] = 31.0f;
		if ( !Run(SparkOpsLogprob(&ops,queue,Whole(&logits),Whole(&targets),Whole(&result),Whole(&argmax),&params),"logprob") )
			break;
		l[poke] = saved;
		for ( row = 0u; row < rows; row++ )
		{
			const float *values = l + (uint64_t)row * VOCAB;
			double best = -INFINITY, total = 0.0;
			uint32_t best_index = 0u;
			for ( i = 0u; i < VOCAB; i++ )
				if ( values[i] > best )
				{
					best = values[i];
					best_index = i;
				}
			for ( i = 0u; i < VOCAB; i++ )
				total += exp(values[i] - best);
			const double normalizer = best + log(total), tolerance = Gamma(VOCAB) + 4.0 * FUNCTION_ERROR * (1.0 + fabs(normalizer));
			Observe(check,r[row * 4u + 1u],normalizer,tolerance);
			Observe(check,r[row * 4u + 2u],values[t[row]] - normalizer,tolerance + UNIT_F32 * fabs(values[t[row]]));
			Observe(check,a[row],best_index,0.0);
		}
	}
	snprintf(shape,sizeof(shape),"rows %u, vocabulary 248320, tied maximum",rows);
	Report("logprob_rows",shape,&clean,&corrupted,"log-normalizer within gamma_V + 4*16 eps32 (1+|lse|); argmax exact with the lowest index on ties");
	SparkDeviceFree(&logits);
	SparkDeviceFree(&targets);
	SparkDeviceFree(&result);
	SparkDeviceFree(&argmax);
}

int main(int argc, char **argv)
{
	if ( argc > 1 && (receipt = fopen(argv[1],"w")) == 0 )
	{
		printf("FAIL cannot write %s\n",argv[1]);
		return(1);
	}
	if ( receipt != 0 )
		fputs("[\n",receipt);
	if ( SparkDeviceInitialize(0) == SPARK_STATUS_OK && strcmp(SparkDeviceGetProperties()->target,"metal.apple10") != 0 )
	{
		printf("SKIP test_metal_kernels (the kernel library needs an Apple family 10 GPU; this device is %s)\n",SparkDeviceGetProperties()->target);
		SparkDeviceShutdown();
		return(0);
	}
	if ( SparkDeviceGetProperties() == 0 || SparkDeviceQueueCreate(&queue,"kernel tests") != SPARK_STATUS_OK ||
		SparkDeviceOpsOpen(&ops,SPARK_KERNEL_LIBRARY) != SPARK_STATUS_OK )
	{
		printf("FAIL device setup: %s\n",SparkDeviceLastError()->text);
		return(1);
	}
	printf("device %s (%s), %u compute units\n",SparkDeviceGetProperties()->name,SparkDeviceGetProperties()->target,SparkDeviceGetProperties()->compute_units);
	TestGather();
	TestRmsNormResidual(1u);
	TestRmsNormResidual(64u);
	TestRmsNormGate(1u);
	TestRmsNormGate(33u);
	TestHeadRope(1u,24u,0u,512u,"24 query heads in the fused row");
	TestHeadRope(17u,4u,12288u,256u,"4 key heads in the fused row");
	TestLinear(1u,5120u,16384u,0u,0u,UINT64_MAX);
	TestLinear(1u,5120u,96u,0u,0u,UINT64_MAX);
	TestLinear(1u,5120u,14336u,0u,0u,UINT64_MAX);
	TestLinear(1u,6144u,5120u,0u,0u,UINT64_MAX);
	TestLinear(1u,5120u,34816u,0u,0u,UINT64_MAX);
	TestLinear(1u,17408u,5120u,0u,1u,UINT64_MAX);
	TestLinear(2u,17408u,5120u,0u,1u,UINT64_MAX);
	TestLinear(3u,17408u,5120u,0u,0u,UINT64_MAX);
	TestLinear(7u,5120u,1000u,0u,0u,UINT64_MAX);
	TestLinear(8u,5120u,1000u,0u,0u,UINT64_MAX);
	TestLinear(2u,6144u,5120u,0u,0u,UINT64_MAX);
	TestLinear(4u,17408u,5120u,0u,0u,UINT64_MAX);
	TestLinear(5u,17408u,5120u,0u,0u,UINT64_MAX);
	TestLinear(16u,5120u,96u,0u,0u,UINT64_MAX);
	TestLinear(17u,6144u,5120u,0u,0u,UINT64_MAX);
	TestLinear(64u,5120u,34816u,0u,0u,20000u);
	TestLinear(2u,5120u,248320u,1u,0u,UINT64_MAX);
	TestLinear(1u,5120u,248320u,1u,0u,UINT64_MAX);
	TestLinear(130u,5120u,200u,0u,0u,UINT64_MAX);
	TestLinear(512u,5120u,16384u,0u,0u,20000u);
	TestLinear(512u,5120u,96u,0u,0u,UINT64_MAX);
	TestLinear(512u,6144u,5120u,0u,0u,20000u);
	TestLinear(512u,17408u,5120u,0u,0u,20000u);
	TestLinear(77u,17408u,5120u,0u,0u,20000u);
	TestLinear(2048u,5120u,34816u,0u,0u,20000u);
	TestLinear(64u,5120u,248320u,1u,0u,20000u);
	TestLinearRowIndependence(2u,1u,17408u,5120u,0u,1u);
	TestLinearRowIndependence(3u,64u,5120u,34816u,0u,0u);
	TestLinearRowIndependence(2u,1u,5120u,248320u,1u,0u);
	TestLinearRowIndependence(64u,5u,17408u,5120u,0u,0u);
	TestLinearRowIndependence(16u,9u,5120u,96u,0u,0u);
	TestLinearRowIndependence(32u,17u,5120u,16384u,0u,0u);
	TestLinearRowIndependence(32u,16u,6144u,5120u,0u,0u);
	TestLinearRowIndependence(40u,8u,5120u,248320u,1u,0u);
	TestCausalConv();
	TestGatedDelta();
	TestAttention(1u,1u,0u);
	TestAttention(1000u,1u,0u);
	TestAttention(8192u,4u,0u);
	TestAttention(300u,40u,1u);
	TestAttention(2048u,64u,1u);
	TestSiluMul(1u);
	TestSiluMul(10u);
	TestAddRows(1u);
	TestAddRows(9u);
	TestLogprob(1u);
	TestLogprob(3u);
	if ( receipt != 0 )
	{
		fputs("\n]\n",receipt);
		fclose(receipt);
	}
	SparkDeviceOpsClose(&ops);
	SparkDeviceQueueDestroy(queue);
	if ( failures != 0 )
	{
		printf("metal kernels: %d failure(s)\n",failures);
		return(1);
	}
	printf("PASS metal kernels: every operation matches its float64 reference and every corrupted input fails\n");
	return(0);
}
