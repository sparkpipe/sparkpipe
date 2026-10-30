#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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

static void TestRmsNormWeightless(uint32_t rows)
{
	enum { DIM = 6656 };
	SparkDeviceBuffer hidden;
	uint16_t *h = (uint16_t *)Shared(&hidden,(uint64_t)rows * DIM * 2u,"weightless norm hidden");
	uint16_t *original = (uint16_t *)malloc((size_t)rows * DIM * 2u);
	SparkNormParams params;
	Check clean, corrupted;
	char shape[64];
	uint32_t pass, row, i;
	FillBf16(h,(uint64_t)rows * DIM,3.0);
	memcpy(original,h,(size_t)rows * DIM * 2u);
	memset(&params,0,sizeof(params));
	params.rows = rows;
	params.dimension = DIM;
	params.input_stride = params.output_stride = DIM;
	params.epsilon = 1e-5f;
	params.weightless = 1u;
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		memset(check,0,sizeof(*check));
		memcpy(h,original,(size_t)rows * DIM * 2u);
		if ( pass == 1u )
			h[(uint64_t)(rows - 1u) * DIM + 9u] = ToBf16(Bf16(original[(uint64_t)(rows - 1u) * DIM + 9u]) * -4.0f);
		if ( !Run(SparkOpsRmsNormResidual(&ops,queue,Whole(&hidden),0,Whole(&hidden),Whole(&hidden),&params),"rmsnorm_residual weightless") )
			break;
		for ( row = 0u; row < rows; row++ )
		{
			double square = 0.0;
			for ( i = 0u; i < DIM; i++ )
				square += (double)Bf16(original[(uint64_t)row * DIM + i]) * Bf16(original[(uint64_t)row * DIM + i]);
			const double scale = 1.0 / sqrt(square / DIM + 1e-5);
			const double scale_error = Gamma(DIM) + FUNCTION_ERROR;
			for ( i = 0u; i < DIM; i++ )
			{
				const double y = Bf16(original[(uint64_t)row * DIM + i]) * scale;
				Observe(check,Bf16(h[(uint64_t)row * DIM + i]),y,UNIT_BF16 * fabs(y) + (1.0 + UNIT_BF16) * (scale_error + 2.0 * UNIT_F32) * fabs(y));
			}
		}
	}
	snprintf(shape,sizeof(shape),"rows %u, dim 6656, in place, no weight",rows);
	Report("rmsnorm_weightless",shape,&clean,&corrupted,"u|y| + (1+u)(gamma_6656 + 16 eps32 + 2 eps32)|y|, u = 2^-8, eps32 = 2^-24");
	free(original);
	SparkDeviceFree(&hidden);
}

static void TestRmsNormGate(uint32_t rows, uint32_t sigmoid_gate)
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
	params.sigmoid_gate = sigmoid_gate;
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
					const double activation = sigmoid_gate != 0u ? Sigmoid(z) : Silu(z), y = weighted * activation;
					const double tolerance = UNIT_BF16 * fabs(y) + fabs(activation) * second + (2.0 * FUNCTION_ERROR + 2.0 * UNIT_F32) * fabs(y);
					Observe(check,Bf16(o[(uint64_t)row * STRIDE + head * HEAD + i]),y,tolerance);
				}
			}
	}
	snprintf(shape,sizeof(shape),"rows %u, 48 heads x 128, %s gate in fused row",rows,sigmoid_gate != 0u ? "sigmoid" : "silu");
	Report("rmsnorm_gate",shape,&clean,&corrupted,"u|y| + |act(z)| (flip slack of both bf16 roundings, propagated) + (2*16 + 2) eps32 |y|");
	SparkDeviceFree(&input);
	SparkDeviceFree(&gate);
	SparkDeviceFree(&weight);
	SparkDeviceFree(&out);
}

typedef struct RopeCase
{
	uint32_t head;
	uint32_t rope;
	uint32_t pairs;
	double theta;
	uint32_t weighted;
	uint32_t rounded;
	double scale;
	double multiply;
	uint32_t interleaved;
	uint32_t output_head_stride;
	uint32_t plain;
} RopeCase;

static void TestHeadRopeCase(uint32_t rows, uint32_t heads, uint32_t offset, uint32_t head_stride, const RopeCase *c, const char *label)
{
	const uint32_t HEAD = c->head, ROPE = c->rope, HALF = c->rope / 2u, STRIDE = offset + heads * head_stride;
	const uint32_t OUT = c->output_head_stride != 0u ? c->output_head_stride : HEAD;
	SparkDeviceBuffer input, weight, positions, out;
	uint16_t *x = (uint16_t *)Shared(&input,(uint64_t)rows * STRIDE * sizeof(uint16_t),"rope input");
	float *w = (float *)Shared(&weight,HEAD * sizeof(float),"rope weight");
	uint32_t *pos = (uint32_t *)Shared(&positions,rows * sizeof(uint32_t),"rope positions");
	uint16_t *o = (uint16_t *)Shared(&out,(uint64_t)rows * heads * OUT * sizeof(uint16_t),"rope out");
	double *normed = (double *)malloc(HEAD * sizeof(double)), *slack = (double *)malloc(HEAD * sizeof(double));
	SparkHeadRopeParams params;
	Check clean, corrupted;
	char shape[128];
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
	params.output_stride = heads * OUT;
	params.output_head_stride = c->output_head_stride;
	params.interleaved = c->interleaved;
	params.plain = c->plain;
	params.rope_dimension = ROPE;
	params.epsilon = 1e-6f;
	params.has_weight = c->weighted;
	params.rope_rounded_terms = c->rounded;
	params.scale = (float)c->scale;
	params.multiply = (float)c->multiply;
	for ( i = 0u; i < c->pairs; i++ )
		params.inverse_frequency[i] = (float)(1.0 / pow(c->theta,(double)(2u * i) / ROPE));
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
				double square = 0.0;
				for ( i = 0u; i < HEAD; i++ )
					square += (double)Bf16(v[i]) * Bf16(v[i]);
				const double scale = c->plain != 0u ? 1.0 : 1.0 / sqrt(square / HEAD + 1e-6);
				for ( i = 0u; i < HEAD; i++ )
				{
					const double gain = c->weighted != 0u ? w[i] : 1.0, multiply = params.multiply;
					const double first = RoundBf16(Bf16(v[i]) * scale * gain);
					const double first_slack = FlipSlack(Bf16(v[i]) * scale * gain,Gamma(HEAD) + FUNCTION_ERROR + 3.0 * UNIT_F32);
					normed[i] = multiply != 0.0 ? RoundBf16(first * multiply) : first;
					slack[i] = multiply != 0.0 ? fabs(multiply) * first_slack + FlipSlack(first * multiply,2.0 * UNIT_F32) +
						(first_slack != 0.0 ? UNIT_BF16 * fabs(normed[i]) : 0.0) : first_slack;
				}
				for ( i = 0u; i < HEAD; i++ )
				{
					double y = normed[i], tolerance = slack[i];
					if ( i < ROPE )
					{
						const uint32_t low = c->interleaved != 0u ? i % 2u == 0u : i < HALF, pair = c->interleaved != 0u ? i / 2u : i % HALF;
						const uint32_t other = c->interleaved != 0u ? (low ? i + 1u : i - 1u) : (low ? i + HALF : i - HALF);
						const double angle = (double)(float)((float)pos[row] * params.inverse_frequency[pair]);
						const double partner = low ? -normed[other] : normed[other];
						const double scale = c->scale != 0.0 ? c->scale : 1.0;
						if ( c->rounded != 0u )
						{
							const double cosine = RoundBf16(cos(angle) * scale), sine = RoundBf16(sin(angle) * scale);
							y = RoundBf16(RoundBf16(normed[i] * cosine) + RoundBf16(partner * sine));
							tolerance = UNIT_BF16 * (fabs(y) + fabs(normed[i] * cosine) + fabs(partner * sine) + fabs(normed[i]) + fabs(partner)) +
								slack[i] + slack[other];
						}
						else
						{
							y = (normed[i] * cos(angle) + partner * sin(angle)) * scale;
							tolerance = UNIT_BF16 * fabs(y) + (slack[i] * fabs(cos(angle)) + slack[other] * fabs(sin(angle)) +
								(4.0 * FUNCTION_ERROR + 4.0 * UNIT_F32) * (fabs(normed[i]) + fabs(partner))) * scale;
						}
					}
					Observe(check,Bf16(o[(uint64_t)row * heads * OUT + head * OUT + i]),y,tolerance);
				}
			}
	}
	snprintf(shape,sizeof(shape),"rows %u, %s",rows,label);
	Report("head_rmsnorm_rope",shape,&clean,&corrupted,c->rounded != 0u ?
		"bf16 cos, sin and products: u (|y| + |x cos| + |p sin| + |x| + |p|) + flip slack of both normed elements" :
		"rotary span: u|y| + flip slack of the normed bf16 rounding (both elements) + (4*16+4) eps32 (|x|+|partner|), times the rope scale; "
		"elsewhere the flip slack only");
	free(normed);
	free(slack);
	SparkDeviceFree(&input);
	SparkDeviceFree(&weight);
	SparkDeviceFree(&positions);
	SparkDeviceFree(&out);
}

static void TestHeadRope(uint32_t rows, uint32_t heads, uint32_t offset, uint32_t head_stride, const char *label)
{
	static const RopeCase qwen = { 256u, 64u, 32u, 10000000.0, 1u, 0u, 0.0, 0.0, 0u, 0u, 0u };
	char text[96];
	snprintf(text,sizeof(text),"%s, rope 64, theta 1e7",label);
	TestHeadRopeCase(rows,heads,offset,head_stride,&qwen,text);
}

static uint32_t SampleIndex(uint64_t index, uint64_t total, uint64_t samples)
{
	if ( samples >= total )
		return((uint32_t)index);
	return((uint32_t)((index * 2654435761ull + 12345u) % total));
}

static void TestLinear(uint32_t rows, uint32_t K, uint32_t N, uint32_t output_f32, uint32_t projection, uint64_t samples)
{
	SparkDeviceBuffer x, w, y;
	uint16_t *xv = (uint16_t *)Shared(&x,(uint64_t)rows * K * 2u,"linear x");
	uint16_t *wv = (uint16_t *)Shared(&w,(uint64_t)N * K * 2u,"linear w");
	void *yv = Shared(&y,(uint64_t)rows * N * (output_f32 != 0u ? 4u : 2u),"linear y");
	SparkGemvParams params;
	Check clean, corrupted;
	char shape[96];
	uint64_t total = (uint64_t)rows * N, index;
	uint32_t pass, k;
	FillBf16(xv,(uint64_t)rows * K,1.0);
	FillBf16(wv,(uint64_t)N * K,1.0 / sqrt((double)K));
	memset(&params,0,sizeof(params));
	params.rows = rows;
	params.input_dimension = K;
	params.output_dimension = N;
	params.input_stride = K;
	params.output_stride = N;
	params.output_f32 = output_f32;
	params.projection = projection;
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		const uint64_t poke = (uint64_t)(rows - 1u) * K + K / 2u;
		const uint16_t saved = xv[poke];
		memset(check,0,sizeof(*check));
		if ( pass == 1u )
			xv[poke] = ToBf16(Bf16(saved) + (Bf16(saved) >= 0.0f ? -64.0f : 64.0f));
		if ( !Run(SparkOpsLinear(&ops,queue,Whole(&x),Whole(&w),Whole(&y),&params),"linear") )
			break;
		xv[poke] = saved;
		for ( index = 0u; index < (samples < total ? samples : total); index++ )
		{
			const uint64_t element = SampleIndex(index,total,samples);
			const uint32_t row = (uint32_t)(element / N), n = (uint32_t)(element % N);
			double sum = 0.0, magnitude = 0.0, got;
			for ( k = 0u; k < K; k++ )
			{
				const double term = Bf16(xv[(uint64_t)row * K + k]) * Bf16(wv[(uint64_t)n * K + k]);
				sum += term;
				magnitude += fabs(term);
			}
			got = output_f32 != 0u ? ((float *)yv)[element] : Bf16(((uint16_t *)yv)[element]);
			Observe(check,got,sum,(output_f32 != 0u ? 0.0 : UNIT_BF16 * fabs(sum)) + (1.0 + UNIT_BF16) * Gamma(K) * magnitude);
		}
	}
	snprintf(shape,sizeof(shape),"rows %u, K %u -> N %u%s%s",rows,K,N,output_f32 != 0u ? ", f32 out" : "",samples < total ? ", sampled" : "");
	Report(rows > SPARK_OPS_DECODE_MAX_ROWS ? "gemm" : projection == SPARK_PROJECTION_ROW_TILES || rows > SPARK_OPS_GEMV_MAX_ROWS ? "gemm rows" : "gemv",shape,
		&clean,&corrupted,output_f32 != 0u ? "gamma_K sum|x w|: fp32 accumulation of exact bf16 products" : "u|y| + (1+u) gamma_K sum|x w|, u = 2^-8");
	SparkDeviceFree(&x);
	SparkDeviceFree(&w);
	SparkDeviceFree(&y);
}

static double E4m3(uint8_t code)
{
	const uint32_t exponent = (code >> 3u) & 15u, mantissa = code & 7u;
	const double magnitude = exponent == 0u ? ldexp((double)mantissa,-9) : ldexp(1.0 + (double)mantissa / 8.0,(int)exponent - 7);
	return((code & 0x80u) != 0u ? -magnitude : magnitude);
}

static void TestLinearFp8(uint32_t rows, uint32_t K, uint32_t N, uint32_t row_exact, uint64_t samples)
{
	const uint32_t blocks = (K + SPARK_FP8_SCALE_K - 1u) / SPARK_FP8_SCALE_K;
	const uint32_t rounded = rows > SPARK_OPS_FP8_GEMV_ROWS && row_exact == 0u ? 1u : 0u;
	SparkDeviceBuffer x, w, scale, dequant, y;
	uint16_t *xv = (uint16_t *)Shared(&x,(uint64_t)rows * K * 2u,"fp8 linear x");
	uint8_t *wv = (uint8_t *)Shared(&w,(uint64_t)N * K,"fp8 linear w");
	float *sv = (float *)Shared(&scale,(uint64_t)N * blocks * sizeof(float),"fp8 linear scale");
	uint16_t *yv = (uint16_t *)Shared(&y,(uint64_t)rows * N * 2u,"fp8 linear y");
	SparkGemvParams params;
	Check clean, corrupted;
	char shape[96];
	uint64_t total = (uint64_t)rows * N, index;
	uint32_t pass, k;
	Shared(&dequant,(uint64_t)N * K * 2u,"fp8 linear dequant");
	FillBf16(xv,(uint64_t)rows * K,1.0);
	for ( index = 0u; index < (uint64_t)N * K; index++ )
	{
		uint8_t code = (uint8_t)(NextRandom() & 0xffu);
		wv[index] = (code & 0x7fu) == 0x7fu ? (uint8_t)(code ^ 1u) : code;
	}
	for ( index = 0u; index < (uint64_t)N * blocks; index++ )
		sv[index] = (float)ldexp(1.0 + 0.5 * (double)(NextRandom() % 1024u) / 1024.0,-8 - (int)(NextRandom() % 4u));
	memset(&params,0,sizeof(params));
	params.rows = rows;
	params.input_dimension = K;
	params.output_dimension = N;
	params.input_stride = K;
	params.output_stride = N;
	params.projection = SPARK_PROJECTION_ROW_TILES;
	params.row_exact = row_exact;
	params.scale_stride = blocks;
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		const uint64_t poke = (uint64_t)(rows - 1u) * K + K / 2u;
		const uint16_t saved = xv[poke];
		memset(check,0,sizeof(*check));
		if ( pass == 1u )
			xv[poke] = ToBf16(Bf16(saved) + (Bf16(saved) >= 0.0f ? -64.0f : 64.0f));
		if ( !Run(SparkOpsLinearFp8(&ops,queue,Whole(&x),Whole(&w),Whole(&scale),Whole(&dequant),Whole(&y),&params),"linear fp8") )
			break;
		xv[poke] = saved;
		if ( pass == 0u && rounded != 0u )
		{
			uint64_t wrong = 0u;
			for ( index = 0u; index < (uint64_t)N * K; index++ )
				if ( ((const uint16_t *)dequant.host)[index] != ToBf16((float)E4m3(wv[index]) * sv[(index / K) * blocks + (index % K) / SPARK_FP8_SCALE_K]) )
					wrong++;
			if ( wrong != 0u )
			{
				printf("FAIL linear_fp8 dequant: %llu of %llu weights are not bf16(e4m3 x scale)\n",(unsigned long long)wrong,(unsigned long long)N * K);
				failures++;
			}
		}
		for ( index = 0u; index < (samples < total ? samples : total); index++ )
		{
			const uint64_t element = SampleIndex(index,total,samples);
			const uint32_t row = (uint32_t)(element / N), n = (uint32_t)(element % N);
			double sum = 0.0, magnitude = 0.0;
			for ( k = 0u; k < K; k++ )
			{
				const float block = sv[(uint64_t)n * blocks + k / SPARK_FP8_SCALE_K];
				const double weight = rounded != 0u ? RoundBf16((double)(float)((float)E4m3(wv[(uint64_t)n * K + k]) * block)) : E4m3(wv[(uint64_t)n * K + k]) * block;
				const double term = Bf16(xv[(uint64_t)row * K + k]) * weight;
				sum += term;
				magnitude += fabs(term);
			}
			Observe(check,Bf16(yv[element]),sum,UNIT_BF16 * fabs(sum) + (1.0 + UNIT_BF16) * Gamma(K + K / SPARK_FP8_CHUNK + 2u) * magnitude);
		}
	}
	snprintf(shape,sizeof(shape),"rows %u, K %u -> N %u%s%s",rows,K,N,rounded != 0u ? ", bf16 dequant" : ", per-chunk scale",
		samples < total ? ", sampled" : "");
	Report(rounded != 0u ? "linear_fp8 dequant" : row_exact != 0u ? "linear_fp8 exact" : "linear_fp8 gemv",shape,&clean,&corrupted,
		rounded != 0u ? "u|y| + (1+u) gamma_K sum|x bf16(w s)|" : "u|y| + (1+u) gamma_(K + K/16 + 2) sum|x w s|, u = 2^-8");
	SparkDeviceFree(&x);
	SparkDeviceFree(&w);
	SparkDeviceFree(&scale);
	SparkDeviceFree(&dequant);
	SparkDeviceFree(&y);
}

static void TestLinearRowIndependence(uint32_t rows, uint32_t other_rows, uint32_t K, uint32_t N, uint32_t output_f32, uint32_t projection,
	uint32_t exact)
{
	const uint32_t width = K, element = output_f32 != 0u ? (uint32_t)sizeof(float) : (uint32_t)sizeof(uint16_t);
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
	params.projection = projection;
	params.row_exact = exact;
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
	snprintf(shape,sizeof(shape),"rows %u vs %u, K %u -> N %u%s%s%s",rows,other_rows,K,N,output_f32 != 0u ? ", f32 out" : "",
		projection == SPARK_PROJECTION_ROW_TILES ? ", row tiles" : "",exact != 0u ? ", row-exact" : "");
	Report("linear row independence",shape,&clean,&corrupted,"exact: a row's output bits do not depend on the other rows or the row count");
	SparkDeviceFree(&x);
	SparkDeviceFree(&w);
	SparkDeviceFree(&y);
	SparkDeviceFree(&ox);
	SparkDeviceFree(&oy);
}

static double GeluTanh(double x)
{
	const double cube = x * x * x;
	return(0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * cube))));
}

static void TestGeluMul(uint32_t rows)
{
	enum { DIM = 2112 };
	SparkDeviceBuffer input, output;
	uint16_t *in = (uint16_t *)Shared(&input,(uint64_t)rows * 2u * DIM * sizeof(uint16_t),"gelu input");
	uint16_t *out = (uint16_t *)Shared(&output,(uint64_t)rows * DIM * sizeof(uint16_t),"gelu output");
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
		const uint64_t poke = (uint64_t)(rows - 1u) * 2u * DIM + DIM + 777u;
		const uint16_t saved = in[poke];
		memset(check,0,sizeof(*check));
		if ( pass == 1u )
			in[poke] = ToBf16(Bf16(saved) * -1.0f + 1.0f);
		if ( !Run(SparkOpsGatedMul(&ops,queue,Whole(&input),Whole(&output),&params,SPARK_GATED_GELU_TANH),"gelu_tanh_mul") )
			break;
		in[poke] = saved;
		for ( row = 0u; row < rows; row++ )
			for ( i = 0u; i < DIM; i++ )
			{
				const double gate = Bf16(in[(uint64_t)row * 2u * DIM + i]), up = Bf16(in[(uint64_t)row * 2u * DIM + DIM + i]);
				const double activated = GeluTanh(gate), y = RoundBf16(activated) * up;
				const double error = 0.5 * fabs(gate) * (FUNCTION_ERROR + 8.0 * UNIT_F32) + 8.0 * UNIT_F32 * fabs(activated);
				const double flip = fmax(fabs(RoundBf16(activated - error) - RoundBf16(activated)),fabs(RoundBf16(activated + error) - RoundBf16(activated)));
				const double tolerance = UNIT_BF16 * fabs(y) + fabs(up) * flip + 3.0 * UNIT_F32 * fabs(y);
				Observe(check,Bf16(out[(uint64_t)row * DIM + i]),y,tolerance);
			}
	}
	snprintf(shape,sizeof(shape),"rows %u, 2112 gate | 2112 up",rows);
	Report("gelu_tanh_mul",shape,&clean,&corrupted,"u|y| + |up| flip slack of bf16(gelu) for an absolute error 0.5|g|(16+8) eps32 + 8 eps32 |gelu| (the 1 + tanh cancellation) + 3 eps32 |y|");
	SparkDeviceFree(&input);
	SparkDeviceFree(&output);
}

static void TestGatedMulRouted(uint32_t routes, uint32_t activation)
{
	enum { DIM = 704 };
	const char *label = activation == SPARK_GATED_GELU_TANH ? "gelu_tanh_mul_routed" : activation == SPARK_GATED_SILU_CLAMP ? "silu_clamp_mul_routed" : "silu_mul_routed";
	const double limit = activation == SPARK_GATED_SILU_CLAMP ? 4.0 : INFINITY;
	SparkDeviceBuffer input, weights, output;
	uint16_t *in = (uint16_t *)Shared(&input,(uint64_t)routes * 2u * DIM * sizeof(uint16_t),"routed gate | up");
	float *w = (float *)Shared(&weights,(uint64_t)routes * sizeof(float),"route weights");
	uint16_t *out = (uint16_t *)Shared(&output,(uint64_t)routes * DIM * sizeof(uint16_t),"routed activation");
	SparkSiluMulParams params;
	Check clean, corrupted;
	char shape[64];
	uint32_t pass, row, i;
	FillBf16(in,(uint64_t)routes * 2u * DIM,activation == SPARK_GATED_SILU_CLAMP ? 3.0 : 2.0);
	for ( row = 0u; row < routes; row++ )
		w[row] = (float)(0.05 + 0.3 * Uniform());
	memset(&params,0,sizeof(params));
	params.rows = routes;
	params.dimension = DIM;
	params.input_stride = 2u * DIM;
	params.output_stride = DIM;
	params.limit = (float)limit;
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		const uint32_t poke = routes - 1u;
		const float saved = w[poke];
		memset(check,0,sizeof(*check));
		if ( pass == 1u )
			w[poke] = saved * 3.0f;
		if ( !Run(SparkOpsGatedMulRouted(&ops,queue,Whole(&input),Whole(&weights),Whole(&output),&params,activation),label) )
			break;
		w[poke] = saved;
		for ( row = 0u; row < routes; row++ )
			for ( i = 0u; i < DIM; i++ )
			{
				const double gate = Bf16(in[(uint64_t)row * 2u * DIM + i]), up = Bf16(in[(uint64_t)row * 2u * DIM + DIM + i]), weight = w[row];
				if ( activation == SPARK_GATED_GELU_TANH )
				{
					const double activated = GeluTanh(gate), y = RoundBf16(activated) * up * weight;
					const double error = 0.5 * fabs(gate) * (FUNCTION_ERROR + 8.0 * UNIT_F32) + 8.0 * UNIT_F32 * fabs(activated);
					const double flip = fmax(fabs(RoundBf16(activated - error) - RoundBf16(activated)),fabs(RoundBf16(activated + error) - RoundBf16(activated)));
					Observe(check,Bf16(out[(uint64_t)row * DIM + i]),y,UNIT_BF16 * fabs(y) + fabs(up * weight) * flip + 4.0 * UNIT_F32 * fabs(y));
				}
				else
				{
					const double y = fmin(Silu(gate),limit) * fmin(fmax(up,-limit),limit) * weight;
					Observe(check,Bf16(out[(uint64_t)row * DIM + i]),y,UNIT_BF16 * fabs(y) + (3.0 * FUNCTION_ERROR + 2.0 * UNIT_F32) * fabs(y));
				}
			}
	}
	snprintf(shape,sizeof(shape),"%u routes, 704 gate | 704 up, route weights%s",routes,activation == SPARK_GATED_SILU_CLAMP ? ", limit 4" : "");
	Report(label,shape,&clean,&corrupted,activation == SPARK_GATED_GELU_TANH ?
		"the gelu_tanh_mul bound with the route weight folded in: u|y| + |up w| flip slack + 4 eps32 |y|" : "u|y| + (3*16 + 2) eps32 |y|");
	SparkDeviceFree(&input);
	SparkDeviceFree(&weights);
	SparkDeviceFree(&output);
}

static void TestScaleAndSoftcap(void)
{
	enum { COUNT = 3u * 2816u, LOGITS = 4096u };
	SparkDeviceBuffer rows, logits;
	uint16_t *x = (uint16_t *)Shared(&rows,COUNT * sizeof(uint16_t),"scale rows");
	float *z = (float *)Shared(&logits,LOGITS * sizeof(float),"softcap logits");
	uint16_t *saved = (uint16_t *)malloc(COUNT * sizeof(uint16_t));
	float *saved_logits = (float *)malloc(LOGITS * sizeof(float));
	Check clean, corrupted;
	uint32_t pass, i;
	FillBf16(x,COUNT,1.0);
	for ( i = 0u; i < LOGITS; i++ )
		z[i] = (float)(40.0 * Normal());
	memcpy(saved,x,COUNT * sizeof(uint16_t));
	memcpy(saved_logits,z,LOGITS * sizeof(float));
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		memset(check,0,sizeof(*check));
		memcpy(x,saved,COUNT * sizeof(uint16_t));
		if ( pass == 1u )
			x[1234] = ToBf16(Bf16(x[1234]) + 3.0f);
		if ( !Run(SparkOpsScaleRows(&ops,queue,Whole(&rows),COUNT,53.0f),"scale_rows") )
			break;
		for ( i = 0u; i < COUNT; i++ )
			Observe(check,Bf16(x[i]),RoundBf16(Bf16(saved[i]) * 53.0),UNIT_BF16 * fabs(Bf16(saved[i]) * 53.0) * 1e-3);
	}
	Report("scale_rows","3 rows x 2816, times 53",&clean,&corrupted,"exact: bf16 x times 53 is exact in fp32, then one rounding");
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		memset(check,0,sizeof(*check));
		memcpy(z,saved_logits,LOGITS * sizeof(float));
		if ( pass == 1u )
			z[77] += 9.0f;
		if ( !Run(SparkOpsSoftcap(&ops,queue,Whole(&logits),LOGITS,30.0f,0.0f),"softcap") )
			break;
		for ( i = 0u; i < LOGITS; i++ )
		{
			const double y = 30.0 * tanh((double)saved_logits[i] / 30.0);
			Observe(check,z[i],y,(2.0 * FUNCTION_ERROR + 2.0 * UNIT_F32) * fabs(y) + 30.0 * UNIT_F32);
		}
	}
	Report("softcap_rows","4096 fp32 logits, cap 30",&clean,&corrupted,"(2*16+2) eps32 |y| + 30 eps32");
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		const float multiply = 0.19611613513818404f;
		memset(check,0,sizeof(*check));
		memcpy(z,saved_logits,LOGITS * sizeof(float));
		if ( pass == 1u )
			z[77] += 9.0f;
		if ( !Run(SparkOpsSoftcap(&ops,queue,Whole(&logits),LOGITS,20.0f,multiply),"softcap scaled") )
			break;
		for ( i = 0u; i < LOGITS; i++ )
		{
			const double scaled = (double)saved_logits[i] * multiply, y = 20.0 * tanh(scaled / 20.0);
			Observe(check,z[i],y,(2.0 * FUNCTION_ERROR + 2.0 * UNIT_F32) * fabs(y) + UNIT_F32 * fabs(scaled) + 20.0 * UNIT_F32);
		}
	}
	Report("softcap_rows","4096 fp32 logits, times 0.196, cap 20",&clean,&corrupted,"(2*16+2) eps32 |y| + eps32 |z m| + 20 eps32");
	free(saved);
	free(saved_logits);
	SparkDeviceFree(&rows);
	SparkDeviceFree(&logits);
}

static void TestRoutesNonFinite(void)
{
	enum { EXPERTS = 128u, TOP = 8u, ROWS = 3u };
	SparkDeviceBuffer logits, scale, experts, weights, pinned;
	uint16_t *l = (uint16_t *)Shared(&logits,(uint64_t)ROWS * EXPERTS * sizeof(uint16_t),"non-finite route logits");
	float *sc = (float *)Shared(&scale,EXPERTS * sizeof(float),"non-finite expert scale");
	uint32_t *e = (uint32_t *)Shared(&experts,(uint64_t)ROWS * TOP * sizeof(uint32_t),"non-finite route experts");
	uint32_t *pin = (uint32_t *)Shared(&pinned,(uint64_t)ROWS * TOP * sizeof(uint32_t),"non-finite pinned experts");
	SparkTensor pinned_tensor = Whole(&pinned);
	SparkRouteParams params;
	uint32_t row, k, j, error, valid = 1u, mode;
	Shared(&weights,(uint64_t)ROWS * TOP * sizeof(float),"non-finite route weights");
	FillBf16(l,(uint64_t)ROWS * EXPERTS,2.0);
	for ( k = 0u; k < EXPERTS; k++ )
		sc[k] = 1.0f;
	for ( row = 0u; row < ROWS; row++ )
		for ( k = 0u; k < TOP; k++ )
			pin[row * TOP + k] = (row * 11u + k * 5u) % EXPERTS;
	l[1u * EXPERTS + 5u] = 0x7fc0u;
	l[2u * EXPERTS + 9u] = 0x7f80u;
	memset(&params,0,sizeof(params));
	params.experts = EXPERTS;
	params.top_k = TOP;
	params.logit_stride = EXPERTS;
	for ( mode = 0u; mode < 2u; mode++ )
	{
		const char *name = mode == 0u ? "route_topk" : "route_pinned";
		params.rows = 1u;
		SparkDeviceOpsTakeError(&ops,&error);
		if ( !Run(SparkOpsRouteTopk(&ops,queue,Whole(&logits),Whole(&scale),mode != 0u ? &pinned_tensor : 0,Whole(&experts),Whole(&weights),&params),name) )
			continue;
		SparkDeviceOpsTakeError(&ops,&error);
		if ( error != 0u )
		{
			printf("FAIL %-22s finite logits raised the device error word (%u)\n",name,error);
			failures++;
		}
		params.rows = ROWS;
		memset(e,0xff,(size_t)ROWS * TOP * sizeof(uint32_t));
		if ( !Run(SparkOpsRouteTopk(&ops,queue,Whole(&logits),Whole(&scale),mode != 0u ? &pinned_tensor : 0,Whole(&experts),Whole(&weights),&params),name) )
			continue;
		SparkDeviceOpsTakeError(&ops,&error);
		for ( row = 0u; row < ROWS; row++ )
			for ( k = 0u; k < TOP; k++ )
			{
				valid &= e[row * TOP + k] < EXPERTS ? 1u : 0u;
				for ( j = 0u; j < k; j++ )
					valid &= e[row * TOP + k] != e[row * TOP + j] ? 1u : 0u;
			}
		if ( error < 2u || error > ROWS || valid == 0u )
		{
			printf("FAIL %-22s a NaN or +Inf logit: error word %u (expected row 1 or 2), experts %s\n",name,error,
				valid != 0u ? "distinct and in range" : "OUT OF RANGE OR REPEATED");
			failures++;
		}
		else
			printf("PASS %-22s a NaN or +Inf logit raises the device error word for row %u; every row still picks %u distinct experts in range\n",name,
				error - 1u,TOP);
	}
	SparkDeviceFree(&logits);
	SparkDeviceFree(&scale);
	SparkDeviceFree(&experts);
	SparkDeviceFree(&weights);
	SparkDeviceFree(&pinned);
}

static void TestRoutesPinned(uint32_t rows)
{
	enum { EXPERTS = 128u, TOP = 8u };
	SparkDeviceBuffer logits, scale, pinned, experts, weights;
	uint16_t *l = (uint16_t *)Shared(&logits,(uint64_t)rows * EXPERTS * sizeof(uint16_t),"pinned logits");
	float *sc = (float *)Shared(&scale,EXPERTS * sizeof(float),"pinned scale");
	uint32_t *pv = (uint32_t *)Shared(&pinned,(uint64_t)rows * TOP * sizeof(uint32_t),"pinned experts");
	uint32_t *e = (uint32_t *)Shared(&experts,(uint64_t)rows * TOP * sizeof(uint32_t),"pinned out experts");
	float *wt = (float *)Shared(&weights,(uint64_t)rows * TOP * sizeof(float),"pinned weights");
	SparkTensor pin = Whole(&pinned);
	SparkRouteParams params;
	Check clean, corrupted;
	char shape[64];
	uint32_t pass, row, k, x;
	FillBf16(l,(uint64_t)rows * EXPERTS,2.0);
	for ( x = 0u; x < EXPERTS; x++ )
		sc[x] = (float)(0.5 + Uniform());
	for ( row = 0u; row < rows; row++ )
		for ( k = 0u; k < TOP; k++ )
			pv[row * TOP + k] = (row * 37u + (TOP - k) * 13u) % EXPERTS;
	memset(&params,0,sizeof(params));
	params.rows = rows;
	params.experts = EXPERTS;
	params.top_k = TOP;
	params.width = 1u;
	params.logit_stride = EXPERTS;
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		const uint32_t poke = (rows - 1u) * EXPERTS + pv[(rows - 1u) * TOP + 2u];
		const uint16_t saved = l[poke];
		memset(check,0,sizeof(*check));
		if ( pass == 1u )
			l[poke] = ToBf16(Bf16(saved) + 1.0f);
		if ( !Run(SparkOpsRouteTopk(&ops,queue,Whole(&logits),Whole(&scale),&pin,Whole(&experts),Whole(&weights),&params),"route_topk pinned") )
			break;
		l[poke] = saved;
		for ( row = 0u; row < rows; row++ )
		{
			double best = -INFINITY, total = 0.0, probability[EXPERTS], picked = 0.0;
			for ( x = 0u; x < EXPERTS; x++ )
				best = fmax(best,Bf16(l[row * EXPERTS + x]));
			for ( x = 0u; x < EXPERTS; x++ )
				total += (probability[x] = exp(Bf16(l[row * EXPERTS + x]) - best));
			for ( k = 0u; k < TOP; k++ )
				picked += probability[pv[row * TOP + k]] / total;
			for ( k = 0u; k < TOP; k++ )
			{
				const uint32_t chosen = pv[row * TOP + k];
				const double weight = probability[chosen] / total / picked * sc[chosen];
				Observe(check,e[row * TOP + k] == chosen ? 0.0 : 1.0,0.0,0.0);
				Observe(check,wt[row * TOP + k],weight,(TOP + EXPERTS + 2.0 * FUNCTION_ERROR / UNIT_F32 + 8.0) * UNIT_F32 * fabs(weight));
			}
		}
	}
	snprintf(shape,sizeof(shape),"rows %u, 128 experts, 8 pinned",rows);
	Report("route_pinned",shape,&clean,&corrupted,"experts copied; weights of the pinned experts (k + E + 2*16 + 8) eps32 relative");
	SparkDeviceFree(&logits);
	SparkDeviceFree(&scale);
	SparkDeviceFree(&pinned);
	SparkDeviceFree(&experts);
	SparkDeviceFree(&weights);
}

static void TestRoutes(uint32_t rows)
{
	enum { EXPERTS = 128u, TOP = 8u, WIDTH = 2816u };
	SparkDeviceBuffer logits, scale, experts, weights, outputs, combined;
	uint16_t *l = (uint16_t *)Shared(&logits,(uint64_t)rows * EXPERTS * sizeof(uint16_t),"route logits");
	float *sc = (float *)Shared(&scale,EXPERTS * sizeof(float),"expert scale");
	uint32_t *e = (uint32_t *)Shared(&experts,(uint64_t)rows * TOP * sizeof(uint32_t),"route experts");
	float *wt = (float *)Shared(&weights,(uint64_t)rows * TOP * sizeof(float),"route weights");
	uint16_t *eo = (uint16_t *)Shared(&outputs,(uint64_t)rows * TOP * WIDTH * sizeof(uint16_t),"expert outputs");
	uint16_t *co = (uint16_t *)Shared(&combined,(uint64_t)rows * WIDTH * sizeof(uint16_t),"combined");
	SparkRouteParams params;
	Check clean, corrupted;
	char shape[64];
	uint32_t pass, row, k, x, i;
	FillBf16(l,(uint64_t)rows * EXPERTS,2.0);
	FillBf16(eo,(uint64_t)rows * TOP * WIDTH,1.0);
	for ( i = 0u; i < EXPERTS; i++ )
		sc[i] = (float)(0.5 + Uniform());
	memset(&params,0,sizeof(params));
	params.rows = rows;
	params.experts = EXPERTS;
	params.top_k = TOP;
	params.width = WIDTH;
	params.logit_stride = EXPERTS;
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		const uint32_t poke = (rows - 1u) * EXPERTS + 5u;
		const uint16_t saved = l[poke];
		memset(check,0,sizeof(*check));
		if ( pass == 1u )
			l[poke] = ToBf16(9.0f);
		if ( !Run(SparkOpsRouteTopk(&ops,queue,Whole(&logits),Whole(&scale),0,Whole(&experts),Whole(&weights),&params),"route_topk") )
			break;
		l[poke] = saved;
		for ( row = 0u; row < rows; row++ )
		{
			double best = -INFINITY, total = 0.0, probability[EXPERTS], picked = 0.0;
			uint32_t chosen[TOP], taken[EXPERTS];
			memset(taken,0,sizeof(taken));
			for ( x = 0u; x < EXPERTS; x++ )
				best = fmax(best,Bf16(l[row * EXPERTS + x]));
			for ( x = 0u; x < EXPERTS; x++ )
				total += (probability[x] = exp(Bf16(l[row * EXPERTS + x]) - best));
			for ( k = 0u; k < TOP; k++ )
			{
				uint32_t winner = 0u;
				double value = -1.0;
				for ( x = 0u; x < EXPERTS; x++ )
					if ( taken[x] == 0u && probability[x] > value )
					{
						value = probability[x];
						winner = x;
					}
				taken[winner] = 1u;
				chosen[k] = winner;
				picked += value / total;
			}
			for ( k = 0u; k < TOP; k++ )
			{
				const double weight = probability[chosen[k]] / total / picked * sc[chosen[k]];
				Observe(check,e[row * TOP + k] == chosen[k] ? 0.0 : 1.0,0.0,0.0);
				Observe(check,wt[row * TOP + k],weight,(TOP + EXPERTS + 2.0 * FUNCTION_ERROR / UNIT_F32 + 8.0) * UNIT_F32 * fabs(weight));
			}
		}
	}
	snprintf(shape,sizeof(shape),"rows %u, 128 experts, top 8",rows);
	Report("route_topk",shape,&clean,&corrupted,"experts exact; weights (k + E + 2*16 + 8) eps32 relative");
	for ( row = 0u; row < rows; row++ )
		for ( k = 0u; k < TOP; k++ )
			e[row * TOP + k] = (uint32_t)((row * 37u + (TOP - k) * 13u) % EXPERTS);
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		const uint64_t poke = ((uint64_t)(rows - 1u) * TOP + 3u) * WIDTH + 99u;
		const uint16_t saved = eo[poke];
		memset(check,0,sizeof(*check));
		if ( pass == 1u )
			eo[poke] = ToBf16(Bf16(saved) + 5.0f);
		if ( !Run(SparkOpsRouteCombine(&ops,queue,Whole(&outputs),Whole(&combined),&params),"route_combine") )
			break;
		eo[poke] = saved;
		for ( row = 0u; row < rows; row++ )
			for ( i = 0u; i < WIDTH; i++ )
			{
				float total = 0.0f;
				double magnitude = 0.0;
				for ( k = 0u; k < TOP; k++ )
				{
					total += (float)Bf16(eo[((uint64_t)row * TOP + k) * WIDTH + i]);
					magnitude += fabs(Bf16(eo[((uint64_t)row * TOP + k) * WIDTH + i]));
				}
				Observe(check,Bf16(co[(uint64_t)row * WIDTH + i]),RoundBf16((double)total),UNIT_BF16 * 1e-3 * magnitude);
			}
	}
	snprintf(shape,sizeof(shape),"rows %u, top 8 x 2816 in route order",rows);
	Report("route_combine",shape,&clean,&corrupted,"exact: the fp32 sum of the routes' bf16 outputs in route order, rounded once to bf16");
	SparkDeviceFree(&logits);
	SparkDeviceFree(&scale);
	SparkDeviceFree(&experts);
	SparkDeviceFree(&weights);
	SparkDeviceFree(&outputs);
	SparkDeviceFree(&combined);
}

static void TestLinearRouted(uint32_t rows)
{
	enum { K = 2816u, N = 704u, EXPERTS = 16u, TOP = 8u };
	const uint32_t routes = rows * TOP;
	SparkDeviceBuffer x, w, y, experts, single;
	uint16_t *xv = (uint16_t *)Shared(&x,(uint64_t)rows * K * sizeof(uint16_t),"routed x");
	uint16_t *wv = (uint16_t *)Shared(&w,(uint64_t)EXPERTS * N * K * sizeof(uint16_t),"routed w");
	uint16_t *yv = (uint16_t *)Shared(&y,(uint64_t)routes * N * sizeof(uint16_t),"routed y");
	uint32_t *ev = (uint32_t *)Shared(&experts,routes * sizeof(uint32_t),"routed experts");
	uint16_t *sv = (uint16_t *)Shared(&single,(uint64_t)N * sizeof(uint16_t),"routed single");
	SparkGemvParams params, alone;
	Check clean, corrupted;
	char shape[64];
	uint32_t pass, route, n;
	FillBf16(xv,(uint64_t)rows * K,1.0);
	FillBf16(wv,(uint64_t)EXPERTS * N * K,1.0 / sqrt((double)K));
	for ( route = 0u; route < routes; route++ )
		ev[route] = (route * 7u + 3u) % EXPERTS;
	memset(&params,0,sizeof(params));
	params.rows = routes;
	params.input_dimension = K;
	params.output_dimension = N;
	params.input_stride = K;
	params.output_stride = N;
	params.route_divisor = TOP;
	memset(&clean,0,sizeof(clean));
	memset(&corrupted,0,sizeof(corrupted));
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		const uint32_t victim = routes - 1u;
		if ( pass == 1u )
			ev[victim] = (ev[victim] + 1u) % EXPERTS;
		if ( !Run(SparkOpsLinearRouted(&ops,queue,Whole(&x),Whole(&w),Whole(&y),Whole(&experts),EXPERTS,&params),"linear_routed") )
			break;
		if ( pass == 1u )
			ev[victim] = (ev[victim] + EXPERTS - 1u) % EXPERTS;
		for ( route = 0u; route < routes; route++ )
		{
			alone = params;
			alone.rows = 1u;
			alone.route_divisor = 0u;
			if ( !Run(SparkOpsLinear(&ops,queue,SparkTensorSlice(Whole(&x),(uint64_t)(route / TOP) * K * sizeof(uint16_t),K * sizeof(uint16_t)),
				SparkTensorSlice(Whole(&w),(uint64_t)ev[route] * N * K * sizeof(uint16_t),(uint64_t)N * K * sizeof(uint16_t)),Whole(&single),&alone),
				"linear for one expert") )
				break;
			for ( n = 0u; n < N; n++ )
				Observe(check,yv[(uint64_t)route * N + n] == sv[n] ? 0.0 : 1.0,0.0,0.0);
		}
	}
	snprintf(shape,sizeof(shape),"rows %u x top 8, K 2816 -> N 704, 16 experts",rows);
	Report("linear_routed",shape,&clean,&corrupted,"exact: each route's bits equal the one-row GEMV of its expert");
	SparkDeviceFree(&x);
	SparkDeviceFree(&w);
	SparkDeviceFree(&y);
	SparkDeviceFree(&experts);
	SparkDeviceFree(&single);
}

static void TestLinearRoutedSkip(uint32_t rows)
{
	enum { K = 2816u, N = 704u, EXPERTS = 16u, TOP = 8u };
	const uint32_t routes = rows * TOP;
	SparkDeviceBuffer x, w, y, experts, single;
	uint16_t *xv = (uint16_t *)Shared(&x,(uint64_t)rows * K * sizeof(uint16_t),"skip x");
	uint16_t *wv = (uint16_t *)Shared(&w,(uint64_t)EXPERTS * N * K * sizeof(uint16_t),"skip w");
	uint16_t *yv = (uint16_t *)Shared(&y,(uint64_t)routes * N * sizeof(uint16_t),"skip y");
	uint32_t *ev = (uint32_t *)Shared(&experts,routes * sizeof(uint32_t),"skip experts");
	uint16_t *sv = (uint16_t *)Shared(&single,(uint64_t)N * sizeof(uint16_t),"skip single");
	SparkGemvParams params, alone;
	Check clean, corrupted;
	char shape[64];
	uint32_t pass, route, n;
	FillBf16(xv,(uint64_t)rows * K,1.0);
	FillBf16(wv,(uint64_t)EXPERTS * N * K,1.0 / sqrt((double)K));
	for ( route = 0u; route < routes; route++ )
		ev[route] = route % 3u == 1u ? SPARK_ROUTE_SKIP : (route * 7u + 3u) % EXPERTS;
	memset(&params,0,sizeof(params));
	params.rows = routes;
	params.input_dimension = K;
	params.output_dimension = N;
	params.input_stride = K;
	params.output_stride = N;
	params.route_divisor = TOP;
	memset(&clean,0,sizeof(clean));
	memset(&corrupted,0,sizeof(corrupted));
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		FillBf16(yv,(uint64_t)routes * N,1.0);
		if ( pass == 1u )
			ev[1] = 5u;
		if ( !Run(SparkOpsLinearRouted(&ops,queue,Whole(&x),Whole(&w),Whole(&y),Whole(&experts),EXPERTS,&params),"linear_routed skip") )
			break;
		if ( pass == 1u )
			ev[1] = SPARK_ROUTE_SKIP;
		for ( route = 0u; route < routes; route++ )
		{
			if ( ev[route] == SPARK_ROUTE_SKIP )
			{
				for ( n = 0u; n < N; n++ )
					Observe(check,yv[(uint64_t)route * N + n] == 0u ? 0.0 : 1.0,0.0,0.0);
				continue;
			}
			alone = params;
			alone.rows = 1u;
			alone.route_divisor = 0u;
			if ( !Run(SparkOpsLinear(&ops,queue,SparkTensorSlice(Whole(&x),(uint64_t)(route / TOP) * K * sizeof(uint16_t),K * sizeof(uint16_t)),
				SparkTensorSlice(Whole(&w),(uint64_t)ev[route] * N * K * sizeof(uint16_t),(uint64_t)N * K * sizeof(uint16_t)),Whole(&single),&alone),
				"linear for one expert") )
				break;
			for ( n = 0u; n < N; n++ )
				Observe(check,yv[(uint64_t)route * N + n] == sv[n] ? 0.0 : 1.0,0.0,0.0);
		}
	}
	snprintf(shape,sizeof(shape),"rows %u x top 8, every third route skipped",rows);
	Report("linear_routed_skip",shape,&clean,&corrupted,"exact: a skipped route writes +0 rows, the others equal their expert's one-row GEMV");
	SparkDeviceFree(&x);
	SparkDeviceFree(&w);
	SparkDeviceFree(&y);
	SparkDeviceFree(&experts);
	SparkDeviceFree(&single);
}

static void TestHeadGate(uint32_t rows, uint32_t heads, uint32_t activation)
{
	enum { DIM = 128u };
	const uint32_t stride = heads + (heads & 1u);
	SparkDeviceBuffer values, logits;
	uint16_t *v = (uint16_t *)Shared(&values,(uint64_t)rows * heads * DIM * 2u,"head gate values");
	uint16_t *g = (uint16_t *)Shared(&logits,(uint64_t)rows * stride * 2u,"head gate logits");
	uint16_t *original = (uint16_t *)malloc((size_t)rows * heads * DIM * 2u);
	SparkHeadGateParams params;
	Check clean, corrupted;
	char shape[64];
	uint32_t pass, row, head, i;
	FillBf16(original,(uint64_t)rows * heads * DIM,2.0);
	for ( i = 0u; i < rows * stride; i++ )
		g[i] = ToBf16((float)(Normal() * 8.0));
	g[0] = ToBf16(25.0f);
	g[1] = ToBf16(-30.0f);
	memset(&params,0,sizeof(params));
	params.rows = rows;
	params.heads = heads;
	params.head_dimension = DIM;
	params.gate_stride = stride;
	params.activation = activation;
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		const uint16_t saved = g[(rows - 1u) * stride + heads - 1u];
		memset(check,0,sizeof(*check));
		memcpy(v,original,(size_t)rows * heads * DIM * 2u);
		if ( pass == 1u )
			g[(rows - 1u) * stride + heads - 1u] = ToBf16(Bf16(saved) + 4.0f);
		if ( !Run(SparkOpsHeadGate(&ops,queue,Whole(&values),Whole(&logits),&params),"head_gate") )
			break;
		g[(rows - 1u) * stride + heads - 1u] = saved;
		for ( row = 0u; row < rows; row++ )
			for ( head = 0u; head < heads; head++ )
			{
				const float logit = Bf16(g[row * stride + head]);
				const double raw = activation == SPARK_HEAD_GATE_SIGMOID_BF16 ? Sigmoid(logit) : (logit > 20.0f ? logit : log1p(exp((double)logit)));
				const double gate = RoundBf16(raw), gate_slack = FlipSlack(raw,4.0 * FUNCTION_ERROR);
				for ( i = 0u; i < DIM; i++ )
				{
					const uint64_t at = ((uint64_t)row * heads + head) * DIM + i;
					const double expected = RoundBf16(Bf16(original[at]) * gate);
					Observe(check,Bf16(v[at]),expected,fabs(expected) * UNIT_BF16 + fabs(Bf16(original[at])) * gate_slack * (1.0 + UNIT_BF16) + 1e-30);
				}
			}
	}
	snprintf(shape,sizeof(shape),"rows %u, %u heads x 128",rows,heads);
	Report(activation == SPARK_HEAD_GATE_SIGMOID_BF16 ? "head_gate_sigmoid" : "head_gate_softplus",shape,&clean,&corrupted,
		"bf16(v bf16(gate)), gate softplus (log1p(exp), Metal's log1p emulated) or sigmoid: u|y| + |v| flip slack of the gate within 4*16 eps32");
	free(original);
	SparkDeviceFree(&values);
	SparkDeviceFree(&logits);
}

static void TestRouteLocal(void)
{
	enum { ROWS = 37u, TOP = 8u, EXPERTS = 128u, FIRST = 48u, LOCAL = 16u };
	SparkDeviceBuffer experts;
	uint32_t *ev = (uint32_t *)Shared(&experts,ROWS * TOP * sizeof(uint32_t),"local experts");
	uint32_t original[ROWS * TOP];
	SparkRouteParams params;
	Check clean, corrupted;
	uint32_t pass, i;
	for ( i = 0u; i < ROWS * TOP; i++ )
		original[i] = (uint32_t)(NextRandom() % EXPERTS);
	original[0] = FIRST;
	original[1] = FIRST + LOCAL - 1u;
	original[2] = FIRST + LOCAL;
	original[3] = FIRST - 1u;
	memset(&params,0,sizeof(params));
	params.rows = ROWS;
	params.experts = EXPERTS;
	params.top_k = TOP;
	params.first_expert = FIRST;
	params.local_experts = LOCAL;
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		memset(check,0,sizeof(*check));
		memcpy(ev,original,sizeof(original));
		if ( pass == 1u )
			ev[0] = FIRST + 1u;
		if ( !Run(SparkOpsRouteLocal(&ops,queue,Whole(&experts),&params),"route_local") )
			break;
		for ( i = 0u; i < ROWS * TOP; i++ )
		{
			const uint32_t expected = original[i] - FIRST < LOCAL ? original[i] - FIRST : SPARK_ROUTE_SKIP;
			Observe(check,ev[i] == expected ? 0.0 : 1.0,0.0,0.0);
		}
	}
	Report("route_local","rows 37 x top 8, experts 48..63 of 128",&clean,&corrupted,"exact: local experts renumbered from 0, the rest skipped");
	SparkDeviceFree(&experts);
}

static void TestGatherSlice(void)
{
	enum { SLICE = 512u, RANKS = 8u, RANK = 3u, WIDTH = 2816u, ROWS = 24u };
	SparkDeviceBuffer table, ids, out;
	uint16_t *t = (uint16_t *)Shared(&table,(uint64_t)SLICE * WIDTH * 2u,"slice table");
	uint32_t *id = (uint32_t *)Shared(&ids,ROWS * 4u,"slice ids");
	uint16_t *o = (uint16_t *)Shared(&out,(uint64_t)ROWS * WIDTH * 2u,"slice out");
	Check clean, corrupted;
	uint32_t pass, row, i, error = 0u;
	FillBf16(t,(uint64_t)SLICE * WIDTH,1.0);
	for ( row = 0u; row < ROWS; row++ )
		id[row] = row % 2u == 0u ? RANK * SLICE + (uint32_t)(NextRandom() % SLICE) : (uint32_t)(NextRandom() % (SLICE * RANKS));
	id[1] = RANK * SLICE - 1u;
	id[3] = (RANK + 1u) * SLICE;
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		const uint64_t victim = (uint64_t)(id[0] - RANK * SLICE) * WIDTH + 9u;
		const uint16_t saved = t[victim];
		memset(check,0,sizeof(*check));
		FillBf16(o,(uint64_t)ROWS * WIDTH,1.0);
		if ( pass == 1u )
			t[victim] = ToBf16(Bf16(saved) + 1.0f);
		if ( !Run(SparkOpsGatherSlice(&ops,queue,Whole(&table),Whole(&ids),Whole(&out),ROWS,WIDTH,SLICE,RANK * SLICE),"gather_slice") )
			break;
		t[victim] = saved;
		for ( row = 0u; row < ROWS; row++ )
			for ( i = 0u; i < WIDTH; i++ )
			{
				const uint32_t local = id[row] - RANK * SLICE;
				Observe(check,o[(uint64_t)row * WIDTH + i] == (local < SLICE ? t[(uint64_t)local * WIDTH + i] : 0u) ? 0.0 : 1.0,0.0,0.0);
			}
	}
	if ( SparkDeviceOpsTakeError(&ops,&error) != SPARK_STATUS_OK )
	{
		printf("FAIL gather_slice: a sliced gather raised the device error word (row %u)\n",error - 1u);
		failures++;
	}
	Report("gather_slice","rows 24, width 2816, rank 3 of 8 holding 512 rows",&clean,&corrupted,
		"exact: rows in the rank's slice are copied, every other row is +0");
	SparkDeviceFree(&table);
	SparkDeviceFree(&ids);
	SparkDeviceFree(&out);
}

static void TestSegmentedMap(void)
{
	enum { WIDTH = 4096u, ROWS = 640u, STRIDE = 1u << 20, GATHERED = 9u };
	const uint64_t bytes = (uint64_t)ROWS * WIDTH * sizeof(uint16_t);
	char path[] = "/tmp/spark_segmented_map_XXXXXX";
	uint16_t *values = (uint16_t *)malloc((size_t)bytes);
	SparkDeviceBuffer mapped, ids, out;
	uint32_t *id = (uint32_t *)Shared(&ids,GATHERED * 4u,"segment ids");
	uint16_t *o = (uint16_t *)Shared(&out,(uint64_t)GATHERED * WIDTH * 2u,"segment out");
	Check clean, corrupted;
	uint32_t pass, row, i, first;
	int fd = mkstemp(path);
	FillBf16(values,(uint64_t)ROWS * WIDTH,1.0);
	if ( fd < 0 || write(fd,values,(size_t)bytes) != (ssize_t)bytes ||
		SparkDeviceMapFileSegments(&mapped,fd,0u,bytes,STRIDE,"segmented file") != SPARK_STATUS_OK )
	{
		printf("FAIL segmented_map: cannot write or map %s in 1 MiB segments\n",path);
		failures++;
		if ( fd >= 0 )
			close(fd);
		unlink(path);
		free(values);
		return;
	}
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		memset(check,0,sizeof(*check));
		for ( first = 0u; first + STRIDE / (WIDTH * 2u) <= ROWS; first += 61u )
		{
			const uint32_t window = STRIDE / (WIDTH * 2u);
			for ( row = 0u; row < GATHERED; row++ )
				id[row] = (row * 13u) % window;
			if ( pass == 1u && first == 61u )
				id[2] = (id[2] + 1u) % window;
			if ( !Run(SparkOpsGather(&ops,queue,SparkTensorAt(&mapped,(uint64_t)first * WIDTH * 2u,(uint64_t)window * WIDTH * 2u),Whole(&ids),Whole(&out),
				GATHERED,WIDTH,window),"segmented gather") )
				break;
			if ( pass == 1u && first == 61u )
				id[2] = (id[2] + window - 1u) % window;
			for ( row = 0u; row < GATHERED; row++ )
				for ( i = 0u; i < WIDTH; i++ )
					Observe(check,o[(uint64_t)row * WIDTH + i] == values[(uint64_t)(first + id[row]) * WIDTH + i] ? 0.0 : 1.0,0.0,0.0);
		}
	}
	Report("segmented_map","640 rows of 8 KiB in 1 MiB segments, windows of 1 MiB at every 61st row",&clean,&corrupted,
		"exact: a tensor read through its segment equals the file bytes");
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		const uint32_t window = (2u * STRIDE + 16384u) / (WIDTH * 2u);
		const uint64_t start = (uint64_t)STRIDE - 8192u;
		memset(check,0,sizeof(*check));
		for ( row = 0u; row < GATHERED; row++ )
			id[row] = (row * 71u) % window;
		if ( pass == 1u )
			id[4] = (id[4] + 1u) % window;
		if ( !Run(SparkOpsGather(&ops,queue,SparkTensorAt(&mapped,start,(uint64_t)window * WIDTH * 2u),Whole(&ids),Whole(&out),GATHERED,WIDTH,window),
			"wide segmented gather") )
			break;
		if ( pass == 1u )
			id[4] = (id[4] + window - 1u) % window;
		for ( row = 0u; row < GATHERED; row++ )
			for ( i = 0u; i < WIDTH; i++ )
				Observe(check,o[(uint64_t)row * WIDTH + i] == values[start / 2u + (uint64_t)id[row] * WIDTH + i] ? 0.0 : 1.0,0.0,0.0);
	}
	Report("segmented_map","a binding of two strides and 16 KiB that starts 8 KiB before a stride boundary",&clean,&corrupted,
		"exact: a binding wider than a stride reads the file bytes across both boundaries");
	if ( SparkDeviceSegmented(SparkTensorAt(&mapped,0u,8192u).range) != 1u || SparkDeviceSegmented(Whole(&ids).range) != 0u )
	{
		printf("FAIL segmented_map: the segmented query does not tell the mapped file from a shared buffer\n");
		failures++;
	}
	else
		printf("PASS segmented_map          the segmented query names the mapped file and not a shared buffer\n");
	SparkDeviceFree(&mapped);
	SparkDeviceFree(&ids);
	SparkDeviceFree(&out);
	close(fd);
	unlink(path);
	free(values);
}

static void TestRankKernels(uint32_t ranks, uint32_t rows, uint32_t width)
{
	const uint64_t elements = (uint64_t)rows * width;
	SparkDeviceBuffer lanes, full, wide, wide_full, ordered;
	uint16_t *lv = (uint16_t *)Shared(&lanes,(uint64_t)ranks * elements * 2u,"rank lanes");
	uint16_t *fv = (uint16_t *)Shared(&full,elements * 2u,"rank sum");
	uint64_t *wv = (uint64_t *)Shared(&wide,(uint64_t)ranks * rows * 8u,"rank max lanes");
	uint64_t *mv = (uint64_t *)Shared(&wide_full,(uint64_t)rows * 8u,"rank max");
	uint16_t *ov = (uint16_t *)Shared(&ordered,(uint64_t)ranks * elements * 2u,"rank rows");
	Check clean, corrupted;
	char shape[96];
	uint64_t i;
	uint32_t pass, rank, row, column;
	for ( i = 0u; i < (uint64_t)ranks * elements; i++ )
		lv[i] = ToBf16((float)ldexp(Normal(),(int)(NextRandom() % 24u) - 12));
	for ( i = 0u; i < (uint64_t)ranks * rows; i++ )
		wv[i] = NextRandom();
	snprintf(shape,sizeof(shape),"%u ranks, rows %u, width %u",ranks,rows,width);
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		const uint64_t victim = (uint64_t)(ranks - 1u) * elements + 5u;
		const uint16_t saved = lv[victim];
		memset(check,0,sizeof(*check));
		if ( pass == 1u )
			lv[victim] = ToBf16(Bf16(saved) + 64.0f);
		if ( !Run(SparkOpsRankSumBf16(&ops,queue,Whole(&lanes),Whole(&full),ranks,elements),"rank_sum_bf16") )
			break;
		lv[victim] = saved;
		for ( i = 0u; i < elements; i++ )
		{
			float total = 0.0f;
			for ( rank = 0u; rank < ranks; rank++ )
				total += Bf16(lv[(uint64_t)rank * elements + i]);
			Observe(check,fv[i] == ToBf16(total) ? 0.0 : 1.0,0.0,0.0);
		}
	}
	Report("rank_sum_bf16",shape,&clean,&corrupted,"exact: an f32 sum from +0 over ranks 0..N-1 and one rounding to bf16, CUDA's all-reduce order");
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		const uint64_t saved = wv[(uint64_t)(ranks - 1u) * rows];
		memset(check,0,sizeof(*check));
		if ( pass == 1u )
			wv[(uint64_t)(ranks - 1u) * rows] = UINT64_MAX;
		if ( !Run(SparkOpsRankMaxU64(&ops,queue,Whole(&wide),Whole(&wide_full),ranks,rows),"rank_max_u64") )
			break;
		wv[(uint64_t)(ranks - 1u) * rows] = saved;
		for ( row = 0u; row < rows; row++ )
		{
			uint64_t best = 0u;
			for ( rank = 0u; rank < ranks; rank++ )
				best = wv[(uint64_t)rank * rows + row] > best ? wv[(uint64_t)rank * rows + row] : best;
			Observe(check,mv[row] == best ? 0.0 : 1.0,0.0,0.0);
		}
	}
	Report("rank_max_u64",shape,&clean,&corrupted,"exact: the largest packed (score, token) word of every rank");
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		const uint16_t saved = lv[elements + 3u];
		memset(check,0,sizeof(*check));
		if ( pass == 1u )
			lv[elements + 3u] = (uint16_t)(saved ^ 0x0100u);
		if ( !Run(SparkOpsRankRows(&ops,queue,Whole(&lanes),Whole(&ordered),ranks,rows,width),"rank_rows") )
			break;
		lv[elements + 3u] = saved;
		for ( row = 0u; row < rows; row++ )
			for ( rank = 0u; rank < ranks; rank++ )
				for ( column = 0u; column < width; column++ )
					Observe(check,ov[((uint64_t)row * ranks + rank) * width + column] == lv[((uint64_t)rank * rows + row) * width + column] ? 0.0 : 1.0,0.0,0.0);
	}
	Report("rank_rows",shape,&clean,&corrupted,"exact: rank-major gathered rows become rows of every rank's columns in rank order");
	SparkDeviceFree(&lanes);
	SparkDeviceFree(&full);
	SparkDeviceFree(&wide);
	SparkDeviceFree(&wide_full);
	SparkDeviceFree(&ordered);
}

static double E2m1(uint32_t nibble)
{
	static const double magnitude[8] = { 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 };
	return((nibble & 8u) != 0u ? -magnitude[nibble & 7u] : magnitude[nibble & 7u]);
}

static void TestLinearRoutedMxfp4(uint32_t rows, uint32_t K, uint32_t N)
{
	enum { EXPERTS = 16u, TOP = 8u };
	const uint32_t routes = rows * TOP, groups = K / SPARK_MXFP4_GROUP;
	SparkDeviceBuffer x, w, scale, y, experts;
	uint16_t *xv = (uint16_t *)Shared(&x,(uint64_t)rows * K * sizeof(uint16_t),"mxfp4 routed x");
	uint8_t *wv = (uint8_t *)Shared(&w,(uint64_t)EXPERTS * N * (K / 2u),"mxfp4 routed w");
	uint8_t *sv = (uint8_t *)Shared(&scale,(uint64_t)EXPERTS * N * groups,"mxfp4 routed scale");
	uint16_t *yv = (uint16_t *)Shared(&y,(uint64_t)routes * N * sizeof(uint16_t),"mxfp4 routed y");
	uint32_t *ev = (uint32_t *)Shared(&experts,routes * sizeof(uint32_t),"mxfp4 routed experts");
	SparkGemvParams params;
	Check clean, corrupted;
	char shape[80];
	uint64_t index;
	uint32_t pass, route, n, k;
	FillBf16(xv,(uint64_t)rows * K,1.0);
	for ( index = 0u; index < (uint64_t)EXPERTS * N * (K / 2u); index++ )
		wv[index] = (uint8_t)(NextRandom() & 0xffu);
	for ( index = 0u; index < (uint64_t)EXPERTS * N * groups; index++ )
		sv[index] = (uint8_t)(120u + NextRandom() % 6u);
	for ( route = 0u; route < routes; route++ )
		ev[route] = (route * 7u + 3u) % EXPERTS;
	memset(&params,0,sizeof(params));
	params.rows = routes;
	params.input_dimension = K;
	params.output_dimension = N;
	params.input_stride = K;
	params.output_stride = N;
	params.route_divisor = TOP;
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		const uint32_t victim = routes - 1u;
		memset(check,0,sizeof(*check));
		if ( pass == 1u )
			ev[victim] = (ev[victim] + 1u) % EXPERTS;
		if ( !Run(SparkOpsLinearRoutedMxfp4(&ops,queue,Whole(&x),Whole(&w),Whole(&scale),Whole(&y),Whole(&experts),EXPERTS,&params),"linear_routed_mxfp4") )
			break;
		if ( pass == 1u )
			ev[victim] = (ev[victim] + EXPERTS - 1u) % EXPERTS;
		for ( route = 0u; route < routes; route++ )
			for ( n = 0u; n < N; n++ )
			{
				const uint64_t neuron = (uint64_t)ev[route] * N + n;
				double sum = 0.0, magnitude = 0.0;
				for ( k = 0u; k < K; k++ )
				{
					const uint8_t byte = wv[neuron * (K / 2u) + k / 2u];
					const double weight = E2m1((k % 2u) == 0u ? byte & 15u : byte >> 4u) * ldexp(1.0,(int)sv[neuron * groups + k / SPARK_MXFP4_GROUP] - 127);
					const double term = weight * Bf16(xv[(uint64_t)(route / TOP) * K + k]);
					sum += term;
					magnitude += fabs(term);
				}
				Observe(check,Bf16(yv[(uint64_t)route * N + n]),sum,UNIT_BF16 * fabs(sum) + (1.0 + UNIT_BF16) * Gamma(K + K / SPARK_MXFP4_GROUP + 2u) * magnitude);
			}
	}
	snprintf(shape,sizeof(shape),"rows %u x top 8, K %u -> N %u, 16 experts",rows,K,N);
	Report("linear_routed_mxfp4",shape,&clean,&corrupted,"u|y| + (1+u) gamma_(K + K/32 + 2) sum|x w s|, u = 2^-8");
	SparkDeviceFree(&x);
	SparkDeviceFree(&w);
	SparkDeviceFree(&scale);
	SparkDeviceFree(&y);
	SparkDeviceFree(&experts);
}

static float Ue4m3(uint8_t code)
{
	const uint32_t value = code & 0x7fu, exponent = value >> 3, mantissa = value & 7u;
	if ( value == 0x7fu )
		return(448.0f);
	if ( exponent == 0u )
		return((float)ldexp((double)mantissa / 8.0,-6));
	return((float)ldexp(1.0 + (double)mantissa / 8.0,(int)exponent - 7));
}

static void TestLinearRoutedNvfp4(uint32_t rows, uint32_t K, uint32_t N)
{
	enum { EXPERTS = 16u, TOP = 10u };
	const uint32_t routes = rows * TOP, groups = K / SPARK_NVFP4_GROUP, chunks = K / SPARK_NVFP4_CHUNK, lanes = SPARK_NVFP4_LANES(chunks);
	SparkDeviceBuffer x, w, scale, global, y, experts;
	uint16_t *xv = (uint16_t *)Shared(&x,(uint64_t)rows * K * sizeof(uint16_t),"nvfp4 routed x");
	uint8_t *wv = (uint8_t *)Shared(&w,(uint64_t)EXPERTS * N * (K / 2u),"nvfp4 routed w");
	uint8_t *sv = (uint8_t *)Shared(&scale,(uint64_t)EXPERTS * N * groups,"nvfp4 routed scale");
	float *gv = (float *)Shared(&global,EXPERTS * sizeof(float),"nvfp4 routed global scale");
	uint16_t *yv = (uint16_t *)Shared(&y,(uint64_t)routes * N * sizeof(uint16_t),"nvfp4 routed y");
	uint32_t *ev = (uint32_t *)Shared(&experts,routes * sizeof(uint32_t),"nvfp4 routed experts");
	SparkGemvParams params;
	Check clean, corrupted;
	char shape[80];
	uint64_t index;
	uint32_t pass, route, n, sub, c, j, step;
	FillBf16(xv,(uint64_t)rows * K,1.0);
	for ( index = 0u; index < (uint64_t)EXPERTS * N * (K / 2u); index++ )
		wv[index] = (uint8_t)(NextRandom() & 0xffu);
	for ( index = 0u; index < (uint64_t)EXPERTS * N * groups; index++ )
		sv[index] = (uint8_t)(40u + NextRandom() % 48u);
	for ( index = 0u; index < EXPERTS; index++ )
		gv[index] = (float)(1e-3 * (1.0 + Uniform()));
	for ( route = 0u; route < routes; route++ )
		ev[route] = (route * 7u + 3u) % EXPERTS;
	memset(&params,0,sizeof(params));
	params.rows = routes;
	params.input_dimension = K;
	params.output_dimension = N;
	params.input_stride = K;
	params.output_stride = N;
	params.route_divisor = TOP;
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		const uint32_t victim = routes - 1u;
		memset(check,0,sizeof(*check));
		if ( pass == 1u )
			ev[victim] = (ev[victim] + 1u) % EXPERTS;
		if ( !Run(SparkOpsLinearRoutedNvfp4(&ops,queue,Whole(&x),Whole(&w),Whole(&scale),Whole(&global),Whole(&y),Whole(&experts),EXPERTS,&params),
			"linear_routed_nvfp4") )
			break;
		if ( pass == 1u )
			ev[victim] = (ev[victim] + EXPERTS - 1u) % EXPERTS;
		for ( route = 0u; route < routes; route++ )
			for ( n = 0u; n < N; n++ )
			{
				const uint64_t neuron = (uint64_t)ev[route] * N + n;
				const uint16_t *row = xv + (uint64_t)(route / TOP) * K;
				float acc[32], next[32];
				for ( sub = 0u; sub < lanes; sub++ )
				{
					acc[sub] = 0.0f;
					for ( c = sub; c < chunks; c += lanes )
					{
						float low = 0.0f, high = 0.0f;
						for ( j = 0u; j < SPARK_NVFP4_CHUNK; j++ )
						{
							const uint8_t byte = wv[neuron * (K / 2u) + c * (SPARK_NVFP4_CHUNK / 2u) + j / 2u];
							const float weight = (float)E2m1((j % 2u) == 0u ? byte & 15u : byte >> 4u), value = Bf16(row[c * SPARK_NVFP4_CHUNK + j]);
							if ( j < SPARK_NVFP4_GROUP )
								low = fmaf(weight,value,low);
							else
								high = fmaf(weight,value,high);
						}
						const float low_scale = Ue4m3(sv[neuron * groups + 2u * c]) * gv[ev[route]];
						const float high_scale = Ue4m3(sv[neuron * groups + 2u * c + 1u]) * gv[ev[route]];
						acc[sub] = fmaf(high_scale,high,fmaf(low_scale,low,acc[sub]));
					}
				}
				for ( step = lanes / 2u; step > 0u; step >>= 1u )
				{
					for ( sub = 0u; sub < lanes; sub++ )
						next[sub] = acc[sub] + acc[sub ^ step];
					memcpy(acc,next,lanes * sizeof(float));
				}
				Observe(check,Bf16(yv[(uint64_t)route * N + n]),Bf16(ToBf16(acc[0])),0.0);
			}
	}
	snprintf(shape,sizeof(shape),"rows %u x top 10, K %u -> N %u, %u lanes",rows,K,N,lanes);
	Report("linear_routed_nvfp4",shape,&clean,&corrupted,"exact: CUDA's skinny lanes, chunk order, fma chain and xor reduction");
	SparkDeviceFree(&x);
	SparkDeviceFree(&w);
	SparkDeviceFree(&scale);
	SparkDeviceFree(&global);
	SparkDeviceFree(&y);
	SparkDeviceFree(&experts);
}

static void TestSharedExpert(uint32_t rows)
{
	enum { WIDTH = 2560u, TOP = 10u };
	static const double factors[3] = { 1.0 - 2.0 * FUNCTION_ERROR, 1.0, 1.0 + 2.0 * FUNCTION_ERROR };
	SparkDeviceBuffer outs, shared, combined, logits;
	uint16_t *ov = (uint16_t *)Shared(&outs,(uint64_t)rows * TOP * WIDTH * 2u,"shared expert routes");
	uint16_t *sv = (uint16_t *)Shared(&shared,(uint64_t)rows * WIDTH * 2u,"shared expert rows");
	uint16_t *cv = (uint16_t *)Shared(&combined,(uint64_t)rows * WIDTH * 2u,"shared expert combined");
	uint16_t *lv = (uint16_t *)Shared(&logits,(uint64_t)rows * 2u,"shared expert gate logits");
	uint16_t *original = (uint16_t *)malloc((uint64_t)rows * WIDTH * 2u);
	SparkRouteParams route;
	Check clean, corrupted;
	char shape[64];
	uint32_t pass, row, i, k;
	if ( original == 0 )
	{
		printf("FAIL shared expert: test buffer\n");
		failures++;
		return;
	}
	FillBf16(ov,(uint64_t)rows * TOP * WIDTH,1.0);
	FillBf16(sv,(uint64_t)rows * WIDTH,1.0);
	FillBf16(lv,rows,2.0);
	memcpy(original,sv,(size_t)rows * WIDTH * 2u);
	memset(&route,0,sizeof(route));
	route.rows = rows;
	route.top_k = TOP;
	route.width = WIDTH;
	route.logit_stride = 1u;
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		memset(check,0,sizeof(*check));
		memcpy(sv,original,(size_t)rows * WIDTH * 2u);
		if ( pass == 1u )
			lv[rows - 1u] = ToBf16(Bf16(lv[rows - 1u]) + 3.0f);
		if ( !Run(SparkOpsSigmoidGateRows(&ops,queue,Whole(&shared),Whole(&logits),&route),"sigmoid_gate_rows") )
			break;
		if ( pass == 1u )
			lv[rows - 1u] = ToBf16(Bf16(lv[rows - 1u]) - 3.0f);
		for ( row = 0u; row < rows; row++ )
			for ( i = 0u; i < WIDTH; i++ )
			{
				const double v = Bf16(original[(uint64_t)row * WIDTH + i]), sigma = Sigmoid(Bf16(lv[row])), expected = RoundBf16(v * sigma);
				double tolerance = 0.0;
				for ( uint32_t f = 0u; f < 3u; f++ )
					tolerance = fmax(tolerance,fabs(RoundBf16(v * sigma * factors[f]) - expected));
				Observe(check,Bf16(sv[(uint64_t)row * WIDTH + i]),expected,tolerance);
			}
	}
	snprintf(shape,sizeof(shape),"rows %u x 2560",rows);
	Report("sigmoid_gate_rows",shape,&clean,&corrupted,"bf16(v sigmoid(bf16 logit)): sigmoid to 2*16 eps32");
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		const uint16_t saved = sv[(uint64_t)(rows - 1u) * WIDTH + 13u];
		memset(check,0,sizeof(*check));
		if ( pass == 1u )
			sv[(uint64_t)(rows - 1u) * WIDTH + 13u] = ToBf16(Bf16(saved) + 2.0f);
		if ( !Run(SparkOpsRouteCombineShared(&ops,queue,Whole(&outs),Whole(&shared),Whole(&combined),&route),"route_combine_shared") )
			break;
		sv[(uint64_t)(rows - 1u) * WIDTH + 13u] = saved;
		for ( row = 0u; row < rows; row++ )
			for ( i = 0u; i < WIDTH; i++ )
			{
				float total = 0.0f;
				for ( k = 0u; k < TOP; k++ )
					total += Bf16(ov[((uint64_t)row * TOP + k) * WIDTH + i]);
				total += Bf16(sv[(uint64_t)row * WIDTH + i]);
				Observe(check,Bf16(cv[(uint64_t)row * WIDTH + i]),Bf16(ToBf16(total)),0.0);
			}
	}
	snprintf(shape,sizeof(shape),"rows %u x top 10 + shared, width 2560",rows);
	Report("route_combine_shared",shape,&clean,&corrupted,"exact: routes in order, then the shared row, in FP32, rounded once");
	free(original);
	SparkDeviceFree(&outs);
	SparkDeviceFree(&shared);
	SparkDeviceFree(&combined);
	SparkDeviceFree(&logits);
}

static int64_t NgramWrapMul(int64_t a, int64_t b)
{
	return((int64_t)((uint64_t)a * (uint64_t)b));
}

static void TestNgram(void)
{
	enum { ROWS = 8u, SEQUENCES = 2u, ORDERS = 2u, HEADS = 8u, COLUMNS = ORDERS * HEADS, DIM = 160u, STRIDE = 4u, PRIME = 1009u, EOS = 7u };
	static const uint32_t tokens_init[ROWS] = { 11u, 12u, 7u, 13u, 14u, 21u, 22u, 23u };
	static const uint32_t positions_init[ROWS] = { 0u, 1u, 2u, 3u, 4u, 10u, 11u, 12u };
	SparkDeviceBuffer tokens, positions, begin, state, history, multipliers, primes, offsets, ids, payload, scale, embed;
	uint32_t *tv = (uint32_t *)Shared(&tokens,ROWS * 4u,"ngram tokens");
	uint32_t *pv = (uint32_t *)Shared(&positions,ROWS * 4u,"ngram positions");
	uint32_t *bv = (uint32_t *)Shared(&begin,(SEQUENCES + 1u) * 4u,"ngram row begin");
	uint32_t *sv = (uint32_t *)Shared(&state,SEQUENCES * 4u,"ngram slots");
	int32_t *hv = (int32_t *)Shared(&history,2u * STRIDE * 4u,"ngram history");
	int64_t *mv = (int64_t *)Shared(&multipliers,(ORDERS + 1u) * 8u,"ngram multipliers");
	int64_t *qv = (int64_t *)Shared(&primes,COLUMNS * 8u,"ngram primes");
	int64_t *ov = (int64_t *)Shared(&offsets,COLUMNS * 8u,"ngram offsets");
	int64_t *iv = (int64_t *)Shared(&ids,ROWS * COLUMNS * 8u,"ngram ids");
	uint8_t *yv = (uint8_t *)Shared(&payload,(uint64_t)COLUMNS * PRIME * DIM,"ngram table");
	float *fv = (float *)Shared(&scale,4u,"ngram scale");
	uint16_t *ev = (uint16_t *)Shared(&embed,ROWS * COLUMNS * DIM * 2u,"ngram embed");
	SparkOpsSequences sequences;
	SparkNgramParams params;
	Check clean, corrupted;
	int64_t expected[ROWS][COLUMNS];
	uint32_t pass, row, column, c;
	uint64_t index;
	memcpy(tv,tokens_init,sizeof(tokens_init));
	memcpy(pv,positions_init,sizeof(positions_init));
	bv[0] = 0u;
	bv[1] = 5u;
	bv[2] = ROWS;
	sv[0] = 1u;
	sv[1] = 0u;
	for ( index = 0u; index <= ORDERS; index++ )
		mv[index] = (int64_t)(NextRandom() | 1u);
	for ( column = 0u; column < COLUMNS; column++ )
	{
		qv[column] = PRIME - 2u * column;
		ov[column] = (int64_t)column * PRIME;
	}
	for ( index = 0u; index < (uint64_t)COLUMNS * PRIME * DIM; index++ )
		yv[index] = (uint8_t)(NextRandom() & 0x7eu);
	fv[0] = 0.375f;
	for ( row = 0u; row < ROWS; row++ )
	{
		const uint32_t lane = row < 5u ? 0u : 1u, begin_row = lane == 0u ? 0u : 5u, into = row - begin_row, slot = lane == 0u ? 1u : 0u;
		const int32_t carried[2] = { 31, 32 };
		int64_t shifted[ORDERS + 1u];
		uint32_t blocked = 0u, shift, order;
		for ( shift = 0u; shift <= ORDERS; shift++ )
		{
			int32_t source = -1;
			if ( positions_init[row] >= shift )
				source = shift <= into ? (int32_t)tokens_init[begin_row + into - shift] : (slot == 0u ? carried[shift - into - 1u] : 0);
			if ( positions_init[row] < shift || (shift >= 1u && source == (int32_t)EOS) )
				blocked = 1u;
			shifted[shift] = blocked != 0u ? (int64_t)EOS : source;
		}
		int64_t rolling = NgramWrapMul(shifted[0],mv[0]);
		for ( order = 1u; order <= ORDERS; order++ )
		{
			rolling ^= NgramWrapMul(shifted[order],mv[order]);
			for ( uint32_t head = 0u; head < HEADS; head++ )
				expected[row][(order - 1u) * HEADS + head] = rolling % qv[(order - 1u) * HEADS + head] + ov[(order - 1u) * HEADS + head];
		}
	}
	memset(&params,0,sizeof(params));
	params.rows = ROWS;
	params.sequences = SEQUENCES;
	params.orders = ORDERS;
	params.heads = HEADS;
	params.pad_token = EOS;
	params.block_token = EOS;
	params.block_from = 1u;
	params.history_stride = STRIDE;
	params.head_dimension = DIM;
	params.commit = 1u;
	params.table_rows = (uint64_t)COLUMNS * PRIME;
	sequences.state_index = Whole(&state);
	sequences.row_begin = Whole(&begin);
	sequences.sequences = SEQUENCES;
	sequences.commit = 1u;
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		memset(check,0,sizeof(*check));
		memset(hv,0,2u * STRIDE * 4u);
		hv[0] = 31;
		hv[1] = 32;
		if ( pass == 1u )
			tv[6] = tokens_init[6] + 1u;
		if ( !Run(SparkOpsNgramHash(&ops,queue,Whole(&tokens),Whole(&positions),&sequences,Whole(&history),Whole(&multipliers),Whole(&primes),Whole(&offsets),
			Whole(&ids),&params),"ngram_hash") )
			break;
		if ( !Run(SparkOpsNgramGatherFp8(&ops,queue,Whole(&ids),Whole(&payload),Whole(&scale),Whole(&embed),&params),"ngram_gather_fp8") )
			break;
		if ( !Run(SparkOpsNgramHistory(&ops,queue,Whole(&tokens),0,&sequences,Whole(&history),&params),"ngram_history") )
			break;
		tv[6] = tokens_init[6];
		for ( row = 0u; row < ROWS; row++ )
			for ( column = 0u; column < COLUMNS; column++ )
			{
				const int64_t id = expected[row][column];
				Observe(check,(double)iv[row * COLUMNS + column],(double)id,0.0);
				for ( c = 0u; c < DIM; c++ )
				{
					const double value = id >= 0 && (uint64_t)id < params.table_rows ? Bf16(ToBf16((float)E4m3(yv[(uint64_t)id * DIM + c]) * fv[0])) : 0.0;
					Observe(check,Bf16(ev[((uint64_t)row * COLUMNS + column) * DIM + c]),value,0.0);
				}
			}
		Observe(check,hv[STRIDE],14.0,0.0);
		Observe(check,hv[STRIDE + 1u],13.0,0.0);
		Observe(check,hv[0],23.0,0.0);
		Observe(check,hv[1],22.0,0.0);
	}
	Report("ngram_hash_gather_history","2 sequences, 8 rows, 2 orders x 8 heads, fp8 rows of 160",&clean,&corrupted,
		"exact: CUDA's hash (int64 wrap, C remainder, EOS blocking, carried history), e4m3 x scale rows, last tokens kept");
	SparkDeviceFree(&tokens);
	SparkDeviceFree(&positions);
	SparkDeviceFree(&begin);
	SparkDeviceFree(&state);
	SparkDeviceFree(&history);
	SparkDeviceFree(&multipliers);
	SparkDeviceFree(&primes);
	SparkDeviceFree(&offsets);
	SparkDeviceFree(&ids);
	SparkDeviceFree(&payload);
	SparkDeviceFree(&scale);
	SparkDeviceFree(&embed);
}

static void TestMimoKernels(uint32_t rows)
{
	enum { DIM = 4096u, FFN = 2048u, EXPERTS = 256u, TOP = 8u };
	const uint32_t routes = rows * TOP;
	SparkDeviceBuffer gate_up, act, outs, weights, combined, logits, bias, experts, route_weights, hidden, norm_weight, normed;
	uint16_t *gv = (uint16_t *)Shared(&gate_up,(uint64_t)rows * 2u * FFN * 2u,"mimo gate up");
	uint16_t *av = (uint16_t *)Shared(&act,(uint64_t)rows * FFN * 2u,"mimo act");
	uint16_t *ov = (uint16_t *)Shared(&outs,(uint64_t)routes * DIM * 2u,"mimo expert outputs");
	float *wv = (float *)Shared(&weights,(uint64_t)routes * sizeof(float),"mimo route weights in");
	uint16_t *cv = (uint16_t *)Shared(&combined,(uint64_t)rows * DIM * 2u,"mimo combined");
	float *lv = (float *)Shared(&logits,(uint64_t)rows * EXPERTS * sizeof(float),"mimo router logits");
	float *bv = (float *)Shared(&bias,EXPERTS * sizeof(float),"mimo router bias");
	uint32_t *ev = (uint32_t *)Shared(&experts,(uint64_t)routes * sizeof(uint32_t),"mimo chosen experts");
	float *rv = (float *)Shared(&route_weights,(uint64_t)routes * sizeof(float),"mimo chosen weights");
	uint16_t *hv = (uint16_t *)Shared(&hidden,(uint64_t)rows * DIM * 2u,"mimo hidden");
	uint16_t *nw = (uint16_t *)Shared(&norm_weight,DIM * 2u,"mimo norm weight");
	uint16_t *nv = (uint16_t *)Shared(&normed,(uint64_t)rows * DIM * 2u,"mimo normed");
	SparkSiluMulParams silu;
	SparkRouteParams route;
	SparkNormParams norm;
	Check clean, corrupted;
	char shape[64];
	uint32_t pass, row, i, k;
	FillBf16(gv,(uint64_t)rows * 2u * FFN,2.0);
	FillBf16(ov,(uint64_t)routes * DIM,1.0);
	FillBf16(hv,(uint64_t)rows * DIM,2.0);
	FillBf16(nw,DIM,1.0);
	for ( i = 0u; i < routes; i++ )
		wv[i] = (float)fabs(Normal()) / (float)TOP;
	for ( i = 0u; i < rows * EXPERTS; i++ )
		lv[i] = (float)(2.0 * Normal());
	for ( i = 0u; i < EXPERTS; i++ )
		bv[i] = (float)(0.1 * Normal());
	memset(&silu,0,sizeof(silu));
	silu.rows = rows;
	silu.dimension = FFN;
	silu.input_stride = 2u * FFN;
	silu.output_stride = FFN;
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		const uint16_t saved = gv[(uint64_t)(rows - 1u) * 2u * FFN + 7u];
		memset(check,0,sizeof(*check));
		if ( pass == 1u )
			gv[(uint64_t)(rows - 1u) * 2u * FFN + 7u] = ToBf16(Bf16(saved) + 4.0f);
		if ( !Run(SparkOpsGatedMul(&ops,queue,Whole(&gate_up),Whole(&act),&silu,SPARK_GATED_SILU_ROUNDED),"silu_rounded_mul") )
			break;
		gv[(uint64_t)(rows - 1u) * 2u * FFN + 7u] = saved;
		for ( row = 0u; row < rows; row++ )
			for ( i = 0u; i < FFN; i++ )
			{
				const double g = Bf16(gv[(uint64_t)row * 2u * FFN + i]), u = Bf16(gv[(uint64_t)row * 2u * FFN + FFN + i]);
				const double activated = RoundBf16(Silu(g)), y = RoundBf16(activated * u);
				Observe(check,Bf16(av[(uint64_t)row * FFN + i]),y,UNIT_BF16 * fabs(y) + fabs(u) * FlipSlack(Silu(g),2.0 * FUNCTION_ERROR));
			}
	}
	snprintf(shape,sizeof(shape),"rows %u, ffn 2048",rows);
	Report("silu_rounded_mul",shape,&clean,&corrupted,"bf16(bf16(silu(g)) u): u|y| + |u| flip slack of the silu rounding");
	memset(&route,0,sizeof(route));
	route.rows = rows;
	route.top_k = TOP;
	route.width = DIM;
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		const float saved = wv[routes - 1u];
		memset(check,0,sizeof(*check));
		if ( pass == 1u )
			wv[routes - 1u] = saved + 1.0f;
		if ( !Run(SparkOpsRouteCombineWeighted(&ops,queue,Whole(&outs),Whole(&weights),Whole(&combined),&route),"route_combine_weighted") )
			break;
		wv[routes - 1u] = saved;
		for ( row = 0u; row < rows; row++ )
			for ( i = 0u; i < DIM; i++ )
			{
				float total = 0.0f;
				for ( k = 0u; k < TOP; k++ )
					total = fmaf(Bf16(ov[((uint64_t)row * TOP + k) * DIM + i]),wv[row * TOP + k],total);
				Observe(check,Bf16(cv[(uint64_t)row * DIM + i]),Bf16(ToBf16(total)),0.0);
			}
	}
	snprintf(shape,sizeof(shape),"rows %u x top 8, width 4096",rows);
	Report("route_combine_weighted",shape,&clean,&corrupted,"exact: an FP32 FMA chain in route order, one bf16 rounding");
	route.experts = EXPERTS;
	route.logit_stride = EXPERTS;
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		memset(check,0,sizeof(*check));
		if ( !Run(SparkOpsRouteSigmoid(&ops,queue,Whole(&logits),Whole(&bias),Whole(&experts),Whole(&route_weights),&route),"route_sigmoid") )
			break;
		for ( row = 0u; row < rows; row++ )
		{
			double score[EXPERTS], selection[EXPERTS], total = 0.0;
			uint32_t chosen[TOP], pick, expert;
			for ( expert = 0u; expert < EXPERTS; expert++ )
			{
				score[expert] = Sigmoid(lv[row * EXPERTS + expert]);
				selection[expert] = (double)(float)score[expert] + bv[expert];
			}
			for ( pick = 0u; pick < TOP; pick++ )
			{
				uint32_t best = 0u;
				for ( expert = 1u; expert < EXPERTS; expert++ )
					if ( selection[expert] > selection[best] )
						best = expert;
				chosen[pick] = best;
				selection[best] = -INFINITY;
				total += score[best];
			}
			for ( pick = 0u; pick < TOP; pick++ )
			{
				const uint32_t got = pass == 1u && pick == TOP - 1u ? (ev[row * TOP + pick] + 1u) % EXPERTS : ev[row * TOP + pick];
				const double weight = score[chosen[pick]] / total;
				Observe(check,(double)got,(double)chosen[pick],0.0);
				Observe(check,rv[row * TOP + pick],weight,8.0 * FUNCTION_ERROR * weight);
			}
		}
	}
	snprintf(shape,sizeof(shape),"rows %u, 256 experts, top 8, bias",rows);
	Report("route_sigmoid",shape,&clean,&corrupted,"experts exact; weights to 8*16 eps32 relative");
	memset(&norm,0,sizeof(norm));
	norm.rows = rows;
	norm.dimension = DIM;
	norm.input_stride = norm.output_stride = DIM;
	norm.epsilon = 1e-6f;
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		const uint16_t saved = hv[(uint64_t)(rows - 1u) * DIM + 11u];
		memset(check,0,sizeof(*check));
		if ( pass == 1u )
			hv[(uint64_t)(rows - 1u) * DIM + 11u] = ToBf16(Bf16(saved) * -4.0f);
		if ( !Run(SparkOpsRmsNormBf16(&ops,queue,Whole(&hidden),Whole(&norm_weight),Whole(&normed),&norm),"rmsnorm_bf16") )
			break;
		hv[(uint64_t)(rows - 1u) * DIM + 11u] = saved;
		for ( row = 0u; row < rows; row++ )
		{
			double square = 0.0;
			for ( i = 0u; i < DIM; i++ )
				square += (double)Bf16(hv[(uint64_t)row * DIM + i]) * Bf16(hv[(uint64_t)row * DIM + i]);
			const double scale = 1.0 / sqrt(square / DIM + 1e-6);
			for ( i = 0u; i < DIM; i++ )
			{
				const double exact = Bf16(hv[(uint64_t)row * DIM + i]) * scale, w = Bf16(nw[i]), y = RoundBf16(RoundBf16(exact) * w);
				const double slack = fabs(w) * FlipSlack(exact,Gamma(DIM) + FUNCTION_ERROR + 3.0 * UNIT_F32);
				Observe(check,Bf16(nv[(uint64_t)row * DIM + i]),y,UNIT_BF16 * fabs(y) + slack + (slack != 0.0 ? UNIT_BF16 * fabs(y) : 0.0));
			}
		}
	}
	snprintf(shape,sizeof(shape),"rows %u, dim 4096, bf16 weight",rows);
	Report("rmsnorm_bf16",shape,&clean,&corrupted,"bf16(bf16(x s) w): u|y| + |w| flip slack of the first rounding");
	SparkDeviceFree(&gate_up);
	SparkDeviceFree(&act);
	SparkDeviceFree(&outs);
	SparkDeviceFree(&weights);
	SparkDeviceFree(&combined);
	SparkDeviceFree(&logits);
	SparkDeviceFree(&bias);
	SparkDeviceFree(&experts);
	SparkDeviceFree(&route_weights);
	SparkDeviceFree(&hidden);
	SparkDeviceFree(&norm_weight);
	SparkDeviceFree(&normed);
}

static double HyperGate(double value, double sigmoid_factor)
{
	const double scaled = RoundBf16(value / 4.0);
	return(RoundBf16(scaled * Sigmoid(scaled) * sigmoid_factor));
}

static double HyperInjectWeight(double logit, double sigmoid_factor)
{
	return(RoundBf16(2.0 * RoundBf16(Sigmoid(RoundBf16(logit / 4.0)) * sigmoid_factor)));
}

static double HyperKeyGate(double total, double sigmoid_factor)
{
	const double score = RoundBf16(RoundBf16(total) / sqrt(2560.0));
	const double magnitude = RoundBf16(sqrt(fmax(fabs(score),1e-6)));
	return(RoundBf16(Sigmoid(score < 0.0 ? -magnitude : (score > 0.0 ? magnitude : 0.0)) * sigmoid_factor));
}

static void TestHyperKernels(uint32_t rows)
{
	enum { STREAMS = 4u, WIDTH = 2560u, LOW = 320u, TOTAL = STREAMS * WIDTH };
	static const double factors[3] = { 1.0 - 2.0 * FUNCTION_ERROR, 1.0, 1.0 + 2.0 * FUNCTION_ERROR };
	SparkDeviceBuffer hidden, streams, gain, normed, low, up, mixed, inject_weight, inject, output, key, query, value, gated, conv;
	uint16_t *hv = (uint16_t *)Shared(&hidden,(uint64_t)rows * WIDTH * 2u,"hyper hidden");
	uint16_t *sv = (uint16_t *)Shared(&streams,(uint64_t)rows * TOTAL * 2u,"hyper streams");
	float *gv = (float *)Shared(&gain,TOTAL * sizeof(float),"hyper group gain");
	uint16_t *nv = (uint16_t *)Shared(&normed,(uint64_t)rows * TOTAL * 2u,"hyper normed");
	uint16_t *lv = (uint16_t *)Shared(&low,(uint64_t)rows * LOW * 2u,"hyper low rank");
	uint16_t *uv = (uint16_t *)Shared(&up,(uint64_t)rows * TOTAL * 2u,"hyper up");
	uint16_t *mv = (uint16_t *)Shared(&mixed,(uint64_t)rows * WIDTH * 2u,"hyper mixed");
	uint16_t *wv = (uint16_t *)Shared(&inject_weight,(uint64_t)STREAMS * TOTAL * 2u,"hyper inject weight");
	float *iv = (float *)Shared(&inject,(uint64_t)rows * STREAMS * sizeof(float),"hyper inject");
	uint16_t *ov = (uint16_t *)Shared(&output,(uint64_t)rows * WIDTH * 2u,"hyper sublayer output");
	uint16_t *kv = (uint16_t *)Shared(&key,(uint64_t)rows * TOTAL * 2u,"hyper key");
	uint16_t *qv = (uint16_t *)Shared(&query,(uint64_t)rows * TOTAL * 2u,"hyper query");
	uint16_t *vv = (uint16_t *)Shared(&value,(uint64_t)rows * WIDTH * 2u,"hyper value");
	uint16_t *xv = (uint16_t *)Shared(&gated,(uint64_t)rows * TOTAL * 2u,"hyper gated");
	uint16_t *cv = (uint16_t *)Shared(&conv,(uint64_t)rows * TOTAL * 2u,"hyper conv");
	uint16_t *original = (uint16_t *)malloc((uint64_t)rows * TOTAL * 2u);
	float inject_values[64];
	SparkHyperParams hyper;
	Check clean, corrupted;
	char shape[64];
	uint32_t pass, row, stream, i;
	if ( original == 0 || rows * STREAMS > 64u )
	{
		printf("FAIL hyper: test buffers\n");
		failures++;
		return;
	}
	FillBf16(hv,(uint64_t)rows * WIDTH,1.0);
	FillBf16(nv,(uint64_t)rows * TOTAL,1.0);
	FillBf16(uv,(uint64_t)rows * TOTAL,2.0);
	FillBf16(wv,(uint64_t)STREAMS * TOTAL,0.02);
	FillBf16(ov,(uint64_t)rows * WIDTH,1.0);
	FillBf16(kv,(uint64_t)rows * TOTAL,1.0);
	FillBf16(qv,(uint64_t)rows * TOTAL,1.0);
	FillBf16(vv,(uint64_t)rows * WIDTH,1.0);
	FillBf16(xv,(uint64_t)rows * TOTAL,1.0);
	FillBf16(cv,(uint64_t)rows * TOTAL,1.0);
	for ( i = 0u; i < TOTAL; i++ )
		gv[i] = (float)(1.0 + 0.1 * Normal());
	for ( i = 0u; i < rows * STREAMS; i++ )
		inject_values[i] = (float)(2.0 * Uniform());
	memset(&hyper,0,sizeof(hyper));
	hyper.rows = rows;
	hyper.streams = STREAMS;
	hyper.width = WIDTH;
	hyper.epsilon = 1e-6f;
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		const uint16_t saved = hv[(uint64_t)(rows - 1u) * WIDTH + 5u];
		memset(check,0,sizeof(*check));
		if ( pass == 1u )
			hv[(uint64_t)(rows - 1u) * WIDTH + 5u] = ToBf16(Bf16(saved) + 1.0f);
		if ( !Run(SparkOpsHyperExpand(&ops,queue,Whole(&hidden),Whole(&streams),&hyper),"hyper_expand") )
			break;
		hv[(uint64_t)(rows - 1u) * WIDTH + 5u] = saved;
		for ( row = 0u; row < rows; row++ )
			for ( i = 0u; i < TOTAL; i++ )
				Observe(check,Bf16(sv[(uint64_t)row * TOTAL + i]),Bf16(hv[(uint64_t)row * WIDTH + i % WIDTH]),0.0);
	}
	snprintf(shape,sizeof(shape),"rows %u, 4 streams x 2560",rows);
	Report("hyper_expand",shape,&clean,&corrupted,"exact copy into every stream");
	FillBf16(sv,(uint64_t)rows * TOTAL,1.5);
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		const uint16_t saved = sv[(uint64_t)(rows - 1u) * TOTAL + 3u * WIDTH + 9u];
		memset(check,0,sizeof(*check));
		if ( pass == 1u )
			sv[(uint64_t)(rows - 1u) * TOTAL + 3u * WIDTH + 9u] = ToBf16(Bf16(saved) * -4.0f);
		if ( !Run(SparkOpsGroupRmsNorm(&ops,queue,Whole(&streams),Whole(&gain),Whole(&normed),&hyper),"group_rmsnorm") )
			break;
		sv[(uint64_t)(rows - 1u) * TOTAL + 3u * WIDTH + 9u] = saved;
		for ( row = 0u; row < rows; row++ )
			for ( stream = 0u; stream < STREAMS; stream++ )
			{
				const uint16_t *x = sv + (uint64_t)row * TOTAL + stream * WIDTH;
				double square = 0.0;
				for ( i = 0u; i < WIDTH; i++ )
					square += (double)Bf16(x[i]) * Bf16(x[i]);
				const double scale = 1.0 / sqrt(square / WIDTH + 1e-6);
				for ( i = 0u; i < WIDTH; i++ )
				{
					const double exact = Bf16(x[i]) * scale * gv[stream * WIDTH + i];
					Observe(check,Bf16(nv[(uint64_t)row * TOTAL + stream * WIDTH + i]),RoundBf16(exact),
						FlipSlack(exact,Gamma(WIDTH) + FUNCTION_ERROR + 3.0 * UNIT_F32));
				}
			}
	}
	snprintf(shape,sizeof(shape),"rows %u, 4 groups x 2560, f32 gain",rows);
	Report("group_rmsnorm",shape,&clean,&corrupted,"bf16(x s w): flip slack of the one rounding");
	FillBf16(lv,(uint64_t)rows * LOW,4.0);
	memcpy(original,lv,(size_t)rows * LOW * 2u);
	hyper.width = LOW;
	hyper.streams = 1u;
	hyper.divisor = 4.0f;
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		memset(check,0,sizeof(*check));
		memcpy(lv,original,(size_t)rows * LOW * 2u);
		if ( pass == 1u )
			lv[(uint64_t)(rows - 1u) * LOW + 17u] = ToBf16(Bf16(lv[(uint64_t)(rows - 1u) * LOW + 17u]) + 2.0f);
		if ( !Run(SparkOpsHyperGate(&ops,queue,Whole(&low),&hyper),"hyper_gate") )
			break;
		for ( i = 0u; i < rows * LOW; i++ )
		{
			const double expected = HyperGate(Bf16(original[i]),1.0);
			double tolerance = 0.0;
			for ( uint32_t f = 0u; f < 3u; f++ )
				tolerance = fmax(tolerance,fabs(HyperGate(Bf16(original[i]),factors[f]) - expected));
			Observe(check,Bf16(lv[i]),expected,tolerance);
		}
	}
	snprintf(shape,sizeof(shape),"rows %u x 320, divisor 4",rows);
	Report("hyper_gate",shape,&clean,&corrupted,"v = bf16(x / 4), bf16(v sigmoid(v)): exp to 2*16 eps32");
	hyper.width = WIDTH;
	hyper.streams = STREAMS;
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		const uint16_t saved = uv[(uint64_t)(rows - 1u) * TOTAL + 2u * WIDTH + 33u];
		memset(check,0,sizeof(*check));
		if ( pass == 1u )
			uv[(uint64_t)(rows - 1u) * TOTAL + 2u * WIDTH + 33u] = ToBf16(Bf16(saved) - 6.0f);
		if ( !Run(SparkOpsHyperMix(&ops,queue,Whole(&up),Whole(&normed),Whole(&mixed),&hyper),"hyper_mix") )
			break;
		uv[(uint64_t)(rows - 1u) * TOTAL + 2u * WIDTH + 33u] = saved;
		for ( row = 0u; row < rows; row++ )
			for ( i = 0u; i < WIDTH; i++ )
			{
				float total = 0.0f, low_total = 0.0f, high_total = 0.0f;
				for ( stream = 0u; stream < STREAMS; stream++ )
				{
					const uint64_t at = (uint64_t)row * TOTAL + stream * WIDTH + i;
					const double n = Bf16(nv[at]), sigma = Sigmoid(Bf16(uv[at]));
					double lowest = INFINITY, highest = -INFINITY;
					for ( uint32_t f = 0u; f < 3u; f++ )
					{
						const double term = RoundBf16(RoundBf16(sigma * factors[f]) * n);
						lowest = fmin(lowest,term);
						highest = fmax(highest,term);
					}
					total += (float)RoundBf16(RoundBf16(sigma) * n);
					low_total += (float)lowest;
					high_total += (float)highest;
				}
				const double expected = RoundBf16(total / 4.0f);
				Observe(check,Bf16(mv[(uint64_t)row * WIDTH + i]),expected,
					fmax(fabs(RoundBf16(low_total / 4.0f) - expected),fabs(RoundBf16(high_total / 4.0f) - expected)));
			}
	}
	snprintf(shape,sizeof(shape),"rows %u, 4 streams x 2560",rows);
	Report("hyper_mix",shape,&clean,&corrupted,"bf16(sum bf16(bf16(sigmoid(u)) n) / 4) in stream order: sigmoid to 2*16 eps32");
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		const uint16_t saved = wv[2u * TOTAL + 100u];
		memset(check,0,sizeof(*check));
		if ( pass == 1u )
			wv[2u * TOTAL + 100u] = ToBf16(Bf16(saved) + 50.0f);
		if ( !Run(SparkOpsHyperInjectWeights(&ops,queue,Whole(&normed),Whole(&inject_weight),Whole(&inject),&hyper),"hyper_inject_weights") )
			break;
		wv[2u * TOTAL + 100u] = saved;
		for ( row = 0u; row < rows; row++ )
			for ( stream = 0u; stream < STREAMS; stream++ )
			{
				double sum = 0.0, magnitude = 0.0, tolerance = 0.0;
				for ( i = 0u; i < TOTAL; i++ )
				{
					const double product = (double)Bf16(nv[(uint64_t)row * TOTAL + i]) * Bf16(wv[(uint64_t)stream * TOTAL + i]);
					sum += product;
					magnitude += fabs(product);
				}
				const double bound = Gamma(TOTAL) * magnitude, expected = HyperInjectWeight(RoundBf16(sum),1.0);
				for ( uint32_t edge = 0u; edge < 3u; edge++ )
					for ( uint32_t f = 0u; f < 3u; f++ )
						tolerance = fmax(tolerance,fabs(HyperInjectWeight(RoundBf16(sum + ((double)edge - 1.0) * bound),factors[f]) - expected));
				Observe(check,iv[row * STREAMS + stream],expected,tolerance);
			}
	}
	snprintf(shape,sizeof(shape),"rows %u, 4 x 10240 dot",rows);
	Report("hyper_inject_weights",shape,&clean,&corrupted,"bf16(2 bf16(sigmoid(bf16(bf16(n.w) / 4)))): sum to gamma(10240), sigmoid 2*16 eps32");
	for ( i = 0u; i < rows * STREAMS; i++ )
		iv[i] = inject_values[i];
	FillBf16(sv,(uint64_t)rows * TOTAL,1.0);
	memcpy(original,sv,(size_t)rows * TOTAL * 2u);
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		memset(check,0,sizeof(*check));
		memcpy(sv,original,(size_t)rows * TOTAL * 2u);
		if ( pass == 1u )
			iv[rows * STREAMS - 1u] = inject_values[rows * STREAMS - 1u] + 1.0f;
		if ( !Run(SparkOpsHyperInject(&ops,queue,Whole(&streams),Whole(&output),Whole(&inject),&hyper),"hyper_inject") )
			break;
		iv[rows * STREAMS - 1u] = inject_values[rows * STREAMS - 1u];
		for ( row = 0u; row < rows; row++ )
			for ( i = 0u; i < TOTAL; i++ )
			{
				const float injected = (float)RoundBf16(Bf16(ov[(uint64_t)row * WIDTH + i % WIDTH]) * inject_values[row * STREAMS + i / WIDTH]);
				Observe(check,Bf16(sv[(uint64_t)row * TOTAL + i]),Bf16(ToBf16(Bf16(original[(uint64_t)row * TOTAL + i]) + injected)),0.0);
			}
	}
	snprintf(shape,sizeof(shape),"rows %u, 4 streams x 2560",rows);
	Report("hyper_inject",shape,&clean,&corrupted,"exact: bf16(r + bf16(out w))");
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		const uint16_t saved = qv[(uint64_t)(rows - 1u) * TOTAL + WIDTH + 7u];
		memset(check,0,sizeof(*check));
		if ( pass == 1u )
			qv[(uint64_t)(rows - 1u) * TOTAL + WIDTH + 7u] = ToBf16(Bf16(saved) + 300.0f);
		if ( !Run(SparkOpsHyperKeyGate(&ops,queue,Whole(&key),Whole(&query),Whole(&value),Whole(&gated),&hyper),"hyper_key_gate") )
			break;
		qv[(uint64_t)(rows - 1u) * TOTAL + WIDTH + 7u] = saved;
		for ( row = 0u; row < rows; row++ )
			for ( stream = 0u; stream < STREAMS; stream++ )
			{
				const uint64_t base = (uint64_t)row * TOTAL + stream * WIDTH;
				double sum = 0.0, magnitude = 0.0, gates[9];
				for ( i = 0u; i < WIDTH; i++ )
				{
					const double product = RoundBf16((double)Bf16(kv[base + i]) * Bf16(qv[base + i]));
					sum += product;
					magnitude += fabs(product);
				}
				const double bound = Gamma(WIDTH) * magnitude, gate = HyperKeyGate(sum,1.0);
				for ( uint32_t edge = 0u; edge < 3u; edge++ )
					for ( uint32_t f = 0u; f < 3u; f++ )
						gates[edge * 3u + f] = HyperKeyGate(sum + ((double)edge - 1.0) * bound,factors[f]);
				for ( i = 0u; i < WIDTH; i++ )
				{
					const double v = Bf16(vv[(uint64_t)row * WIDTH + i]), expected = RoundBf16(gate * v);
					double tolerance = 0.0;
					for ( uint32_t g = 0u; g < 9u; g++ )
						tolerance = fmax(tolerance,fabs(RoundBf16(gates[g] * v) - expected));
					Observe(check,Bf16(xv[base + i]),expected,tolerance);
				}
			}
	}
	snprintf(shape,sizeof(shape),"rows %u, 4 streams x 2560",rows);
	Report("hyper_key_gate",shape,&clean,&corrupted,"bf16(gate v), gate from bf16(sum bf16(k q)): sum to gamma(2560), sigmoid 2*16 eps32");
	FillBf16(cv,(uint64_t)rows * TOTAL,1.0);
	memcpy(original,sv,(size_t)rows * TOTAL * 2u);
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		const uint16_t saved = cv[(uint64_t)(rows - 1u) * TOTAL + 21u];
		memset(check,0,sizeof(*check));
		memcpy(sv,original,(size_t)rows * TOTAL * 2u);
		if ( pass == 1u )
			cv[(uint64_t)(rows - 1u) * TOTAL + 21u] = ToBf16(Bf16(saved) + 3.0f);
		if ( !Run(SparkOpsHyperAdd(&ops,queue,Whole(&streams),Whole(&gated),Whole(&conv),&hyper),"hyper_add") )
			break;
		cv[(uint64_t)(rows - 1u) * TOTAL + 21u] = saved;
		for ( i = 0u; i < rows * TOTAL; i++ )
		{
			const float added = (float)RoundBf16(Bf16(xv[i]) + Bf16(cv[i]));
			Observe(check,Bf16(sv[i]),Bf16(ToBf16(Bf16(original[i]) + added)),0.0);
		}
	}
	snprintf(shape,sizeof(shape),"rows %u, 4 streams x 2560",rows);
	Report("hyper_add",shape,&clean,&corrupted,"exact: bf16(r + bf16(g + c))");
	free(original);
	SparkDeviceFree(&hidden);
	SparkDeviceFree(&streams);
	SparkDeviceFree(&gain);
	SparkDeviceFree(&normed);
	SparkDeviceFree(&low);
	SparkDeviceFree(&up);
	SparkDeviceFree(&mixed);
	SparkDeviceFree(&inject_weight);
	SparkDeviceFree(&inject);
	SparkDeviceFree(&output);
	SparkDeviceFree(&key);
	SparkDeviceFree(&query);
	SparkDeviceFree(&value);
	SparkDeviceFree(&gated);
	SparkDeviceFree(&conv);
}

static void TestBranchNorms(uint32_t rows, uint32_t branches)
{
	enum { DIM = 2816u };
	SparkDeviceBuffer hidden, first, second, weights, scalar, routed;
	uint16_t *h = (uint16_t *)Shared(&hidden,(uint64_t)rows * DIM * sizeof(uint16_t),"branch hidden");
	uint16_t *a = (uint16_t *)Shared(&first,(uint64_t)rows * DIM * sizeof(uint16_t),"branch first");
	uint16_t *b = (uint16_t *)Shared(&second,(uint64_t)rows * DIM * sizeof(uint16_t),"branch second");
	float *wv = (float *)Shared(&weights,3u * DIM * sizeof(float),"branch weights");
	float *sc = (float *)Shared(&scalar,sizeof(float),"branch scalar");
	uint16_t *r = (uint16_t *)Shared(&routed,(uint64_t)rows * DIM * sizeof(uint16_t),"routed input");
	uint16_t *saved = (uint16_t *)malloc((uint64_t)rows * DIM * sizeof(uint16_t));
	SparkTensor w1 = SparkTensorSlice(Whole(&weights),0u,DIM * sizeof(float)), w2 = SparkTensorSlice(Whole(&weights),DIM * sizeof(float),DIM * sizeof(float));
	SparkTensor wo = SparkTensorSlice(Whole(&weights),2u * DIM * sizeof(float),DIM * sizeof(float)), sct = Whole(&scalar), bt = Whole(&second);
	SparkBranchNormParams params;
	Check clean, corrupted;
	char shape[64];
	uint32_t pass, row, i;
	FillBf16(h,(uint64_t)rows * DIM,1.0);
	FillBf16(a,(uint64_t)rows * DIM,3.0);
	FillBf16(b,(uint64_t)rows * DIM,0.5);
	for ( i = 0u; i < 3u * DIM; i++ )
		wv[i] = (float)(0.5 + Uniform());
	sc[0] = 0.8125f;
	memcpy(saved,h,(uint64_t)rows * DIM * sizeof(uint16_t));
	memset(&params,0,sizeof(params));
	params.rows = rows;
	params.dimension = DIM;
	params.branch_stride = params.hidden_stride = DIM;
	params.branches = branches;
	params.has_scalar = branches == 2u ? 1u : 0u;
	params.epsilon = 1e-6f;
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		const uint64_t poke = (uint64_t)(rows - 1u) * DIM + 1000u;
		const uint16_t keep = a[poke];
		memset(check,0,sizeof(*check));
		memcpy(h,saved,(uint64_t)rows * DIM * sizeof(uint16_t));
		if ( pass == 1u )
			a[poke] = ToBf16(Bf16(keep) + 8.0f);
		if ( !Run(SparkOpsRmsNormBranches(&ops,queue,Whole(&hidden),Whole(&first),branches == 2u ? &bt : 0,branches == 2u ? &w1 : 0,branches == 2u ? &w2 : 0,wo,
			branches == 2u ? &sct : 0,&params),"rmsnorm_branches") )
			break;
		a[poke] = keep;
		for ( row = 0u; row < rows; row++ )
		{
			double sa = 0.0, sb = 0.0, sc2 = 0.0, combined[DIM], slack[DIM];
			for ( i = 0u; i < DIM; i++ )
			{
				sa += (double)Bf16(a[row * DIM + i]) * Bf16(a[row * DIM + i]);
				sb += (double)Bf16(b[row * DIM + i]) * Bf16(b[row * DIM + i]);
			}
			sa = 1.0 / sqrt(sa / DIM + 1e-6);
			sb = 1.0 / sqrt(sb / DIM + 1e-6);
			for ( i = 0u; i < DIM; i++ )
			{
				const double x = Bf16(a[row * DIM + i]) * sa * (branches == 2u ? wv[i] : wv[2u * DIM + i]);
				if ( branches == 2u )
				{
					const double z = Bf16(b[row * DIM + i]) * sb * wv[DIM + i];
					combined[i] = RoundBf16(RoundBf16(x) + RoundBf16(z));
					slack[i] = UNIT_BF16 * (fabs(x) + fabs(z) + fabs(combined[i]));
				}
				else
				{
					combined[i] = RoundBf16(x);
					slack[i] = FlipSlack(x,Gamma(DIM) + FUNCTION_ERROR + 3.0 * UNIT_F32);
				}
				sc2 += combined[i] * combined[i];
			}
			sc2 = 1.0 / sqrt(sc2 / DIM + 1e-6);
			for ( i = 0u; i < DIM; i++ )
			{
				const double normed = branches == 2u ? RoundBf16(combined[i] * sc2 * wv[2u * DIM + i]) : combined[i];
				const double added = RoundBf16(Bf16(saved[row * DIM + i]) + normed), y = branches == 2u ? added * sc[0] : added;
				const double tolerance = UNIT_BF16 * (fabs(y) + fabs(added) + fabs(normed)) + (slack[i] + UNIT_BF16 * fabs(normed)) * (branches == 2u ? 2.0 : 1.0);
				Observe(check,Bf16(h[row * DIM + i]),y,tolerance);
			}
		}
	}
	snprintf(shape,sizeof(shape),"rows %u, %u branch%s x 2816",rows,branches,branches == 2u ? "es, layer scalar" : "");
	Report("rmsnorm_branches",shape,&clean,&corrupted,"u (|y| + |sum| + |normed|) + the flip slack of each branch's bf16 rounding");
	{
		SparkBranchNormParams scaled = params;
		scaled.output_scale = (float)(1.0 / sqrt((double)DIM));
		for ( pass = 0u; pass < 2u; pass++ )
		{
			Check *check = pass == 0u ? &clean : &corrupted;
			const uint64_t poke = (uint64_t)(rows - 1u) * DIM + 2000u;
			const uint16_t keep = a[poke];
			memset(check,0,sizeof(*check));
			if ( pass == 1u )
				a[poke] = ToBf16(Bf16(keep) + 8.0f);
			if ( !Run(SparkOpsRmsNormScaled(&ops,queue,Whole(&first),wo,Whole(&routed),&scaled),"rmsnorm_scaled") )
				break;
			a[poke] = keep;
			for ( row = 0u; row < rows; row++ )
			{
				double square = 0.0;
				for ( i = 0u; i < DIM; i++ )
					square += (double)Bf16(a[row * DIM + i]) * Bf16(a[row * DIM + i]);
				square = 1.0 / sqrt(square / DIM + 1e-6);
				for ( i = 0u; i < DIM; i++ )
				{
					const double normed = RoundBf16(Bf16(a[row * DIM + i]) * square), gained = RoundBf16(normed * wv[2u * DIM + i]);
					const double y = gained * scaled.output_scale;
					Observe(check,Bf16(r[row * DIM + i]),y,UNIT_BF16 * (fabs(y) + 2.0 * fabs(gained) * scaled.output_scale) +
						fabs(wv[2u * DIM + i]) * scaled.output_scale * FlipSlack(Bf16(a[row * DIM + i]) * square,Gamma(DIM) + FUNCTION_ERROR + 3.0 * UNIT_F32));
				}
			}
		}
		snprintf(shape,sizeof(shape),"rows %u x 2816, weightless norm, scale, 1/sqrt(2816)",rows);
		Report("rmsnorm_scaled",shape,&clean,&corrupted,"u (|y| + 2|gained| c) + |w| c flip slack of the normed rounding");
	}
	free(saved);
	SparkDeviceFree(&hidden);
	SparkDeviceFree(&first);
	SparkDeviceFree(&second);
	SparkDeviceFree(&weights);
	SparkDeviceFree(&scalar);
	SparkDeviceFree(&routed);
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

static void TestCausalConvDilated(void)
{
	enum { CHANNELS = 10240, TAPS = 4, DILATION = 3, SPAN = (TAPS - 1) * DILATION + 1, FIRST = 7, SECOND = 4, ROWS = FIRST + SECOND };
	SparkDeviceBuffer window, index, begin, input, weight, output;
	uint16_t *win = (uint16_t *)Shared(&window,(uint64_t)CHANNELS * SPAN * 2u,"dilated conv window");
	uint32_t *slot = (uint32_t *)Shared(&index,4u,"dilated conv slot");
	uint32_t *rb = (uint32_t *)Shared(&begin,8u,"dilated conv begin");
	uint16_t *in = (uint16_t *)Shared(&input,(uint64_t)ROWS * CHANNELS * 2u,"dilated conv input");
	uint16_t *w = (uint16_t *)Shared(&weight,CHANNELS * TAPS * 2u,"dilated conv weight");
	uint16_t *out = (uint16_t *)Shared(&output,(uint64_t)ROWS * CHANNELS * 2u,"dilated conv output");
	SparkOpsSequences sequences;
	SparkConvParams params;
	Check clean, corrupted;
	uint32_t pass, row, c, tap;
	FillBf16(in,(uint64_t)ROWS * CHANNELS,1.0);
	FillBf16(w,CHANNELS * TAPS,0.5);
	memset(&params,0,sizeof(params));
	params.channels = CHANNELS;
	params.row_stride = CHANNELS;
	params.sequences = 1u;
	params.kernel_taps = TAPS;
	params.dilation = DILATION;
	params.commit = 1u;
	sequences.state_index = Whole(&index);
	sequences.row_begin = Whole(&begin);
	sequences.sequences = 1u;
	sequences.commit = 1u;
	slot[0] = 0u;
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		const uint16_t saved = w[77u * TAPS];
		memset(check,0,sizeof(*check));
		memset(win,0,(size_t)CHANNELS * SPAN * 2u);
		if ( pass == 1u )
			w[77u * TAPS] = ToBf16(Bf16(saved) + 2.0f);
		rb[0] = 0u;
		rb[1] = FIRST;
		if ( !Run(SparkOpsCausalConv(&ops,queue,Whole(&window),&sequences,Whole(&input),Whole(&weight),Whole(&output),&params),"causal_conv dilated first") )
			break;
		rb[0] = FIRST;
		rb[1] = ROWS;
		if ( !Run(SparkOpsCausalConv(&ops,queue,Whole(&window),&sequences,Whole(&input),Whole(&weight),Whole(&output),&params),"causal_conv dilated second") )
			break;
		w[77u * TAPS] = saved;
		for ( row = 0u; row < ROWS; row++ )
			for ( c = 0u; c < CHANNELS; c++ )
			{
				float total = 0.0f;
				for ( tap = 0u; tap < TAPS; tap++ )
				{
					const int32_t source = (int32_t)row - (int32_t)((TAPS - 1u - tap) * DILATION);
					total += (source >= 0 ? Bf16(in[(uint64_t)source * CHANNELS + c]) : 0.0f) * Bf16(w[c * TAPS + tap]);
				}
				const double rounded = RoundBf16(total), y = rounded * Sigmoid(rounded);
				Observe(check,Bf16(out[(uint64_t)row * CHANNELS + c]),y,UNIT_BF16 * fabs(y) + 2.0 * FUNCTION_ERROR * fabs(y));
			}
	}
	Report("causal_conv_dilated","10240 channels, 4 taps dilation 3, 7 rows then 4",&clean,&corrupted,
		"u|y| + 2*16 eps32 |y|: taps x[t-9], x[t-6], x[t-3], x[t] summed in fp32 oldest first, rounded to bf16, then swish; the window carries over");
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

static void RouteGroupsReference(const float *logits, const float *bias, uint32_t grouped, uint32_t *chosen, double *weight)
{
	enum { EXPERTS = 512u, GROUPS = 8u, OPEN = 4u, TOP = 8u, PER = EXPERTS / GROUPS };
	float score[EXPERTS], selection[EXPERTS], group_score[GROUPS], total = 0.0f;
	uint32_t open[GROUPS], expert, group, pick;
	for ( expert = 0u; expert < EXPERTS; expert++ )
	{
		score[expert] = (float)Sigmoid(logits[expert]);
		selection[expert] = score[expert] + bias[expert];
	}
	for ( group = 0u; group < GROUPS; group++ )
	{
		float first = -INFINITY, second = -INFINITY;
		for ( expert = group * PER; expert < (group + 1u) * PER; expert++ )
			if ( selection[expert] > first )
			{
				second = first;
				first = selection[expert];
			}
			else if ( selection[expert] > second )
				second = selection[expert];
		group_score[group] = first + second;
		open[group] = grouped != 0u ? 0u : 1u;
	}
	for ( pick = 0u; grouped != 0u && pick < OPEN; pick++ )
	{
		uint32_t best = GROUPS;
		for ( group = 0u; group < GROUPS; group++ )
			if ( open[group] == 0u && (best == GROUPS || group_score[group] > group_score[best]) )
				best = group;
		open[best] = 1u;
	}
	for ( pick = 0u; pick < TOP; pick++ )
	{
		uint32_t best = EXPERTS;
		for ( expert = 0u; expert < EXPERTS; expert++ )
			if ( open[expert / PER] != 0u && (best == EXPERTS || selection[expert] > selection[best]) )
				best = expert;
		chosen[pick] = best;
		selection[best] = -INFINITY;
		total += score[best];
	}
	for ( pick = 0u; pick < TOP; pick++ )
		weight[pick] = score[chosen[pick]] / (total + 1e-20f) * 2.5;
}

static void TestRouteGroups(uint32_t rows)
{
	enum { EXPERTS = 512u, TOP = 8u };
	SparkDeviceBuffer logits, bias, experts, weights;
	float *lv = (float *)Shared(&logits,(uint64_t)rows * EXPERTS * 4u,"group route logits");
	float *bv = (float *)Shared(&bias,EXPERTS * 4u,"group route bias");
	uint32_t *ev = (uint32_t *)Shared(&experts,(uint64_t)rows * TOP * 4u,"group route experts");
	float *wv = (float *)Shared(&weights,(uint64_t)rows * TOP * 4u,"group route weights");
	SparkRouteParams route;
	Check clean, corrupted;
	char shape[64];
	uint32_t pass, row, pick, chosen[TOP];
	double weight[TOP];
	for ( row = 0u; row < rows * EXPERTS; row++ )
		lv[row] = (float)(Normal() * 2.0);
	for ( row = 0u; row < EXPERTS; row++ )
		bv[row] = (float)(Normal() * 0.05);
	memset(&route,0,sizeof(route));
	route.rows = rows;
	route.experts = EXPERTS;
	route.top_k = TOP;
	route.width = 1u;
	route.logit_stride = EXPERTS;
	route.scale = 2.5f;
	route.groups = 8u;
	route.topk_groups = 4u;
	if ( Run(SparkOpsRouteSigmoid(&ops,queue,Whole(&logits),Whole(&bias),Whole(&experts),Whole(&weights),&route),"route_sigmoid groups") )
		for ( pass = 0u; pass < 2u; pass++ )
		{
			Check *check = pass == 0u ? &clean : &corrupted;
			memset(check,0,sizeof(*check));
			for ( row = 0u; row < rows; row++ )
			{
				RouteGroupsReference(lv + (uint64_t)row * EXPERTS,bv,pass == 0u ? 1u : 0u,chosen,weight);
				for ( pick = 0u; pick < TOP; pick++ )
				{
					Observe(check,(double)ev[row * TOP + pick],(double)chosen[pick],0.0);
					Observe(check,wv[row * TOP + pick],weight[pick],8.0 * FUNCTION_ERROR * weight[pick]);
				}
			}
		}
	snprintf(shape,sizeof(shape),"rows %u, 512 experts in 8 groups, 4 open, top 8",rows);
	Report("route_sigmoid_groups",shape,&clean,&corrupted,"experts exact, weights to 8*16 eps32 relative; the control routes without the groups");
	SparkDeviceFree(&logits);
	SparkDeviceFree(&bias);
	SparkDeviceFree(&experts);
	SparkDeviceFree(&weights);
}

static void ChannelDeltaStep(DeltaReference *ref, const uint16_t *mix, const uint16_t *decay_logits, const uint16_t *beta_logits, const float *a_log,
	const float *dt_bias, uint32_t row, double *out, double *tolerance)
{
	enum { H = 4, D = 128, W = H * D, MIX = 3 * W };
	const uint16_t *m = mix + (uint64_t)row * MIX;
	uint32_t head, i, v;
	for ( head = 0u; head < H; head++ )
	{
		double q[D], k[D], decay[D], qs = 0.0, ks = 0.0, largest = 0.0;
		for ( i = 0u; i < D; i++ )
		{
			q[i] = Bf16(m[head * D + i]);
			k[i] = Bf16(m[W + head * D + i]);
			qs += q[i] * q[i];
			ks += k[i] * k[i];
			decay[i] = exp(-5.0 * Sigmoid(exp((double)a_log[head]) * ((double)Bf16(decay_logits[(uint64_t)row * W + head * D + i]) + dt_bias[head * D + i])));
			if ( decay[i] > largest )
				largest = decay[i];
		}
		for ( i = 0u; i < D; i++ )
		{
			q[i] *= 1.0 / sqrt(qs + 1e-6) / sqrt((double)D);
			k[i] *= 1.0 / sqrt(ks + 1e-6);
		}
		const double beta = Sigmoid(Bf16(beta_logits[(uint64_t)row * H + head]));
		for ( v = 0u; v < D; v++ )
		{
			double *s = ref->state + ((uint64_t)head * D) * D + v;
			double predicted = 0.0, result = 0.0, magnitude = 0.0, qnorm = 0.0, smax = 0.0;
			const double value = Bf16(m[2u * W + head * D + v]);
			for ( i = 0u; i < D; i++ )
			{
				s[(uint64_t)i * D] *= decay[i];
				predicted += s[(uint64_t)i * D] * k[i];
			}
			for ( i = 0u; i < D; i++ )
				s[(uint64_t)i * D] += beta * (value - predicted) * k[i];
			for ( i = 0u; i < D; i++ )
			{
				result += s[(uint64_t)i * D] * q[i];
				magnitude += fabs(s[(uint64_t)i * D] * q[i]);
				qnorm += fabs(q[i]);
				if ( fabs(s[(uint64_t)i * D]) > smax )
					smax = fabs(s[(uint64_t)i * D]);
			}
			double *b = ref->bound + (uint64_t)head * D + v;
			*b = largest * *b + (4.0 * Gamma(D) + 64.0 * FUNCTION_ERROR) * (smax + fabs(value));
			out[head * D + v] = result;
			tolerance[head * D + v] = UNIT_BF16 * fabs(result) + (1.0 + UNIT_BF16) * (Gamma(D) * magnitude + *b * qnorm + 4.0 * FUNCTION_ERROR * magnitude);
		}
	}
}

static void TestChannelDelta(void)
{
	enum { H = 4, D = 128, W = H * D, MIX = 3 * W, FIRST = 200, SECOND = 3, ROWS = FIRST + SECOND };
	SparkDeviceBuffer state, index, begin, mix, decays, betas, alog, bias, scratch, out;
	float *s = (float *)Shared(&state,(uint64_t)2u * H * D * D * 4u,"channel delta state");
	uint32_t *slot = (uint32_t *)Shared(&index,4u,"channel delta slot");
	uint32_t *rb = (uint32_t *)Shared(&begin,8u,"channel delta begin");
	uint16_t *m = (uint16_t *)Shared(&mix,(uint64_t)ROWS * MIX * 2u,"channel delta qkv");
	uint16_t *f = (uint16_t *)Shared(&decays,(uint64_t)ROWS * W * 2u,"channel delta decay logits");
	uint16_t *bl = (uint16_t *)Shared(&betas,(uint64_t)ROWS * H * 2u,"channel delta beta logits");
	float *al = (float *)Shared(&alog,H * 4u,"channel delta a_log");
	float *db = (float *)Shared(&bias,W * 4u,"channel delta dt_bias");
	uint16_t *o = (uint16_t *)Shared(&out,(uint64_t)ROWS * W * 2u,"channel delta out");
	const float *prepared = (const float *)Shared(&scratch,SparkOpsChannelDeltaScratchBytes(ROWS,H,D),"channel delta scratch");
	DeltaReference ref;
	double *ref_out = (double *)malloc(W * sizeof(double)), *ref_tol = (double *)malloc(W * sizeof(double));
	SparkOpsSequences sequences;
	SparkDeltaParams params;
	Check clean, corrupted;
	uint32_t pass, row, i;
	FillBf16(m,(uint64_t)ROWS * MIX,1.0);
	FillBf16(f,(uint64_t)ROWS * W,2.0);
	FillBf16(bl,(uint64_t)ROWS * H,2.0);
	for ( i = 0u; i < H; i++ )
		al[i] = (float)(Normal() * 0.5);
	for ( i = 0u; i < W; i++ )
		db[i] = (float)(Normal() * 0.5);
	ref.state = (double *)malloc((size_t)H * D * D * sizeof(double));
	ref.bound = (double *)malloc((size_t)H * D * sizeof(double));
	memset(&params,0,sizeof(params));
	params.sequences = 1u;
	params.key_heads = params.value_heads = H;
	params.key_dimension = params.value_dimension = D;
	params.mix_stride = MIX;
	params.key_offset = W;
	params.value_offset = 2u * W;
	params.gate_stride = W;
	params.beta_stride = H;
	params.output_stride = W;
	params.commit = 1u;
	params.epsilon = 1e-6f;
	params.minimum_log_decay = -5.0f;
	sequences.state_index = Whole(&index);
	sequences.row_begin = Whole(&begin);
	sequences.sequences = 1u;
	sequences.commit = 1u;
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		float saved[D];
		memcpy(saved,db + 3u * D,sizeof(saved));
		memset(check,0,sizeof(*check));
		memset(s,0,(size_t)2u * H * D * D * 4u);
		memset(ref.state,0,(size_t)H * D * D * sizeof(double));
		memset(ref.bound,0,(size_t)H * D * sizeof(double));
		slot[0] = 1u;
		for ( i = 0u; pass == 1u && i < D; i++ )
			db[3u * D + i] = saved[i] - 8.0f;
		params.rows = FIRST;
		rb[0] = 0u;
		rb[1] = FIRST;
		if ( !Run(SparkOpsChannelDelta(&ops,queue,Whole(&state),&sequences,Whole(&mix),Whole(&decays),Whole(&betas),Whole(&alog),Whole(&bias),Whole(&scratch),
			Whole(&out),&params),"channel_delta first") )
			break;
		params.rows = SECOND;
		rb[0] = 0u;
		rb[1] = SECOND;
		if ( !Run(SparkOpsChannelDelta(&ops,queue,Whole(&state),&sequences,SparkTensorSlice(Whole(&mix),(uint64_t)FIRST * MIX * 2u,(uint64_t)SECOND * MIX * 2u),
			SparkTensorSlice(Whole(&decays),(uint64_t)FIRST * W * 2u,(uint64_t)SECOND * W * 2u),SparkTensorSlice(Whole(&betas),(uint64_t)FIRST * H * 2u,
			(uint64_t)SECOND * H * 2u),Whole(&alog),Whole(&bias),Whole(&scratch),SparkTensorSlice(Whole(&out),(uint64_t)FIRST * W * 2u,(uint64_t)SECOND * W * 2u),
			&params),"channel_delta second") )
			break;
		memcpy(db + 3u * D,saved,sizeof(saved));
		for ( row = 0u; row < ROWS; row++ )
		{
			ChannelDeltaStep(&ref,m,f,bl,al,db,row,ref_out,ref_tol);
			for ( i = 0u; i < W; i++ )
				Observe(check,Bf16(o[(uint64_t)row * W + i]),ref_out[i],ref_tol[i]);
		}
	}
	Report("channel_delta_rule","4 heads x 128, per-channel decay, 200 rows then 3",&clean,&corrupted,
		"the gated_delta_rule bound with B_t = max(decay) B_{t-1} + (4 gamma_128 + 64*16 eps32)(max|S| + |v|)");
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		const uint64_t poke = (uint64_t)FIRST * W + 2u * D + 77u;
		const uint16_t saved = f[poke];
		memset(check,0,sizeof(*check));
		if ( pass == 1u )
			f[poke] = ToBf16(Bf16(saved) + 3.0f);
		params.rows = SECOND;
		params.commit = sequences.commit = 0u;
		params.doubled_beta = 1u;
		rb[0] = 0u;
		rb[1] = SECOND;
		if ( !Run(SparkOpsChannelDelta(&ops,queue,Whole(&state),&sequences,SparkTensorSlice(Whole(&mix),(uint64_t)FIRST * MIX * 2u,(uint64_t)SECOND * MIX * 2u),
			SparkTensorSlice(Whole(&decays),(uint64_t)FIRST * W * 2u,(uint64_t)SECOND * W * 2u),SparkTensorSlice(Whole(&betas),(uint64_t)FIRST * H * 2u,
			(uint64_t)SECOND * H * 2u),Whole(&alog),Whole(&bias),Whole(&scratch),SparkTensorSlice(Whole(&out),(uint64_t)FIRST * W * 2u,(uint64_t)SECOND * W * 2u),
			&params),"channel_delta prepare") )
			break;
		f[poke] = saved;
		for ( row = 0u; row < SECOND; row++ )
		{
			for ( i = 0u; i < W; i++ )
			{
				const double expected = exp(-5.0 * Sigmoid(exp((double)al[i / D]) * ((double)Bf16(f[(uint64_t)(FIRST + row) * W + i]) + db[i])));
				Observe(check,prepared[(uint64_t)2u * SECOND * W + (uint64_t)row * W + i],expected,64.0 * FUNCTION_ERROR * expected);
			}
			for ( i = 0u; i < H; i++ )
			{
				const double sigmoid = Sigmoid(Bf16(bl[(uint64_t)(FIRST + row) * H + i]));
				Observe(check,prepared[(uint64_t)3u * SECOND * W + (uint64_t)row * H + i],2.0 * RoundBf16(sigmoid),2.0 * FlipSlack(sigmoid,4.0 * FUNCTION_ERROR));
			}
		}
	}
	Report("channel_delta_prepare","3 rows, 4 heads x 128, doubled beta",&clean,&corrupted,
		"decay exp(lower sigmoid(exp(a) (f + b))) to 64*16 eps32 relative; beta 2 bf16(sigmoid) to its flip slack within 4*16 eps32");
	free(ref.state);
	free(ref.bound);
	free(ref_out);
	free(ref_tol);
	SparkDeviceFree(&state);
	SparkDeviceFree(&index);
	SparkDeviceFree(&begin);
	SparkDeviceFree(&mix);
	SparkDeviceFree(&decays);
	SparkDeviceFree(&betas);
	SparkDeviceFree(&alog);
	SparkDeviceFree(&bias);
	SparkDeviceFree(&scratch);
	SparkDeviceFree(&out);
}

typedef struct AttentionFixture
{
	SparkDeviceBuffer pool, table, sequence, positions, keys, values, query, gate, partial, output, sinks;
	SparkOpsPaged paged;
	uint32_t context;
	uint32_t pages;
} AttentionFixture;

enum { ATT_SLOTS = 64, ATT_MAX_DIM = 512 };

typedef struct AttentionCase
{
	uint32_t heads;
	uint32_t kv;
	uint32_t dim;
	uint32_t qkv;
	uint32_t value_offset;
	uint32_t window;
	uint32_t gate;
	float scale;
	uint32_t separate;
	uint32_t vdim;
	uint32_t sinks;
} AttentionCase;

static AttentionCase att = { 24u, 4u, 256u, 14336u, 13312u, 0u, 1u, 1.0f / 16.0f, 0u, 0u, 0u };

static uint32_t AttVdim(void)
{
	return(att.vdim != 0u ? att.vdim : att.dim);
}

static void AttentionFill(AttentionFixture *f, uint32_t context, uint32_t sequences)
{
	SparkKvParams params;
	uint32_t *table, *sequence, *position, row, s, p;
	f->context = context;
	f->pages = (context + ATT_SLOTS - 1u) / ATT_SLOTS;
	Shared(&f->pool,(uint64_t)sequences * f->pages * ATT_SLOTS * att.kv * (att.dim + AttVdim()) * 2u,"attention pool");
	if ( att.sinks != 0u )
	{
		float *sink = (float *)Shared(&f->sinks,att.heads * sizeof(float),"attention sinks");
		for ( s = 0u; s < att.heads; s++ )
			sink[s] = (float)(2.0 * Normal());
	}
	table = (uint32_t *)Shared(&f->table,(uint64_t)sequences * f->pages * 4u,"attention table");
	for ( s = 0u; s < sequences; s++ )
		for ( p = 0u; p < f->pages; p++ )
			table[s * f->pages + p] = (sequences * f->pages - 1u) - (s * f->pages + p);
	sequence = (uint32_t *)Shared(&f->sequence,(uint64_t)sequences * context * 4u,"attention fill sequence");
	position = (uint32_t *)Shared(&f->positions,(uint64_t)sequences * context * 4u,"attention fill positions");
	FillBf16((uint16_t *)Shared(&f->keys,(uint64_t)sequences * context * att.kv * att.dim * 2u,"attention keys"),(uint64_t)sequences * context * att.kv * att.dim,1.0);
	FillBf16((uint16_t *)Shared(&f->values,(uint64_t)sequences * context * att.qkv * 2u,"attention values"),(uint64_t)sequences * context * att.qkv,1.0);
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
	f->paged.kv_heads = att.kv;
	memset(&params,0,sizeof(params));
	params.rows = sequences * context;
	params.kv_heads = att.kv;
	params.head_dimension = att.dim;
	params.page_slots = ATT_SLOTS;
	params.pages_per_sequence = f->pages;
	params.key_stride = att.kv * att.dim;
	params.value_stride = att.qkv;
	params.value_offset = att.value_offset;
	params.value_dimension = att.vdim;
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
	if ( f->sinks.bytes != 0u )
		SparkDeviceFree(&f->sinks);
}

static void AttentionReference(const AttentionFixture *f, const uint16_t *query, const uint16_t *gate, uint32_t sequence, uint32_t position,
	uint32_t head, double *out, double *tolerance)
{
	const uint16_t *keys = (const uint16_t *)f->keys.host, *values = (const uint16_t *)f->values.host;
	const uint32_t kv = head / (att.heads / att.kv), length = position + 1u, low = att.window != 0u && length > att.window ? length - att.window : 0u;
	double *score = (double *)malloc(length * sizeof(double)), *slack = (double *)malloc(length * sizeof(double)), best = -INFINITY, total = 0.0;
	const uint32_t vdim = AttVdim();
	const double sink = att.sinks != 0u ? ((const float *)f->sinks.host)[head] : -INFINITY;
	uint32_t j, d;
	for ( j = 0u; j < low; j++ )
		score[j] = slack[j] = 0.0;
	for ( j = low; j < length; j++ )
	{
		const uint16_t *k = keys + ((uint64_t)sequence * f->context + j) * att.kv * att.dim + kv * att.dim;
		double dot = 0.0, magnitude = 0.0;
		for ( d = 0u; d < att.dim; d++ )
		{
			dot += (double)Bf16(query[head * att.dim + d]) * Bf16(k[d]);
			magnitude += fabs((double)Bf16(query[head * att.dim + d]) * Bf16(k[d]));
		}
		score[j] = dot * att.scale;
		slack[j] = (Gamma(att.dim + 2.0) * magnitude * att.scale + 2.0 * FUNCTION_ERROR * (1.0 + fabs(score[j])));
		if ( score[j] > best )
			best = score[j];
	}
	if ( sink > best )
		best = sink;
	for ( j = low; j < length; j++ )
	{
		score[j] = exp(score[j] - best);
		total += score[j];
	}
	if ( att.sinks != 0u )
		total += exp(sink - best);
	for ( d = 0u; d < vdim; d++ )
	{
		double value = 0.0, spread = 0.0, magnitude = 0.0;
		for ( j = low; j < length; j++ )
		{
			const double v = Bf16(values[((uint64_t)sequence * f->context + j) * att.qkv + att.value_offset + kv * vdim + d]);
			value += score[j] / total * v;
			magnitude += score[j] / total * fabs(v);
			spread += score[j] / total * fabs(v) * 2.0 * slack[j];
		}
		const double sigmoid = att.gate != 0u ? Sigmoid(Bf16(gate[att.separate != 0u ? head * att.dim + d : head * 2u * att.dim + att.dim + d])) : 1.0;
		const double attended = RoundBf16(value), g = att.gate == SPARK_ATTENTION_GATE_SIGMOID_BF16 ? RoundBf16(sigmoid) : sigmoid;
		const double gate_slack = att.gate == SPARK_ATTENTION_GATE_SIGMOID_BF16 ? FlipSlack(sigmoid,2.0 * FUNCTION_ERROR) : 0.0;
		const double absolute = 2.0 * spread + Gamma(length + 8.0) * magnitude + 4.0 * FUNCTION_ERROR * fabs(value);
		const double slack = fabs(RoundBf16(value - absolute) - attended) > fabs(RoundBf16(value + absolute) - attended) ?
			fabs(RoundBf16(value - absolute) - attended) : fabs(RoundBf16(value + absolute) - attended);
		out[d] = attended * g;
		tolerance[d] = (att.gate != 0u ? UNIT_BF16 * fabs(out[d]) : 0.0) + g * slack + fabs(attended) * gate_slack + 4.0 * FUNCTION_ERROR * fabs(out[d]);
	}
	free(score);
	free(slack);
}

static SparkAttentionParams AttentionParamsFor(const AttentionFixture *f, uint32_t rows, uint32_t sequences)
{
	SparkAttentionParams params;
	memset(&params,0,sizeof(params));
	params.rows = rows;
	params.heads = att.heads;
	params.kv_heads = att.kv;
	params.head_dimension = att.dim;
	params.page_slots = ATT_SLOTS;
	params.pages_per_sequence = f->pages;
	params.pool_pages = sequences * f->pages;
	params.query_stride = att.heads * att.dim;
	params.output_stride = att.heads * AttVdim();
	params.scale = att.scale;
	params.value_dimension = att.vdim;
	params.has_sinks = att.sinks;
	params.output_gate = att.gate;
	params.window = att.window;
	params.gate_stride = att.separate != 0u ? att.heads * att.dim : att.qkv;
	params.gate_offset = att.separate != 0u ? 0u : att.dim;
	params.gate_head_stride = (att.separate != 0u ? 1u : 2u) * att.dim;
	return(params);
}

static void TestAttention(uint32_t context, uint32_t rows, uint32_t prefill)
{
	AttentionFixture f;
	SparkAttentionParams params;
	uint16_t *query, *gate, *output;
	uint32_t *sequence, *position, row, head, d, pass;
	double ref[ATT_MAX_DIM], tolerance[ATT_MAX_DIM];
	Check clean, corrupted;
	char shape[128], window[24], values_text[32];
	const uint32_t sequences = prefill != 0u ? 2u : 1u;
	memset(&f,0,sizeof(f));
	AttentionFill(&f,context,sequences);
	query = (uint16_t *)Shared(&f.query,(uint64_t)rows * att.heads * att.dim * 2u,"attention query");
	gate = (uint16_t *)Shared(&f.gate,(uint64_t)rows * att.qkv * 2u,"attention gate");
	output = (uint16_t *)Shared(&f.output,(uint64_t)rows * att.heads * AttVdim() * 2u,"attention output");
	Shared(&f.partial,SparkOpsAttentionScratchBytes(rows,att.heads,AttVdim(),context),"attention partial");
	const SparkTensor sinks = Whole(&f.sinks);
	FillBf16(query,(uint64_t)rows * att.heads * att.dim,att.scale < 0.5f ? 1.0 : 1.0 / 16.0);
	FillBf16(gate,(uint64_t)rows * att.qkv,1.0);
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
	params = AttentionParamsFor(&f,rows,sequences);
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		uint16_t *keys = (uint16_t *)f.keys.host;
		const uint32_t poke_position = att.window != 0u && context > att.window ? context - 1u - att.window / 2u : (context - 1u) / 2u;
		const uint32_t poked_rows = context > 1u ? 2u : 1u, value_position = poke_position + poked_rows - 1u;
		const uint64_t poke = ((uint64_t)(sequences - 1u) * f.context + poke_position) * att.kv * att.dim + (att.kv - 1u) * att.dim + 17u;
		const uint16_t saved = keys[poke];
		const float probe = Bf16(query[((uint64_t)(prefill != 0u ? rows / 2u : 0u) * att.heads + (att.heads / att.kv) * (att.kv - 1u)) * att.dim + 17u]);
		SparkKvParams store;
		memset(check,0,sizeof(*check));
		uint16_t *values = (uint16_t *)f.values.host;
		const uint64_t value_poke = ((uint64_t)(sequences - 1u) * f.context + value_position) * att.qkv + att.value_offset + (att.kv - 1u) * AttVdim() + 17u;
		const uint16_t value_saved = values[value_poke];
		if ( pass == 1u )
		{
			keys[poke] = ToBf16(Bf16(saved) + (probe < 0.0f ? -64.0f : 64.0f));
			values[value_poke] = ToBf16(Bf16(value_saved) + 4.0f);
			memset(&store,0,sizeof(store));
			store.rows = poked_rows;
			store.kv_heads = att.kv;
			store.head_dimension = att.dim;
			store.page_slots = ATT_SLOTS;
			store.pages_per_sequence = f.pages;
			store.key_stride = att.kv * att.dim;
			store.value_stride = att.qkv;
			store.value_offset = att.value_offset;
			store.value_dimension = att.vdim;
			store.pool_pages = sequences * f.pages;
			{
				SparkDeviceBuffer one_sequence, one_position;
				SparkOpsPaged single = f.paged;
				const uint64_t first_row = (uint64_t)(sequences - 1u) * f.context + poke_position;
				uint32_t *poke_sequences = (uint32_t *)Shared(&one_sequence,2u * 4u,"poke sequence");
				uint32_t *poke_positions = (uint32_t *)Shared(&one_position,2u * 4u,"poke position");
				poke_sequences[0] = poke_sequences[1] = sequences - 1u;
				poke_positions[0] = poke_position;
				poke_positions[1] = poke_position + 1u;
				single.sequence_of_row = Whole(&one_sequence);
				single.positions = Whole(&one_position);
				Run(SparkOpsKvStore(&ops,queue,&single,SparkTensorSlice(Whole(&f.keys),first_row * att.kv * att.dim * 2u,(uint64_t)poked_rows * att.kv * att.dim * 2u),
					SparkTensorSlice(Whole(&f.values),first_row * att.qkv * 2u,(uint64_t)poked_rows * att.qkv * 2u),&store),"kv_store poke");
				SparkDeviceFree(&one_sequence);
				SparkDeviceFree(&one_position);
			}
			keys[poke] = saved;
			values[value_poke] = value_saved;
		}
		if ( !Run(SparkOpsAttention(&ops,queue,&f.paged,Whole(&f.query),Whole(&f.gate),att.sinks != 0u ? &sinks : 0,Whole(&f.partial),Whole(&f.output),&params,position,sequence),
			"attention") )
			break;
		for ( row = 0u; row < rows; row++ )
			for ( head = 0u; head < att.heads; head++ )
			{
				AttentionReference(&f,query + (uint64_t)row * att.heads * att.dim,gate + (uint64_t)row * params.gate_stride,sequence[row],position[row],head,ref,
					tolerance);
				for ( d = 0u; d < AttVdim(); d++ )
					Observe(check,Bf16(output[((uint64_t)row * att.heads + head) * AttVdim() + d]),ref[d],tolerance[d]);
			}
	}
	window[0] = 0;
	if ( att.window != 0u )
		snprintf(window,sizeof(window),", window %u",att.window);
	snprintf(values_text,sizeof(values_text),", values %u%s",att.vdim,att.sinks != 0u ? ", sinks" : "");
	snprintf(shape,sizeof(shape),"%s rows %u, context %u, %u q / %u kv heads x %u%s%s",prefill != 0u ? "prefill" : "decode",rows,context,att.heads,att.kv,
		att.dim,window,att.vdim != 0u ? values_text : att.gate == SPARK_ATTENTION_GATE_SIGMOID_BF16 ? ", bf16 gate" :
		att.gate != 0u ? "" : ", no gate");
	Report(prefill != 0u ? "attention_prefill" : "attention_decode",shape,&clean,&corrupted,
		"u|y| + sigmoid(g) bf16 flip slack of o within e + |o| flip slack of a bf16 gate + 4*16 eps32 |y|; e = 2 sum p|v| 2 ds + gamma_L sum p|v|, ds = gamma_258 sum|q k|/16 + 2*16 eps32 (1+|s|)");
	AttentionFree(&f);
	SparkDeviceFree(&f.query);
	SparkDeviceFree(&f.gate);
	SparkDeviceFree(&f.partial);
	SparkDeviceFree(&f.output);
}

static int AttentionPagesRun(AttentionFixture *f, uint32_t rows, const uint32_t *position, const uint32_t *sequence, uint32_t sequences,
	uint32_t *error)
{
	const SparkAttentionParams params = AttentionParamsFor(f,rows,sequences);
	SparkDeviceOpsTakeError(&ops,error);
	if ( !Run(SparkOpsAttention(&ops,queue,&f->paged,Whole(&f->query),Whole(&f->gate),0,Whole(&f->partial),Whole(&f->output),&params,position,sequence),
		"attention pages") )
		return(0);
	SparkDeviceOpsTakeError(&ops,error);
	return(1);
}

static void AttentionPagesExpect(const char *what, int ran, uint32_t error, uint32_t first, uint32_t last)
{
	if ( ran && error != 0u && error - 1u >= first && error - 1u <= last )
		printf("PASS attention_pages        %s raises the device error word for row %u\n",what,error - 1u);
	else
	{
		printf("FAIL attention_pages        %s: error word %u, expected a row in %u..%u\n",what,error,first,last);
		failures++;
	}
}

static void TestAttentionPages(void)
{
	enum { CONTEXT = 300, SEQUENCES = 2, PREFILL_ROWS = 64, BAD_PAGE = 2 };
	AttentionFixture f;
	SparkKvParams store;
	SparkDeviceBuffer one_sequence, one_position, decode_sequence, decode_position;
	uint32_t *sequence, *position, *table, row, error, saved;
	uint32_t *decode_rows[2];
	int ran;
	memset(&f,0,sizeof(f));
	AttentionFill(&f,CONTEXT,SEQUENCES);
	Shared(&f.query,(uint64_t)PREFILL_ROWS * att.heads * att.dim * 2u,"pages query");
	Shared(&f.gate,(uint64_t)PREFILL_ROWS * att.qkv * 2u,"pages gate");
	Shared(&f.output,(uint64_t)PREFILL_ROWS * att.heads * att.dim * 2u,"pages output");
	Shared(&f.partial,SparkOpsAttentionScratchBytes(PREFILL_ROWS,att.heads,att.dim,CONTEXT),"pages partial");
	FillBf16((uint16_t *)f.query.host,(uint64_t)PREFILL_ROWS * att.heads * att.dim,att.scale < 0.5f ? 1.0 : 1.0 / 16.0);
	FillBf16((uint16_t *)f.gate.host,(uint64_t)PREFILL_ROWS * att.qkv,1.0);
	SparkDeviceFree(&f.sequence);
	SparkDeviceFree(&f.positions);
	sequence = (uint32_t *)Shared(&f.sequence,PREFILL_ROWS * 4u,"pages sequence");
	position = (uint32_t *)Shared(&f.positions,PREFILL_ROWS * 4u,"pages positions");
	for ( row = 0u; row < PREFILL_ROWS; row++ )
	{
		sequence[row] = row < PREFILL_ROWS / 2u ? 0u : 1u;
		position[row] = CONTEXT - PREFILL_ROWS / 2u + row % (PREFILL_ROWS / 2u);
	}
	decode_rows[0] = (uint32_t *)Shared(&decode_sequence,SEQUENCES * 4u,"pages decode sequence");
	decode_rows[1] = (uint32_t *)Shared(&decode_position,SEQUENCES * 4u,"pages decode positions");
	for ( row = 0u; row < SEQUENCES; row++ )
	{
		decode_rows[0][row] = row;
		decode_rows[1][row] = CONTEXT - 1u;
	}
	table = (uint32_t *)f.table.host;
	f.paged.sequence_of_row = Whole(&f.sequence);
	f.paged.positions = Whole(&f.positions);
	ran = AttentionPagesRun(&f,PREFILL_ROWS,position,sequence,SEQUENCES,&error);
	if ( !ran || error != 0u )
	{
		printf("FAIL attention_pages        mapped pages raised the device error word (%u)\n",error);
		failures++;
	}
	saved = table[(SEQUENCES - 1u) * f.pages + BAD_PAGE];
	table[(SEQUENCES - 1u) * f.pages + BAD_PAGE] = SEQUENCES * f.pages;
	ran = AttentionPagesRun(&f,PREFILL_ROWS,position,sequence,SEQUENCES,&error);
	AttentionPagesExpect("prefill over an unmapped page",ran,error,PREFILL_ROWS / 2u,PREFILL_ROWS - 1u);
	f.paged.sequence_of_row = Whole(&decode_sequence);
	f.paged.positions = Whole(&decode_position);
	ran = AttentionPagesRun(&f,SEQUENCES,decode_rows[1],decode_rows[0],SEQUENCES,&error);
	AttentionPagesExpect("decode over an unmapped page",ran,error,SEQUENCES - 1u,SEQUENCES - 1u);
	memset(&store,0,sizeof(store));
	store.rows = 1u;
	store.kv_heads = att.kv;
	store.head_dimension = att.dim;
	store.page_slots = ATT_SLOTS;
	store.pages_per_sequence = f.pages;
	store.key_stride = att.kv * att.dim;
	store.value_stride = att.qkv;
	store.value_offset = att.value_offset;
	store.pool_pages = SEQUENCES * f.pages;
	{
		SparkOpsPaged single = f.paged;
		*(uint32_t *)Shared(&one_sequence,4u,"pages store sequence") = SEQUENCES - 1u;
		*(uint32_t *)Shared(&one_position,4u,"pages store position") = BAD_PAGE * ATT_SLOTS + 5u;
		single.sequence_of_row = Whole(&one_sequence);
		single.positions = Whole(&one_position);
		SparkDeviceOpsTakeError(&ops,&error);
		ran = Run(SparkOpsKvStore(&ops,queue,&single,SparkTensorSlice(Whole(&f.keys),0u,att.kv * att.dim * 2u),
			SparkTensorSlice(Whole(&f.values),0u,att.qkv * 2u),&store),"kv_store unmapped page");
		SparkDeviceOpsTakeError(&ops,&error);
		AttentionPagesExpect("a KV store into an unmapped page",ran,error,0u,0u);
		SparkDeviceFree(&one_sequence);
		SparkDeviceFree(&one_position);
	}
	table[(SEQUENCES - 1u) * f.pages + BAD_PAGE] = saved;
	ran = AttentionPagesRun(&f,SEQUENCES,decode_rows[1],decode_rows[0],SEQUENCES,&error);
	if ( !ran || error != 0u )
	{
		printf("FAIL attention_pages        the restored page table still raised the device error word (%u)\n",error);
		failures++;
	}
	else
		printf("PASS attention_pages        mapped pages raise nothing before and after the unmapped-page controls\n");
	decode_rows[1][0] = f.pages * ATT_SLOTS;
	{
		const SparkAttentionParams params = AttentionParamsFor(&f,SEQUENCES,SEQUENCES);
		const SparkStatus status = SparkOpsAttention(&ops,queue,&f.paged,Whole(&f.query),Whole(&f.gate),0,Whole(&f.partial),Whole(&f.output),&params,
			decode_rows[1],decode_rows[0]);
		if ( status == SPARK_STATUS_OK )
		{
			printf("FAIL attention_pages        a position past the sequence's pages was not refused\n");
			failures++;
			SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER);
		}
		else
			printf("PASS attention_pages        a position past the sequence's pages is refused before launch\n");
	}
	AttentionFree(&f);
	SparkDeviceFree(&f.query);
	SparkDeviceFree(&f.gate);
	SparkDeviceFree(&f.partial);
	SparkDeviceFree(&f.output);
	SparkDeviceFree(&decode_sequence);
	SparkDeviceFree(&decode_position);
}

static void TestSiluMul(uint32_t rows, double limit)
{
	const uint32_t activation = isfinite(limit) ? SPARK_GATED_SILU_CLAMP : SPARK_GATED_SILU;
	enum { DIM = 17408 };
	SparkDeviceBuffer input, output;
	uint16_t *in = (uint16_t *)Shared(&input,(uint64_t)rows * 2u * DIM * 2u,"silu input");
	uint16_t *out = (uint16_t *)Shared(&output,(uint64_t)rows * DIM * 2u,"silu output");
	SparkSiluMulParams params;
	Check clean, corrupted;
	char shape[64];
	uint32_t pass, row, i;
	FillBf16(in,(uint64_t)rows * 2u * DIM,isfinite(limit) ? 3.0 : 2.0);
	memset(&params,0,sizeof(params));
	params.rows = rows;
	params.dimension = DIM;
	params.input_stride = 2u * DIM;
	params.output_stride = DIM;
	params.limit = (float)limit;
	for ( pass = 0u; pass < 2u; pass++ )
	{
		Check *check = pass == 0u ? &clean : &corrupted;
		const uint64_t poke = (uint64_t)(rows - 1u) * 2u * DIM + DIM + 4321u;
		const uint16_t saved = in[poke];
		memset(check,0,sizeof(*check));
		if ( pass == 1u )
			in[poke] = ToBf16(Bf16(saved) * -1.0f + 1.0f);
		if ( !Run(SparkOpsGatedMul(&ops,queue,Whole(&input),Whole(&output),&params,activation),"silu_mul") )
			break;
		in[poke] = saved;
		for ( row = 0u; row < rows; row++ )
			for ( i = 0u; i < DIM; i++ )
			{
				const double y = fmin(Silu(Bf16(in[(uint64_t)row * 2u * DIM + i])),limit) * fmin(fmax(Bf16(in[(uint64_t)row * 2u * DIM + DIM + i]),-limit),limit);
				Observe(check,Bf16(out[(uint64_t)row * DIM + i]),y,UNIT_BF16 * fabs(y) + 3.0 * FUNCTION_ERROR * fabs(y));
			}
	}
	snprintf(shape,sizeof(shape),"rows %u, 17408 gate | 17408 up%s",rows,isfinite(limit) ? ", limit 5" : "");
	Report(isfinite(limit) ? "silu_clamp_mul" : "silu_mul",shape,&clean,&corrupted,"u|y| + 3*16 eps32 |y|");
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
	TestGatherSlice();
	TestSegmentedMap();
	TestRankKernels(8u,3u,2816u);
	TestRankKernels(2u,17u,1024u);
	TestRankKernels(8u,2u,65536u);
	TestRouteLocal();
	TestHeadGate(3u,10u,SPARK_HEAD_GATE_SOFTPLUS);
	TestHeadGate(17u,5u,SPARK_HEAD_GATE_SOFTPLUS);
	TestHeadGate(5u,4u,SPARK_HEAD_GATE_SIGMOID_BF16);
	TestLinearRoutedSkip(3u);
	TestRmsNormResidual(1u);
	TestRmsNormResidual(64u);
	TestRmsNormGate(1u,0u);
	TestRmsNormGate(33u,0u);
	TestRmsNormGate(3u,1u);
	TestHeadRope(1u,24u,0u,512u,"24 query heads in the fused row");
	TestHeadRope(17u,4u,12288u,256u,"4 key heads in the fused row");
	{
		static const RopeCase sliding = { 256u, 256u, 128u, 10000.0, 1u, 1u, 0.0, 0.0, 0u, 0u, 0u }, full = { 512u, 512u, 64u, 1000000.0, 1u, 1u, 0.0, 0.0, 0u, 0u, 0u },
			value = { 512u, 0u, 0u, 1.0, 0u, 0u, 0.0, 0.0, 0u, 0u, 0u }, yarn = { 128u, 64u, 32u, 500000.0, 1u, 1u, 1.4852030263919618, 0.0, 0u, 0u, 0u },
			plain = { 128u, 64u, 32u, 500000.0, 1u, 0u, 1.4852030263919618, 0.0, 0u, 0u, 0u };
		TestHeadRopeCase(3u,16u,0u,256u,&sliding,"16 heads x 256, whole-head rope with bf16 terms, theta 1e4");
		TestHeadRopeCase(2u,2u,8192u,512u,&full,"2 heads x 512, 64 of 256 pairs rotated, bf16 terms, theta 1e6");
		TestHeadRopeCase(3u,10u,0u,128u,&yarn,"10 heads x 128, 64 dims rotated, bf16 terms of cos and sin times 1.485");
		TestHeadRopeCase(3u,10u,0u,128u,&plain,"10 heads x 128, 64 dims rotated in FP32, cos and sin times 1.485");
		TestHeadRopeCase(2u,2u,8192u,512u,&value,"2 heads x 512, weightless norm, no rope");
		{
			static const RopeCase latent = { 64u, 64u, 32u, 6000000.0, 0u, 0u, 0.0, 0.0, 1u, 576u, 1u };
			TestHeadRopeCase(3u,8u,128u,192u,&latent,"8 heads, 64 interleaved dims of 192-wide heads into 576-wide rows, plain, theta 6e6");
		}
	}
	{
		static const RopeCase sliding = { 128u, 128u, 64u, 500000.0, 0u, 1u, 0.0, 3.87, 0u, 0u, 0u }, full = { 128u, 0u, 0u, 1.0, 0u, 0u, 0.0, 3.87, 0u, 0u, 0u };
		TestHeadRopeCase(3u,32u,0u,128u,&sliding,"32 heads x 128, weightless, times 3.87, bf16-term rope, theta 5e5");
		TestHeadRopeCase(2u,32u,0u,128u,&full,"32 heads x 128, weightless, times 3.87, no rope");
	}
	TestRmsNormWeightless(1u);
	TestRmsNormWeightless(9u);
	TestLinear(1u,5120u,16384u,0u,SPARK_PROJECTION_GEMV,UINT64_MAX);
	TestLinear(1u,5120u,96u,0u,SPARK_PROJECTION_GEMV,UINT64_MAX);
	TestLinear(1u,5120u,14336u,0u,SPARK_PROJECTION_GEMV,UINT64_MAX);
	TestLinear(1u,6144u,5120u,0u,SPARK_PROJECTION_GEMV,UINT64_MAX);
	TestLinear(1u,5120u,34816u,0u,SPARK_PROJECTION_GEMV,UINT64_MAX);
	TestLinear(3u,17408u,5120u,0u,SPARK_PROJECTION_GEMV,UINT64_MAX);
	TestLinear(7u,5120u,1000u,0u,SPARK_PROJECTION_GEMV,UINT64_MAX);
	TestLinear(8u,5120u,1000u,0u,SPARK_PROJECTION_GEMV,UINT64_MAX);
	TestLinear(2u,6144u,5120u,0u,SPARK_PROJECTION_GEMV,UINT64_MAX);
	TestLinear(4u,17408u,5120u,0u,SPARK_PROJECTION_GEMV,UINT64_MAX);
	TestLinear(5u,17408u,5120u,0u,SPARK_PROJECTION_GEMV,UINT64_MAX);
	TestLinear(16u,5120u,96u,0u,SPARK_PROJECTION_GEMV,UINT64_MAX);
	TestLinear(17u,6144u,5120u,0u,SPARK_PROJECTION_GEMV,UINT64_MAX);
	TestLinear(64u,5120u,34816u,0u,SPARK_PROJECTION_GEMV,20000u);
	TestLinear(2u,5120u,248320u,1u,SPARK_PROJECTION_GEMV,UINT64_MAX);
	TestLinear(1u,5120u,248320u,1u,SPARK_PROJECTION_GEMV,UINT64_MAX);
	TestLinear(130u,5120u,200u,0u,SPARK_PROJECTION_GEMV,UINT64_MAX);
	TestLinear(512u,5120u,16384u,0u,SPARK_PROJECTION_GEMV,20000u);
	TestLinear(512u,5120u,96u,0u,SPARK_PROJECTION_GEMV,UINT64_MAX);
	TestLinear(512u,6144u,5120u,0u,SPARK_PROJECTION_GEMV,20000u);
	TestLinear(512u,17408u,5120u,0u,SPARK_PROJECTION_GEMV,20000u);
	TestLinear(77u,17408u,5120u,0u,SPARK_PROJECTION_GEMV,20000u);
	TestLinear(2048u,5120u,34816u,0u,SPARK_PROJECTION_GEMV,20000u);
	TestLinear(64u,5120u,248320u,1u,SPARK_PROJECTION_GEMV,20000u);
	{
		static const uint32_t shapes[][2] = { { 2816u, 4224u }, { 2112u, 2816u }, { 2816u, 8192u }, { 4096u, 2816u }, { 8192u, 2816u },
			{ 2816u, 1024u }, { 2816u, 128u }, { 704u, 2816u } }, row_counts[] = { 1u, 4u, 9u, 64u, 512u };
		uint32_t s, c;
		for ( s = 0u; s < sizeof(shapes) / sizeof(shapes[0]); s++ )
			for ( c = 0u; c < sizeof(row_counts) / sizeof(row_counts[0]); c++ )
				TestLinear(row_counts[c],shapes[s][0],shapes[s][1],0u,SPARK_PROJECTION_GEMV,20000u);
		TestLinear(1u,2816u,262144u,1u,SPARK_PROJECTION_GEMV,20000u);
		TestLinear(64u,2816u,262144u,1u,SPARK_PROJECTION_GEMV,20000u);
	}
	TestLinearRowIndependence(3u,64u,5120u,34816u,0u,SPARK_PROJECTION_GEMV,0u);
	TestLinearRowIndependence(2u,1u,5120u,248320u,1u,SPARK_PROJECTION_GEMV,0u);
	TestLinearRowIndependence(64u,5u,17408u,5120u,0u,SPARK_PROJECTION_GEMV,0u);
	TestLinearRowIndependence(16u,9u,5120u,96u,0u,SPARK_PROJECTION_GEMV,0u);
	TestLinearRowIndependence(32u,17u,5120u,16384u,0u,SPARK_PROJECTION_GEMV,0u);
	TestLinearRowIndependence(32u,16u,6144u,5120u,0u,SPARK_PROJECTION_GEMV,0u);
	TestLinearRowIndependence(40u,8u,5120u,248320u,1u,SPARK_PROJECTION_GEMV,0u);
	TestLinearRowIndependence(23u,1u,5120u,248320u,1u,SPARK_PROJECTION_GEMV,1u);
	TestLinearRowIndependence(13u,1u,5120u,34816u,0u,SPARK_PROJECTION_GEMV,1u);
	TestLinear(1u,5120u,34816u,0u,SPARK_PROJECTION_ROW_TILES,UINT64_MAX);
	TestLinear(2u,6144u,5120u,0u,SPARK_PROJECTION_ROW_TILES,UINT64_MAX);
	TestLinear(1u,5120u,248320u,1u,SPARK_PROJECTION_ROW_TILES,20000u);
	TestLinearRowIndependence(64u,1u,17408u,5120u,0u,SPARK_PROJECTION_ROW_TILES,1u);
	TestLinearRowIndependence(32u,1u,5120u,248320u,1u,SPARK_PROJECTION_ROW_TILES,1u);
	TestLinearRowIndependence(13u,2u,6144u,5120u,0u,SPARK_PROJECTION_ROW_TILES,0u);
	TestLinearFp8(1u,5120u,1000u,0u,UINT64_MAX);
	TestLinearFp8(3u,17408u,512u,0u,20000u);
	TestLinearFp8(8u,5120u,2048u,0u,20000u);
	TestLinearFp8(16u,5120u,1000u,0u,20000u);
	TestLinearFp8(64u,5120u,1000u,1u,20000u);
	TestLinearFp8(17u,5120u,512u,0u,20000u);
	TestLinearFp8(130u,5120u,512u,0u,20000u);
	TestCausalConv();
	TestCausalConvDilated();
	TestGatedDelta();
	TestChannelDelta();
	TestAttention(1u,1u,0u);
	TestAttention(1000u,1u,0u);
	TestAttention(8192u,4u,0u);
	TestAttention(300u,40u,1u);
	TestAttention(2048u,64u,1u);
	{
		const AttentionCase qwen = att, sliding = { 16u, 8u, 256u, 8192u, 6144u, 1024u, 0u, 1.0f, 0u, 0u, 0u }, full = { 16u, 2u, 512u, 10240u, 9216u, 0u, 0u, 1.0f, 0u, 0u, 0u },
			muse_sliding = { 32u, 2u, 128u, 4608u, 4352u, 2048u, SPARK_ATTENTION_GATE_SIGMOID_BF16, 0.08838834764831845f, 1u, 0u, 0u },
			muse_full = { 32u, 2u, 128u, 4608u, 4352u, 0u, SPARK_ATTENTION_GATE_SIGMOID_BF16, 0.08838834764831845f, 1u, 0u, 0u },
			mimo_sliding = { 16u, 4u, 192u, 512u, 0u, 128u, 0u, 0.07216878364870322f, 0u, 128u, 1u },
			mimo_full = { 16u, 4u, 192u, 512u, 0u, 0u, 0u, 0.07216878364870322f, 0u, 128u, 0u },
			latent = { 8u, 1u, 576u, 512u, 0u, 0u, 0u, 0.07216878364870322f, 0u, 512u, 0u };
		att = sliding;
		TestAttention(2048u,1u,0u);
		TestAttention(2048u,64u,1u);
		att = full;
		TestAttention(1000u,1u,0u);
		TestAttention(300u,40u,1u);
		att = muse_sliding;
		TestAttention(3000u,1u,0u);
		TestAttention(300u,40u,1u);
		att = muse_full;
		TestAttention(1000u,4u,0u);
		TestAttention(2048u,64u,1u);
		att = mimo_sliding;
		TestAttention(1000u,2u,0u);
		TestAttention(300u,40u,1u);
		att = mimo_full;
		TestAttention(1000u,4u,0u);
		TestAttention(300u,40u,1u);
		att = latent;
		TestAttention(1000u,2u,0u);
		TestAttention(300u,40u,1u);
		att = qwen;
	}
	TestAttentionPages();
	TestGeluMul(2u);
	TestGatedMulRouted(24u,SPARK_GATED_GELU_TANH);
	TestGatedMulRouted(24u,SPARK_GATED_SILU);
	TestGatedMulRouted(24u,SPARK_GATED_SILU_CLAMP);
	TestScaleAndSoftcap();
	TestRoutes(3u);
	TestRoutesPinned(3u);
	TestRoutesNonFinite();
	TestLinearRouted(2u);
	TestLinearRoutedMxfp4(2u,4096u,512u);
	TestMimoKernels(3u);
	TestHyperKernels(3u);
	TestSharedExpert(3u);
	TestRouteGroups(5u);
	TestNgram();
	TestLinearRoutedNvfp4(2u,2560u,96u);
	TestLinearRoutedNvfp4(2u,640u,160u);
	TestLinearRoutedNvfp4(1u,256u,64u);
	TestLinearRoutedMxfp4(1u,2048u,1024u);
	TestBranchNorms(2u,1u);
	TestBranchNorms(3u,2u);
	TestSiluMul(1u,INFINITY);
	TestSiluMul(10u,INFINITY);
	TestSiluMul(3u,5.0);
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
