#include "sparkpipe/spark_clock.h"
#include "sparkpipe/spark_drafter_module.h"
#include "sparkpipe/spark_gpu.h"
#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_device.h"
#include "sparkpipe/spark_weightd_lazy_pack.h"
#include "inference/kernels/dense_rows.cuh"
#include "inference/kernels/norm.cuh"
#include "runtime/launch.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define K31_HIDDEN 7168u
#define K31_TAPS 5u
#define K31_TAP_WIDTH (K31_TAPS * K31_HIDDEN)
#define K31_LAYERS 5u
#define K31_KV_RANK 512u
#define K31_ROPE 64u
#define K31_LATENT (K31_KV_RANK + K31_ROPE)
#define K31_PAGE_POSITIONS 4096u
#define K31_PAGE_BYTES ((uint64_t)K31_LAYERS * K31_PAGE_POSITIONS * K31_LATENT * sizeof(uint16_t))
#define K31_PAGES_MAX 4096u
#define K31_SEQUENCES_MAX 64u
#define K31_INGEST_ROWS 256u
#define K31_THREADS 256u
#define K31_PACK_MAGIC 0x4B505244u
#define K31_PACK_VERSION 1u
#define K31_PACK_ALIGNMENT 128u
#define K31_PI 3.14159265358979323846

typedef struct K31Layer
{
	const uint16_t *kv_a;
	const uint16_t *kv_a_norm;
} K31Layer;

typedef struct K31Sequence
{
	uint64_t id;
	uint64_t held_end;
	uint64_t last_used;
	uint32_t page_count;
	uint16_t **pages;
} K31Sequence;

typedef struct K31State
{
	SparkWeightdLazyPack *pack;
	SparkGpuStream stream;
	uint32_t multiprocessors;
	float epsilon;
	float rope_gain;
	float *rope_inverse;
	const uint16_t *context_proj;
	const uint16_t *context_norm;
	K31Layer layers[K31_LAYERS];
	uint16_t *host_taps;
	uint16_t *taps;
	uint16_t *features_raw;
	uint16_t *features;
	uint16_t *latent_raw;
	uint32_t *positions;
	uint32_t *host_positions;
	uint64_t context_bytes;
	uint64_t pages_used;
	uint64_t clock;
	K31Sequence sequences[K31_SEQUENCES_MAX];
	uint64_t frames;
	uint64_t rows;
} K31State;

__global__ __launch_bounds__(K31_THREADS, 1)
void K31RmsRowsKernel(const uint16_t *__restrict__ input, const uint16_t *__restrict__ weight, uint16_t *__restrict__ output, uint32_t dimension, float epsilon)
{
	__shared__ float reduction[K31_THREADS / 32u];
	const uint16_t *row = input + (uint64_t)blockIdx.x * dimension;
	uint16_t *out = output + (uint64_t)blockIdx.x * dimension;
	float total = 0.0f, scale;
	for ( uint32_t index = threadIdx.x; index < dimension; index += K31_THREADS )
	{
		const float value = LmBf16ToFloat(row[index]);
		total += value * value;
	}
	total = LmBlockSum<K31_THREADS>(total, reduction);
	scale = rsqrtf(total / (float)dimension + epsilon);
	for ( uint32_t index = threadIdx.x; index < dimension; index += K31_THREADS )
		out[index] = LmFloatToBf16(LmBf16ToFloat(row[index]) * scale * LmBf16ToFloat(weight[index]));
}

__global__ __launch_bounds__(K31_THREADS, 1)
void K31LatentRowsKernel(const uint16_t *__restrict__ latent, const uint16_t *__restrict__ norm, const float *__restrict__ inverse, float gain,
	const uint32_t *__restrict__ positions, uint16_t *const *__restrict__ destinations, float epsilon)
{
	__shared__ float reduction[K31_THREADS / 32u];
	const uint16_t *row = latent + (uint64_t)blockIdx.x * K31_LATENT;
	uint16_t *out = destinations[blockIdx.x];
	float total = 0.0f, scale;
	for ( uint32_t index = threadIdx.x; index < K31_KV_RANK; index += K31_THREADS )
	{
		const float value = LmBf16ToFloat(row[index]);
		total += value * value;
	}
	total = LmBlockSum<K31_THREADS>(total, reduction);
	scale = rsqrtf(total / (float)K31_KV_RANK + epsilon);
	for ( uint32_t index = threadIdx.x; index < K31_KV_RANK; index += K31_THREADS )
		out[index] = LmFloatToBf16(LmBf16ToFloat(row[index]) * scale * LmBf16ToFloat(norm[index]));
	if ( threadIdx.x < K31_ROPE / 2u )
	{
		const uint32_t pair = threadIdx.x;
		const float angle = (float)positions[blockIdx.x] * inverse[pair];
		const float cosine = cosf(angle) * gain, sine = sinf(angle) * gain;
		const float even = LmBf16ToFloat(row[K31_KV_RANK + 2u * pair]), odd = LmBf16ToFloat(row[K31_KV_RANK + 2u * pair + 1u]);
		out[K31_KV_RANK + 2u * pair] = LmFloatToBf16(even * cosine - odd * sine);
		out[K31_KV_RANK + 2u * pair + 1u] = LmFloatToBf16(even * sine + odd * cosine);
	}
}

static SparkStatus K31DeviceOpen(void)
{
	const SparkStatus status = SparkDeviceInitialize(0);
	return status == SPARK_STATUS_DUPLICATE ? SPARK_STATUS_OK : status;
}

static SparkStatus K31Refuse(SparkDrafterModuleBind *bind, SparkStatus status, const char *format, const char *detail)
{
	if ( bind != 0 )
		snprintf(bind->text, sizeof(bind->text), format, detail);
	return status;
}

static SparkStatus K31LoadManifest(const char *pack_path, SparkJsonDocument *document, uint64_t *payload_base)
{
	uint8_t header[16];
	uint32_t magic, version;
	uint64_t bytes;
	char *text;
	SparkStatus status;
	FILE *file = fopen(pack_path, "rb");
	if ( file == 0 )
		return SPARK_STATUS_NOT_FOUND;
	if ( fread(header, 1u, sizeof(header), file) != sizeof(header) )
	{
		(void)fclose(file);
		return SPARK_STATUS_IO_ERROR;
	}
	memcpy(&magic, header, sizeof(magic));
	memcpy(&version, header + sizeof(magic), sizeof(version));
	memcpy(&bytes, header + sizeof(magic) + sizeof(version), sizeof(bytes));
	if ( magic != K31_PACK_MAGIC || version != K31_PACK_VERSION || bytes == 0u || bytes > (UINT64_C(64) << 20) )
	{
		(void)fclose(file);
		return SPARK_STATUS_VALIDATION_FAILED;
	}
	text = (char *)malloc((size_t)bytes);
	if ( text == 0 || fread(text, 1u, (size_t)bytes, file) != (size_t)bytes )
	{
		free(text);
		(void)fclose(file);
		return SPARK_STATUS_IO_ERROR;
	}
	(void)fclose(file);
	status = SparkJsonParseText(text, (size_t)bytes, document);
	free(text);
	*payload_base = (16u + bytes + K31_PACK_ALIGNMENT - 1u) / K31_PACK_ALIGNMENT * K31_PACK_ALIGNMENT;
	return status;
}

static SparkStatus K31Tensor(const K31State *state, const SparkJsonDocument *document, uint64_t payload_base, const char *name, uint64_t elements,
	const uint16_t **pointer)
{
	const int32_t tensors = SparkJsonFindObjectMember(document, SparkJsonGetRootToken(document), "tensors");
	const int32_t entry = tensors < 0 ? -1 : SparkJsonFindObjectMember(document, tensors, name);
	const int32_t offset_token = entry < 0 ? -1 : SparkJsonFindObjectMember(document, entry, "offset");
	const int32_t bytes_token = entry < 0 ? -1 : SparkJsonFindObjectMember(document, entry, "bytes");
	uint64_t offset = 0u, bytes = 0u;
	const void *slice = 0;
	if ( offset_token < 0 || bytes_token < 0 || SparkJsonGetUInt64(document, offset_token, &offset) != SPARK_STATUS_OK ||
		SparkJsonGetUInt64(document, bytes_token, &bytes) != SPARK_STATUS_OK || bytes != elements * sizeof(uint16_t) )
	{
		fprintf(stderr, "k31 dspark: pack tensor %s is missing or is not %llu bf16 elements\n", name, (unsigned long long)elements);
		return SPARK_STATUS_SCHEMA_ERROR;
	}
	if ( SparkWeightdLazyPackSlice(state->pack, payload_base + offset, bytes, &slice) != SPARK_STATUS_OK )
		return SPARK_STATUS_IO_ERROR;
	*pointer = (const uint16_t *)slice;
	return SPARK_STATUS_OK;
}

static float K31ConfigFloat(const SparkJsonDocument *document, int32_t object, const char *name, float fallback)
{
	const int32_t token = object < 0 ? -1 : SparkJsonFindObjectMember(document, object, name);
	float value = fallback;
	if ( token >= 0 )
		(void)SparkJsonGetFloat(document, token, &value);
	return value;
}

static double K31YarnScale(double scale, double weight)
{
	return scale <= 1.0 || weight == 0.0 ? 1.0 : 0.1 * weight * log(scale) + 1.0;
}

static SparkStatus K31Rope(K31State *state, const SparkJsonDocument *document)
{
	const int32_t config = SparkJsonFindObjectMember(document, SparkJsonGetRootToken(document), "config");
	const int32_t checkpoint = config < 0 ? -1 : SparkJsonFindObjectMember(document, config, "checkpoint_config");
	const int32_t rope = checkpoint < 0 ? -1 : SparkJsonFindObjectMember(document, checkpoint, "rope_parameters");
	const double base = K31ConfigFloat(document, rope, "rope_theta", 10000.0f), factor = K31ConfigFloat(document, rope, "factor", 1.0f);
	const double original = K31ConfigFloat(document, rope, "original_max_position_embeddings", 4096.0f);
	const double fast = K31ConfigFloat(document, rope, "beta_fast", 32.0f), slow = K31ConfigFloat(document, rope, "beta_slow", 1.0f);
	const double mscale = K31ConfigFloat(document, rope, "mscale", 1.0f), mscale_all = K31ConfigFloat(document, rope, "mscale_all_dim", 0.0f);
	const double attention = K31ConfigFloat(document, rope, "attn_factor", 1.0f);
	const double dimension = (double)K31_ROPE;
	float host[K31_ROPE / 2u];
	double low, high;
	if ( rope < 0 || checkpoint < 0 )
		return SPARK_STATUS_SCHEMA_ERROR;
	state->epsilon = K31ConfigFloat(document, checkpoint, "rms_norm_eps", 1e-6f);
	low = floor(dimension * log(original / (fast * 2.0 * K31_PI)) / (2.0 * log(base)));
	high = ceil(dimension * log(original / (slow * 2.0 * K31_PI)) / (2.0 * log(base)));
	low = low < 0.0 ? 0.0 : low;
	high = high > dimension - 1.0 ? dimension - 1.0 : high;
	if ( low == high )
		high += 0.001;
	for ( uint32_t pair = 0u; pair < K31_ROPE / 2u; pair++ )
	{
		const double extrapolated = 1.0 / pow(base, (2.0 * pair) / dimension);
		double ramp = ((double)pair - low) / (high - low);
		ramp = ramp < 0.0 ? 0.0 : ramp > 1.0 ? 1.0 : ramp;
		host[pair] = (float)(extrapolated / factor * ramp + extrapolated * (1.0 - ramp));
	}
	state->rope_gain = (float)(K31YarnScale(factor, mscale) / K31YarnScale(factor, mscale_all) * attention);
	if ( SparkGpuMalloc((void **)&state->rope_inverse, sizeof(host)) != SPARK_GPU_SUCCESS ||
		SparkGpuMemcpy(state->rope_inverse, host, sizeof(host), SPARK_GPU_COPY_HOST_TO_DEVICE) != SPARK_GPU_SUCCESS )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	return SPARK_STATUS_OK;
}

static SparkStatus K31Weights(K31State *state, const char *pack_path)
{
	SparkJsonDocument document;
	uint64_t payload_base = 0u;
	char name[96];
	SparkStatus status;
	memset(&document, 0, sizeof(document));
	status = K31LoadManifest(pack_path, &document, &payload_base);
	if ( status == SPARK_STATUS_OK )
		status = K31Rope(state, &document);
	if ( status == SPARK_STATUS_OK )
		status = K31Tensor(state, &document, payload_base, "context_proj.weight", (uint64_t)K31_HIDDEN * K31_TAP_WIDTH, &state->context_proj);
	if ( status == SPARK_STATUS_OK )
		status = K31Tensor(state, &document, payload_base, "context_norm.weight", K31_HIDDEN, &state->context_norm);
	for ( uint32_t layer = 0u; layer < K31_LAYERS && status == SPARK_STATUS_OK; layer++ )
	{
		snprintf(name, sizeof(name), "layers.%u.self_attn.kv_a_proj_with_mqa.weight", layer);
		status = K31Tensor(state, &document, payload_base, name, (uint64_t)K31_LATENT * K31_HIDDEN, &state->layers[layer].kv_a);
		snprintf(name, sizeof(name), "layers.%u.self_attn.kv_a_layernorm.weight", layer);
		if ( status == SPARK_STATUS_OK )
			status = K31Tensor(state, &document, payload_base, name, K31_KV_RANK, &state->layers[layer].kv_a_norm);
	}
	SparkJsonDocumentDestroy(&document);
	return status;
}

static SparkStatus K31Scratch(K31State *state)
{
	int multiprocessors = 0;
	if ( SparkGpuStreamCreateWithFlags(&state->stream, SPARK_GPU_STREAM_NON_BLOCKING) != SPARK_GPU_SUCCESS ||
		SparkGpuDeviceGetAttribute(&multiprocessors, SPARK_GPU_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0) != SPARK_GPU_SUCCESS ||
		SparkGpuMallocHost((void **)&state->host_taps, (size_t)K31_INGEST_ROWS * K31_TAP_WIDTH * sizeof(uint16_t)) != SPARK_GPU_SUCCESS ||
		SparkGpuMallocHost((void **)&state->host_positions, (size_t)K31_INGEST_ROWS * sizeof(uint32_t)) != SPARK_GPU_SUCCESS ||
		SparkGpuMalloc((void **)&state->taps, (size_t)K31_INGEST_ROWS * K31_TAP_WIDTH * sizeof(uint16_t)) != SPARK_GPU_SUCCESS ||
		SparkGpuMalloc((void **)&state->features_raw, (size_t)K31_INGEST_ROWS * K31_HIDDEN * sizeof(uint16_t)) != SPARK_GPU_SUCCESS ||
		SparkGpuMalloc((void **)&state->features, (size_t)K31_INGEST_ROWS * K31_HIDDEN * sizeof(uint16_t)) != SPARK_GPU_SUCCESS ||
		SparkGpuMalloc((void **)&state->latent_raw, (size_t)K31_INGEST_ROWS * K31_LATENT * sizeof(uint16_t)) != SPARK_GPU_SUCCESS ||
		SparkGpuMalloc((void **)&state->positions, (size_t)K31_INGEST_ROWS * sizeof(uint32_t)) != SPARK_GPU_SUCCESS )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	state->multiprocessors = (uint32_t)multiprocessors;
	return SPARK_STATUS_OK;
}

static void K31SequenceFree(K31State *state, K31Sequence *sequence)
{
	for ( uint32_t page = 0u; page < sequence->page_count; page++ )
		if ( sequence->pages[page] != 0 )
		{
			(void)SparkGpuFree(sequence->pages[page]);
			state->pages_used--;
		}
	free(sequence->pages);
	memset(sequence, 0, sizeof(*sequence));
}

static K31Sequence *K31SequenceFind(K31State *state, uint64_t id, uint32_t create)
{
	K31Sequence *empty = 0, *oldest = 0;
	for ( uint32_t index = 0u; index < K31_SEQUENCES_MAX; index++ )
	{
		K31Sequence *sequence = &state->sequences[index];
		if ( sequence->pages != 0 && sequence->id == id )
		{
			sequence->last_used = ++state->clock;
			return sequence;
		}
		if ( sequence->pages == 0 && empty == 0 )
			empty = sequence;
		if ( sequence->pages != 0 && (oldest == 0 || sequence->last_used < oldest->last_used) )
			oldest = sequence;
	}
	if ( create == 0u )
		return 0;
	if ( empty == 0 )
	{
		K31SequenceFree(state, oldest);
		empty = oldest;
	}
	empty->pages = (uint16_t **)calloc(K31_PAGES_MAX, sizeof(uint16_t *));
	if ( empty->pages == 0 )
		return 0;
	empty->id = id;
	empty->last_used = ++state->clock;
	return empty;
}

static K31Sequence *K31SequenceOldestBut(K31State *state, const K31Sequence *keep)
{
	K31Sequence *oldest = 0;
	for ( uint32_t index = 0u; index < K31_SEQUENCES_MAX; index++ )
	{
		K31Sequence *sequence = &state->sequences[index];
		if ( sequence->pages != 0 && sequence != keep && (oldest == 0 || sequence->last_used < oldest->last_used) )
			oldest = sequence;
	}
	return oldest;
}

static SparkStatus K31SequencePages(K31State *state, K31Sequence *sequence, uint64_t end)
{
	const uint64_t needed = (end + K31_PAGE_POSITIONS - 1u) / K31_PAGE_POSITIONS;
	if ( needed > K31_PAGES_MAX )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	for ( uint64_t page = sequence->page_count; page < needed; page++ )
	{
		while ( (state->pages_used + 1u) * K31_PAGE_BYTES > state->context_bytes )
		{
			K31Sequence *oldest = K31SequenceOldestBut(state, sequence);
			if ( oldest == 0 )
				return SPARK_STATUS_CAPACITY_EXCEEDED;
			K31SequenceFree(state, oldest);
		}
		if ( SparkGpuMalloc((void **)&sequence->pages[page], (size_t)K31_PAGE_BYTES) != SPARK_GPU_SUCCESS )
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		state->pages_used++;
		sequence->page_count = (uint32_t)page + 1u;
	}
	return SPARK_STATUS_OK;
}

static uint16_t *K31Row(const K31Sequence *sequence, uint32_t layer, uint64_t position)
{
	uint16_t *page = sequence->pages[position / K31_PAGE_POSITIONS];
	return page + ((uint64_t)layer * K31_PAGE_POSITIONS + position % K31_PAGE_POSITIONS) * K31_LATENT;
}

static SparkStatus K31WriteRows(K31State *state, K31Sequence *sequence, uint64_t first, uint32_t rows)
{
	uint16_t *host_rows[K31_INGEST_ROWS];
	uint16_t **device_rows = 0;
	SparkStatus status = SPARK_STATUS_OK;
	if ( LmDenseRows<LmBf16Format>(state->context_proj, LmScaleTensorNone(), state->taps, state->features_raw, (float *)0, rows, K31_TAP_WIDTH,
		K31_HIDDEN, 0u, state->multiprocessors, state->stream) != LM_LAUNCH_OK )
		return SPARK_STATUS_INTERNAL_ERROR;
	LM_LAUNCH((K31RmsRowsKernel), rows, K31_THREADS, 0, state->stream, state->features_raw, state->context_norm, state->features, K31_HIDDEN, state->epsilon);
	if ( SparkGpuMalloc((void **)&device_rows, sizeof(host_rows)) != SPARK_GPU_SUCCESS )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	for ( uint32_t layer = 0u; layer < K31_LAYERS && status == SPARK_STATUS_OK; layer++ )
	{
		for ( uint32_t row = 0u; row < rows; row++ )
			host_rows[row] = K31Row(sequence, layer, first + row);
		if ( LmDenseRows<LmBf16Format>(state->layers[layer].kv_a, LmScaleTensorNone(), state->features, state->latent_raw, (float *)0, rows, K31_HIDDEN,
			K31_LATENT, 0u, state->multiprocessors, state->stream) != LM_LAUNCH_OK ||
			SparkGpuMemcpyAsync(device_rows, host_rows, (size_t)rows * sizeof(uint16_t *), SPARK_GPU_COPY_HOST_TO_DEVICE, state->stream) != SPARK_GPU_SUCCESS )
			status = SPARK_STATUS_INTERNAL_ERROR;
		if ( status == SPARK_STATUS_OK )
			LM_LAUNCH((K31LatentRowsKernel), rows, K31_THREADS, 0, state->stream, state->latent_raw, state->layers[layer].kv_a_norm, state->rope_inverse,
				state->rope_gain, state->positions, device_rows, state->epsilon);
		if ( status == SPARK_STATUS_OK && SparkGpuStreamSynchronize(state->stream) != SPARK_GPU_SUCCESS )
			status = SPARK_STATUS_IO_ERROR;
	}
	(void)SparkGpuFree(device_rows);
	return status;
}

static SparkStatus K31Ingest(void *opaque, const SparkTapStreamReaderFrame *frame)
{
	K31State *state = (K31State *)opaque;
	const uint32_t rows = frame->header.row_count;
	const uint64_t hidden = frame->hello != 0 ? frame->hello->hidden : 0u;
	K31Sequence *sequence;
	SparkStatus status;
	if ( rows == 0u )
		return SPARK_STATUS_OK;
	if ( frame->required_count != K31_TAPS || hidden != K31_HIDDEN || frame->rows == 0 )
		return SPARK_STATUS_SCHEMA_ERROR;
	for ( uint32_t tap = 0u; tap < K31_TAPS; tap++ )
		if ( ((uint64_t)frame->required_index[tap] + 1u) * K31_HIDDEN > frame->row_elements )
			return SPARK_STATUS_SCHEMA_ERROR;
	sequence = K31SequenceFind(state, frame->header.sequence_id, 1u);
	if ( sequence == 0 )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	status = K31SequencePages(state, sequence, frame->header.first_position + rows);
	for ( uint32_t done = 0u; done < rows && status == SPARK_STATUS_OK; done += K31_INGEST_ROWS )
	{
		const uint32_t part = rows - done < K31_INGEST_ROWS ? rows - done : K31_INGEST_ROWS;
		for ( uint32_t row = 0u; row < part; row++ )
		{
			const uint16_t *source = frame->rows + (uint64_t)(done + row) * frame->row_elements;
			for ( uint32_t tap = 0u; tap < K31_TAPS; tap++ )
				memcpy(state->host_taps + (uint64_t)row * K31_TAP_WIDTH + (uint64_t)tap * K31_HIDDEN,
					source + (uint64_t)frame->required_index[tap] * K31_HIDDEN, K31_HIDDEN * sizeof(uint16_t));
			state->host_positions[row] = (uint32_t)(frame->header.first_position + done + row);
		}
		if ( SparkGpuMemcpyAsync(state->taps, state->host_taps, (size_t)part * K31_TAP_WIDTH * sizeof(uint16_t), SPARK_GPU_COPY_HOST_TO_DEVICE,
			state->stream) != SPARK_GPU_SUCCESS ||
			SparkGpuMemcpyAsync(state->positions, state->host_positions, (size_t)part * sizeof(uint32_t), SPARK_GPU_COPY_HOST_TO_DEVICE,
				state->stream) != SPARK_GPU_SUCCESS )
			status = SPARK_STATUS_IO_ERROR;
		if ( status == SPARK_STATUS_OK )
			status = K31WriteRows(state, sequence, frame->header.first_position + done, part);
	}
	if ( status == SPARK_STATUS_OK && frame->header.first_position + rows > sequence->held_end )
		sequence->held_end = frame->header.first_position + rows;
	state->frames++;
	state->rows += rows;
	return status;
}

static SparkStatus K31Close(void *opaque)
{
	K31State *state = (K31State *)opaque;
	SparkStatus status = SPARK_STATUS_OK;
	if ( state == 0 )
		return SPARK_STATUS_OK;
	for ( uint32_t index = 0u; index < K31_SEQUENCES_MAX; index++ )
		if ( state->sequences[index].pages != 0 )
			K31SequenceFree(state, &state->sequences[index]);
	(void)SparkGpuFree(state->taps);
	(void)SparkGpuFree(state->features_raw);
	(void)SparkGpuFree(state->features);
	(void)SparkGpuFree(state->latent_raw);
	(void)SparkGpuFree(state->positions);
	(void)SparkGpuFree(state->rope_inverse);
	if ( state->host_taps != 0 )
		(void)SparkGpuFreeHost(state->host_taps);
	if ( state->host_positions != 0 )
		(void)SparkGpuFreeHost(state->host_positions);
	if ( state->stream != 0 )
		(void)SparkGpuStreamDestroy(state->stream);
	if ( state->pack != 0 )
		status = SparkWeightdLazyPackDestroy(state->pack);
	free(state);
	return status;
}

static SparkStatus K31Open(const SparkDrafterModuleLane *lane, void **state_out, SparkDrafterModuleBind *bind)
{
	SparkWeightdLazyAttachRequest request;
	SparkWeightdLazyPackOptions options;
	K31State *state;
	size_t free_bytes = 0u, total_bytes = 0u;
	uint64_t need, started;
	SparkStatus status;
	*state_out = 0;
	if ( lane == 0 || lane->abi_version != SPARK_DRAFTER_MODULE_ABI_VERSION || lane->descriptor_bytes < sizeof(*lane) )
		return K31Refuse(bind, SPARK_STATUS_ABI_MISMATCH, "the lane descriptor is not drafter ABI %s", "1");
	if ( K31DeviceOpen() != SPARK_STATUS_OK || SparkGpuMemGetInfo(&free_bytes, &total_bytes) != SPARK_GPU_SUCCESS )
		return K31Refuse(bind, SPARK_STATUS_IO_ERROR, "lane %s cannot read the device's free memory", lane->lane_name);
	bind->device_free_bytes = (uint64_t)free_bytes;
	bind->device_total_bytes = (uint64_t)total_bytes;
	need = lane->pack_bytes + 255u + lane->context_bytes;
	if ( need > (uint64_t)free_bytes )
	{
		snprintf(bind->text, sizeof(bind->text), "lane %s needs %llu bytes of device memory (pack %llu + context cap %llu), %llu of %llu free",
			lane->lane_name, (unsigned long long)need, (unsigned long long)lane->pack_bytes, (unsigned long long)lane->context_bytes,
			(unsigned long long)free_bytes, (unsigned long long)total_bytes);
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	}
	if ( lane->context_bytes < K31_PAGE_BYTES )
		return K31Refuse(bind, SPARK_STATUS_INVALID_ARGUMENT, "lane %s has a context cap below one context page", lane->lane_name);
	state = (K31State *)calloc(1u, sizeof(*state));
	if ( state == 0 )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	memset(&request, 0, sizeof(request));
	memset(&options, 0, sizeof(options));
	request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
	request.identity.arena_bytes = lane->pack_bytes;
	snprintf(request.identity.model, sizeof(request.identity.model), "%s", lane->model);
	snprintf(request.identity.revision, sizeof(request.identity.revision), "%s", lane->revision);
	snprintf(request.identity.pack_sha256, sizeof(request.identity.pack_sha256), "%s", lane->pack_sha256);
	snprintf(request.pack_path, sizeof(request.pack_path), "%s", lane->pack_path);
	options.timeout_nanoseconds = lane->timeout_ns;
	state->context_bytes = lane->context_bytes;
	started = SparkMonotonicNs();
	status = SparkWeightdLazyPackCreate(lane->weightd_socket, &request, &options, &state->pack);
	if ( status != SPARK_STATUS_OK )
	{
		snprintf(bind->text, sizeof(bind->text), "lane %s: the pack %s did not bind", lane->lane_name, lane->pack_path);
		free(state);
		return status;
	}
	status = K31Weights(state, lane->pack_path);
	if ( status == SPARK_STATUS_OK )
		status = K31Scratch(state);
	if ( status != SPARK_STATUS_OK )
	{
		snprintf(bind->text, sizeof(bind->text), "lane %s: the pack %s is not a K3.1 Pro DSpark pack this module reads", lane->lane_name, lane->pack_path);
		(void)K31Close(state);
		return status;
	}
	bind->bind_ns = SparkMonotonicNs() - started;
	bind->device_bytes = state->pack->spine_allocation_bytes;
	bind->first_hold = state->pack->attached.loaded_from_pack;
	snprintf(bind->text, sizeof(bind->text), "lane %s bound %s", lane->lane_name, lane->pack_path);
	*state_out = state;
	return SPARK_STATUS_OK;
}

static SparkStatus K31Propose(void *opaque, const SparkDraftServerRequest *request, SparkDraftBridgeNode *nodes, uint32_t capacity, uint32_t *count)
{
	(void)opaque;
	*count = 0u;
	if ( capacity == 0u )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	nodes[0].token_id = request->anchor_token_id;
	nodes[0].parent_index = SPARK_DRAFT_BRIDGE_ROOT_PARENT_INDEX;
	nodes[0].depth = 0u;
	nodes[0].source_bit = request->speculator_mask & (~request->speculator_mask + 1u);
	nodes[0].score = 1.0f;
	*count = 1u;
	return SPARK_STATUS_OK;
}

static const SparkDrafterModuleInterface k31_interface = {
	SPARK_DRAFTER_MODULE_ABI_VERSION,
	(uint32_t)sizeof(SparkDrafterModuleInterface),
	"k31-dspark",
	K31Open,
	K31Ingest,
	K31Propose,
	K31Close,
};

extern "C" __attribute__((visibility("default"))) const SparkDrafterModuleInterface *spark_drafter_module_interface(void)
{
	return &k31_interface;
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_k31_dspark_context_rows(void *opaque, uint64_t sequence_id, uint32_t layer,
	uint64_t first, uint32_t count, uint16_t *host)
{
	K31State *state = (K31State *)opaque;
	K31Sequence *sequence = state != 0 ? K31SequenceFind(state, sequence_id, 0u) : 0;
	if ( sequence == 0 || layer >= K31_LAYERS || first + count > sequence->held_end )
		return SPARK_STATUS_NOT_FOUND;
	for ( uint32_t row = 0u; row < count; row++ )
		if ( SparkGpuMemcpy(host + (uint64_t)row * K31_LATENT, K31Row(sequence, layer, first + row), K31_LATENT * sizeof(uint16_t),
			SPARK_GPU_COPY_DEVICE_TO_HOST) != SPARK_GPU_SUCCESS )
			return SPARK_STATUS_IO_ERROR;
	return SPARK_STATUS_OK;
}
