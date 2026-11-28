#ifndef SPARKPIPE_DSPARK_DRAFT_STAGE_COMMON_CUH
#define SPARKPIPE_DSPARK_DRAFT_STAGE_COMMON_CUH

#include "sparkpipe/spark_clock.h"
#include "sparkpipe/spark_device.h"
#include "sparkpipe/spark_draft_tree_pack.h"
#include "sparkpipe/spark_drafter_module.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_gpu.h"
#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_weightd_lazy_pack.h"
#include "inference/kernels/dense_rows.cuh"
#include "inference/kernels/drafter_blackwell.cuh"
#include "inference/kernels/kernel_math.cuh"
#include "inference/kernels/norm.cuh"
#include "runtime/launch.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <type_traits>

#define DSPARK_PACK_MAGIC 0x4B505244u
#define DSPARK_PACK_VERSION 1u
#define DSPARK_PACK_ALIGNMENT 128u
#define DSPARK_PI 3.14159265358979323846
#define DSPARK_INVERSE_PI 0.31830988618379067154
#define DSPARK_THREADS 256u
#define DSPARK_WARP 32u
#define DSPARK_WARPS (DSPARK_THREADS / DSPARK_WARP)
#define DSPARK_PICK_THREADS 1024u

#define DSPARK_DEVICE(field, count) (SparkGpuMalloc((void **)&(field), (size_t)(count) * sizeof(*(field))) == SPARK_GPU_SUCCESS)

enum DsparkNormMode
{
	DSPARK_NORM_F32 = 0,
	DSPARK_NORM_BF16_OUT = 1,
	DSPARK_NORM_BF16_BOTH = 2,
};

template<typename T>
static inline bool DsparkAllocHost(T **ptr, size_t count)
{
	return SparkGpuMallocHost((void **)ptr, count * sizeof(T)) == SPARK_GPU_SUCCESS;
}

static inline void DsparkFreeDeviceBuffers(void *const *buffers, uint32_t count)
{
	for ( uint32_t index = 0u; index < count; index++ )
		(void)SparkGpuFree(buffers[index]);
}

static inline void DsparkFreeHostBuffers(void *const *buffers, uint32_t count)
{
	for ( uint32_t index = 0u; index < count; index++ )
		if ( buffers[index] != 0 )
			(void)SparkGpuFreeHost(buffers[index]);
}

static inline SparkStatus DsparkReadback(void *host, const void *device, size_t bytes)
{
	if ( SparkGpuMemcpy(host, device, bytes, SPARK_GPU_COPY_DEVICE_TO_HOST) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return SPARK_STATUS_OK;
}

static inline SparkStatus DsparkSettle(SparkGpuStream stream, SparkStatus status)
{
	if ( SparkGpuStreamSynchronize(stream) != SPARK_GPU_SUCCESS && status == SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return status;
}

static inline SparkStatus DsparkRefuse(SparkDrafterModuleBind *bind, SparkStatus status, const char *format, const char *detail)
{
	if ( bind != 0 )
		snprintf(bind->text, sizeof(bind->text), format, detail);
	SPARK_FAIL(status);
}

static inline int32_t DsparkMember(const SparkJsonDocument *document, int32_t object, const char *name)
{
	return object < 0 ? -1 : SparkJsonFindObjectMember(document, object, name);
}

static inline double DsparkYarnScale(double scale, double weight)
{
	return scale <= 1.0 || weight == 0.0 ? 1.0 : 0.1 * weight * log(scale) + 1.0;
}

static inline SparkStatus DsparkLoadManifest(const char *pack_path, SparkJsonDocument *document, uint64_t *payload_base)
{
	uint8_t header[16];
	uint32_t magic, version;
	uint64_t bytes;
	char *text;
	SparkStatus status;
	FILE *file = fopen(pack_path, "rb");
	if ( file == 0 )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	if ( fread(header, 1u, sizeof(header), file) != sizeof(header) )
	{
		(void)fclose(file);
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	memcpy(&magic, header, sizeof(magic));
	memcpy(&version, header + sizeof(magic), sizeof(version));
	memcpy(&bytes, header + sizeof(magic) + sizeof(version), sizeof(bytes));
	if ( magic != DSPARK_PACK_MAGIC || version != DSPARK_PACK_VERSION || bytes == 0u || bytes > (UINT64_C(64) << 20) )
	{
		(void)fclose(file);
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	}
	text = (char *)malloc((size_t)bytes);
	if ( text == 0 || fread(text, 1u, (size_t)bytes, file) != (size_t)bytes )
	{
		free(text);
		(void)fclose(file);
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	(void)fclose(file);
	status = SparkJsonParseText(text, (size_t)bytes, document);
	free(text);
	*payload_base = (sizeof(header) + bytes + DSPARK_PACK_ALIGNMENT - 1u) / DSPARK_PACK_ALIGNMENT * DSPARK_PACK_ALIGNMENT;
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	return SPARK_STATUS_OK;
}

static inline SparkStatus DsparkTensorBytes(SparkWeightdLazyPack *pack, const SparkJsonDocument *document, uint64_t payload_base, const char *tag,
	const char *name, uint64_t expected_bytes, const void **pointer)
{
	const int32_t tensors = SparkJsonFindObjectMember(document, SparkJsonGetRootToken(document), "tensors");
	const int32_t entry = tensors < 0 ? -1 : SparkJsonFindObjectMember(document, tensors, name);
	const int32_t offset_token = entry < 0 ? -1 : SparkJsonFindObjectMember(document, entry, "offset");
	const int32_t bytes_token = entry < 0 ? -1 : SparkJsonFindObjectMember(document, entry, "bytes");
	uint64_t offset = 0u, bytes = 0u;
	const void *slice = 0;
	if ( offset_token < 0 || bytes_token < 0 || SparkJsonGetUInt64(document, offset_token, &offset) != SPARK_STATUS_OK ||
		SparkJsonGetUInt64(document, bytes_token, &bytes) != SPARK_STATUS_OK || bytes != expected_bytes )
	{
		fprintf(stderr, "%s: pack tensor %s is missing or is not %llu bytes\n", tag, name, (unsigned long long)expected_bytes);
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	if ( SparkWeightdLazyPackSlice(pack, payload_base + offset, bytes, &slice) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	*pointer = slice;
	return SPARK_STATUS_OK;
}

static inline SparkStatus DsparkTensorBf16(SparkWeightdLazyPack *pack, const SparkJsonDocument *document, uint64_t payload_base, const char *tag,
	const char *name, uint64_t expected_elements, const uint16_t **pointer)
{
	const void *slice = 0;
	const SparkStatus status = DsparkTensorBytes(pack, document, payload_base, tag, name, expected_elements * sizeof(uint16_t), &slice);
	if ( status == SPARK_STATUS_OK )
		*pointer = (const uint16_t *)slice;
	return status;
}

typedef struct DsparkExpectedU64
{
	int32_t object;
	const char *name;
	uint64_t value;
} DsparkExpectedU64;

typedef struct DsparkExpectedBool
{
	int32_t object;
	const char *name;
	bool value;
} DsparkExpectedBool;

typedef struct DsparkBf16Spec
{
	const char *name;
	uint64_t elements;
	const uint16_t **pointer;
} DsparkBf16Spec;

static inline SparkStatus DsparkCheckU64Table(const SparkJsonDocument *document, const char *tag, const DsparkExpectedU64 *items, uint32_t count)
{
	for ( uint32_t index = 0u; index < count; index++ )
	{
		uint64_t value = 0u;
		const int32_t token = DsparkMember(document, items[index].object, items[index].name);
		if ( token < 0 || SparkJsonGetUInt64(document, token, &value) != SPARK_STATUS_OK || value != items[index].value )
		{
			fprintf(stderr, "%s: checkpoint_config %s is not %llu, the shape this module is built for\n", tag, items[index].name,
				(unsigned long long)items[index].value);
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		}
	}
	return SPARK_STATUS_OK;
}

static inline SparkStatus DsparkCheckBoolTable(const SparkJsonDocument *document, const char *tag, const DsparkExpectedBool *items, uint32_t count)
{
	for ( uint32_t index = 0u; index < count; index++ )
	{
		bool flag = false;
		const int32_t token = DsparkMember(document, items[index].object, items[index].name);
		if ( token < 0 || SparkJsonGetBoolean(document, token, &flag) != SPARK_STATUS_OK || flag != items[index].value )
		{
			fprintf(stderr, "%s: checkpoint_config %s does not match module contract\n", tag, items[index].name);
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		}
	}
	return SPARK_STATUS_OK;
}

static inline SparkStatus DsparkLoadBf16Table(SparkWeightdLazyPack *pack, const SparkJsonDocument *document, uint64_t payload_base, const char *tag,
	const DsparkBf16Spec *specs, uint32_t count)
{
	for ( uint32_t index = 0u; index < count; index++ )
	{
		const SparkStatus status = DsparkTensorBf16(pack, document, payload_base, tag, specs[index].name, specs[index].elements, specs[index].pointer);
		if ( status != SPARK_STATUS_OK )
			return status;
	}
	return SPARK_STATUS_OK;
}

static inline SparkStatus DsparkLoadPrefixedBf16Table(SparkWeightdLazyPack *pack, const SparkJsonDocument *document, uint64_t payload_base, const char *tag,
	const char *prefix_fmt, uint32_t layer, const DsparkBf16Spec *specs, uint32_t count)
{
	for ( uint32_t index = 0u; index < count; index++ )
	{
		char name[128];
		snprintf(name, sizeof(name), prefix_fmt, layer, specs[index].name);
		const SparkStatus status = DsparkTensorBf16(pack, document, payload_base, tag, name, specs[index].elements, specs[index].pointer);
		if ( status != SPARK_STATUS_OK )
			return status;
	}
	return SPARK_STATUS_OK;
}

static inline SparkStatus DsparkGemm(const uint16_t *weight, const uint16_t *input, uint16_t *output, uint32_t rows, uint32_t input_dimension,
	uint32_t output_dimension, uint32_t multiprocessors, SparkGpuStream stream)
{
	if ( LmDenseRows<LmBf16Format>(weight, LmScaleTensorNone(), input, output, (float *)0, rows, input_dimension, output_dimension, 0u,
		multiprocessors, stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return SPARK_STATUS_OK;
}

static inline SparkStatus DsparkGemmF32(const uint16_t *weight, const uint16_t *input, float *output, uint32_t rows, uint32_t input_dimension,
	uint32_t output_dimension, uint32_t multiprocessors, SparkGpuStream stream)
{
	if ( LmDenseRows<LmBf16Format>(weight, LmScaleTensorNone(), input, (uint16_t *)0, output, rows, input_dimension, output_dimension, 0u,
		multiprocessors, stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return SPARK_STATUS_OK;
}

static inline SparkStatus DsparkValidateLaneAndBindPack(const SparkDrafterModuleLane *lane, uint64_t min_context_bytes, const char *below_context_fmt,
	SparkDrafterModuleBind *bind, SparkWeightdLazyPack **pack_out, SparkGpuStream *stream_out, uint64_t *started_out)
{
	SparkWeightdLazyAttachRequest request;
	SparkWeightdLazyPackOptions options;
	size_t free_bytes = 0u, total_bytes = 0u;
	uint64_t need;
	SparkStatus status;
	if ( lane == 0 || lane->abi_version != SPARK_DRAFTER_MODULE_ABI_VERSION || lane->descriptor_bytes < sizeof(*lane) )
		return DsparkRefuse(bind, SPARK_STATUS_ABI_MISMATCH, "the lane descriptor is not drafter ABI %s", "1");
	{
		const SparkStatus device = SparkDeviceInitialize(0);
		if ( (device != SPARK_STATUS_OK && device != SPARK_STATUS_DUPLICATE) || SparkGpuMemGetInfo(&free_bytes, &total_bytes) != SPARK_GPU_SUCCESS )
			return DsparkRefuse(bind, SPARK_STATUS_IO_ERROR, "lane %s cannot read the device's free memory", lane->lane_name);
	}
	bind->device_free_bytes = (uint64_t)free_bytes;
	bind->device_total_bytes = (uint64_t)total_bytes;
	need = lane->pack_bytes + 255u + lane->context_bytes;
	if ( need > (uint64_t)free_bytes )
	{
		snprintf(bind->text, sizeof(bind->text), "lane %s needs %llu bytes of device memory (pack %llu + context cap %llu), %llu of %llu free",
			lane->lane_name, (unsigned long long)need, (unsigned long long)lane->pack_bytes, (unsigned long long)lane->context_bytes,
			(unsigned long long)free_bytes, (unsigned long long)total_bytes);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	if ( lane->context_bytes < min_context_bytes )
		return DsparkRefuse(bind, SPARK_STATUS_INVALID_ARGUMENT, below_context_fmt, lane->lane_name);
	memset(&request, 0, sizeof(request));
	memset(&options, 0, sizeof(options));
	request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
	request.identity.arena_bytes = lane->pack_bytes;
	snprintf(request.identity.model, sizeof(request.identity.model), "%s", lane->model);
	snprintf(request.identity.revision, sizeof(request.identity.revision), "%s", lane->revision);
	snprintf(request.identity.pack_sha256, sizeof(request.identity.pack_sha256), "%s", lane->pack_sha256);
	snprintf(request.pack_path, sizeof(request.pack_path), "%s", lane->pack_path);
	options.timeout_nanoseconds = lane->timeout_ns;
	*started_out = SparkMonotonicNs();
	status = SparkWeightdLazyPackCreate(lane->weightd_socket, &request, &options, pack_out);
	if ( status != SPARK_STATUS_OK )
	{
		snprintf(bind->text, sizeof(bind->text), "lane %s: the pack %s did not bind", lane->lane_name, lane->pack_path);
		SPARK_FAIL(status);
	}
	if ( stream_out != 0 && SparkGpuStreamCreateWithFlags(stream_out, SPARK_GPU_STREAM_NON_BLOCKING) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return SPARK_STATUS_OK;
}

static inline void DsparkFillBindSuccess(const SparkDrafterModuleLane *lane, const SparkWeightdLazyPack *pack, uint64_t started,
	SparkDrafterModuleBind *bind, uint32_t capacity, bool show_windows)
{
	bind->bind_ns = SparkMonotonicNs() - started;
	bind->device_bytes = pack->spine_allocation_bytes;
	bind->first_hold = pack->attached.loaded_from_pack;
	if ( show_windows )
		snprintf(bind->text, sizeof(bind->text), "lane %s bound %s with %u sequence windows", lane->lane_name, lane->pack_path, capacity);
	else
		snprintf(bind->text, sizeof(bind->text), "lane %s bound %s", lane->lane_name, lane->pack_path);
}

template<typename StateT, typename CloseFn, typename InitFn, typename CapFn>
static inline SparkStatus DsparkDrafterOpen(const SparkDrafterModuleLane *lane, uint64_t min_context_bytes, const char *below_context_fmt,
	const char *bad_pack_fmt, bool show_windows, void **state_out, SparkDrafterModuleBind *bind, CloseFn close_fn, InitFn init_fn, CapFn cap_fn)
{
	StateT *state;
	uint64_t started = 0u;
	SparkStatus status;
	*state_out = 0;
	state = (StateT *)calloc(1u, sizeof(*state));
	if ( state == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = DsparkValidateLaneAndBindPack(lane, min_context_bytes, below_context_fmt, bind, &state->pack, &state->stream, &started);
	if ( status != SPARK_STATUS_OK )
	{
		(void)close_fn(state);
		SPARK_FAIL(status);
	}
	status = init_fn(state);
	if ( status != SPARK_STATUS_OK )
	{
		snprintf(bind->text, sizeof(bind->text), bad_pack_fmt, lane->lane_name, lane->pack_path);
		(void)close_fn(state);
		SPARK_FAIL(status);
	}
	DsparkFillBindSuccess(lane, state->pack, started, bind, cap_fn(state), show_windows);
	*state_out = state;
	return SPARK_STATUS_OK;
}

static __device__ __forceinline__ uint32_t DsparkSwizzle(uint32_t row, uint32_t chunk)
{
	return chunk ^ (row & 7u);
}

static __device__ __forceinline__ void DsparkLoadX4(uint32_t address, uint32_t r[4])
{
	asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
		: "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(address));
}

static __device__ __forceinline__ void DsparkLoadX4Trans(uint32_t address, uint32_t r[4])
{
	asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
		: "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(address));
}

static __device__ __forceinline__ void DsparkStage16Cg(uint32_t destination, const void *source, uint32_t bytes)
{
	asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(destination), "l"(source), "r"(bytes) : "memory");
}

template<uint32_t LANES>
static __device__ __forceinline__ void DsparkWarpSampleSubwarp(float value, uint32_t candidate, uint32_t lane, uint32_t seed, uint32_t position,
	float nucleus, uint32_t top_k, uint32_t *choice_out, uint32_t *runner_out, float *runner_prob_out)
{
	const uint32_t active = (1u << LANES) - 1u;
	float top = value, probability, sum, before = 0.0f;
	uint32_t rank = 0u, winner_rank, runner_rank, choice = 0u, runner = 0u;
	float runner_prob = 0.0f;
	unsigned long long key;
	for ( uint32_t other = 0u; other < LANES; other++ )
	{
		const float ov = __shfl_sync(active, value, other);
		rank += ov > value || (ov == value && other < lane) ? 1u : 0u;
	}
	for ( uint32_t mask = LANES / 2u; mask > 0u; mask >>= 1u )
		top = fmaxf(top, __shfl_xor_sync(active, top, mask));
	probability = expf(value - top);
	sum = probability;
	for ( uint32_t mask = LANES / 2u; mask > 0u; mask >>= 1u )
		sum += __shfl_xor_sync(active, sum, mask);
	probability /= sum;
	for ( uint32_t other = 0u; other < LANES; other++ )
	{
		const float op = __shfl_sync(active, probability, other);
		const float ov = __shfl_sync(active, value, other);
		if ( ov > value || (ov == value && other < lane) )
			before += op;
	}
	key = (((nucleus <= 0.0f) || (before < nucleus)) && (top_k == 0u || rank < top_k)) || rank == 0u
		? ((unsigned long long)SparkMathSampleOrdered(value + SparkMathGumbelNoise(seed, position, candidate)) << 32u) |
			(unsigned long long)(0xffffffffu - rank)
		: 0ull;
	for ( uint32_t mask = LANES / 2u; mask > 0u; mask >>= 1u )
	{
		const unsigned long long other = __shfl_xor_sync(active, key, mask);
		key = other > key ? other : key;
	}
	winner_rank = 0xffffffffu - (uint32_t)key;
	runner_rank = winner_rank == 0u ? 1u : 0u;
	for ( uint32_t other = 0u; other < LANES; other++ )
	{
		const uint32_t other_rank = __shfl_sync(active, rank, other);
		const float other_prob = __shfl_sync(active, probability, other);
		if ( other_rank == winner_rank )
			choice = other;
		if ( other_rank == runner_rank )
		{
			runner = other;
			runner_prob = other_prob;
		}
	}
	*choice_out = choice;
	if ( runner_out != 0 )
		*runner_out = runner;
	if ( runner_prob_out != 0 )
		*runner_prob_out = runner_prob;
}

static __device__ __forceinline__ void DsparkRotation(uint32_t position, float inverse, float gain, float *cosine, float *sine)
{
	const float angle = (float)position * inverse;
	const double half_turns = (double)angle * DSPARK_INVERSE_PI;
	const float reduced = (float)(half_turns - 2.0 * rint(0.5 * half_turns));
	float s, c;
	sincospif(reduced, &s, &c);
	*cosine = c * gain;
	*sine = s * gain;
}

static __device__ __forceinline__ void DsparkRotationUnit(uint32_t position, float inverse, float *cosine, float *sine)
{
	const float angle = (float)position * inverse;
	const double half_turns = (double)angle * DSPARK_INVERSE_PI;
	const float reduced = (float)(half_turns - 2.0 * rint(0.5 * half_turns));
	sincospif(reduced, sine, cosine);
}

static __device__ __forceinline__ float DsparkWarpSum(float value)
{
	for ( uint32_t mask = DSPARK_WARP / 2u; mask > 0u; mask >>= 1u )
		value += __shfl_xor_sync(0xffffffffu, value, mask);
	return value;
}

static __device__ __forceinline__ float DsparkWarpMax(float value)
{
	for ( uint32_t mask = DSPARK_WARP / 2u; mask > 0u; mask >>= 1u )
		value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, mask));
	return value;
}

static __device__ __forceinline__ unsigned long long DsparkWarpKey(unsigned long long key)
{
	for ( uint32_t mask = DSPARK_WARP / 2u; mask > 0u; mask >>= 1u )
	{
		const unsigned long long other = __shfl_xor_sync(0xffffffffu, key, mask);
		key = other > key ? other : key;
	}
	return key;
}

static __device__ __forceinline__ unsigned long long DsparkRowKey(const float *row, uint32_t slot)
{
	return ((unsigned long long)SparkMathSampleOrdered(row[slot]) << 32u) | (unsigned long long)(0xffffffffu - slot);
}

static __device__ __forceinline__ uint32_t DsparkKeySlot(unsigned long long key)
{
	return 0xffffffffu - (uint32_t)key;
}

static __device__ __forceinline__ void DsparkTop2Insert(unsigned long long candidate, unsigned long long *first, unsigned long long *second)
{
	if ( candidate > *first )
	{
		*second = *first;
		*first = candidate;
	}
	else if ( candidate > *second && candidate != *first )
		*second = candidate;
}

template<uint32_t VOCAB, bool HAS_D2T>
static __device__ __forceinline__ uint32_t DsparkMapDraftToken(const int64_t *__restrict__ d2t, uint32_t draft_id)
{
	if constexpr ( HAS_D2T )
	{
		const int64_t mapped = (int64_t)draft_id + d2t[draft_id];
		return mapped >= 0 && mapped < (int64_t)VOCAB ? (uint32_t)mapped : 0u;
	}
	else
		return draft_id;
}

template<uint32_t HALF, uint32_t DIM>
__global__ __launch_bounds__(DSPARK_THREADS, 1)
void DsparkRopeTableKernel(float *__restrict__ inverse, float theta)
{
	if ( threadIdx.x < HALF )
		inverse[threadIdx.x] = 1.0f / powf(theta, (float)(2u * threadIdx.x) / (float)DIM);
}

template<uint32_t HALF, uint32_t DIM>
__global__ __launch_bounds__(DSPARK_THREADS, 1)
void DsparkYarnTableKernel(float *__restrict__ inverse, double base, double factor, double original, double beta_fast, double beta_slow)
{
	if ( threadIdx.x < HALF )
	{
		const double dimension = (double)DIM;
		double low = floor(dimension * log(original / (beta_fast * 2.0 * DSPARK_PI)) / (2.0 * log(base)));
		double high = ceil(dimension * log(original / (beta_slow * 2.0 * DSPARK_PI)) / (2.0 * log(base)));
		const double extrapolated = 1.0 / pow(base, (2.0 * (double)threadIdx.x) / dimension);
		double ramp;
		low = low < 0.0 ? 0.0 : low;
		high = high > dimension - 1.0 ? dimension - 1.0 : high;
		if ( low == high )
			high += 0.001;
		ramp = ((double)threadIdx.x - low) / (high - low);
		ramp = ramp < 0.0 ? 0.0 : ramp > 1.0 ? 1.0 : ramp;
		inverse[threadIdx.x] = (float)(extrapolated / factor * ramp + extrapolated * (1.0 - ramp));
	}
}

template<bool ROUND_NORM, bool DEP_WAIT = false, typename SrcT = uint16_t>
__global__ __launch_bounds__(DSPARK_THREADS, 2)
void DsparkRmsKernel(const SrcT *__restrict__ input, const uint16_t *__restrict__ weight, uint16_t *__restrict__ output, uint32_t dimension,
	float epsilon)
{
	if constexpr ( DEP_WAIT )
		LmDependentWait();
	__shared__ float reduction[DSPARK_WARPS];
	const uint64_t row = blockIdx.x;
	const SrcT *src = input + row * dimension;
	uint16_t *dst = output + row * dimension;
	float total = 0.0f, scale;
	for ( uint32_t index = threadIdx.x; index < dimension; index += DSPARK_THREADS )
	{
		float value;
		if constexpr ( std::is_same_v<SrcT, float> )
			value = src[index];
		else
			value = LmBf16ToFloat(__ldg(src + index));
		total += value * value;
	}
	total = LmBlockSum<DSPARK_THREADS>(total, reduction);
	scale = rsqrtf(total / (float)dimension + epsilon);
	for ( uint32_t index = threadIdx.x; index < dimension; index += DSPARK_THREADS )
	{
		float raw;
		if constexpr ( std::is_same_v<SrcT, float> )
			raw = src[index];
		else
			raw = LmBf16ToFloat(__ldg(src + index));
		const float scaled = raw * scale;
		const float normalized = ROUND_NORM ? LmBf16ToFloat(LmFloatToBf16(scaled)) : scaled;
		dst[index] = LmFloatToBf16(normalized * LmBf16ToFloat(__ldg(weight + index)));
	}
	LmDependentRelease();
}

template<bool ROUND_NORM>
__global__ __launch_bounds__(DSPARK_THREADS, 2)
void DsparkResidualRmsKernel(const uint16_t *__restrict__ base, const uint16_t *__restrict__ addend, const uint16_t *__restrict__ weight,
	uint16_t *__restrict__ residual, uint16_t *__restrict__ output, uint32_t dimension, float epsilon)
{
	__shared__ float reduction[DSPARK_WARPS];
	const uint64_t row = blockIdx.x;
	const uint64_t base_offset = row * dimension;
	float total = 0.0f, scale;
	for ( uint32_t index = threadIdx.x; index < dimension; index += DSPARK_THREADS )
	{
		const uint16_t sum_bf16 = LmFloatToBf16(LmBf16ToFloat(__ldg(base + base_offset + index)) + LmBf16ToFloat(__ldg(addend + base_offset + index)));
		const float value = LmBf16ToFloat(sum_bf16);
		residual[base_offset + index] = sum_bf16;
		total += value * value;
	}
	total = LmBlockSum<DSPARK_THREADS>(total, reduction);
	scale = rsqrtf(total / (float)dimension + epsilon);
	for ( uint32_t index = threadIdx.x; index < dimension; index += DSPARK_THREADS )
	{
		const float scaled = LmBf16ToFloat(residual[base_offset + index]) * scale;
		const float normalized = ROUND_NORM ? LmBf16ToFloat(LmFloatToBf16(scaled)) : scaled;
		output[base_offset + index] = LmFloatToBf16(normalized * LmBf16ToFloat(__ldg(weight + index)));
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(DSPARK_THREADS, 2)
static void DsparkResidualKernel(const uint16_t *__restrict__ base, const uint16_t *__restrict__ addend, uint16_t *__restrict__ output, uint32_t count)
{
	for ( uint32_t index = blockIdx.x * DSPARK_THREADS + threadIdx.x; index < count; index += gridDim.x * DSPARK_THREADS )
		output[index] = LmFloatToBf16(LmBf16ToFloat(__ldg(base + index)) + LmBf16ToFloat(__ldg(addend + index)));
	LmDependentRelease();
}

__global__ __launch_bounds__(DSPARK_THREADS, 2)
static void DsparkSwigluKernel(const uint16_t *__restrict__ gate, const uint16_t *__restrict__ up, uint16_t *__restrict__ output, uint32_t count)
{
	const uint4 *g4 = (const uint4 *)gate;
	const uint4 *u4 = (const uint4 *)up;
	uint4 *o4 = (uint4 *)output;
	for ( uint32_t vec = blockIdx.x * DSPARK_THREADS + threadIdx.x; vec < count / 8u; vec += gridDim.x * DSPARK_THREADS )
	{
		const uint4 gv = __ldg(g4 + vec), uv = __ldg(u4 + vec);
		const uint32_t ga[4] = {gv.x, gv.y, gv.z, gv.w};
		const uint32_t ua[4] = {uv.x, uv.y, uv.z, uv.w};
		uint32_t oa[4];
		for ( uint32_t i = 0u; i < 4u; i++ )
		{
			const float g0 = LmBf16ToFloat((uint16_t)(ga[i] & 0xffffu));
			const float g1 = LmBf16ToFloat((uint16_t)(ga[i] >> 16u));
			const uint16_t r0 = LmFloatToBf16(g0 / (1.0f + expf(-g0)) * LmBf16ToFloat((uint16_t)(ua[i] & 0xffffu)));
			const uint16_t r1 = LmFloatToBf16(g1 / (1.0f + expf(-g1)) * LmBf16ToFloat((uint16_t)(ua[i] >> 16u)));
			oa[i] = (uint32_t)r0 | ((uint32_t)r1 << 16u);
		}
		o4[vec] = make_uint4(oa[0], oa[1], oa[2], oa[3]);
	}
	LmDependentRelease();
}

template<uint32_t HIDDEN, typename DraftT>
__global__ __launch_bounds__(DSPARK_THREADS, 2)
void DsparkEmbedBf16Kernel(const uint16_t *__restrict__ embed, const DraftT *__restrict__ draft, uint16_t *__restrict__ hidden)
{
	const uint32_t row = blockIdx.x, token = row == 0u ? draft->anchor : draft->mask;
	const uint4 *src = (const uint4 *)(embed + (uint64_t)token * HIDDEN);
	uint4 *dst = (uint4 *)(hidden + (uint64_t)row * HIDDEN);
	for ( uint32_t vec = threadIdx.x; vec < HIDDEN / 8u; vec += DSPARK_THREADS )
		dst[vec] = __ldg(src + vec);
	LmDependentRelease();
}

template<uint32_t HIDDEN, typename DraftT>
__global__ __launch_bounds__(DSPARK_THREADS, 2)
void DsparkEmbedF32Kernel(const uint16_t *__restrict__ embed, const DraftT *__restrict__ draft, float *__restrict__ hidden)
{
	const uint32_t row = blockIdx.x, token = row == 0u ? draft->anchor : draft->mask;
	const uint4 *src = (const uint4 *)(embed + (uint64_t)token * HIDDEN);
	float4 *dst = (float4 *)(hidden + (uint64_t)row * HIDDEN);
	for ( uint32_t index = threadIdx.x; index < HIDDEN / 8u; index += DSPARK_THREADS )
	{
		const uint4 p = __ldg(src + index);
		dst[2u * index] = make_float4(__uint_as_float(p.x << 16u), __uint_as_float(p.x & 0xffff0000u),
			__uint_as_float(p.y << 16u), __uint_as_float(p.y & 0xffff0000u));
		dst[2u * index + 1u] = make_float4(__uint_as_float(p.z << 16u), __uint_as_float(p.z & 0xffff0000u),
			__uint_as_float(p.w << 16u), __uint_as_float(p.w & 0xffff0000u));
	}
	LmDependentRelease();
}

template<uint32_t HEAD_DIM, uint32_t HEAD_THREADS, uint32_t WINDOW, uint32_t COLUMNS, DsparkNormMode NORM_MODE, bool USE_GAIN, typename DraftT>
__global__ __launch_bounds__(HEAD_THREADS, 4)
void DsparkHeadNormRopeRingKernel(const uint16_t *__restrict__ input, const uint16_t *__restrict__ weight, const float *__restrict__ inverse,
	float rope_gain, const uint32_t *__restrict__ positions, uint32_t heads, float epsilon, uint16_t *__restrict__ output,
	uint16_t *__restrict__ cache_keys, const DraftT *__restrict__ draft, uint64_t layer_offset, uint32_t use_ring)
{
	if ( draft != 0 && output == 0 )
		LmDependentWait();
	constexpr uint32_t HALF = HEAD_DIM / 2u;
	__shared__ float reduction[HEAD_THREADS / DSPARK_WARP];
	__shared__ float normed[HEAD_DIM];
	const uint32_t row = blockIdx.x / heads, head = blockIdx.x % heads, lane = threadIdx.x;
	const uint32_t position = positions[row];
	const float value = lane < HEAD_DIM ? LmBf16ToFloat(__ldg(input + ((uint64_t)row * heads + head) * HEAD_DIM + lane)) : 0.0f;
	const float scale = rsqrtf(LmBlockSum<HEAD_THREADS>(value * value, reduction) / (float)HEAD_DIM + epsilon);
	if ( lane < HEAD_DIM )
	{
		const float w = LmBf16ToFloat(__ldg(weight + lane));
		if constexpr ( NORM_MODE == DSPARK_NORM_BF16_BOTH )
			normed[lane] = LmBf16ToFloat(LmFloatToBf16(LmBf16ToFloat(LmFloatToBf16(value * scale)) * w));
		else if constexpr ( NORM_MODE == DSPARK_NORM_BF16_OUT )
			normed[lane] = LmBf16ToFloat(LmFloatToBf16(value * scale * w));
		else
			normed[lane] = value * scale * w;
	}
	__syncthreads();
	if ( lane < HEAD_DIM )
	{
		float cosine, sine, rotated;
		if constexpr ( USE_GAIN )
			DsparkRotation(position, __ldg(inverse + (lane % HALF)), rope_gain, &cosine, &sine);
		else
			DsparkRotationUnit(position, __ldg(inverse + (lane % HALF)), &cosine, &sine);
		rotated = lane < HALF ? normed[lane] * cosine - normed[lane + HALF] * sine : normed[lane] * cosine + normed[lane - HALF] * sine;
		if ( output != 0 )
			output[((uint64_t)row * heads + head) * HEAD_DIM + lane] = LmFloatToBf16(rotated);
		else
		{
			uint16_t *dst_keys = cache_keys != 0 ? cache_keys : draft->cache + layer_offset;
			const uint32_t column = use_ring != 0u ? position % WINDOW : draft->block_at + row;
			dst_keys[((uint64_t)head * COLUMNS + column) * HEAD_DIM + lane] = LmFloatToBf16(rotated);
		}
	}
	LmDependentRelease();
}

template<uint32_t KV_WIDTH, uint32_t HEAD_DIM, uint32_t WINDOW, uint32_t COLUMNS, typename DraftT>
__global__ __launch_bounds__(DSPARK_THREADS, 2)
void DsparkStoreValuesRingKernel(const uint16_t *__restrict__ values, const uint32_t *__restrict__ positions, const DraftT *__restrict__ draft,
	uint64_t layer_offset, uint16_t *__restrict__ cache)
{
	if ( draft != 0 )
		LmDependentWait();
	const uint32_t row = blockIdx.x;
	const uint32_t column = positions != 0 ? positions[row] % WINDOW : draft->block_at + row;
	uint16_t *dst = cache != 0 ? cache : draft->cache + layer_offset;
	const uint4 *vsrc = (const uint4 *)(values + (uint64_t)row * KV_WIDTH);
	for ( uint32_t vec = threadIdx.x; vec < KV_WIDTH / 8u; vec += DSPARK_THREADS )
	{
		const uint32_t index = vec * 8u;
		const uint32_t head = index / HEAD_DIM;
		const uint32_t col = index % HEAD_DIM;
		*(uint4 *)(dst + ((uint64_t)head * COLUMNS + column) * HEAD_DIM + col) = __ldg(vsrc + vec);
	}
	LmDependentRelease();
}

template<uint32_t HEADS, uint32_t KV_HEADS, uint32_t BLOCK, uint32_t COLUMNS, uint32_t HEAD_DIM, uint32_t SPLITS, bool CAUSAL, bool PAGED,
	bool GROUP_PARTIALS, typename DraftT>
__global__ __launch_bounds__(HEAD_DIM, 4)
void DsparkAttentionMergeKernel(const uint16_t *__restrict__ q, const uint16_t *__restrict__ block_k, const uint16_t *__restrict__ block_v,
	uint64_t keys_offset, uint64_t values_offset, const DraftT *__restrict__ draft, float scale, const float *__restrict__ partials,
	uint16_t *__restrict__ output)
{
	LmDependentWait();
	constexpr uint32_t GROUP = HEADS / KV_HEADS;
	constexpr uint32_t Q_WIDTH = HEADS * HEAD_DIM;
	__shared__ float reduction[HEAD_DIM / DSPARK_WARP];
	const uint32_t row = blockIdx.x / HEADS, head = blockIdx.x % HEADS, kv_head = head / GROUP, lane = threadIdx.x;
	const uint32_t splits = draft->splits;
	const float own = LmBf16ToFloat(__ldg(q + ((uint64_t)row * HEADS + head) * HEAD_DIM + lane));
	float top = -INFINITY, sum = 0.0f, accumulated = 0.0f;
	for ( uint32_t split = 0u; split < splits; split++ )
	{
		const float *part;
		if constexpr ( GROUP_PARTIALS )
		{
			const uint32_t query_index = (head % GROUP) * BLOCK + row;
			part = partials + (((uint64_t)kv_head * SPLITS + split) * (GROUP * BLOCK) + query_index) * (HEAD_DIM + 2u);
		}
		else
			part = partials + (((uint64_t)head * SPLITS + split) * BLOCK + row) * (HEAD_DIM + 2u);
		const float part_top = part[HEAD_DIM], next = fmaxf(top, part_top);
		float carry, weight;
		if ( part_top == -INFINITY )
			continue;
		carry = top == -INFINITY ? 0.0f : expf(top - next);
		weight = expf(part_top - next);
		sum = sum * carry + part[HEAD_DIM + 1u] * weight;
		accumulated = accumulated * carry + part[lane] * weight;
		top = next;
	}
	const uint32_t key_end = CAUSAL ? (row + 1u) : BLOCK;
	for ( uint32_t key = 0u; key < key_end; key++ )
	{
		float k_val, v_val;
		if constexpr ( PAGED )
		{
			const uint64_t kv_at = ((uint64_t)key * KV_HEADS + kv_head) * HEAD_DIM + lane;
			k_val = LmBf16ToFloat(block_k[kv_at]);
			v_val = LmBf16ToFloat(block_v[kv_at]);
		}
		else
		{
			const uint64_t kv_at = ((uint64_t)kv_head * COLUMNS + draft->block_at + key) * HEAD_DIM + lane;
			k_val = LmBf16ToFloat(__ldg(draft->cache + keys_offset + kv_at));
			v_val = LmBf16ToFloat(__ldg(draft->cache + values_offset + kv_at));
		}
		const float score = LmBlockSum<HEAD_DIM>(own * k_val, reduction) * scale;
		const float next = fmaxf(top, score), carry = top == -INFINITY ? 0.0f : expf(top - next), weight = expf(score - next);
		sum = sum * carry + weight;
		accumulated = accumulated * carry + weight * v_val;
		top = next;
	}
	output[(uint64_t)row * Q_WIDTH + head * HEAD_DIM + lane] = LmFloatToBf16(accumulated / sum);
	LmDependentRelease();
}

template<uint32_t WINDOW>
struct DsparkRingSequence
{
	uint64_t id;
	uint64_t control_generation;
	uint64_t request_id;
	uint64_t last_used;
	uint32_t live;
	uint32_t retired;
	int64_t positions[WINDOW];
	uint32_t tokens[WINDOW];
	uint16_t *cache;
};

template<uint32_t WINDOW, uint32_t MAX_SEQ>
struct DsparkRingPool
{
	using Seq = DsparkRingSequence<WINDOW>;
	Seq *sequences;
	uint32_t capacity;
	uint64_t clock;
	uint64_t gaps;
	uint64_t donor_fills;

	bool Init(uint64_t context_bytes, uint64_t seq_elements, uint16_t **caches_out)
	{
		uint64_t cap = context_bytes / (seq_elements * sizeof(uint16_t));
		cap = cap < 1u ? 1u : cap > MAX_SEQ ? MAX_SEQ : cap;
		capacity = (uint32_t)cap;
		sequences = (Seq *)calloc(cap, sizeof(Seq));
		if ( sequences == 0 || !DSPARK_DEVICE(*caches_out, cap * seq_elements) )
			return false;
		for ( uint32_t index = 0u; index < capacity; index++ )
			sequences[index].cache = *caches_out + (uint64_t)index * seq_elements;
		return true;
	}

	static uint64_t CheckedFrom(const Seq *sequence, uint64_t first, uint64_t position)
	{
		uint64_t checked = first;
		for ( uint64_t at = first; at < position; at++ )
			if ( Held(sequence, at) == 0u )
				checked = at + 1u;
		return checked;
	}

	template<uint32_t LAYERS, uint32_t KV_HEADS, uint32_t COLUMNS, uint32_t HEAD_DIM>
	static SparkStatus CopyKvRange(Seq *dst, const Seq *src, uint32_t slot, uint32_t count, SparkGpuStream stream)
	{
		constexpr uint64_t LAYER_KV = (uint64_t)KV_HEADS * COLUMNS * HEAD_DIM;
		const size_t bytes = (size_t)count * HEAD_DIM * sizeof(uint16_t);
		for ( uint32_t layer = 0u; layer < LAYERS; layer++ )
		{
			const uint64_t k_off = (uint64_t)(2u * layer) * LAYER_KV;
			const uint64_t v_off = k_off + LAYER_KV;
			for ( uint32_t head = 0u; head < KV_HEADS; head++ )
			{
				const uint64_t at = ((uint64_t)head * COLUMNS + slot) * HEAD_DIM;
				if ( SparkGpuMemcpyAsync(dst->cache + k_off + at, src->cache + k_off + at, bytes, SPARK_GPU_COPY_DEVICE_TO_DEVICE, stream) !=
						SPARK_GPU_SUCCESS ||
					SparkGpuMemcpyAsync(dst->cache + v_off + at, src->cache + v_off + at, bytes, SPARK_GPU_COPY_DEVICE_TO_DEVICE, stream) !=
						SPARK_GPU_SUCCESS )
					SPARK_FAIL(SPARK_STATUS_IO_ERROR);
			}
		}
		return SPARK_STATUS_OK;
	}

	template<uint32_t LAYERS, uint64_t LAYER_KV>
	SparkStatus CacheReadback(uint64_t sequence_id, uint32_t layer, uint16_t *keys, uint16_t *values, int64_t *positions) const
	{
		const Seq *sequence = Live(sequence_id);
		SparkStatus status;
		if ( sequence == 0 || layer >= LAYERS || keys == 0 || values == 0 || positions == 0 )
			SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
		memcpy(positions, sequence->positions, sizeof(sequence->positions));
		status = DsparkReadback(keys, sequence->cache + (uint64_t)(2u * layer) * LAYER_KV, LAYER_KV * sizeof(uint16_t));
		if ( status == SPARK_STATUS_OK )
			status = DsparkReadback(values, sequence->cache + (uint64_t)(2u * layer + 1u) * LAYER_KV, LAYER_KV * sizeof(uint16_t));
		return status;
	}

	static void Reset(Seq *sequence)
	{
		for ( uint32_t slot = 0u; slot < WINDOW; slot++ )
		{
			sequence->positions[slot] = -1;
			sequence->tokens[slot] = 0u;
		}
		sequence->retired = 0u;
	}

	Seq *Live(uint64_t id) const
	{
		for ( uint32_t index = 0u; index < capacity; index++ )
			if ( sequences[index].live != 0u && sequences[index].retired == 0u && sequences[index].id == id )
				return &sequences[index];
		return 0;
	}

	Seq *Claim() const
	{
		Seq *oldest = 0;
		for ( uint32_t index = 0u; index < capacity; index++ )
		{
			Seq *sequence = &sequences[index];
			if ( sequence->live == 0u )
				return sequence;
			if ( oldest == 0 || sequence->last_used < oldest->last_used )
				oldest = sequence;
		}
		return oldest;
	}

	static uint32_t Held(const Seq *sequence, uint64_t position)
	{
		return sequence->positions[position % WINDOW] == (int64_t)position ? 1u : 0u;
	}

	static uint32_t Usable(const Seq *sequence, const Seq *donor, const SparkDraftServerRequest *request, uint64_t position)
	{
		return Held(sequence, position) == 0u && Held(donor, position) != 0u && position < request->committed_token_count &&
			donor->tokens[position % WINDOW] == request->committed_token_ids[position] ? 1u : 0u;
	}

	template<typename CopySlotsFn>
	SparkStatus FillFromDonors(Seq *sequence, const SparkDraftServerRequest *request, uint64_t first, CopySlotsFn copy_slots)
	{
		uint8_t tried[MAX_SEQ];
		memset(tried, 0, sizeof(tried));
		for ( ;; )
		{
			Seq *donor = 0;
			uint32_t chosen = 0u, missing = 0u;
			uint64_t position = first;
			for ( uint64_t at = first; at < request->position; at++ )
				missing += Held(sequence, at) == 0u ? 1u : 0u;
			if ( missing == 0u )
				return SPARK_STATUS_OK;
			for ( uint32_t index = 0u; index < capacity; index++ )
			{
				Seq *other = &sequences[index];
				if ( other->live != 0u && other != sequence && tried[index] == 0u && (donor == 0 || other->last_used > donor->last_used) )
				{
					donor = other;
					chosen = index;
				}
			}
			if ( donor == 0 )
				return SPARK_STATUS_OK;
			tried[chosen] = 1u;
			while ( position < request->position )
			{
				uint64_t stop = position;
				if ( Usable(sequence, donor, request, position) == 0u )
				{
					position++;
					continue;
				}
				while ( stop < request->position && Usable(sequence, donor, request, stop) != 0u && (stop == position || stop % WINDOW != 0u) )
					stop++;
				if ( copy_slots(sequence, donor, (uint32_t)(position % WINDOW), (uint32_t)(stop - position)) != SPARK_STATUS_OK )
					SPARK_FAIL(SPARK_STATUS_IO_ERROR);
				for ( uint64_t at = position; at < stop; at++ )
				{
					sequence->positions[at % WINDOW] = (int64_t)at;
					sequence->tokens[at % WINDOW] = donor->tokens[at % WINDOW];
					donor_fills++;
				}
				position = stop;
			}
		}
	}

	template<uint32_t BLOCK, uint32_t SLIDING, bool PER_ROW_MASK, typename DraftT>
	static void Span(Seq *sequence, uint64_t checked, uint64_t position, DraftT *draft)
	{
		const uint64_t count = position - checked;
		const uint32_t first = count != 0u ? (uint32_t)(checked % WINDOW) : 0u, end = first + (uint32_t)count;
		uint32_t newer = 0u;
		memset(draft->valid, 0, sizeof(draft->valid));
		draft->masked = 0u;
		if constexpr ( !PER_ROW_MASK )
		{
			if ( count == WINDOW )
			{
				draft->start = 0u;
				draft->block_at = WINDOW;
				return;
			}
		}
		for ( uint32_t slot = end; slot < WINDOW && slot < end + BLOCK; slot++ )
			newer |= count != 0u && sequence->positions[slot] > (int64_t)(position - 1u) ? 1u : 0u;
		if ( (!PER_ROW_MASK || position <= SLIDING - BLOCK) && end <= WINDOW && newer == 0u )
		{
			for ( uint32_t slot = end; slot < WINDOW && slot < end + BLOCK; slot++ )
				sequence->positions[slot] = -1;
			draft->start = first;
			draft->block_at = end;
			return;
		}
		for ( uint64_t at = checked; at < position; at++ )
		{
			if constexpr ( PER_ROW_MASK )
			{
				const uint64_t gap = position - at;
				const uint32_t max_row = gap <= SLIDING - BLOCK ? BLOCK - 1u : (uint32_t)((SLIDING - 1u) - gap);
				draft->valid[at % WINDOW] = (uint8_t)((1u << (max_row + 1u)) - 1u);
			}
			else
				draft->valid[at % WINDOW] = 1u;
		}
		draft->masked = 1u;
		draft->start = 0u;
		draft->block_at = WINDOW;
	}

	template<uint32_t TAPS, uint32_t HIDDEN, uint32_t INGEST_ROWS, typename IngestRowsFn>
	SparkStatus IngestFrame(const SparkTapStreamReaderFrame *frame, uint16_t *host_taps, uint32_t *host_positions, uint16_t *taps,
		uint32_t *positions, SparkGpuStream stream, uint64_t *frames_counter, uint64_t *rows_counter, IngestRowsFn ingest_rows)
	{
		constexpr uint32_t TAP_WIDTH = TAPS * HIDDEN;
		const uint32_t rows = frame->header.row_count;
		const uint64_t hidden = frame->hello != 0 ? frame->hello->hidden : 0u;
		const uint32_t skip = rows > WINDOW ? rows - WINDOW : 0u;
		const uint64_t first = frame->header.first_position + skip;
		Seq *sequence;
		SparkStatus status = SPARK_STATUS_OK;
		if ( rows == 0u )
			return SPARK_STATUS_OK;
		if ( frame->required_count != TAPS || hidden != HIDDEN || frame->rows == 0 )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		for ( uint32_t tap = 0u; tap < TAPS; tap++ )
			if ( ((uint64_t)frame->required_index[tap] + 1u) * HIDDEN > frame->row_elements )
				SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		if ( frame->header.first_position + rows > UINT32_MAX )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		sequence = Live(frame->header.sequence_id);
		if ( sequence != 0 && frame->header.first_position != 0u &&
			(sequence->control_generation != frame->header.control_generation || sequence->request_id != frame->header.request_id) )
		{
			sequence->retired = 1u;
			sequence->last_used = ++clock;
			sequence = 0;
		}
		if ( sequence == 0 )
		{
			sequence = Claim();
			Reset(sequence);
			sequence->live = 1u;
			sequence->id = frame->header.sequence_id;
		}
		else if ( frame->header.first_position == 0u )
			Reset(sequence);
		sequence->control_generation = frame->header.control_generation;
		sequence->request_id = frame->header.request_id;
		sequence->last_used = ++clock;
		for ( uint32_t row = skip; row < rows; row++ )
			sequence->positions[(frame->header.first_position + row) % WINDOW] = -1;
		for ( uint32_t done = skip; done < rows && status == SPARK_STATUS_OK; done += INGEST_ROWS )
		{
			const uint32_t part = rows - done < INGEST_ROWS ? rows - done : INGEST_ROWS;
			for ( uint32_t row = 0u; row < part; row++ )
			{
				const uint16_t *source = frame->rows + (uint64_t)(done + row) * frame->row_elements;
				for ( uint32_t tap = 0u; tap < TAPS; tap++ )
					memcpy(host_taps + (uint64_t)row * TAP_WIDTH + (uint64_t)tap * HIDDEN,
						source + (uint64_t)frame->required_index[tap] * HIDDEN, HIDDEN * sizeof(uint16_t));
				host_positions[row] = (uint32_t)(frame->header.first_position + done + row);
			}
			if ( SparkGpuMemcpyAsync(taps, host_taps, (size_t)part * TAP_WIDTH * sizeof(uint16_t), SPARK_GPU_COPY_HOST_TO_DEVICE, stream) !=
					SPARK_GPU_SUCCESS ||
				SparkGpuMemcpyAsync(positions, host_positions, (size_t)part * sizeof(uint32_t), SPARK_GPU_COPY_HOST_TO_DEVICE, stream) !=
					SPARK_GPU_SUCCESS )
			{
				(void)SparkGpuStreamSynchronize(stream);
				SPARK_FAIL(SPARK_STATUS_IO_ERROR);
			}
			status = ingest_rows(sequence, part);
		}
		if ( status == SPARK_STATUS_OK )
		{
			for ( uint32_t slot = 0u; slot < WINDOW; slot++ )
				if ( sequence->positions[slot] >= (int64_t)first )
					sequence->positions[slot] = -1;
			for ( uint32_t row = skip; row < rows; row++ )
			{
				const uint64_t position = frame->header.first_position + row;
				sequence->positions[position % WINDOW] = (int64_t)position;
				sequence->tokens[position % WINDOW] = frame->tokens != 0 ? frame->tokens[row] : 0u;
			}
		}
		(*frames_counter)++;
		(*rows_counter) += rows;
		return status;
	}

	template<typename CopySlotsFn>
	SparkStatus PrepareProposeContext(const SparkDraftServerRequest *request, uint64_t first, SparkGpuStream stream, Seq **seq_out,
		uint64_t *checked_out, CopySlotsFn copy_slots)
	{
		Seq *sequence = Live(request->sequence_id);
		uint64_t checked = first;
		SparkStatus status;
		if ( sequence == 0 || (request->position != 0u && Held(sequence, request->position - 1u) == 0u) )
			return SPARK_DRAFTER_MODULE_PROPOSE_AWAIT_TAPS;
		status = FillFromDonors(sequence, request, first, copy_slots);
		if ( status != SPARK_STATUS_OK )
			return DsparkSettle(stream, status);
		for ( uint64_t position = first; position < request->position; position++ )
			if ( Held(sequence, position) == 0u )
				checked = position + 1u;
		if ( checked != first )
			gaps++;
		if ( request->committed_token_count < request->position )
			return DsparkSettle(stream, SPARK_DRAFTER_MODULE_PROPOSE_NO_CONTEXT);
		for ( uint64_t position = checked; position < request->position; position++ )
			if ( sequence->tokens[position % WINDOW] != request->committed_token_ids[position] )
				return DsparkSettle(stream, SPARK_DRAFTER_MODULE_PROPOSE_NO_CONTEXT);
		*seq_out = sequence;
		*checked_out = checked;
		return SPARK_STATUS_OK;
	}
};

struct DsparkPagedSequence
{
	uint64_t id;
	uint64_t control_generation;
	uint64_t request_id;
	uint64_t held_end;
	uint64_t verified;
	uint64_t last_used;
	uint32_t retired;
	uint32_t page_count;
	uint16_t **pages;
	uint8_t *held;
	uint32_t *tokens;
};

template<uint32_t PAGE_POSITIONS, uint32_t PAGES_MAX, uint64_t PAGE_BYTES, uint32_t SEQUENCES_MAX, uint32_t RUNS_MAX>
struct DsparkPagedPool
{
	static_assert(SEQUENCES_MAX <= 64u, "one bit per sequence marks the donors tried");
	uint64_t context_bytes;
	uint64_t pages_used;
	uint64_t clock;
	DsparkPagedSequence sequences[SEQUENCES_MAX];

	void Free(DsparkPagedSequence *sequence)
	{
		for ( uint32_t page = 0u; page < sequence->page_count; page++ )
			if ( sequence->pages[page] != 0 )
			{
				(void)SparkGpuFree(sequence->pages[page]);
				pages_used--;
			}
		free(sequence->pages);
		free(sequence->held);
		free(sequence->tokens);
		memset(sequence, 0, sizeof(*sequence));
	}

	void FreeAll()
	{
		for ( uint32_t index = 0u; index < SEQUENCES_MAX; index++ )
			if ( sequences[index].pages != 0 )
				Free(&sequences[index]);
	}

	DsparkPagedSequence *Find(uint64_t id, uint32_t create)
	{
		DsparkPagedSequence *empty = 0, *oldest = 0;
		for ( uint32_t index = 0u; index < SEQUENCES_MAX; index++ )
		{
			DsparkPagedSequence *sequence = &sequences[index];
			if ( sequence->pages != 0 && sequence->retired == 0u && sequence->id == id )
			{
				sequence->last_used = ++clock;
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
			Free(oldest);
			empty = oldest;
		}
		empty->pages = (uint16_t **)calloc(PAGES_MAX, sizeof(uint16_t *));
		if ( empty->pages == 0 )
			return 0;
		empty->id = id;
		empty->last_used = ++clock;
		return empty;
	}

	DsparkPagedSequence *OldestBut(const DsparkPagedSequence *keep, const DsparkPagedSequence *donor)
	{
		DsparkPagedSequence *oldest = 0;
		for ( uint32_t index = 0u; index < SEQUENCES_MAX; index++ )
		{
			DsparkPagedSequence *sequence = &sequences[index];
			if ( sequence->pages != 0 && sequence != keep && sequence != donor && (oldest == 0 || sequence->last_used < oldest->last_used) )
				oldest = sequence;
		}
		return oldest;
	}

	SparkStatus EnsurePages(DsparkPagedSequence *sequence, uint64_t end, const DsparkPagedSequence *donor)
	{
		const uint64_t needed = (end + PAGE_POSITIONS - 1u) / PAGE_POSITIONS;
		const uint64_t had = (uint64_t)sequence->page_count * PAGE_POSITIONS;
		uint8_t *held;
		uint32_t *tokens;
		if ( needed > PAGES_MAX )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		if ( needed <= sequence->page_count )
			return SPARK_STATUS_OK;
		held = (uint8_t *)realloc(sequence->held, (size_t)(needed * PAGE_POSITIONS));
		if ( held == 0 )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		sequence->held = held;
		memset(held + had, 0, (size_t)(needed * PAGE_POSITIONS - had));
		tokens = (uint32_t *)realloc(sequence->tokens, (size_t)(needed * PAGE_POSITIONS) * sizeof(uint32_t));
		if ( tokens == 0 )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		sequence->tokens = tokens;
		memset(tokens + had, 0, (size_t)(needed * PAGE_POSITIONS - had) * sizeof(uint32_t));
		for ( uint64_t page = sequence->page_count; page < needed; page++ )
		{
			while ( (pages_used + 1u) * PAGE_BYTES > context_bytes )
			{
				DsparkPagedSequence *oldest = OldestBut(sequence, donor);
				if ( oldest == 0 )
					SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
				Free(oldest);
			}
			if ( SparkGpuMalloc((void **)&sequence->pages[page], (size_t)PAGE_BYTES) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
			pages_used++;
			sequence->page_count = (uint32_t)page + 1u;
		}
		return SPARK_STATUS_OK;
	}

	static void Forget(DsparkPagedSequence *sequence, uint64_t first)
	{
		const uint64_t capacity = (uint64_t)sequence->page_count * PAGE_POSITIONS;
		if ( first < capacity )
			memset(sequence->held + first, 0, (size_t)(capacity - first));
		if ( sequence->held_end > first )
			sequence->held_end = first;
		if ( sequence->verified > first )
			sequence->verified = first;
	}

	static uint32_t Covers(const DsparkPagedSequence *sequence, uint64_t position)
	{
		if ( position == 0u )
			return 1u;
		return sequence != 0 && position - 1u < (uint64_t)sequence->page_count * PAGE_POSITIONS && sequence->held[position - 1u] != 0u ? 1u : 0u;
	}

	static uint32_t Complete(const DsparkPagedSequence *sequence, uint64_t position)
	{
		return position <= (uint64_t)sequence->page_count * PAGE_POSITIONS && memchr(sequence->held, 0, (size_t)position) == 0 ? 1u : 0u;
	}

	static uint32_t TokensMatch(DsparkPagedSequence *sequence, const SparkDraftServerRequest *request)
	{
		const uint64_t capacity = (uint64_t)sequence->page_count * PAGE_POSITIONS;
		const uint64_t end = request->position < capacity ? request->position : capacity;
		uint32_t complete = end == request->position ? 1u : 0u;
		if ( (uint64_t)request->committed_token_count < request->position )
			return 0u;
		for ( uint64_t at = sequence->verified; at < end; at++ )
		{
			if ( sequence->held[at] == 0u )
				complete = 0u;
			else if ( sequence->tokens[at] != request->committed_token_ids[at] )
				return 0u;
		}
		if ( complete != 0u && end > sequence->verified )
			sequence->verified = end;
		return 1u;
	}

	template<typename DraftT>
	static SparkStatus BuildRuns(const DsparkPagedSequence *sequence, uint64_t position, DraftT *draft, uint32_t *held_out)
	{
		const uint64_t capacity = sequence != 0 ? (uint64_t)sequence->page_count * PAGE_POSITIONS : 0u;
		const uint64_t limit = position < capacity ? position : capacity;
		uint64_t at = 0u, held = 0u;
		draft->runs = 0u;
		while ( at < limit )
		{
			const uint8_t *start = (const uint8_t *)memchr(sequence->held + at, 1, (size_t)(limit - at));
			const uint8_t *stop;
			uint64_t begin, finish;
			if ( start == 0 )
				break;
			begin = (uint64_t)(start - sequence->held);
			stop = (const uint8_t *)memchr(start, 0, (size_t)(limit - begin));
			finish = stop != 0 ? (uint64_t)(stop - sequence->held) : limit;
			if ( draft->runs == RUNS_MAX )
				SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
			draft->run_first[draft->runs] = (uint32_t)held;
			draft->run_position[draft->runs] = (uint32_t)begin;
			draft->runs++;
			held += finish - begin;
			at = finish;
		}
		*held_out = (uint32_t)held;
		return SPARK_STATUS_OK;
	}

	template<typename CopyRowsFn>
	SparkStatus CopyPrefix(DsparkPagedSequence *sequence, const DsparkPagedSequence *donor, const SparkDraftServerRequest *request,
		CopyRowsFn copy_rows)
	{
		const uint64_t donor_capacity = (uint64_t)donor->page_count * PAGE_POSITIONS;
		const uint64_t shared = request->position < donor_capacity ? request->position : donor_capacity;
		uint64_t prefix = 0u, at = 0u;
		SparkStatus status;
		while ( prefix < shared && donor->tokens[prefix] == request->committed_token_ids[prefix] )
			prefix++;
		if ( prefix == 0u )
			return SPARK_STATUS_OK;
		status = EnsurePages(sequence, prefix, donor);
		while ( status == SPARK_STATUS_OK && at < prefix )
		{
			uint64_t stop;
			if ( donor->held[at] == 0u || sequence->held[at] != 0u )
			{
				at++;
				continue;
			}
			stop = at + 1u;
			while ( stop < prefix && donor->held[stop] != 0u && sequence->held[stop] == 0u )
				stop++;
			status = copy_rows(sequence, donor, at, stop);
			if ( status == SPARK_STATUS_OK )
			{
				memset(sequence->held + at, 1, (size_t)(stop - at));
				memcpy(sequence->tokens + at, donor->tokens + at, (size_t)(stop - at) * sizeof(uint32_t));
			}
			at = stop;
		}
		return status;
	}

	template<typename CopyRowsFn>
	SparkStatus FillFromDonors(DsparkPagedSequence *sequence, const SparkDraftServerRequest *request, CopyRowsFn copy_rows)
	{
		uint64_t tried = 0u;
		SparkStatus status = SPARK_STATUS_OK;
		while ( status == SPARK_STATUS_OK && Complete(sequence, request->position) == 0u )
		{
			DsparkPagedSequence *donor = 0;
			uint32_t chosen = SEQUENCES_MAX;
			for ( uint32_t index = 0u; index < SEQUENCES_MAX; index++ )
			{
				DsparkPagedSequence *other = &sequences[index];
				if ( other->pages != 0 && other != sequence && ((tried >> index) & 1u) == 0u && (donor == 0 || other->last_used > donor->last_used) )
				{
					donor = other;
					chosen = index;
				}
			}
			if ( donor == 0 )
				break;
			tried |= UINT64_C(1) << chosen;
			status = CopyPrefix(sequence, donor, request, copy_rows);
		}
		return status;
	}

	template<uint32_t TAPS, uint32_t HIDDEN, uint32_t INGEST_ROWS, typename BeforeChunksFn, typename WriteChunkFn>
	SparkStatus IngestFrame(const SparkTapStreamReaderFrame *frame, uint16_t *host_taps, uint32_t *host_positions, uint16_t *taps,
		uint32_t *positions, SparkGpuStream stream, uint64_t *frames_counter, uint64_t *rows_counter, BeforeChunksFn before_chunks,
		WriteChunkFn write_chunk)
	{
		constexpr uint32_t TAP_WIDTH = TAPS * HIDDEN;
		const uint32_t rows = frame->header.row_count;
		const uint64_t first = frame->header.first_position, end = first + rows;
		const uint64_t hidden = frame->hello != 0 ? frame->hello->hidden : 0u;
		DsparkPagedSequence *sequence;
		SparkStatus status;
		if ( rows == 0u )
			return SPARK_STATUS_OK;
		if ( frame->required_count != TAPS || hidden != HIDDEN || frame->rows == 0 )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		for ( uint32_t tap = 0u; tap < TAPS; tap++ )
			if ( ((uint64_t)frame->required_index[tap] + 1u) * HIDDEN > frame->row_elements )
				SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		if ( end > (uint64_t)PAGES_MAX * PAGE_POSITIONS )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		sequence = Find(frame->header.sequence_id, 0u);
		if ( sequence != 0 && first != 0u &&
			(sequence->control_generation != frame->header.control_generation || sequence->request_id != frame->header.request_id) )
		{
			sequence->retired = 1u;
			sequence = 0;
		}
		if ( sequence == 0 )
		{
			sequence = Find(frame->header.sequence_id, 1u);
			if ( sequence == 0 )
				SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
			sequence->control_generation = frame->header.control_generation;
			sequence->request_id = frame->header.request_id;
		}
		status = EnsurePages(sequence, end, 0);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( first == 0u )
		{
			Forget(sequence, 0u);
			sequence->control_generation = frame->header.control_generation;
			sequence->request_id = frame->header.request_id;
		}
		Forget(sequence, first);
		status = before_chunks(sequence);
		for ( uint32_t done = 0u; done < rows && status == SPARK_STATUS_OK; done += INGEST_ROWS )
		{
			const uint32_t part = rows - done < INGEST_ROWS ? rows - done : INGEST_ROWS;
			for ( uint32_t row = 0u; row < part; row++ )
			{
				const uint16_t *source = frame->rows + (uint64_t)(done + row) * frame->row_elements;
				for ( uint32_t tap = 0u; tap < TAPS; tap++ )
					memcpy(host_taps + (uint64_t)row * TAP_WIDTH + (uint64_t)tap * HIDDEN,
						source + (uint64_t)frame->required_index[tap] * HIDDEN, HIDDEN * sizeof(uint16_t));
				host_positions[row] = (uint32_t)(first + done + row);
			}
			if ( SparkGpuMemcpyAsync(taps, host_taps, (size_t)part * TAP_WIDTH * sizeof(uint16_t), SPARK_GPU_COPY_HOST_TO_DEVICE, stream) !=
					SPARK_GPU_SUCCESS ||
				SparkGpuMemcpyAsync(positions, host_positions, (size_t)part * sizeof(uint32_t), SPARK_GPU_COPY_HOST_TO_DEVICE, stream) !=
					SPARK_GPU_SUCCESS )
			{
				(void)SparkGpuStreamSynchronize(stream);
				SPARK_FAIL(SPARK_STATUS_IO_ERROR);
			}
			status = write_chunk(sequence, first + done, part);
		}
		if ( status == SPARK_STATUS_OK )
		{
			memset(sequence->held + first, 1, rows);
			if ( frame->tokens != 0 )
				memcpy(sequence->tokens + first, frame->tokens, (size_t)rows * sizeof(uint32_t));
			sequence->held_end = end;
		}
		(*frames_counter)++;
		(*rows_counter) += rows;
		return status;
	}
};

template<uint32_t BLOCK, uint32_t BRANCHES, uint32_t BRANCH_LEN>
struct DsparkMarkovPicks
{
	uint32_t tokens[BLOCK];
	float scores[BLOCK];
	uint32_t alt_tokens[BLOCK];
	float alt_scores[BLOCK];
	uint32_t branch_fork[BRANCHES];
	uint32_t branch_len[BRANCHES];
	uint32_t branch_tokens[BRANCHES][BRANCH_LEN];
	float branch_scores[BRANCHES][BRANCH_LEN];
};

template<uint32_t DRAFT_VOCAB, uint32_t MARKOV, uint32_t CANDIDATES, typename DraftT>
__global__ __launch_bounds__(DSPARK_THREADS, 2)
void DsparkMarkovRowKernel(uint32_t step, const DraftT *__restrict__ draft, const uint32_t *__restrict__ tokens,
	const uint32_t *__restrict__ candidates, const uint16_t *__restrict__ logits, const uint16_t *__restrict__ markov_previous,
	const uint16_t *__restrict__ markov_next, float *__restrict__ row)
{
	LmDependentWait();
	__shared__ float vector[MARKOV];
	const uint32_t previous = step == 0u ? draft->anchor : tokens[step - 1u];
	const uint32_t warp = threadIdx.x / DSPARK_WARP, lane = threadIdx.x % DSPARK_WARP;
	for ( uint32_t index = threadIdx.x; index < MARKOV; index += DSPARK_THREADS )
		vector[index] = LmBf16ToFloat(__ldg(markov_previous + (uint64_t)previous * MARKOV + index));
	__syncthreads();
	for ( uint32_t slot = blockIdx.x * DSPARK_WARPS + warp; slot < CANDIDATES; slot += gridDim.x * DSPARK_WARPS )
	{
		const uint32_t candidate = candidates[step * CANDIDATES + slot];
		const uint16_t *edge = markov_next + (uint64_t)candidate * MARKOV;
		float sum = 0.0f;
		for ( uint32_t element = lane; element < MARKOV; element += DSPARK_WARP )
			sum += LmBf16ToFloat(__ldg(edge + element)) * vector[element];
		sum = DsparkWarpSum(sum);
		if ( lane == 0u )
			row[slot] = LmBf16ToFloat(__ldg(logits + (uint64_t)step * DRAFT_VOCAB + candidate)) + sum;
	}
	LmDependentRelease();
}

template<uint32_t HIDDEN, uint32_t VOCAB, uint32_t MARKOV, uint32_t CANDIDATES, uint32_t SAMPLED, bool HAS_D2T, typename DraftT, typename PicksT>
__global__ __launch_bounds__(DSPARK_PICK_THREADS, 1)
void DsparkMarkovPickKernel(uint32_t step, const DraftT *__restrict__ draft, const float *__restrict__ row, const uint32_t *__restrict__ candidates,
	const int64_t *__restrict__ d2t, const uint16_t *__restrict__ y, const uint16_t *__restrict__ markov_previous,
	const uint16_t *__restrict__ confidence_weight, const uint16_t *__restrict__ confidence_bias, PicksT *__restrict__ picks)
{
	LmDependentWait();
	constexpr uint32_t CONFIDENCE = HIDDEN + MARKOV;
	__shared__ unsigned long long keys[CANDIDATES];
	__shared__ unsigned long long best[DSPARK_PICK_THREADS / DSPARK_WARP];
	__shared__ unsigned long long second_best[DSPARK_PICK_THREADS / DSPARK_WARP];
	__shared__ float reduction[DSPARK_PICK_THREADS / DSPARK_WARP];
	__shared__ uint32_t chosen;
	__shared__ uint32_t alt_chosen;
	__shared__ float peak_scaled;
	const uint32_t anchor = draft->anchor, position = draft->position, top_k = draft->top_k;
	const unsigned long long seed = (unsigned long long)draft->seed;
	const float inverse_temperature = draft->inverse_temperature, nucleus = draft->nucleus;
	const uint32_t previous = step == 0u ? anchor : picks->tokens[step - 1u];
	const uint32_t warp = threadIdx.x / DSPARK_WARP, lane = threadIdx.x % DSPARK_WARP;
	const float scale = inverse_temperature > 0.0f ? inverse_temperature : 1.0f;
	float dot = 0.0f, denom = 0.0f;
	if ( inverse_temperature > 0.0f )
	{
		for ( uint32_t index = threadIdx.x; index < CANDIDATES; index += DSPARK_PICK_THREADS )
			keys[index] = DsparkRowKey(row, index);
		__syncthreads();
		for ( uint32_t size = 2u; size <= CANDIDATES; size <<= 1u )
			for ( uint32_t stride = size >> 1u; stride > 0u; stride >>= 1u )
			{
				for ( uint32_t index = threadIdx.x; index < CANDIDATES; index += DSPARK_PICK_THREADS )
				{
					const uint32_t other = index ^ stride;
					if ( other > index )
					{
						const unsigned long long a = keys[index], b = keys[other];
						if ( ((index & size) == 0u) ? a < b : a > b )
						{
							keys[index] = b;
							keys[other] = a;
						}
					}
				}
				__syncthreads();
			}
		if ( warp == 0u )
		{
			const uint32_t slot = DsparkKeySlot(keys[lane]);
			const uint32_t token = DsparkMapDraftToken<VOCAB, HAS_D2T>(d2t, candidates[step * CANDIDATES + slot]);
			const float scaled = row[slot] * inverse_temperature;
			float peak = DsparkWarpMax(scaled), probability = expf(scaled - peak), total = DsparkWarpSum(probability), running, keyed;
			uint32_t winner, runner;
			unsigned long long key;
			probability /= total;
			running = probability;
			for ( uint32_t offset = 1u; offset < SAMPLED; offset <<= 1u )
			{
				const float earlier = __shfl_up_sync(0xffffffffu, running, offset);
				if ( lane >= offset )
					running += earlier;
			}
			keyed = (((nucleus <= 0.0f) || (running - probability < nucleus)) && (top_k == 0u || lane < top_k)) || lane == 0u
				? scaled + SparkMathGumbelNoise(seed, position + step, token) : -INFINITY;
			key = DsparkWarpKey(((unsigned long long)SparkMathSampleOrdered(keyed) << 32u) | (unsigned long long)(0xffffffffu - lane));
			winner = 0xffffffffu - (uint32_t)key;
			runner = winner == 0u ? 1u : 0u;
			if ( lane == winner )
				chosen = slot;
			if ( lane == runner )
				alt_chosen = slot;
			if ( lane == 0u )
				peak_scaled = peak;
		}
	}
	else
	{
		unsigned long long first = 0ull, second = 0ull;
		for ( uint32_t index = threadIdx.x; index < CANDIDATES; index += DSPARK_PICK_THREADS )
			DsparkTop2Insert(DsparkRowKey(row, index), &first, &second);
		for ( uint32_t mask = DSPARK_WARP / 2u; mask > 0u; mask >>= 1u )
		{
			const unsigned long long other_first = __shfl_xor_sync(0xffffffffu, first, mask);
			const unsigned long long other_second = __shfl_xor_sync(0xffffffffu, second, mask);
			DsparkTop2Insert(other_first, &first, &second);
			DsparkTop2Insert(other_second, &first, &second);
		}
		if ( lane == 0u )
		{
			best[warp] = first;
			second_best[warp] = second;
		}
		__syncthreads();
		if ( threadIdx.x == 0u )
		{
			first = best[0];
			second = second_best[0];
			for ( uint32_t index = 1u; index < DSPARK_PICK_THREADS / DSPARK_WARP; index++ )
			{
				DsparkTop2Insert(best[index], &first, &second);
				DsparkTop2Insert(second_best[index], &first, &second);
			}
			chosen = DsparkKeySlot(first);
			alt_chosen = DsparkKeySlot(second);
			peak_scaled = row[chosen];
		}
	}
	__syncthreads();
	for ( uint32_t index = threadIdx.x; index < CANDIDATES; index += DSPARK_PICK_THREADS )
		denom += expf(row[index] * scale - peak_scaled);
	denom = LmBlockSum<DSPARK_PICK_THREADS>(denom, reduction);
	for ( uint32_t index = threadIdx.x; index < CONFIDENCE; index += DSPARK_PICK_THREADS )
	{
		const float feature = index < HIDDEN ? LmBf16ToFloat(__ldg(y + (uint64_t)step * HIDDEN + index))
			: LmBf16ToFloat(__ldg(markov_previous + (uint64_t)previous * MARKOV + index - HIDDEN));
		dot += LmBf16ToFloat(__ldg(confidence_weight + index)) * feature;
	}
	dot = LmBlockSum<DSPARK_PICK_THREADS>(dot, reduction);
	if ( threadIdx.x == 0u )
	{
		picks->tokens[step] = DsparkMapDraftToken<VOCAB, HAS_D2T>(d2t, candidates[step * CANDIDATES + chosen]);
		picks->scores[step] = 1.0f / (1.0f + expf(-(dot + LmBf16ToFloat(confidence_bias[0]))));
		picks->alt_tokens[step] = DsparkMapDraftToken<VOCAB, HAS_D2T>(d2t, candidates[step * CANDIDATES + alt_chosen]);
		picks->alt_scores[step] = denom > 0.0f ? expf(row[alt_chosen] * scale - peak_scaled) / denom : 0.0f;
	}
	LmDependentRelease();
}

template<uint32_t BLOCK, uint32_t BRANCHES, typename DraftT, typename PicksT>
__global__ __launch_bounds__(DSPARK_WARP, 1)
void DsparkBranchSelectKernel(PicksT *__restrict__ picks, const DraftT *__restrict__ draft, float min_score)
{
	LmDependentWait();
	uint32_t used = 0u;
	if ( threadIdx.x == 0u )
	{
		const uint32_t drafts = draft->drafts;
		for ( uint32_t branch = 0u; branch < BRANCHES; branch++ )
		{
			picks->branch_fork[branch] = BLOCK;
			picks->branch_len[branch] = 0u;
		}
		for ( uint32_t step = 0u; step + 1u < drafts; step++ )
		{
			const float score = picks->alt_scores[step];
			uint32_t pos = used;
			if ( picks->alt_tokens[step] == picks->tokens[step] || !(score >= min_score) )
				continue;
			while ( pos > 0u && score > picks->branch_scores[pos - 1u][0] )
			{
				if ( pos < BRANCHES )
				{
					picks->branch_fork[pos] = picks->branch_fork[pos - 1u];
					picks->branch_len[pos] = picks->branch_len[pos - 1u];
					picks->branch_tokens[pos][0] = picks->branch_tokens[pos - 1u][0];
					picks->branch_scores[pos][0] = picks->branch_scores[pos - 1u][0];
				}
				pos--;
			}
			if ( pos < BRANCHES )
			{
				picks->branch_fork[pos] = step;
				picks->branch_len[pos] = 1u;
				picks->branch_tokens[pos][0] = picks->alt_tokens[step];
				picks->branch_scores[pos][0] = score;
				if ( used < BRANCHES )
					used++;
			}
		}
	}
	LmDependentRelease();
}

template<uint32_t DRAFT_VOCAB, uint32_t MARKOV, uint32_t CANDIDATES, typename DraftT, typename PicksT>
__global__ __launch_bounds__(DSPARK_THREADS, 2)
void DsparkBranchRowKernel(uint32_t offset, const DraftT *__restrict__ draft, const PicksT *__restrict__ picks,
	const uint32_t *__restrict__ candidates, const uint16_t *__restrict__ logits, const uint16_t *__restrict__ markov_previous,
	const uint16_t *__restrict__ markov_next, float *__restrict__ rows)
{
	LmDependentWait();
	__shared__ float vector[MARKOV];
	const uint32_t drafts = draft->drafts;
	const uint32_t branch = blockIdx.y, fork = picks->branch_fork[branch], step = fork + offset;
	const uint32_t warp = threadIdx.x / DSPARK_WARP, lane = threadIdx.x % DSPARK_WARP;
	if ( fork < drafts && step < drafts )
	{
		const uint32_t previous = picks->branch_tokens[branch][offset - 1u];
		float *row = rows + (uint64_t)branch * CANDIDATES;
		for ( uint32_t index = threadIdx.x; index < MARKOV; index += DSPARK_THREADS )
			vector[index] = LmBf16ToFloat(__ldg(markov_previous + (uint64_t)previous * MARKOV + index));
		__syncthreads();
		for ( uint32_t slot = blockIdx.x * DSPARK_WARPS + warp; slot < CANDIDATES; slot += gridDim.x * DSPARK_WARPS )
		{
			const uint32_t candidate = candidates[step * CANDIDATES + slot];
			const uint16_t *edge = markov_next + (uint64_t)candidate * MARKOV;
			float sum = 0.0f;
			for ( uint32_t element = lane; element < MARKOV; element += DSPARK_WARP )
				sum += LmBf16ToFloat(__ldg(edge + element)) * vector[element];
			sum = DsparkWarpSum(sum);
			if ( lane == 0u )
				row[slot] = LmBf16ToFloat(__ldg(logits + (uint64_t)step * DRAFT_VOCAB + candidate)) + sum;
		}
	}
	LmDependentRelease();
}

template<uint32_t VOCAB, uint32_t CANDIDATES, bool HAS_D2T, typename DraftT, typename PicksT>
__global__ __launch_bounds__(DSPARK_PICK_THREADS, 1)
void DsparkBranchPickKernel(uint32_t offset, const DraftT *__restrict__ draft, const float *__restrict__ rows,
	const uint32_t *__restrict__ candidates, const int64_t *__restrict__ d2t, PicksT *__restrict__ picks)
{
	LmDependentWait();
	__shared__ unsigned long long best[DSPARK_PICK_THREADS / DSPARK_WARP];
	__shared__ float reduction[DSPARK_PICK_THREADS / DSPARK_WARP];
	__shared__ uint32_t chosen;
	__shared__ float peak_scaled;
	const uint32_t drafts = draft->drafts;
	const float scale = draft->scale;
	const uint32_t branch = blockIdx.x, fork = picks->branch_fork[branch], step = fork + offset;
	const uint32_t warp = threadIdx.x / DSPARK_WARP, lane = threadIdx.x % DSPARK_WARP;
	if ( fork < drafts && step < drafts )
	{
		const float *row = rows + (uint64_t)branch * CANDIDATES;
		unsigned long long first = 0ull;
		float denom = 0.0f;
		for ( uint32_t index = threadIdx.x; index < CANDIDATES; index += DSPARK_PICK_THREADS )
		{
			const unsigned long long key = DsparkRowKey(row, index);
			if ( key > first )
				first = key;
		}
		first = DsparkWarpKey(first);
		if ( lane == 0u )
			best[warp] = first;
		__syncthreads();
		if ( threadIdx.x == 0u )
		{
			first = best[0];
			for ( uint32_t index = 1u; index < DSPARK_PICK_THREADS / DSPARK_WARP; index++ )
				if ( best[index] > first )
					first = best[index];
			chosen = DsparkKeySlot(first);
			peak_scaled = row[chosen] * scale;
		}
		__syncthreads();
		for ( uint32_t index = threadIdx.x; index < CANDIDATES; index += DSPARK_PICK_THREADS )
			denom += expf(row[index] * scale - peak_scaled);
		denom = LmBlockSum<DSPARK_PICK_THREADS>(denom, reduction);
		if ( threadIdx.x == 0u )
		{
			picks->branch_tokens[branch][offset] = DsparkMapDraftToken<VOCAB, HAS_D2T>(d2t, candidates[step * CANDIDATES + chosen]);
			picks->branch_scores[branch][offset] = denom > 0.0f ? 1.0f / denom : 0.0f;
			picks->branch_len[branch] = offset + 1u;
		}
	}
	LmDependentRelease();
}

template<uint32_t BLOCK, uint32_t HIDDEN, uint32_t VOCAB, uint32_t DRAFT_VOCAB, uint32_t MARKOV, uint32_t CANDIDATES, uint32_t SAMPLED,
	uint32_t BRANCHES, bool HAS_D2T, typename DraftT, typename PicksT>
static inline SparkStatus DsparkLaunchMarkovHeads(const uint16_t *head, const uint16_t *y, uint16_t *logits, SparkDrafterTopkBf16Workspace *topk_ws,
	uint32_t *candidates, float *markov_row, const uint16_t *markov_previous, const uint16_t *markov_next, const int64_t *d2t,
	const uint16_t *confidence_weight, const uint16_t *confidence_bias, const DraftT *draft, PicksT *picks, PicksT *host_picks,
	uint32_t branch_steps, uint32_t tree_enabled, uint32_t multiprocessors, SparkGpuStream stream)
{
	SparkStatus status = DsparkGemm(head, y, logits, BLOCK, HIDDEN, DRAFT_VOCAB, multiprocessors, stream);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( SparkDrafterTopkBf16Launch(logits, BLOCK, DRAFT_VOCAB, CANDIDATES, topk_ws, candidates, stream) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	for ( uint32_t step = 0u; step < BLOCK; step++ )
	{
		if ( LM_LAUNCH_DEPENDENT((DsparkMarkovRowKernel<DRAFT_VOCAB, MARKOV, CANDIDATES, DraftT>), CANDIDATES / DSPARK_WARPS, DSPARK_THREADS, 0,
				stream, step, draft, picks->tokens, candidates, logits, markov_previous, markov_next, markov_row) != SPARK_GPU_SUCCESS ||
			LM_LAUNCH_DEPENDENT((DsparkMarkovPickKernel<HIDDEN, VOCAB, MARKOV, CANDIDATES, SAMPLED, HAS_D2T, DraftT, PicksT>), 1u,
				DSPARK_PICK_THREADS, 0, stream, step, draft, markov_row, candidates, d2t, y, markov_previous, confidence_weight, confidence_bias,
				picks) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	}
	if ( tree_enabled != 0u )
	{
		if ( LM_LAUNCH_DEPENDENT((DsparkBranchSelectKernel<BLOCK, BRANCHES, DraftT, PicksT>), 1u, DSPARK_WARP, 0, stream, picks, draft,
				SPARK_DRAFT_TREE_DEFAULT_MIN_SCORE) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		for ( uint32_t offset = 1u; offset < branch_steps; offset++ )
		{
			if ( LM_LAUNCH_DEPENDENT((DsparkBranchRowKernel<DRAFT_VOCAB, MARKOV, CANDIDATES, DraftT, PicksT>),
					dim3(CANDIDATES / DSPARK_WARPS, BRANCHES), DSPARK_THREADS, 0, stream, offset, draft, picks, candidates, logits, markov_previous,
					markov_next, markov_row) != SPARK_GPU_SUCCESS ||
				LM_LAUNCH_DEPENDENT((DsparkBranchPickKernel<VOCAB, CANDIDATES, HAS_D2T, DraftT, PicksT>), BRANCHES, DSPARK_PICK_THREADS, 0, stream,
					offset, draft, markov_row, candidates, d2t, picks) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
	}
	if ( SparkGpuMemcpyAsync(host_picks, picks, sizeof(PicksT), SPARK_GPU_COPY_DEVICE_TO_HOST, stream) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	if ( SparkGpuGetLastError() != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return SPARK_STATUS_OK;
}

template<uint32_t BLOCK, uint32_t BRANCHES, typename PicksT>
static inline uint32_t DsparkPackMarkovTree(uint32_t anchor_token_id, uint32_t source, const PicksT *picks, uint32_t tree, uint32_t drafts,
	uint32_t requested, uint32_t limit, SparkDraftBridgeNode *nodes)
{
	SparkDraftTreeBranch branches[BLOCK];
	uint32_t branch_count = 0u;
	if ( tree != 0u )
		for ( uint32_t step = 0u; step < drafts; step++ )
		{
			uint32_t matched = BRANCHES;
			for ( uint32_t b = 0u; b < BRANCHES; b++ )
				if ( picks->branch_len[b] > 0u && picks->branch_fork[b] == step )
					matched = b;
			branches[branch_count].fork_step = step;
			if ( matched < BRANCHES )
			{
				const uint32_t len = picks->branch_len[matched];
				branches[branch_count].length = len;
				for ( uint32_t k = 0u; k < len; k++ )
				{
					branches[branch_count].tokens[k] = picks->branch_tokens[matched][k];
					branches[branch_count].scores[k] = picks->branch_scores[matched][k];
				}
			}
			else
			{
				branches[branch_count].length = 1u;
				branches[branch_count].tokens[0] = picks->alt_tokens[step];
				branches[branch_count].scores[0] = picks->alt_scores[step];
			}
			branch_count++;
		}
	return SparkDraftTreePackBranches(anchor_token_id, source, picks->tokens, picks->scores, branches, branch_count, drafts, requested, limit,
		SPARK_DRAFT_TREE_DEFAULT_MIN_SCORE, nodes);
}

template<uint32_t DRAFTS, uint32_t BRANCH_LEN, typename PicksT>
static inline uint32_t DsparkPackGridTree(uint32_t anchor_token_id, uint32_t source, const PicksT *picks, uint32_t tree, uint32_t drafts,
	uint32_t requested, uint32_t limit, SparkDraftBridgeNode *nodes)
{
	SparkDraftTreeBranch branches[DRAFTS];
	uint32_t branch_count = 0u;
	if ( tree != 0u )
		for ( uint32_t step = 0u; step < drafts; step++ )
		{
			const uint32_t len = drafts - step < BRANCH_LEN ? drafts - step : BRANCH_LEN;
			branches[branch_count].fork_step = step;
			branches[branch_count].length = len;
			for ( uint32_t k = 0u; k < len; k++ )
			{
				branches[branch_count].tokens[k] = picks->branch_tokens[step][k];
				branches[branch_count].scores[k] = picks->branch_scores[step][k];
			}
			branch_count++;
		}
	return SparkDraftTreePackBranches(anchor_token_id, source, picks->tokens, picks->scores, branches, branch_count, drafts, requested, limit,
		SPARK_DRAFT_TREE_DEFAULT_MIN_SCORE, nodes);
}

#endif
