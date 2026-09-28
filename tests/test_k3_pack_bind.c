#include "sparkpipe/spark_k3_resident_decode_stage_module.h"
#include "sparkpipe/spark_status.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define TEST_K3_MANIFEST_BYTES 65536u
#define TEST_K3_TENSOR_BYTES 64u

typedef struct TestK3PackSpec
{
	uint32_t magic;
	uint32_t version;
	uint32_t first_layer;
	uint32_t layers;
	uint32_t total_layers;
	uint32_t truncate_payload;
	uint32_t omit_layer;
	const char *omit_tensor;
} TestK3PackSpec;

static const char *const TestK3Every[] =
{
	"attn_norm_weight", "attnres_attn_weight", "mlp_norm_weight",
	"attnres_mlp_weight"
};

static const char *const TestK3Kda[] =
{
	"kda_qkv_beta_weight", "kda_q_conv_weight", "kda_k_conv_weight",
	"kda_v_conv_weight", "kda_decay_down_weight", "kda_decay_up_weight",
	"kda_decay_bias", "kda_head_log_scale", "kda_gate_weight",
	"kda_out_norm_weight", "kda_out_weight"
};

static const char *const TestK3Mla[] =
{
	"mla_q_down_weight", "mla_q_norm_weight", "mla_q_up_weight",
	"mla_kv_a_weight", "mla_kv_a_norm_weight", "mla_kv_b_value_weight",
	"mla_gate_weight", "mla_out_weight"
};

static const char *const TestK3Moe[] =
{
	"router_weight", "router_bias", "expert_w1_weight", "expert_w2_weight",
	"shared_w1_weight", "shared_w2_weight", "routed_down_weight",
	"routed_up_weight", "routed_norm_weight"
};

static const char *const TestK3Dense[] =
{
	"dense_gate_up_weight", "dense_down_weight"
};

static int32_t TestK3Check(int32_t condition, const char *what)
{
	printf("%s: %s\n", condition ? "PASS" : "FAIL", what);
	return(condition ? 0 : 1);
}

static void TestK3WriteLe(uint8_t *bytes, uint64_t value, uint32_t count)
{
	uint32_t index;
	for (index = 0u; index < count; index++)
		bytes[index] = (uint8_t)(value >> (8u * index));
}

static uint32_t TestK3Append(char *manifest, uint32_t used, const char *format, ...)
{
	va_list arguments;
	int written;
	if ( used >= TEST_K3_MANIFEST_BYTES )
		return(TEST_K3_MANIFEST_BYTES);
	va_start(arguments, format);
	written = vsnprintf(manifest + used, TEST_K3_MANIFEST_BYTES - used, format, arguments);
	va_end(arguments);
	if ( written < 0 || (uint32_t)written >= TEST_K3_MANIFEST_BYTES - used )
		return(TEST_K3_MANIFEST_BYTES);
	return(used + (uint32_t)written);
}

static uint32_t TestK3AppendGroup(char *manifest, uint32_t used, uint32_t *count,
	const TestK3PackSpec *spec, uint32_t layer, const char *const *names,
	uint32_t name_count)
{
	uint32_t index;
	for (index = 0u; index < name_count && used < TEST_K3_MANIFEST_BYTES; index++)
	{
		const char *kind = strncmp(names[index], "expert_", 7u) == 0 ?
			"mxfp4_ws_interleaved_v1" : "bf16";
		const char *interleave = strcmp(names[index], "expert_w1_weight") == 0 ?
			",\"interleave\":{\"tile_k\":32}" : "";
		if ( spec->omit_tensor != 0 && layer == spec->omit_layer &&
			strcmp(names[index], spec->omit_tensor) == 0 )
			continue;
		used = TestK3Append(manifest, used,
			"%s\"model.layers.%u.%s\":{\"offset\":%u,\"bytes\":%u,"
			"\"kind\":\"%s\",\"shape\":[4,8]%s}",
			*count == 0u ? "" : ",", layer, names[index],
			*count * TEST_K3_TENSOR_BYTES, TEST_K3_TENSOR_BYTES, kind, interleave);
		*count += 1u;
	}
	return(used);
}

static int32_t TestK3WritePack(const char *path, const TestK3PackSpec *spec)
{
	static char manifest[TEST_K3_MANIFEST_BYTES];
	uint8_t header[16];
	uint8_t zero[TEST_K3_TENSOR_BYTES];
	uint32_t used, count = 0u, layer, index;
	uint64_t payload_base;
	FILE *file;
	int32_t ok;
	used = TestK3Append(manifest, 0u,
		"{\"config\":{\"hidden\":%u,\"layers\":%u,\"first_layer\":%u,"
		"\"total_layers\":%u,\"experts\":%u,\"top_k\":%u,\"latent\":%u,"
		"\"intermediate\":%u,\"group\":32,\"vocab\":%u,\"kda_heads\":%u,"
		"\"kda_head\":%u,\"heads\":%u,\"kv_lora\":%u,\"rope\":%u,\"v_head\":%u,"
		"\"nope\":%u,\"shared\":%u,\"q_lora\":%u},\"tensors\":{",
		SPARK_K3_MODEL_HIDDEN_DIMENSION, spec->layers, spec->first_layer,
		spec->total_layers, SPARK_K3_MODEL_MOE_EXPERT_COUNT,
		SPARK_K3_MODEL_MOE_TOP_K,
		SPARK_K3_MODEL_MOE_ROUTED_EXPERT_HIDDEN_DIMENSION,
		SPARK_K3_MODEL_MOE_INTERMEDIATE_DIMENSION,
		SPARK_K3_MODEL_OUTPUT_VOCAB_COUNT, SPARK_K3_MODEL_KDA_HEAD_COUNT,
		SPARK_K3_MODEL_KDA_HEAD_KEY_DIMENSION, SPARK_K3_MODEL_MLA_HEAD_COUNT,
		SPARK_K3_MODEL_MLA_LATENT_DIMENSION, SPARK_K3_MODEL_MLA_ROPE_DIMENSION,
		SPARK_K3_MODEL_MLA_VALUE_HEAD_DIMENSION,
		SPARK_K3_MODEL_MLA_QK_NOPE_HEAD_DIMENSION,
		SPARK_K3_MODEL_MOE_SHARED_EXPERT_COUNT,
		SPARK_K3_MODEL_MLA_QUERY_A_DIMENSION);
	for (layer = spec->first_layer; layer < spec->first_layer + spec->layers; layer++)
	{
		used = TestK3AppendGroup(manifest, used, &count, spec, layer, TestK3Every,
			sizeof(TestK3Every) / sizeof(TestK3Every[0]));
		if ( SPARK_K3_MODEL_LAYER_IS_MLA(layer) )
			used = TestK3AppendGroup(manifest, used, &count, spec, layer, TestK3Mla,
				sizeof(TestK3Mla) / sizeof(TestK3Mla[0]));
		else
			used = TestK3AppendGroup(manifest, used, &count, spec, layer, TestK3Kda,
				sizeof(TestK3Kda) / sizeof(TestK3Kda[0]));
		if ( layer == 0u )
			used = TestK3AppendGroup(manifest, used, &count, spec, layer, TestK3Dense,
				sizeof(TestK3Dense) / sizeof(TestK3Dense[0]));
		else
			used = TestK3AppendGroup(manifest, used, &count, spec, layer, TestK3Moe,
				sizeof(TestK3Moe) / sizeof(TestK3Moe[0]));
	}
	used = TestK3Append(manifest, used, "}}");
	if ( used >= TEST_K3_MANIFEST_BYTES )
		return(0);
	memset(header, 0, sizeof(header));
	memset(zero, 0, sizeof(zero));
	TestK3WriteLe(header, spec->magic, 4u);
	TestK3WriteLe(header + 4u, spec->version, 4u);
	TestK3WriteLe(header + 8u, used, 8u);
	payload_base = 16u + (uint64_t)used;
	payload_base += (SPARK_K3_PACK_ALIGNMENT - (payload_base % SPARK_K3_PACK_ALIGNMENT)) %
		SPARK_K3_PACK_ALIGNMENT;
	file = fopen(path, "wb");
	if ( file == 0 )
		return(0);
	ok = fwrite(header, 1u, sizeof(header), file) == sizeof(header) &&
		fwrite(manifest, 1u, used, file) == used;
	for (index = 16u + used; ok != 0 && index < payload_base; index++)
		ok = fwrite(zero, 1u, 1u, file) == 1u;
	for (index = 0u; ok != 0 && index + spec->truncate_payload < count; index++)
		ok = fwrite(zero, 1u, sizeof(zero), file) == sizeof(zero);
	return(fclose(file) == 0 && ok != 0);
}

static TestK3PackSpec TestK3Spec(uint32_t first_layer, uint32_t layers)
{
	TestK3PackSpec spec;
	memset(&spec, 0, sizeof(spec));
	spec.magic = SPARK_K3_PACK_MAGIC;
	spec.version = SPARK_K3_PACK_FORMAT_VERSION;
	spec.first_layer = first_layer;
	spec.layers = layers;
	spec.total_layers = SPARK_K3_MODEL_LAYER_COUNT;
	return(spec);
}

static int32_t TestK3PackAndBind(const char *path)
{
	TestK3PackSpec spec = TestK3Spec(0u, 4u);
	SparkK3Pack pack;
	SparkK3PackEntry entry;
	SparkK3BoundLayer bound;
	uint32_t tile_k = 0u;
	int32_t failures = 0;
	if ( TestK3Check(TestK3WritePack(path, &spec), "synthetic stage slice 0+4 written") != 0 )
		return(1);
	if ( TestK3Check(SparkK3PackOpen(path, &pack) == SPARK_STATUS_OK, "the V2 pack opens") != 0 )
		return(failures + 1);
	failures += TestK3Check(pack.version == SPARK_K3_PACK_FORMAT_VERSION &&
		pack.payload_base % SPARK_K3_PACK_ALIGNMENT == 0u &&
		pack.config.first_layer == 0u && pack.config.layers == 4u &&
		pack.config.total_layers == SPARK_K3_MODEL_LAYER_COUNT &&
		pack.config.hidden == SPARK_K3_MODEL_HIDDEN_DIMENSION &&
		pack.config.experts == SPARK_K3_MODEL_MOE_EXPERT_COUNT,
		"the manifest config and the aligned payload base are read");
	failures += TestK3Check(SparkK3PackLoadEntry(&pack, "model.layers.1.expert_w1_weight",
		&entry) == SPARK_STATUS_OK && entry.kind == SPARK_K3_PACK_KIND_MXFP4_WS_INTERLEAVED_V1 &&
		entry.bytes == TEST_K3_TENSOR_BYTES && entry.shape_count == 2u &&
		entry.shape[0] == 4u && entry.shape[1] == 8u &&
		entry.payload_offset % TEST_K3_TENSOR_BYTES == 0u,
		"an interleaved expert entry resolves with its kind, extent and shape");
	failures += TestK3Check(SparkK3PackLoadEntry(&pack, "model.layers.0.kda_gate_up_weight",
		&entry) == SPARK_STATUS_NOT_FOUND, "the retired low-rank gate name is absent");
	failures += TestK3Check(SparkK3PackLoadInterleaveTileK(&pack,
		"model.layers.1.expert_w1_weight", &tile_k) == SPARK_STATUS_OK && tile_k == 32u,
		"the interleave tile_k is read from the manifest");
	failures += TestK3Check(SparkK3PackLoadInterleaveTileK(&pack,
		"model.layers.1.router_weight", &tile_k) == SPARK_STATUS_VALIDATION_FAILED,
		"a tensor without an interleave block has no tile_k");
	failures += TestK3Check(SparkK3BindLayer(&pack, 0u, &bound) == SPARK_STATUS_OK &&
		bound.layer_is_gdn == 1u && bound.layer_is_dense == 1u &&
		bound.tensor_count == 17u && SparkK3BoundEntry(&bound, "kda_gate_weight") != 0 &&
		SparkK3BoundEntry(&bound, "dense_down_weight") != 0 &&
		SparkK3BoundEntry(&bound, "expert_w1_weight") == 0,
		"layer 0 binds 17 tensors as the dense KDA layer");
	failures += TestK3Check(SparkK3BindLayer(&pack, 1u, &bound) == SPARK_STATUS_OK &&
		bound.layer_is_gdn == 1u && bound.layer_is_dense == 0u &&
		bound.tensor_count == 24u && SparkK3BoundEntry(&bound, "expert_w1_weight") != 0 &&
		SparkK3BoundEntry(&bound, "dense_gate_up_weight") == 0,
		"layer 1 binds 24 tensors as a routed KDA layer");
	failures += TestK3Check(SparkK3BindLayer(&pack, 3u, &bound) == SPARK_STATUS_OK &&
		bound.layer_is_gdn == 0u && bound.layer_is_dense == 0u &&
		bound.tensor_count == 21u && SparkK3BoundEntry(&bound, "mla_gate_weight") != 0 &&
		SparkK3BoundEntry(&bound, "kda_gate_weight") == 0,
		"layer 3 binds 21 tensors as a routed MLA layer");
	failures += TestK3Check(SparkK3BindLayer(&pack, 4u, &bound) == SPARK_STATUS_VALIDATION_FAILED,
		"a layer outside the pack's slice has no tensors to bind");
	failures += TestK3Check(SparkK3BindLayer(&pack, SPARK_K3_MODEL_LAYER_COUNT, &bound) ==
		SPARK_STATUS_VALIDATION_FAILED, "a layer past the model is refused");
	SparkK3PackClose(&pack);
	failures += TestK3Check(pack.private_state == 0, "close releases the manifest");
	remove(path);
	return(failures);
}

static int32_t TestK3RefusedPacks(const char *path)
{
	TestK3PackSpec spec;
	SparkK3Pack pack;
	SparkK3PackEntry entry;
	SparkK3BoundLayer bound;
	int32_t failures = 0;
	spec = TestK3Spec(0u, 4u);
	spec.magic ^= 1u;
	failures += TestK3Check(TestK3WritePack(path, &spec) &&
		SparkK3PackOpen(path, &pack) == SPARK_STATUS_VALIDATION_FAILED &&
		pack.private_state == 0, "a wrong magic is refused");
	spec = TestK3Spec(0u, 4u);
	spec.version = 1u;
	failures += TestK3Check(TestK3WritePack(path, &spec) &&
		SparkK3PackOpen(path, &pack) == SPARK_STATUS_VALIDATION_FAILED,
		"a V1 pack is refused");
	spec = TestK3Spec(0u, 4u);
	spec.truncate_payload = 1u;
	failures += TestK3Check(TestK3WritePack(path, &spec) &&
		SparkK3PackOpen(path, &pack) == SPARK_STATUS_OK, "a truncated pack opens its manifest");
	if ( pack.private_state != 0 )
	{
		failures += TestK3Check(SparkK3PackLoadEntry(&pack, "model.layers.3.routed_norm_weight",
			&entry) == SPARK_STATUS_VALIDATION_FAILED &&
			SparkK3PackLoadEntry(&pack, "model.layers.0.attn_norm_weight", &entry) ==
			SPARK_STATUS_OK, "an entry past the end of the file is refused");
		SparkK3PackClose(&pack);
	}
	spec = TestK3Spec(0u, 4u);
	spec.omit_layer = 1u;
	spec.omit_tensor = "routed_up_weight";
	failures += TestK3Check(TestK3WritePack(path, &spec) &&
		SparkK3PackOpen(path, &pack) == SPARK_STATUS_OK, "a pack missing one tensor opens");
	if ( pack.private_state != 0 )
	{
		failures += TestK3Check(SparkK3BindLayer(&pack, 1u, &bound) ==
			SPARK_STATUS_VALIDATION_FAILED &&
			SparkK3BindLayer(&pack, 2u, &bound) == SPARK_STATUS_OK,
			"the layer missing a required tensor is refused, its neighbour binds");
		SparkK3PackClose(&pack);
	}
	remove(path);
	failures += TestK3Check(SparkK3PackOpen(path, &pack) == SPARK_STATUS_IO_ERROR,
		"a missing pack is an IO error");
	return(failures);
}

static int32_t TestK3ModuleSlices(const char *path)
{
	static SparkK3ModuleState state;
	TestK3PackSpec spec;
	SparkK3PoolSizing expected;
	int32_t failures = 0;
	spec = TestK3Spec(0u, 4u);
	failures += TestK3Check(TestK3WritePack(path, &spec), "synthetic stage slice 0+4 written");
	failures += TestK3Check(SparkK3ModuleInitialize(&state, path, 0u, 4u) == SPARK_STATUS_OK &&
		state.bound_count == 4u && state.first_layer == 0u && state.layer_count == 4u &&
		state.sizing.mla_layer_count == 1u && state.sizing.kda_layer_count == 3u &&
		state.bound[3].layer_is_gdn == 0u && state.bound[0].layer_is_dense == 1u,
		"module init binds the explicit slice 0+4 as 3 KDA + 1 MLA");
	SparkK3ModuleDestroy(&state);
	failures += TestK3Check(SparkK3ModuleInitialize(&state, path, SPARK_K3_MODULE_DERIVE_SLICE,
		SPARK_K3_MODULE_DERIVE_SLICE) == SPARK_STATUS_OK && state.first_layer == 0u &&
		state.layer_count == 4u && state.bound_count == 4u,
		"DERIVE_SLICE takes the slice from the pack manifest");
	SparkK3ModuleDestroy(&state);
	failures += TestK3Check(SparkK3ModuleInitialize(&state, path, 0u, 3u) ==
		SPARK_STATUS_VALIDATION_FAILED && state.pack.private_state == 0 &&
		state.bound_count == 0u, "a slice that disagrees with the pack is refused and released");
	failures += TestK3Check(SparkK3ModuleInitialize(&state, path, 1u, 4u) ==
		SPARK_STATUS_VALIDATION_FAILED, "a shifted slice is refused");
	spec = TestK3Spec(SPARK_K3_MODEL_LAYER_COUNT - 4u, 4u);
	SparkK3PoolSizingForSlice(spec.first_layer, spec.layers, &expected);
	failures += TestK3Check(TestK3WritePack(path, &spec) &&
		SparkK3ModuleInitialize(&state, path, SPARK_K3_MODULE_DERIVE_SLICE,
		SPARK_K3_MODULE_DERIVE_SLICE) == SPARK_STATUS_OK && state.bound_count == 4u &&
		state.sizing.mla_layer_count == 2u && state.sizing.kda_layer_count == 2u &&
		state.sizing.kda_slot_bytes_per_sequence == expected.kda_slot_bytes_per_sequence &&
		state.bound[2].layer_is_gdn == 0u && state.bound[3].layer_is_gdn == 0u &&
		state.bound[3].layer_index == SPARK_K3_MODEL_LAYER_COUNT - 1u,
		"the last slice binds layer 91 and the final layer 92 as MLA");
	SparkK3ModuleDestroy(&state);
	spec = TestK3Spec(0u, 4u);
	spec.total_layers = SPARK_K3_MODEL_LAYER_COUNT - 1u;
	failures += TestK3Check(TestK3WritePack(path, &spec) &&
		SparkK3ModuleInitialize(&state, path, SPARK_K3_MODULE_DERIVE_SLICE,
		SPARK_K3_MODULE_DERIVE_SLICE) == SPARK_STATUS_VALIDATION_FAILED,
		"a pack cut from a different layer count is refused");
	spec = TestK3Spec(0u, 4u);
	spec.omit_layer = 2u;
	spec.omit_tensor = "kda_out_weight";
	failures += TestK3Check(TestK3WritePack(path, &spec) &&
		SparkK3ModuleInitialize(&state, path, 0u, 4u) == SPARK_STATUS_VALIDATION_FAILED &&
		state.pack.private_state == 0 && state.bound_count == 0u,
		"module init refuses a slice with an unbindable layer and releases it");
	remove(path);
	return(failures);
}

static int32_t TestK3RealPack(const char *path)
{
	static SparkK3ModuleState state;
	static SparkK3ModuleState wrong_slice;
	SparkK3PoolSizing expected;
	uint32_t first_layer, layer_count;
	int32_t failures = 0;
	if ( TestK3Check(SparkK3ModuleInitialize(&state, path, SPARK_K3_MODULE_DERIVE_SLICE,
		SPARK_K3_MODULE_DERIVE_SLICE) == SPARK_STATUS_OK, "the rank pack binds its slice") != 0 )
		return(1);
	first_layer = state.first_layer;
	layer_count = state.layer_count;
	SparkK3PoolSizingForSlice(first_layer, layer_count, &expected);
	failures += TestK3Check(state.pack.config.hidden == SPARK_K3_MODEL_HIDDEN_DIMENSION &&
		state.pack.config.experts == SPARK_K3_MODEL_MOE_EXPERT_COUNT &&
		state.pack.config.kda_heads == SPARK_K3_MODEL_KDA_HEAD_COUNT,
		"the pack config is the model header's geometry");
	failures += TestK3Check(state.bound_count == layer_count &&
		state.sizing.mla_layer_count == expected.mla_layer_count &&
		state.sizing.kda_slot_bytes_per_sequence == expected.kda_slot_bytes_per_sequence,
		"every layer of the slice binds and the pool is sized for it");
	SparkK3ModuleDestroy(&state);
	failures += TestK3Check(SparkK3ModuleInitialize(&wrong_slice, path,
		first_layer, layer_count + 1u) == SPARK_STATUS_VALIDATION_FAILED,
		"a slice that disagrees with the pack is refused");
	SparkK3ModuleDestroy(&wrong_slice);
	return(failures);
}

int main(int argc, char **argv)
{
	char path[256];
	int32_t failures;
	if ( argc > 2 )
	{
		fprintf(stderr, "usage: test_k3_pack_bind [RANK_PACK]\n");
		return(2);
	}
	snprintf(path, sizeof(path), "/tmp/test_k3_pack_bind-%d.pack", (int)getpid());
	failures = TestK3PackAndBind(path);
	failures += TestK3RefusedPacks(path);
	failures += TestK3ModuleSlices(path);
	if ( argc == 2 )
		failures += TestK3RealPack(argv[1]);
	printf("test_k3_pack_bind: %d failures\n", (int)failures);
	return(failures != 0 ? 1 : 0);
}
