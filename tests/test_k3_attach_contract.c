#include "sparkpipe/spark_k3_resident_decode_stage_runner.h"
#include "sparkpipe/spark_k3_pack_load.h"
#include "sparkpipe/spark_status.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define ADAPTER_SO "build/libk3_serving_adapter.so"

static const char *const k3_layer0_tensors[] =
{
	"attn_norm_weight", "attnres_attn_weight",
	"kda_qkv_beta_weight", "kda_q_conv_weight", "kda_k_conv_weight",
	"kda_v_conv_weight", "kda_decay_down_weight", "kda_decay_up_weight",
	"kda_decay_bias", "kda_head_log_scale", "kda_gate_weight",
	"kda_out_norm_weight", "kda_out_weight",
	"mlp_norm_weight", "attnres_mlp_weight",
	"dense_gate_up_weight", "dense_down_weight"
};

typedef SparkStatus (*InitFunction)(SparkK3StageRunner *,
	const SparkK3StageRunnerConfiguration *);

static int expect(int condition, const char *what)
{
	printf("%s: %s\n", condition ? "PASS" : "FAIL", what);
	return(condition ? 0 : 1);
}

static uint32_t SparkK3ProbeWriteLe32(uint8_t *p, uint32_t v)
{
	p[0] = (uint8_t)v;
	p[1] = (uint8_t)(v >> 8u);
	p[2] = (uint8_t)(v >> 16u);
	p[3] = (uint8_t)(v >> 24u);
	return(v);
}

static long SparkK3ProbeWritePack(const char *path)
{
	char tensors[4096];
	char manifest[4608];
	uint8_t header[16];
	FILE *file;
	size_t manifest_bytes, written;
	uint64_t payload_base;
	uint32_t index;
	long file_bytes;
	tensors[0] = '\0';
	for ( index = 0u; index < sizeof(k3_layer0_tensors) / sizeof(k3_layer0_tensors[0]);
		index++ )
	{
		char entry[192];
		snprintf(entry, sizeof(entry),
			"%s\"model.layers.0.%s\":{\"offset\":%u,\"bytes\":8,"
			"\"kind\":\"bf16\",\"shape\":[1,1]}",
			index != 0u ? "," : "", k3_layer0_tensors[index],
			(uint32_t)(index * 8u));
		strcat(tensors, entry);
	}
	snprintf(manifest, sizeof(manifest),
		"{\"config\":{\"hidden\":64,\"layers\":1,\"first_layer\":0,"
		"\"total_layers\":93,\"experts\":1,\"top_k\":1,\"latent\":1,"
		"\"intermediate\":1,\"group\":1,\"vocab\":32,\"kda_heads\":1,"
		"\"kda_head\":1,\"heads\":1,\"kv_lora\":1,\"rope\":1,\"v_head\":1,"
		"\"nope\":1,\"shared\":1,\"q_lora\":1},\"tensors\":{%s}}",
		tensors);
	manifest_bytes = strlen(manifest);
	memset(header, 0, sizeof(header));
	SparkK3ProbeWriteLe32(header, 0x4B33504Bu);
	SparkK3ProbeWriteLe32(header + 4u, SPARK_K3_PACK_FORMAT_VERSION);
	SparkK3ProbeWriteLe32(header + 8u, (uint32_t)manifest_bytes);
	payload_base = 16ull + manifest_bytes;
	payload_base += (SPARK_K3_PACK_ALIGNMENT -
		(payload_base % SPARK_K3_PACK_ALIGNMENT)) % SPARK_K3_PACK_ALIGNMENT;
	file_bytes = (long)(payload_base + 256u);
	file = fopen(path, "wb");
	if ( file == 0 )
		return(-1);
	written = fwrite(header, 1u, 16u, file);
	written += fwrite(manifest, 1u, manifest_bytes, file);
	for ( index = 16u + (uint32_t)manifest_bytes;
		written < (size_t)file_bytes && index < (uint32_t)file_bytes;
		index++ )
	{
		uint8_t zero = 0u;
		written += fwrite(&zero, 1u, 1u, file);
	}
	fclose(file);
	return(written == (size_t)file_bytes ? file_bytes : -1);
}

static SparkStatus SparkK3ProbeInitialize(InitFunction init,
	const char *pack_path, SparkK3StageRunner *runner)
{
	SparkK3StageRunnerConfiguration config;
	memset(&config, 0, sizeof(config));
	config.abi_version = SPARK_K3_STAGE_RUNNER_ABI_VERSION;
	config.descriptor_bytes = (uint32_t)sizeof(config);
	config.stage_index = 0u;
	config.stage_count = 1u;
	config.tp_degree = 1u;
	config.max_active_sequence_count = 1u;
	config.max_input_row_count = 1u;
	config.resident_sequence_capacity = 1u;
	config.kv_pages_per_sequence = 1u;
	config.multiprocessors = 1u;
	config.rank_pack_path = pack_path;
	return(init(runner, &config));
}

int main(void)
{
	void *adapter;
	InitFunction init;
	SparkK3StageRunner runner;
	char pack_path[256];
	char digest[65];
	char pool_env[64];
	long pack_bytes;
	uint32_t i;
	int failures = 0;
	snprintf(pack_path, sizeof(pack_path), "/tmp/k3attach-%d.k3pack", (int)getpid());
	pack_bytes = SparkK3ProbeWritePack(pack_path);
	failures += expect(pack_bytes > 0, "synthetic manifest-only pack written");
	for ( i = 0u; i < 64u; i++ )
		digest[i] = "0123456789abcdef"[i % 16u];
	digest[64] = '\0';
	unsetenv("SPARK_WEIGHTD_SOCKET");
	unsetenv("SPARK_WEIGHTD_ATTACH");
	unsetenv("SPARK_WEIGHTD_PACK_SHA256");
	unsetenv("SPARK_WEIGHTD_EXPERT_POOL_BYTES");
	unsetenv("SPARK_WEIGHTD_SPINE_BUDGET_BYTES");
	adapter = dlopen(ADAPTER_SO, RTLD_NOW);
	if ( adapter == 0 )
	{
		printf("SKIP: %s (%s)\n", ADAPTER_SO, dlerror());
		remove(pack_path);
		return(0);
	}
	init = (InitFunction)dlsym(adapter, "SparkK3StageRunnerInitialize");
	failures += expect(init != 0, "SparkK3StageRunnerInitialize resolved");
	memset(&runner, 0, sizeof(runner));
	failures += expect(SparkK3ProbeInitialize(init, pack_path, &runner) ==
		SPARK_STATUS_BUSY,
		"weightd absent fails closed with BUSY (no direct load)");
	if ( runner.private_state != 0 )
		failures += expect(0, "runner state leaked on failure");
	snprintf(pool_env, sizeof(pool_env), "%ld", pack_bytes);
	setenv("SPARK_WEIGHTD_SOCKET", "/tmp/k3attach-probe-absent.sock", 1);
	setenv("SPARK_WEIGHTD_PACK_SHA256", digest, 1);
	setenv("SPARK_WEIGHTD_EXPERT_POOL_BYTES", pool_env, 1);
	setenv("SPARK_WEIGHTD_SPINE_BUDGET_BYTES", pool_env, 1);
	failures += expect(SparkK3ProbeInitialize(init, pack_path, &runner) ==
		SPARK_STATUS_IO_ERROR,
		"configured weightd with dead socket fails closed (no direct load)");
	dlclose(adapter);
	remove(pack_path);
	printf("test_k3_attach_contract: %d failures\n", failures);
	return(failures != 0 ? 1 : 0);
}
