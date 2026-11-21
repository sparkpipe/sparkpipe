#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <string>

#include "model-families/k3/source/layer.cuh"

#define SHAPES_RANKS 16u

static uint32_t failures;

static void Tp16(K3LayerBuffers *b)
{
	memset(b, 0, sizeof(*b));
	b->tp_sharded = 1u;
	b->kda_qkvb_rows = K3_KDA_QKVB_FUSED_ROWS / SHAPES_RANKS;
	b->kda_gate_rows = K3_KDA_V_DIM / SHAPES_RANKS;
	b->kda_decay_up_rows = K3_KDA_QK_DIM / SHAPES_RANKS;
	b->kda_out_input = K3_KDA_V_DIM / SHAPES_RANKS;
	b->mla_q_up_rows = K3_MLA_Q_DIM / SHAPES_RANKS;
	b->mla_gate_rows = K3_MLA_OUT_DIM / SHAPES_RANKS;
	b->mla_out_input = K3_MLA_OUT_DIM / SHAPES_RANKS;
	b->mla_heads_rank = K3_MLA_HEADS / SHAPES_RANKS;
	b->shared_w1_rows = 2u * K3_SHARED_INTERMEDIATE / SHAPES_RANKS;
	b->shared_w2_input = K3_SHARED_INTERMEDIATE / SHAPES_RANKS;
	b->routed_up_input = K3_ROUTED_EXPERT_HIDDEN / SHAPES_RANKS;
	b->routed_down_rows = K3_ROUTED_EXPERT_HIDDEN / SHAPES_RANKS;
	b->dense_gate_up_rows = 2u * K3_DENSE_INTERMEDIATE / SHAPES_RANKS;
	b->dense_down_input = K3_DENSE_INTERMEDIATE / SHAPES_RANKS;
}

static std::string Captured(const K3LayerBuffers *b, int32_t *status)
{
	char chunk[512];
	size_t count;
	std::string text;
	FILE *sink = tmpfile();
	fflush(stderr);
	const int saved = dup(fileno(stderr));
	dup2(fileno(sink), fileno(stderr));
	*status = K3ProjectionShapes(b);
	fflush(stderr);
	dup2(saved, fileno(stderr));
	close(saved);
	rewind(sink);
	while ( (count = fread(chunk, 1u, sizeof(chunk), sink)) > 0u )
		text.append(chunk, count);
	fclose(sink);
	return text;
}

static void Expect(const char *name, uint32_t condition)
{
	if ( condition == 0u )
	{
		printf("FAIL %s\n", name);
		failures++;
	}
	else
		printf("PASS %s\n", name);
}

static void Refuses(const char *name, const K3LayerBuffers *b, const char *message)
{
	int32_t status = LM_LAUNCH_OK;
	spark_last_error_site.file = 0;
	const std::string text = Captured(b, &status);
	const char *file = spark_last_error_site.file != 0 ? spark_last_error_site.file : "";
	if ( status != LM_LAUNCH_ERR_SHAPE || text.find(message) == std::string::npos || strstr(file, "model-families/k3/source/layer.cuh") == 0 )
	{
		printf("FAIL %s: status %d, site %s, stderr:\n%s", name, status, file, text.c_str());
		failures++;
		return;
	}
	printf("PASS %s refuses at load with \"%s\"\n", name, message);
}

int main(void)
{
	K3LayerBuffers b;
	int32_t status = LM_LAUNCH_ERR_SHAPE;
	Tp16(&b);
	const std::string text = Captured(&b, &status);
	Expect("the K3 and K3.1 Pro TP16 geometry passes the load check", status == LM_LAUNCH_OK && text.empty() ? 1u : 0u);
	Expect("routed_up's TP16 slice of 224 takes the tiled GEMM for prefill by shape",
		b.routed_up_input == 224u && K3ProjectStreams(b.routed_up_input, (const uint16_t *)0) == 0u ? 1u : 0u);
	Expect("shared_w2's TP16 slice of 384 takes the stream GEMM for prefill by shape",
		b.shared_w2_input == 384u && K3ProjectStreams(b.shared_w2_input, (const uint16_t *)0) != 0u ? 1u : 0u);
	Expect("an accumulating projection takes the tiled GEMM whatever its input",
		K3ProjectStreams(K3_HIDDEN, (const uint16_t *)0x10000) == 0u ? 1u : 0u);
	Tp16(&b);
	b.routed_up_input = 200u;
	Refuses("a routed_up slice of 200", &b, "k3 projection routed_up: this rank's 7168 x 200 slice has an input that is not a nonzero multiple of 32");
	Tp16(&b);
	b.kda_qkvb_rows = 2311u;
	Refuses("an odd kda_qkv_beta slice", &b, "k3 projection kda_qkv_beta: this rank's 2311 x 7168 slice takes the stream GEMM for prefill, which needs an even output");
	if ( failures != 0u )
	{
		printf("FAIL k3 projection shapes: %u cases\n", failures);
		return 1;
	}
	printf("PASS k3 projection shapes: production TP16 slices pass the load check, routed_up takes the tiled GEMM by shape, and an unfit slice refuses at load\n");
	return 0;
}
