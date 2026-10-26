#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_glm53_full_tp16_stage_model.h"
#include "glm52_stage.cuh"

struct ProbeGeometry : Glm52Dimensions
{
	static constexpr uint32_t kTpDegree = SPARK_GLM53_FULL_TP16_STAGE_TP_DEGREE;
	static constexpr uint32_t kIndexLayers = SPARK_GLM53_FULL_TP16_STAGE_INDEX_LAYERS;
	static const char *Tag(void) { return "glm_attach_kv_probe"; }
	static const char *ReportTag(void) { return "GLM-ATTACH-KV-PROBE"; }
	static const char *WeightdModel(void) { return "glm52_stage"; }
	static const char *Revision(void) { return "probe"; }
	static const char *Contract(void) { return "probe"; }
};

typedef GlmModel<ProbeGeometry> ProbeModel;
typedef GlmShape<ProbeGeometry> ProbeShape;

#define PROBE_PAGES_PER_SEQUENCE 4u
#define PROBE_POOL_PAGES 32u
#define PROBE_SEQUENCES 2u
#define PROBE_INDEX_PAGE_BYTES (GlmIndexKv<ProbeShape>::kPageBytes / SPARK_GLM53_FULL_TP16_STAGE_TP_DEGREE)

static int probe_failures;
static uint8_t probe_pool[8];
static uint8_t probe_index[8];
static uint32_t probe_table[PROBE_SEQUENCES * PROBE_PAGES_PER_SEQUENCE];

static ProbeModel::State *ProbeState(void)
{
	ProbeModel::State *state = (ProbeModel::State *)calloc(1u, sizeof(ProbeModel::State));
	if ( state == 0 )
	{
		printf("FAIL glm_attach_kv_cuda: no memory for the stage state\n");
		exit(1);
	}
	state->tp_degree = SPARK_GLM53_FULL_TP16_STAGE_TP_DEGREE;
	state->slots = PROBE_SEQUENCES;
	state->max_positions = PROBE_PAGES_PER_SEQUENCE * GLM_PAGE_SLOTS;
	state->shard.degree = SPARK_GLM53_FULL_TP16_STAGE_TP_DEGREE;
	state->shard.rank = 3u;
	state->shard.grain = GLM_PAGE_SLOTS;
	return state;
}

static SparkStageRunnerKv ProbeBinding(const ProbeModel::State *state)
{
	SparkStageRunnerKv kv;
	memset(&kv, 0, sizeof(kv));
	kv.pool = probe_pool;
	kv.layer_page_bytes = (uint64_t)GLM_PAGE_SLOTS * ProbeShape::kSlotBytes;
	kv.layer_stride_bytes = kv.layer_page_bytes / state->tp_degree;
	kv.layer_count = ProbeShape::kLayers;
	kv.page_table = probe_table;
	kv.page_table_stride = PROBE_PAGES_PER_SEQUENCE;
	kv.pool_page_count = PROBE_POOL_PAGES;
	kv.sequence_count = PROBE_SEQUENCES;
	kv.context_shard = state->shard;
	kv.second_pool = probe_index;
	kv.second_layer_stride_bytes = (uint64_t)PROBE_POOL_PAGES * PROBE_INDEX_PAGE_BYTES;
	kv.second_layer_page_bytes = GlmIndexKv<ProbeShape>::kPageBytes;
	kv.second_layer_count = ProbeGeometry::kIndexLayers;
	return kv;
}

static void ProbeCase(const char *name, void (*bend)(SparkStageRunnerKv *), int expect_ok)
{
	ProbeModel::State *state = ProbeState();
	SparkStageRunnerKv kv = ProbeBinding(state);
	SparkStatus status;
	if ( bend != 0 )
		bend(&kv);
	status = ProbeModel::AttachKv(state, &kv);
	if ( expect_ok != 0 && (status != SPARK_STATUS_OK || state->kv_attached != 1u || state->index_layer_stride != kv.second_layer_stride_bytes ||
		state->page_table_stride != kv.page_table_stride) )
	{
		printf("FAIL %s: a binding that covers every position and pool page is refused (status %d attached %u)\n", name, (int)status, state->kv_attached);
		probe_failures++;
	}
	if ( expect_ok == 0 && (status == SPARK_STATUS_OK || state->kv_attached != 0u || state->index_pool != 0 || state->page_table != 0) )
	{
		printf("FAIL %s: the binding is attached (status %d attached %u)\n", name, (int)status, state->kv_attached);
		probe_failures++;
	}
	free(state);
}

static void ProbeShortIndexLayers(SparkStageRunnerKv *kv)
{
	kv->second_layer_stride_bytes = (uint64_t)(PROBE_POOL_PAGES - 1u) * PROBE_INDEX_PAGE_BYTES;
}

static void ProbeShortPageTable(SparkStageRunnerKv *kv)
{
	kv->page_table_stride = PROBE_PAGES_PER_SEQUENCE - 1u;
}

static void ProbeNoPool(SparkStageRunnerKv *kv)
{
	kv->pool = 0;
}

static void ProbeNoPageTable(SparkStageRunnerKv *kv)
{
	kv->page_table = 0;
}

static void ProbeNoPoolPages(SparkStageRunnerKv *kv)
{
	kv->pool_page_count = 0u;
}

static void ProbeWideStrides(SparkStageRunnerKv *kv)
{
	kv->page_table_stride = PROBE_PAGES_PER_SEQUENCE + 4u;
	kv->second_layer_stride_bytes = (uint64_t)(PROBE_POOL_PAGES + 8u) * PROBE_INDEX_PAGE_BYTES;
}

int main(void)
{
	ProbeCase("exact strides of a context-sharded binding", 0, 1);
	ProbeCase("wider strides", ProbeWideStrides, 1);
	ProbeCase("index layers shorter than the pool", ProbeShortIndexLayers, 0);
	ProbeCase("page table shorter than a sequence", ProbeShortPageTable, 0);
	ProbeCase("no latent pool", ProbeNoPool, 0);
	ProbeCase("no page table", ProbeNoPageTable, 0);
	ProbeCase("no pool pages", ProbeNoPoolPages, 0);
	if ( probe_failures != 0 )
	{
		printf("FAIL glm_attach_kv_cuda: %d failures\n", probe_failures);
		return(1);
	}
	printf("PASS glm_attach_kv_cuda: GLM attaches KV only when index layers span every pool page at this rank's share of a page and the page table spans a full sequence\n");
	return(0);
}
