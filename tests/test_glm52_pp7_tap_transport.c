/* PP7 cross-stage tap/sideband transport (host, no GPU): the DFlash2
 * drafter's tap ring under the flagship [12,11x6] pipeline.
 *
 * White-box on purpose (test_glm52_pp7_configure pattern): the harness
 * includes the module translation unit so the speculator configure chain,
 * the tap-sideband wire format and the transport hooks run on host, with
 * the CUDA memcpys executing as real byte moves (tests/cuda_stub) and the
 * TapStore launcher re-implemented with its exact device semantics on host
 * buffers. Pinned here:
 *
 *   1. Topology roles - with the speculator armed, every PP7 stage
 *      configures: capture owners {0,1,3,4,6} for taps {7,22,38,54,69},
 *      drafter role on the head stage only, relay groups per stage;
 *      TP>1 fanout and unknown grids refuse loudly (the adapter's TP1
 *      doctrine, mirrored module-side);
 *   2. The wire record - block layout [header][records][payload groups],
 *      exact extents both ends derive from shared facts, records carrying
 *      (lane slot, position, tap row) so any split or reordered delivery
 *      reconciles into the same arena cells;
 *   3. THE SEQUENCE PROOF - the same tokens walked through the seven
 *      synthetic stage packs (forced taps crossing EVERY used boundary
 *      0->1 .. 5->6, relays forwarding, head ingesting) leave the
 *      drafter's tap arena BYTE-IDENTICAL to a TP1 single-stage run of
 *      the same tokens, verify frame split across waves included;
 *   4. Split-frame reconciliation - the draft tail folds accept depths and
 *      stages anchors from FRAME-relative host arrays (a verify frame
 *      split across waves reconciles; the pre-transport first_row read
 *      would have staged the wrong anchor), and a starved hop fails the
 *      draft loudly (tap_coverage_missing) instead of drafting from
 *      partial taps;
 *   5. The frame contract - tap sidebands are required exactly when the
 *      stage's role says so on speculating frames, with EXACT byte
 *      extents, and forbidden otherwise.
 */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "spark_glm52_resident_decode_stage_module.c"

#define TEST_MODEL_REVISION "pp7-taps"
#define TEST_PACK_DIR "build/tmp/glm52_tap_transport"
#define TEST_LANES 2u
#define TEST_ROWS_PER_LANE SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_VERIFY_ROW_COUNT
#define TEST_FRAME_ROWS (TEST_LANES * TEST_ROWS_PER_LANE)
#define TEST_HIDDEN SPARK_GLM52_MODEL_HIDDEN_DIMENSION
#define TEST_TAP_COUNT SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_TAP_COUNT
#define TEST_FULL_MASK ((1u << TEST_TAP_COUNT) - 1u)
#define TEST_LANE_STRIDE_ELEMENTS \
	(SPARK_DSPARK_AUX_LAYER_COUNT * TEST_HIDDEN)

static const uint32_t TestPp7Counts[SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT] =
	SPARK_GLM52_MODEL_DSPARK_PP_STAGE_LAYER_COUNTS_INITIALIZER;
static const uint32_t TestPp7Firsts[SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT] =
	SPARK_GLM52_MODEL_DSPARK_PP_STAGE_FIRST_LAYER_INITIALIZER;

static uint32_t TestHexDigit(char digit)
{
	if ( digit >= '0' && digit <= '9' )
		return((uint32_t)(digit - '0'));
	return((uint32_t)(digit - 'a') + 10u);
}

static void TestHexSha(const char *hex,uint8_t out[SPARK_GLM52_STAGEPACK_SHA256_BYTES])
{
	uint32_t index;
	for (index = 0u; index < SPARK_GLM52_STAGEPACK_SHA256_BYTES; index++)
		out[index] = (uint8_t)(TestHexDigit(hex[2u*index]) * 16u +
			TestHexDigit(hex[2u*index + 1u]));
}

/* One synthetic stage pack: real header + directory + shapes at tp_degree=1,
 * payload bytes claimed but left as file holes (never read here). Same
 * builder as the pp7 dry proof. */
static void TestBuildStagePack(const char *path,uint32_t stage_count,
	uint32_t stage_index,uint32_t first_layer,uint32_t layer_count)
{
	uint8_t sha[SPARK_GLM52_STAGEPACK_SHA256_BYTES];
	SparkGlm52StagePackHeader header;
	SparkGlm52StagePackEntry entries[SPARK_GLM52_STAGEPACK_MAX_TENSOR_COUNT];
	SparkStagePackShape shape;
	FILE *file;
	uint64_t offset,file_bytes,directory_end;
	uint32_t kind,layer,count;
	uint8_t contract_sha[SPARK_GLM52_STAGEPACK_SHA256_BYTES];

	memset(&header,0,sizeof(header));
	memset(entries,0,sizeof(entries));
	count = 0u;
	directory_end = 0u;
	TestHexSha("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",contract_sha);
	TestHexSha("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",sha);
	for (kind = 0u; kind < SPARK_GLM52_STAGEPACK_TENSOR_KIND_COUNT; kind++)
	{
		uint32_t is_global = kind <= SPARK_GLM52_STAGEPACK_TENSOR_LM_HEAD;
		if ( is_global != 0u )
		{
			uint32_t owns_embedding = kind == SPARK_GLM52_STAGEPACK_TENSOR_EMBEDDING && stage_index == 0u && first_layer == 0u;
			uint32_t owns_head = (kind == SPARK_GLM52_STAGEPACK_TENSOR_FINAL_NORM || kind == SPARK_GLM52_STAGEPACK_TENSOR_LM_HEAD) && stage_index + 1u == stage_count;
			if ( owns_embedding == 0u && owns_head == 0u )
				continue;
			assert(SparkGlm52StagePackExpectedShape(kind,SPARK_GLM52_STAGEPACK_GLOBAL_LAYER,GLM52_EXPERT_WEIGHT_CODEC,1u,&shape) == 0);
			goto add_entry;
		}
		for (layer = first_layer; layer < first_layer + layer_count; layer++)
		{
			if ( (kind >= SPARK_GLM52_STAGEPACK_TENSOR_INDEX_Q && kind <= SPARK_GLM52_STAGEPACK_TENSOR_INDEX_NORM_BIAS) != 0u && SPARK_GLM52_MODEL_LAYER_HAS_FULL_INDEXER(layer) == 0u )
				continue;
			if ( (kind == SPARK_GLM52_STAGEPACK_TENSOR_DENSE_GATE_UP || kind == SPARK_GLM52_STAGEPACK_TENSOR_DENSE_DOWN) != 0u && layer >= SPARK_GLM52_MODEL_FIRST_ROUTED_LAYER )
				continue;
			if ( kind >= SPARK_GLM52_STAGEPACK_TENSOR_ROUTER != 0u && layer < SPARK_GLM52_MODEL_FIRST_ROUTED_LAYER )
				continue;
			assert(SparkGlm52StagePackExpectedShape(kind,layer,GLM52_EXPERT_WEIGHT_CODEC,1u,&shape) == 0);
add_entry:
			assert(count < SPARK_GLM52_STAGEPACK_MAX_TENSOR_COUNT);
			entries[count].tensor_kind = kind;
			entries[count].layer_index = is_global != 0u ? SPARK_GLM52_STAGEPACK_GLOBAL_LAYER : layer;
			entries[count].payload_type = shape.payload_type;
			entries[count].weight_codec = shape.weight_codec;
			entries[count].scale_encoding = shape.scale_encoding;
			entries[count].group_count = shape.group_count;
			entries[count].rows = shape.rows;
			entries[count].columns = shape.columns;
			entries[count].payload_bytes = SparkStagePackPayloadBytes(&shape);
			entries[count].scale_bytes = SparkStagePackScaleBytes(&shape);
			count++;
			if ( is_global != 0u )
				break;
		}
	}
	offset = 512u + (uint64_t)count * SPARK_GLM52_STAGEPACK_ENTRY_BYTES;
	for (kind = 0u; kind < count; kind++)
	{
		offset = (offset + SPARK_GLM52_STAGEPACK_ALIGNMENT_BYTES - 1u) & ~((uint64_t)SPARK_GLM52_STAGEPACK_ALIGNMENT_BYTES - 1u);
		entries[kind].payload_offset = offset;
		offset = entries[kind].payload_offset + entries[kind].payload_bytes;
		if ( entries[kind].scale_bytes != 0u )
		{
			offset = (offset + SPARK_GLM52_STAGEPACK_ALIGNMENT_BYTES - 1u) & ~((uint64_t)SPARK_GLM52_STAGEPACK_ALIGNMENT_BYTES - 1u);
			entries[kind].scale_offset = offset;
			offset = entries[kind].scale_offset + entries[kind].scale_bytes;
		}
	}
	file_bytes = offset;

	header.magic = SPARK_GLM52_STAGEPACK_MAGIC;
	header.format_version = SPARK_GLM52_STAGEPACK_FORMAT_VERSION;
	header.header_bytes = SPARK_GLM52_STAGEPACK_HEADER_BYTES;
	header.directory_entry_bytes = SPARK_GLM52_STAGEPACK_ENTRY_BYTES;
	header.codec_abi_version = SPARK_WEIGHT_CODEC_ABI_VERSION;
	header.flags = 0u;
	header.tensor_count = count;
	header.stage_count = stage_count;
	header.stage_index = stage_index;
	header.first_layer_index = first_layer;
	header.layer_count = layer_count;
	header.total_layer_count = SPARK_GLM52_MODEL_LAYER_COUNT;
	header.hidden_dimension = SPARK_GLM52_MODEL_HIDDEN_DIMENSION;
	header.vocab_count = SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT;
	header.routed_expert_count = SPARK_GLM52_MODEL_MOE_EXPERT_COUNT;
	header.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16;
	header.expert_weight_codec = GLM52_EXPERT_WEIGHT_CODEC;
	header.kv_cache_codec = SPARK_WEIGHT_CODEC_BF16;
	header.reserved0 = 1u; /* tp_degree */
	header.reserved1 = 0u; /* tp_rank */
	header.directory_offset = 512u;
	header.file_bytes = file_bytes;
	memcpy(header.model_revision,TEST_MODEL_REVISION,sizeof(TEST_MODEL_REVISION));
	memcpy(header.contract_sha256,contract_sha,sizeof(contract_sha));
	memcpy(header.source_config_sha256,sha,sizeof(sha));
	memcpy(header.pack_recipe_sha256,sha,sizeof(sha));
	directory_end = header.directory_offset + (uint64_t)count * SPARK_GLM52_STAGEPACK_ENTRY_BYTES;
	assert(directory_end <= entries[0].payload_offset);

	file = fopen(path,"wb");
	assert(file != 0);
	assert(fwrite(&header,1,sizeof(header),file) == sizeof(header));
	assert(fseeko(file,(off_t)header.directory_offset,SEEK_SET) == 0);
	for (kind = 0u; kind < count; kind++)
		assert(fwrite(&entries[kind],1,sizeof(entries[kind]),file) == sizeof(entries[kind]));
	assert(fseeko(file,(off_t)(file_bytes - 1u),SEEK_SET) == 0);
	assert(fputc(0,file) != EOF);
	assert(fclose(file) == 0);
}

static void TestFillContext(SparkGlm52ResidentDecodeStageNodeContext *context,
	uint32_t stage_count,uint32_t stage_index,uint32_t first_layer,
	uint32_t layer_count,uint32_t tp_degree,const char *pack_path)
{
	memset(context,0,sizeof(*context));
	context->abi_version = SPARK_GLM52_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION;
	context->descriptor_bytes = SPARK_GLM52_RESIDENT_DECODE_STAGE_NODE_CONTEXT_BYTES;
	context->stage_count = stage_count;
	context->stage_index = stage_index;
	context->first_layer_index = first_layer;
	context->layer_count = layer_count;
	context->expert_weight_codec = GLM52_EXPERT_WEIGHT_CODEC;
	context->resident_sequence_capacity = 8u;
	context->pipeline_slot_count = 2u;
	context->max_sequence_positions = 4096u;
	context->execution_row_capacity = 4u;
	context->tp_degree = tp_degree;
	context->tp_rank = 0u;
	context->stage_pack_path = pack_path;
	context->model_revision = TEST_MODEL_REVISION;
	context->kv_backing_directory = "/tmp";
}

/* Configure one stage through the real chain with the speculator env armed
 * (or disarmed for the disabled variant) and return the configured state. */
static void TestConfigureSpeculatorStage(uint32_t stage_count,uint32_t stage_index,
	uint32_t tp_degree,const char *path,struct SparkGlm52ModuleState *state,
	SparkStatus expected_status)
{
	SparkGlm52ResidentDecodeStageNodeContext context;
	SparkFirmwareModuleConfiguration configuration;
	SparkFirmwareModuleHostServices services;
	const SparkGlm52ResidentDecodeStageGeometry *geometry;
	const char *pack_path = 0;

	geometry = SparkGlm52ResidentDecodeStageGeometryFor(stage_count);
	if ( expected_status == SPARK_STATUS_OK )
		assert(geometry != 0);
	TestFillContext(&context,stage_count,stage_index,
		geometry != 0 ? SparkGlm52ResidentDecodeStageFirstLayer(geometry,stage_index) : 0u,
		geometry != 0 ? geometry->stage_layer_counts[stage_index] : 0u,
		tp_degree,path);
	memset(&configuration,0,sizeof(configuration));
	configuration.model_revision = TEST_MODEL_REVISION;
	memset(&services,0,sizeof(services));
	services.node_context = &context;
	services.execution_stream = (void *)0x1;
	memset(state,0,sizeof(*state));
	assert(SparkGlm52ModuleConfigure(state,&configuration,&services,&pack_path) == expected_status);
}

/* The per-stage transport roles the PP7 split must derive. */
static const uint32_t TestExpectedGroupsIn[SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT] = { 0u,1u,2u,2u,3u,4u,4u };
static const uint32_t TestExpectedGroupsOut[SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT] = { 1u,2u,2u,3u,4u,4u,0u };
static const uint32_t TestExpectedTapOwner[TEST_TAP_COUNT] = { 0u,1u,3u,4u,6u };

static void TestTopologyRoles(void)
{
	static struct SparkGlm52ModuleState state;
	static const uint32_t tp8_groups_in[1u] = { 0u };
	static const uint32_t tp8_groups_out[1u] = { 0u };
	char path[256];
	uint32_t stage,tap;

	mkdir(TEST_PACK_DIR,0755);
	assert(setenv("SPARK_GLM52_STAGE_SPECULATOR","1",1) == 0);
	assert(setenv("SPARK_GLM52_DSPARK_MANIFEST","tap-transport-manifest",1) == 0);
	assert(setenv("SPARK_GLM52_DSPARK_CONFIG","tap-transport-config",1) == 0);
	assert(setenv("SPARK_GLM52_DSPARK_SAFETENSORS","tap-transport-safetensors",1) == 0);
	for (stage = 0u; stage < SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT; stage++)
	{
		snprintf(path,sizeof(path),"%s/stage%u.fp8.glm52sp",TEST_PACK_DIR,stage);
		TestBuildStagePack(path,SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT,stage,
			TestPp7Firsts[stage],TestPp7Counts[stage]);
		TestConfigureSpeculatorStage(SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT,stage,1u,path,&state,SPARK_STATUS_OK);
		assert(state.speculator_enabled == 1u);
		assert(state.dspark_drafter_role == (stage + 1u == SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT));
		assert(state.dspark_tap_groups_in == TestExpectedGroupsIn[stage]);
		assert(state.dspark_tap_groups_out == TestExpectedGroupsOut[stage]);
	}
	/* TP8 standalone keeps its pre-transport shape: local ring, no wire. */
	snprintf(path,sizeof(path),"%s/tp8.fp8.glm52sp",TEST_PACK_DIR);
	TestBuildStagePack(path,SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_COUNT,0u,0u,SPARK_GLM52_MODEL_LAYER_COUNT);
	TestConfigureSpeculatorStage(SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_COUNT,0u,1u,path,&state,SPARK_STATUS_OK);
	assert(state.dspark_drafter_role == 1u && state.dspark_tap_groups_in == 0u && state.dspark_tap_groups_out == 0u);
	assert(tp8_groups_in[0] == 0u && tp8_groups_out[0] == 0u);

	/* Owner table: taps {7,22,38,54,69} land on stages {0,1,3,4,6}. */
	for (tap = 0u; tap < TEST_TAP_COUNT; tap++)
	{
		assert(SparkGlm52ResidentDecodeStageDsparkTapOwnerStage(SparkGlm52ResidentDecodeStageGeometryFor(SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT),tap) == TestExpectedTapOwner[tap]);
		assert(SparkGlm52ResidentDecodeStageDsparkTapOwnerStage(SparkGlm52ResidentDecodeStageGeometryFor(1u),tap) == 0u);
	}
	assert(SparkGlm52ResidentDecodeStageOwningLayer(SparkGlm52ResidentDecodeStageGeometryFor(SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT),SPARK_GLM52_MODEL_LAYER_COUNT + 3u) == UINT32_MAX);

	/* Loud refusals: TP fanout has no draft transport; a retired grid is
	 * unknown. Both refuse at configure, never silently arm. */
	snprintf(path,sizeof(path),"%s/stage0.fp8.glm52sp",TEST_PACK_DIR);
	TestConfigureSpeculatorStage(SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT,0u,2u,path,&state,SPARK_STATUS_INVALID_ARGUMENT);
	TestConfigureSpeculatorStage(13u,3u,1u,"unused",&state,SPARK_STATUS_INVALID_ARGUMENT);
	printf("topology roles: owners {0,1,3,4,6}, head-only drafter, relay groups pinned; tp>1 and unknown grids refused\n");
}

static void TestWireFormat(void)
{
	/* The wire is a cross-process contract: sizes are pinned exactly so a
	 * silent layout drift cannot slip through a rebuild. */
	assert(SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_TAP_SIDEBAND_HEADER_BYTES == 32u);
	assert(SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_TAP_SIDEBAND_RECORD_BYTES == 24u);
	assert(sizeof(SparkGlm52ResidentDecodeStageDsparkTapSidebandHeader) == 32u);
	assert(sizeof(SparkGlm52ResidentDecodeStageDsparkTapSidebandRecord) == 24u);
	/* 8 rows x 2 groups: 32 + 16*24 + 16*6144*2 = 197024. */
	assert(SparkGlm52ResidentDecodeStageDsparkTapSidebandBytes(8u,2u) == 32ull + 16ull * 24ull + 16ull * (uint64_t)TEST_HIDDEN * 2ull);
	assert(SparkGlm52ResidentDecodeStageDsparkTapSidebandBytes(0u,0u) == 32ull);
	/* Group bases: metadata first, then group-major payloads. */
	assert(SparkGlm52DsparkTapGroupBaseBytes(16u,3u,0u) == 32ull + 48ull * 24ull);
	assert(SparkGlm52DsparkTapGroupBaseBytes(16u,3u,1u) == 32ull + 48ull * 24ull + 16ull * (uint64_t)TEST_HIDDEN * 2ull);
	printf("wire format: 32-byte header, 24-byte records, closed-form extents and group bases\n");
}

/* ---- transport simulation -------------------------------------------- */

typedef struct TestStageRun
{
	struct SparkGlm52ModuleState state;
	SparkGlm52ExecutionSlot slot;
	SparkGlm52ResidentDecodeStageBatchView batch;
	SparkGlm52ResidentDecodeStageFrameContext context;
	SparkGlm52TpChain chain;
	uint8_t *outbound; /* full wire block this stage emits */
	uint64_t outbound_bytes;
	uint8_t *inbound;  /* leg-received block (process-boundary case) */
	uint16_t *hidden;  /* execution-row hidden buffer */
} TestStageRun;

static uint32_t TestRowSlot(uint32_t row) { return(row % TEST_LANES); }
static uint64_t TestRowPosition(uint32_t row)
{
	return(100ull + (row % TEST_LANES) * 400ull + row / TEST_LANES);
}
static uint64_t TestRowSequence(uint32_t row) { return(7000ull + TestRowSlot(row)); }
static uint32_t TestRowToken(uint32_t row) { return(1000u + row); }

static uint16_t TestHiddenPattern(uint32_t global_layer,uint32_t frame_row,uint32_t element)
{
	uint32_t mixed = global_layer * 2654435761u + frame_row * 97u + element * 31u + 1u;
	mixed ^= mixed >> 13u;
	mixed *= 1274126177u;
	mixed ^= mixed >> 16u;
	return((uint16_t)(mixed & 0xffffu) | 1u);
}

/* Recorder backend: the drafter engine's entry points succeed and record,
 * so the draft tail can be driven end-to-end on host (numerics stay gated
 * by the epoch-3 validator on hardware). */
static SparkGlm52DsparkDraftBackendStage TestStagedRecords[8u];
static uint32_t TestStagedCount;
static SparkGlm52DsparkDraftRequest TestDraftRequests[8u];
static uint32_t TestDraftRequestCount;
static uint32_t TestResetLaneCount;

SparkStatus SparkGlm52DsparkDraftBackendInitialize(SparkGlm52DsparkDraftBackend *backend,const SparkGlm52DsparkDraftBackendConfiguration *configuration)
{
	(void)backend;(void)configuration;
	return(SPARK_STATUS_OK);
}
SparkStatus SparkGlm52DsparkDraftBackendTeardown(SparkGlm52DsparkDraftBackend *backend)
{
	(void)backend;
	return(SPARK_STATUS_OK);
}
SparkStatus SparkGlm52DsparkDraftBackendModelContract(const SparkGlm52DsparkDraftBackend *backend,SparkGlm52DsparkModelContract *contract_out)
{
	(void)backend;(void)contract_out;
	return(1);
}
SparkStatus SparkGlm52DsparkDraftBackendTapOutputPointers(SparkGlm52DsparkDraftBackend *backend,uint32_t lane_index,void *tap_output_bf16[SPARK_DSPARK_AUX_LAYER_COUNT],uint64_t *lane_stride_bytes_out)
{
	(void)backend;(void)lane_index;(void)tap_output_bf16;(void)lane_stride_bytes_out;
	return(1);
}
SparkStatus SparkGlm52DsparkDraftBackendStageBatch(SparkGlm52DsparkDraftBackend *backend,const SparkGlm52DsparkDraftBackendStage *stages,uint32_t stage_count)
{
	uint32_t index;
	(void)backend;
	assert(stage_count <= 8u);
	for (index = 0u; index < stage_count; index++)
		TestStagedRecords[index] = stages[index];
	TestStagedCount = stage_count;
	return(SPARK_STATUS_OK);
}
SparkStatus SparkGlm52DsparkDraftBackendLaunchDraftBatch(SparkGlm52DsparkDraftBackend *backend,const SparkGlm52DsparkDraftRequest *requests,uint32_t lane_count)
{
	uint32_t index;
	(void)backend;
	assert(lane_count <= 8u);
	for (index = 0u; index < lane_count; index++)
		TestDraftRequests[index] = requests[index];
	TestDraftRequestCount = lane_count;
	return(SPARK_STATUS_OK);
}
SparkStatus SparkGlm52DsparkDraftBackendTakeBatchResults(SparkGlm52DsparkDraftBackend *backend,SparkGlm52DsparkDraftResult *results,uint32_t result_capacity,uint32_t *result_count)
{
	uint32_t lane,index,count;
	(void)backend;
	count = TestDraftRequestCount > 0u ? 3u : 0u;
	if ( count > result_capacity )
		count = result_capacity;
	for (lane = 0u; lane < count; lane++)
	{
		results[lane].token_count = 3u;
		for (index = 0u; index < 3u; index++)
		{
			results[lane].token_ids[index] = 500u + index;
			results[lane].confidence_milli[index] = 900u;
		}
	}
	*result_count = count;
	return(SPARK_STATUS_OK);
}
SparkStatus SparkGlm52DsparkDraftBackendResetLanes(SparkGlm52DsparkDraftBackend *backend,const uint32_t *lane_indices,uint32_t lane_count)
{
	(void)backend;(void)lane_indices;
	TestResetLaneCount += lane_count;
	return(SPARK_STATUS_OK);
}

/* The TapStore launcher with its exact device semantics on host buffers:
 * row r of the contiguous source lands at
 * arena_base + tap_row_indices[r]*row_stride + tap_index*hidden + column. */
cudaError_t SparkGlm52LaunchDsparkTapStore(cudaStream_t stream,const void *hidden_bf16,const uint32_t *tap_row_indices,uint32_t tap_index,uint32_t row_count,uint32_t hidden_dimension,uint16_t *arena_base,uint64_t arena_row_stride_elements)
{
	uint32_t row,column;
	(void)stream;
	if ( hidden_bf16 == 0 || tap_row_indices == 0 || arena_base == 0 || row_count == 0u || hidden_dimension == 0u )
		return(1);
	for (row = 0u; row < row_count; row++)
		for (column = 0u; column < hidden_dimension; column++)
			arena_base[(uint64_t)tap_row_indices[row] * arena_row_stride_elements +
				(uint64_t)tap_index * hidden_dimension + column] =
				((const uint16_t *)hidden_bf16)[(uint64_t)row * hidden_dimension + column];
	return(0);
}

static void TestBuildRun(TestStageRun *run,uint32_t stage_count,uint32_t stage_index,
	uint16_t *arena,const SparkGlm52ResidentDecodeStageBatchView *batch)
{
	struct SparkGlm52ModuleState *state;
	uint64_t rows_bound;
	uint32_t lane_seed,row_seed;
	memset(run,0,sizeof(*run));
	state = &run->state;
	state->geometry = SparkGlm52ResidentDecodeStageGeometryFor(stage_count);
	assert(state->geometry != 0);
	state->stage_index = stage_index;
	state->first_layer_index = SparkGlm52ResidentDecodeStageFirstLayer(state->geometry,stage_index);
	state->layer_count = state->geometry->stage_layer_counts[stage_index];
	state->owns_embedding = stage_index == 0u;
	state->owns_final_head = stage_index + 1u == stage_count;
	state->speculator_enabled = 1u;
	state->dspark_drafter_role = state->owns_final_head;
	state->dspark_tap_groups_in = SparkGlm52ResidentDecodeStageDsparkTapGroupsIn(state->geometry,stage_index);
	state->dspark_tap_groups_out = state->owns_final_head != 0u ? 0u : SparkGlm52ResidentDecodeStageDsparkTapGroupsOut(state->geometry,stage_index);
	state->resident_sequence_capacity = 8u;
	state->execution_row_capacity = 4u;
	state->max_sequence_positions = 4096u;
	state->execution_stream = (void *)0x1;
	state->tp_degree = 1u;
	state->dspark_tap_rows_host = (uint32_t *)malloc(64u * sizeof(uint32_t));
	rows_bound = (uint64_t)state->resident_sequence_capacity * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_VERIFY_ROW_COUNT;
	state->dspark_tap_wire_rows_capacity = rows_bound;
	state->dspark_tap_wire_host = (uint8_t *)calloc(1u,(size_t)(SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_TAP_SIDEBAND_HEADER_BYTES + rows_bound * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_TAP_SIDEBAND_RECORD_BYTES));
	state->dspark_lane_tap_mask = (uint32_t *)calloc(state->resident_sequence_capacity,sizeof(uint32_t));
	state->dspark_lane_staged_counts = (uint32_t *)calloc(state->resident_sequence_capacity,sizeof(uint32_t));
	state->dspark_lane_tap_generations = (uint64_t *)calloc(state->resident_sequence_capacity,sizeof(uint64_t));
	state->dspark_lane_pending_verify = (uint8_t *)calloc(state->resident_sequence_capacity,sizeof(uint8_t));
	state->dspark_lane_verify_starts = (uint64_t *)calloc(state->resident_sequence_capacity,sizeof(uint64_t));
	state->dspark_lane_verify_walked = (uint32_t *)calloc(state->resident_sequence_capacity,sizeof(uint32_t));
	state->dspark_lane_last_sequence_ids = (uint64_t *)calloc(state->resident_sequence_capacity,sizeof(uint64_t));
	assert(state->dspark_tap_rows_host != 0 && state->dspark_tap_wire_host != 0 && state->dspark_lane_tap_mask != 0 &&
		state->dspark_lane_staged_counts != 0 && state->dspark_lane_tap_generations != 0 &&
		state->dspark_lane_pending_verify != 0 && state->dspark_lane_verify_starts != 0 &&
		state->dspark_lane_verify_walked != 0 && state->dspark_lane_last_sequence_ids != 0);
	for (lane_seed = 0u; lane_seed < state->resident_sequence_capacity; lane_seed++)
		state->dspark_lane_tap_generations[lane_seed] = 1u;
	if ( state->dspark_drafter_role != 0u )
	{
		assert(arena != 0);
		state->dspark_backend.device_tap_arena_bf16 = arena;
		state->dspark_backend.tap_arena_lane_stride_bytes = (uint64_t)TEST_LANE_STRIDE_ELEMENTS * 2u;
		state->dspark_backend.completion_event = (void *)0x1;
	}
	run->hidden = (uint16_t *)calloc((uint64_t)state->execution_row_capacity * TEST_HIDDEN,sizeof(uint16_t));
	assert(run->hidden != 0);
	run->slot.hidden_bf16 = run->hidden;
	run->slot.host_token_ids = (uint32_t *)calloc(TEST_FRAME_ROWS + 8u,sizeof(uint32_t));
	run->slot.host_resident_slots = (uint32_t *)calloc(TEST_FRAME_ROWS + 8u,sizeof(uint32_t));
	run->slot.host_positions = (uint32_t *)calloc(TEST_FRAME_ROWS + 8u,sizeof(uint32_t));
	run->slot.host_output_token_ids = (uint32_t *)calloc(TEST_FRAME_ROWS + 8u,sizeof(uint32_t));
	run->slot.dspark_tap_row_indices = (uint32_t *)malloc(8u * sizeof(uint32_t));
	assert(run->slot.host_token_ids != 0 && run->slot.host_resident_slots != 0 && run->slot.host_positions != 0 &&
		run->slot.host_output_token_ids != 0 && run->slot.dspark_tap_row_indices != 0);
	/* What SparkGlm52StageHostBatch does in production: stage the frame's
	 * decode facts into the slot's host arrays once, whole-frame. */
	for (row_seed = 0u; row_seed < batch->row_count; row_seed++)
	{
		run->slot.host_resident_slots[row_seed] = batch->row_resident_slots[row_seed];
		run->slot.host_positions[row_seed] = (uint32_t)batch->row_positions[row_seed];
		run->slot.host_token_ids[row_seed] = batch->token_ids[row_seed];
	}

	run->batch = *batch;
	memset(&run->context,0,sizeof(run->context));
	run->context.abi_version = SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION;
	run->context.descriptor_bytes = sizeof(run->context);
	run->context.flags = SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_FLAG_PREFILL |
		(state->owns_embedding == 0u ? SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_FLAG_HIDDEN_INPUT : 0u) |
		(state->owns_final_head == 0u ? SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_FLAG_HIDDEN_OUTPUT : 0u) |
		SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_SPECULATIVE_VERIFY |
		SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DSPARK_DRAFT_AFTER;
	run->context.batch = &run->batch;
	run->context.dspark_tap_sideband_output = 0;
	run->context.dspark_tap_sideband_input = 0;
	memset(&run->chain,0,sizeof(run->chain));
	run->chain.state = state;
	run->chain.slot = &run->slot;
	run->chain.slot_index = 0u;
	run->chain.context = &run->context;
	run->chain.batch = &run->batch;
	run->chain.active = 1u;
	run->chain.dspark_capture = 1u;
	run->chain.dspark_draft_after = state->dspark_drafter_role;
}

static void TestDestroyRun(TestStageRun *run)
{
	free(run->state.dspark_tap_rows_host);
	free(run->state.dspark_tap_wire_host);
	free(run->state.dspark_lane_tap_mask);
	free(run->state.dspark_lane_staged_counts);
	free(run->state.dspark_lane_tap_generations);
	free(run->state.dspark_lane_pending_verify);
	free(run->state.dspark_lane_verify_starts);
	free(run->state.dspark_lane_verify_walked);
	free(run->state.dspark_lane_last_sequence_ids);
	free(run->hidden);
	free(run->slot.host_token_ids);
	free(run->slot.host_resident_slots);
	free(run->slot.host_positions);
	free(run->slot.host_output_token_ids);
	free(run->slot.dspark_tap_row_indices);
	free(run->outbound);
	free(run->inbound);
}

/* Walk one stage's full frame the way the chain would: per wave, upload the
 * ring rows, consume the inbound block, then walk every local layer,
 * filling the post-layer hidden with the shared pattern and firing the
 * capture hook at the aux layers. */
static void TestWalkStage(TestStageRun *run)
{
	struct SparkGlm52ModuleState *state = &run->state;
	uint32_t first_row,wave_rows,local_layer,wave_row,element;
	assert(SparkGlm52DsparkTapFrameBegin(state,&run->context,1u) == SPARK_STATUS_OK);
	first_row = 0u;
	while ( first_row < run->batch.row_count )
	{
		wave_rows = SparkGlm52RoundMajorWaveRows(&run->batch,first_row);
		assert(wave_rows > 0u && wave_rows <= state->execution_row_capacity);
		run->chain.first_row = first_row;
		run->chain.wave_rows = wave_rows;
		run->chain.next_wave_row = first_row + wave_rows;
		SparkGlm52DsparkUploadTapRows(&run->chain);
		SparkGlm52DsparkIngestWave(&run->chain);
		for (local_layer = 0u; local_layer < state->layer_count; local_layer++)
		{
			uint32_t global_layer = state->first_layer_index + local_layer;
			uint32_t tap_index;
			for (wave_row = 0u; wave_row < wave_rows; wave_row++)
				for (element = 0u; element < TEST_HIDDEN; element++)
					run->hidden[(uint64_t)wave_row * TEST_HIDDEN + element] =
						TestHiddenPattern(global_layer,first_row + wave_row,element);
			tap_index = SparkGlm52DsparkTapIndexForLayer(global_layer);
			if ( tap_index != UINT32_MAX )
				SparkGlm52DsparkCaptureLayer(&run->chain,tap_index);
		}
		first_row += wave_rows;
	}
}

static void TestBuildBatch(SparkGlm52ResidentDecodeStageBatchView *batch,
	uint32_t *slots,uint64_t *positions,uint64_t *sequences,uint32_t *tokens)
{
	uint32_t row;
	for (row = 0u; row < TEST_FRAME_ROWS; row++)
	{
		slots[row] = TestRowSlot(row);
		positions[row] = TestRowPosition(row);
		sequences[row] = TestRowSequence(row);
		tokens[row] = TestRowToken(row);
	}
	memset(batch,0,sizeof(*batch));
	batch->abi_version = SPARK_GLM52_RESIDENT_DECODE_STAGE_BATCH_VIEW_ABI_VERSION;
	batch->descriptor_bytes = sizeof(*batch);
	batch->row_count = TEST_FRAME_ROWS;
	batch->active_sequence_count = TEST_LANES;
	batch->token_ids = tokens;
	batch->row_resident_slots = slots;
	batch->row_positions = positions;
	batch->row_sequence_ids = sequences;
}

static void TestTransportSequenceParity(void)
{
	static TestStageRun pp7[SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT];
	TestStageRun tp1;
	SparkGlm52ResidentDecodeStageBatchView batch;
	static uint32_t slots[TEST_FRAME_ROWS];
	static uint64_t positions[TEST_FRAME_ROWS];
	static uint64_t sequences[TEST_FRAME_ROWS];
	static uint32_t tokens[TEST_FRAME_ROWS];
	uint16_t *arena_tp1,*arena_pp7;
	size_t arena_bytes;
	uint32_t stage,row,element,tap;
	const SparkGlm52ResidentDecodeStageDsparkTapSidebandHeader *header;
	const SparkGlm52ResidentDecodeStageDsparkTapSidebandRecord *records;

	TestBuildBatch(&batch,slots,positions,sequences,tokens);
	arena_bytes = (size_t)8u * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_TAP_ROWS_PER_LANE * TEST_TAP_COUNT * TEST_HIDDEN * sizeof(uint16_t);
	arena_tp1 = (uint16_t *)calloc(1u,arena_bytes);
	arena_pp7 = (uint16_t *)calloc(1u,arena_bytes);
	assert(arena_tp1 != 0 && arena_pp7 != 0);

	/* TP1 reference: one stage, whole ring local, same tokens. */
	TestBuildRun(&tp1,SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_COUNT,0u,arena_tp1,&batch);
	TestWalkStage(&tp1);
	for (row = 0u; row < TEST_LANES; row++)
		assert(tp1.state.dspark_lane_tap_mask[row] == TEST_FULL_MASK);

	/* PP7 pipeline: seven synthetic stage packs, taps forced across EVERY
	 * used boundary. Each stage's inbound block IS the previous stage's
	 * outbound block - the memcpys move real bytes. */
	for (stage = 0u; stage < SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT; stage++)
	{
		TestBuildRun(&pp7[stage],SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT,stage,
			stage + 1u == SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT ? arena_pp7 : 0,&batch);
		pp7[stage].outbound_bytes = pp7[stage].state.dspark_tap_groups_out > 0u ?
			SparkGlm52ResidentDecodeStageDsparkTapSidebandBytes(TEST_FRAME_ROWS,pp7[stage].state.dspark_tap_groups_out) : 0u;
		if ( pp7[stage].outbound_bytes > 0u )
		{
			pp7[stage].outbound = (uint8_t *)calloc(1u,(size_t)pp7[stage].outbound_bytes);
			assert(pp7[stage].outbound != 0);
			pp7[stage].context.dspark_tap_sideband_output = pp7[stage].outbound;
			pp7[stage].context.dspark_tap_sideband_output_bytes = pp7[stage].outbound_bytes;
		}
		if ( stage > 0u )
		{
			pp7[stage].context.dspark_tap_sideband_input = pp7[stage - 1u].outbound;
			pp7[stage].context.dspark_tap_sideband_input_bytes = pp7[stage - 1u].outbound_bytes;
			/* Every used boundary carries exactly the upstream group set:
			 * emitter groups_out == consumer groups_in, same row count. */
			assert(pp7[stage - 1u].state.dspark_tap_groups_out == pp7[stage].state.dspark_tap_groups_in);
			assert(pp7[stage - 1u].outbound_bytes == SparkGlm52ResidentDecodeStageDsparkTapSidebandBytes(TEST_FRAME_ROWS,pp7[stage].state.dspark_tap_groups_in));
		}
		TestWalkStage(&pp7[stage]);
	}

	/* THE PROOF: the drafter observes a byte-identical tap sequence. */
	assert(memcmp(arena_tp1,arena_pp7,arena_bytes) == 0);
	/* Spot-check one cell against the direct pattern (guards against a
	 * symmetric double-write cancellation hiding behind memcmp). */
	{
		uint32_t ring_row = 1u * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_TAP_ROWS_PER_LANE + (uint32_t)(TestRowPosition(3) % SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_TAP_ROWS_PER_LANE);
		uint16_t expected = TestHiddenPattern(22u,3u,777u);
		assert(arena_pp7[(uint64_t)ring_row * TEST_LANE_STRIDE_ELEMENTS + 1ull * TEST_HIDDEN + 777u] == expected);
	}

	/* Wire blocks: every non-head stage emitted a well-formed block whose
	 * records name (lane slot, position, tap row) for every row x group. */
	for (stage = 0u; stage + 1u < SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT; stage++)
	{
		uint32_t groups = pp7[stage].state.dspark_tap_groups_out;
		header = (const SparkGlm52ResidentDecodeStageDsparkTapSidebandHeader *)(const void *)pp7[stage].outbound;
		assert(header->magic == SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_TAP_SIDEBAND_MAGIC);
		assert(header->format_version == SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_TAP_SIDEBAND_FORMAT_VERSION);
		assert(header->record_count == TEST_FRAME_ROWS * groups);
		assert(header->row_count == TEST_FRAME_ROWS);
		assert(header->group_count == groups);
		assert(header->element_count == TEST_HIDDEN);
		records = (const SparkGlm52ResidentDecodeStageDsparkTapSidebandRecord *)(pp7[stage].outbound + header->header_bytes);
		for (tap = 0u; tap < groups; tap++)
			for (row = 0u; row < TEST_FRAME_ROWS; row++)
			{
				const SparkGlm52ResidentDecodeStageDsparkTapSidebandRecord *record = &records[tap * TEST_FRAME_ROWS + row];
				assert(record->tap_index == tap);
				assert(record->resident_slot == TestRowSlot(row));
				assert(record->position == TestRowPosition(row));
				assert(record->tap_row == TestRowSlot(row) * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_TAP_ROWS_PER_LANE +
					(uint32_t)(TestRowPosition(row) % SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_TAP_ROWS_PER_LANE));
				assert(record->element_count == TEST_HIDDEN);
			}
		/* Payload group 0 of the emitter equals the TP1 capture source for
		 * tap 0: the same pattern bytes the local run wrote. */
		for (row = 0u; row < TEST_FRAME_ROWS; row++)
			for (element = 0u; element < 32u; element++)
				assert(((const uint16_t *)(const void *)(pp7[stage].outbound + SparkGlm52DsparkTapGroupBaseBytes(TEST_FRAME_ROWS,groups,0u)))[(uint64_t)row * TEST_HIDDEN + element] == TestHiddenPattern(7u,row,element));
	}
	/* Head stage: inbound block only, nothing emitted. */
	assert(pp7[SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT - 1u].outbound == 0);
	assert(pp7[SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT - 1u].state.dspark_lane_tap_mask[0] == TEST_FULL_MASK);
	assert(pp7[SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT - 1u].state.dspark_lane_tap_mask[1] == TEST_FULL_MASK);

	for (stage = 0u; stage < SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT; stage++)
		TestDestroyRun(&pp7[stage]);
	TestDestroyRun(&tp1);
	free(arena_tp1);
	free(arena_pp7);
	printf("transport sequence parity: PP7 taps across every boundary are byte-identical to the TP1 run (verify frame split over waves)\n");
}

/* Split-frame reconciliation: a verify frame whose rows split across waves
 * must fold accept depths and stage anchors from FRAME-relative host
 * arrays, and a starved hop must refuse the draft loudly. */
static void TestSplitFrameDraftTail(void)
{
	static TestStageRun head;
	SparkGlm52ResidentDecodeStageBatchView batch;
	static uint32_t slots[TEST_FRAME_ROWS];
	static uint64_t positions[TEST_FRAME_ROWS];
	static uint64_t sequences[TEST_FRAME_ROWS];
	static uint32_t tokens[TEST_FRAME_ROWS];
	uint16_t *arena;
	SparkGlm52ResidentDecodeStageDsparkDraftView view;
	static uint32_t draft_ids[TEST_LANES * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_DRAFT_TOKEN_COUNT];
	static uint32_t confidence[TEST_LANES * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_DRAFT_TOKEN_COUNT];
	static uint32_t accepted[TEST_LANES];
	SparkModelDriverFrame frame;
	uint32_t lane,j;
	size_t arena_bytes;

	TestBuildBatch(&batch,slots,positions,sequences,tokens);
	arena_bytes = (size_t)8u * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_TAP_ROWS_PER_LANE * TEST_TAP_COUNT * TEST_HIDDEN * sizeof(uint16_t);
	arena = (uint16_t *)calloc(1u,arena_bytes);
	assert(arena != 0);
	TestBuildRun(&head,SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT,SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT - 1u,arena,&batch);
	/* Inbound block shape only (contents irrelevant to the tail): stage 5's
	 * four upstream groups. */
	head.outbound_bytes = SparkGlm52ResidentDecodeStageDsparkTapSidebandBytes(TEST_FRAME_ROWS,4u);
	head.outbound = (uint8_t *)calloc(1u,(size_t)head.outbound_bytes);
	assert(head.outbound != 0);
	head.context.dspark_tap_sideband_input = head.outbound;
	head.context.dspark_tap_sideband_input_bytes = head.outbound_bytes;
	head.chain.frame = &frame;
	memset(&frame,0,sizeof(frame));
	frame.request_id = 4242u;
	/* The adapter-owned draft view: the tail folds accept depths into it
	 * and hands back the next block's ids. */
	memset(&view,0,sizeof(view));
	view.abi_version = SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_DRAFT_VIEW_ABI_VERSION;
	view.descriptor_bytes = sizeof(view);
	view.requested_token_count = 3u;
	view.sequence_id = 1u;
	view.draft_token_ids = draft_ids;
	view.confidence_milli = confidence;
	view.accepted_token_counts = accepted;
	head.context.dspark_draft = &view;

	/* Walk the frame (waves of 2 rows: the verify frame splits 8 ways). */
	TestWalkStage(&head);

	/* Emissions: lane 0 accepts depth 3, lane 1 accepts depth 5. */
	for (lane = 0u; lane < TEST_LANES; lane++)
		for (j = 0u; j < TEST_ROWS_PER_LANE; j++)
		{
			uint32_t row = j * TEST_LANES + lane;
			uint32_t depth = lane == 0u ? 3u : 5u;
			head.slot.host_output_token_ids[row] = j < depth ?
				head.slot.host_token_ids[row + TEST_LANES] : 90000u + row;
		}

	/* Starved hop: clear one tap bit -> the draft refuses LOUDLY. The
	 * adapter-owned draft view above is what the successful tail fills. */
	head.state.dspark_lane_tap_mask[1] &= ~(1u << 3);
	assert(SparkGlm52DsparkRunDraftTail(&head.chain) == SPARK_STATUS_INTERNAL_ERROR);
	head.state.dspark_lane_tap_mask[1] |= 1u << 3;

	assert(SparkGlm52DsparkRunDraftTail(&head.chain) == SPARK_STATUS_OK);
	/* Accept depths folded from frame-relative rows. */
	assert(accepted[0] == 3u && accepted[1] == 5u);
	assert(view.accepted_token_counts[0] == 3u && view.accepted_token_counts[1] == 5u);
	assert(view.draft_token_count == 3u);
	assert(view.verified_row_count == TEST_ROWS_PER_LANE);
	/* Anchors staged from the FRAME base: lane 0's anchor is row 6, lane
	 * 1's row 11 - the pre-transport last-wave first_row(14) read would
	 * have staged rows 20/25, past the frame. */
	assert(TestStagedCount == TEST_LANES);
	assert(TestStagedRecords[0].token_id == TestRowToken(6u));
	assert(TestStagedRecords[1].token_id == TestRowToken(11u));
	assert(TestStagedRecords[0].backend_lane_index == 0u && TestStagedRecords[1].backend_lane_index == 1u);
	assert(TestStagedRecords[0].tap_row_index == 0u * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_TAP_ROWS_PER_LANE +
		(uint32_t)(TestRowPosition(6u) % SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_TAP_ROWS_PER_LANE));
	assert(TestStagedRecords[1].tap_row_index == 1u * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_TAP_ROWS_PER_LANE +
		(uint32_t)(TestRowPosition(11u) % SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_TAP_ROWS_PER_LANE));
	assert(TestStagedRecords[0].sequence_position == 1u && TestStagedRecords[1].sequence_position == 1u);
	assert(TestStagedRecords[0].tap_generation == 2u && TestStagedRecords[1].tap_generation == 2u);
	assert(TestDraftRequestCount == TEST_LANES);
	/* Draft ids landed in the adapter-owned view; pending verify armed. */
	assert(draft_ids[0] == 500u && draft_ids[2] == 502u);
	assert(confidence[0] == 900u);
	assert(head.state.dspark_lane_pending_verify[0] == 1u && head.state.dspark_lane_pending_verify[1] == 1u);
	assert(head.state.dspark_lane_verify_starts[0] == TestRowPosition(0u));
	assert(head.state.dspark_lane_verify_starts[1] == TestRowPosition(1u));
	assert(head.state.dspark_lane_verify_walked[0] == TEST_ROWS_PER_LANE);

	TestDestroyRun(&head);
	free(arena);
	printf("split-frame reconciliation: frame-relative accept fold + anchor staging; starved hop refuses loudly\n");
}

/* ---- frame contract --------------------------------------------------- */

static void TestDummyCompletion(void *context,const SparkModelDriverCompletion *completion)
{
	(void)context;(void)completion;
}

static SparkStatus TestValidate(struct SparkGlm52ModuleState *state,
	SparkModelDriverFrame *frame,SparkGlm52ResidentDecodeStageFrameContext *context)
{
	const SparkGlm52ResidentDecodeStageFrameContext *resolved = 0;
	(void)context;
	return(SparkGlm52ValidateFrame(state,frame,&resolved));
}

/* Prepare a well-formed frame for this stage from scratch (fresh ABI
 * fields, DSA sidebands per the geometry rule); the caller then sets the
 * tap-sideband members it wants and drives SparkGlm52ValidateFrame
 * directly. */
static void TestPrepareFrame(struct SparkGlm52ModuleState *state,
	SparkGlm52ResidentDecodeStageFrameContext *context,
	SparkModelDriverFrame *frame,SparkModelDriverBuffer *buffer,
	SparkGlm52ResidentDecodeStageBatchView *batch,uint32_t speculative)
{
	static uint32_t slots[TEST_FRAME_ROWS];
	static uint64_t positions[TEST_FRAME_ROWS];
	static uint64_t sequences[TEST_FRAME_ROWS];
	static uint32_t tokens[TEST_FRAME_ROWS];
	static uint32_t accepts[TEST_FRAME_ROWS];
	static SparkGlm52ResidentDecodeStageDsparkDraftView view;
	static uint32_t draft_ids[TEST_LANES * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_DRAFT_TOKEN_COUNT];
	uint32_t row;
	memset(frame,0,sizeof(*frame));
	memset(buffer,0,sizeof(*buffer));
	memset(context,0,sizeof(*context));
	for (row = 0u; row < TEST_FRAME_ROWS; row++)
	{
		slots[row] = TestRowSlot(row);
		positions[row] = TestRowPosition(row);
		sequences[row] = TestRowSequence(row);
		tokens[row] = TestRowToken(row);
		accepts[row] = 0u;
	}
	memset(batch,0,sizeof(*batch));
	batch->abi_version = SPARK_GLM52_RESIDENT_DECODE_STAGE_BATCH_VIEW_ABI_VERSION;
	batch->descriptor_bytes = sizeof(*batch);
	batch->row_count = TEST_FRAME_ROWS;
	batch->active_sequence_count = TEST_LANES;
	batch->token_ids = tokens;
	batch->row_resident_slots = slots;
	batch->row_positions = positions;
	batch->row_sequence_ids = sequences;
	memset(&view,0,sizeof(view));
	view.abi_version = SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_DRAFT_VIEW_ABI_VERSION;
	view.descriptor_bytes = sizeof(view);
	view.requested_token_count = 3u;
	view.sequence_id = 7u;
	view.draft_token_ids = draft_ids;
	memset(context,0,sizeof(*context));
	context->abi_version = SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION;
	context->descriptor_bytes = sizeof(*context);
	context->flags = (speculative != 0u ?
		SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_FLAG_PREFILL |
		SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_SPECULATIVE_VERIFY |
		SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DSPARK_DRAFT_AFTER :
		SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_FLAG_PREFILL) |
		(state->owns_embedding == 0u ? SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_FLAG_HIDDEN_INPUT : 0u) |
		(state->owns_final_head == 0u ? SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_FLAG_HIDDEN_OUTPUT : 0u) |
		(SparkGlm52ResidentDecodeStageRequiresSidebandInput(state->geometry,state->stage_index) != 0u ? SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_FLAG_SIDEBAND_INPUT : 0u) |
		(SparkGlm52ResidentDecodeStageRequiresSidebandOutput(state->geometry,state->stage_index) != 0u ? SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_FLAG_SIDEBAND_OUTPUT : 0u);
	context->batch = batch;
	if ( speculative != 0u )
	{
		context->dspark_draft = &view;
		context->previous_verify_accepts = accepts;
	}
	frame->flags = SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL;
	frame->active_slot_count = TEST_LANES;
	frame->new_token_count = TEST_FRAME_ROWS;
	frame->user_context = context;
	frame->execution_stream = state->execution_stream;
	frame->completion_function = TestDummyCompletion;
	if ( state->owns_final_head != 0u )
	{
		buffer->flags = SPARK_MODEL_DRIVER_BUFFER_FLAG_WRITE;
		buffer->address = (void *)0x2;
		buffer->bytes = (uint64_t)TEST_FRAME_ROWS * sizeof(uint32_t);
		frame->buffers = buffer;
		frame->buffer_count = 1u;
	}
	/* Hidden boundaries: upstream stages require both legs. */
	if ( state->owns_embedding == 0u )
	{
		context->hidden_input_bf16 = malloc((size_t)((uint64_t)TEST_FRAME_ROWS * 2u * TEST_HIDDEN * 2u));
		assert(context->hidden_input_bf16 != 0);
		context->hidden_input_bytes = (uint64_t)TEST_FRAME_ROWS * 2u * TEST_HIDDEN * 2u;
	}
	if ( state->owns_final_head == 0u )
	{
		context->hidden_output_bf16 = malloc((size_t)((uint64_t)TEST_FRAME_ROWS * 2u * TEST_HIDDEN * 2u));
		assert(context->hidden_output_bf16 != 0);
		context->hidden_output_bytes = (uint64_t)TEST_FRAME_ROWS * 2u * TEST_HIDDEN * 2u;
	}
	/* The DSA sideband (kind 1) rides its own rule; satisfy it so the tap
	 * assertions below see only their own failures. */
	if ( SparkGlm52ResidentDecodeStageRequiresSidebandInput(state->geometry,state->stage_index) != 0u )
	{
		context->sideband_input = malloc((size_t)((uint64_t)TEST_FRAME_ROWS * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_SIDEBAND_BYTES_PER_ROW));
		assert(context->sideband_input != 0);
		context->sideband_input_bytes = (uint64_t)TEST_FRAME_ROWS * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_SIDEBAND_BYTES_PER_ROW;
	}
	if ( SparkGlm52ResidentDecodeStageRequiresSidebandOutput(state->geometry,state->stage_index) != 0u )
	{
		context->sideband_output = malloc((size_t)((uint64_t)TEST_FRAME_ROWS * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_SIDEBAND_BYTES_PER_ROW));
		assert(context->sideband_output != 0);
		context->sideband_output_bytes = (uint64_t)TEST_FRAME_ROWS * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_SIDEBAND_BYTES_PER_ROW;
	}
}

static void TestFrameContract(void)
{
	static struct SparkGlm52ModuleState stage0,stage2,stage6;
	char path[256];
	SparkGlm52ResidentDecodeStageFrameContext context;
	SparkGlm52ResidentDecodeStageBatchView batch;
	SparkModelDriverFrame frame;
	SparkModelDriverBuffer buffer;
	uint64_t in_bytes,out_bytes;

	snprintf(path,sizeof(path),"%s/stage0.fp8.glm52sp",TEST_PACK_DIR);
	TestConfigureSpeculatorStage(SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT,0u,1u,path,&stage0,SPARK_STATUS_OK);
	snprintf(path,sizeof(path),"%s/stage2.fp8.glm52sp",TEST_PACK_DIR);
	TestConfigureSpeculatorStage(SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT,2u,1u,path,&stage2,SPARK_STATUS_OK);
	snprintf(path,sizeof(path),"%s/stage6.fp8.glm52sp",TEST_PACK_DIR);
	TestConfigureSpeculatorStage(SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT,SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT - 1u,1u,path,&stage6,SPARK_STATUS_OK);
	stage0.dspark_tap_wire_rows_capacity = (uint64_t)stage0.resident_sequence_capacity * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_VERIFY_ROW_COUNT;
	stage2.dspark_tap_wire_rows_capacity = stage0.dspark_tap_wire_rows_capacity;
	stage6.dspark_tap_wire_rows_capacity = stage0.dspark_tap_wire_rows_capacity;

	/* Stage 0: emits one group, consumes none. Exact extents both ways. */
	in_bytes = SparkGlm52ResidentDecodeStageDsparkTapSidebandBytes(TEST_FRAME_ROWS,stage0.dspark_tap_groups_in);
	out_bytes = SparkGlm52ResidentDecodeStageDsparkTapSidebandBytes(TEST_FRAME_ROWS,stage0.dspark_tap_groups_out);
	assert(in_bytes == 32ull && out_bytes > 32ull);
	{
		void *out = malloc((size_t)out_bytes);
		assert(out != 0);
		TestPrepareFrame(&stage0,&context,&frame,&buffer,&batch,1u);
		context.dspark_tap_sideband_output = out;
		context.dspark_tap_sideband_output_bytes = out_bytes;
		assert(TestValidate(&stage0,&frame,&context) == SPARK_STATUS_OK);
		/* Short output block: off-by-one refuses. */
		context.dspark_tap_sideband_output_bytes = out_bytes - 1u;
		assert(TestValidate(&stage0,&frame,&context) == SPARK_STATUS_CAPACITY_EXCEEDED);
		/* Missing output block on an emitting stage: starved downstream. */
		context.dspark_tap_sideband_output = 0;
		context.dspark_tap_sideband_output_bytes = 0u;
		assert(TestValidate(&stage0,&frame,&context) == SPARK_STATUS_CAPACITY_EXCEEDED);
		free(out);
	}

	/* Stage 2 (pure relay): both directions required. */
	in_bytes = SparkGlm52ResidentDecodeStageDsparkTapSidebandBytes(TEST_FRAME_ROWS,stage2.dspark_tap_groups_in);
	out_bytes = SparkGlm52ResidentDecodeStageDsparkTapSidebandBytes(TEST_FRAME_ROWS,stage2.dspark_tap_groups_out);
	assert(in_bytes == out_bytes && in_bytes > 32ull);
	{
		void *in = malloc((size_t)in_bytes);
		void *out = malloc((size_t)out_bytes);
		assert(in != 0 && out != 0);
		TestPrepareFrame(&stage2,&context,&frame,&buffer,&batch,1u);
		context.dspark_tap_sideband_input = in;
		context.dspark_tap_sideband_input_bytes = in_bytes;
		context.dspark_tap_sideband_output = out;
		context.dspark_tap_sideband_output_bytes = out_bytes;
		assert(TestValidate(&stage2,&frame,&context) == SPARK_STATUS_OK);
		/* Inbound extent disagreement: misrouted or stale block. */
		context.dspark_tap_sideband_input_bytes = in_bytes + 1u;
		assert(TestValidate(&stage2,&frame,&context) == SPARK_STATUS_CAPACITY_EXCEEDED);
		/* Missing inbound block on a consuming stage: starved pipeline. */
		context.dspark_tap_sideband_input_bytes = in_bytes;
		context.dspark_tap_sideband_input = 0;
		context.dspark_tap_sideband_input_bytes = 0u;
		assert(TestValidate(&stage2,&frame,&context) == SPARK_STATUS_CAPACITY_EXCEEDED);
		free(in);
		free(out);
	}

	/* Stage 6 (head): consumes four groups, emits none. */
	in_bytes = SparkGlm52ResidentDecodeStageDsparkTapSidebandBytes(TEST_FRAME_ROWS,stage6.dspark_tap_groups_in);
	assert(in_bytes > 32ull && stage6.dspark_tap_groups_out == 0u);
	{
		void *in = malloc((size_t)in_bytes);
		assert(in != 0);
		TestPrepareFrame(&stage6,&context,&frame,&buffer,&batch,1u);
		context.dspark_tap_sideband_input = in;
		context.dspark_tap_sideband_input_bytes = in_bytes;
		assert(TestValidate(&stage6,&frame,&context) == SPARK_STATUS_OK);
		/* An outbound block on the head: nothing to emit, refuse. */
		context.dspark_tap_sideband_output = in;
		context.dspark_tap_sideband_output_bytes = in_bytes;
		assert(TestValidate(&stage6,&frame,&context) == SPARK_STATUS_CAPACITY_EXCEEDED);
		/* A plain prefill frame carries no tap sidebands even here. */
		TestPrepareFrame(&stage6,&context,&frame,&buffer,&batch,0u);
		assert(TestValidate(&stage6,&frame,&context) == SPARK_STATUS_OK);
		context.dspark_tap_sideband_input = in;
		context.dspark_tap_sideband_input_bytes = in_bytes;
		assert(TestValidate(&stage6,&frame,&context) == SPARK_STATUS_CAPACITY_EXCEEDED);
		free(in);
	}
	printf("frame contract: role-exact tap sidebands with exact extents; non-speculating frames carry none\n");
}

/* The boundary leg's refusal law, exercised in-process over real
 * SOCK_STREAM socketpairs: every disagreement is a loud refusal with the
 * derived extent named - alien preamble (misrouted/self-consistent-but-
 * different extent), truncated stream, trailing over-send bytes, dead
 * peer - and one well-formed round-trip carries the block byte-identical. */
static void TestBoundaryLegRefusals(void)
{
	static const uint8_t payload[64] = {1u,2u,3u,4u};
	uint8_t received[64];
	uint8_t scratch[64u + 8u];
	static uint8_t lying[8u + 32u];
	static uint8_t over[8u + 64u + 3u];
	int pair[2],dead[2],fresh[2];
	ssize_t sent;
	uint64_t expected = sizeof(payload);

	assert(socketpair(AF_UNIX,SOCK_STREAM,0,pair) == 0);
	assert(socketpair(AF_UNIX,SOCK_STREAM,0,dead) == 0);
	close(dead[1]);

	/* Argument refusals. */
	assert(SparkGlm52DsparkTapSidebandLegSend(-1,payload,expected) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkGlm52DsparkTapSidebandLegSend(pair[0],0,expected) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkGlm52DsparkTapSidebandLegSend(pair[0],payload,0u) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkGlm52DsparkTapSidebandLegReceive(-1,received,expected,scratch,sizeof(scratch)) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkGlm52DsparkTapSidebandLegReceive(pair[1],0,expected,scratch,sizeof(scratch)) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkGlm52DsparkTapSidebandLegReceive(pair[1],received,0u,scratch,sizeof(scratch)) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkGlm52DsparkTapSidebandLegReceive(pair[1],received,expected,0,sizeof(scratch)) == SPARK_STATUS_INVALID_ARGUMENT);

	/* Well-formed round-trip: byte-identical block out the far end. */
	assert(SparkGlm52DsparkTapSidebandLegSend(pair[0],payload,expected) == SPARK_STATUS_OK);
	assert(SparkGlm52DsparkTapSidebandLegReceive(pair[1],received,expected,scratch,sizeof(scratch)) == SPARK_STATUS_OK);
	assert(memcmp(received,payload,sizeof(payload)) == 0);

	/* Truncated stream: the declared extent never fully arrives. */
	SparkGlm52DsparkTapLegStoreLe64(lying,expected);
	memcpy(lying + 8u,payload,16u); /* half the payload, then stop */
	sent = send(pair[0],lying,(size_t)(8u + 16u),MSG_NOSIGNAL);
	assert(sent == (ssize_t)(8u + 16u));
	shutdown(pair[0],SHUT_WR);
	assert(SparkGlm52DsparkTapSidebandLegReceive(pair[1],received,expected,scratch,sizeof(scratch)) == SPARK_STATUS_VALIDATION_FAILED);
	close(pair[0]);
	close(pair[1]);
	assert(socketpair(AF_UNIX,SOCK_STREAM,0,fresh) == 0);
	pair[0] = fresh[0];
	pair[1] = fresh[1];

	/* Alien preamble: a self-consistent sender that derived a DIFFERENT
	 * extent than ours - refused unread. */
	SparkGlm52DsparkTapLegStoreLe64(lying,32u);
	memset(lying + 8u,0xa5,sizeof(lying) - 8u);
	sent = send(pair[0],lying,sizeof(lying),MSG_NOSIGNAL);
	assert(sent == (ssize_t)sizeof(lying));
	assert(SparkGlm52DsparkTapSidebandLegReceive(pair[1],received,expected,scratch,sizeof(scratch)) == SPARK_STATUS_VALIDATION_FAILED);
	/* The refusal was UNREAD by law, so the half-consumed frame stays
	 * pending on this leg - every negative case gets a fresh pair. */
	close(pair[0]);
	close(pair[1]);
	assert(socketpair(AF_UNIX,SOCK_STREAM,0,fresh) == 0);
	pair[0] = fresh[0];
	pair[1] = fresh[1];

	/* Over-send behind a CORRECT extent: the post-frame peek catches it. */
	SparkGlm52DsparkTapLegStoreLe64(over,expected);
	memset(over + 8u,0x5a,sizeof(over) - 8u);
	sent = send(pair[0],over,sizeof(over),MSG_NOSIGNAL);
	assert(sent == (ssize_t)sizeof(over));
	assert(SparkGlm52DsparkTapSidebandLegReceive(pair[1],received,expected,scratch,sizeof(scratch)) == SPARK_STATUS_VALIDATION_FAILED);

	/* Dead peer: recv returns zero - refused, never guessed around. */
	assert(SparkGlm52DsparkTapSidebandLegReceive(dead[0],received,expected,scratch,sizeof(scratch)) == SPARK_STATUS_VALIDATION_FAILED);

	close(pair[0]);
	close(pair[1]);
	close(dead[0]);
	printf("boundary leg refusals: truncation, alien extent, over-send and dead peer all refuse loudly; well-formed block round-trips byte-identical\n");
}

/*
 * THE REAL HANDOFF: stages 0..5 walk in a CHILD process; stage 5's final
 * outbound block crosses the process boundary through the leg (one
 * SEQPACKET message; each end derives its extent independently); the
 * PARENT's head stage ingests it and its drafter arena must be
 * BYTE-IDENTICAL to the TP1 single-stage run of the same tokens.
 */
static void TestTransportProcessBoundary(void)
{
	static TestStageRun pp7[SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT];
	TestStageRun tp1;
	SparkGlm52ResidentDecodeStageBatchView batch;
	static uint32_t slots[TEST_FRAME_ROWS];
	static uint64_t positions[TEST_FRAME_ROWS];
	static uint64_t sequences[TEST_FRAME_ROWS];
	static uint32_t tokens[TEST_FRAME_ROWS];
	uint16_t *arena_tp1,*arena_pp7;
	size_t arena_bytes;
	uint64_t head_in_bytes;
	static uint8_t scratch[
		SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_TAP_SIDEBAND_HEADER_BYTES +
		4u * TEST_FRAME_ROWS * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_TAP_SIDEBAND_RECORD_BYTES +
		4u * TEST_FRAME_ROWS * TEST_HIDDEN * 2u];
	int pair[2];
	pid_t pid;
	int status = -1;
	uint32_t stage;

	TestBuildBatch(&batch,slots,positions,sequences,tokens);
	arena_bytes = (size_t)8u * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_TAP_ROWS_PER_LANE * TEST_TAP_COUNT * TEST_HIDDEN * sizeof(uint16_t);
	arena_tp1 = (uint16_t *)calloc(1u,arena_bytes);
	arena_pp7 = (uint16_t *)calloc(1u,arena_bytes);
	assert(arena_tp1 != 0 && arena_pp7 != 0);

	/* TP1 reference first - the golden lives in the parent across fork. */
	TestBuildRun(&tp1,SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_COUNT,0u,arena_tp1,&batch);
	TestWalkStage(&tp1);

	/* Both halves exist before fork; the child inherits copy-on-write and
	 * touches only its own half. */
	for (stage = 0u; stage < SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT; stage++)
	{
		TestStageRun *run = &pp7[stage];
		TestBuildRun(run,SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT,stage,
			stage + 1u == SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT ? arena_pp7 : 0,&batch);
		if ( stage + 1u < SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT )
		{
			run->outbound_bytes = run->state.dspark_tap_groups_out > 0u ?
				SparkGlm52ResidentDecodeStageDsparkTapSidebandBytes(TEST_FRAME_ROWS,run->state.dspark_tap_groups_out) : 0u;
			if ( run->outbound_bytes > 0u )
			{
				run->outbound = (uint8_t *)calloc(1u,(size_t)run->outbound_bytes);
				assert(run->outbound != 0);
				run->context.dspark_tap_sideband_output = run->outbound;
				run->context.dspark_tap_sideband_output_bytes = run->outbound_bytes;
			}
			if ( stage > 0u )
			{
				run->context.dspark_tap_sideband_input = pp7[stage - 1u].outbound;
				run->context.dspark_tap_sideband_input_bytes = pp7[stage - 1u].outbound_bytes;
			}
		}
		else
		{
			head_in_bytes = SparkGlm52ResidentDecodeStageDsparkTapSidebandBytes(TEST_FRAME_ROWS,run->state.dspark_tap_groups_in);
			run->inbound = (uint8_t *)calloc(1u,(size_t)head_in_bytes);
			assert(run->inbound != 0);
		}
	}

	assert(socketpair(AF_UNIX,SOCK_STREAM,0,pair) == 0);
	pid = fork();
	assert(pid >= 0);
	if ( pid == 0 )
	{
		/* Child: upstream stages chained in-process (those hops are the
		 * in-process proof's), then ship stage 5's final outbound block
		 * across the REAL boundary leg. Any failure exits non-zero. */
		uint64_t shipped_bytes;
		SparkStatus ship;
		close(pair[1]); /* the child never reads its own leg */
		for (stage = 0u; stage + 1u < SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT; stage++)
			TestWalkStage(&pp7[stage]);
		shipped_bytes = pp7[SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT - 2u].outbound_bytes;
		ship = SparkGlm52DsparkTapSidebandLegSend(pair[0],
			pp7[SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT - 2u].outbound,shipped_bytes);
		_exit(ship == SPARK_STATUS_OK ? 0 : 70);
	}
	/* Parent: drop our copy of the leg's write end NOW - a dead child then
	 * surfaces as loud EOF at the receiver instead of an eternal wait -
	 * receive through the leg into the head's input buffer, run the head
	 * stage, demand the TP1 arena byte-for-byte, and only then reap. */
	close(pair[0]);
	{
		TestStageRun *head = &pp7[SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT - 1u];
		assert(SparkGlm52DsparkTapSidebandLegReceive(pair[1],head->inbound,head_in_bytes,scratch,sizeof(scratch)) == SPARK_STATUS_OK);
		head->context.dspark_tap_sideband_input = head->inbound;
		head->context.dspark_tap_sideband_input_bytes = head_in_bytes;
		TestWalkStage(head);
		assert(waitpid(pid,&status,0) == pid);
		assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
	}
	assert(memcmp(arena_tp1,arena_pp7,arena_bytes) == 0);
	assert(pp7[SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT - 1u].state.dspark_lane_tap_mask[0] == TEST_FULL_MASK);
	assert(pp7[SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT - 1u].state.dspark_lane_tap_mask[1] == TEST_FULL_MASK);

	for (stage = 0u; stage < SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT; stage++)
		TestDestroyRun(&pp7[stage]);
	TestDestroyRun(&tp1);
	free(arena_tp1);
	free(arena_pp7);
	close(pair[1]); /* the leg's write end was dropped right after fork */
	printf("boundary handoff: child-shipped kind-2 block crosses the process boundary verbatim; parent drafter arena byte-identical to TP1\n");
}

int main(void)
{
	setvbuf(stdout,0,_IOLBF,0);
	TestTopologyRoles();
	TestWireFormat();
	TestTransportSequenceParity();
	TestSplitFrameDraftTail();
	TestFrameContract();
	TestBoundaryLegRefusals();
	TestTransportProcessBoundary();
	printf("PASS glm52 pp7 tap transport: roles, wire record, byte-identical cross-stage sequence, split-frame reconciliation, frame contract, boundary-leg refusals, two-process handoff\n");
	return(0);
}

/* Link-only stubs: the dry path never reaches the CUDA launchers, but the
 * whole module translation unit is linked, so their symbols must resolve. */
#include <cuda_runtime.h>
cudaError_t SparkStageLaunchAccumAdd(cudaStream_t stream,void *destination_bf16,const void *source_bf16,uint32_t row_count,uint32_t width)
{
	(void)stream;(void)destination_bf16;(void)source_bf16;(void)row_count;(void)width;
	return(1);
}
cudaError_t SparkStageLaunchAccumU64Max(cudaStream_t stream,uint64_t *destination,const uint64_t *source,uint32_t element_count)
{
	(void)stream;(void)destination;(void)source;(void)element_count;
	return(1);
}
int32_t SparkGlm52ConfigureCudaModule(uint32_t *multiprocessor_count)
{
	(void)multiprocessor_count;
	return(-1);
}
int32_t SparkGlm52LaunchCudaWaveBegin(const SparkGlm52CudaWave *wave) { (void)wave; return(-1); }
int32_t SparkGlm52LaunchCudaLayerAttention(const SparkGlm52CudaWave *wave,uint32_t local_layer) { (void)wave;(void)local_layer; return(-1); }
int32_t SparkGlm52LaunchCudaLayerMlp(const SparkGlm52CudaWave *wave,uint32_t local_layer) { (void)wave;(void)local_layer; return(-1); }
int32_t SparkGlm52LaunchCudaWaveHead(const SparkGlm52CudaWave *wave) { (void)wave; return(-1); }
/* Device launchers the walk chain references (cuda.cu definitions): the
 * transport paths above never reach them, but the symbols must resolve. */
cudaError_t SparkGlm52LaunchHeadMaxlocPack(cudaStream_t stream,const float *scores,const uint32_t *token_ids,uint64_t *maxloc,uint32_t row_count,uint32_t rank_offset)
{
	(void)stream;(void)scores;(void)token_ids;(void)maxloc;(void)row_count;(void)rank_offset;
	return(1);
}
cudaError_t SparkGlm52LaunchHeadMaxlocUnpack(cudaStream_t stream,const uint64_t *maxloc,uint32_t *token_ids,uint32_t row_count)
{
	(void)stream;(void)maxloc;(void)token_ids;(void)row_count;
	return(1);
}
