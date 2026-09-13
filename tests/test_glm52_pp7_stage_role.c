/* PP7 stage-role config gate (host, no GPU): the adapter must be able to
 * LOAD a generated per-(stage,rank) config - WHICH stage of the [12,11x6]
 * split this process serves - cross-check its recorded geometry against
 * the compiled table, wire the module node context from that table (never
 * from the JSON), and pin the drafter to the head-owning stage.
 *
 * White-box on purpose: the harness includes the adapter translation unit
 * directly so SparkGlm52ServingLoadConfiguration, ApplyStageRole and
 * ArmSpeculation are drivable without loading a model driver (this dry
 * path never reaches dlopen; the neutral validators and the loader are
 * link stubs pinned for real by test_model_serving_adapter).
 *
 * Built twice by the Makefile:
 *   build/test_glm52_pp7_stage_role       (default 8x8 fanout defines)
 *   build/test_glm52_pp7_stage_role_tp1   (-DSPARK_GLM52_SERVING_STAGE_COUNT=1
 *                                          -DSPARK_GLM52_SERVING_TP_DEGREE=1,
 *                                          the qwen36-doctrine TP1 variant)
 * The TP1 variant is the only one where the drafter may arm at all, so it
 * carries the non-head REFUSAL and the head-stage ARMED pins; the default
 * binary proves those same configs still refuse under TP>1 (doctrine
 * unchanged) plus every config-law pin.
 *
 * The positive configs mirror tools/glm52_gen_deployment.py member-for-
 * member (schema 4 + the pp_* stage-role stanza), so "the generator's
 * output loads" is a pinned property, not an assertion by inspection. */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "spark_glm52_serving_adapter.c"

#define TEST_DIR "build/tmp/glm52_pp7_stage_role"

static const uint32_t TestPp7Counts[SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT] =
	SPARK_GLM52_MODEL_DSPARK_PP_STAGE_LAYER_COUNTS_INITIALIZER;
static const uint32_t TestPp7Firsts[SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT] =
	SPARK_GLM52_MODEL_DSPARK_PP_STAGE_FIRST_LAYER_INITIALIZER;

static int g_checks = 0;

#define CHECK(condition) \
	do { g_checks++; if ( !(condition) ) { \
		(void)fprintf(stderr,"FAIL %s:%d check #%d: %s\n",__FILE__,__LINE__,g_checks,#condition); \
		return(1); } } while (0)

/* One firmware config exactly as the deployment generator emits it.
 * mutation != 0 rewrites one value AFTER generation (negative space);
 * extra_member adds an unknown key (exact-member law). */
static int TestWriteConfig(const char *path,uint32_t tp_degree,
	uint32_t stage,uint32_t rank,int use_pp_stanza,const char *mutation,
	const char *extra_member,int force_collective)
{
	FILE *file;
	file = fopen(path,"w");
	CHECK(file != 0);
	fprintf(file,"{\n");
	fprintf(file," \"schema_version\": %u,\n",
		SPARK_GLM52_SERVING_ADAPTER_CONFIGURATION_SCHEMA_VERSION);
	fprintf(file," \"model_revision\": \"%s\",\n",GLM52_MODEL_REVISION);
	fprintf(file," \"expert_weight_codec\": \"%s\",\n",GLM52_EXPERT_CODEC_NAME);
	if ( use_pp_stanza )
		fprintf(file," \"stage_pack_path\": \"packs/glm52_stage.%u.tp%u.fp8.glm52sp\",\n",stage,rank);
	else
		fprintf(file," \"stage_pack_path\": \"packs/glm52_tp8_rank%02u.fp8.glm52sp\",\n",rank);
	fprintf(file," \"max_sequence_positions\": 4096,\n");
	fprintf(file," \"execution_row_capacity\": 16,\n");
	fprintf(file," \"tp_degree\": %u,\n",tp_degree);
	fprintf(file," \"tp_rank\": %u,\n",rank);
	if ( (force_collective > 0) || (force_collective >= 0 && tp_degree > 1u) )
	{
		/* Minimal valid nccl-backend collective (base-member set).
		 * force_collective > 0 emits one even at tp_degree 1 (the
		 * FORBIDDEN-at-tp1 negative); force_collective < 0 suppresses
		 * one even above tp1 (the REQUIRED-above-tp1 negative). */
		uint32_t peer;
		fprintf(file," \"tp_collective\": {\n");
		fprintf(file,"  \"backend\": \"nccl\",\n");
		fprintf(file,"  \"backend_module_path\": \"lib/hidden_transport.so\",\n");
		fprintf(file,"  \"collective_identifier\": %llu,\n",
			(unsigned long long)(8811223344556678ULL + stage));
		fprintf(file,"  \"listen_port\": %u,\n",(unsigned)(63620u + stage * tp_degree + rank));
		fprintf(file,"  \"connect_timeout_milli\": 180000,\n");
		fprintf(file,"  \"operation_timeout_milli\": 30000,\n");
		fprintf(file,"  \"peer_hosts\": [");
		for (peer=0u; peer<tp_degree; peer++)
			fprintf(file,"%s\"spark%u\"",peer != 0u ? "," : "",(unsigned)peer);
		fprintf(file,"],\n");
		fprintf(file,"  \"peer_ports\": [");
		for (peer=0u; peer<tp_degree; peer++)
			fprintf(file,"%s%u",peer != 0u ? "," : "",(unsigned)(63620u + stage * tp_degree + peer));
		fprintf(file,"]\n");
		fprintf(file," },\n");
	}
	fprintf(file," \"speculation_enabled\": %s,\n",use_pp_stanza && stage == SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT - 1u ? "true" : "false");
	fprintf(file," \"speculation_draft_count\": 7,\n");
	fprintf(file," \"dspark_pack_path\": \"packs/glm52_dspark_drafter\"");
	if ( use_pp_stanza )
	{
		uint32_t first = TestPp7Firsts[stage];
		uint32_t count = TestPp7Counts[stage];
		uint32_t stage_count = (uint32_t)SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT;
		uint32_t stage_index = stage;
		if ( mutation != 0 )
		{
			if ( strcmp(mutation,"count13") == 0 )
				stage_count = 13u;
			else if ( strcmp(mutation,"index_oob") == 0 )
				stage_index = (uint32_t)SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT;
			else if ( strcmp(mutation,"first33") == 0 && stage == 3u )
				first = 33u;
			else if ( strcmp(mutation,"layers12") == 0 && stage == 4u )
				count = 12u;
		}
		fprintf(file,",\n \"pp_stage_count\": %u",stage_count);
		fprintf(file,",\n \"pp_stage_index\": %u",stage_index);
		fprintf(file,",\n \"pp_stage_first_layer\": %u",first);
		fprintf(file,",\n \"pp_stage_layer_count\": %u",count);
	}
	if ( extra_member != 0 )
		fprintf(file,",\n \"%s\": 1",extra_member);
	fprintf(file,"\n}\n");
	CHECK(fclose(file) == 0);
	return(0);
}

/* Drive LoadConfiguration + ApplyStageRole exactly as Initialize does
 * (minus the driver load this dry path never reaches). */
static SparkStatus TestLoadAndApply(
	const SparkModelServingAdapterConfiguration *configuration,
	SparkGlm52ServingState *state)
{
	uint32_t max_sequence_positions,execution_row_capacity,tp_degree,tp_rank;
	SparkStatus status;
	memset(state,0,sizeof(*state));
	state->resident_sequence_capacity = configuration->runtime_limits.resident_sequence_capacity;
	state->pipeline_slot_count = configuration->runtime_limits.max_inflight_submission_count;
	state->max_active_sequence_count = configuration->runtime_limits.max_active_sequence_count;
	state->max_input_row_count = configuration->runtime_limits.max_input_row_count;
	status = SparkGlm52ServingLoadConfiguration(configuration->adapter_configuration_path,
		configuration->runtime_root,state,&max_sequence_positions,&execution_row_capacity,
		&tp_degree,&tp_rank);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SparkGlm52ServingApplyStageRole(state,configuration,max_sequence_positions,
		execution_row_capacity,tp_degree,tp_rank));
}

static void TestFillConfiguration(
	SparkModelServingAdapterConfiguration *configuration,
	const char *config_path,const char *runtime_root,
	uint32_t node_stage,uint32_t node_rank)
{
	memset(configuration,0,sizeof(*configuration));
	configuration->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	configuration->descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_CONFIGURATION_BYTES;
	configuration->rank_index = node_rank;
	configuration->stage_index = node_stage;
	configuration->runtime_limits.max_inflight_submission_count = 2u;
	configuration->runtime_limits.max_active_sequence_count = 16u;
	configuration->runtime_limits.max_input_row_count = 16u;
	configuration->runtime_limits.resident_sequence_capacity = 16u;
	configuration->runtime_limits.kv_logical_page_capacity = 0u;
	configuration->runtime_limits.kv_physical_page_capacity = 0u;
	configuration->runtime_root = runtime_root;
	configuration->node_id = "node";
	configuration->node_target = "target";
	configuration->adapter_configuration_path = config_path;
	configuration->driver_shared_object_path = "unused.so";
	configuration->driver_program_name = SPARK_GLM52_SERVING_PROGRAM_NAME;
	configuration->kv_backing_directory = "/tmp";
}

int main(void)
{
	SparkModelServingAdapterConfiguration configuration;
	SparkGlm52ServingState state;
	char path[512];
	char runtime_root[256];
	uint32_t stage;

	CHECK(getcwd(runtime_root,sizeof(runtime_root)) != 0);
	CHECK(strlen(runtime_root) + sizeof("/" TEST_DIR "/root") < sizeof(runtime_root));
	(void)mkdir(TEST_DIR,0755);
	(void)mkdir(TEST_DIR "/root",0755);
	strcat(runtime_root,"/" TEST_DIR "/root");

#if SPARK_GLM52_SERVING_TP_DEGREE != 1u

	/* === Default (8x8 flat fanout) build === */

	/* Flat shape unchanged: a tp8 config with no pp_* stanza still maps
	 * tp_rank onto the flat node index and fills the single-stage context. */
	TestWriteConfig(TEST_DIR "/flat.json",SPARK_GLM52_SERVING_TP_DEGREE,0u,3u,0,0,0,0);
	TestFillConfiguration(&configuration,TEST_DIR "/flat.json",runtime_root,3u,3u);
	CHECK(TestLoadAndApply(&configuration,&state) == SPARK_STATUS_OK);
	CHECK(state.pp_role_present == 0u);
	CHECK(state.node_context.stage_count == SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_COUNT);
	CHECK(state.node_context.stage_index == 0u);
	CHECK(state.node_context.first_layer_index == 0u);
	CHECK(state.node_context.layer_count == SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_LAYERS_PER_STAGE);
	/* Flat identity law intact: tp_rank must equal the node's stage_index. */
	TestFillConfiguration(&configuration,TEST_DIR "/flat.json",runtime_root,4u,3u);
	CHECK(TestLoadAndApply(&configuration,&state) == SPARK_STATUS_SCHEMA_ERROR);

	/* Every [12,11x6] stage config now LOADS and wires its stage role from
	 * the compiled geometry table - the property the generated tree never
	 * had before (schema desync + unknown members refused it outright). */
	for (stage=0u; stage<SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT; stage++)
	{
		snprintf(path,sizeof(path),TEST_DIR "/stage%u.json",stage);
		TestWriteConfig(path,SPARK_GLM52_SERVING_TP_DEGREE,stage,(stage*2u+1u)%8u,1,0,0,0);
		TestFillConfiguration(&configuration,path,runtime_root,stage,(stage*2u+1u)%8u);
		CHECK(TestLoadAndApply(&configuration,&state) == SPARK_STATUS_OK);
		CHECK(state.pp_role_present == 1u);
		CHECK(state.pp_stage_count == (uint32_t)SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT);
		CHECK(state.pp_stage_index == stage);
		CHECK(state.node_context.stage_count == (uint32_t)SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT);
		CHECK(state.node_context.stage_index == stage);
		CHECK(state.node_context.first_layer_index == TestPp7Firsts[stage]);
		CHECK(state.node_context.layer_count == TestPp7Counts[stage]);
	}
	/* Node identity mismatch: the resident names another stage -> refuse. */
	TestFillConfiguration(&configuration,TEST_DIR "/stage2.json",runtime_root,5u,2u);
	CHECK(TestLoadAndApply(&configuration,&state) == SPARK_STATUS_SCHEMA_ERROR);

#else

	/* === TP1 variant build (the shape that may carry a drafter) === */

	/* Flat TP1 shape unchanged. */
	TestWriteConfig(TEST_DIR "/tp1_flat.json",1u,0u,0u,0,0,0,0);
	TestFillConfiguration(&configuration,TEST_DIR "/tp1_flat.json",runtime_root,0u,0u);
	CHECK(TestLoadAndApply(&configuration,&state) == SPARK_STATUS_OK);
	CHECK(state.pp_role_present == 0u);
	CHECK(state.node_context.layer_count == SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_LAYERS_PER_STAGE);

	/* Every stage of the split loads at tp_degree 1 with no collective. */
	for (stage=0u; stage<SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT; stage++)
	{
		snprintf(path,sizeof(path),TEST_DIR "/tp1_stage%u.json",stage);
		TestWriteConfig(path,1u,stage,0u,1,0,0,0);
		TestFillConfiguration(&configuration,path,runtime_root,stage,0u);
		CHECK(TestLoadAndApply(&configuration,&state) == SPARK_STATUS_OK);
		CHECK(state.node_context.stage_index == stage);
		CHECK(state.node_context.first_layer_index == TestPp7Firsts[stage]);
		CHECK(state.node_context.layer_count == TestPp7Counts[stage]);
	}
	/* Drafter PIN: speculation requested on a non-head stage refuses
	 * loudly - the drafter engine lives ONLY on the head-owning stage. */
	snprintf(path,sizeof(path),TEST_DIR "/tp1_relay.json");
	TestWriteConfig(path,1u,4u,0u,1,0,0,0);
	TestFillConfiguration(&configuration,path,runtime_root,4u,0u);
	CHECK(TestLoadAndApply(&configuration,&state) == SPARK_STATUS_OK);
	state.speculate = 1u; /* as speculation_enabled=true would set it */
	CHECK(SparkGlm52ServingArmSpeculation(&state,1u) == SPARK_STATUS_UNSUPPORTED);
	CHECK(state.spec_stamp_scratch == 0);
	/* Head-owning stage ARMED: the flagship split's drafter process. */
	snprintf(path,sizeof(path),TEST_DIR "/tp1_head.json");
	TestWriteConfig(path,1u,SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT - 1u,0u,1,0,0,0);
	TestFillConfiguration(&configuration,path,runtime_root,
		SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT - 1u,0u);
	CHECK(TestLoadAndApply(&configuration,&state) == SPARK_STATUS_OK);
	CHECK(state.speculate == 1u); /* config said true on the head stage */
	CHECK(SparkGlm52ServingArmSpeculation(&state,1u) == SPARK_STATUS_OK);
	CHECK(state.spec_stamp_scratch != 0);
	free(state.spec_stamp_scratch);
	state.spec_stamp_scratch = 0;

#endif

	/* === Negative space: the stanza law (both builds run these at their
	 * native tp degree via the same writer) === */
#if SPARK_GLM52_SERVING_TP_DEGREE != 1u
#define TEST_TP SPARK_GLM52_SERVING_TP_DEGREE
#define TEST_RANK 3u
#else
#define TEST_TP 1u
#define TEST_RANK 0u
#endif
	/* Partial stanza: two of four members is a refusal, not a guess. */
	TestWriteConfig(TEST_DIR "/partial.json",TEST_TP,1u,TEST_RANK,1,0,0,0);
	{
		/* Rewrite without the last member. */
		FILE *file = fopen(TEST_DIR "/partial.json","r");
		char buffer[4096];
		size_t bytes;
		CHECK(file != 0);
		bytes = fread(buffer,1,sizeof(buffer) - 1u,file);
		buffer[bytes] = '\0';
		CHECK(fclose(file) == 0);
		{
			char *cut = strstr(buffer,"\"pp_stage_layer_count\"");
			char *comma;
			CHECK(cut != 0);
			/* Cut at the comma PRECEDING the member regardless of the
			 * exact whitespace the writer used. */
			comma = cut;
			while (comma > buffer && *comma != ',')
				comma--;
			CHECK(comma > buffer && *comma == ',');
			*comma = '\0';
			strcat(buffer,"\n}\n");
		}
		file = fopen(TEST_DIR "/partial.json","w");
		CHECK(file != 0);
		fwrite(buffer,1,strlen(buffer),file);
		CHECK(fclose(file) == 0);
	}
	snprintf(path,sizeof(path),TEST_DIR "/partial.json");
	TestFillConfiguration(&configuration,path,runtime_root,1u,TEST_RANK);
	CHECK(TestLoadAndApply(&configuration,&state) == SPARK_STATUS_SCHEMA_ERROR);

	/* Unknown geometry: the retired uniform PP13 grid cannot come back
	 * through a config file either. */
	TestWriteConfig(TEST_DIR "/pp13.json",TEST_TP,3u,TEST_RANK,1,"count13",0,0);
	TestFillConfiguration(&configuration,TEST_DIR "/pp13.json",runtime_root,3u,TEST_RANK);
	CHECK(TestLoadAndApply(&configuration,&state) == SPARK_STATUS_SCHEMA_ERROR);

	/* Geometry disagreement: prefix-sum violation (33 vs 34) ... */
	TestWriteConfig(TEST_DIR "/first33.json",TEST_TP,3u,TEST_RANK,1,"first33",0,0);
	TestFillConfiguration(&configuration,TEST_DIR "/first33.json",runtime_root,3u,TEST_RANK);
	CHECK(TestLoadAndApply(&configuration,&state) == SPARK_STATUS_SCHEMA_ERROR);
	/* ... and a layer-count lie (12 vs 11). */
	TestWriteConfig(TEST_DIR "/layers12.json",TEST_TP,4u,TEST_RANK,1,"layers12",0,0);
	TestFillConfiguration(&configuration,TEST_DIR "/layers12.json",runtime_root,4u,TEST_RANK);
	CHECK(TestLoadAndApply(&configuration,&state) == SPARK_STATUS_SCHEMA_ERROR);

	/* Out-of-range stage index. */
	TestWriteConfig(TEST_DIR "/oob.json",TEST_TP,6u,TEST_RANK,1,"index_oob",0,0);
	TestFillConfiguration(&configuration,TEST_DIR "/oob.json",runtime_root,6u,TEST_RANK);
	CHECK(TestLoadAndApply(&configuration,&state) == SPARK_STATUS_SCHEMA_ERROR);

	/* Exact-member law intact WITH the new members present: one alien key
	 * still refuses the whole document. */
	TestWriteConfig(TEST_DIR "/alien.json",TEST_TP,2u,TEST_RANK,1,0,"bogus_member",0);
	TestFillConfiguration(&configuration,TEST_DIR "/alien.json",runtime_root,2u,TEST_RANK);
	CHECK(TestLoadAndApply(&configuration,&state) == SPARK_STATUS_SCHEMA_ERROR);

#if SPARK_GLM52_SERVING_TP_DEGREE != 1u
	/* Zero collective_identifier: not a configuration - the module would
	 * elide every reduce and serve rank-local math on a TP-degree pack.
	 * White-box sibling of the kv_backing_directory pin in pp7_configure;
	 * this is the include-the-TU fixture the improvement-pass seed asked
	 * for, so the refusal is pinned against the real parse path. (Only
	 * above tp1 - at tp1 the stanza itself is forbidden, pinned below.) */
	TestWriteConfig(TEST_DIR "/zeroid.json",TEST_TP,1u,TEST_RANK,1,0,0,0);
	{
		FILE *file = fopen(TEST_DIR "/zeroid.json","r");
		char buffer[4096];
		size_t bytes;
		char *hit,*digit;
		const char needle[] = "collective_identifier";
		CHECK(file != 0);
		bytes = fread(buffer,1,sizeof(buffer)-1u,file);
		buffer[bytes] = '\0';
		fclose(file);
		hit = strstr(buffer,needle);
		CHECK(hit != 0);
		digit = hit + strlen(needle);
		while (*digit != '\0' && (*digit < '0' || *digit > '9'))
			digit++;
		CHECK(*digit >= '0' && *digit <= '9');
		*digit++ = '0';
		while (*digit >= '0' && *digit <= '9')
			*digit++ = ' ';
		file = fopen(TEST_DIR "/zeroid.json","w");
		CHECK(file != 0);
		fwrite(buffer,1,strlen(buffer),file);
		CHECK(fclose(file) == 0);
	}
	TestFillConfiguration(&configuration,TEST_DIR "/zeroid.json",runtime_root,1u,TEST_RANK);
	CHECK(TestLoadAndApply(&configuration,&state) == SPARK_STATUS_SCHEMA_ERROR);
#endif

#if SPARK_GLM52_SERVING_TP_DEGREE != 1u
	/* Collective REQUIRED above tp1: a degree-8 config without the stanza
	 * refuses. (The four-schema law also fixed the inverse latent hole:
	 * with one member list that omitted tp_collective, NO generated
	 * multi-rank config could ever load - the count never matched.) */
	TestWriteConfig(TEST_DIR "/nocoll.json",SPARK_GLM52_SERVING_TP_DEGREE,0u,3u,0,0,0,-1);
	TestFillConfiguration(&configuration,TEST_DIR "/nocoll.json",runtime_root,3u,3u);
	CHECK(TestLoadAndApply(&configuration,&state) == SPARK_STATUS_SCHEMA_ERROR);
#else
	/* Collective FORBIDDEN at tp1: a single-rank process carrying a peer
	 * topology is a deployment lie - refused loudly, never parsed. */
	TestWriteConfig(TEST_DIR "/coll_tp1.json",1u,6u,0u,1,0,0,1);
	TestFillConfiguration(&configuration,TEST_DIR "/coll_tp1.json",runtime_root,6u,0u);
	CHECK(TestLoadAndApply(&configuration,&state) == SPARK_STATUS_SCHEMA_ERROR);
#endif

	printf("PASS glm52 pp7 stage-role configs: "
#if SPARK_GLM52_SERVING_TP_DEGREE != 1u
		"flat unchanged, all 7 [12,11x6] stage roles wire the compiled triple"
#else
		"TP1 variant: all 7 stages load, drafter armed on head stage ONLY "
		"(non-head refused loudly)"
#endif
		", stanza law + geometry cross-checks refuse every negative (%d checks)\n",
		g_checks);
	return(0);
}

/* Link-only stubs: this dry path never loads a model driver nor runs the
 * submission path, but the whole adapter TU is linked, so the symbols it
 * references must resolve. The real implementations live in
 * src/spark_driver_loader.c and runtime/model_serving_adapter.c and are
 * pinned by test_model_serving_adapter - they are NOT re-proven here. */
void SparkLoadedModelDriverReset(SparkLoadedModelDriver *driver)
{
	(void)driver;
}
SparkStatus SparkLoadModelDriver(const char *driver_path,const char *expected_target,
	SparkLoadedModelDriver *driver,char *error_buffer,uint32_t error_buffer_bytes)
{
	(void)driver_path;(void)expected_target;(void)driver;(void)error_buffer;(void)error_buffer_bytes;
	return(SPARK_STATUS_INTERNAL_ERROR); /* never reached on this dry path */
}
void SparkUnloadModelDriver(SparkLoadedModelDriver *driver)
{
	(void)driver;
}
const SparkModelDriverProgramDescriptor *SparkFindLoadedModelDriverProgram(
	const SparkLoadedModelDriver *driver,const char *program_name)
{
	(void)driver;(void)program_name;
	return(0); /* never reached on this dry path */
}
SparkStatus SparkModelServingAdapterValidateRuntimeLimits(
	const SparkModelServingAdapterDescriptor *descriptor,
	const SparkModelServingRuntimeLimits *runtime_limits)
{
	(void)descriptor;(void)runtime_limits;
	return(SPARK_STATUS_OK); /* neutral contract pinned elsewhere */
}
SparkStatus SparkModelServingAdapterValidateRuntimeSubmission(
	const SparkModelServingAdapterDescriptor *descriptor,
	const SparkModelServingRuntimeLimits *runtime_limits,
	const SparkModelServingSubmission *submission)
{
	(void)descriptor;(void)runtime_limits;(void)submission;
	return(SPARK_STATUS_INTERNAL_ERROR); /* submit path never driven here */
}
SparkStatus SparkModelServingAdapterStreamOrderedProgress(
	void *adapter_state,uint32_t maximum_step_count)
{
	(void)adapter_state;(void)maximum_step_count;
	return(SPARK_STATUS_INTERNAL_ERROR); /* progress path never driven here */
}
