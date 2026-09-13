/* MTP>0 pack-load regression guard: mtp_hc_head_scale_value must be seeded
 * from the pack's 1x1 MTP_HC_HEAD_SCALE tensor by the real load path.
 *
 * The P0-A fix seeds BOTH gate scalars inside SparkDsv4ModuleBindGlobal;
 * tests/test_dsv4_head_scale_value.c pins the bind seam directly. This test
 * goes one level up and drives the actual SparkDsv4ModuleLoadPack pipeline
 * over a synthesized MTP>0 stage pack:
 *
 *   header geometry compare -> directory read -> per-entry validation ->
 *   device-region load -> global bind (final-head gate scales AND the
 *   draft-stage extras incl. the MTP gate scale) -> VerifyCoverage.
 *
 * The stage slice is the model tail (first_layer_index =
 * SPARK_DSV4_MODEL_LAYER_COUNT - 1, layer_count = 1), so derived ownership
 * is participates_final_head=1 / owns_embedding=0: the pack carries the
 * final-head globals (whose 1x1 HC_HEAD_SCALE feeds hc_head_scale_value),
 * the shared embedding every MTP>0 rank requires, and the nine draft-stage
 * extras (whose 1x1 MTP_HC_HEAD_SCALE feeds mtp_hc_head_scale_value).
 *
 * The pack is synthesized IN this translation unit through the module's own
 * helpers: entry sets come verbatim from SparkDsv4ModuleExpectedLayerBits,
 * shapes from SparkDsv4ModuleResolvedShape, byte counts from
 * SparkDsv4StagePackPayloadBytes/ScaleBytes, and the directory size from
 * SparkDsv4StagePackExpectedTensorCount - so synthesis cannot drift from
 * what the loader demands, and any coverage-rule change fails here first.
 *
 * Device-region loads are hermetic on the host: regions at most
 * LOAD_REGION_INLINE_BYTES come off the file (that is where the planted
 * gate scales live, so file->device->seed flows real bytes), anything
 * larger - the multi-GB expert/embedding tensors - is satisfied by a lazy
 * allocation nobody dereferences. The full chunked fread+h2d mechanics are
 * exercised by the GPU validator; they are not what this guard exists for.
 *
 * Compiled twice by the Python driver: FLASH defines and
 * -DSPARK_DSV4_PRO_BUILD=1 (Pro aliases; SPARK_DSV4_PRO_MTP_LAYER_COUNT is
 * 3u), so both model variants prove the MTP seeding.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "spark_dsv4_resident_decode_stage_module.c"

#ifndef TEST_VARIANT_NAME
#define TEST_VARIANT_NAME "flash"
#endif

/* Regions up to this many bytes are read from the pack file; larger ones
 * are lazy allocations (the expert/embedding giants). */
#define LOAD_REGION_INLINE_BYTES 4096u
#define PLANTED_GLOBAL_HEAD_SCALE 1.375f
#define PLANTED_MTP_HEAD_SCALE 2.5f

static uint32_t failure_count;

static void Require(int condition,const char *label)
{
	if ( condition == 0 )
	{
		fprintf(stderr,"FAIL %s\n",label);
		failure_count++;
		return;
	}
	printf("PASS %s\n",label);
	fflush(stdout);
}

/* --- host stand-ins for the shared stage-module IO surface -------------- */

SparkStatus SparkStageModulePackRead(
	const char *module_tag,FILE *file,uint64_t offset,void *destination,
	uint64_t bytes)
{
	if ( destination == 0 || bytes == 0u || offset > (uint64_t)INT64_MAX ||
		fseeko(file,(off_t)offset,SEEK_SET) != 0 )
		return(SPARK_STATUS_IO_ERROR);
	return(fread(destination,1u,(size_t)bytes,file) == (size_t)bytes ?
		SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR);
	(void)module_tag;
}

SparkStatus SparkStageModuleLoadDeviceRegion(
	SparkStageModuleLedger *ledger,FILE *file,uint64_t offset,uint64_t bytes,
	void **pointer)
{
	void *allocation;
	if ( pointer == 0 || bytes == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*pointer = 0;
	allocation = calloc(1u,(size_t)bytes);
	if ( allocation == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( bytes <= LOAD_REGION_INLINE_BYTES &&
		SparkStageModulePackRead("mtp_pack_test",file,offset,
			allocation,bytes) != SPARK_STATUS_OK )
	{
		free(allocation);
		return(SPARK_STATUS_IO_ERROR);
	}
	*pointer = allocation;
	return(SPARK_STATUS_OK);
	(void)ledger;
}

SparkStatus SparkStageModuleCudaStatus(
	const char *module_tag,cudaError_t error,const char *site)
{
	if ( error == cudaSuccess )
		return(SPARK_STATUS_OK);
	fprintf(stderr,"%s %s cuda_error=%d\n",module_tag,site,(int)error);
	return(SPARK_STATUS_IO_ERROR);
}

/* --- pack synthesis ----------------------------------------------------- */

typedef struct PackBuilder
{
	FILE *file;
	SparkDsv4StagePackHeader header;
	SparkDsv4StagePackEntry *directory;
	const SparkDsv4ModuleState *state;
	uint32_t entry_count;
	uint64_t cursor;
} PackBuilder;

static void BuilderAdd(PackBuilder *builder,uint32_t tensor_kind,
	uint32_t layer_index,uint32_t is_global)
{
	SparkDsv4StagePackEntry *entry;
	SparkDsv4StagePackTensorShape shape;
	(void)is_global;
	entry = &builder->directory[builder->entry_count];
	memset(entry,0,sizeof(*entry));
	entry->tensor_kind = tensor_kind;
	entry->layer_index = layer_index;
	if ( SparkDsv4ModuleResolvedShape(builder->state,entry,&shape) != 0 )
	{
		fprintf(stderr,"FAIL resolved_shape kind=%u layer=%u\n",
			tensor_kind,layer_index);
		failure_count++;
		return;
	}
	entry->weight_format = shape.weight_format;
	entry->rows = shape.rows;
	entry->columns = shape.columns;
	entry->payload_offset = builder->cursor;
	builder->cursor += SparkDsv4StagePackPayloadBytes(
		entry->weight_format,entry->rows,entry->columns);
	if ( SparkDsv4StagePackScaleBytes(entry->weight_format,
			entry->rows,entry->columns) != 0u )
	{
		entry->scale_offset = builder->cursor;
		builder->cursor += SparkDsv4StagePackScaleBytes(
			entry->weight_format,entry->rows,entry->columns);
	}
	builder->entry_count++;
}

/* Writes every byte that fits the inline window; the giants stay holes. */
static void BuilderMaterializeInline(PackBuilder *builder)
{
	uint32_t index;
	for (index = 0u; index < builder->entry_count; index++)
	{
		SparkDsv4StagePackEntry *entry = &builder->directory[index];
		uint64_t payload_bytes = SparkDsv4StagePackPayloadBytes(
			entry->weight_format,entry->rows,entry->columns);
		uint64_t scale_bytes = SparkDsv4StagePackScaleBytes(
			entry->weight_format,entry->rows,entry->columns);
		uint64_t offset;
		if ( payload_bytes > LOAD_REGION_INLINE_BYTES )
			continue;
		offset = entry->payload_offset;
		fseeko(builder->file,(off_t)offset,SEEK_SET);
		for (; payload_bytes != 0u; payload_bytes--,offset++)
			fputc(0x5a,builder->file);
		if ( scale_bytes != 0u && scale_bytes <= LOAD_REGION_INLINE_BYTES )
		{
			fseeko(builder->file,(off_t)entry->scale_offset,SEEK_SET);
			for (; scale_bytes != 0u; scale_bytes--)
				fputc(0x7f,builder->file);
		}
	}
}

/* Plants one little-endian float at an entry's payload offset. */
static void BuilderPlantFloat(PackBuilder *builder,uint32_t tensor_kind,
	float value)
{
	uint32_t index;
	for (index = 0u; index < builder->entry_count; index++)
	{
		SparkDsv4StagePackEntry *entry = &builder->directory[index];
		if ( entry->tensor_kind == tensor_kind &&
			entry->layer_index == SPARK_DSV4_STAGEPACK_GLOBAL_LAYER )
		{
			fseeko(builder->file,(off_t)entry->payload_offset,SEEK_SET);
			fwrite(&value,sizeof(value),1u,builder->file);
			return;
		}
	}
	fprintf(stderr,"FAIL plant_target kind=%u\n",tensor_kind);
	failure_count++;
}

static int BuildPack(const char *path,float global_head_scale,
	float mtp_head_scale)
{
	PackBuilder builder;
	/* ~6 MB each: static storage so two copies never overflow the
	 * 8 MB main-thread stack. */
	static SparkDsv4ModuleState template_state;
	uint32_t kind,stage;
	uint64_t expected_count;
	memset(&template_state,0,sizeof(template_state));
	template_state.first_layer_index = SPARK_DSV4_MODEL_LAYER_COUNT - 1u;
	template_state.layer_count = 1u;
	template_state.tp_degree = 1u;
	template_state.owns_embedding = 0u;
	template_state.participates_final_head = 1u;
	memset(&builder,0,sizeof(builder));
	builder.state = &template_state;
	builder.file = fopen(path,"wb");
	if ( builder.file == 0 )
		return(0);
	SparkDsv4StagePackExpectedGeometry(&builder.header,
		template_state.first_layer_index,template_state.layer_count);
	expected_count = SparkDsv4StagePackExpectedTensorCount(
		template_state.first_layer_index,template_state.layer_count);
	builder.header.tensor_count = (uint32_t)expected_count;
	/* Payloads start after the directory; the loader sweeps for
	 * records below this floor since the bounds hardening. */
	builder.cursor = SPARK_DSV4_STAGEPACK_HEADER_BYTES +
		(uint64_t)builder.header.tensor_count *
		SPARK_DSV4_STAGEPACK_ENTRY_BYTES;
	builder.directory = calloc(expected_count,sizeof(*builder.directory));
	/* Backbone tail layer: its exact kind set comes from the module's own
	 * coverage rule (HCA adds the compressor quartet, hash routing picks
	 * the bias side, and so on). */
	for (kind = 0u; kind < SPARK_DSV4_STAGEPACK_TENSOR_KIND_COUNT; kind++)
	{
		if ( (SparkDsv4ModuleExpectedLayerBits(
			template_state.first_layer_index) & (1ull << kind)) == 0u )
			continue;
		BuilderAdd(&builder,kind,
			template_state.first_layer_index,0u);
	}
	/* The DSpark draft layers ride standard kinds under the MTP range. */
	for (stage = 0u; stage < SPARK_DSV4_MODEL_MTP_LAYER_COUNT; stage++)
	{
		uint32_t mtp_layer = SPARK_DSV4_STAGEPACK_MTP_LAYER(stage);
		uint64_t mtp_bits = SparkDsv4ModuleExpectedLayerBits(mtp_layer);
		for (kind = 0u; kind < SPARK_DSV4_STAGEPACK_TENSOR_KIND_COUNT; kind++)
		{
			if ( (mtp_bits & (1ull << kind)) == 0u )
				continue;
			BuilderAdd(&builder,kind,mtp_layer,0u);
		}
	}
	/* Globals: the shared embedding (expected whenever the draft rides
	 * along), the five final-head tensors, and the nine draft-stage extras
	 * whose 1x1 MTP_HC_HEAD_SCALE seeds the mirror under test. */
	BuilderAdd(&builder,SPARK_DSV4_STAGEPACK_TENSOR_EMBEDDING,
		SPARK_DSV4_STAGEPACK_GLOBAL_LAYER,1u);
	for (kind = SPARK_DSV4_STAGEPACK_TENSOR_FINAL_NORM;
		kind <= SPARK_DSV4_STAGEPACK_TENSOR_HC_HEAD_SCALE; kind++)
		BuilderAdd(&builder,kind,SPARK_DSV4_STAGEPACK_GLOBAL_LAYER,1u);
	for (kind = SPARK_DSV4_STAGEPACK_TENSOR_MTP_MAIN_PROJ;
		kind <= SPARK_DSV4_STAGEPACK_TENSOR_MTP_CONFIDENCE_PROJ; kind++)
		BuilderAdd(&builder,kind,SPARK_DSV4_STAGEPACK_GLOBAL_LAYER,1u);
	if ( (uint64_t)builder.entry_count != expected_count )
	{
		fprintf(stderr,"FAIL entry_count synthesized=%u expected=%llu\n",
			builder.entry_count,(unsigned long long)expected_count);
		failure_count++;
	}
	/* Generic fill first; the planted gate scales go on top so the
	 * values survive into the file the loader reads. */
	BuilderMaterializeInline(&builder);
	BuilderPlantFloat(&builder,SPARK_DSV4_STAGEPACK_TENSOR_HC_HEAD_SCALE,
		global_head_scale);
	BuilderPlantFloat(&builder,SPARK_DSV4_STAGEPACK_TENSOR_MTP_HC_HEAD_SCALE,
		mtp_head_scale);
	builder.header.file_bytes = builder.cursor;
	fseeko(builder.file,0L,SEEK_SET);
	fwrite(&builder.header,sizeof(builder.header),1u,builder.file);
	fwrite(builder.directory,sizeof(*builder.directory),
		builder.entry_count,builder.file);
	fclose(builder.file);
	free(builder.directory);
	return(failure_count == 0u);
}

static void SeedTailStageState(SparkDsv4ModuleState *state)
{
	memset(state,0,sizeof(*state));
	state->first_layer_index = SPARK_DSV4_MODEL_LAYER_COUNT - 1u;
	state->layer_count = 1u;
	state->tp_degree = 1u;
	state->owns_embedding = 0u;
	state->participates_final_head = 1u;
}

int main(void)
{
	/* See note above: 6 MB struct, kept off the stack. */
	static SparkDsv4ModuleState state;
	char path[256];
	SparkStatus status;
	float zero = 0.0f;
	snprintf(path,sizeof(path),"build/tmp/mtp_pack_%s_%ld.dsv4sp",
		TEST_VARIANT_NAME,(long)getpid());
	if ( SPARK_DSV4_MODEL_MTP_LAYER_COUNT == 0u )
	{
		fprintf(stderr,"FAIL variant_has_no_mtp_layers\n");
		return(1);
	}
	/* Positive: an MTP>0 pack seeds BOTH gate scalars through the full
	 * load path. */
	if ( BuildPack(path,PLANTED_GLOBAL_HEAD_SCALE,
		PLANTED_MTP_HEAD_SCALE) == 0 )
		return(1);
	SeedTailStageState(&state);
	status = SparkDsv4ModuleLoadPack(&state,path);
	Require(status == SPARK_STATUS_OK,"load_mtp_pack_ok");
	Require(state.mtp.hc_head_scale_f32 != 0,
		"mtp_bind_keeps_device_pointer");
	Require(state.mtp_hc_head_scale_value != 0.0f,
		"mtp_head_scale_nonzero_after_load");
	Require(state.mtp_hc_head_scale_value == PLANTED_MTP_HEAD_SCALE,
		"mtp_head_scale_matches_pack_tensor");
	Require(isfinite(state.mtp_hc_head_scale_value) != 0,
		"mtp_head_scale_finite_after_load");
	Require(state.hc_head_scale_value == PLANTED_GLOBAL_HEAD_SCALE,
		"global_head_scale_seeded_by_same_load");
	Require(state.mtp.main_proj.payload != 0 &&
		state.mtp.confidence_proj.payload != 0,
		"mtp_extras_bound");
	/* Negative: a pack whose draft gate scale is zero must be refused -
	 * the pre-fix world where seeding never ran. */
	BuildPack(path,PLANTED_GLOBAL_HEAD_SCALE,zero);
	SeedTailStageState(&state);
	status = SparkDsv4ModuleLoadPack(&state,path);
	Require(status == SPARK_STATUS_VALIDATION_FAILED,
		"load_refuses_zero_mtp_head_scale");
	if ( failure_count != 0u )
	{
		fprintf(stderr,"FAIL dsv4 mtp head-scale pack-load unit (%s)\n",
			TEST_VARIANT_NAME);
		return(1);
	}
	printf("PASS dsv4 mtp_hc_head_scale_value seeded on MTP>0 pack load "
		"(%s)\n",TEST_VARIANT_NAME);
	return(0);
}
