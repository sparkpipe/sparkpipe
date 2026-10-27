#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "spark_dsv41_flash_stagepack_format.h"

#define ENTRIES_MAX 2048u

static uint32_t failures;

static void Check(int32_t condition, const char *message)
{
	if ( !condition )
	{
		fprintf(stderr, "FAIL test_dsv41_flash_pack_directory: %s\n", message);
		failures++;
	}
}

static uint64_t Align(uint64_t value)
{
	return (value + SPARK_DSV41_FLASH_STAGEPACK_ALIGNMENT_BYTES - 1u) / SPARK_DSV41_FLASH_STAGEPACK_ALIGNMENT_BYTES * SPARK_DSV41_FLASH_STAGEPACK_ALIGNMENT_BYTES;
}

static uint64_t Place(SparkDsv41FlashStagePackEntry *entry, uint32_t kind, uint32_t layer, uint64_t cursor)
{
	SparkDsv41FlashStagePackTensorShape shape;
	memset(entry, 0, sizeof(*entry));
	SparkDsv41FlashStagePackExpectedShape(kind, SPARK_WEIGHT_CODEC_MXFP4_E2M1, 16u, &shape);
	entry->tensor_kind = kind;
	entry->layer_index = layer;
	entry->payload_type = shape.payload_type;
	entry->weight_codec = shape.weight_codec;
	entry->scale_encoding = shape.scale_encoding;
	entry->group_count = shape.group_count;
	entry->rows = shape.rows;
	entry->columns = shape.columns;
	entry->payload_offset = cursor;
	entry->payload_bytes = SparkDsv41FlashStagePackExpectedPayloadBytes(&shape);
	cursor = Align(cursor + entry->payload_bytes);
	entry->scale_bytes = SparkDsv41FlashStagePackExpectedScaleBytes(&shape);
	if ( entry->scale_bytes != 0u )
	{
		entry->scale_offset = cursor;
		cursor = Align(cursor + entry->scale_bytes);
	}
	return cursor;
}

static uint32_t Build(SparkDsv41FlashStagePackHeader *header, SparkDsv41FlashStagePackEntry *entries)
{
	static const uint32_t globals[] = { SPARK_DSV41_FLASH_STAGEPACK_TENSOR_EMBEDDING, SPARK_DSV41_FLASH_STAGEPACK_TENSOR_FINAL_NORM,
		SPARK_DSV41_FLASH_STAGEPACK_TENSOR_LM_HEAD };
	uint32_t count = 0u, layer, kind;
	uint64_t cursor;
	for ( layer = 0u; layer < SPARK_DSV41_FLASH_MODEL_LAYER_COUNT; layer++ )
		for ( kind = 0u; kind < SPARK_DSV41_FLASH_STAGEPACK_TENSOR_KIND_COUNT; kind++ )
			if ( SparkDsv41FlashStagePackKindInLayer(kind, layer) != 0u )
				count++;
	count += 3u;
	memset(header, 0, sizeof(*header));
	header->tp_degree = 16u;
	header->expert_weight_codec = SPARK_WEIGHT_CODEC_MXFP4_E2M1;
	header->tensor_count = count;
	header->directory_offset = 512u;
	cursor = Align(header->directory_offset + (uint64_t)count * sizeof(SparkDsv41FlashStagePackEntry));
	count = 0u;
	for ( kind = 0u; kind < 3u; kind++ )
		cursor = Place(&entries[count++], globals[kind], SPARK_DSV41_FLASH_STAGEPACK_GLOBAL_LAYER, cursor);
	for ( layer = 0u; layer < SPARK_DSV41_FLASH_MODEL_LAYER_COUNT; layer++ )
		for ( kind = 0u; kind < SPARK_DSV41_FLASH_STAGEPACK_TENSOR_KIND_COUNT; kind++ )
			if ( SparkDsv41FlashStagePackKindInLayer(kind, layer) != 0u )
				cursor = Place(&entries[count++], kind, layer, cursor);
	header->file_bytes = cursor;
	return count;
}

static uint32_t Find(const SparkDsv41FlashStagePackEntry *entries, uint32_t count, uint32_t kind, uint32_t layer)
{
	for ( uint32_t index = 0u; index < count; index++ )
		if ( entries[index].tensor_kind == kind && entries[index].layer_index == layer )
			return index;
	return UINT32_MAX;
}

static int32_t Run(const SparkDsv41FlashStagePackHeader *header, const SparkDsv41FlashStagePackEntry *entries)
{
	uint32_t at = 0u;
	return SparkDsv41FlashStagePackDirectoryCheck(header, entries, &at);
}

int main(void)
{
	static SparkDsv41FlashStagePackEntry clean[ENTRIES_MAX], entries[ENTRIES_MAX];
	SparkDsv41FlashStagePackHeader header, edited;
	uint32_t count, swa = UINT32_MAX, layer, q_b, ffn_norm, attn_norm, w13, sink;
	count = Build(&header, clean);
	Check(count == 998u, "the TP16 pack contract holds 998 tensors, as the 1017 packs on the fleet do");
	Check(Run(&header, clean) == 0, "a directory built from the pack contract passes");
	for ( layer = 0u; layer < SPARK_DSV41_FLASH_MODEL_LAYER_COUNT && swa == UINT32_MAX; layer++ )
		if ( SPARK_DSV41_FLASH_MODEL_LAYER_IS_SWA_ONLY(layer) )
			swa = layer;
	q_b = Find(clean, count, SPARK_DSV41_FLASH_STAGEPACK_TENSOR_Q_B, 3u);
	ffn_norm = Find(clean, count, SPARK_DSV41_FLASH_STAGEPACK_TENSOR_FFN_NORM, 3u);
	attn_norm = Find(clean, count, SPARK_DSV41_FLASH_STAGEPACK_TENSOR_ATTN_NORM, 3u);
	w13 = Find(clean, count, SPARK_DSV41_FLASH_STAGEPACK_TENSOR_EXPERT_W13_SLICE, 3u);
	sink = Find(clean, count, SPARK_DSV41_FLASH_STAGEPACK_TENSOR_ATTN_SINK, 3u);
	Check(swa != UINT32_MAX && q_b != UINT32_MAX && ffn_norm != UINT32_MAX && attn_norm != UINT32_MAX && w13 != UINT32_MAX && sink != UINT32_MAX,
		"the contract has an SWA-only layer and layer 3 holds q_b, both norms, the expert w13 slice and the sinks");
	if ( failures != 0u )
		return 1;

	memcpy(entries, clean, sizeof(clean));
	entries[q_b].rows += SPARK_DSV41_FLASH_STAGEPACK_FP8_BLOCK;
	entries[q_b].payload_bytes += (uint64_t)SPARK_DSV41_FLASH_STAGEPACK_FP8_BLOCK * entries[q_b].columns;
	Check(Run(&header, entries) == SPARK_DSV41_FLASH_STAGEPACK_CHECK_SHAPE, "q_b with another rank count's rows is refused for its shape");

	memcpy(entries, clean, sizeof(clean));
	entries[w13].payload_bytes -= 1u;
	Check(Run(&header, entries) == SPARK_DSV41_FLASH_STAGEPACK_CHECK_BYTES, "an expert slice one byte short is refused");

	memcpy(entries, clean, sizeof(clean));
	entries[q_b].scale_bytes /= 2u;
	Check(Run(&header, entries) == SPARK_DSV41_FLASH_STAGEPACK_CHECK_BYTES, "fp8 scales for half the blocks are refused");

	memcpy(entries, clean, sizeof(clean));
	entries[sink].payload_offset += 4u;
	Check(Run(&header, entries) == SPARK_DSV41_FLASH_STAGEPACK_CHECK_BYTES, "a misaligned tensor is refused");

	memcpy(entries, clean, sizeof(clean));
	edited = header;
	edited.file_bytes = entries[count - 1u].payload_offset + 1u;
	Check(Run(&edited, entries) == SPARK_DSV41_FLASH_STAGEPACK_CHECK_BYTES, "a tensor past the end of the file is refused");

	memcpy(entries, clean, sizeof(clean));
	entries[ffn_norm].tensor_kind = SPARK_DSV41_FLASH_STAGEPACK_TENSOR_ATTN_NORM;
	Check(Run(&header, entries) == SPARK_DSV41_FLASH_STAGEPACK_CHECK_LAYER, "a layer holding one kind twice is refused");

	memcpy(entries, clean, sizeof(clean));
	entries[ffn_norm].payload_offset = entries[attn_norm].payload_offset;
	Check(Run(&header, entries) == SPARK_DSV41_FLASH_STAGEPACK_CHECK_OVERLAP, "two tensors sharing bytes are refused");

	memcpy(entries, clean, sizeof(clean));
	entries[0].layer_index = 0u;
	Check(Run(&header, entries) == SPARK_DSV41_FLASH_STAGEPACK_CHECK_GLOBAL, "a global tensor with a layer is refused");

	memcpy(entries, clean, sizeof(clean));
	edited = header;
	edited.tensor_count = count - 1u;
	Check(Run(&edited, entries) == SPARK_DSV41_FLASH_STAGEPACK_CHECK_LAYER_SET, "a pack missing the last layer's last tensor is refused");

	memcpy(entries, clean, sizeof(clean));
	{
		SparkDsv41FlashStagePackTensorShape shape;
		SparkDsv41FlashStagePackExpectedShape(SPARK_DSV41_FLASH_STAGEPACK_TENSOR_INDEXER_K_NORM, SPARK_WEIGHT_CODEC_MXFP4_E2M1, 16u, &shape);
		entries[sink].layer_index = swa;
		entries[sink].tensor_kind = SPARK_DSV41_FLASH_STAGEPACK_TENSOR_INDEXER_K_NORM;
		entries[sink].payload_type = shape.payload_type;
		entries[sink].weight_codec = shape.weight_codec;
		entries[sink].scale_encoding = shape.scale_encoding;
		entries[sink].group_count = shape.group_count;
		entries[sink].rows = shape.rows;
		entries[sink].columns = shape.columns;
		Check(entries[sink].payload_bytes == SparkDsv41FlashStagePackExpectedPayloadBytes(&shape), "the sinks and the indexer key norm are the same size");
		Check(Run(&header, entries) == SPARK_DSV41_FLASH_STAGEPACK_CHECK_LAYER, "a well-formed indexer tensor in an SWA-only layer is refused");
	}

	if ( failures != 0u )
		return 1;
	printf("PASS dsv41 flash pack directory: a contract-built directory of %u tensors passes; wrong rows, short payloads, half scales, misalignment, "
		"tensors past the file, duplicates, overlaps, a global with a layer, a missing tensor and a misplaced indexer tensor are each refused\n", count);
	return 0;
}
