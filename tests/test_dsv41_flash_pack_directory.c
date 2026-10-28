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

#define CONTRACT_HEX "00112233445566778899aabbccddeeff102132435465768798a9bacbdcedfe0f"
#define ENGRAM_RANK 3u
#define ENGRAM_EMBED_ROWS 24000386u

static void BuildIdentity(SparkDsv41FlashStagePackHeader *header)
{
	static const uint8_t contract[SPARK_DSV41_FLASH_STAGEPACK_SHA256_BYTES] = { 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb,
		0xcc, 0xdd, 0xee, 0xff, 0x10, 0x21, 0x32, 0x43, 0x54, 0x65, 0x76, 0x87, 0x98, 0xa9, 0xba, 0xcb, 0xdc, 0xed, 0xfe, 0x0f };
	memcpy(header->contract_sha256, contract, sizeof(contract));
	memset(header->model_revision, 0, sizeof(header->model_revision));
	strcpy(header->model_revision, "1bae0e1088993b9b82de08f59de5be5a34e42170");
	memset(header->source_config_sha256, 0x8b, sizeof(header->source_config_sha256));
	memset(header->pack_recipe_sha256, 0x5c, sizeof(header->pack_recipe_sha256));
	header->tp_rank = ENGRAM_RANK;
}

static uint64_t PlaceEngram(SparkDsv41FlashStagePackEntry *entry, uint32_t which, uint32_t slot, uint64_t cursor)
{
	memset(entry, 0, sizeof(*entry));
	entry->tensor_kind = 2u * slot + which;
	entry->layer_index = which == 0u ? SPARK_DSV41_FLASH_MODEL_ENGRAM_LAYER_0 : SPARK_DSV41_FLASH_MODEL_ENGRAM_LAYER_1;
	entry->group_count = 1u;
	if ( slot < 2u )
	{
		entry->payload_type = SPARK_DSV41_FLASH_STAGEPACK_PAYLOAD_PACKED_WEIGHT;
		entry->weight_codec = SPARK_WEIGHT_CODEC_FP8_E4M3;
		entry->scale_encoding = SPARK_WEIGHT_SCALE_ENCODING_E8M0;
		entry->rows = slot == 0u ? ENGRAM_EMBED_ROWS : SPARK_DSV41_FLASH_ENGRAM_KV_ROWS;
		entry->columns = slot == 0u ? SPARK_DSV41_FLASH_MODEL_ENGRAM_HEAD_DIMENSION : SPARK_DSV41_FLASH_ENGRAM_EMBED_COLUMNS;
		entry->payload_bytes = (uint64_t)entry->rows * entry->columns;
		entry->scale_bytes = slot == 0u ? (uint64_t)entry->rows * (entry->columns / SPARK_DSV41_FLASH_STAGEPACK_FP8_BLOCK) :
			(uint64_t)(entry->rows / SPARK_DSV41_FLASH_STAGEPACK_FP8_BLOCK) * (entry->columns / SPARK_DSV41_FLASH_STAGEPACK_FP8_BLOCK);
	}
	else
	{
		entry->payload_type = SPARK_DSV41_FLASH_STAGEPACK_PAYLOAD_BF16;
		entry->weight_codec = SPARK_WEIGHT_CODEC_BF16;
		entry->scale_encoding = SPARK_WEIGHT_SCALE_ENCODING_NONE;
		entry->rows = SPARK_DSV41_FLASH_MODEL_HC_MULT;
		entry->columns = SPARK_DSV41_FLASH_MODEL_HIDDEN_DIMENSION;
		entry->payload_bytes = (uint64_t)entry->rows * entry->columns * sizeof(uint16_t);
	}
	entry->payload_offset = cursor;
	cursor = Align(cursor + entry->payload_bytes);
	if ( entry->scale_bytes != 0u )
	{
		entry->scale_offset = cursor;
		cursor = Align(cursor + entry->scale_bytes);
	}
	return cursor;
}

static void BuildEngram(SparkDsv41FlashStagePackHeader *engram, SparkDsv41FlashStagePackEntry *entries, const SparkDsv41FlashStagePackHeader *pack)
{
	uint64_t cursor;
	memset(engram, 0, sizeof(*engram));
	engram->magic = SPARK_DSV41_FLASH_ENGRAM_MAGIC;
	engram->format_version = SPARK_DSV41_FLASH_ENGRAM_FORMAT_VERSION;
	engram->header_bytes = sizeof(*engram);
	engram->directory_entry_bytes = sizeof(*entries);
	engram->tensor_count = SPARK_DSV41_FLASH_ENGRAM_TENSOR_COUNT;
	engram->tp_degree = pack->tp_degree;
	engram->tp_rank = ENGRAM_RANK;
	engram->directory_offset = Align(sizeof(*engram));
	memcpy(engram->model_revision, pack->model_revision, sizeof(engram->model_revision));
	memcpy(engram->source_config_sha256, pack->source_config_sha256, sizeof(engram->source_config_sha256));
	cursor = Align(engram->directory_offset + SPARK_DSV41_FLASH_ENGRAM_TENSOR_COUNT * sizeof(*entries));
	for ( uint32_t which = 0u; which < 2u; which++ )
		for ( uint32_t slot = 0u; slot < 4u; slot++ )
			cursor = PlaceEngram(&entries[which * 4u + slot], which, slot, cursor);
	engram->file_bytes = cursor;
}

static int32_t RunEngram(const SparkDsv41FlashStagePackEntry *entries, uint64_t file_bytes, uint32_t *at)
{
	*at = UINT32_MAX;
	return SparkDsv41FlashEngramDirectoryCheck(entries, file_bytes, at);
}

static void TestIdentityAndEngram(const SparkDsv41FlashStagePackHeader *built)
{
	SparkDsv41FlashStagePackHeader pack = *built, edited, engram, shard;
	SparkDsv41FlashStagePackEntry clean[SPARK_DSV41_FLASH_ENGRAM_TENSOR_COUNT], entries[SPARK_DSV41_FLASH_ENGRAM_TENSOR_COUNT];
	uint32_t at;
	BuildIdentity(&pack);
	Check(SparkDsv41FlashStagePackIdentityCheck(&pack, CONTRACT_HEX) == 0, "a pack carrying the compiled contract, a revision and both source digests passes");
	edited = pack;
	edited.contract_sha256[31] ^= 1u;
	Check(SparkDsv41FlashStagePackIdentityCheck(&edited, CONTRACT_HEX) == SPARK_DSV41_FLASH_STAGEPACK_CHECK_CONTRACT, "a contract digest one bit off is refused");
	Check(SparkDsv41FlashStagePackIdentityCheck(&pack, "00112233") == SPARK_DSV41_FLASH_STAGEPACK_CHECK_CONTRACT, "a short compiled contract is refused");
	Check(SparkDsv41FlashStagePackIdentityCheck(&pack, "00112233445566778899AABBCCDDEEFF102132435465768798A9BACBDCEDFE0F") == SPARK_DSV41_FLASH_STAGEPACK_CHECK_CONTRACT,
		"a compiled contract that is not lowercase hex is refused");
	edited = pack;
	edited.model_revision[0] = '\0';
	Check(SparkDsv41FlashStagePackIdentityCheck(&edited, CONTRACT_HEX) == SPARK_DSV41_FLASH_STAGEPACK_CHECK_IDENTITY, "a pack with no model revision is refused");
	edited = pack;
	memset(edited.model_revision, 'a', sizeof(edited.model_revision));
	Check(SparkDsv41FlashStagePackIdentityCheck(&edited, CONTRACT_HEX) == SPARK_DSV41_FLASH_STAGEPACK_CHECK_IDENTITY, "an unterminated model revision is refused");
	edited = pack;
	memset(edited.source_config_sha256, 0, sizeof(edited.source_config_sha256));
	Check(SparkDsv41FlashStagePackIdentityCheck(&edited, CONTRACT_HEX) == SPARK_DSV41_FLASH_STAGEPACK_CHECK_IDENTITY, "a zero source config digest is refused");
	edited = pack;
	memset(edited.pack_recipe_sha256, 0, sizeof(edited.pack_recipe_sha256));
	Check(SparkDsv41FlashStagePackIdentityCheck(&edited, CONTRACT_HEX) == SPARK_DSV41_FLASH_STAGEPACK_CHECK_IDENTITY, "a zero pack recipe digest is refused");

	BuildEngram(&engram, clean, &pack);
	Check(SparkDsv41FlashEngramHeaderCheck(&engram, &pack, ENGRAM_RANK, engram.file_bytes) == 0, "this rank's engram shard of the pack's checkpoint passes");
	Check(RunEngram(clean, engram.file_bytes, &at) == 0, "an engram directory laid out as the shard cutter writes it passes");
	shard = engram;
	shard.magic ^= 1u;
	Check(SparkDsv41FlashEngramHeaderCheck(&shard, &pack, ENGRAM_RANK, engram.file_bytes) == SPARK_DSV41_FLASH_STAGEPACK_CHECK_ENGRAM_HEADER, "a wrong magic is refused");
	shard = engram;
	shard.format_version = 2u;
	Check(SparkDsv41FlashEngramHeaderCheck(&shard, &pack, ENGRAM_RANK, engram.file_bytes) == SPARK_DSV41_FLASH_STAGEPACK_CHECK_ENGRAM_HEADER, "another shard format is refused");
	shard = engram;
	shard.tensor_count = 7u;
	Check(SparkDsv41FlashEngramHeaderCheck(&shard, &pack, ENGRAM_RANK, engram.file_bytes) == SPARK_DSV41_FLASH_STAGEPACK_CHECK_ENGRAM_HEADER, "a shard of seven tensors is refused");
	Check(SparkDsv41FlashEngramHeaderCheck(&engram, &pack, ENGRAM_RANK, engram.file_bytes - 1u) == SPARK_DSV41_FLASH_STAGEPACK_CHECK_ENGRAM_HEADER,
		"a shard file one byte short of its header's size is refused");
	shard = engram;
	shard.directory_offset = engram.file_bytes;
	Check(SparkDsv41FlashEngramHeaderCheck(&shard, &pack, ENGRAM_RANK, engram.file_bytes) == SPARK_DSV41_FLASH_STAGEPACK_CHECK_ENGRAM_HEADER, "a directory past the file is refused");
	Check(SparkDsv41FlashEngramHeaderCheck(&engram, &pack, ENGRAM_RANK + 1u, engram.file_bytes) == SPARK_DSV41_FLASH_STAGEPACK_CHECK_ENGRAM_RANK,
		"another rank's shard (equal row counts on ranks 0-14) is refused");
	shard = engram;
	shard.tp_degree = 8u;
	Check(SparkDsv41FlashEngramHeaderCheck(&shard, &pack, ENGRAM_RANK, engram.file_bytes) == SPARK_DSV41_FLASH_STAGEPACK_CHECK_ENGRAM_RANK, "a shard cut for another degree is refused");
	shard = engram;
	shard.model_revision[39] = shard.model_revision[39] == '0' ? '1' : '0';
	Check(SparkDsv41FlashEngramHeaderCheck(&shard, &pack, ENGRAM_RANK, engram.file_bytes) == SPARK_DSV41_FLASH_STAGEPACK_CHECK_ENGRAM_CHECKPOINT,
		"a shard of another model revision is refused");
	shard = engram;
	shard.source_config_sha256[0] ^= 1u;
	Check(SparkDsv41FlashEngramHeaderCheck(&shard, &pack, ENGRAM_RANK, engram.file_bytes) == SPARK_DSV41_FLASH_STAGEPACK_CHECK_ENGRAM_CHECKPOINT,
		"a shard of another source config is refused");

	memcpy(entries, clean, sizeof(clean));
	entries[0] = clean[4];
	entries[4] = clean[0];
	Check(RunEngram(entries, engram.file_bytes, &at) == SPARK_DSV41_FLASH_STAGEPACK_CHECK_ENGRAM_ENTRY && at == 0u, "the two layers' embeddings swapped are refused");
	memcpy(entries, clean, sizeof(clean));
	entries[3].rows = SPARK_DSV41_FLASH_MODEL_HC_MULT + 1u;
	entries[3].payload_bytes = (uint64_t)entries[3].rows * entries[3].columns * sizeof(uint16_t);
	Check(RunEngram(entries, engram.file_bytes, &at) == SPARK_DSV41_FLASH_STAGEPACK_CHECK_ENGRAM_ENTRY && at == 3u, "a k weight with another row count is refused, as q is");
	memcpy(entries, clean, sizeof(clean));
	entries[0].scale_bytes -= 1u;
	Check(RunEngram(entries, engram.file_bytes, &at) == SPARK_DSV41_FLASH_STAGEPACK_CHECK_ENGRAM_ENTRY && at == 0u, "embedding scales one byte short are refused");
	memcpy(entries, clean, sizeof(clean));
	entries[1].weight_codec = SPARK_WEIGHT_CODEC_BF16;
	Check(RunEngram(entries, engram.file_bytes, &at) == SPARK_DSV41_FLASH_STAGEPACK_CHECK_ENGRAM_ENTRY && at == 1u, "a key/value weight in another codec is refused");
	memcpy(entries, clean, sizeof(clean));
	entries[5].layer_index = SPARK_DSV41_FLASH_MODEL_ENGRAM_LAYER_0;
	Check(RunEngram(entries, engram.file_bytes, &at) == SPARK_DSV41_FLASH_STAGEPACK_CHECK_ENGRAM_ENTRY && at == 5u, "a second-layer tensor marked for the first layer is refused");
	memcpy(entries, clean, sizeof(clean));
	Check(RunEngram(entries, entries[7].payload_offset + 1u, &at) == SPARK_DSV41_FLASH_STAGEPACK_CHECK_ENGRAM_ENTRY && at == 7u, "a tensor past the end of the shard is refused");
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

	TestIdentityAndEngram(&header);
	if ( failures != 0u )
		return 1;
	printf("PASS dsv41 flash pack directory: a contract-built directory of %u tensors passes; wrong rows, short payloads, half scales, misalignment, "
		"tensors past the file, duplicates, overlaps, a global with a layer, a missing tensor and a misplaced indexer tensor are each refused; "
		"a pack off the compiled contract or without a revision or source digests is refused; an engram shard of another rank, degree, "
		"format, size or checkpoint is refused, and so is every malformed engram entry\n", count);
	return 0;
}
