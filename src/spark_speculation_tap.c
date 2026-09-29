#include "sparkpipe/spark_speculation_tap.h"

#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_error_site.h"

static void SparkSpeculationTapPut32(uint8_t *bytes,uint32_t value)
{
	bytes[0] = (uint8_t)value;
	bytes[1] = (uint8_t)(value >> 8);
	bytes[2] = (uint8_t)(value >> 16);
	bytes[3] = (uint8_t)(value >> 24);
}

static void SparkSpeculationTapPut64(uint8_t *bytes,uint64_t value)
{
	SparkSpeculationTapPut32(bytes,(uint32_t)value);
	SparkSpeculationTapPut32(bytes + 4,(uint32_t)(value >> 32));
}

static uint32_t SparkSpeculationTapGet32(const uint8_t *bytes)
{
	return((uint32_t)bytes[0] | (uint32_t)bytes[1] << 8 | (uint32_t)bytes[2] << 16 | (uint32_t)bytes[3] << 24);
}

static uint64_t SparkSpeculationTapGet64(const uint8_t *bytes)
{
	return((uint64_t)SparkSpeculationTapGet32(bytes) | (uint64_t)SparkSpeculationTapGet32(bytes + 4) << 32);
}

static SparkStatus SparkSpeculationTapParseLayer(const char **cursor,uint32_t *layer_out)
{
	const char *text = *cursor;
	uint64_t value = 0u;
	uint32_t digits = 0u;
	while ( *text >= '0' && *text <= '9' && digits < 10u )
	{
		value = value * 10u + (uint64_t)(*text - '0');
		text++;
		digits++;
	}
	if ( digits == 0u || value > UINT32_MAX || (*text >= '0' && *text <= '9') )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	*layer_out = (uint32_t)value;
	*cursor = text;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkSpeculationTapSetParse(const char *text,uint32_t layer_count,uint32_t stream_count,uint32_t hidden_dimension,SparkSpeculationTapSet *set)
{
	const char *cursor;
	uint32_t layer;
	uint64_t row_bytes,record_bytes;
	SparkStatus status;
	if ( text == 0 || set == 0 || layer_count == 0u || stream_count == 0u || hidden_dimension == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(set,0,sizeof(*set));
	if ( strncmp(text,"mean:",5u) == 0 )
		set->reduction = SPARK_SPECULATION_TAP_REDUCTION_MEAN;
	else if ( strncmp(text,"all:",4u) == 0 )
		set->reduction = SPARK_SPECULATION_TAP_REDUCTION_ALL;
	else
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	cursor = strchr(text,':') + 1;
	for (;;)
	{
		status = SparkSpeculationTapParseLayer(&cursor,&layer);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		if ( set->tap_count == SPARK_SPECULATION_TAP_MAX_TAPS )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		if ( layer >= layer_count || (set->tap_count != 0u && layer <= set->layers[set->tap_count - 1u]) )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		set->layers[set->tap_count++] = layer;
		if ( *cursor == '\0' )
			break;
		if ( *cursor != ',' )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		cursor++;
	}
	row_bytes = (uint64_t)hidden_dimension * (set->reduction == SPARK_SPECULATION_TAP_REDUCTION_ALL ? stream_count : 1u) * sizeof(uint16_t);
	record_bytes = row_bytes * set->tap_count;
	if ( record_bytes > UINT32_MAX / 2u )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	set->stream_count = stream_count;
	set->hidden_dimension = hidden_dimension;
	set->layer_count = layer_count;
	set->row_elements = (uint32_t)(row_bytes / sizeof(uint16_t));
	set->row_bytes = (uint32_t)row_bytes;
	set->record_bytes = (uint32_t)record_bytes;
	return(SPARK_STATUS_OK);
}

uint32_t SparkSpeculationTapSetOrdinal(const SparkSpeculationTapSet *set,uint32_t layer)
{
	uint32_t index;
	if ( set == 0 )
		return(UINT32_MAX);
	for (index=0u; index<set->tap_count; index++)
		if ( set->layers[index] == layer )
			return(index);
	return(UINT32_MAX);
}

uint64_t SparkSpeculationTapSetFingerprint(const SparkSpeculationTapSet *set)
{
	uint8_t bytes[24u + 4u * SPARK_SPECULATION_TAP_MAX_TAPS];
	uint32_t index;
	if ( set == 0 )
		return(0u);
	memset(bytes,0,sizeof(bytes));
	SparkSpeculationTapPut32(bytes,SPARK_SPECULATION_TAP_FRAGMENT_MAGIC);
	SparkSpeculationTapPut32(bytes + 4,set->reduction);
	SparkSpeculationTapPut32(bytes + 8,set->tap_count);
	SparkSpeculationTapPut32(bytes + 12,set->stream_count);
	SparkSpeculationTapPut32(bytes + 16,set->hidden_dimension);
	SparkSpeculationTapPut32(bytes + 20,set->layer_count);
	for (index=0u; index<set->tap_count && index<SPARK_SPECULATION_TAP_MAX_TAPS; index++)
		SparkSpeculationTapPut32(bytes + 24 + 4u * index,set->layers[index]);
	return(SparkHashBytes(UINT64_C(1469598103934665603),bytes,(uint32_t)sizeof(bytes)));
}

uint32_t SparkSpeculationTapFragmentCount(uint32_t record_bytes,uint32_t payload_max)
{
	if ( payload_max == 0u || payload_max > SPARK_SPECULATION_TAP_FRAGMENT_PAYLOAD_MAX )
		return(0u);
	return(SparkCeilDivU32(record_bytes,payload_max));
}

SparkStatus SparkSpeculationTapEncodeFragment(const SparkSpeculationTapRecord *record,uint64_t fingerprint,const uint8_t *payload,uint32_t record_bytes,uint32_t offset,uint32_t bytes,uint8_t *out,uint32_t out_capacity,uint32_t *out_length)
{
	if ( record == 0 || payload == 0 || out == 0 || out_length == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*out_length = 0u;
	if ( (record->flags & ~SPARK_SPECULATION_TAP_FLAG_MASK) != 0u || record->reserved != 0u || record_bytes == 0u || bytes == 0u || bytes > SPARK_SPECULATION_TAP_FRAGMENT_PAYLOAD_MAX || offset >= record_bytes || bytes > record_bytes - offset )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( out_capacity < SPARK_SPECULATION_TAP_FRAGMENT_HEADER_BYTES + bytes )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	SparkSpeculationTapPut32(out,SPARK_SPECULATION_TAP_FRAGMENT_MAGIC);
	SparkSpeculationTapPut32(out + 4,SPARK_SPECULATION_TAP_FRAGMENT_VERSION | SPARK_SPECULATION_TAP_FRAGMENT_KIND << 16);
	SparkSpeculationTapPut64(out + 8,record->engine_generation);
	SparkSpeculationTapPut64(out + 16,record->sequence_id);
	SparkSpeculationTapPut64(out + 24,record->position);
	SparkSpeculationTapPut64(out + 32,record->serial);
	SparkSpeculationTapPut64(out + 40,fingerprint);
	SparkSpeculationTapPut32(out + 48,record->token_id);
	SparkSpeculationTapPut32(out + 52,record->next_token_id);
	SparkSpeculationTapPut32(out + 56,record->flags);
	SparkSpeculationTapPut32(out + 60,record_bytes);
	SparkSpeculationTapPut32(out + 64,offset);
	SparkSpeculationTapPut32(out + 68,0u);
	memcpy(out + SPARK_SPECULATION_TAP_FRAGMENT_HEADER_BYTES,payload + offset,bytes);
	*out_length = SPARK_SPECULATION_TAP_FRAGMENT_HEADER_BYTES + bytes;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkSpeculationTapDecodeFragment(const uint8_t *bytes,uint32_t length,SparkSpeculationTapFragment *fragment)
{
	uint32_t word;
	if ( bytes == 0 || fragment == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(fragment,0,sizeof(*fragment));
	if ( length <= SPARK_SPECULATION_TAP_FRAGMENT_HEADER_BYTES || length > SPARK_SPECULATION_TAP_FRAGMENT_BYTES_MAX || SparkSpeculationTapGet32(bytes) != SPARK_SPECULATION_TAP_FRAGMENT_MAGIC )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	word = SparkSpeculationTapGet32(bytes + 4);
	if ( (word & 0xffffu) != SPARK_SPECULATION_TAP_FRAGMENT_VERSION )
		SPARK_FAIL(SPARK_STATUS_ABI_MISMATCH);
	if ( (word >> 16) != SPARK_SPECULATION_TAP_FRAGMENT_KIND )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	fragment->record.engine_generation = SparkSpeculationTapGet64(bytes + 8);
	fragment->record.sequence_id = SparkSpeculationTapGet64(bytes + 16);
	fragment->record.position = SparkSpeculationTapGet64(bytes + 24);
	fragment->record.serial = SparkSpeculationTapGet64(bytes + 32);
	fragment->fingerprint = SparkSpeculationTapGet64(bytes + 40);
	fragment->record.token_id = SparkSpeculationTapGet32(bytes + 48);
	fragment->record.next_token_id = SparkSpeculationTapGet32(bytes + 52);
	fragment->record.flags = SparkSpeculationTapGet32(bytes + 56);
	fragment->record_bytes = SparkSpeculationTapGet32(bytes + 60);
	fragment->offset = SparkSpeculationTapGet32(bytes + 64);
	fragment->bytes = length - SPARK_SPECULATION_TAP_FRAGMENT_HEADER_BYTES;
	fragment->payload = bytes + SPARK_SPECULATION_TAP_FRAGMENT_HEADER_BYTES;
	if ( (fragment->record.flags & ~SPARK_SPECULATION_TAP_FLAG_MASK) != 0u || SparkSpeculationTapGet32(bytes + 68) != 0u || fragment->record_bytes == 0u || fragment->offset >= fragment->record_bytes || fragment->bytes > fragment->record_bytes - fragment->offset )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkSpeculationTapAssemblerInitialize(SparkSpeculationTapAssembler *assembler,uint64_t fingerprint,uint32_t record_bytes,uint8_t *payload)
{
	if ( assembler == 0 || payload == 0 || record_bytes == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(assembler,0,sizeof(*assembler));
	assembler->fingerprint = fingerprint;
	assembler->record_bytes = record_bytes;
	assembler->payload = payload;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkSpeculationTapAssemblerAccept(SparkSpeculationTapAssembler *assembler,const uint8_t *bytes,uint32_t length,uint32_t *complete_out)
{
	SparkSpeculationTapFragment fragment;
	SparkStatus status;
	if ( assembler == 0 || assembler->payload == 0 || complete_out == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*complete_out = 0u;
	status = SparkSpeculationTapDecodeFragment(bytes,length,&fragment);
	if ( status != SPARK_STATUS_OK )
	{
		assembler->rejected++;
		SPARK_RETURN(status);
	}
	if ( fragment.fingerprint != assembler->fingerprint || fragment.record_bytes != assembler->record_bytes )
	{
		assembler->rejected++;
		SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
	}
	if ( fragment.offset == 0u )
	{
		if ( assembler->active != 0u )
			assembler->abandoned++;
		assembler->active = 1u;
		assembler->received = 0u;
		assembler->record = fragment.record;
	}
	else if ( assembler->active == 0u || fragment.offset != assembler->received || memcmp(&fragment.record,&assembler->record,sizeof(fragment.record)) != 0 )
	{
		if ( assembler->active != 0u )
			assembler->abandoned++;
		assembler->active = 0u;
		return(SPARK_STATUS_OK);
	}
	memcpy(assembler->payload + fragment.offset,fragment.payload,fragment.bytes);
	assembler->received += fragment.bytes;
	if ( assembler->received == assembler->record_bytes )
	{
		assembler->active = 0u;
		assembler->completed++;
		*complete_out = 1u;
	}
	return(SPARK_STATUS_OK);
}

static void SparkSpeculationTapDumpHeader(const SparkSpeculationTapDump *dump,const char *model_tag,uint64_t engine_generation,uint32_t tp_rank,uint8_t header[SPARK_SPECULATION_TAP_DUMP_HEADER_BYTES])
{
	uint32_t index;
	memset(header,0,SPARK_SPECULATION_TAP_DUMP_HEADER_BYTES);
	SparkSpeculationTapPut32(header,SPARK_SPECULATION_TAP_DUMP_MAGIC);
	SparkSpeculationTapPut32(header + 4,SPARK_SPECULATION_TAP_DUMP_VERSION);
	SparkSpeculationTapPut32(header + 8,SPARK_SPECULATION_TAP_DUMP_HEADER_BYTES);
	SparkSpeculationTapPut32(header + 12,SPARK_SPECULATION_TAP_DUMP_RECORD_HEADER_BYTES);
	SparkSpeculationTapPut32(header + 16,dump->set.reduction);
	SparkSpeculationTapPut32(header + 20,dump->set.tap_count);
	SparkSpeculationTapPut32(header + 24,dump->set.stream_count);
	SparkSpeculationTapPut32(header + 28,dump->set.hidden_dimension);
	SparkSpeculationTapPut32(header + 32,dump->set.layer_count);
	SparkSpeculationTapPut32(header + 36,dump->set.row_elements);
	SparkSpeculationTapPut32(header + 40,dump->set.row_bytes);
	SparkSpeculationTapPut32(header + 44,SPARK_SPECULATION_TAP_DTYPE_BF16);
	for (index=0u; index<dump->set.tap_count; index++)
		SparkSpeculationTapPut32(header + 48 + 4u * index,dump->set.layers[index]);
	SparkSpeculationTapPut64(header + 80,SparkSpeculationTapSetFingerprint(&dump->set));
	SparkSpeculationTapPut64(header + 88,engine_generation);
	SparkSpeculationTapPut32(header + 96,tp_rank);
	SparkSpeculationTapPut32(header + 100,dump->flags);
	SparkSpeculationTapPut64(header + 104,dump->records);
	if ( model_tag != 0 )
		memcpy(header + 112,model_tag,strnlen(model_tag,SPARK_SPECULATION_TAP_MODEL_TAG_BYTES));
}

SparkStatus SparkSpeculationTapDumpOpen(SparkSpeculationTapDump *dump,const char *path,const SparkSpeculationTapSet *set,const char *model_tag,uint64_t engine_generation,uint32_t tp_rank,uint64_t max_bytes)
{
	uint8_t header[SPARK_SPECULATION_TAP_DUMP_HEADER_BYTES];
	if ( dump == 0 || path == 0 || set == 0 || set->tap_count == 0u || set->record_bytes == 0u || max_bytes < SPARK_SPECULATION_TAP_DUMP_HEADER_BYTES )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(dump,0,sizeof(*dump));
	dump->set = *set;
	dump->max_bytes = max_bytes;
	dump->file = fopen(path,"wbx");
	if ( dump->file == 0 )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	SparkSpeculationTapDumpHeader(dump,model_tag,engine_generation,tp_rank,header);
	if ( fwrite(header,1u,sizeof(header),dump->file) != sizeof(header) )
	{
		fclose(dump->file);
		dump->file = 0;
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	dump->bytes = sizeof(header);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkSpeculationTapDumpAppend(SparkSpeculationTapDump *dump,const SparkSpeculationTapRecord *record,const uint8_t *payload)
{
	uint8_t header[SPARK_SPECULATION_TAP_DUMP_RECORD_HEADER_BYTES];
	uint64_t bytes;
	if ( dump == 0 || dump->file == 0 || record == 0 || payload == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (record->flags & ~SPARK_SPECULATION_TAP_FLAG_MASK) != 0u || record->reserved != 0u )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	bytes = SPARK_SPECULATION_TAP_DUMP_RECORD_HEADER_BYTES + (uint64_t)dump->set.record_bytes;
	if ( dump->bytes + bytes > dump->max_bytes )
	{
		dump->flags |= SPARK_SPECULATION_TAP_DUMP_FLAG_TRUNCATED;
		dump->refused++;
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	SparkSpeculationTapPut64(header,record->sequence_id);
	SparkSpeculationTapPut64(header + 8,record->position);
	SparkSpeculationTapPut32(header + 16,record->token_id);
	SparkSpeculationTapPut32(header + 20,record->next_token_id);
	SparkSpeculationTapPut32(header + 24,record->flags);
	SparkSpeculationTapPut32(header + 28,0u);
	SparkSpeculationTapPut64(header + 32,record->serial);
	if ( fwrite(header,1u,sizeof(header),dump->file) != sizeof(header) || fwrite(payload,1u,dump->set.record_bytes,dump->file) != dump->set.record_bytes )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	dump->bytes += bytes;
	dump->records++;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkSpeculationTapDumpClose(SparkSpeculationTapDump *dump)
{
	uint8_t word[12];
	int failed;
	if ( dump == 0 || dump->file == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	dump->flags |= SPARK_SPECULATION_TAP_DUMP_FLAG_CLOSED;
	SparkSpeculationTapPut32(word,dump->flags);
	SparkSpeculationTapPut64(word + 4,dump->records);
	failed = fflush(dump->file) != 0 || fseek(dump->file,100L,SEEK_SET) != 0 || fwrite(word,1u,sizeof(word),dump->file) != sizeof(word);
	failed |= fclose(dump->file) != 0;
	dump->file = 0;
	if ( failed )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return(SPARK_STATUS_OK);
}
