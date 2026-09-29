#include "sparkpipe/spark_score_dump.h"

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "sparkpipe/spark_sha256.h"

#define SPARK_SCORE_DUMP_ROWS_MAGIC "SPSCORE1"
#define SPARK_SCORE_DUMP_TIER2_OUT_MAGIC "SPSCORT2"
#define SPARK_SCORE_DUMP_PROBE_MAGIC "SPPROBE1"
#define SPARK_SCORE_DUMP_TIER2_MAGIC "SPTIER21"
#define SPARK_SCORE_DUMP_TABLE_ENTRY_FIXED_BYTES 16u

uint64_t SparkScoreDumpKeyNext(uint64_t previous, uint32_t token)
{
	uint64_t hash = previous;
	uint32_t index;
	for (index = 0u; index < 4u; index++)
	{
		hash ^= (uint64_t)((token >> (8u * index)) & 0xffu);
		hash *= SPARK_SCORE_DUMP_KEY_PRIME;
	}
	return(hash);
}

SparkStatus SparkScoreDumpShard(uint32_t vocabulary, uint32_t tp_degree, uint32_t tp_rank, uint32_t *begin, uint32_t *end)
{
	uint32_t width;
	if ( begin == 0 || end == 0 || tp_degree == 0u || tp_rank >= tp_degree || vocabulary == 0u || vocabulary % tp_degree != 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	width = vocabulary / tp_degree;
	*begin = tp_rank * width;
	*end = *begin + width;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkScoreDumpKeysInitialize(SparkScoreDumpKeys *keys, uint32_t slot_count, uint32_t position_count)
{
	if ( keys == 0 || slot_count == 0u || position_count == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(keys, 0, sizeof(*keys));
	keys->keys = (uint64_t *)calloc((size_t)slot_count * position_count, sizeof(uint64_t));
	keys->known = (uint32_t *)calloc(slot_count, sizeof(uint32_t));
	if ( keys->keys == 0 || keys->known == 0 )
	{
		SparkScoreDumpKeysDestroy(keys);
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	keys->slot_count = slot_count;
	keys->position_count = position_count;
	return(SPARK_STATUS_OK);
}

void SparkScoreDumpKeysDestroy(SparkScoreDumpKeys *keys)
{
	if ( keys == 0 )
		return;
	free(keys->keys);
	free(keys->known);
	memset(keys, 0, sizeof(*keys));
}

uint32_t SparkScoreDumpKeysAdvance(SparkScoreDumpKeys *keys, uint32_t slot, uint32_t position, uint32_t token, uint64_t *key)
{
	uint64_t previous;
	uint64_t *row;
	if ( keys == 0 || key == 0 || keys->keys == 0 || slot >= keys->slot_count || position >= keys->position_count )
		return(0u);
	row = keys->keys + (uint64_t)slot * keys->position_count;
	if ( position == 0u )
		previous = SPARK_SCORE_DUMP_KEY_SEED;
	else if ( position <= keys->known[slot] )
		previous = row[position - 1u];
	else
	{
		keys->known[slot] = 0u;
		return(0u);
	}
	row[position] = SparkScoreDumpKeyNext(previous, token);
	keys->known[slot] = position + 1u;
	*key = row[position];
	return(1u);
}

static uint64_t SparkScoreDumpLoad64(const uint8_t *bytes)
{
	uint64_t value;
	memcpy(&value, bytes, sizeof(value));
	return(value);
}

static uint32_t SparkScoreDumpLoad32(const uint8_t *bytes)
{
	uint32_t value;
	memcpy(&value, bytes, sizeof(value));
	return(value);
}

static int32_t SparkScoreDumpEntryCompare(uint64_t key_a, uint32_t position_a, uint64_t key_b, uint32_t position_b)
{
	if ( key_a != key_b )
		return(key_a < key_b ? -1 : 1);
	if ( position_a != position_b )
		return(position_a < position_b ? -1 : 1);
	return(0);
}

void SparkScoreDumpTableDestroy(SparkScoreDumpTable *table)
{
	if ( table == 0 )
		return;
	free(table->bytes);
	memset(table, 0, sizeof(*table));
}

static SparkStatus SparkScoreDumpReadWhole(const char *path, uint8_t **bytes_out, uint64_t *count_out)
{
	FILE *file;
	uint8_t *bytes;
	long length;
	file = fopen(path, "rb");
	if ( file == 0 )
		return(SPARK_STATUS_NOT_FOUND);
	if ( fseek(file, 0, SEEK_END) != 0 || (length = ftell(file)) < 0 || fseek(file, 0, SEEK_SET) != 0 )
	{
		fclose(file);
		return(SPARK_STATUS_IO_ERROR);
	}
	bytes = (uint8_t *)malloc(length > 0 ? (size_t)length : 1u);
	if ( bytes == 0 )
	{
		fclose(file);
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	if ( length > 0 && fread(bytes, 1u, (size_t)length, file) != (size_t)length )
	{
		free(bytes);
		fclose(file);
		return(SPARK_STATUS_IO_ERROR);
	}
	fclose(file);
	*bytes_out = bytes;
	*count_out = (uint64_t)length;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkScoreDumpTableLoad(const char *path, const char *magic, uint32_t require_ids, SparkScoreDumpTable *table)
{
	SparkScoreDumpProbeHeader header;
	SparkSha256Context sha;
	SparkStatus status;
	uint64_t index,expected;
	const uint8_t *entry;
	const uint8_t *previous;
	if ( path == 0 || magic == 0 || table == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(table, 0, sizeof(*table));
	status = SparkScoreDumpReadWhole(path, &table->bytes, &table->byte_count);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( table->byte_count < sizeof(header) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK )
	{
		memcpy(&header, table->bytes, sizeof(header));
		if ( memcmp(header.magic, magic, sizeof(header.magic)) != 0 || header.version != SPARK_SCORE_DUMP_VERSION || (require_ids != 0u) != (header.id_width != 0u) || header.id_width > 4096u )
			status = SPARK_STATUS_SCHEMA_ERROR;
	}
	if ( status == SPARK_STATUS_OK )
	{
		table->id_width = header.id_width;
		table->entry_bytes = SPARK_SCORE_DUMP_TABLE_ENTRY_FIXED_BYTES + 4u * header.id_width;
		table->entry_count = header.entry_count;
		expected = sizeof(header) + table->entry_count * table->entry_bytes;
		if ( table->entry_count > (table->byte_count / table->entry_bytes) || expected != table->byte_count )
			status = SPARK_STATUS_SCHEMA_ERROR;
	}
	for (index = 0u; status == SPARK_STATUS_OK && index < table->entry_count; index++)
	{
		entry = table->bytes + sizeof(header) + index * table->entry_bytes;
		if ( SparkScoreDumpLoad32(entry + 12u) > table->id_width )
			status = SPARK_STATUS_SCHEMA_ERROR;
		if ( status == SPARK_STATUS_OK && index > 0u )
		{
			previous = entry - table->entry_bytes;
			if ( SparkScoreDumpEntryCompare(SparkScoreDumpLoad64(previous), SparkScoreDumpLoad32(previous + 8u), SparkScoreDumpLoad64(entry), SparkScoreDumpLoad32(entry + 8u)) >= 0 )
				status = SPARK_STATUS_SCHEMA_ERROR;
		}
	}
	if ( status == SPARK_STATUS_OK )
	{
		SparkSha256Initialize(&sha);
		SparkSha256Update(&sha, table->bytes, (size_t)table->byte_count);
		SparkSha256Finalize(&sha, table->sha256);
	}
	if ( status != SPARK_STATUS_OK )
		SparkScoreDumpTableDestroy(table);
	return(status);
}

const uint32_t *SparkScoreDumpTableFind(const SparkScoreDumpTable *table, uint64_t key, uint32_t position, uint32_t *id_count)
{
	uint64_t low,high,middle;
	const uint8_t *entry;
	int32_t order;
	if ( table == 0 || table->bytes == 0 )
		return(0);
	low = 0u;
	high = table->entry_count;
	while ( low < high )
	{
		middle = low + (high - low) / 2u;
		entry = table->bytes + sizeof(SparkScoreDumpProbeHeader) + middle * table->entry_bytes;
		order = SparkScoreDumpEntryCompare(SparkScoreDumpLoad64(entry), SparkScoreDumpLoad32(entry + 8u), key, position);
		if ( order == 0 )
		{
			if ( id_count != 0 )
				*id_count = SparkScoreDumpLoad32(entry + 12u);
			return((const uint32_t *)(const void *)(entry + SPARK_SCORE_DUMP_TABLE_ENTRY_FIXED_BYTES));
		}
		if ( order < 0 )
			low = middle + 1u;
		else
			high = middle;
	}
	return(0);
}

static SparkStatus SparkScoreDumpCreate(const char *directory, const char *name, uint32_t rank, char *path, FILE **file)
{
	int descriptor;
	int written;
	written = snprintf(path, SPARK_SCORE_DUMP_PATH_BYTES, "%s/%s.r%02u.bin", directory, name, rank);
	if ( written <= 0 || (size_t)written >= SPARK_SCORE_DUMP_PATH_BYTES )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	descriptor = open(path, O_WRONLY | O_CREAT | O_EXCL, 0640);
	if ( descriptor < 0 )
		return(errno == EEXIST ? SPARK_STATUS_DUPLICATE : SPARK_STATUS_IO_ERROR);
	*file = fdopen(descriptor, "wb");
	if ( *file == 0 )
	{
		close(descriptor);
		return(SPARK_STATUS_IO_ERROR);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkScoreDumpWriteAll(FILE *file, const void *bytes, size_t count)
{
	if ( count == 0u )
		return(SPARK_STATUS_OK);
	return(fwrite(bytes, 1u, count, file) == count ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR);
}

static void SparkScoreDumpRelease(SparkScoreDumpWriter *writer)
{
	if ( writer->rows != 0 )
		fclose(writer->rows);
	if ( writer->tier2 != 0 )
		fclose(writer->tier2);
	writer->rows = 0;
	writer->tier2 = 0;
	SparkScoreDumpTableDestroy(&writer->probes);
	SparkScoreDumpTableDestroy(&writer->tier2_rows);
	SparkScoreDumpKeysDestroy(&writer->keys);
}

SparkStatus SparkScoreDumpOpen(const SparkScoreDumpConfig *config, SparkScoreDumpWriter *writer)
{
	SparkScoreDumpFileHeader tier2_header;
	SparkStatus status;
	struct stat info;
	if ( config == 0 || writer == 0 || config->directory == 0 || config->directory[0] == '\0' || config->hidden_dimension == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(writer, 0, sizeof(*writer));
	if ( stat(config->directory, &info) != 0 || !S_ISDIR(info.st_mode) )
		return(SPARK_STATUS_NOT_FOUND);
	memcpy(writer->header.magic, SPARK_SCORE_DUMP_ROWS_MAGIC, sizeof(writer->header.magic));
	writer->header.version = SPARK_SCORE_DUMP_VERSION;
	writer->header.header_bytes = (uint32_t)sizeof(writer->header);
	writer->header.tp_rank = config->tp_rank;
	writer->header.tp_degree = config->tp_degree;
	writer->header.vocabulary = config->vocabulary;
	writer->header.hidden_dimension = config->hidden_dimension;
	writer->header.top_k = SPARK_SCORE_DUMP_TOP_K;
	if ( config->arm_digest != 0 )
		memcpy(writer->header.arm_digest, config->arm_digest, sizeof(writer->header.arm_digest));
	status = SparkScoreDumpShard(config->vocabulary, config->tp_degree, config->tp_rank, &writer->header.shard_begin, &writer->header.shard_end);
	if ( status == SPARK_STATUS_OK && writer->header.shard_end - writer->header.shard_begin < SPARK_SCORE_DUMP_TOP_K )
		status = SPARK_STATUS_INVALID_ARGUMENT;
	if ( status == SPARK_STATUS_OK )
		status = SparkScoreDumpKeysInitialize(&writer->keys, config->slot_count, config->position_count);
	if ( status == SPARK_STATUS_OK && config->probe_path != 0 )
		status = SparkScoreDumpTableLoad(config->probe_path, SPARK_SCORE_DUMP_PROBE_MAGIC, 1u, &writer->probes);
	if ( status == SPARK_STATUS_OK && config->probe_path != 0 )
		memcpy(writer->header.probe_sha256, writer->probes.sha256, sizeof(writer->header.probe_sha256));
	if ( status == SPARK_STATUS_OK && config->tier2_path != 0 )
	{
		status = SparkScoreDumpTableLoad(config->tier2_path, SPARK_SCORE_DUMP_TIER2_MAGIC, 0u, &writer->tier2_rows);
		if ( status == SPARK_STATUS_OK )
		{
			memcpy(writer->header.tier2_sha256, writer->tier2_rows.sha256, sizeof(writer->header.tier2_sha256));
			writer->header.tier2 = 1u;
		}
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkScoreDumpCreate(config->directory, "score", config->tp_rank, writer->rows_path, &writer->rows);
	if ( status == SPARK_STATUS_OK )
		status = SparkScoreDumpWriteAll(writer->rows, &writer->header, sizeof(writer->header));
	if ( status == SPARK_STATUS_OK && writer->header.tier2 != 0u )
	{
		status = SparkScoreDumpCreate(config->directory, "tier2", config->tp_rank, writer->tier2_path, &writer->tier2);
		tier2_header = writer->header;
		memcpy(tier2_header.magic, SPARK_SCORE_DUMP_TIER2_OUT_MAGIC, sizeof(tier2_header.magic));
		if ( status == SPARK_STATUS_OK )
			status = SparkScoreDumpWriteAll(writer->tier2, &tier2_header, sizeof(tier2_header));
	}
	writer->end.record_kind = SPARK_SCORE_DUMP_RECORD_END;
	if ( status != SPARK_STATUS_OK )
		SparkScoreDumpRelease(writer);
	return(status);
}

SparkStatus SparkScoreDumpWriteRow(SparkScoreDumpWriter *writer, const SparkScoreDumpRowRecord *row, const uint32_t *probe_ids, const float *probe_logits)
{
	SparkStatus status;
	uint32_t index;
	if ( writer == 0 || writer->rows == 0 || row == 0 || row->record_kind != SPARK_SCORE_DUMP_RECORD_ROW || (row->probe_count != 0u && (probe_ids == 0 || probe_logits == 0)) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkScoreDumpWriteAll(writer->rows, row, sizeof(*row));
	for (index = 0u; status == SPARK_STATUS_OK && index < row->probe_count; index++)
	{
		status = SparkScoreDumpWriteAll(writer->rows, probe_ids + index, sizeof(uint32_t));
		if ( status == SPARK_STATUS_OK )
			status = SparkScoreDumpWriteAll(writer->rows, probe_logits + index, sizeof(float));
	}
	if ( status != SPARK_STATUS_OK )
		return(status);
	writer->end.row_count++;
	if ( (row->flags & SPARK_SCORE_DUMP_ROW_KEY_VALID) == 0u )
		writer->end.keyless_row_count++;
	if ( (row->flags & SPARK_SCORE_DUMP_ROW_PROBED) != 0u )
		writer->end.probed_row_count++;
	if ( (row->flags & SPARK_SCORE_DUMP_ROW_NONFINITE) != 0u )
		writer->end.nonfinite_row_count++;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkScoreDumpWriteTier2(SparkScoreDumpWriter *writer, uint64_t key, uint32_t position, const float *logits)
{
	SparkStatus status;
	uint32_t width;
	if ( writer == 0 || writer->tier2 == 0 || logits == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	width = writer->header.shard_end - writer->header.shard_begin;
	status = SparkScoreDumpWriteAll(writer->tier2, &key, sizeof(key));
	if ( status == SPARK_STATUS_OK )
		status = SparkScoreDumpWriteAll(writer->tier2, &position, sizeof(position));
	if ( status == SPARK_STATUS_OK )
		status = SparkScoreDumpWriteAll(writer->tier2, &width, sizeof(width));
	if ( status == SPARK_STATUS_OK )
		status = SparkScoreDumpWriteAll(writer->tier2, logits, (size_t)width * sizeof(float));
	if ( status == SPARK_STATUS_OK )
		writer->end.tier2_row_count++;
	return(status);
}

void SparkScoreDumpNoteWave(SparkScoreDumpWriter *writer, uint32_t skipped)
{
	if ( writer == 0 )
		return;
	if ( skipped != 0u )
		writer->end.skipped_wave_count++;
	else
		writer->end.wave_count++;
}

SparkStatus SparkScoreDumpClose(SparkScoreDumpWriter *writer)
{
	SparkStatus status;
	if ( writer == 0 || writer->rows == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkScoreDumpWriteAll(writer->rows, &writer->end, sizeof(writer->end));
	if ( status == SPARK_STATUS_OK && (fflush(writer->rows) != 0 || fsync(fileno(writer->rows)) != 0) )
		status = SPARK_STATUS_IO_ERROR;
	if ( status == SPARK_STATUS_OK && writer->tier2 != 0 && (fflush(writer->tier2) != 0 || fsync(fileno(writer->tier2)) != 0) )
		status = SPARK_STATUS_IO_ERROR;
	SparkScoreDumpRelease(writer);
	return(status);
}
