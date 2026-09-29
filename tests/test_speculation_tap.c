#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "sparkpipe/spark_speculation_tap.h"

#define TEST_LAYERS 45u
#define TEST_STREAMS 4u
#define TEST_HIDDEN 4096u

static void Require(int condition,const char *what)
{
	if ( condition )
		return;
	fprintf(stderr,"FAIL %s\n",what);
	exit(1);
}

static uint32_t Get32(const uint8_t *bytes)
{
	return((uint32_t)bytes[0] | (uint32_t)bytes[1] << 8 | (uint32_t)bytes[2] << 16 | (uint32_t)bytes[3] << 24);
}

static uint64_t Get64(const uint8_t *bytes)
{
	return((uint64_t)Get32(bytes) | (uint64_t)Get32(bytes + 4) << 32);
}

static void Fill(uint8_t *payload,uint32_t bytes,uint64_t seed)
{
	uint32_t index;
	for (index=0u; index<bytes; index++)
		payload[index] = (uint8_t)((seed * 131u + index * 7u + (index >> 8)) & 0xffu);
}

static void TestParse(void)
{
	SparkSpeculationTapSet set,other;
	Require(SparkSpeculationTapSetParse("mean:5,14,24,33,42,44",TEST_LAYERS,TEST_STREAMS,TEST_HIDDEN,&set) == SPARK_STATUS_OK,"dflash2 taps plus the last layer parse");
	Require(set.reduction == SPARK_SPECULATION_TAP_REDUCTION_MEAN && set.tap_count == 6u && set.row_elements == TEST_HIDDEN && set.row_bytes == 8192u && set.record_bytes == 6u * 8192u,"mean rows are one hidden row of bf16 per tap");
	Require(SparkSpeculationTapSetOrdinal(&set,44u) == 5u && SparkSpeculationTapSetOrdinal(&set,5u) == 0u && SparkSpeculationTapSetOrdinal(&set,6u) == UINT32_MAX,"ordinal maps layers to tap slots and refuses other layers");
	Require(SparkSpeculationTapSetParse("all:44",TEST_LAYERS,TEST_STREAMS,TEST_HIDDEN,&other) == SPARK_STATUS_OK && other.reduction == SPARK_SPECULATION_TAP_REDUCTION_ALL && other.row_bytes == 32768u && other.record_bytes == 32768u,"all keeps every stream");
	Require(SparkSpeculationTapSetFingerprint(&set) != SparkSpeculationTapSetFingerprint(&other),"fingerprint separates tap sets");
	Require(SparkSpeculationTapSetParse("all:5,14,24,33,42,44",TEST_LAYERS,TEST_STREAMS,TEST_HIDDEN,&other) == SPARK_STATUS_OK && SparkSpeculationTapSetFingerprint(&set) != SparkSpeculationTapSetFingerprint(&other),"fingerprint covers the reduction");
	Require(SparkSpeculationTapSetParse("mean:5,14,24,33,42,43",TEST_LAYERS,TEST_STREAMS,TEST_HIDDEN,&other) == SPARK_STATUS_OK && SparkSpeculationTapSetFingerprint(&set) != SparkSpeculationTapSetFingerprint(&other),"fingerprint covers every layer index");
	Require(SparkSpeculationTapSetParse("mean:5,14,24,33,42,44",TEST_LAYERS,TEST_STREAMS,TEST_HIDDEN,&other) == SPARK_STATUS_OK && SparkSpeculationTapSetFingerprint(&set) == SparkSpeculationTapSetFingerprint(&other),"fingerprint is stable");
	Require(SparkSpeculationTapSetParse("mean:",TEST_LAYERS,TEST_STREAMS,TEST_HIDDEN,&other) == SPARK_STATUS_PARSE_ERROR,"empty layer list refused");
	Require(SparkSpeculationTapSetParse("mean:5,",TEST_LAYERS,TEST_STREAMS,TEST_HIDDEN,&other) == SPARK_STATUS_PARSE_ERROR,"trailing comma refused");
	Require(SparkSpeculationTapSetParse("mean:5;6",TEST_LAYERS,TEST_STREAMS,TEST_HIDDEN,&other) == SPARK_STATUS_PARSE_ERROR,"other separators refused");
	Require(SparkSpeculationTapSetParse("sum:5",TEST_LAYERS,TEST_STREAMS,TEST_HIDDEN,&other) == SPARK_STATUS_PARSE_ERROR,"unknown reduction refused");
	Require(SparkSpeculationTapSetParse("mean:45",TEST_LAYERS,TEST_STREAMS,TEST_HIDDEN,&other) == SPARK_STATUS_SCHEMA_ERROR,"layer past the model refused");
	Require(SparkSpeculationTapSetParse("mean:5,5",TEST_LAYERS,TEST_STREAMS,TEST_HIDDEN,&other) == SPARK_STATUS_SCHEMA_ERROR,"duplicate layer refused");
	Require(SparkSpeculationTapSetParse("mean:14,5",TEST_LAYERS,TEST_STREAMS,TEST_HIDDEN,&other) == SPARK_STATUS_SCHEMA_ERROR,"unordered layers refused");
	Require(SparkSpeculationTapSetParse("mean:1,2,3,4,5,6,7,8,9",TEST_LAYERS,TEST_STREAMS,TEST_HIDDEN,&other) == SPARK_STATUS_CAPACITY_EXCEEDED,"more than eight taps refused");
	Require(SparkSpeculationTapSetParse("mean:99999999999",TEST_LAYERS,TEST_STREAMS,TEST_HIDDEN,&other) == SPARK_STATUS_PARSE_ERROR,"overlong index refused");
}

static SparkSpeculationTapRecord Record(uint64_t position)
{
	SparkSpeculationTapRecord record;
	memset(&record,0,sizeof(record));
	record.engine_generation = 0x0102030405060708u;
	record.sequence_id = 77u;
	record.position = position;
	record.serial = position + 1000u;
	record.token_id = (uint32_t)(position * 3u + 1u);
	record.flags = SPARK_SPECULATION_TAP_FLAG_DECODE;
	return(record);
}

static void TestFragments(void)
{
	SparkSpeculationTapSet set;
	SparkSpeculationTapRecord record = Record(300u);
	SparkSpeculationTapFragment fragment;
	SparkSpeculationTapAssembler assembler;
	uint8_t *payload,*received,datagram[SPARK_SPECULATION_TAP_FRAGMENT_BYTES_MAX],saved[SPARK_SPECULATION_TAP_FRAGMENT_BYTES_MAX];
	uint32_t length,complete,offset,fragments;
	uint64_t fingerprint;
	Require(SparkSpeculationTapSetParse("mean:5,14,24,33,42,44",TEST_LAYERS,TEST_STREAMS,TEST_HIDDEN,&set) == SPARK_STATUS_OK,"set parses");
	fingerprint = SparkSpeculationTapSetFingerprint(&set);
	payload = (uint8_t *)malloc(set.record_bytes);
	received = (uint8_t *)calloc(1u,set.record_bytes);
	Require(payload != 0 && received != 0,"buffers");
	Fill(payload,set.record_bytes,record.position);
	fragments = SparkSpeculationTapFragmentCount(set.record_bytes,SPARK_SPECULATION_TAP_FRAGMENT_PAYLOAD_MAX);
	Require(fragments == 6u && SparkSpeculationTapFragmentCount(set.record_bytes,SPARK_SPECULATION_TAP_FRAGMENT_PAYLOAD_MAX + 1u) == 0u,"one 8 KiB fragment per mean tap; payload over 8 KiB refused");
	Require(SPARK_SPECULATION_TAP_FRAGMENT_BYTES_MAX + 28u <= 9000u,"a fragment fits one datagram at MTU 9000");
	Require(SparkSpeculationTapEncodeFragment(&record,fingerprint,payload,set.record_bytes,8192u,8192u,datagram,sizeof(datagram),&length) == SPARK_STATUS_OK && length == 64u + 8192u,"fragment encodes");
	Require(memcmp(datagram,"SPT1",4u) == 0 && datagram[4] == 1u && datagram[6] == 3u,"magic SPT1, version 1, kind 3");
	Require(Get64(datagram + 8) == record.engine_generation && Get64(datagram + 16) == 77u && Get64(datagram + 24) == 300u && Get64(datagram + 32) == 1300u && Get64(datagram + 40) == fingerprint,"generation, sequence, position, serial, fingerprint at fixed offsets");
	Require(Get32(datagram + 48) == 901u && Get32(datagram + 52) == SPARK_SPECULATION_TAP_FLAG_DECODE && Get32(datagram + 56) == set.record_bytes && Get32(datagram + 60) == 8192u,"token, flags, record bytes, offset at fixed offsets");
	Require(memcmp(datagram + 64,payload + 8192u,8192u) == 0,"payload slice follows the 64-byte header");
	Require(SparkSpeculationTapDecodeFragment(datagram,length,&fragment) == SPARK_STATUS_OK && memcmp(&fragment.record,&record,sizeof(record)) == 0 && fragment.offset == 8192u && fragment.bytes == 8192u && fragment.fingerprint == fingerprint,"decode inverts encode");
	memcpy(saved,datagram,length);
	saved[0] ^= 1u;
	Require(SparkSpeculationTapDecodeFragment(saved,length,&fragment) == SPARK_STATUS_PARSE_ERROR,"bad magic refused");
	memcpy(saved,datagram,length);
	saved[4] = 2u;
	Require(SparkSpeculationTapDecodeFragment(saved,length,&fragment) == SPARK_STATUS_ABI_MISMATCH,"other version refused");
	memcpy(saved,datagram,length);
	saved[6] = 1u;
	Require(SparkSpeculationTapDecodeFragment(saved,length,&fragment) == SPARK_STATUS_SCHEMA_ERROR,"relay request kind refused as a tap");
	memcpy(saved,datagram,length);
	saved[52] = 0x08u;
	Require(SparkSpeculationTapDecodeFragment(saved,length,&fragment) == SPARK_STATUS_SCHEMA_ERROR,"unknown flag bits refused");
	memcpy(saved,datagram,length);
	saved[61] = 0xc0u;
	Require(SparkSpeculationTapDecodeFragment(saved,length,&fragment) == SPARK_STATUS_SCHEMA_ERROR,"slice past the record refused");
	Require(SparkSpeculationTapDecodeFragment(datagram,64u,&fragment) == SPARK_STATUS_PARSE_ERROR,"empty payload refused");
	record.flags = 0x20u;
	Require(SparkSpeculationTapEncodeFragment(&record,fingerprint,payload,set.record_bytes,0u,8192u,datagram,sizeof(datagram),&length) == SPARK_STATUS_SCHEMA_ERROR,"encoder refuses unknown flags");
	record = Record(300u);
	Require(SparkSpeculationTapAssemblerInitialize(&assembler,fingerprint,set.record_bytes,received) == SPARK_STATUS_OK,"assembler");
	for (offset=0u; offset<set.record_bytes; offset+=8192u)
	{
		Require(SparkSpeculationTapEncodeFragment(&record,fingerprint,payload,set.record_bytes,offset,8192u,datagram,sizeof(datagram),&length) == SPARK_STATUS_OK,"encode slice");
		Require(SparkSpeculationTapAssemblerAccept(&assembler,datagram,length,&complete) == SPARK_STATUS_OK,"accept slice");
		Require(complete == (offset + 8192u == set.record_bytes ? 1u : 0u),"record completes on its last slice only");
	}
	Require(memcmp(received,payload,set.record_bytes) == 0 && assembler.completed == 1u && memcmp(&assembler.record,&record,sizeof(record)) == 0,"assembled record equals the sent one");
	Require(SparkSpeculationTapEncodeFragment(&record,fingerprint,payload,set.record_bytes,0u,8192u,datagram,sizeof(datagram),&length) == SPARK_STATUS_OK && SparkSpeculationTapAssemblerAccept(&assembler,datagram,length,&complete) == SPARK_STATUS_OK,"first slice");
	Require(SparkSpeculationTapEncodeFragment(&record,fingerprint,payload,set.record_bytes,16384u,8192u,datagram,sizeof(datagram),&length) == SPARK_STATUS_OK && SparkSpeculationTapAssemblerAccept(&assembler,datagram,length,&complete) == SPARK_STATUS_OK && complete == 0u && assembler.abandoned == 1u && assembler.active == 0u,"a missing slice abandons the record");
	Require(SparkSpeculationTapEncodeFragment(&record,fingerprint,payload,set.record_bytes,0u,8192u,datagram,sizeof(datagram),&length) == SPARK_STATUS_OK && SparkSpeculationTapAssemblerAccept(&assembler,datagram,length,&complete) == SPARK_STATUS_OK,"restart");
	record.position = 301u;
	Require(SparkSpeculationTapEncodeFragment(&record,fingerprint,payload,set.record_bytes,8192u,8192u,datagram,sizeof(datagram),&length) == SPARK_STATUS_OK && SparkSpeculationTapAssemblerAccept(&assembler,datagram,length,&complete) == SPARK_STATUS_OK && complete == 0u && assembler.abandoned == 2u,"a slice of another record abandons the open one");
	record.position = 302u;
	Require(SparkSpeculationTapEncodeFragment(&record,fingerprint + 1u,payload,set.record_bytes,0u,8192u,datagram,sizeof(datagram),&length) == SPARK_STATUS_OK && SparkSpeculationTapAssemblerAccept(&assembler,datagram,length,&complete) == SPARK_STATUS_TARGET_MISMATCH && assembler.rejected == 1u,"another tap set is refused by fingerprint");
	free(payload);
	free(received);
}

static void TestDump(void)
{
	SparkSpeculationTapSet set;
	SparkSpeculationTapDump dump;
	SparkSpeculationTapRecord record;
	char path[] = "/tmp/spark_tap_dump_testXXXXXX";
	uint8_t payload[2u * 3u * 2u],*file_bytes;
	FILE *file;
	long size;
	int fd;
	uint32_t index;
	Require(SparkSpeculationTapSetParse("mean:0,2",3u,4u,3u,&set) == SPARK_STATUS_OK && set.record_bytes == sizeof(payload),"tiny set");
	fd = mkstemp(path);
	Require(fd >= 0,"temp path");
	close(fd);
	Require(SparkSpeculationTapDumpOpen(&dump,path,&set,"model",9u,15u,1u << 20) == SPARK_STATUS_IO_ERROR,"an existing file is never overwritten");
	unlink(path);
	Require(SparkSpeculationTapDumpOpen(&dump,path,&set,"model",9u,15u,128u + 2u * (32u + sizeof(payload))) == SPARK_STATUS_OK,"dump opens");
	for (index=0u; index<2u; index++)
	{
		record = Record(index + 10u);
		Fill(payload,sizeof(payload),index);
		Require(SparkSpeculationTapDumpAppend(&dump,&record,payload) == SPARK_STATUS_OK,"append within budget");
	}
	Require(SparkSpeculationTapDumpAppend(&dump,&record,payload) == SPARK_STATUS_CAPACITY_EXCEEDED && dump.refused == 1u,"budget refusal is counted");
	Require(SparkSpeculationTapDumpClose(&dump) == SPARK_STATUS_OK,"close");
	file = fopen(path,"rb");
	Require(file != 0 && fseek(file,0L,SEEK_END) == 0,"reopen");
	size = ftell(file);
	Require(size == (long)(128u + 2u * (32u + sizeof(payload))),"header plus two records");
	file_bytes = (uint8_t *)malloc((size_t)size);
	Require(fseek(file,0L,SEEK_SET) == 0 && fread(file_bytes,1u,(size_t)size,file) == (size_t)size,"read back");
	fclose(file);
	Require(memcmp(file_bytes,"SPTD",4u) == 0 && Get32(file_bytes + 4) == 1u && Get32(file_bytes + 8) == 128u && Get32(file_bytes + 12) == 32u,"magic, version, header and record header sizes");
	Require(Get32(file_bytes + 16) == SPARK_SPECULATION_TAP_REDUCTION_MEAN && Get32(file_bytes + 20) == 2u && Get32(file_bytes + 24) == 4u && Get32(file_bytes + 28) == 3u && Get32(file_bytes + 32) == 3u,"reduction, taps, streams, hidden, layers");
	Require(Get32(file_bytes + 36) == 3u && Get32(file_bytes + 40) == 6u && Get32(file_bytes + 44) == SPARK_SPECULATION_TAP_DTYPE_BF16 && Get32(file_bytes + 48) == 0u && Get32(file_bytes + 52) == 2u,"row elements, row bytes, dtype, layer list");
	Require(Get64(file_bytes + 80) == SparkSpeculationTapSetFingerprint(&set) && Get64(file_bytes + 88) == 9u && Get32(file_bytes + 96) == 15u,"fingerprint, engine generation, rank");
	Require(Get32(file_bytes + 100) == (SPARK_SPECULATION_TAP_DUMP_FLAG_TRUNCATED | SPARK_SPECULATION_TAP_DUMP_FLAG_CLOSED) && Get64(file_bytes + 104) == 2u && memcmp(file_bytes + 112,"model",6u) == 0,"close records truncation, record count and model tag");
	Fill(payload,sizeof(payload),1u);
	Require(Get64(file_bytes + 128 + 44) == 77u && Get64(file_bytes + 128 + 44 + 8) == 11u && Get32(file_bytes + 128 + 44 + 16) == 34u && Get32(file_bytes + 128 + 44 + 20) == SPARK_SPECULATION_TAP_FLAG_DECODE && Get64(file_bytes + 128 + 44 + 24) == 1011u,"record header: sequence, position, token, flags, serial");
	Require(memcmp(file_bytes + 128 + 44 + 32,payload,sizeof(payload)) == 0,"record rows follow its header");
	free(file_bytes);
	unlink(path);
}

int main(void)
{
	TestParse();
	TestFragments();
	TestDump();
	printf("test_speculation_tap: ok\n");
	return(0);
}
