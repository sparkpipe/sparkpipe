#include "sparkpipe/spark_weightd_receipt.h"
#include "sparkpipe/spark_ck128.h"
#include "sparkpipe/spark_sha256.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct ToolSink
{
	SparkSha256Context sha;
	SparkCk128Context ck;
	uint32_t hash;
} ToolSink;

static SparkStatus tool_sink(void *context,const uint8_t *data,uint64_t offset,uint64_t bytes)
{
	ToolSink *sink = (ToolSink *)context;
	(void)offset;
	if ( (sink->hash & 1u) != 0u )
		SparkSha256Update(&sink->sha,data,(size_t)bytes);
	if ( (sink->hash & 2u) != 0u )
		SparkCk128Update(&sink->ck,data,(size_t)bytes);
	return(SPARK_STATUS_OK);
}

static int tool_digest(const char *pack,const char *given,char hex[65])
{
	SparkStatus status;
	if ( given != 0 )
	{
		if ( SparkSha256HexIsValid(given) == 0 )
		{
			fprintf(stderr,"weightd_receipt: %s is not a lowercase sha256 hex digest\n",given);
			return(2);
		}
		memcpy(hex,given,SPARK_SHA256_HEX_BYTES);
		return(0);
	}
	status = SparkWeightdPackDigestRead(pack,hex);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"weightd_receipt: %s.sha256 unreadable status=%s; pass the digest explicitly\n",pack,SparkStatusToString(status));
		return(2);
	}
	return(0);
}

static int tool_open(const char *pack,struct stat *info)
{
	int fd = open(pack,O_RDONLY | O_CLOEXEC);
	if ( fd < 0 || fstat(fd,info) != 0 || S_ISREG(info->st_mode) == 0 || info->st_size <= 0 )
	{
		fprintf(stderr,"weightd_receipt: cannot open regular pack %s errno=%d\n",pack,errno);
		if ( fd >= 0 )
			(void)close(fd);
		return(-1);
	}
	return(fd);
}

static void tool_report(const char *verb,const char *pack,const SparkWeightdStreamStats *stats)
{
	double seconds = (double)stats->elapsed_ns / 1e9;
	double work = (double)stats->sink_ns / 1e9;
	printf("weightd_receipt %s path=%s bytes=%llu seconds=%.3f gbps=%.2f io=%s read_wait_s=%.3f work_s=%.3f work_gbps=%.2f\n",
		verb,pack,(unsigned long long)stats->bytes,seconds,seconds > 0.0 ? (double)stats->bytes / seconds / 1e9 : 0.0,
		stats->direct != 0u ? "direct" : "buffered",(double)stats->wait_ns / 1e9,work,work > 0.0 ? (double)stats->bytes / work / 1e9 : 0.0);
}

static int tool_stream(const char *pack,uint32_t hash,const char *expected,int record)
{
	SparkWeightdStreamStats stats;
	ToolSink sink;
	struct stat before,after;
	uint8_t digest[32];
	char sha_hex[SPARK_SHA256_HEX_BYTES],ck_hex[SPARK_CK128_HEX_BYTES];
	SparkStatus status;
	int fd = tool_open(pack,&before);
	if ( fd < 0 )
		return(1);
	memset(&sink,0,sizeof(sink));
	sink.hash = hash;
	SparkSha256Initialize(&sink.sha);
	SparkCk128Initialize(&sink.ck);
	status = SparkWeightdPackStream(pack,fd,(uint64_t)before.st_size,tool_sink,&sink,&stats);
	if ( status != SPARK_STATUS_OK || fstat(fd,&after) != 0 || SparkWeightdPackStatSame(&before,&after) == 0 )
	{
		fprintf(stderr,"weightd_receipt: %s read failed or changed while read status=%s\n",pack,SparkStatusToString(status));
		(void)close(fd);
		return(1);
	}
	tool_report(hash == 0u ? "read" : hash == 1u ? "sha256" : hash == 2u ? "ck128" : "verify",pack,&stats);
	if ( record == 0 && hash != 1u )
	{
		(void)close(fd);
		return(0);
	}
	SparkSha256Finalize(&sink.sha,digest);
	SparkSha256DigestToHex(digest,sha_hex);
	SparkCk128Finalize(&sink.ck,digest);
	SparkCk128DigestToHex(digest,ck_hex);
	if ( strcmp(sha_hex,expected) != 0 )
	{
		fprintf(stderr,"weightd_receipt: HASH MISMATCH path=%s expected=%s computed=%s\n",pack,expected,sha_hex);
		(void)close(fd);
		return(3);
	}
	status = record != 0 ? SparkWeightdReceiptRecord(pack,fd,&before,sha_hex,ck_hex,"tool-verify") : SPARK_STATUS_OK;
	(void)close(fd);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"weightd_receipt: %s verified but the receipt was not written status=%s\n",pack,SparkStatusToString(status));
		return(1);
	}
	printf("weightd_receipt verified path=%s sha256=%s ck128=%s\n",pack,sha_hex,(hash & 2u) != 0u ? ck_hex : "-");
	return(0);
}

static int tool_check(const char *pack,const char *expected)
{
	struct stat info;
	const char *reason = "absent";
	SparkStatus status;
	int fd = tool_open(pack,&info);
	if ( fd < 0 )
		return(1);
	status = SparkWeightdReceiptCheck(pack,fd,expected,0,&reason);
	(void)close(fd);
	printf("weightd_receipt check path=%s status=%s reason=%s\n",pack,SparkStatusToString(status),reason);
	return(status == SPARK_STATUS_OK ? 0 : 1);
}

static int tool_show(const char *pack)
{
	char primary[SPARK_WEIGHTD_RECEIPT_PATH_BYTES],fallback[SPARK_WEIGHTD_RECEIPT_PATH_BYTES];
	const char *paths[2];
	SparkWeightdReceipt receipt;
	struct stat info;
	SparkStatus status;
	int fd = tool_open(pack,&info),shown = 0;
	if ( fd < 0 )
		return(1);
	(void)close(fd);
	if ( SparkWeightdReceiptLocate(pack,&info,primary,fallback) != SPARK_STATUS_OK )
		return(1);
	paths[0] = primary;
	paths[1] = fallback;
	for (int i=0; i<2; i++)
	{
		if ( paths[i][0] == 0 )
			continue;
		status = SparkWeightdReceiptLoad(paths[i],&receipt);
		printf("receipt %s status=%s\n",paths[i],SparkStatusToString(status));
		if ( status != SPARK_STATUS_OK )
			continue;
		shown = 1;
		printf("  device=%llu inode=%llu size=%llu mtime_ns=%llu ctime_ns=%llu\n  sha256=%s ck128=%s\n  verifier=%s verified_unix_ns=%llu\n",
			(unsigned long long)receipt.device,(unsigned long long)receipt.inode,(unsigned long long)receipt.size,
			(unsigned long long)receipt.mtime_ns,(unsigned long long)receipt.ctime_ns,
			receipt.sha256[0] != 0 ? receipt.sha256 : "-",receipt.ck128[0] != 0 ? receipt.ck128 : "-",
			receipt.verifier,(unsigned long long)receipt.verified_unix_ns);
	}
	return(shown != 0 ? 0 : 1);
}

int main(int argc,char **argv)
{
	char expected[65];
	int status;
	if ( argc < 3 )
	{
		fprintf(stderr,"usage: weightd_receipt verify <pack> [sha256]\n"
			"       weightd_receipt check <pack> [sha256]\n"
			"       weightd_receipt read <pack>\n"
			"       weightd_receipt hash <pack> [sha256]\n"
			"       weightd_receipt ck128 <pack>\n"
			"       weightd_receipt show <pack>\n");
		return(2);
	}
	if ( strcmp(argv[1],"read") == 0 )
		return(tool_stream(argv[2],0u,0,0));
	if ( strcmp(argv[1],"ck128") == 0 )
		return(tool_stream(argv[2],2u,0,0));
	if ( strcmp(argv[1],"show") == 0 )
		return(tool_show(argv[2]));
	status = tool_digest(argv[2],argc > 3 ? argv[3] : 0,expected);
	if ( status != 0 )
		return(status);
	if ( strcmp(argv[1],"verify") == 0 )
		return(tool_stream(argv[2],3u,expected,1));
	if ( strcmp(argv[1],"hash") == 0 )
		return(tool_stream(argv[2],1u,expected,0));
	if ( strcmp(argv[1],"check") == 0 )
		return(tool_check(argv[2],expected));
	fprintf(stderr,"weightd_receipt: unknown command %s\n",argv[1]);
	return(2);
}
