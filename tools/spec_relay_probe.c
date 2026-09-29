#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "sparkpipe/spark_speculation_relay_link.h"

#define PROBE_VOCAB 1000000u
#define PROBE_GENERATION 0x70726f6265u
#define PROBE_SEQUENCE 7u

static int Usage(void)
{
	fprintf(stderr,"usage: spark_speculation_relay_probe draftd LOCAL TAPS LAYERS STREAMS HIDDEN SECONDS\n"
		"       spark_speculation_relay_probe engine LOCAL PEER TAPS LAYERS STREAMS HIDDEN ROUNDS ROWS AWAIT_US PACE_US DEPTH\n");
	return(2);
}

static uint32_t Arg(const char *text)
{
	return((uint32_t)strtoul(text,0,10));
}

static int Compare64(const void *a,const void *b)
{
	uint64_t x = *(const uint64_t *)a,y = *(const uint64_t *)b;
	return(x < y ? -1 : x > y ? 1 : 0);
}

static int RunDraftd(char **argv)
{
	SparkSpeculationTapSet set;
	SparkSpeculationTapAssembler assembler;
	SparkSpeculationRelayFrame frame,reply;
	struct sockaddr_in local,from;
	socklen_t from_length;
	uint8_t buffer[SPARK_SPECULATION_TAP_FRAGMENT_BYTES_MAX],bytes[SPARK_SPECULATION_RELAY_FRAME_BYTES];
	uint8_t *payload;
	uint64_t end,requests = 0u,foreign = 0u;
	uint32_t complete,index;
	ssize_t length;
	int fd,buffer_bytes = 8 << 20;
	char host[64];
	const char *colon = strrchr(argv[2],':');
	if ( colon == 0 || (size_t)(colon - argv[2]) >= sizeof(host) || SparkSpeculationTapSetParse(argv[3],Arg(argv[4]),Arg(argv[5]),Arg(argv[6]),&set) != SPARK_STATUS_OK )
		return(Usage());
	memcpy(host,argv[2],(size_t)(colon - argv[2]));
	host[colon - argv[2]] = '\0';
	memset(&local,0,sizeof(local));
	local.sin_family = AF_INET;
	local.sin_port = htons((uint16_t)Arg(colon + 1));
	payload = (uint8_t *)malloc(set.record_bytes);
	fd = socket(AF_INET,SOCK_DGRAM,0);
	if ( payload == 0 || fd < 0 || inet_pton(AF_INET,host,&local.sin_addr) != 1 || bind(fd,(const struct sockaddr *)&local,sizeof(local)) != 0 )
	{
		fprintf(stderr,"draftd cannot bind %s\n",argv[2]);
		return(1);
	}
	(void)setsockopt(fd,SOL_SOCKET,SO_RCVBUF,&buffer_bytes,sizeof(buffer_bytes));
	(void)SparkSpeculationTapAssemblerInitialize(&assembler,SparkSpeculationTapSetFingerprint(&set),set.record_bytes,payload);
	end = SparkSpeculationRelayNowNs() + (uint64_t)Arg(argv[7]) * UINT64_C(1000000000);
	printf("RELAY-PROBE-DRAFTD listening=%s taps=%s record_bytes=%u\n",argv[2],argv[3],set.record_bytes);
	fflush(stdout);
	while ( SparkSpeculationRelayNowNs() < end )
	{
		from_length = sizeof(from);
		length = recvfrom(fd,buffer,sizeof(buffer),MSG_DONTWAIT,(struct sockaddr *)&from,&from_length);
		if ( length <= 0 )
			continue;
		if ( (uint32_t)length == SPARK_SPECULATION_RELAY_FRAME_BYTES && SparkSpeculationRelayDecode(buffer,(uint32_t)length,PROBE_VOCAB,&frame) == SPARK_STATUS_OK && frame.kind == SPARK_SPECULATION_RELAY_KIND_REQUEST )
		{
			reply = frame;
			reply.kind = SPARK_SPECULATION_RELAY_KIND_DRAFT;
			reply.token_count = frame.requested_token_count;
			memset(reply.token_ids,0,sizeof(reply.token_ids));
			for (index=0u; index<reply.token_count; index++)
				reply.token_ids[index] = (frame.anchor_token + 1u + index) % PROBE_VOCAB;
			if ( SparkSpeculationRelayEncode(&reply,PROBE_VOCAB,bytes) == SPARK_STATUS_OK )
				(void)sendto(fd,bytes,sizeof(bytes),0,(const struct sockaddr *)&from,from_length);
			requests++;
			continue;
		}
		if ( SparkSpeculationTapAssemblerAccept(&assembler,buffer,(uint32_t)length,&complete) != SPARK_STATUS_OK )
			foreign++;
	}
	printf("RELAY-PROBE-DRAFTD requests=%llu tap_records=%llu abandoned=%llu rejected=%llu foreign=%llu\n",(unsigned long long)requests,(unsigned long long)assembler.completed,(unsigned long long)assembler.abandoned,(unsigned long long)assembler.rejected,(unsigned long long)foreign);
	close(fd);
	free(payload);
	return(0);
}

static int RunEngine(char **argv)
{
	static SparkSpeculationRelayRemote remote;
	SparkSpeculationTapSet set;
	SparkSpeculationRelayLink link;
	SparkSpeculationTapRecord record;
	SparkSpeculationPolicyDraftResult result;
	struct timespec pace;
	uint64_t *send_ns,start,position = 0u,fingerprint;
	uint32_t rounds,rows,await_us,pace_us,depth,round,row,token;
	uint8_t *payload;
	if ( SparkSpeculationTapSetParse(argv[4],Arg(argv[5]),Arg(argv[6]),Arg(argv[7]),&set) != SPARK_STATUS_OK )
		return(Usage());
	rounds = Arg(argv[8]);
	rows = Arg(argv[9]);
	await_us = Arg(argv[10]);
	pace_us = Arg(argv[11]);
	depth = Arg(argv[12]);
	if ( rounds == 0u || await_us == 0u || depth == 0u || depth > SPARK_SPECULATION_RELAY_MAX_TOKENS )
		return(Usage());
	payload = (uint8_t *)malloc(set.record_bytes);
	send_ns = (uint64_t *)calloc(rounds,sizeof(uint64_t));
	if ( payload == 0 || send_ns == 0 || SparkSpeculationRelayLinkOpen(&link,argv[2],argv[3]) != SPARK_STATUS_OK ||
		SparkSpeculationRelayRemoteInitialize(&remote,&link,PROBE_GENERATION,PROBE_VOCAB,(uint64_t)await_us * 1000u,0u) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"engine cannot open %s>%s\n",argv[2],argv[3]);
		return(1);
	}
	memset(payload,0x5a,set.record_bytes);
	fingerprint = SparkSpeculationTapSetFingerprint(&set);
	pace.tv_sec = pace_us / 1000000u;
	pace.tv_nsec = (long)(pace_us % 1000000u) * 1000L;
	for (round=0u; round<rounds; round++)
	{
		start = SparkSpeculationRelayNowNs();
		for (row=0u; row<rows; row++)
		{
			memset(&record,0,sizeof(record));
			record.engine_generation = PROBE_GENERATION;
			record.sequence_id = PROBE_SEQUENCE;
			record.position = position;
			record.serial = position + 1u;
			record.token_id = (uint32_t)(position % PROBE_VOCAB);
			record.next_token_id = (uint32_t)((position + 1u) % PROBE_VOCAB);
			record.flags = SPARK_SPECULATION_TAP_FLAG_DECODE;
			(void)SparkSpeculationRelayLinkSendTap(&link,fingerprint,&record,payload,set.record_bytes,SPARK_SPECULATION_TAP_FRAGMENT_PAYLOAD_MAX);
			token = (uint32_t)(position % PROBE_VOCAB);
			(void)SparkSpeculationRelayRemoteObserve(&remote,PROBE_SEQUENCE,position,&token,1u);
			position++;
		}
		send_ns[round] = SparkSpeculationRelayNowNs() - start;
		if ( rows == 0u )
			position++;
		token = (uint32_t)(position % PROBE_VOCAB);
		(void)SparkSpeculationRelayRemoteObserve(&remote,PROBE_SEQUENCE,position,&token,1u);
		(void)SparkSpeculationRelayRemoteRound(&remote,PROBE_SEQUENCE,position,depth,&result);
		if ( pace_us != 0u )
			nanosleep(&pace,0);
	}
	qsort(send_ns,rounds,sizeof(uint64_t),Compare64);
	printf("RELAY-PROBE-ENGINE rounds=%u rows=%u taps=%s record_bytes=%u tap_bytes_per_round=%llu send_us_p50=%.1f send_us_p99=%.1f requests=%llu in_time=%llu misses=%llu late=%llu stale=%llu unsent=%llu send_errors=%llu tap_unsent=%llu rtt_p50_us=%llu rtt_p99_us=%llu rtt_max_us=%.1f await_us=%u pace_us=%u depth=%u\n",
		rounds,rows,argv[4],set.record_bytes,(unsigned long long)rows * set.record_bytes,
		(double)send_ns[rounds / 2u] / 1000.0,(double)send_ns[(uint64_t)rounds * 99u / 100u] / 1000.0,
		(unsigned long long)remote.requests,(unsigned long long)remote.answered_in_time,(unsigned long long)remote.deadline_misses,(unsigned long long)remote.answered_late,(unsigned long long)remote.mailbox.stale,
		(unsigned long long)remote.unsent,(unsigned long long)link.send_errors,(unsigned long long)link.tap_records_unsent,
		(unsigned long long)SparkSpeculationRelayRemoteRttPercentileNs(&remote,500u) / 1000u,(unsigned long long)SparkSpeculationRelayRemoteRttPercentileNs(&remote,990u) / 1000u,
		(double)remote.rtt_max_ns / 1000.0,await_us,pace_us,depth);
	SparkSpeculationRelayLinkClose(&link);
	free(payload);
	free(send_ns);
	return(0);
}

int main(int argc,char **argv)
{
	if ( argc == 8 && strcmp(argv[1],"draftd") == 0 )
		return(RunDraftd(argv));
	if ( argc == 13 && strcmp(argv[1],"engine") == 0 )
		return(RunEngine(argv));
	return(Usage());
}
