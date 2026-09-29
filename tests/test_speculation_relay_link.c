#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "sparkpipe/spark_speculation_relay_link.h"

#define TEST_VOCAB 5000u
#define TEST_SEQUENCE 41u
#define TEST_PROMPT 24u
#define TEST_GENERATE 240u
#define TEST_DEPTH 5u
#define TEST_LATE_EVERY 5u
#define TEST_LATE_NS 6000000u
#define TEST_AWAIT_NS 2000000u
#define TEST_GENERATION 0x5eedu

static void Require(int condition,const char *what)
{
	if ( condition )
		return;
	fprintf(stderr,"FAIL %s\n",what);
	exit(1);
}

static uint32_t Target(uint64_t position)
{
	return((uint32_t)((position * 2654435761u + 17u) % TEST_VOCAB));
}

static void RowFill(uint8_t *payload,uint32_t bytes,uint64_t position)
{
	uint32_t index;
	for (index=0u; index<bytes; index++)
		payload[index] = (uint8_t)(position * 13u + index * 5u);
}

typedef struct Draftd
{
	int socket_fd;
	volatile int stop;
	SparkSpeculationTapSet set;
	uint64_t requests;
	uint64_t tap_records;
	uint64_t tap_bad;
	uint64_t tap_gaps;
	uint64_t tap_next_position;
	uint64_t token_mismatch;
	uint32_t good_tokens;
} Draftd;

static void SleepNs(uint64_t ns)
{
	struct timespec wait;
	wait.tv_sec = (time_t)(ns / 1000000000u);
	wait.tv_nsec = (long)(ns % 1000000000u);
	nanosleep(&wait,0);
}

static void *DraftdMain(void *context)
{
	Draftd *draftd = (Draftd *)context;
	SparkSpeculationTapAssembler assembler;
	SparkSpeculationRelayFrame frame,reply;
	struct sockaddr_in from;
	socklen_t from_length;
	uint8_t buffer[SPARK_SPECULATION_TAP_FRAGMENT_BYTES_MAX],expected_payload[8192u * 2u],bytes[SPARK_SPECULATION_RELAY_FRAME_BYTES];
	uint8_t *payload = (uint8_t *)malloc(draftd->set.record_bytes);
	uint32_t complete,index;
	ssize_t length;
	Require(payload != 0 && draftd->set.record_bytes <= sizeof(expected_payload),"draftd buffers");
	Require(SparkSpeculationTapAssemblerInitialize(&assembler,SparkSpeculationTapSetFingerprint(&draftd->set),draftd->set.record_bytes,payload) == SPARK_STATUS_OK,"draftd assembler");
	while ( draftd->stop == 0 )
	{
		from_length = sizeof(from);
		length = recvfrom(draftd->socket_fd,buffer,sizeof(buffer),MSG_DONTWAIT,(struct sockaddr *)&from,&from_length);
		if ( length <= 0 )
			continue;
		if ( (uint32_t)length == SPARK_SPECULATION_RELAY_FRAME_BYTES && SparkSpeculationRelayDecode(buffer,(uint32_t)length,TEST_VOCAB,&frame) == SPARK_STATUS_OK )
		{
			draftd->requests++;
			if ( frame.anchor_token != Target(frame.anchor_position) )
				draftd->token_mismatch++;
			if ( draftd->requests % TEST_LATE_EVERY == 0u )
				SleepNs(TEST_LATE_NS);
			reply = frame;
			reply.kind = SPARK_SPECULATION_RELAY_KIND_DRAFT;
			reply.token_count = frame.requested_token_count;
			memset(reply.token_ids,0,sizeof(reply.token_ids));
			for (index=0u; index<reply.token_count; index++)
				reply.token_ids[index] = index < draftd->good_tokens ? Target(frame.anchor_position + 1u + index) : (Target(frame.anchor_position + 1u + index) + 1u) % TEST_VOCAB;
			Require(SparkSpeculationRelayEncode(&reply,TEST_VOCAB,bytes) == SPARK_STATUS_OK,"draftd encodes");
			Require(sendto(draftd->socket_fd,bytes,sizeof(bytes),0,(const struct sockaddr *)&from,from_length) == (ssize_t)sizeof(bytes),"draftd replies");
			continue;
		}
		if ( SparkSpeculationTapAssemblerAccept(&assembler,buffer,(uint32_t)length,&complete) != SPARK_STATUS_OK )
		{
			draftd->tap_bad++;
			continue;
		}
		if ( complete == 0u )
			continue;
		RowFill(expected_payload,draftd->set.record_bytes,assembler.record.position);
		if ( assembler.record.sequence_id != TEST_SEQUENCE || assembler.record.token_id != Target(assembler.record.position) || assembler.record.next_token_id != Target(assembler.record.position + 1u) || memcmp(payload,expected_payload,draftd->set.record_bytes) != 0 )
			draftd->tap_bad++;
		if ( assembler.record.position != draftd->tap_next_position )
			draftd->tap_gaps++;
		draftd->tap_next_position = assembler.record.position + 1u;
		draftd->tap_records++;
	}
	free(payload);
	return(0);
}

static void StartDraftd(Draftd *draftd,char *endpoint,uint32_t good_tokens)
{
	struct sockaddr_in address;
	socklen_t length = sizeof(address);
	int buffer_bytes = 8 << 20;
	memset(draftd,0,sizeof(*draftd));
	draftd->good_tokens = good_tokens;
	Require(SparkSpeculationTapSetParse("mean:1,3",4u,4u,4096u,&draftd->set) == SPARK_STATUS_OK,"draftd tap set");
	draftd->socket_fd = socket(AF_INET,SOCK_DGRAM,0);
	Require(draftd->socket_fd >= 0,"draftd socket");
	(void)setsockopt(draftd->socket_fd,SOL_SOCKET,SO_RCVBUF,&buffer_bytes,sizeof(buffer_bytes));
	memset(&address,0,sizeof(address));
	address.sin_family = AF_INET;
	address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
	Require(bind(draftd->socket_fd,(const struct sockaddr *)&address,sizeof(address)) == 0 && getsockname(draftd->socket_fd,(struct sockaddr *)&address,&length) == 0,"draftd binds");
	snprintf(endpoint,64u,"127.0.0.1:%u",(unsigned)ntohs(address.sin_port));
}

typedef struct Outcome
{
	uint32_t stream[TEST_PROMPT + TEST_GENERATE + 16u];
	uint32_t produced;
	uint64_t rounds;
	uint64_t zero_rounds;
	uint64_t accepted;
} Outcome;

static void SendTaps(SparkSpeculationRelayLink *link,const SparkSpeculationTapSet *set,uint64_t first,uint64_t count,uint32_t flags)
{
	SparkSpeculationTapRecord record;
	uint8_t payload[8192u * 2u];
	uint64_t position;
	for (position=first; position<first+count; position++)
	{
		memset(&record,0,sizeof(record));
		record.engine_generation = TEST_GENERATION;
		record.sequence_id = TEST_SEQUENCE;
		record.position = position;
		record.serial = position;
		record.token_id = Target(position);
		record.next_token_id = Target(position + 1u);
		record.flags = flags;
		RowFill(payload,set->record_bytes,position);
		Require(SparkSpeculationRelayLinkSendTap(link,SparkSpeculationTapSetFingerprint(set),&record,payload,set->record_bytes,SPARK_SPECULATION_TAP_FRAGMENT_PAYLOAD_MAX) == SPARK_STATUS_OK,"taps sent");
	}
}

static void RunGreedy(uint32_t shadow,uint64_t await_ns,uint64_t pace_ns,uint32_t good_tokens,Outcome *outcome,Draftd *draftd_out,SparkSpeculationRelayRemote *remote)
{
	SparkSpeculationRelayLink link;
	SparkSpeculationPolicyDraftResult result;
	Draftd draftd;
	pthread_t thread;
	char endpoint[64];
	uint32_t tokens[TEST_PROMPT],index,accepted,committed;
	uint64_t anchor;
	StartDraftd(&draftd,endpoint,good_tokens);
	Require(SparkSpeculationRelayLinkOpen(&link,"127.0.0.1:0",endpoint) == SPARK_STATUS_OK,"engine link opens");
	Require(pthread_create(&thread,0,DraftdMain,&draftd) == 0,"draftd thread");
	Require(SparkSpeculationRelayRemoteInitialize(remote,&link,TEST_GENERATION,TEST_VOCAB,await_ns,shadow) == SPARK_STATUS_OK,"remote");
	memset(outcome,0,sizeof(*outcome));
	for (index=0u; index<TEST_PROMPT; index++)
		tokens[index] = Target(index);
	SendTaps(&link,&draftd.set,0u,TEST_PROMPT - 1u,SPARK_SPECULATION_TAP_FLAG_PREFILL);
	Require(SparkSpeculationRelayRemoteObserve(remote,TEST_SEQUENCE,0u,tokens,TEST_PROMPT) == SPARK_STATUS_OK,"prompt observed");
	memcpy(outcome->stream,tokens,sizeof(tokens));
	outcome->produced = TEST_PROMPT;
	while ( outcome->produced < TEST_PROMPT + TEST_GENERATE )
	{
		anchor = outcome->produced - 1u;
		Require(SparkSpeculationRelayRemoteRound(remote,TEST_SEQUENCE,anchor,TEST_DEPTH,&result) == SPARK_STATUS_OK,"round never fails a request");
		Require(shadow == 0u || result.token_count == 0u,"shadow never changes the round shape");
		accepted = 0u;
		while ( accepted < result.token_count && result.token_ids[accepted] == Target(anchor + 1u + accepted) )
			accepted++;
		committed = accepted + 1u;
		if ( outcome->produced + committed > TEST_PROMPT + TEST_GENERATE )
			committed = TEST_PROMPT + TEST_GENERATE - outcome->produced;
		for (index=0u; index<committed; index++)
			outcome->stream[outcome->produced + index] = Target(anchor + 1u + index);
		SendTaps(&link,&draftd.set,anchor,committed,result.token_count != 0u ? SPARK_SPECULATION_TAP_FLAG_VERIFY : SPARK_SPECULATION_TAP_FLAG_DECODE);
		Require(SparkSpeculationRelayRemoteObserve(remote,TEST_SEQUENCE,anchor + 1u,outcome->stream + outcome->produced,committed) == SPARK_STATUS_OK,"committed tokens observed");
		outcome->produced += committed;
		outcome->rounds++;
		outcome->zero_rounds += result.token_count == 0u ? 1u : 0u;
		outcome->accepted += accepted;
		if ( pace_ns != 0u )
			SleepNs(pace_ns);
	}
	SleepNs(3u * TEST_LATE_NS);
	Require(SparkSpeculationRelayRemotePoll(remote) == SPARK_STATUS_OK,"final poll");
	draftd.stop = 1;
	Require(pthread_join(thread,0) == 0,"draftd joins");
	close(draftd.socket_fd);
	SparkSpeculationRelayLinkClose(&link);
	remote->link = 0;
	*draftd_out = draftd;
}

static void TestDeadlineRounds(void)
{
	static SparkSpeculationRelayRemote remote;
	Outcome outcome;
	Draftd draftd;
	uint32_t index;
	RunGreedy(0u,TEST_AWAIT_NS,0u,3u,&outcome,&draftd,&remote);
	for (index=0u; index<outcome.produced; index++)
		Require(outcome.stream[index] == Target(index),"served stream equals the target stream with a remote drafter and late drafts");
	Require(remote.requests == outcome.rounds && draftd.requests == outcome.rounds,"one request per round reaches draftd");
	Require(draftd.token_mismatch == 0u,"requests carry the committed anchor token");
	Require(remote.deadline_misses >= 1u && remote.deadline_misses == outcome.zero_rounds,"every late draft is a counted zero-draft round");
	Require(remote.answered_in_time + remote.deadline_misses == outcome.rounds,"each round is answered in time or missed");
	Require(remote.answered_late + remote.mailbox.stale >= 1u,"late drafts are counted when they land, as late for the open round or stale for a closed one");
	Require(outcome.accepted == 3u * remote.answered_in_time,"in-time drafts commit their correct prefix");
	Require(remote.rtt_count == remote.answered_in_time && SparkSpeculationRelayRemoteRttPercentileNs(&remote,500u) > 0u && SparkSpeculationRelayRemoteRttPercentileNs(&remote,990u) <= TEST_AWAIT_NS + SPARK_SPECULATION_RELAY_RTT_BUCKET_NS,"round trip percentiles are recorded for in-time drafts");
	Require(draftd.tap_bad == 0u && draftd.tap_gaps == 0u && draftd.tap_records == TEST_PROMPT + TEST_GENERATE - 1u,"every committed position reaches draftd once, in order, with intact rows");
	printf("deadline: rounds=%llu zero=%llu in_time=%llu late=%llu stale=%llu p50=%lluus p99=%lluus\n",(unsigned long long)outcome.rounds,(unsigned long long)outcome.zero_rounds,(unsigned long long)remote.answered_in_time,(unsigned long long)remote.answered_late,(unsigned long long)remote.mailbox.stale,
		(unsigned long long)SparkSpeculationRelayRemoteRttPercentileNs(&remote,500u) / 1000u,(unsigned long long)SparkSpeculationRelayRemoteRttPercentileNs(&remote,990u) / 1000u);
}

static void TestShadow(void)
{
	static SparkSpeculationRelayRemote remote;
	Outcome outcome;
	Draftd draftd;
	uint32_t index;
	RunGreedy(1u,0u,1000000u,2u,&outcome,&draftd,&remote);
	for (index=0u; index<outcome.produced; index++)
		Require(outcome.stream[index] == Target(index),"shadow mode serves the plain stream");
	Require(outcome.zero_rounds == outcome.rounds && outcome.rounds == TEST_GENERATE,"shadow rounds are plain decode steps");
	Require(remote.shadow_rounds + remote.shadow_evicted + remote.deadline_misses >= remote.requests - 1u,"every answered shadow draft is scored or evicted");
	Require(remote.shadow_rounds >= TEST_GENERATE / 2u && remote.shadow_accepted == 2u * remote.shadow_rounds && remote.shadow_accepted_at[2] == remote.shadow_rounds,"shadow scoring counts the correct prefix of each draft");
	Require(remote.shadow_proposed == (uint64_t)TEST_DEPTH * remote.shadow_rounds,"shadow scoring counts proposed tokens");
	printf("shadow: rounds=%llu scored=%llu accepted=%llu evicted=%llu late=%llu\n",(unsigned long long)outcome.rounds,(unsigned long long)remote.shadow_rounds,(unsigned long long)remote.shadow_accepted,(unsigned long long)remote.shadow_evicted,(unsigned long long)remote.answered_late);
}

static void TestUnsynced(void)
{
	SparkSpeculationRelayLink link;
	static SparkSpeculationRelayRemote remote;
	SparkSpeculationPolicyDraftResult result;
	uint32_t tokens[3] = {1u,2u,3u};
	Require(SparkSpeculationRelayLinkOpen(&link,"127.0.0.1:0","127.0.0.1:9") == SPARK_STATUS_OK,"link to a closed port opens");
	Require(SparkSpeculationRelayRemoteInitialize(&remote,&link,1u,TEST_VOCAB,0u,0u) == SPARK_STATUS_INVALID_ARGUMENT,"a serving drafter needs a deadline");
	Require(SparkSpeculationRelayRemoteInitialize(&remote,&link,1u,TEST_VOCAB,100000u,0u) == SPARK_STATUS_OK,"remote");
	Require(SparkSpeculationRelayRemoteRound(&remote,5u,2u,4u,&result) == SPARK_STATUS_OK && result.token_count == 0u && remote.unsynced == 1u,"a round before the anchor is observed is a counted zero-draft round");
	Require(SparkSpeculationRelayRemoteObserve(&remote,5u,0u,tokens,3u) == SPARK_STATUS_OK,"observe");
	Require(SparkSpeculationRelayRemoteRound(&remote,5u,2u,17u,&result) == SPARK_STATUS_CAPACITY_EXCEEDED,"depth past the frame is refused, not trimmed");
	Require(SparkSpeculationRelayRemoteRound(&remote,5u,2u,4u,&result) == SPARK_STATUS_OK && result.token_count == 0u && remote.deadline_misses == 1u,"a silent peer is a counted zero-draft round");
	Require(SparkSpeculationRelayRemoteRound(&remote,5u,2u,4u,&result) == SPARK_STATUS_OK && result.token_count == 0u,"a refused port never fails the round");
	SparkSpeculationRelayLinkClose(&link);
	Require(SparkSpeculationRelayLinkOpen(&link,"127.0.0.1:0","127.0.0.1:0") == SPARK_STATUS_PARSE_ERROR && link.socket_fd < 0,"a peer needs a port");
	Require(SparkSpeculationRelayLinkOpen(&link,"localhost:1","127.0.0.1:9") == SPARK_STATUS_PARSE_ERROR && link.socket_fd < 0,"endpoints are numeric");
}

int main(void)
{
	TestUnsynced();
	TestDeadlineRounds();
	TestShadow();
	printf("test_speculation_relay_link: ok\n");
	return(0);
}
