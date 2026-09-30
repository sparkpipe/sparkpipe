#pragma once

#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

static void SparkGlm52RunChain(SparkGlm52TpChain *chain);

static const char *const SparkGlm52VerifyDrafterNames[6] = {"none","lookup","oracle","adversary","recorded","relay"};

static SparkStatus SparkGlm52RelayIo(SparkGlm52ModuleState *state,void *bytes,size_t count,uint32_t sending)
{
	uint8_t *cursor = (uint8_t *)bytes;
	ssize_t moved;
	while ( count != 0u )
	{
		moved = sending != 0u ? send(state->relay_socket,cursor,count,MSG_NOSIGNAL) : recv(state->relay_socket,cursor,count,0);
		if ( moved <= 0 )
		{
			fprintf(stderr,"GLM52-RELAY %s failed rank=%u; the relay drafter is closed\n",sending != 0u ? "send" : "receive",state->tp_rank);
			close(state->relay_socket);
			state->relay_connected = 0u;
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		}
		cursor += moved;
		count -= (size_t)moved;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52RelayConnect(SparkGlm52ModuleState *state,const char *endpoint)
{
	char host[SPARK_GLM52_RELAY_ENDPOINT_BYTES];
	const char *colon = strrchr(endpoint,':');
	struct addrinfo hints,*found = 0;
	struct timeval timeout;
	int fd,flag = 1;
	if ( colon == 0 || colon == endpoint || (size_t)(colon - endpoint) >= sizeof(host) || colon[1] == '\0' )
	{
		fprintf(stderr,"GLM52-RELAY endpoint %s must be HOST:PORT\n",endpoint);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	memcpy(host,endpoint,(size_t)(colon - endpoint));
	host[colon - endpoint] = '\0';
	memset(&hints,0,sizeof(hints));
	hints.ai_family = AF_INET;
	hints.ai_socktype = SOCK_STREAM;
	if ( getaddrinfo(host,colon + 1,&hints,&found) != 0 || found == 0 )
	{
		fprintf(stderr,"GLM52-RELAY cannot resolve %s\n",endpoint);
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	}
	fd = socket(found->ai_family,found->ai_socktype,found->ai_protocol);
	timeout.tv_sec = SPARK_GLM52_RELAY_TIMEOUT_SECONDS;
	timeout.tv_usec = 0;
	if ( fd < 0 || setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout)) != 0 || setsockopt(fd,SOL_SOCKET,SO_SNDTIMEO,&timeout,sizeof(timeout)) != 0 ||
		setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&flag,sizeof(flag)) != 0 || connect(fd,found->ai_addr,found->ai_addrlen) != 0 )
	{
		fprintf(stderr,"GLM52-RELAY cannot connect to %s rank=%u\n",endpoint,state->tp_rank);
		if ( fd >= 0 )
			close(fd);
		freeaddrinfo(found);
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	freeaddrinfo(found);
	state->relay_socket = fd;
	state->relay_connected = 1u;
	fprintf(stderr,"GLM52-RELAY connected %s rank=%u\n",endpoint,state->tp_rank);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52RelayCommit(SparkGlm52ModuleState *state,uint64_t sequence_id,uint32_t position,uint32_t input_token,uint32_t output_token,const uint16_t *hidden)
{
	uint8_t header[24];
	uint32_t words[2] = {input_token,output_token},magic = SPARK_GLM52_RELAY_COMMIT_MAGIC,count = 1u;
	SparkStatus status;
	if ( state->relay_connected == 0u )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	memcpy(header,&magic,4u);
	memcpy(header + 4u,&state->tp_rank,4u);
	memcpy(header + 8u,&sequence_id,8u);
	memcpy(header + 16u,&position,4u);
	memcpy(header + 20u,&count,4u);
	status = SparkGlm52RelayIo(state,header,sizeof(header),1u);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52RelayIo(state,words,sizeof(words),1u);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52RelayIo(state,(void *)hidden,SPARK_GLM52_MODEL_HIDDEN_DIMENSION * sizeof(uint16_t),1u);
	if ( status == SPARK_STATUS_OK )
		state->relay_rows++;
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm52RelayDraftTokens(void *context,const SparkSpeculationPolicyDraftRequest *request,SparkSpeculationPolicyDraftResult *result)
{
	SparkGlm52ModuleState *state = (SparkGlm52ModuleState *)context;
	uint8_t header[32];
	uint32_t magic = SPARK_GLM52_RELAY_DRAFT_MAGIC,position = (uint32_t)request->sequence_position,count = request->requested_token_count,zero = 0u,answer[2];
	uint64_t begin = SparkGlm52NowNs();
	SparkStatus status;
	if ( state->relay_connected == 0u )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	memcpy(header,&magic,4u);
	memcpy(header + 4u,&state->tp_rank,4u);
	memcpy(header + 8u,&request->sequence_id,8u);
	memcpy(header + 16u,&position,4u);
	memcpy(header + 20u,&count,4u);
	memcpy(header + 24u,&state->relay_anchor,4u);
	memcpy(header + 28u,&zero,4u);
	status = SparkGlm52RelayIo(state,header,sizeof(header),1u);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52RelayIo(state,answer,sizeof(answer),0u);
	if ( status == SPARK_STATUS_OK && (answer[0] != SPARK_GLM52_RELAY_ANSWER_MAGIC || answer[1] > count) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK && answer[1] != 0u )
		status = SparkGlm52RelayIo(state,result->token_ids,(size_t)answer[1] * sizeof(uint32_t),0u);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	result->token_count = answer[1];
	state->relay_drafts++;
	state->relay_wait_ns += SparkGlm52NowNs() - begin;
	return(answer[1] != 0u ? SPARK_STATUS_OK : SPARK_STATUS_NOT_FOUND);
}

static uint32_t SparkGlm52TapActive(const SparkGlm52ModuleState *state)
{
	return(state->tap_file != 0 || (state->relay_connected != 0u && state->tp_rank == 0u) ? 1u : 0u);
}

static uint64_t SparkGlm52TapSequence(const SparkGlm52ModuleState *state,uint32_t slot_index,uint32_t resident)
{
	const SparkGlm52AsyncCompletion *async = &state->completions[slot_index];
	uint32_t lane;
	for (lane=0u; lane<async->lane_count; lane++)
		if ( async->lane_indices[lane] == resident )
			return(async->lane_sequence_ids[lane]);
	return(0u);
}

static SparkStatus SparkGlm52TapConfigure(SparkGlm52ModuleState *state)
{
	const char *path = getenv(SPARK_GLM52_TAP_DUMP_ENV);
	if ( path == 0 || path[0] == '\0' || state->tp_rank != 0u )
		return(SPARK_STATUS_OK);
	if ( state->owns_final_head == 0u )
	{
		fprintf(stderr,"GLM52-TAP-REFUSED %s needs the final head on rank 0\n",SPARK_GLM52_TAP_DUMP_ENV);
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	}
	state->tap_file = fopen(path,"wb");
	if ( state->tap_file == 0 )
	{
		fprintf(stderr,"GLM52-TAP %s cannot be opened\n",path);
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	fprintf(stderr,"GLM52-TAP dump=%s rank=%u row_bytes=%u\n",path,state->tp_rank,(unsigned)(SPARK_GLM52_MODEL_HIDDEN_DIMENSION * sizeof(uint16_t)));
	return(SPARK_STATUS_OK);
}

static void SparkGlm52TapClose(SparkGlm52ModuleState *state,const char *reason)
{
	if ( state->tap_file != 0 )
	{
		fprintf(stderr,"GLM52-TAP closed records=%llu reason=%s\n",(unsigned long long)state->tap_records,reason);
		fclose(state->tap_file);
	}
	state->tap_file = 0;
	free(state->tap_rows);
	state->tap_rows = 0;
	state->tap_capacity = 0u;
}

static void SparkGlm52TapWave(SparkGlm52TpChain *chain)
{
	SparkGlm52ModuleState *state = chain->state;
	uint32_t rows = chain->first_row + chain->wave_rows;
	uint16_t *grown;
	if ( SparkGlm52TapActive(state) == 0u )
		return;
	if ( rows > state->tap_capacity )
	{
		grown = (uint16_t *)realloc(state->tap_rows,(size_t)rows * SPARK_GLM52_MODEL_HIDDEN_DIMENSION * sizeof(uint16_t));
		if ( grown == 0 )
		{
			SparkGlm52TapClose(state,"allocation");
			return;
		}
		state->tap_rows = grown;
		state->tap_capacity = rows;
	}
	if ( cudaMemcpyAsync(state->tap_rows + (size_t)chain->first_row * SPARK_GLM52_MODEL_HIDDEN_DIMENSION,chain->slot->normed_bf16,
		(size_t)chain->wave_rows * SPARK_GLM52_MODEL_HIDDEN_DIMENSION * sizeof(uint16_t),cudaMemcpyDeviceToHost,(cudaStream_t)chain->slot->stream) != cudaSuccess )
		SparkGlm52TapClose(state,"copy");
}

static SparkStatus SparkGlm52TapEmit(SparkGlm52TpChain *chain,uint64_t sequence_id,uint32_t row)
{
	SparkGlm52ModuleState *state = chain->state;
	SparkGlm52ExecutionSlot *slot = chain->slot;
	const uint16_t *hidden = state->tap_rows + (size_t)row * SPARK_GLM52_MODEL_HIDDEN_DIMENSION;
	uint32_t header[5];
	if ( row >= state->tap_capacity )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	if ( state->tap_file != 0 )
	{
		header[0] = SPARK_GLM52_TAP_RECORD_MAGIC;
		header[1] = slot->host_resident_slots[row];
		header[2] = slot->host_positions[row];
		header[3] = slot->host_token_ids[row];
		header[4] = slot->host_output_token_ids[row];
		if ( fwrite(header,sizeof(header),1u,state->tap_file) != 1u || fwrite(hidden,SPARK_GLM52_MODEL_HIDDEN_DIMENSION * sizeof(uint16_t),1u,state->tap_file) != 1u )
		{
			SparkGlm52TapClose(state,"write");
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		}
		state->tap_records++;
	}
	if ( state->relay_connected != 0u && state->tp_rank == 0u )
		return(SparkGlm52RelayCommit(state,sequence_id,slot->host_positions[row],slot->host_token_ids[row],slot->host_output_token_ids[row],hidden));
	return(SPARK_STATUS_OK);
}

static void SparkGlm52TapWrite(SparkGlm52TpChain *chain,SparkStatus status)
{
	SparkGlm52ModuleState *state = chain->state;
	uint32_t row,rows = chain->batch->row_count;
	if ( SparkGlm52TapActive(state) == 0u || chain->steps_budget != 0u || status != SPARK_STATUS_OK )
		return;
	for (row=0u; row<rows; row++)
		if ( SparkGlm52TapEmit(chain,SparkGlm52TapSequence(state,chain->slot_index,chain->slot->host_resident_slots[row]),row) != SPARK_STATUS_OK )
			return;
	if ( state->tap_file != 0 )
		fflush(state->tap_file);
}

static SparkStatus SparkGlm52VerifyReadFile(const char *path,uint8_t **bytes_out,uint64_t *count_out)
{
	FILE *file;
	long bytes;
	uint8_t *buffer;
	*bytes_out = 0;
	*count_out = 0u;
	file = fopen(path,"rb");
	if ( file == 0 )
	{
		fprintf(stderr,"GLM52-VERIFY drafter file %s cannot be opened\n",path);
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	}
	bytes = fseek(file,0,SEEK_END) == 0 ? ftell(file) : -1;
	if ( bytes <= 0 || fseek(file,0,SEEK_SET) != 0 )
	{
		fclose(file);
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	buffer = (uint8_t *)malloc((size_t)bytes);
	if ( buffer == 0 || fread(buffer,1u,(size_t)bytes,file) != (size_t)bytes )
	{
		free(buffer);
		fclose(file);
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	fclose(file);
	*bytes_out = buffer;
	*count_out = (uint64_t)bytes;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52VerifyAllocateDrafter(SparkGlm52ModuleState *state)
{
	SparkStatus status;
	uint64_t bytes;
	uint32_t count;
	if ( state->verify_drafter == SPARK_SPECULATION_VERIFY_DRAFTER_LOOKUP )
	{
		status = SparkSpeculationLookupDraftInitialize(&state->verify_lookup,state->resident_sequence_capacity,state->max_sequence_positions,SPARK_GLM52_VERIFY_LOOKUP_MIN_MATCH,SPARK_GLM52_VERIFY_LOOKUP_MAX_MATCH);
		state->verify_draft_function = SparkSpeculationLookupDraftTokens;
		state->verify_draft_context = &state->verify_lookup;
		SPARK_RETURN(status);
	}
	if ( state->verify_drafter == SPARK_SPECULATION_VERIFY_DRAFTER_RELAY )
	{
		status = SparkGlm52RelayConnect(state,state->verify_drafter_path);
		state->verify_draft_function = SparkGlm52RelayDraftTokens;
		state->verify_draft_context = state;
		SPARK_RETURN(status);
	}
	status = SparkGlm52VerifyReadFile(state->verify_drafter_path,&state->verify_drafter_bytes,&bytes);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( state->verify_drafter == SPARK_SPECULATION_VERIFY_DRAFTER_RECORDED )
	{
		status = SparkSpeculationRecordedDraftInitialize(&state->verify_recorded,state->verify_drafter_bytes,bytes,SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT);
		if ( status == SPARK_STATUS_OK )
			fprintf(stderr,"GLM52-VERIFY recorded drafter %s entries=%llu depth=%u\n",state->verify_drafter_path,(unsigned long long)state->verify_recorded.entry_count,state->verify_recorded.depth);
		state->verify_draft_function = SparkSpeculationRecordedDraftTokens;
		state->verify_draft_context = &state->verify_recorded;
		SPARK_RETURN(status);
	}
	if ( bytes % sizeof(uint32_t) != 0u || bytes / sizeof(uint32_t) < 2u || bytes / sizeof(uint32_t) > state->max_sequence_positions )
	{
		fprintf(stderr,"GLM52-VERIFY reference drafter %s must hold 2..%u little-endian uint32 token ids from position 0\n",state->verify_drafter_path,state->max_sequence_positions);
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	count = (uint32_t)(bytes / sizeof(uint32_t));
	status = SparkSpeculationReferenceDraftInitialize(&state->verify_reference,
		state->verify_drafter == SPARK_SPECULATION_VERIFY_DRAFTER_ORACLE ? SPARK_SPECULATION_REFERENCE_ORACLE : SPARK_SPECULATION_REFERENCE_ADVERSARY,
		SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT,0u,(const uint32_t *)state->verify_drafter_bytes,count);
	state->verify_draft_function = SparkSpeculationReferenceDraftTokens;
	state->verify_draft_context = &state->verify_reference;
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm52VerifyConfigure(SparkGlm52ModuleState *state)
{
	const char *reason = 0;
	uint32_t limit;
	SparkStatus status;
	limit = SparkGlm52ExactWaveRows() < SPARK_GLM52_VERIFY_ROWS_LIMIT ? SparkGlm52ExactWaveRows() : SPARK_GLM52_VERIFY_ROWS_LIMIT;
	if ( limit > state->execution_row_capacity )
		limit = state->execution_row_capacity;
	if ( SparkSpeculationVerifyRowsParse(getenv(SPARK_GLM52_VERIFY_ROWS_ENV),limit,&state->verify_rows_max) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s must be 0 or 2..%u (the exact multi-row wave limit and the execution row capacity)\n",SPARK_GLM52_VERIFY_ROWS_ENV,limit);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	if ( SparkSpeculationVerifyDrafterParse(getenv(SPARK_GLM52_VERIFY_DRAFTER_ENV),state->verify_rows_max,&state->verify_drafter,&state->verify_drafter_path) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s must be lookup, oracle:PATH, adversary:PATH or recorded:PATH when %s is nonzero, and absent when it is 0 or unset\n",SPARK_GLM52_VERIFY_DRAFTER_ENV,SPARK_GLM52_VERIFY_ROWS_ENV);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	status = SparkGlm52TapConfigure(state);
	if ( status != SPARK_STATUS_OK || state->verify_rows_max == 0u )
		SPARK_RETURN(status);
	if ( state->chain_mode == SPARK_TP_CHAIN_MODE_EAGER )
		reason = "needs the linear or graph chain (SPARK_GLM52_CHAIN_MODE)";
	else if ( state->owns_embedding == 0u || state->owns_final_head == 0u )
		reason = "needs every rank to own the embedding and the final head";
	else if ( state->head_certified_fp8_payload == 0 )
		reason = "needs the certified head shadow, the per-row exact head of multi-row waves";
	if ( reason != 0 )
	{
		fprintf(stderr,"GLM52-VERIFY-REFUSED rows=%u reason=%s\n",state->verify_rows_max,reason);
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	}
	status = SparkGlm52VerifyAllocateDrafter(state);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"GLM52-VERIFY drafter %s refused: status=%s\n",SparkGlm52VerifyDrafterNames[state->verify_drafter],SparkStatusToString(status));
		SPARK_RETURN(status);
	}
	fprintf(stderr,"GLM52-VERIFY rows=%u drafter=%s rank=%u\n",state->verify_rows_max,SparkGlm52VerifyDrafterNames[state->verify_drafter],state->tp_rank);
	return(SPARK_STATUS_OK);
}

static void SparkGlm52VerifyRelease(SparkGlm52ModuleState *state)
{
	SparkGlm52TapClose(state,"release");
	if ( state->relay_connected != 0u )
	{
		fprintf(stderr,"GLM52-RELAY closed rank=%u drafts=%llu rows=%llu wait_ms_mean=%.3f\n",state->tp_rank,(unsigned long long)state->relay_drafts,(unsigned long long)state->relay_rows,
			state->relay_drafts != 0u ? (double)state->relay_wait_ns / 1e6 / (double)state->relay_drafts : 0.0);
		close(state->relay_socket);
		state->relay_connected = 0u;
	}
	if ( state->verify_drafter == SPARK_SPECULATION_VERIFY_DRAFTER_LOOKUP )
		SparkSpeculationLookupDraftDestroy(&state->verify_lookup);
	free(state->verify_drafter_bytes);
	state->verify_drafter_bytes = 0;
	state->verify_draft_function = 0;
	state->verify_draft_context = 0;
}

static SparkStatus SparkGlm52VerifyObserveRows(SparkGlm52ModuleState *state,const SparkGlm52ResidentDecodeStageBatchView *batch)
{
	SparkStatus status;
	uint32_t row;
	if ( state->verify_drafter != SPARK_SPECULATION_VERIFY_DRAFTER_LOOKUP || batch->token_ids == 0 )
		return(SPARK_STATUS_OK);
	for (row=0u; row<batch->row_count; row++)
	{
		status = SparkSpeculationLookupDraftObserve(&state->verify_lookup,batch->row_resident_slots[row],batch->row_sequence_ids[row],batch->row_positions[row],&batch->token_ids[row],1u);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52StepsCheck(const SparkGlm52ModuleState *state,const SparkModelDriverFrame *frame,const SparkGlm52ResidentDecodeStageBatchView *batch)
{
	uint32_t steps = frame->tokens_per_sequence,index;
	if ( steps <= 1u )
		return(SPARK_STATUS_OK);
	if ( state->verify_rows_max == 0u || steps > SPARK_MODEL_DRIVER_MAX_TOKENS_PER_SEQUENCE || (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u ||
		batch->token_ids == 0 || batch->row_count != batch->active_sequence_count || frame->buffer_count != 1u || frame->buffers == 0 ||
		frame->buffers[0].bytes < (uint64_t)batch->row_count * steps * sizeof(uint32_t) )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	for (index=0u; index<frame->cache_lane_count; index++)
		if ( (frame->cache_lanes[index].flags & (SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PREFIX | SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PUBLISH)) != 0u )
			SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	return(SPARK_STATUS_OK);
}

static uint32_t SparkGlm52VerifyFit(const SparkGlm52ModuleState *state,uint32_t position,uint32_t rows_max)
{
	SparkGlm52WaveRegimeContext regime;
	uint32_t positions[SPARK_GLM52_VERIFY_ROWS_LIMIT];
	uint32_t row,first;
	for (row=0u; row<rows_max && row<SPARK_GLM52_VERIFY_ROWS_LIMIT; row++)
		positions[row] = position + row;
	regime.positions = positions;
	regime.split_threshold = state->decode_split_context_threshold;
	regime.max_positions = state->max_sequence_positions;
	first = SparkGlm52RowRegime(&regime,0u);
	row = 0u;
	while ( row < rows_max && row < SPARK_GLM52_VERIFY_ROWS_LIMIT && position + row < state->max_sequence_positions && SparkGlm52RowRegime(&regime,row) == first )
		row++;
	return(row);
}

static SparkStatus SparkGlm52StepsDraft(SparkGlm52TpChain *chain,uint32_t *count_out)
{
	SparkGlm52ModuleState *state = chain->state;
	SparkGlm52ExecutionSlot *slot = chain->slot;
	SparkSpeculationPolicyDraftRequest request;
	SparkSpeculationPolicyDraftResult result;
	SparkStatus status;
	uint32_t depth,row;
	*count_out = 0u;
	if ( chain->steps_lanes != 1u || state->verify_draft_function == 0 )
		return(SPARK_STATUS_OK);
	depth = SparkSpeculationVerifyDepth(chain->steps_budget,chain->steps_produced,state->verify_rows_max,SparkGlm52VerifyFit(state,slot->host_positions[0],state->verify_rows_max));
	if ( depth == 0u )
		return(SPARK_STATUS_OK);
	memset(&request,0,sizeof(request));
	memset(&result,0,sizeof(result));
	request.abi_version = SPARK_SPECULATION_ABI_VERSION;
	request.descriptor_bytes = SPARK_SPECULATION_DRAFT_REQUEST_DESCRIPTOR_BYTES;
	request.requested_token_count = depth;
	request.active_sequence_index = chain->steps_lane_slots[0];
	request.request_id = chain->frame->request_id;
	request.sequence_id = chain->steps_sequence_ids[0];
	request.sequence_position = slot->host_positions[0];
	state->relay_anchor = slot->host_token_ids[0];
	status = state->verify_draft_function(state->verify_draft_context,&request,&result);
	if ( status == SPARK_STATUS_NOT_FOUND )
		return(SPARK_STATUS_OK);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( result.token_count > depth )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	for (row=0u; row<result.token_count; row++)
	{
		if ( result.token_ids[row] >= SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT )
			SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
		chain->verify_draft[row] = result.token_ids[row];
		slot->host_token_ids[row + 1u] = result.token_ids[row];
		slot->host_positions[row + 1u] = slot->host_positions[0] + row + 1u;
		slot->host_resident_slots[row + 1u] = chain->steps_lane_slots[0];
	}
	*count_out = result.token_count;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52StepsPlan(SparkGlm52TpChain *chain)
{
	SparkGlm52ExecutionSlot *slot = chain->slot;
	SparkStatus status;
	uint32_t lane,drafts;
	for (lane=0u; lane<chain->steps_lanes; lane++)
	{
		slot->host_token_ids[lane] = chain->steps_produced == 0u ? chain->steps_anchor_tokens[lane] : chain->steps_tokens[lane * chain->steps_budget + chain->steps_produced - 1u];
		slot->host_positions[lane] = (uint32_t)(chain->steps_positions[lane] + chain->steps_produced);
		slot->host_resident_slots[lane] = chain->steps_lane_slots[lane];
	}
	status = SparkGlm52StepsDraft(chain,&drafts);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	chain->verify_draft_count = drafts;
	chain->verify_wave = drafts != 0u ? 1u : 0u;
	chain->batch_copy.row_count = chain->steps_lanes + drafts;
	chain->first_row = 0u;
	chain->wave_rows = SparkGlm52WaveRows(chain,0u);
	chain->next_wave_row = chain->wave_rows;
	chain->graph = 0u;
	if ( chain->wave_rows == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52StepsBegin(SparkGlm52TpChain *chain,const SparkGlm52ResidentDecodeStageBatchView *batch)
{
	SparkGlm52ExecutionSlot *slot = chain->slot;
	uint32_t lane;
	chain->steps_budget = chain->frame->tokens_per_sequence;
	chain->steps_produced = 0u;
	chain->steps_lanes = batch->active_sequence_count;
	for (lane=0u; lane<chain->steps_lanes; lane++)
	{
		chain->steps_anchor_tokens[lane] = slot->host_token_ids[lane];
		chain->steps_positions[lane] = slot->host_positions[lane];
		chain->steps_lane_slots[lane] = slot->host_resident_slots[lane];
		chain->steps_sequence_ids[lane] = batch->row_sequence_ids[lane];
	}
	chain->state->verify_frames++;
	return(SparkGlm52StepsPlan(chain));
}

static SparkStatus SparkGlm52StepsFinish(SparkGlm52TpChain *chain)
{
	SparkGlm52ModuleState *state = chain->state;
	SparkGlm52AsyncCompletion *async = &state->completions[chain->slot_index];
	uint32_t lane,index;
	int length;
	char positions[256];
	if ( async->lane_count != chain->steps_lanes || chain->steps_produced != chain->steps_budget )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	async->steps_token_count = chain->steps_lanes * chain->steps_budget;
	memcpy(async->steps_tokens,chain->steps_tokens,(uint64_t)async->steps_token_count * sizeof(uint32_t));
	async->steps_extra_tokens = chain->steps_budget - 1u;
	for (lane=0u; lane<async->lane_count; lane++)
		async->lane_next_positions[lane] += async->steps_extra_tokens;
	state->verify_tokens += async->steps_token_count;
	if ( state->tp_rank != 0u )
		return(SPARK_STATUS_OK);
	fprintf(stderr,"VERIFY-FRAME lanes=%u position=%llu budget=%u rounds=%u accepted=%u plain=%u | frames=%llu rounds=%llu proposed=%llu accepted=%llu plain=%llu tokens=%llu\n",
		chain->steps_lanes,(unsigned long long)chain->steps_positions[0],chain->steps_budget,chain->verify_rounds,chain->verify_accepted,chain->verify_plain,
		(unsigned long long)state->verify_frames,(unsigned long long)state->verify_rounds,(unsigned long long)state->verify_proposed,(unsigned long long)state->verify_accepted,
		(unsigned long long)state->verify_plain_rounds,(unsigned long long)state->verify_tokens);
	length = snprintf(positions,sizeof(positions),"VERIFY-POSITIONS");
	for (index=0u; index<SPARK_GLM52_VERIFY_ROWS_LIMIT - 1u && length > 0 && (size_t)length < sizeof(positions); index++)
		length += snprintf(positions + length,sizeof(positions) - (size_t)length," p%u=%llu/%llu",index + 1u,
			(unsigned long long)state->verify_position_accepted[index],(unsigned long long)state->verify_position_reached[index]);
	fprintf(stderr,"%s\n",positions);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52StepsAdvance(SparkGlm52TpChain *chain,uint32_t *more_out)
{
	SparkGlm52ModuleState *state = chain->state;
	SparkGlm52ExecutionSlot *slot = chain->slot;
	SparkSpeculationPolicyVerifyResult result;
	SparkStatus status;
	uint32_t committed,lane,index,produced = chain->steps_produced,budget = chain->steps_budget;
	*more_out = 0u;
	if ( slot->host_kv_access_error[0] != 0u )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	if ( chain->verify_draft_count != 0u )
	{
		status = SparkSpeculationPolicyResolveVerifierTokens(chain->verify_draft,chain->verify_draft_count,slot->host_output_token_ids,chain->verify_draft_count + 1u,SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT,&result);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		committed = result.committed_token_count;
		if ( committed != result.accepted_draft_token_count + 1u || committed > budget - produced )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		memcpy(chain->steps_tokens + produced,slot->host_output_token_ids,(uint64_t)committed * sizeof(uint32_t));
		state->verify_rounds++;
		state->verify_proposed += chain->verify_draft_count;
		state->verify_accepted += result.accepted_draft_token_count;
		for (index=0u; index<chain->verify_draft_count && index<=result.accepted_draft_token_count; index++)
		{
			state->verify_position_reached[index]++;
			if ( index < result.accepted_draft_token_count )
				state->verify_position_accepted[index]++;
		}
		chain->verify_rounds++;
		chain->verify_accepted += result.accepted_draft_token_count;
		for (index=0u; index<committed && SparkGlm52TapActive(state) != 0u; index++)
		{
			status = SparkGlm52TapEmit(chain,chain->steps_sequence_ids[0],index);
			if ( status != SPARK_STATUS_OK )
				SPARK_RETURN(status);
		}
	}
	else
	{
		committed = 1u;
		for (lane=0u; lane<chain->steps_lanes; lane++)
		{
			chain->steps_tokens[lane * budget + produced] = slot->host_output_token_ids[lane];
			status = SparkGlm52TapActive(state) != 0u ? SparkGlm52TapEmit(chain,chain->steps_sequence_ids[lane],lane) : SPARK_STATUS_OK;
			if ( status != SPARK_STATUS_OK )
				SPARK_RETURN(status);
		}
		state->verify_plain_rounds++;
		chain->verify_plain++;
	}
	if ( state->verify_drafter == SPARK_SPECULATION_VERIFY_DRAFTER_LOOKUP )
		for (lane=0u; lane<chain->steps_lanes; lane++)
		{
			status = SparkSpeculationLookupDraftObserve(&state->verify_lookup,chain->steps_lane_slots[lane],chain->steps_sequence_ids[lane],chain->steps_positions[lane] + produced + 1u,chain->steps_tokens + lane * budget + produced,committed);
			if ( status != SPARK_STATUS_OK )
				SPARK_RETURN(status);
		}
	chain->steps_produced = produced + committed;
	if ( chain->steps_produced >= budget )
		return(SparkGlm52StepsFinish(chain));
	status = SparkGlm52StepsPlan(chain);
	*more_out = status == SPARK_STATUS_OK ? 1u : 0u;
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm52StepsRekey(SparkGlm52TpChain *chain)
{
	SparkGlm52ModuleState *state = chain->state;
	if ( state->tp_degree == 1u || state->tp_collective_disabled != 0u || state->tp_device_collective_initialized == 0u )
		return(SPARK_STATUS_OK);
	return(SparkTpDeviceCollectiveChainKey(&state->tp_device_collective,chain->frame->request_id & SPARK_TP_DEVICE_COLLECTIVE_CHAIN_ID_MASK));
}

static uint32_t SparkGlm52StepsContinue(SparkGlm52TpChain *chain,SparkStatus *status)
{
	uint32_t more = 0u;
	if ( *status != SPARK_STATUS_OK || chain->steps_budget == 0u )
		return(0u);
	*status = SparkGlm52StepsAdvance(chain,&more);
	if ( *status == SPARK_STATUS_OK && more != 0u )
		*status = SparkGlm52StepsRekey(chain);
	if ( *status != SPARK_STATUS_OK || more == 0u )
		return(0u);
	SparkGlm52RunChain(chain);
	return(1u);
}
