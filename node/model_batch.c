#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_model_batch_engine.h"
#include "sparkpipe/spark_model_resident_deployment.h"

#define SPARK_MODEL_BATCH_FILE_SCHEMA_VERSION 1u
#define SPARK_MODEL_BATCH_POLL_TIMEOUT_MS 10
#define SPARK_MODEL_BATCH_STAGE_PROFILE_EVENT_CAPACITY_MAX 1048576u

typedef struct SparkModelBatchFileRequest
{
	SparkModelBatchSubmitRequest request;
	uint32_t *prompt_token_ids;
} SparkModelBatchFileRequest;

typedef struct SparkModelBatchFile
{
	SparkModelBatchEngineConfiguration engine;
	uint32_t maximum_new_submissions_per_progress;
	uint32_t sequential_submissions;
	uint32_t request_count;
	SparkModelBatchFileRequest *requests;
} SparkModelBatchFile;

typedef struct SparkModelBatchOutput
{
	uint32_t terminal_count;
	uint32_t error_count;
	uint32_t write_failed;
	uint32_t stage_completion_count;
	uint32_t stage_completion_capacity;
	uint32_t dropped_stage_completion_count;
	SparkModelPipelineStageCompletion *stage_completions;
	uint8_t *pending_output;
	uint32_t pending_output_bytes;
	uint32_t pending_output_capacity;
} SparkModelBatchOutput;

static const char *const SparkModelBatchFileMembers[] =
{
	"schema_version",
	"connect_timeout_ms",
	"request_capacity",
	"max_context_tokens",
	"max_prefill_rows_per_submission",
	"maximum_messages_per_rank_per_progress",
	"maximum_new_submissions_per_progress",
	"stop_token_ids",
	"requests"
};

static const char *const SparkModelBatchRequestMembers[] =
{
	"request_id",
	"sequence_id",
	"priority",
	"output_token_budget",
	"prompt_token_ids"
};

static int32_t SparkModelBatchJsonMember(
	const SparkJsonDocument *document,
	int32_t object,
	const char *name)
{
	return(SparkJsonFindObjectMember(document,object,name));
}

static SparkStatus SparkModelBatchJsonU32(
	const SparkJsonDocument *document,
	int32_t object,
	const char *name,
	uint32_t *value)
{
	int32_t token;
	token = SparkModelBatchJsonMember(document,object,name);
	return(token < 0 ? SPARK_STATUS_SCHEMA_ERROR : SparkJsonGetUInt32(document,token,value));
}

static SparkStatus SparkModelBatchJsonU64(
	const SparkJsonDocument *document,
	int32_t object,
	const char *name,
	uint64_t *value)
{
	int32_t token;
	token = SparkModelBatchJsonMember(document,object,name);
	return(token < 0 ? SPARK_STATUS_SCHEMA_ERROR : SparkJsonGetUInt64(document,token,value));
}

static int SparkModelBatchCompareU64(const void *left,const void *right)
{
	uint64_t a,b;
	a = *(const uint64_t *)left;
	b = *(const uint64_t *)right;
	return(a < b ? -1 : a > b ? 1 : 0);
}

static void SparkModelBatchFileDestroy(SparkModelBatchFile *file)
{
	uint32_t index;
	if ( file == 0 )
		return;
	for (index=0u; index<file->request_count; index++)
		free(file->requests[index].prompt_token_ids);
	free(file->requests);
	memset(file,0,sizeof(*file));
}

static SparkStatus SparkModelBatchParseTokenArray(
	const SparkJsonDocument *document,
	int32_t array,
	uint32_t maximum_count,
	uint32_t **tokens_out,
	uint32_t *count_out)
{
	uint32_t *tokens;
	uint32_t count,index;
	int32_t element;
	SparkStatus status;
	if ( !SparkJsonTokenIsType(document,array,SPARK_JSON_TOKEN_ARRAY) )
		return(SPARK_STATUS_SCHEMA_ERROR);
	count = SparkJsonGetArrayElementCount(document,array);
	if ( count == 0u || count > maximum_count )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	tokens = (uint32_t *)calloc(count,sizeof(*tokens));
	if ( tokens == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = SPARK_STATUS_OK;
	element = SparkJsonGetArrayElementFirst(document,array);
	for (index=0u; status==SPARK_STATUS_OK && index<count; index++)
	{
		if ( element < 0 )
			status = SPARK_STATUS_SCHEMA_ERROR;
		else
			status = SparkJsonGetUInt32(document,element,&tokens[index]);
		element = SparkJsonGetArrayElementNext(document,array,element);
	}
	if ( status != SPARK_STATUS_OK )
		free(tokens);
	else
	{
		*tokens_out = tokens;
		*count_out = count;
	}
	return(status);
}

static SparkStatus SparkModelBatchParseRequest(
	const SparkJsonDocument *document,
	int32_t object,
	uint32_t max_context_tokens,
	SparkModelBatchFileRequest *file_request)
{
	SparkModelBatchSubmitRequest *request;
	int32_t tokens;
	SparkStatus status;
	if ( !SparkJsonTokenIsType(document,object,SPARK_JSON_TOKEN_OBJECT) )
		return(SPARK_STATUS_SCHEMA_ERROR);
	status = SparkJsonValidateObjectMembersExact(document,object,SparkModelBatchRequestMembers,(uint32_t)(sizeof(SparkModelBatchRequestMembers) / sizeof(SparkModelBatchRequestMembers[0])));
	request = &file_request->request;
	memset(request,0,sizeof(*request));
	request->abi_version = SPARK_MODEL_BATCH_ENGINE_ABI_VERSION;
	request->descriptor_bytes = SPARK_MODEL_BATCH_SUBMIT_REQUEST_BYTES;
	if ( status == SPARK_STATUS_OK )
		status = SparkModelBatchJsonU64(document,object,"request_id",&request->request_id);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelBatchJsonU64(document,object,"sequence_id",&request->sequence_id);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelBatchJsonU32(document,object,"priority",&request->priority);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelBatchJsonU32(document,object,"output_token_budget",&request->output_token_budget);
	tokens = status == SPARK_STATUS_OK ? SparkModelBatchJsonMember(document,object,"prompt_token_ids") : -1;
	if ( status == SPARK_STATUS_OK )
		status = tokens < 0 ? SPARK_STATUS_SCHEMA_ERROR : SparkModelBatchParseTokenArray(document,tokens,max_context_tokens,&file_request->prompt_token_ids,&request->prompt_token_count);
	request->prompt_token_ids = file_request->prompt_token_ids;
	if ( status == SPARK_STATUS_OK && (request->request_id == 0u || request->sequence_id == 0u || request->output_token_budget == 0u || request->output_token_budget > max_context_tokens - request->prompt_token_count) )
		status = SPARK_STATUS_CAPACITY_EXCEEDED;
	return(status);
}

static SparkStatus SparkModelBatchValidateUniqueIds(
	const SparkModelBatchFile *file)
{
	uint64_t *request_ids,*sequence_ids;
	uint32_t index;
	SparkStatus status;
	request_ids = (uint64_t *)calloc(file->request_count,sizeof(*request_ids));
	sequence_ids = (uint64_t *)calloc(file->request_count,sizeof(*sequence_ids));
	if ( request_ids == 0 || sequence_ids == 0 )
	{
		free(sequence_ids);
		free(request_ids);
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	for (index=0u; index<file->request_count; index++)
	{
		request_ids[index] = file->requests[index].request.request_id;
		sequence_ids[index] = file->requests[index].request.sequence_id;
	}
	qsort(request_ids,file->request_count,sizeof(*request_ids),SparkModelBatchCompareU64);
	qsort(sequence_ids,file->request_count,sizeof(*sequence_ids),SparkModelBatchCompareU64);
	status = SPARK_STATUS_OK;
	for (index=1u; status==SPARK_STATUS_OK && index<file->request_count; index++)
		if ( request_ids[index] == request_ids[index - 1u] || sequence_ids[index] == sequence_ids[index - 1u] )
			status = SPARK_STATUS_DUPLICATE;
	free(sequence_ids);
	free(request_ids);
	return(status);
}

static SparkStatus SparkModelBatchParseRequests(
	const SparkJsonDocument *document,
	int32_t array,
	SparkModelBatchFile *file)
{
	uint32_t index;
	int32_t element;
	SparkStatus status;
	if ( !SparkJsonTokenIsType(document,array,SPARK_JSON_TOKEN_ARRAY) )
		return(SPARK_STATUS_SCHEMA_ERROR);
	file->request_count = SparkJsonGetArrayElementCount(document,array);
	if ( file->request_count == 0u || file->request_count > file->engine.request_capacity )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	file->requests = (SparkModelBatchFileRequest *)calloc(file->request_count,sizeof(*file->requests));
	if ( file->requests == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = SPARK_STATUS_OK;
	element = SparkJsonGetArrayElementFirst(document,array);
	for (index=0u; status==SPARK_STATUS_OK && index<file->request_count; index++)
	{
		if ( element < 0 )
			status = SPARK_STATUS_SCHEMA_ERROR;
		else
			status = SparkModelBatchParseRequest(document,element,file->engine.max_context_tokens,&file->requests[index]);
		element = SparkJsonGetArrayElementNext(document,array,element);
	}
	return(status == SPARK_STATUS_OK ? SparkModelBatchValidateUniqueIds(file) : status);
}

static SparkStatus SparkModelBatchParseStopTokens(
	const SparkJsonDocument *document,
	int32_t array,
	SparkModelBatchFile *file)
{
	uint32_t index,left;
	int32_t element;
	SparkStatus status;
	if ( !SparkJsonTokenIsType(document,array,SPARK_JSON_TOKEN_ARRAY) )
		return(SPARK_STATUS_SCHEMA_ERROR);
	file->engine.stop_token_count = SparkJsonGetArrayElementCount(document,array);
	if ( file->engine.stop_token_count > SPARK_MODEL_BATCH_ENGINE_MAX_STOP_TOKEN_COUNT )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = SPARK_STATUS_OK;
	element = SparkJsonGetArrayElementFirst(document,array);
	for (index=0u; status==SPARK_STATUS_OK && index<file->engine.stop_token_count; index++)
	{
		if ( element < 0 )
			status = SPARK_STATUS_SCHEMA_ERROR;
		else
			status = SparkJsonGetUInt32(document,element,&file->engine.stop_token_ids[index]);
		element = SparkJsonGetArrayElementNext(document,array,element);
	}
	for (index=0u; status==SPARK_STATUS_OK && index<file->engine.stop_token_count; index++)
		for (left=0u; left<index; left++)
			if ( file->engine.stop_token_ids[left] == file->engine.stop_token_ids[index] )
				status = SPARK_STATUS_DUPLICATE;
	return(status);
}

static SparkStatus SparkModelBatchLoadFile(
	const char *path,
	SparkModelBatchFile *file)
{
	SparkJsonDocument document;
	uint32_t schema_version;
	int32_t root,requests,stop_tokens;
	SparkStatus status;
	memset(file,0,sizeof(*file));
	SparkJsonDocumentReset(&document);
	status = SparkJsonLoadFile(path,&document);
	root = status == SPARK_STATUS_OK ? SparkJsonGetRootToken(&document) : -1;
	if ( status == SPARK_STATUS_OK && !SparkJsonTokenIsType(&document,root,SPARK_JSON_TOKEN_OBJECT) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK )
		status = SparkJsonValidateObjectMembersExact(&document,root,SparkModelBatchFileMembers,(uint32_t)(sizeof(SparkModelBatchFileMembers) / sizeof(SparkModelBatchFileMembers[0])));
	if ( status == SPARK_STATUS_OK )
		status = SparkModelBatchJsonU32(&document,root,"schema_version",&schema_version);
	if ( status == SPARK_STATUS_OK && schema_version != SPARK_MODEL_BATCH_FILE_SCHEMA_VERSION )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK )
		status = SparkModelBatchJsonU32(&document,root,"connect_timeout_ms",&file->engine.connect_timeout_ms);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelBatchJsonU32(&document,root,"request_capacity",&file->engine.request_capacity);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelBatchJsonU32(&document,root,"max_context_tokens",&file->engine.max_context_tokens);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelBatchJsonU32(&document,root,"max_prefill_rows_per_submission",&file->engine.max_prefill_rows_per_submission);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelBatchJsonU32(&document,root,"maximum_messages_per_rank_per_progress",&file->engine.maximum_messages_per_rank_per_progress);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelBatchJsonU32(&document,root,"maximum_new_submissions_per_progress",&file->maximum_new_submissions_per_progress);
	stop_tokens = status == SPARK_STATUS_OK ? SparkModelBatchJsonMember(&document,root,"stop_token_ids") : -1;
	if ( status == SPARK_STATUS_OK )
		status = stop_tokens < 0 ? SPARK_STATUS_SCHEMA_ERROR : SparkModelBatchParseStopTokens(&document,stop_tokens,file);
	requests = status == SPARK_STATUS_OK ? SparkModelBatchJsonMember(&document,root,"requests") : -1;
	if ( status == SPARK_STATUS_OK )
		status = requests < 0 ? SPARK_STATUS_SCHEMA_ERROR : SparkModelBatchParseRequests(&document,requests,file);
	SparkJsonDocumentDestroy(&document);
	if ( status != SPARK_STATUS_OK )
		SparkModelBatchFileDestroy(file);
	return(status);
}

static const char *SparkModelBatchEventName(uint32_t kind)
{
	switch (kind)
	{
	case SPARK_MODEL_BATCH_EVENT_REQUEST_ACCEPTED:
		return("accepted");
	case SPARK_MODEL_BATCH_EVENT_TOKEN:
		return("token");
	case SPARK_MODEL_BATCH_EVENT_REQUEST_COMPLETED:
		return("completed");
	case SPARK_MODEL_BATCH_EVENT_REQUEST_CANCELLED:
		return("cancelled");
	case SPARK_MODEL_BATCH_EVENT_ERROR:
		return("error");
	default:
		return("unknown");
	}
}

/* Emission is decoupled from the decode loop: tokens are formatted into a
 * user-space buffer and written to stdout WITHOUT blocking. A slow reader
 * (the harness draining stdout) would otherwise stall the only producer of
 * the next decode submission, turning reader latency into wall-clock. */
static int32_t SparkModelBatchOutputReserve(
	SparkModelBatchOutput *output,
	uint32_t bytes)
{
	uint32_t capacity,required;
	uint8_t *grown;
	if ( bytes <= output->pending_output_capacity - output->pending_output_bytes )
		return(0);
	required = output->pending_output_bytes + bytes;
	capacity = output->pending_output_capacity != 0u ? output->pending_output_capacity : 4096u;
	while ( capacity < required )
		capacity *= 2u;
	grown = (uint8_t *)realloc(output->pending_output,capacity);
	if ( grown == 0 )
		return(-1);
	output->pending_output = grown;
	output->pending_output_capacity = capacity;
	return(0);
}

static void SparkModelBatchFlushOutput(SparkModelBatchOutput *output)
{
	ssize_t written;
	if ( output->write_failed != 0u )
		return;
	while ( output->pending_output_bytes != 0u )
	{
		written = write(STDOUT_FILENO,output->pending_output,output->pending_output_bytes);
		if ( written > 0 )
		{
			memmove(output->pending_output,output->pending_output + written,
				output->pending_output_bytes - (uint32_t)written);
			output->pending_output_bytes -= (uint32_t)written;
			continue;
		}
		if ( written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK) )
			return;  /* reader is behind; keep bytes, retry next loop iteration */
		output->write_failed = 1u;
		return;
	}
}

static void SparkModelBatchWriteEvent(
	void *event_context,
	const SparkModelBatchEvent *event)
{
	SparkModelBatchOutput *output;
	char line[512];
	int32_t length;
	output = (SparkModelBatchOutput *)event_context;
	length = snprintf(line,sizeof(line),"{\"schema_version\":1,\"event\":\"%s\",\"status\":%u,\"request_id\":%llu,\"sequence_id\":%llu,\"request_handle\":%llu,\"token_id\":%u,\"token_index\":%u,\"generated_token_count\":%u,\"stop_token\":%s",SparkModelBatchEventName(event->kind),event->status,(unsigned long long)event->request_id,(unsigned long long)event->sequence_id,(unsigned long long)event->request_handle,event->token_id,event->token_index,event->generated_token_count,(event->flags & SPARK_MODEL_BATCH_EVENT_FLAG_STOP_TOKEN) != 0u ? "true" : "false");
	if ( length >= 0 && event->kind == SPARK_MODEL_BATCH_EVENT_REQUEST_COMPLETED && (size_t)length < sizeof(line) )
		length += snprintf(line + length,sizeof(line) - (size_t)length,",\"model_extension_kind\":%u,\"first_draft_miss\":%u,\"first_draft_policy\":%u",event->model_extension_kind,event->first_draft_miss_count,event->first_draft_policy);
	if ( length >= 0 && (size_t)length < sizeof(line) )
		length += snprintf(line + length,sizeof(line) - (size_t)length,"}\n");
	if ( length < 0 || (size_t)length >= sizeof(line) ||
		SparkModelBatchOutputReserve(output,(uint32_t)length) != 0 )
		output->write_failed = 1u;
	else
	{
		memcpy(output->pending_output + output->pending_output_bytes,line,(size_t)length);
		output->pending_output_bytes += (uint32_t)length;
		SparkModelBatchFlushOutput(output);
	}
	if ( event->kind == SPARK_MODEL_BATCH_EVENT_REQUEST_COMPLETED || event->kind == SPARK_MODEL_BATCH_EVENT_REQUEST_CANCELLED || event->kind == SPARK_MODEL_BATCH_EVENT_ERROR )
		output->terminal_count++;
	if ( event->kind == SPARK_MODEL_BATCH_EVENT_ERROR )
		output->error_count++;
}

static void SparkModelBatchRecordStageCompletion(
	void *completion_context,
	const SparkModelPipelineStageCompletion *completion)
{
	SparkModelBatchOutput *output;
	output = (SparkModelBatchOutput *)completion_context;
	if ( output->stage_completion_count == output->stage_completion_capacity )
	{
		output->dropped_stage_completion_count++;
		return;
	}
	output->stage_completions[output->stage_completion_count++] = *completion;
}

static SparkStatus SparkModelBatchInitializeStageProfile(
	const SparkModelBatchFile *file,
	uint32_t stage_count,
	SparkModelBatchOutput *output)
{
	uint64_t event_capacity,submission_capacity;
	uint32_t index;
	submission_capacity = 0u;
	for (index=0u; index<file->request_count; index++)
		submission_capacity += (uint64_t)file->requests[index].request.prompt_token_count + file->requests[index].request.output_token_budget + 1u;
	event_capacity = submission_capacity * stage_count;
	if ( event_capacity > SPARK_MODEL_BATCH_STAGE_PROFILE_EVENT_CAPACITY_MAX )
		event_capacity = SPARK_MODEL_BATCH_STAGE_PROFILE_EVENT_CAPACITY_MAX;
	if ( event_capacity == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	output->stage_completion_capacity = (uint32_t)event_capacity;
	output->stage_completions = (SparkModelPipelineStageCompletion *)calloc(output->stage_completion_capacity,sizeof(output->stage_completions[0]));
	return(output->stage_completions != 0 ? SPARK_STATUS_OK : SPARK_STATUS_CAPACITY_EXCEEDED);
}

static int32_t SparkModelBatchWriteStageProfile(
	const SparkModelBatchOutput *output)
{
	const SparkModelPipelineStageCompletion *completion;
	uint32_t index;
	for (index=0u; index<output->stage_completion_count; index++)
	{
		completion = &output->stage_completions[index];
		if ( fprintf(stderr,"sparkpipe_stage_profile submission=%llu stage=%u kind=%u lanes=%u rows=%u status=%u flags=%u queue_ns=%llu service_ns=%llu elapsed_ns=%llu completion_ns=%llu device_bytes=%llu host_bytes=%llu\n",(unsigned long long)completion->submission_id,completion->stage_index,completion->work_kind,completion->active_sequence_count,completion->row_count,completion->status,completion->flags,(unsigned long long)completion->queue_delay_ns,(unsigned long long)completion->service_time_ns,(unsigned long long)completion->client_elapsed_ns,(unsigned long long)completion->client_completion_time_ns,(unsigned long long)completion->device_memcpy_bytes,(unsigned long long)completion->host_staging_bytes) < 0 )
			return(-1);
	}
	if ( fprintf(stderr,"sparkpipe_stage_profile_summary events=%u dropped=%u\n",output->stage_completion_count,output->dropped_stage_completion_count) < 0 )
		return(-2);
	return(fflush(stderr) == 0 ? 0 : -3);
}

static int32_t SparkModelBatchWriteJsonString(const char *text)
{
	const uint8_t *cursor;
	uint8_t value;
	if ( fputc('"',stdout) == EOF )
		return(-1);
	for (cursor=(const uint8_t *)text; *cursor!=0u; cursor++)
	{
		value = *cursor;
		if ( value == '"' || value == '\\' )
		{
			if ( fputc('\\',stdout) == EOF || fputc(value,stdout) == EOF )
				return(-2);
		}
		else if ( value < 0x20u )
		{
			if ( fprintf(stdout,"\\u%04x",(uint32_t)value) < 0 )
				return(-3);
		}
		else if ( fputc(value,stdout) == EOF )
			return(-4);
	}
	return(fputc('"',stdout) == EOF ? -5 : 0);
}

static int32_t SparkModelBatchWriteReady(
	const SparkModelServingAdapterDescriptor *descriptor)
{
	if ( fputs("{\"schema_version\":1,\"event\":\"ready\",\"adapter_id\":",stdout) == EOF || SparkModelBatchWriteJsonString(descriptor->adapter_id) < 0 || fputs(",\"model_id\":",stdout) == EOF || SparkModelBatchWriteJsonString(descriptor->model_id) < 0 || fputs(",\"model_revision\":",stdout) == EOF || SparkModelBatchWriteJsonString(descriptor->model_revision) < 0 )
		return(-1);
	if ( fprintf(stdout,",\"stage_count\":%u,\"linear_weight_codec\":%u,\"expert_weight_codec\":%u,\"kv_cache_codec\":%u}\n",descriptor->stage_count,descriptor->linear_weight_codec,descriptor->expert_weight_codec,descriptor->kv_cache_codec) < 0 )
		return(-2);
	return(fflush(stdout) == 0 ? 0 : -3);
}

static SparkStatus SparkModelBatchSubmitAll(
	SparkModelBatchEngine *engine,
	const SparkModelBatchFile *file,
	uint32_t first,
	uint32_t count)
{
	SparkModelBatchRequestHandle handle;
	uint32_t index;
	SparkStatus status;
	status = SPARK_STATUS_OK;
	for (index=first; status==SPARK_STATUS_OK && index<first+count; index++)
		status = SparkModelBatchEngineSubmit(engine,&file->requests[index].request,&handle);
	return(status);
}

static int32_t SparkModelBatchPoll(
	const SparkModelResidentClientPollDescriptor *descriptors,
	uint32_t descriptor_count)
{
	struct pollfd poll_descriptors[SPARK_MODEL_RESIDENT_DEPLOYMENT_MAX_NODE_COUNT];
	uint32_t index;
	int32_t status;
	for (index=0u; index<descriptor_count; index++)
	{
		poll_descriptors[index].fd = descriptors[index].fd;
		poll_descriptors[index].events = 0;
		poll_descriptors[index].revents = 0;
		if ( (descriptors[index].events & SPARK_MODEL_RESIDENT_CLIENT_POLL_READ) != 0u )
			poll_descriptors[index].events |= POLLIN;
		if ( (descriptors[index].events & SPARK_MODEL_RESIDENT_CLIENT_POLL_WRITE) != 0u )
			poll_descriptors[index].events |= POLLOUT;
	}
	status = poll(poll_descriptors,descriptor_count,SPARK_MODEL_BATCH_POLL_TIMEOUT_MS);
	return(status >= 0 || errno == EINTR ? 0 : -1);
}

static SparkStatus SparkModelBatchRun(
	SparkModelBatchEngine *engine,
	const SparkModelBatchFile *file,
	SparkModelBatchOutput *output)
{
	SparkModelResidentClientPollDescriptor descriptors[SPARK_MODEL_RESIDENT_DEPLOYMENT_MAX_NODE_COUNT];
	SparkModelBatchEngineView view;
	uint32_t descriptor_count;
	uint32_t submitted;
	SparkStatus status;
	status = SPARK_STATUS_OK;
	submitted = file->sequential_submissions != 0u ? 0u : file->request_count;
	while ( status == SPARK_STATUS_OK && output->terminal_count<file->request_count && output->write_failed == 0u )
	{
		/* sequential mode: request N+1 admits only after request N reached
		 * terminal - the production arrival pattern the prefix cache serves
		 * (a co-scheduled group prefills every lane from zero in parallel) */
		while ( file->sequential_submissions != 0u && submitted < file->request_count && output->terminal_count == submitted )
		{
			status = SparkModelBatchSubmitAll(engine,file,submitted,1u);
			if ( status != SPARK_STATUS_OK )
				break;
			submitted++;
		}
		if ( status != SPARK_STATUS_OK )
			break;
		if ( file->sequential_submissions != 0u && submitted == file->request_count )
			status = SparkModelBatchEngineCloseAdmission(engine);
		if ( status != SPARK_STATUS_OK )
			break;
		SparkModelBatchFlushOutput(output);
		status = SparkModelBatchEngineProgress(engine,file->maximum_new_submissions_per_progress);
		if ( status == SPARK_STATUS_OK )
			status = SparkModelBatchEngineGetPollDescriptors(engine,descriptors,SPARK_MODEL_RESIDENT_DEPLOYMENT_MAX_NODE_COUNT,&descriptor_count);
		if ( status == SPARK_STATUS_OK && SparkModelBatchPoll(descriptors,descriptor_count) < 0 )
			status = SPARK_STATUS_IO_ERROR;
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkModelBatchEngineGetView(engine,&view);
	if ( status == SPARK_STATUS_OK && (view.live_request_count != 0u || view.inflight_submission_count != 0u) )
		status = SPARK_STATUS_BUSY;
	return(status);
}

static int32_t SparkModelBatchParseArguments(
	int32_t argc,
	char **argv,
	const char **deployment_path,
	const char **runtime_root,
	const char **batch_path,
	uint32_t *profile_stages)
{
	int32_t index;
	*deployment_path = 0;
	*runtime_root = 0;
	*batch_path = 0;
	*profile_stages = 0u;
	for (index=1; index<argc; index++)
	{
		if ( strcmp(argv[index],"--deployment") == 0 && index + 1 < argc )
			*deployment_path = argv[++index];
		else if ( strcmp(argv[index],"--runtime-root") == 0 && index + 1 < argc )
			*runtime_root = argv[++index];
		else if ( strcmp(argv[index],"--batch") == 0 && index + 1 < argc )
			*batch_path = argv[++index];
		else if ( strcmp(argv[index],"--profile-stages") == 0 )
			*profile_stages = 1u;
		else
			return(-1);
	}
	return(*deployment_path != 0 && *runtime_root != 0 && *batch_path != 0 ? 0 : -2);
}

int main(int argc,char **argv)
{
	const SparkModelServingAdapterDescriptor *descriptor;
	SparkModelResidentDeployment deployment;
	SparkModelBatchEngine *engine;
	SparkModelBatchEngineView engine_view;
	SparkModelBatchFile file;
	SparkModelBatchOutput output;
	const char *deployment_path,*runtime_root,*batch_path;
	uint32_t failed_stage_index,profile_stages,view_valid;
	SparkStatus status,destroy_status;
	if ( SparkModelBatchParseArguments(argc,argv,&deployment_path,&runtime_root,&batch_path,&profile_stages) < 0 )
	{
		fprintf(stderr,"usage: sparkpipe_model_batch --deployment PATH --runtime-root PATH --batch PATH [--profile-stages]\n");
		return(2);
	}
	memset(&output,0,sizeof(output));
	memset(&file,0,sizeof(file));
	failed_stage_index = SPARK_MODEL_PIPELINE_CLIENT_INVALID_STAGE_INDEX;
	view_valid = 0u;
	engine = 0;
	SparkModelResidentDeploymentReset(&deployment);
	status = SparkModelResidentDeploymentLoad(deployment_path,&deployment);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelBatchLoadFile(batch_path,&file);
	if ( status == SPARK_STATUS_OK && getenv("SPARK_MODEL_BATCH_SEQUENTIAL") != 0 )
		file.sequential_submissions = 1u;
	if ( status == SPARK_STATUS_OK && profile_stages != 0u )
		status = SparkModelBatchInitializeStageProfile(&file,deployment.node_count,&output);
	if ( status == SPARK_STATUS_OK )
	{
		file.engine.abi_version = SPARK_MODEL_BATCH_ENGINE_ABI_VERSION;
		file.engine.descriptor_bytes = SPARK_MODEL_BATCH_ENGINE_CONFIGURATION_BYTES;
		file.engine.deployment = &deployment;
		file.engine.runtime_root = runtime_root;
		file.engine.event_function = SparkModelBatchWriteEvent;
		file.engine.event_context = &output;
		if ( profile_stages != 0u )
		{
			file.engine.stage_completion_function = SparkModelBatchRecordStageCompletion;
			file.engine.stage_completion_context = &output;
		}
		status = SparkModelBatchEngineConnect(&file.engine,&engine);
	}
	descriptor = status == SPARK_STATUS_OK ? SparkModelBatchEngineGetAdapterDescriptor(engine) : 0;
	if ( status == SPARK_STATUS_OK && (descriptor == 0 || SparkModelBatchWriteReady(descriptor) < 0) )
		status = SPARK_STATUS_IO_ERROR;
	if ( status == SPARK_STATUS_OK )
		status = SparkModelBatchSubmitAll(engine,&file,0u,file.sequential_submissions != 0u ? 0u : file.request_count);
	if ( status == SPARK_STATUS_OK && file.sequential_submissions == 0u )
		status = SparkModelBatchEngineCloseAdmission(engine);
	if ( status == SPARK_STATUS_OK )
		(void)fcntl(STDOUT_FILENO,F_SETFL,fcntl(STDOUT_FILENO,F_GETFL,0) | O_NONBLOCK);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelBatchRun(engine,&file,&output);
	/* Drain whatever the slow reader could not take during the run (now blocking). */
	(void)fcntl(STDOUT_FILENO,F_SETFL,fcntl(STDOUT_FILENO,F_GETFL,0) & ~O_NONBLOCK);
	SparkModelBatchFlushOutput(&output);
	if ( engine != 0 && SparkModelBatchEngineGetView(engine,&engine_view) ==
		SPARK_STATUS_OK )
	{
		view_valid = 1u;
		if ( status != SPARK_STATUS_OK )
			failed_stage_index = engine_view.pipeline.failed_stage_index;
	}
	destroy_status = SparkModelBatchEngineDestroy(engine);
	if ( status == SPARK_STATUS_OK && destroy_status != SPARK_STATUS_OK )
		status = destroy_status;
	if ( profile_stages != 0u && output.stage_completions != 0 && SparkModelBatchWriteStageProfile(&output) < 0 && status == SPARK_STATUS_OK )
		status = SPARK_STATUS_IO_ERROR;
	if ( status == SPARK_STATUS_OK && output.error_count != 0u )
		status = SPARK_STATUS_INVALID_ARGUMENT;
	if ( status != SPARK_STATUS_OK && failed_stage_index != SPARK_MODEL_PIPELINE_CLIENT_INVALID_STAGE_INDEX )
		fprintf(stderr,"sparkpipe_model_batch_failure status=%u stage=%u\n",status,failed_stage_index);
	if ( view_valid != 0u )
		fprintf(stderr,"sparkpipe_model_batch_pipeline submitted=%llu continued=%llu admitted=%llu rejected=%llu leases=%u\n",(unsigned long long)engine_view.pipeline.submitted_count,(unsigned long long)engine_view.pipeline.continued_count,(unsigned long long)engine_view.pipeline.admitted_count,(unsigned long long)engine_view.pipeline.rejected_count,engine_view.pipeline.active_continue_lease_count);
	fprintf(stderr,"sparkpipe_model_batch_status=%u terminal=%u requests=%u\n",status,output.terminal_count,file.request_count);
	SparkModelBatchFileDestroy(&file);
	SparkModelResidentDeploymentDestroy(&deployment);
	free(output.stage_completions);
	free(output.pending_output);
	return(status == SPARK_STATUS_OK ? 0 : 1);
}
