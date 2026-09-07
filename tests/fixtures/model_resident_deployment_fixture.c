#include "model_resident_deployment_fixture.h"

#include <stdio.h>
#include <unistd.h>

static int32_t TestModelResidentWriteText(FILE *stream,const char *text)
{
	const uint8_t *cursor;
	uint8_t value;
	if ( stream == 0 || text == 0 || text[0] == '\0' )
		return(-1);
	if ( fputc('"',stream) == EOF )
		return(-2);
	for (cursor=(const uint8_t *)text; *cursor!=0u; cursor++)
	{
		value = *cursor;
		if ( value == '"' || value == '\\' )
		{
			if ( fputc('\\',stream) == EOF || fputc(value,stream) == EOF )
				return(-3);
		}
		else if ( value < 0x20u )
		{
			if ( fprintf(stream,"\\u%04x",(uint32_t)value) < 0 )
				return(-4);
		}
		else if ( fputc(value,stream) == EOF )
			return(-5);
	}
	return(fputc('"',stream) == EOF ? -6 : 0);
}

static int32_t TestModelResidentWriteEndpoint(
	FILE *stream,
	const SparkModelResidentEndpoint *endpoint)
{
	int32_t status;
	if ( SparkModelResidentEndpointValidate(endpoint) != SPARK_STATUS_OK )
		return(-1);
	if ( endpoint->kind == SPARK_MODEL_RESIDENT_ENDPOINT_KIND_UNIX )
	{
		status = fputs("{\"kind\":\"unix\",\"path\":",stream) == EOF ? -2 : 0;
		if ( status == 0 )
			status = TestModelResidentWriteText(stream,endpoint->unix_socket_path);
	}
	else
	{
		status = fputs("{\"kind\":\"tcp\",\"host\":",stream) == EOF ? -3 : 0;
		if ( status == 0 )
			status = TestModelResidentWriteText(stream,endpoint->tcp_host);
		if ( status == 0 && fprintf(stream,",\"port\":%u",endpoint->tcp_port) < 0 )
			status = -4;
	}
	if ( status == 0 && fputc('}',stream) == EOF )
		status = -5;
	return(status);
}

static int32_t TestModelResidentWriteNode(
	FILE *stream,
	const TestModelResidentDeploymentFixture *fixture,
	uint32_t rank)
{
	int32_t status;
	uint32_t stage;
	stage = fixture->stage_indices != 0 ? fixture->stage_indices[rank] : rank;
	status = fprintf(stream,"%s{\"rank_index\":%u,\"stage_index\":%u,\"runtime_root\":",rank == 0u ? "" : ",",rank,stage) < 0 ? -1 : 0;
	if ( status == 0 )
		status = TestModelResidentWriteText(stream,fixture->runtime_roots[rank]);
	if ( status == 0 && fputs(",\"node_target\":",stream) == EOF )
		status = -2;
	if ( status == 0 )
		status = TestModelResidentWriteText(stream,fixture->node_target);
	if ( status == 0 && fputs(",\"transport_host\":",stream) == EOF )
		status = -3;
	if ( status == 0 )
		status = TestModelResidentWriteText(stream,fixture->transport_hosts[rank]);
	if ( status == 0 && fputs(",\"adapter_configuration_path\":",stream) == EOF )
		status = -4;
	if ( status == 0 )
		status = TestModelResidentWriteText(stream,fixture->adapter_configuration_path);
	if ( status == 0 && fputs(",\"kv_backing_directory\":",stream) == EOF )
		status = -5;
	if ( status == 0 )
		status = fixture->kv_backing_directory != 0 ?
			TestModelResidentWriteText(stream,fixture->kv_backing_directory) :
			(fputs("null",stream) == EOF ? -6 : 0);
	if ( status == 0 && fprintf(stream,
		",\"kv_backing_maximum_bytes\":%llu,\"control_endpoint\":",
		(unsigned long long)fixture->kv_backing_maximum_bytes) < 0 )
		status = -7;
	if ( status == 0 )
		status = TestModelResidentWriteEndpoint(stream,&fixture->control_endpoints[rank]);
	if ( status == 0 && fputc('}',stream) == EOF )
		status = -8;
	return(status);
}

static int32_t TestModelResidentWriteBody(
	FILE *stream,
	const TestModelResidentDeploymentFixture *fixture)
{
	int32_t status;
	uint32_t rank;
	status = fprintf(stream,"{\"schema_version\":2,\"coordinator_rank_index\":%u,\"adapter\":{\"shared_object_path\":",fixture->coordinator_rank_index) < 0 ? -1 : 0;
	if ( status == 0 )
		status = TestModelResidentWriteText(stream,fixture->adapter_shared_object_path);
	if ( status == 0 && fputs("},\"driver\":{\"shared_object_path\":",stream) == EOF )
		status = -2;
	if ( status == 0 )
		status = TestModelResidentWriteText(stream,fixture->driver_shared_object_path);
	if ( status == 0 && fputs(",\"program_name\":",stream) == EOF )
		status = -3;
	if ( status == 0 )
		status = TestModelResidentWriteText(stream,fixture->driver_program_name);
	if ( status == 0 && fputs("},\"transport\":{\"shared_object_path\":",stream) == EOF )
		status = -4;
	if ( status == 0 )
		status = TestModelResidentWriteText(stream,fixture->transport_shared_object_path);
	if ( status == 0 && fputs(",\"mode\":",stream) == EOF )
		status = -5;
	if ( status == 0 )
		status = TestModelResidentWriteText(stream,fixture->transport_mode);
	if ( status == 0 && fprintf(stream,",\"control_port_base\":%u},\"runtime_limits\":{\"max_inflight_submissions\":%u,\"max_active_sequences\":%u,\"max_input_rows\":%u,\"resident_sequence_capacity\":%u,\"kv_logical_page_capacity\":%u,\"kv_physical_page_capacity\":%u},\"nodes\":[",fixture->control_port_base,fixture->runtime_limits.max_inflight_submission_count,fixture->runtime_limits.max_active_sequence_count,fixture->runtime_limits.max_input_row_count,fixture->runtime_limits.resident_sequence_capacity,fixture->runtime_limits.kv_logical_page_capacity,fixture->runtime_limits.kv_physical_page_capacity) < 0 )
		status = -6;
	for (rank=0u; status==0 && rank<fixture->node_count; rank++)
		status = TestModelResidentWriteNode(stream,fixture,rank);
	if ( status == 0 && fputs("]",stream) == EOF )
		status = -7;
	if ( status == 0 && fixture->tokenizer_asset_path != 0 )
	{
		if ( fputs(",\"tokenizer\":{\"path\":",stream) == EOF )
			status = -8;
		else
			status = TestModelResidentWriteText(stream,fixture->tokenizer_asset_path);
		if ( status == 0 && fputc('}',stream) == EOF )
			status = -9;
	}
	if ( status == 0 && fputs("}",stream) == EOF )
		status = -7;
	return(status);
}

int32_t TestModelResidentDeploymentWrite(
	const char *path,
	const TestModelResidentDeploymentFixture *fixture)
{
	FILE *stream;
	int32_t close_status,status;
	if ( path == 0 || path[0] == '\0' || fixture == 0 || fixture->node_count == 0u || fixture->runtime_roots == 0 || fixture->transport_hosts == 0 || fixture->control_endpoints == 0 )
		return(-1);
	stream = fopen(path,"wb");
	if ( stream == 0 )
		return(-2);
	status = TestModelResidentWriteBody(stream,fixture);
	close_status = fclose(stream);
	if ( status != 0 || close_status != 0 )
	{
		unlink(path);
		return(status != 0 ? status : -3);
	}
	return(0);
}
