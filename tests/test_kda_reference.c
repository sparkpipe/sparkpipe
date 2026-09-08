#include <string.h>
#include "sparkpipe/spark_kda_reference.h"

int main(void)
{
	float state[4] = {2.0f,4.0f,6.0f,8.0f};
	float query[2] = {0.5f,-1.0f},key[2] = {1.0f,2.0f};
	float value[2] = {3.0f,5.0f},retention[2] = {0.5f,0.25f},output[2];
	const float first_state[4] = {0.5f,1.5f,0.5f,1.0f};
	const float second_state[4] = {0.375f,1.71875f,0.5f,0.0625f};
	SparkKdaReferenceHead(state,query,key,value,retention,0.5f,2u,2u,output);
	if ( memcmp(state,first_state,sizeof(state)) != 0 || output[0] != -0.25f || output[1] != -0.25f )
		return(1);
	query[0] = 1.0f;
	query[1] = 0.0f;
	key[0] = -0.5f;
	key[1] = 1.0f;
	value[0] = 1.0f;
	value[1] = -2.0f;
	retention[0] = 1.0f;
	retention[1] = 0.5f;
	SparkKdaReferenceHead(state,query,key,value,retention,0.25f,2u,2u,output);
	if ( memcmp(state,second_state,sizeof(state)) != 0 || output[0] != 0.375f || output[1] != 1.71875f )
		return(2);
	state[0] = 2.0f;
	state[1] = 6.0f;
	key[0] = 1.0f;
	key[1] = 2.0f;
	value[0] = 3.0f;
	retention[0] = 0.5f;
	retention[1] = 0.25f;
	SparkKdaReferenceHead(state,query,key,value,retention,0.5f,2u,1u,output);
	return(state[0] != 0.5f || state[1] != 0.5f || output[0] != 0.5f ? 3 : 0);
}
