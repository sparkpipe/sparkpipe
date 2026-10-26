#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_json.h"
#include "tools/token_prompts.h"

static SparkStatus PromptLoadOne(const SparkJsonDocument *document, int32_t object, SparkTokenPrompt *prompt)
{
	const int32_t name = SparkJsonFindObjectMember(document,object,"name");
	const int32_t tokens = SparkJsonFindObjectMember(document,object,"prompt_token_ids");
	char *text = 0;
	uint32_t index;
	int32_t element;
	if ( name < 0 || tokens < 0 || !SparkJsonTokenIsType(document,tokens,SPARK_JSON_TOKEN_ARRAY) ||
		SparkJsonCopyString(document,name,&text) != SPARK_STATUS_OK || strlen(text) >= SPARK_TOKEN_PROMPT_NAME_BYTES )
	{
		free(text);
		return(SPARK_STATUS_PARSE_ERROR);
	}
	snprintf(prompt->name,sizeof(prompt->name),"%s",text);
	free(text);
	prompt->count = SparkJsonGetArrayElementCount(document,tokens);
	prompt->tokens = (uint32_t *)calloc(prompt->count != 0u ? prompt->count : 1u,sizeof(uint32_t));
	if ( prompt->tokens == 0 || prompt->count == 0u )
		return(SPARK_STATUS_PARSE_ERROR);
	for ( index = 0u, element = SparkJsonGetArrayElementFirst(document,tokens); index < prompt->count;
		index++, element = SparkJsonGetArrayElementNext(document,tokens,element) )
		if ( element < 0 || SparkJsonGetUInt32(document,element,&prompt->tokens[index]) != SPARK_STATUS_OK )
			return(SPARK_STATUS_PARSE_ERROR);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkTokenPromptsLoad(const char *path, SparkTokenPrompts *prompts)
{
	SparkJsonDocument document;
	int32_t root, list, element;
	SparkStatus status;
	uint32_t index;
	memset(prompts,0,sizeof(*prompts));
	SparkJsonDocumentReset(&document);
	status = SparkJsonLoadFile(path,&document);
	if ( status != SPARK_STATUS_OK )
		return(status);
	root = SparkJsonGetRootToken(&document);
	list = root >= 0 ? SparkJsonFindObjectMember(&document,root,"prompts") : -1;
	if ( list < 0 || !SparkJsonTokenIsType(&document,list,SPARK_JSON_TOKEN_ARRAY) )
		status = SPARK_STATUS_PARSE_ERROR;
	else
	{
		prompts->count = SparkJsonGetArrayElementCount(&document,list);
		prompts->prompts = (SparkTokenPrompt *)calloc(prompts->count != 0u ? prompts->count : 1u,sizeof(SparkTokenPrompt));
		status = prompts->prompts != 0 && prompts->count != 0u ? SPARK_STATUS_OK : SPARK_STATUS_PARSE_ERROR;
		for ( index = 0u, element = SparkJsonGetArrayElementFirst(&document,list); status == SPARK_STATUS_OK && index < prompts->count;
			index++, element = SparkJsonGetArrayElementNext(&document,list,element) )
			status = PromptLoadOne(&document,element,&prompts->prompts[index]);
	}
	SparkJsonDocumentDestroy(&document);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"token_prompts: %s is not a {\"prompts\": [{\"name\", \"prompt_token_ids\"}]} file\n",path);
		SparkTokenPromptsFree(prompts);
	}
	return(status);
}

void SparkTokenPromptsFree(SparkTokenPrompts *prompts)
{
	uint32_t index;
	for ( index = 0u; prompts->prompts != 0 && index < prompts->count; index++ )
		free(prompts->prompts[index].tokens);
	free(prompts->prompts);
	memset(prompts,0,sizeof(*prompts));
}
