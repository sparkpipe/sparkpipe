#include <stdint.h>

#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_minimax_h3_resident_media_stage_firmware.h"

#ifndef MINIMAX_H3_MODEL_REVISION
#define MINIMAX_H3_MODEL_REVISION "unpinned"
#endif
#ifndef MINIMAX_H3_DIFFUSERS_COMMIT
#define MINIMAX_H3_DIFFUSERS_COMMIT "unpinned"
#endif
#ifndef MINIMAX_H3_CONTRACT_SHA256
#define MINIMAX_H3_CONTRACT_SHA256 "unpinned"
#endif

const char *SparkMinimaxH3ResidentMediaStageModelRevision(void)
{
	return(MINIMAX_H3_MODEL_REVISION);
}

const char *SparkMinimaxH3ResidentMediaStageDiffusersCommit(void)
{
	return(MINIMAX_H3_DIFFUSERS_COMMIT);
}

const char *SparkMinimaxH3ResidentMediaStageContractDigest(void)
{
	return(MINIMAX_H3_CONTRACT_SHA256);
}

uint32_t SparkMinimaxH3ResidentMediaStageRankCount(void)
{
	return(SPARK_MINIMAX_H3_RESIDENT_MEDIA_STAGE_RANK_COUNT);
}

int32_t SparkMinimaxH3ResidentMediaStageStageBlockCount(uint32_t stage_index,
	uint32_t *block_count_out)
{
	if ( block_count_out == 0 )
		SPARK_FAIL(-108);
	if ( stage_index >= SPARK_MINIMAX_H3_RESIDENT_MEDIA_STAGE_PP_STAGE_COUNT )
		SPARK_FAIL(-107);
	if ( stage_index < 3u )
		*block_count_out = SPARK_MINIMAX_H3_RESIDENT_MEDIA_STAGE_DIT_BLOCKS_PER_STAGE;
	else
		*block_count_out = SPARK_MINIMAX_H3_RESIDENT_MEDIA_STAGE_DIT_TAIL_BLOCKS;
	return(0);
}
