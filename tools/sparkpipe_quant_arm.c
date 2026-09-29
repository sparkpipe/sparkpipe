#include <stdio.h>
#include <string.h>

#include "sparkpipe/spark_quant_arm.h"

static int SparkQuantArmToolUsage(const char *program)
{
	fprintf(stderr,"usage: %s (--digest | --canonical | --summary) ARM_JSON\n",program);
	return(2);
}

int main(int argc,char **argv)
{
	static SparkQuantArm arm;
	static char canonical[SPARK_QUANT_ARM_CANONICAL_BYTES];
	char error[1024];
	uint32_t canonical_bytes = 0u;
	if ( argc != 3 || (strcmp(argv[1],"--digest") != 0 && strcmp(argv[1],"--canonical") != 0 && strcmp(argv[1],"--summary") != 0) )
		return(SparkQuantArmToolUsage(argv[0]));
	if ( SparkQuantArmLoadFile(argv[2],&arm,error,(uint32_t)sizeof(error)) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"sparkpipe_quant_arm: REFUSED %s: %s\n",argv[2],error);
		return(1);
	}
	if ( strcmp(argv[1],"--digest") == 0 )
		printf("%s\n",arm.arm_digest);
	else if ( strcmp(argv[1],"--canonical") == 0 )
	{
		if ( SparkQuantArmCanonicalize(&arm,canonical,(uint32_t)sizeof(canonical),&canonical_bytes) != SPARK_STATUS_OK )
			return(1);
		fwrite(canonical,1u,canonical_bytes,stdout);
	}
	else
		printf("{\"arm_id\":\"%s\",\"arm_digest\":\"%s\",\"pack_set_sha256\":\"%s\",\"arm_kv\":\"%s\",\"ranks\":%u}\n",arm.arm_id,arm.arm_digest,arm.pack_set_sha256,arm.kv_text,arm.rank_count);
	return(fflush(stdout) == 0 ? 0 : 1);
}
