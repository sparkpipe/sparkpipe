#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_speculation_verify_plan.h"

static void Require(int condition,const char *what)
{
	if ( condition )
		return;
	fprintf(stderr,"FAIL %s\n",what);
	exit(1);
}

static void TestRows(void)
{
	uint32_t rows = 99u;
	Require(SparkSpeculationVerifyRowsParse(0,8u,&rows) == SPARK_STATUS_OK && rows == 0u,"unset rows mean off");
	Require(SparkSpeculationVerifyRowsParse("0",8u,&rows) == SPARK_STATUS_OK && rows == 0u,"0 means off");
	Require(SparkSpeculationVerifyRowsParse("2",8u,&rows) == SPARK_STATUS_OK && rows == 2u,"2 rows");
	Require(SparkSpeculationVerifyRowsParse("8",8u,&rows) == SPARK_STATUS_OK && rows == 8u,"limit rows");
	Require(SparkSpeculationVerifyRowsParse("8",4u,&rows) == SPARK_STATUS_INVALID_ARGUMENT && rows == 0u,"above the exact row limit");
	Require(SparkSpeculationVerifyRowsParse("1",8u,&rows) == SPARK_STATUS_INVALID_ARGUMENT,"one row is not a verify wave");
	Require(SparkSpeculationVerifyRowsParse("",8u,&rows) == SPARK_STATUS_INVALID_ARGUMENT,"empty");
	Require(SparkSpeculationVerifyRowsParse("4x",8u,&rows) == SPARK_STATUS_INVALID_ARGUMENT,"trailing text");
	Require(SparkSpeculationVerifyRowsParse("x",8u,&rows) == SPARK_STATUS_INVALID_ARGUMENT,"not a digit");
}

static void TestDrafter(void)
{
	uint32_t kind = 99u;
	const char *path = "x";
	Require(SparkSpeculationVerifyDrafterParse(0,0u,&kind,&path) == SPARK_STATUS_OK && kind == SPARK_SPECULATION_VERIFY_DRAFTER_NONE && path == 0,"off needs no drafter");
	Require(SparkSpeculationVerifyDrafterParse("lookup",0u,&kind,&path) == SPARK_STATUS_INVALID_ARGUMENT,"a drafter without rows is refused");
	Require(SparkSpeculationVerifyDrafterParse(0,4u,&kind,&path) == SPARK_STATUS_INVALID_ARGUMENT,"rows without a drafter are refused");
	Require(SparkSpeculationVerifyDrafterParse("lookup",4u,&kind,&path) == SPARK_STATUS_OK && kind == SPARK_SPECULATION_VERIFY_DRAFTER_LOOKUP && path == 0,"lookup");
	Require(SparkSpeculationVerifyDrafterParse("oracle:/a/b",4u,&kind,&path) == SPARK_STATUS_OK && kind == SPARK_SPECULATION_VERIFY_DRAFTER_ORACLE && strcmp(path,"/a/b") == 0,"oracle path");
	Require(SparkSpeculationVerifyDrafterParse("adversary:/c",4u,&kind,&path) == SPARK_STATUS_OK && kind == SPARK_SPECULATION_VERIFY_DRAFTER_ADVERSARY && strcmp(path,"/c") == 0,"adversary path");
	Require(SparkSpeculationVerifyDrafterParse("recorded:t.sprd",4u,&kind,&path) == SPARK_STATUS_OK && kind == SPARK_SPECULATION_VERIFY_DRAFTER_RECORDED && strcmp(path,"t.sprd") == 0,"recorded path");
	Require(SparkSpeculationVerifyDrafterParse("oracle:",4u,&kind,&path) == SPARK_STATUS_INVALID_ARGUMENT,"oracle without a path");
	Require(SparkSpeculationVerifyDrafterParse("recorded:",4u,&kind,&path) == SPARK_STATUS_INVALID_ARGUMENT,"recorded without a path");
	Require(SparkSpeculationVerifyDrafterParse("mtp",4u,&kind,&path) == SPARK_STATUS_INVALID_ARGUMENT,"unknown drafter");
	Require(SparkSpeculationVerifyDrafterParse("lookup ",4u,&kind,&path) == SPARK_STATUS_INVALID_ARGUMENT,"trailing space");
}

static void TestDepth(void)
{
	Require(SparkSpeculationVerifyDepth(8u,0u,8u,8u) == 7u,"full frame, full rows");
	Require(SparkSpeculationVerifyDepth(8u,0u,4u,8u) == 3u,"row cap");
	Require(SparkSpeculationVerifyDepth(8u,5u,8u,8u) == 2u,"budget left 3 gives depth 2");
	Require(SparkSpeculationVerifyDepth(8u,6u,8u,8u) == 1u,"budget left 2 gives depth 1");
	Require(SparkSpeculationVerifyDepth(8u,7u,8u,8u) == 0u,"budget left 1 is a plain step");
	Require(SparkSpeculationVerifyDepth(8u,8u,8u,8u) == 0u,"budget spent");
	Require(SparkSpeculationVerifyDepth(8u,0u,8u,3u) == 2u,"regime fit");
	Require(SparkSpeculationVerifyDepth(8u,0u,8u,1u) == 0u,"no room past the regime boundary");
	Require(SparkSpeculationVerifyDepth(8u,0u,0u,8u) == 0u,"verify off");
	Require(SparkSpeculationVerifyDepth(8u,0u,1u,8u) == 0u,"one row is off");
}

int main(void)
{
	TestRows();
	TestDrafter();
	TestDepth();
	printf("PASS speculation verify plan: rows, drafter kinds, depth under budget, row cap and regime fit\n");
	return(0);
}
