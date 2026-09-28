#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "sparkpipe/spark_dsv4_parallel_shape.h"
#include "sparkpipe/spark_dsv4_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_model_resident_deployment.h"
#include "sparkpipe/spark_status.h"

typedef struct TestDsv4FlashStageConfig
{
	const char *path;
	uint32_t tp_degree;
	uint32_t pp_stage_count;
} TestDsv4FlashStageConfig;

static const TestDsv4FlashStageConfig TestDsv4FlashStageConfigs[] =
{
	{"examples/deployments/dsv4_flash_tp4_stage.json",4u,1u},
	{"examples/deployments/dsv4_flash_tp4_pp4_stage.json",4u,4u}
};

static int32_t failures;

static void TestDsv4Expect(const char *path,const char *name,uint32_t condition)
{
	if ( condition == 0u )
	{
		fprintf(stderr,"FAIL %s: %s\n",path,name);
		failures++;
	}
}

static void TestDsv4CheckFlashGraphCounts(const TestDsv4FlashStageConfig *stage)
{
	SparkDsv4TpShapeDescriptor shape;
	SparkDsv4TpNodeConfig node;
	SparkJsonDocument document;
	int32_t counts,element;
	uint32_t count,index;
	SparkJsonDocumentReset(&document);
	TestDsv4Expect(stage->path,"stage configuration parses",SparkJsonLoadFile(stage->path,&document) == SPARK_STATUS_OK);
	counts = SparkJsonFindObjectMember(&document,SparkJsonGetRootToken(&document),"cuda_graph_count_by_pp_stage");
	TestDsv4Expect(stage->path,"one graph count per pipeline stage",counts >= 0 && SparkJsonGetArrayElementCount(&document,counts) == stage->pp_stage_count);
	for (index=0u; counts>=0 && index<stage->pp_stage_count && index<SparkJsonGetArrayElementCount(&document,counts); index++)
	{
		memset(&shape,0,sizeof(shape));
		shape.abi_version = SPARK_DSV4_PARALLEL_SHAPE_ABI_VERSION;
		shape.tp_degree = stage->tp_degree;
		shape.pp_stage_count = stage->pp_stage_count;
		shape.pp_stage_index = index;
		element = SparkJsonGetArrayElement(&document,counts,index);
		count = 0u;
		TestDsv4Expect(stage->path,"stage shape derives",SparkDsv4TpDeriveNodeConfig(&shape,&node) == SPARK_STATUS_OK);
		TestDsv4Expect(stage->path,"graph count readable",element >= 0 && SparkJsonGetUInt32(&document,element,&count) == SPARK_STATUS_OK);
		TestDsv4Expect(stage->path,"graph count follows the firmware island law",count == SparkDsv4ResidentDecodeStageGraphIslandsPerSlot(node.layer_count));
	}
	SparkJsonDocumentDestroy(&document);
}

static uint32_t TestDsv4ContractEos(const char *path)
{
	SparkJsonDocument document;
	int32_t eos,model;
	uint32_t value;
	value = UINT32_MAX;
	SparkJsonDocumentReset(&document);
	if ( SparkJsonLoadFile(path,&document) == SPARK_STATUS_OK )
	{
		model = SparkJsonFindObjectMember(&document,SparkJsonGetRootToken(&document),"model");
		eos = model >= 0 ? SparkJsonFindObjectMember(&document,model,"eos_token_id") : -1;
		if ( eos < 0 || SparkJsonGetUInt32(&document,eos,&value) != SPARK_STATUS_OK )
			value = UINT32_MAX;
	}
	SparkJsonDocumentDestroy(&document);
	return(value);
}

static void TestDsv4CheckDeployment(const char *path,const char *contract_path,uint32_t node_count)
{
	SparkModelResidentDeployment deployment;
	uint32_t eos;
	eos = TestDsv4ContractEos(contract_path);
	TestDsv4Expect(contract_path,"model EOS present",eos != UINT32_MAX);
	SparkModelResidentDeploymentReset(&deployment);
	TestDsv4Expect(path,"loads through the residentd deployment parser",SparkModelResidentDeploymentLoad(path,&deployment) == SPARK_STATUS_OK);
	TestDsv4Expect(path,"node count",deployment.node_count == node_count);
	TestDsv4Expect(path,"model EOS from the contract",deployment.eos_token_count == 1u && deployment.eos_token_ids[0] == eos);
	TestDsv4Expect(path,"weightd socket",deployment.weightd_socket_path != 0 && deployment.weightd_socket_path[0] == '/');
	SparkModelResidentDeploymentDestroy(&deployment);
}

int main(void)
{
	uint32_t index;
	for (index=0u; index<sizeof(TestDsv4FlashStageConfigs) / sizeof(TestDsv4FlashStageConfigs[0]); index++)
		TestDsv4CheckFlashGraphCounts(&TestDsv4FlashStageConfigs[index]);
	TestDsv4CheckDeployment("deployment/dsv4_pro_tp4pp4/model_resident.json","model_contracts/dsv4_pro.json",16u);
	TestDsv4CheckDeployment("deployment/dsv41_flash_tp4/model_resident.json","model_contracts/dsv41_flash_authoritative.json",4u);
	if ( failures != 0 )
	{
		printf("test_dsv4_committed_configs: FAIL (%d)\n",(int)failures);
		return(1);
	}
	printf("test_dsv4_committed_configs: PASS\n");
	return(0);
}
