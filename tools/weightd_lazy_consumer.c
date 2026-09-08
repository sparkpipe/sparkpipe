// Synthetic fixture probe, not a model pack producer. Failure terminates the
// consumer process; never release an uncertain GPU lease merely to clean up.
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <poll.h>
#include <unistd.h>
#include <cuda_runtime_api.h>
#include "sparkpipe/spark_weightd_lazy_pack.h"
#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_ck128.h"

#define CHUNK UINT64_C(2097152)
#define PACK_BYTES (4u * CHUNK)
#define TIMEOUT UINT64_C(10000000000)

static int32_t prepare(const char *path)
{
	uint8_t data[4096],record[48];
	uint32_t header[4] = {SPARK_WEIGHTD_EXPERT_MANIFEST_MAGIC,2u,8u,0u};
	uint32_t expert,plane,kind,i;
	uint64_t offset,bytes = sizeof(data);
	SparkCk128Context hash;
	FILE *pack,*manifest;
	char sidecar[SPARK_WEIGHTD_PATH_BYTES + 8u];
	if ( snprintf(sidecar,sizeof(sidecar),"%s.experts",path) >= (int32_t)sizeof(sidecar) ) return(-1);
	pack = fopen(path,"wbx");
	if ( pack == 0 ) return(-2);
	manifest = fopen(sidecar,"wbx");
	if ( manifest == 0 ) { fclose(pack); return(-3); }
	if ( fwrite(header,1u,sizeof(header),manifest) != sizeof(header) ) return(-4);
	for (expert=0u; expert<4u; expert++)
	{
		memset(data,(int32_t)expert + 1,sizeof(data));
		for (i=0u; i<CHUNK / sizeof(data); i++)
			if ( fwrite(data,1u,sizeof(data),pack) != sizeof(data) ) return(-5);
		for (plane=0u; plane<2u; plane++)
		{
			memset(record,0,sizeof(record));
			kind = plane;
			offset = ((uint64_t)expert * CHUNK + plane * sizeof(data));
			memcpy(record + 4u,&expert,4u);
			memcpy(record + 8u,&kind,4u);
			memcpy(record + 16u,&offset,8u);
			memcpy(record + 24u,&bytes,8u);
			SparkCk128Initialize(&hash);
			SparkCk128Update(&hash,data,sizeof(data));
			SparkCk128Finalize(&hash,record + 32u);
			if ( fwrite(record,1u,sizeof(record),manifest) != sizeof(record) ) return(-6);
		}
	}
	if ( fclose(pack) != 0 ) return(-7);
	return(fclose(manifest) == 0 ? 0 : -8);
}

static int32_t barrier(void)
{
	struct pollfd input = {STDIN_FILENO,POLLIN,0};
	uint8_t go;
	puts("READY");
	fflush(stdout);
	if ( poll(&input,1u,10000) != 1 || (input.revents & POLLIN) == 0 ) return(-9);
	return(read(STDIN_FILENO,&go,1u) == 1 && go == 71u ? 0 : -10);
}

static int32_t read_leased(SparkWeightdLazyPack *pack,uint32_t first)
{
	SparkWeightdExpertKey keys[2] = {{0u,first},{0u,first + 1u}};
	uint64_t lease = 0u;
	void *address = 0;
	uint8_t output[2][8192];
	uint32_t expert,index;
	SparkStatus status;
	status = SparkWeightdMapAcquire(pack->map,keys,2u,&lease,TIMEOUT);
	if ( status != SPARK_STATUS_OK ) return(-11);
	if ( SparkWeightdMapBeginUse(pack->map,lease,&address) != SPARK_STATUS_OK ) return(-12);
	for (expert=0u; expert<2u; expert++)
		if ( cudaMemcpyAsync(output[expert],(uint8_t *)address + ((first + expert) * CHUNK),sizeof(output[expert]),cudaMemcpyDeviceToHost,0) != cudaSuccess ) return(-13);
	if ( SparkWeightdMapRecordCompletion(pack->map,lease,0) != SPARK_STATUS_OK ) return(-14);
	if ( barrier() != 0 ) return(-15);
	if ( cudaStreamSynchronize(0) != cudaSuccess ) return(-16);
	for (expert=0u; expert<2u; expert++)
		for (index=0u; index<sizeof(output[expert]); index++)
			if ( output[expert][index] != (uint8_t)(first + expert + 1u) ) return(-17);
	if ( SparkWeightdMapRelease(pack->map,lease,TIMEOUT) != SPARK_STATUS_OK ) return(-18);
	return(0);
}

static int32_t reject_pinned(SparkWeightdLazyPack *pack)
{
	SparkWeightdExpertKey key = {0u,3u};
	uint64_t lease = 0u;
	SparkStatus status;
	status = SparkWeightdMapAcquire(pack->map,&key,1u,&lease,TIMEOUT);
	if ( lease != 0u && SparkWeightdMapRelease(pack->map,lease,TIMEOUT) != SPARK_STATUS_OK ) return(-25);
	if ( status != SPARK_STATUS_CAPACITY_EXCEEDED ) return(-26);
	return(0);
}

static int32_t consume(const char *socket,const char *path,uint32_t first)
{
	SparkWeightdLazyAttachRequest request = {0};
	SparkWeightdLazyPack *pack = 0;
	int32_t err;
	if ( strlen(path) >= sizeof(request.pack_path) || first > 3u ) return(-19);
	if ( cudaFree(0) != cudaSuccess ) return(-20);
	request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
	request.identity.arena_bytes = PACK_BYTES;
	memcpy(request.identity.model,"lazy-consumer-probe",20u);
	memcpy(request.pack_path,path,strlen(path) + 1u);
	request.expert_pool_bytes = (3u * CHUNK);
	if ( SparkSha256File(path,request.identity.pack_sha256) != SPARK_STATUS_OK ) return(-21);
	if ( SparkWeightdLazyPackCreate(socket,&request,PACK_BYTES + 255u,TIMEOUT,&pack) != SPARK_STATUS_OK ) return(-22);
	if ( pack->attached.chunk_bytes != CHUNK ) return(-23);
	err = first == 3u ? reject_pinned(pack) : read_leased(pack,first);
	if ( err != 0 ) return(err);
	if ( SparkWeightdLazyPackDestroy(pack) != SPARK_STATUS_OK ) return(-24);
	puts(first == 3u ? "PASS pinned pool rejects fourth chunk" : "PASS consumer-local lazy reads");
	return(0);
}

int main(int argc,char **argv)
{
	int32_t err;
	if ( argc == 3 && strcmp(argv[1],"prepare") == 0 ) err = prepare(argv[2]);
	else if ( argc == 5 && strcmp(argv[3],"consumer") == 0 && strcmp(argv[4],"0") == 0 ) err = consume(argv[1],argv[2],0u);
	else if ( argc == 5 && strcmp(argv[3],"consumer") == 0 && strcmp(argv[4],"1") == 0 ) err = consume(argv[1],argv[2],1u);
	else if ( argc == 5 && strcmp(argv[3],"consumer") == 0 && strcmp(argv[4],"2") == 0 ) err = consume(argv[1],argv[2],2u);
	else if ( argc == 4 && strcmp(argv[3],"pressure") == 0 ) err = consume(argv[1],argv[2],3u);
	else { fprintf(stderr,"usage: %s prepare PACK | SOCKET PACK consumer 0|1|2 | SOCKET PACK pressure\n",argv[0]); return(2); }
	if ( err != 0 ) fprintf(stderr,"lazy probe failed: %d\n",err);
	return(err == 0 ? 0 : 1);
}
