#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "sparkpipe/spark_gpu.h"
#include "runtime/launch.h"

#define CLUSTER_WORDS 8u
#define CLUSTER_MAX_GROUPS 32u
#define CLUSTER_THREADS 64u
#define DEPENDENT_GROUPS 96u
#define DEPENDENT_THREADS 128u
#define DEPENDENT_WORDS (DEPENDENT_GROUPS * DEPENDENT_THREADS)
#define DEPENDENT_SPIN_NS 100000ull
#define SHAPE_WORDS 16u
#define SHAPE_MAX_GROUPS 64u

static int failures;

#define CHECK(condition, what) \
	do { \
		if ( !(condition) ) \
		{ \
			printf("FAIL %s: %s\n",what,SparkGpuGetErrorString(SparkGpuGetLastError())); \
			failures++; \
		} \
	} while (0)

static __device__ void ClusterRecord(uint32_t *out)
{
	__shared__ uint32_t mine;
	uint32_t rank,ranks,cluster,clusters,place,places,peer = 0u;
	asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
	asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(ranks));
	asm volatile("mov.u32 %0, %%clusterid.x;" : "=r"(cluster));
	asm volatile("mov.u32 %0, %%nclusterid.x;" : "=r"(clusters));
	asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(place));
	asm volatile("mov.u32 %0, %%cluster_nctaid.x;" : "=r"(places));
	if ( threadIdx.x == 0u )
		mine = blockIdx.x * 1000u + 17u;
	asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
	if ( threadIdx.x == 0u && ranks > 1u )
	{
		uint32_t local = (uint32_t)__cvta_generic_to_shared(&mine),remote;
		asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(local), "r"(rank ^ 1u));
		asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(peer) : "r"(remote) : "memory");
	}
	asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
	if ( threadIdx.x == 0u )
	{
		uint32_t *row = out + blockIdx.x * CLUSTER_WORDS;
		row[0] = rank;
		row[1] = ranks;
		row[2] = cluster;
		row[3] = clusters;
		row[4] = place;
		row[5] = places;
		row[6] = blockIdx.x;
		row[7] = peer;
	}
}

__global__ void __cluster_dims__(2, 1, 1) ClusterCompiled(uint32_t *out)
{
	ClusterRecord(out);
}

__global__ void ClusterAtLaunch(uint32_t *out)
{
	ClusterRecord(out);
}

static __device__ void ShapeRecord(uint32_t *out)
{
	uint32_t v[SHAPE_WORDS];
	asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(v[0]));
	asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(v[1]));
	asm volatile("mov.u32 %0, %%clusterid.x;" : "=r"(v[2]));
	asm volatile("mov.u32 %0, %%clusterid.y;" : "=r"(v[3]));
	asm volatile("mov.u32 %0, %%clusterid.z;" : "=r"(v[4]));
	asm volatile("mov.u32 %0, %%nclusterid.x;" : "=r"(v[5]));
	asm volatile("mov.u32 %0, %%nclusterid.y;" : "=r"(v[6]));
	asm volatile("mov.u32 %0, %%nclusterid.z;" : "=r"(v[7]));
	asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(v[8]));
	asm volatile("mov.u32 %0, %%cluster_ctaid.y;" : "=r"(v[9]));
	asm volatile("mov.u32 %0, %%cluster_ctaid.z;" : "=r"(v[10]));
	asm volatile("mov.u32 %0, %%cluster_nctaid.x;" : "=r"(v[11]));
	asm volatile("mov.u32 %0, %%cluster_nctaid.y;" : "=r"(v[12]));
	asm volatile("mov.u32 %0, %%cluster_nctaid.z;" : "=r"(v[13]));
	v[14] = blockIdx.x | (blockIdx.y << 10) | (blockIdx.z << 20);
	v[15] = 0x600du;
	if ( threadIdx.x == 0u )
	{
		uint32_t *row = out + (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * SHAPE_WORDS;
		for ( uint32_t index = 0u; index < SHAPE_WORDS; index++ )
			row[index] = v[index];
	}
}

__global__ void __cluster_dims__(2, 2, 2) ShapeCube(uint32_t *out)
{
	ShapeRecord(out);
}

__global__ void __cluster_dims__(1, 3, 2) ShapeColumn(uint32_t *out)
{
	ShapeRecord(out);
}

__global__ void ShapeSingle(uint32_t *out)
{
	ShapeRecord(out);
}

__global__ void DependentFirst(uint32_t *data, unsigned long long spin)
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned long long start,now;
	asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
	do
		asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
	while ( now - start < spin );
	data[index] = index * 2654435761u + 0x9e3779b9u;
}

__global__ void DependentSecond(const uint32_t *data, uint32_t *out)
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	asm volatile("griddepcontrol.wait;" ::: "memory");
	out[index] = (data[index] ^ 0x5a5a5a5au) * 747796405u + 2891336453u;
}

static uint64_t Digest(const void *bytes, uint64_t count)
{
	uint64_t digest = 1469598103934665603ull;
	uint64_t index;
	for ( index = 0u; index < count; index++ )
		digest = (digest ^ ((const uint8_t *)bytes)[index]) * 1099511628211ull;
	return(digest);
}

static void ClusterCheck(const char *what, const uint32_t *rows, uint32_t groups, uint32_t size)
{
	uint32_t group,wrong = 0u;
	for ( group = 0u; group < groups; group++ )
	{
		const uint32_t rank = group % size,first = group - rank;
		const uint32_t expected[CLUSTER_WORDS] = {rank, size, group / size, groups / size, rank, size, group,
			size > 1u ? (first + (rank ^ 1u)) * 1000u + 17u : 0u};
		wrong += memcmp(rows + group * CLUSTER_WORDS,expected,sizeof(expected)) != 0 ? 1u : 0u;
	}
	CHECK(wrong == 0u,what);
	printf("digest %s %016llx\n",what,(unsigned long long)Digest(rows,(uint64_t)groups * CLUSTER_WORDS * sizeof(uint32_t)));
}

static void TestClusters(SparkGpuStream stream, uint32_t *rows, uint32_t *host)
{
	const uint64_t bytes = (uint64_t)CLUSTER_MAX_GROUPS * CLUSTER_WORDS * sizeof(uint32_t);
	CHECK(SparkGpuMemset(rows,0xee,bytes) == SPARK_GPU_SUCCESS,"clear the cluster rows");
	LM_LAUNCH((ClusterCompiled),8u,CLUSTER_THREADS,0u,stream,rows);
	CHECK(SparkGpuGetLastError() == SPARK_GPU_SUCCESS && SparkGpuStreamSynchronize(stream) == SPARK_GPU_SUCCESS &&
		SparkGpuMemcpy(host,rows,bytes,SPARK_GPU_COPY_DEVICE_TO_HOST) == SPARK_GPU_SUCCESS,"launch a kernel compiled for clusters of two");
	ClusterCheck("compiled clusters of 2 in a grid of 8",host,8u,2u);
	CHECK(SparkGpuMemset(rows,0xee,bytes) == SPARK_GPU_SUCCESS,"clear the cluster rows again");
	CHECK(LmLaunchClusterKernel(ClusterAtLaunch,dim3(CLUSTER_MAX_GROUPS),dim3(CLUSTER_THREADS),0u,stream,4u,0u,rows) == SPARK_GPU_SUCCESS &&
		SparkGpuStreamSynchronize(stream) == SPARK_GPU_SUCCESS && SparkGpuMemcpy(host,rows,bytes,SPARK_GPU_COPY_DEVICE_TO_HOST) == SPARK_GPU_SUCCESS,
		"launch clusters of four chosen at launch");
	ClusterCheck("launch clusters of 4 in a grid of 32",host,CLUSTER_MAX_GROUPS,4u);
	CHECK(SparkGpuMemset(rows,0xee,bytes) == SPARK_GPU_SUCCESS,"clear the cluster rows a third time");
	CHECK(LmLaunchClusterKernel(ClusterAtLaunch,dim3(16u),dim3(CLUSTER_THREADS),0u,stream,8u,1u,rows) == SPARK_GPU_SUCCESS &&
		SparkGpuStreamSynchronize(stream) == SPARK_GPU_SUCCESS && SparkGpuMemcpy(host,rows,bytes,SPARK_GPU_COPY_DEVICE_TO_HOST) == SPARK_GPU_SUCCESS,
		"launch dependent clusters of eight");
	ClusterCheck("dependent clusters of 8 in a grid of 16",host,16u,8u);
}

static void ShapeCheck(const char *what, const uint32_t *rows, const uint32_t grid[3], const uint32_t cluster[3])
{
	const uint32_t groups = grid[0] * grid[1] * grid[2];
	uint32_t x,y,z,wrong = 0u;
	for ( z = 0u; z < grid[2]; z++ )
		for ( y = 0u; y < grid[1]; y++ )
			for ( x = 0u; x < grid[0]; x++ )
			{
				const uint32_t *row = rows + (x + grid[0] * (y + grid[1] * z)) * SHAPE_WORDS;
				const uint32_t cx = cluster[0],cy = cluster[1],cz = cluster[2];
				const uint32_t expected[SHAPE_WORDS] = {(x % cx) + cx * ((y % cy) + cy * (z % cz)), cx * cy * cz, x / cx, y / cy, z / cz,
					grid[0] / cx, grid[1] / cy, grid[2] / cz, x % cx, y % cy, z % cz, cx, cy, cz, x | (y << 10) | (z << 20), 0x600du};
				wrong += memcmp(row,expected,sizeof(expected)) != 0 ? 1u : 0u;
			}
	CHECK(wrong == 0u,what);
	printf("digest %s %016llx\n",what,(unsigned long long)Digest(rows,(uint64_t)groups * SHAPE_WORDS * sizeof(uint32_t)));
}

static void TestShapes(SparkGpuStream stream, uint32_t *rows, uint32_t *host)
{
	const uint64_t bytes = (uint64_t)SHAPE_MAX_GROUPS * SHAPE_WORDS * sizeof(uint32_t);
	const uint32_t cube_grid[3] = {4u, 4u, 4u},cube[3] = {2u, 2u, 2u},column_grid[3] = {2u, 6u, 4u},column[3] = {1u, 3u, 2u};
	const uint32_t single_grid[3] = {3u, 2u, 2u},single[3] = {1u, 1u, 1u};
	CHECK(SparkGpuMemset(rows,0xee,bytes) == SPARK_GPU_SUCCESS,"clear the shape rows");
	LM_LAUNCH((ShapeCube),dim3(4u,4u,4u),CLUSTER_THREADS,0u,stream,rows);
	CHECK(SparkGpuGetLastError() == SPARK_GPU_SUCCESS && SparkGpuStreamSynchronize(stream) == SPARK_GPU_SUCCESS &&
		SparkGpuMemcpy(host,rows,bytes,SPARK_GPU_COPY_DEVICE_TO_HOST) == SPARK_GPU_SUCCESS,"launch clusters of 2x2x2 in a grid of 4x4x4");
	ShapeCheck("clusters of 2x2x2 in a grid of 4x4x4",host,cube_grid,cube);
	CHECK(SparkGpuMemset(rows,0xee,bytes) == SPARK_GPU_SUCCESS,"clear the shape rows again");
	LM_LAUNCH((ShapeColumn),dim3(2u,6u,4u),CLUSTER_THREADS,0u,stream,rows);
	CHECK(SparkGpuGetLastError() == SPARK_GPU_SUCCESS && SparkGpuStreamSynchronize(stream) == SPARK_GPU_SUCCESS &&
		SparkGpuMemcpy(host,rows,bytes,SPARK_GPU_COPY_DEVICE_TO_HOST) == SPARK_GPU_SUCCESS,"launch clusters of 1x3x2 in a grid of 2x6x4");
	ShapeCheck("clusters of 1x3x2 in a grid of 2x6x4",host,column_grid,column);
	CHECK(SparkGpuMemset(rows,0xee,bytes) == SPARK_GPU_SUCCESS,"clear the shape rows a third time");
	LM_LAUNCH((ShapeSingle),dim3(3u,2u,2u),CLUSTER_THREADS,0u,stream,rows);
	CHECK(SparkGpuGetLastError() == SPARK_GPU_SUCCESS && SparkGpuStreamSynchronize(stream) == SPARK_GPU_SUCCESS &&
		SparkGpuMemcpy(host,rows,bytes,SPARK_GPU_COPY_DEVICE_TO_HOST) == SPARK_GPU_SUCCESS,"launch a grid of 3x2x2 without clusters");
	ShapeCheck("a grid of 3x2x2 without clusters reads clusters of one block",host,single_grid,single);
}

static void TestDependent(SparkGpuStream stream, uint32_t *data, uint32_t *out, uint32_t *host)
{
	uint32_t round,index,wrong = 0u;
	for ( round = 0u; round < 4u; round++ )
	{
		CHECK(SparkGpuMemset(data,0,DEPENDENT_WORDS * sizeof(uint32_t)) == SPARK_GPU_SUCCESS &&
			SparkGpuMemset(out,0,DEPENDENT_WORDS * sizeof(uint32_t)) == SPARK_GPU_SUCCESS,"clear the dependent buffers");
		CHECK(LM_LAUNCH_DEPENDENT((DependentFirst),DEPENDENT_GROUPS,DEPENDENT_THREADS,0u,stream,data,DEPENDENT_SPIN_NS) == SPARK_GPU_SUCCESS &&
			LM_LAUNCH_DEPENDENT((DependentSecond),DEPENDENT_GROUPS,DEPENDENT_THREADS,0u,stream,data,out) == SPARK_GPU_SUCCESS,
			"launch a writer and a dependent reader");
		CHECK(SparkGpuStreamSynchronize(stream) == SPARK_GPU_SUCCESS &&
			SparkGpuMemcpy(host,out,DEPENDENT_WORDS * sizeof(uint32_t),SPARK_GPU_COPY_DEVICE_TO_HOST) == SPARK_GPU_SUCCESS,"read the reader's output");
		for ( index = 0u; index < DEPENDENT_WORDS; index++ )
			wrong += host[index] != (((index * 2654435761u + 0x9e3779b9u) ^ 0x5a5a5a5au) * 747796405u + 2891336453u) ? 1u : 0u;
	}
	CHECK(wrong == 0u,"every dependent reader passes its wait only after the writer's stores");
	printf("digest dependent reader %016llx\n",(unsigned long long)Digest(host,DEPENDENT_WORDS * sizeof(uint32_t)));
}

#ifdef SPARKPIPE_DEVICE_NVRM
static void TestRefusals(SparkGpuStream stream, uint32_t *rows)
{
	(void)SparkGpuGetLastError();
	CHECK(LmLaunchClusterKernel(ClusterCompiled,dim3(8u),dim3(CLUSTER_THREADS),0u,stream,4u,0u,rows) == SPARK_GPU_ERROR_INVALID_VALUE &&
		strstr(SparkDeviceLastError()->text,"compiled for clusters of 2x1x1") != 0,"a kernel compiled for clusters of two refuses clusters of four");
	CHECK(LmLaunchClusterKernel(ClusterAtLaunch,dim3(6u),dim3(CLUSTER_THREADS),0u,stream,4u,0u,rows) == SPARK_GPU_ERROR_INVALID_VALUE &&
		strstr(SparkDeviceLastError()->text,"whole number of clusters") != 0,"a grid that is not a whole number of clusters is refused");
	CHECK(LmLaunchClusterKernel(ClusterAtLaunch,dim3(CLUSTER_MAX_GROUPS),dim3(CLUSTER_THREADS),0u,stream,16u,0u,rows) == SPARK_GPU_ERROR_INVALID_VALUE &&
		strstr(SparkDeviceLastError()->text,"over the limit") != 0,"a cluster over eight blocks is refused");
	CHECK(LmLaunchClusterKernel(ClusterAtLaunch,dim3(8u),dim3(CLUSTER_THREADS),0u,stream,0u,0u,rows) == SPARK_GPU_ERROR_INVALID_VALUE,
		"a cluster of no blocks is refused");
	(void)SparkGpuGetLastError();
}
#endif

int main(int argc, char **argv)
{
	SparkGpuStream stream = 0;
	uint32_t *rows = 0,*data = 0,*out = 0,*shapes = 0;
	static uint32_t host[DEPENDENT_WORDS];
	const char *flavour = argc > 1 ? argv[1] : "this flavour";
	if ( SparkGpuStreamCreateWithFlags(&stream,SPARK_GPU_STREAM_NON_BLOCKING) != SPARK_GPU_SUCCESS ||
		SparkGpuMalloc(&rows,(uint64_t)CLUSTER_MAX_GROUPS * CLUSTER_WORDS * sizeof(uint32_t)) != SPARK_GPU_SUCCESS ||
		SparkGpuMalloc(&shapes,(uint64_t)SHAPE_MAX_GROUPS * SHAPE_WORDS * sizeof(uint32_t)) != SPARK_GPU_SUCCESS ||
		SparkGpuMalloc(&data,DEPENDENT_WORDS * sizeof(uint32_t)) != SPARK_GPU_SUCCESS || SparkGpuMalloc(&out,DEPENDENT_WORDS * sizeof(uint32_t)) != SPARK_GPU_SUCCESS )
	{
		printf("FAIL set up the stream and buffers: %s\n",SparkGpuGetErrorString(SparkGpuGetLastError()));
		return(1);
	}
	TestClusters(stream,rows,host);
	TestShapes(stream,shapes,host);
	TestDependent(stream,data,out,host);
#ifdef SPARKPIPE_DEVICE_NVRM
	TestRefusals(stream,rows);
#endif
	CHECK(SparkGpuFree(rows) == SPARK_GPU_SUCCESS && SparkGpuFree(shapes) == SPARK_GPU_SUCCESS && SparkGpuFree(data) == SPARK_GPU_SUCCESS && SparkGpuFree(out) == SPARK_GPU_SUCCESS &&
		SparkGpuStreamDestroy(stream) == SPARK_GPU_SUCCESS,"release the buffers and the stream");
	if ( failures != 0 )
	{
		printf("\nFAIL cluster launch %s (%d)\n",flavour,failures);
		return(1);
	}
	printf("PASS cluster launch %s: compiled and launch-time clusters run with CUDA's cluster numbering and shared memory across the cluster, "
		"and dependent readers see their writers' stores\n",flavour);
	return(0);
}
