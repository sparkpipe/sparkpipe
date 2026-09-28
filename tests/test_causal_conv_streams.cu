#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "inference/kernels/linear_attn.cuh"

#define CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { fprintf(stderr,"FAIL line=%d cuda=%s call=%s\n",__LINE__,cudaGetErrorString(e),#call); exit(1); } } while(0)
#define REQUIRE(test) do { if(!(test)) { fprintf(stderr,"FAIL line=%d test=%s\n",__LINE__,#test); exit(1); } } while(0)

#define CONV_THREADS 256u
#define CONV_KERNEL 4u

static uint32_t random_state = 20260928u;

static uint32_t Random()
{
	random_state ^= random_state << 13u;
	random_state ^= random_state >> 17u;
	random_state ^= random_state << 5u;
	return random_state;
}

static uint16_t Bf16(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,sizeof(bits));
	return (uint16_t)((bits+0x7fffu+((bits>>16u)&1u))>>16u);
}

static std::vector<uint16_t> Random16(size_t count,float amplitude)
{
	std::vector<uint16_t> values(count);
	for (auto &value : values) value=Bf16(amplitude*(((int32_t)(Random()%2049u)-1024)/1024.0f));
	return values;
}

template<class T> static T *Upload(const std::vector<T> &host)
{
	T *device;
	CUDA(cudaMalloc(&device,host.size()*sizeof(T)));
	CUDA(cudaMemcpy(device,host.data(),host.size()*sizeof(T),cudaMemcpyHostToDevice));
	return device;
}

template<class T> static std::vector<T> Download(const T *device,size_t count)
{
	std::vector<T> host(count);
	CUDA(cudaMemcpy(host.data(),device,count*sizeof(T),cudaMemcpyDeviceToHost));
	return host;
}

static void Case(uint32_t sequences,uint32_t rows_per_sequence,const uint32_t *channels,uint32_t commit,cudaStream_t stream)
{
	const uint32_t rows=sequences*rows_per_sequence;
	std::vector<uint32_t> index(sequences),begin(sequences+1u);
	std::vector<uint16_t> window[LM_CAUSAL_CONV_STREAMS],input[LM_CAUSAL_CONV_STREAMS],weight[LM_CAUSAL_CONV_STREAMS];
	uint16_t *device_window[2][LM_CAUSAL_CONV_STREAMS],*device_data[2][LM_CAUSAL_CONV_STREAMS],*device_weight[LM_CAUSAL_CONV_STREAMS];
	uint32_t *device_index,*device_begin,stream_index,variant,widest=0u;
	LmCausalConvStreams<uint16_t> fused;
	for (uint32_t sequence=0u; sequence<sequences; sequence++) { index[sequence]=(sequence*5u+1u)%sequences; begin[sequence]=sequence*rows_per_sequence; }
	begin[sequences]=rows;
	device_index=Upload(index); device_begin=Upload(begin);
	for (stream_index=0u; stream_index<LM_CAUSAL_CONV_STREAMS; stream_index++)
	{
		window[stream_index]=Random16((size_t)sequences*channels[stream_index]*CONV_KERNEL,2.0f);
		input[stream_index]=Random16((size_t)rows*channels[stream_index],3.0f);
		weight[stream_index]=Random16((size_t)channels[stream_index]*CONV_KERNEL,1.0f);
		device_weight[stream_index]=Upload(weight[stream_index]);
		for (variant=0u; variant<2u; variant++)
		{
			device_window[variant][stream_index]=Upload(window[stream_index]);
			device_data[variant][stream_index]=Upload(input[stream_index]);
		}
		widest=channels[stream_index] > widest ? channels[stream_index] : widest;
	}
	for (stream_index=0u; stream_index<LM_CAUSAL_CONV_STREAMS; stream_index++)
		LmCausalConvKernel<CONV_THREADS,CONV_KERNEL,LM_CONV_SWISH,uint16_t><<<dim3(sequences,(channels[stream_index]+CONV_THREADS-1u)/CONV_THREADS),CONV_THREADS,0,stream>>>(device_window[0][stream_index],device_index,device_begin,0,device_data[0][stream_index],device_weight[stream_index],device_data[0][stream_index],channels[stream_index],sequences,commit,0);
	for (stream_index=0u; stream_index<LM_CAUSAL_CONV_STREAMS; stream_index++)
	{
		fused.window[stream_index]=device_window[1][stream_index];
		fused.input_bf16[stream_index]=fused.output_bf16[stream_index]=device_data[1][stream_index];
		fused.weight[stream_index]=device_weight[stream_index];
		fused.channels[stream_index]=channels[stream_index];
	}
	LmCausalConvStreamsKernel<CONV_THREADS,CONV_KERNEL,LM_CONV_SWISH,uint16_t><<<dim3(sequences,(widest+CONV_THREADS-1u)/CONV_THREADS,LM_CAUSAL_CONV_STREAMS),CONV_THREADS,0,stream>>>(fused,device_index,device_begin,0,sequences,commit,0);
	CUDA(cudaPeekAtLastError());
	CUDA(cudaStreamSynchronize(stream));
	for (stream_index=0u; stream_index<LM_CAUSAL_CONV_STREAMS; stream_index++)
	{
		REQUIRE(Download(device_data[0][stream_index],input[stream_index].size()) == Download(device_data[1][stream_index],input[stream_index].size()));
		REQUIRE(Download(device_window[0][stream_index],window[stream_index].size()) == Download(device_window[1][stream_index],window[stream_index].size()));
		if (commit == 0u)
			REQUIRE(Download(device_window[1][stream_index],window[stream_index].size()) == window[stream_index]);
		for (variant=0u; variant<2u; variant++) { CUDA(cudaFree(device_window[variant][stream_index])); CUDA(cudaFree(device_data[variant][stream_index])); }
		CUDA(cudaFree(device_weight[stream_index]));
	}
	CUDA(cudaFree(device_index)); CUDA(cudaFree(device_begin));
	printf("PASS sequences=%u rows_per_sequence=%u channels=%u,%u,%u commit=%u\n",sequences,rows_per_sequence,channels[0],channels[1],channels[2],commit);
}

int main(int argc,char **argv)
{
	static const uint32_t glm_shape[3]={512u,512u,512u},uneven[3]={300u,512u,1000u},narrow[3]={96u,64u,33u};
	cudaStream_t stream;
	if (argc < 2 || strcmp(argv[1],"--run") != 0)
	{
		fprintf(stderr,"usage: test_causal_conv_streams --run\n");
		return 2;
	}
	CUDA(cudaStreamCreate(&stream));
	Case(1u,1u,glm_shape,1u,stream);
	Case(8u,1u,glm_shape,1u,stream);
	Case(3u,4u,uneven,1u,stream);
	Case(2u,2u,narrow,0u,stream);
	printf("test_causal_conv_streams PASS\n");
	return 0;
}
