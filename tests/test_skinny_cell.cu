#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "inference/kernels/skinny.cuh"

#define CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { fprintf(stderr,"FAIL line=%d cuda=%s call=%s\n",__LINE__,cudaGetErrorString(e),#call); exit(1); } } while(0)
#define REQUIRE(test) do { if(!(test)) { fprintf(stderr,"FAIL line=%d test=%s\n",__LINE__,#test); exit(1); } } while(0)

static uint32_t random_state = 20261007u;

static uint32_t Random()
{
	random_state ^= random_state << 13u;
	random_state ^= random_state >> 17u;
	random_state ^= random_state << 5u;
	return random_state;
}

static double E2m1(uint32_t nibble)
{
	static const double magnitude[8] = {0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0};
	return (nibble & 8u) != 0u ? -magnitude[nibble & 7u] : magnitude[nibble & 7u];
}

static float Bf16ToFloat(uint16_t value)
{
	uint32_t bits = (uint32_t)value << 16u;
	float result;
	memcpy(&result,&bits,sizeof(result));
	return result;
}

static uint16_t FloatToBf16(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,sizeof(bits));
	return (uint16_t)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
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

struct Experts
{
	uint32_t groups,output,input;
	std::vector<uint8_t> payload,scale,cells;
};

static Experts MakeExperts(uint32_t groups,uint32_t output,uint32_t input)
{
	Experts experts;
	const uint32_t tiles = input / 32u, cells = output / 16u;
	experts.groups = groups;
	experts.output = output;
	experts.input = input;
	experts.payload.resize((size_t)groups * output * input / 2u);
	experts.scale.resize((size_t)groups * output * tiles);
	for (size_t i = 0u; i < experts.payload.size(); i++) experts.payload[i] = (uint8_t)Random();
	for (size_t i = 0u; i < experts.scale.size(); i++) experts.scale[i] = (uint8_t)(120u + Random() % 12u);
	experts.cells.assign((size_t)groups * tiles * cells * 17u * 16u,0u);
	for (uint32_t e = 0u; e < groups; e++)
		for (uint32_t t = 0u; t < tiles; t++)
			for (uint32_t c = 0u; c < cells; c++)
			{
				uint8_t *row = experts.cells.data() + (((size_t)e * tiles * cells + (size_t)t * cells + c) * 17u) * 16u;
				for (uint32_t r = 0u; r < 16u; r++)
				{
					const uint32_t neuron = c * 16u + r;
					memcpy(row + r * 16u,experts.payload.data() + ((size_t)e * output + neuron) * (input / 2u) + t * 16u,16u);
					row[16u * 16u + r] = experts.scale[((size_t)e * output + neuron) * tiles + t];
				}
			}
	return experts;
}

static double Reference(const Experts &experts,uint32_t group,uint32_t neuron,const uint16_t *activation,double *magnitude)
{
	const uint32_t tiles = experts.input / 32u;
	double sum = 0.0;
	*magnitude = 0.0;
	for (uint32_t k = 0u; k < experts.input; k++)
	{
		const uint8_t byte = experts.payload[((size_t)group * experts.output + neuron) * (experts.input / 2u) + k / 2u];
		const double weight = E2m1((k & 1u) != 0u ? byte >> 4u : byte & 15u) *
			ldexp(1.0,(int)experts.scale[((size_t)group * experts.output + neuron) * tiles + k / 32u] - 127);
		const double term = weight * (double)Bf16ToFloat(activation[k]);
		sum += term;
		*magnitude += fabs(term);
	}
	return sum;
}

struct Routing
{
	std::vector<uint32_t> offset,source;
	uint32_t group_rows_max;
};

static Routing MakeRouting(const std::vector<uint32_t> &group_of_row,const std::vector<uint32_t> &source_of_row,uint32_t groups)
{
	Routing routing;
	std::vector<uint32_t> fill;
	routing.offset.assign(groups + 1u,0u);
	routing.group_rows_max = 0u;
	for (uint32_t row = 0u; row < group_of_row.size(); row++) routing.offset[group_of_row[row] + 1u]++;
	for (uint32_t g = 0u; g < groups; g++)
	{
		if (routing.offset[g + 1u] > routing.group_rows_max) routing.group_rows_max = routing.offset[g + 1u];
		routing.offset[g + 1u] += routing.offset[g];
	}
	fill.assign(routing.offset.begin(),routing.offset.end() - 1);
	routing.source.resize(group_of_row.size());
	for (uint32_t row = 0u; row < group_of_row.size(); row++) routing.source[fill[group_of_row[row]]++] = source_of_row[row];
	return routing;
}

static std::vector<uint16_t> Run(const Experts &experts,const uint8_t *weight,const std::vector<uint16_t> &activation,const Routing &routing,uint32_t activation_packed,uint32_t group_rows_max,cudaStream_t stream)
{
	const uint32_t packed = (uint32_t)routing.source.size();
	uint16_t *activation_device = Upload(activation), *output;
	uint32_t *offset = Upload(routing.offset), *source = Upload(routing.source);
	CUDA(cudaMalloc(&output,(size_t)packed * experts.output * sizeof(uint16_t)));
	CUDA(cudaMemset(output,0xff,(size_t)packed * experts.output * sizeof(uint16_t)));
	REQUIRE(LmSkinnyCellExperts(weight,activation_device,output,offset,activation_packed != 0u ? (const uint32_t *)0 : source,
		experts.groups,packed,group_rows_max,activation_packed,experts.input,experts.output,32u,stream) == LM_LAUNCH_OK);
	CUDA(cudaStreamSynchronize(stream));
	std::vector<uint16_t> result = Download(output,(size_t)packed * experts.output);
	CUDA(cudaFree(output));
	CUDA(cudaFree(activation_device));
	CUDA(cudaFree(offset));
	CUDA(cudaFree(source));
	return result;
}

static void CheckAgainstReference(const Experts &experts,const std::vector<uint16_t> &output,const std::vector<uint16_t> &activation,const Routing &routing,uint32_t activation_packed)
{
	double worst = 0.0;
	for (uint32_t g = 0u; g < experts.groups; g++)
		for (uint32_t row = routing.offset[g]; row < routing.offset[g + 1u]; row++)
		{
			const uint16_t *input = activation.data() + (size_t)(activation_packed != 0u ? row : routing.source[row]) * experts.input;
			for (uint32_t n = 0u; n < experts.output; n++)
			{
				double magnitude;
				const double expected = Reference(experts,g,n,input,&magnitude);
				const double got = (double)Bf16ToFloat(output[(size_t)row * experts.output + n]);
				const double error = fabs(got - expected) / (fabs(expected) * 0.0079 + magnitude * 1e-5 + 1e-30);
				if (error > worst) worst = error;
				if (error > 1.0)
				{
					fprintf(stderr,"FAIL group=%u row=%u neuron=%u got=%.8g expected=%.8g magnitude=%.8g\n",g,row,n,got,expected,magnitude);
					exit(1);
				}
			}
		}
	printf("  reference: worst error %.3f of the allowed bound\n",worst);
}

static void Case(uint32_t groups,uint32_t output,uint32_t input,uint32_t tokens,uint32_t top_k,uint32_t activation_packed,cudaStream_t stream)
{
	Experts experts = MakeExperts(groups,output,input);
	uint8_t *weight = Upload(experts.cells);
	std::vector<uint32_t> group_of_row,source_of_row;
	for (uint32_t token = 0u; token < tokens; token++)
		for (uint32_t k = 0u; k < top_k; k++)
		{
			uint32_t g;
			bool fresh;
			do
			{
				g = Random() % groups;
				fresh = true;
				for (uint32_t other = token * top_k; other < group_of_row.size(); other++) fresh = fresh && group_of_row[other] != g;
			} while (!fresh);
			group_of_row.push_back(g);
			source_of_row.push_back(token);
		}
	Routing routing = MakeRouting(group_of_row,source_of_row,groups);
	const uint32_t packed = (uint32_t)group_of_row.size(), activation_rows = activation_packed != 0u ? packed : tokens;
	std::vector<uint16_t> activation((size_t)activation_rows * input);
	for (size_t i = 0u; i < activation.size(); i++) activation[i] = FloatToBf16(((float)(Random() % 2001u) - 1000.0f) / 250.0f);
	printf("case groups=%u output=%u input=%u tokens=%u top_k=%u packed_activation=%u group_rows_max=%u\n",groups,output,input,tokens,top_k,activation_packed,routing.group_rows_max);
	std::vector<uint16_t> batched = Run(experts,weight,activation,routing,activation_packed,routing.group_rows_max,stream);
	CheckAgainstReference(experts,batched,activation,routing,activation_packed);
	if (routing.group_rows_max == 1u)
	{
		std::vector<uint16_t> wide = Run(experts,weight,activation,routing,activation_packed,LM_SKINNY_CELL_BATCH,stream);
		REQUIRE(memcmp(wide.data(),batched.data(),wide.size() * sizeof(uint16_t)) == 0);
		printf("  one-row and four-row batches are bit-identical\n");
	}
	if (activation_packed == 0u && tokens > 1u)
	{
		std::vector<uint32_t> alone_group(group_of_row.begin(),group_of_row.begin() + top_k), alone_source(top_k,0u);
		Routing alone = MakeRouting(alone_group,alone_source,groups);
		std::vector<uint16_t> single = Run(experts,weight,std::vector<uint16_t>(activation.begin(),activation.begin() + input),alone,0u,alone.group_rows_max,stream);
		for (uint32_t g = 0u; g < groups; g++)
			if (alone.offset[g + 1u] > alone.offset[g])
			{
				uint32_t row = routing.offset[g];
				while (routing.source[row] != 0u) row++;
				REQUIRE(memcmp(single.data() + (size_t)alone.offset[g] * output,batched.data() + (size_t)row * output,output * sizeof(uint16_t)) == 0);
			}
		printf("  token 0 alone equals token 0 inside the %u-token batch, bit for bit\n",tokens);
	}
	CUDA(cudaFree(weight));
}

int main(int argc,char **argv)
{
	cudaStream_t stream;
	if (argc < 2 || strcmp(argv[1],"--run") != 0)
	{
		fprintf(stderr,"usage: test_skinny_cell --run\n");
		return 2;
	}
	CUDA(cudaStreamCreate(&stream));
	Case(64u,384u,224u,1u,16u,0u,stream);
	Case(64u,224u,3072u,1u,16u,1u,stream);
	Case(64u,384u,224u,24u,16u,0u,stream);
	Case(64u,224u,3072u,24u,16u,1u,stream);
	Case(8u,64u,96u,5u,3u,0u,stream);
	{
		const std::vector<uint16_t> activation(64u,0u);
		std::vector<uint8_t> weight(64u * 64u);
		uint32_t offset[2] = {0u,1u};
		REQUIRE(LmSkinnyCellExperts(weight.data(),activation.data(),(uint16_t *)activation.data(),offset,offset,1u,1u,1u,1u,48u,32u,32u,stream) == LM_LAUNCH_ERR_SHAPE);
		REQUIRE(LmSkinnyCellExperts(weight.data(),activation.data(),(uint16_t *)activation.data(),offset,offset,1u,1u,1u,1u,64u,24u,32u,stream) == LM_LAUNCH_ERR_SHAPE);
		REQUIRE(LmSkinnyCellExperts(weight.data(),activation.data(),(uint16_t *)activation.data(),offset,offset,1u,1u,1u,1u,128u,32u,128u,stream) == LM_LAUNCH_ERR_SHAPE);
		printf("  K not a whole tile, output not whole cells and tile_k 128 are refused\n");
	}
	printf("test_skinny_cell PASS\n");
	return 0;
}
