#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "inference/kernels/skinny.cuh"

#define CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { fprintf(stderr,"FAIL line=%d cuda=%s call=%s\n",__LINE__,cudaGetErrorString(e),#call); exit(1); } } while(0)
#define REQUIRE(test) do { if(!(test)) { fprintf(stderr,"FAIL line=%d test=%s\n",__LINE__,#test); exit(1); } } while(0)

#define REF_MAX_ROWS 20u
#define REF_ALLOC_ROWS (LM_SKINNY_ROWS_WIDE + 1u)
#define REF_SENTINEL_BF16 0x5a5au
#define REF_SENTINEL_F32 0x5a5a5a5au

static uint32_t random_state = 20261013u;
static uint64_t compared = 0u, failures = 0u, sentinel_failures = 0u;
static uint64_t failures_by_rows[REF_MAX_ROWS + 1u];
static uint64_t compared_by_rows[REF_MAX_ROWS + 1u];

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

static double Bf16Value(uint16_t value)
{
	uint32_t bits=(uint32_t)value<<16u;
	float result;
	memcpy(&result,&bits,sizeof(result));
	return result;
}

static double E4m3Value(uint8_t code)
{
	const uint32_t sign=code>>7u, exponent=(code>>3u)&15u, mantissa=code&7u;
	const double magnitude=exponent==0u ? ldexp((double)mantissa/8.0,-6) : ldexp(1.0+(double)mantissa/8.0,(int)exponent-7);
	return sign!=0u ? -magnitude : magnitude;
}

static uint8_t RandomE4m3()
{
	uint8_t code;
	do code=(uint8_t)(Random()&0xffu); while ((code&0x7fu)==0x7fu || ((code>>3u)&15u)>9u);
	return code;
}

static float BitsF32(uint32_t bits)
{
	float value;
	memcpy(&value,&bits,sizeof(value));
	return value;
}

typedef struct Matrix
{
	uint32_t input,output,fp8;
	std::vector<double> weight;
	std::vector<double> scale;
	void *device_weight;
	float *device_scale;
}
Matrix;

static void BuildBf16(Matrix *m, uint32_t input, uint32_t output)
{
	std::vector<uint16_t> host((size_t)output*input);
	m->input=input; m->output=output; m->fp8=0u; m->device_scale=0;
	m->weight.resize(host.size());
	for (size_t i=0u; i<host.size(); i++) { host[i]=Bf16(0.0625f*(((int32_t)(Random()%2049u)-1024)/1024.0f)); m->weight[i]=Bf16Value(host[i]); }
	CUDA(cudaMalloc(&m->device_weight,host.size()*2u));
	CUDA(cudaMemcpy(m->device_weight,host.data(),host.size()*2u,cudaMemcpyHostToDevice));
}

static void BuildFp8(Matrix *m, uint32_t input, uint32_t output)
{
	const uint32_t groups=(input+LmFp8::kScaleGroup-1u)/LmFp8::kScaleGroup;
	std::vector<uint8_t> host((size_t)output*input);
	std::vector<float> scale((size_t)output*groups);
	m->input=input; m->output=output; m->fp8=1u;
	m->weight.resize(host.size());
	m->scale.resize(scale.size());
	for (size_t i=0u; i<host.size(); i++) { host[i]=RandomE4m3(); m->weight[i]=E4m3Value(host[i]); }
	for (size_t i=0u; i<scale.size(); i++) { scale[i]=ldexpf(1.0f+(float)(Random()%256u)/256.0f,-7-(int)(Random()%3u)); m->scale[i]=scale[i]; }
	CUDA(cudaMalloc(&m->device_weight,host.size()));
	CUDA(cudaMemcpy(m->device_weight,host.data(),host.size(),cudaMemcpyHostToDevice));
	CUDA(cudaMalloc(&m->device_scale,scale.size()*4u));
	CUDA(cudaMemcpy(m->device_scale,scale.data(),scale.size()*4u,cudaMemcpyHostToDevice));
}

static void Release(Matrix *m)
{
	CUDA(cudaFree(m->device_weight));
	if (m->device_scale!=0) CUDA(cudaFree(m->device_scale));
}

static LmScaleTensor Scale(const Matrix *m)
{
	if (m->fp8==0u) return LmScaleTensorNone();
	return LmScaleTensorBuild(m->device_scale,LM_SCALE_ENCODING_F32,1u,m->output,m->input,1u,LmFp8::kScaleGroup);
}

static void Reference(const Matrix *m, const std::vector<double> &activation, std::vector<double> *value, std::vector<double> *magnitude)
{
	const uint32_t groups=(m->input+LmFp8::kScaleGroup-1u)/LmFp8::kScaleGroup;
	value->assign((size_t)REF_MAX_ROWS*m->output,0.0);
	magnitude->assign((size_t)REF_MAX_ROWS*m->output,0.0);
	for (uint32_t r=0u; r<REF_MAX_ROWS; r++)
		for (uint32_t n=0u; n<m->output; n++)
		{
			const double *w=&m->weight[(size_t)n*m->input], *a=&activation[(size_t)r*m->input];
			double sum=0.0, absolute=0.0, term;
			for (uint32_t k=0u; k<m->input; k++)
			{
				term=w[k]*a[k]*(m->fp8!=0u ? m->scale[(size_t)n*groups+k/LmFp8::kScaleGroup] : 1.0);
				sum+=term; absolute+=fabs(term);
			}
			(*value)[(size_t)r*m->output+n]=sum;
			(*magnitude)[(size_t)r*m->output+n]=absolute;
		}
}

static const char *Branch(const Matrix *m, uint32_t rows)
{
	const uint32_t chunks=m->input/(m->fp8!=0u ? 16u : 8u);
	const bool split=m->fp8==0u && m->output<=LM_SKINNY_SPLIT_MAX_OUTPUTS && chunks>=LM_SKINNY_SPLIT_WARPS*LM_WARP_LANES;
	const uint32_t shape=rows>LM_SKINNY_ROWS ? LM_SKINNY_ROWS_WIDE : rows>LM_SKINNY_ROWS_MID ? LM_SKINNY_ROWS : rows!=1u ? LM_SKINNY_ROWS_MID : 1u;
	static char text[64];
	snprintf(text,sizeof(text),"%s rows_t=%u lanes=%u",split ? "split" : "launch",shape,split ? 32u : chunks>=32u ? 32u : chunks>=16u ? 16u : 8u);
	return text;
}

static void Check(const char *path, const Matrix *m, uint32_t rows, uint32_t stride, uint32_t offset, uint32_t f32, const void *device_output,
	const std::vector<double> &value, const std::vector<double> &magnitude)
{
	const size_t count=(size_t)REF_ALLOC_ROWS*stride;
	std::vector<uint8_t> host(count*(f32 ? 4u : 2u));
	uint32_t printed=0u;
	CUDA(cudaMemcpy(host.data(),device_output,host.size(),cudaMemcpyDeviceToHost));
	for (uint32_t r=0u; r<REF_ALLOC_ROWS; r++)
		for (uint32_t c=0u; c<stride; c++)
		{
			const size_t index=(size_t)r*stride+c;
			const bool live=r<rows && c>=offset && c<offset+m->output;
			uint32_t bits;
			double got;
			if (f32) { memcpy(&bits,&host[index*4u],4u); got=BitsF32(bits); }
			else { uint16_t half; memcpy(&half,&host[index*2u],2u); bits=half; got=Bf16Value(half); }
			if (!live)
			{
				if (bits!=(f32 ? REF_SENTINEL_F32 : REF_SENTINEL_BF16))
				{
					sentinel_failures++;
					if (sentinel_failures<=8u) fprintf(stderr,"SENTINEL %s %s rows=%u in=%u out=%u stride=%u offset=%u f32=%u row=%u col=%u bits=%08x\n",path,Branch(m,rows),rows,m->input,m->output,stride,offset,f32,r,c,bits);
				}
				continue;
			}
			const double expect=value[(size_t)r*m->output+c-offset], bound=magnitude[(size_t)r*m->output+c-offset];
			const double tolerance=(f32 ? 1e-6 : 1.0/128.0)*fabs(expect)+1e-5*bound+1e-30;
			compared++; compared_by_rows[rows]++;
			if (!(fabs(got-expect)<=tolerance))
			{
				failures++; failures_by_rows[rows]++;
				if (printed<3u && failures<=40u) fprintf(stderr,"MISMATCH %s %s rows=%u in=%u out=%u stride=%u offset=%u f32=%u row=%u col=%u got=%.6g ref=%.6g tol=%.3g\n",path,Branch(m,rows),rows,m->input,m->output,stride,offset,f32,r,c-offset,got,expect,tolerance);
				printed++;
			}
		}
}

static void *Output(uint32_t stride, uint32_t f32)
{
	void *device;
	const size_t bytes=(size_t)REF_ALLOC_ROWS*stride*(f32 ? 4u : 2u);
	CUDA(cudaMalloc(&device,bytes));
	CUDA(cudaMemset(device,0x5a,bytes));
	return device;
}

static int32_t Single(const Matrix *m, const uint16_t *activation, uint32_t rows, uint32_t stride, uint32_t offset, void *output, uint32_t f32, cudaStream_t stream)
{
	LmSkinnyArguments args;
	if (m->fp8==0u)
		return LmSkinnyDense<LmBf16Format>(m->device_weight,activation,f32 ? 0 : (uint16_t *)output,f32 ? (float *)output : 0,rows,m->input,m->output,stride,offset,stream);
	memset(&args,0,sizeof(args));
	args.weight=(const uint8_t *)m->device_weight;
	args.activation=activation;
	args.output_bf16=f32 ? 0 : (uint16_t *)output;
	args.output_f32=f32 ? (float *)output : 0;
	args.scale=Scale(m);
	args.rows=rows;
	args.input_dimension=m->input;
	args.output_dimension=m->output;
	args.output_row_stride=stride;
	args.output_column_offset=offset;
	return LmSkinnyLaunch<LmFp8>(&args,stream);
}

static void Dense(const Matrix *m, const uint16_t *activation, const std::vector<double> &value, const std::vector<double> &magnitude, cudaStream_t stream)
{
	const uint32_t layouts[2][2]={{0u,0u},{m->output+24u,8u}};
	for (uint32_t layout=0u; layout<2u; layout++)
		for (uint32_t f32=0u; f32<2u; f32++)
		{
			const uint32_t stride=layouts[layout][0]!=0u ? layouts[layout][0] : m->output, offset=layouts[layout][1];
			void *output=Output(stride,f32);
			for (uint32_t rows=1u; rows<=REF_MAX_ROWS; rows++)
			{
				const int32_t status=Single(m,activation,rows,layouts[layout][0],offset,output,f32,stream);
				if (rows>LM_SKINNY_ROWS_WIDE) { REQUIRE(status==LM_LAUNCH_ERR_SHAPE); continue; }
				REQUIRE(status==LM_LAUNCH_OK);
				CUDA(cudaStreamSynchronize(stream));
				Check(m->fp8 ? "fp8" : "bf16",m,rows,stride,offset,f32,output,value,magnitude);
				CUDA(cudaMemset(output,0x5a,(size_t)REF_ALLOC_ROWS*stride*(f32 ? 4u : 2u)));
			}
			CUDA(cudaFree(output));
		}
}

static void Multi(const Matrix *const *parts, uint32_t count, const uint16_t *activation, const std::vector<double> *value, const std::vector<double> *magnitude, uint32_t fp8, cudaStream_t stream)
{
	LmSkinnyDenseTarget targets[LM_SKINNY_MULTI_MAX];
	void *output[LM_SKINNY_MULTI_MAX];
	uint32_t stride[LM_SKINNY_MULTI_MAX], offset[LM_SKINNY_MULTI_MAX], f32[LM_SKINNY_MULTI_MAX], part, rows;
	for (part=0u; part<count; part++)
	{
		f32[part]=part%2u;
		offset[part]=part==1u ? 8u : 0u;
		stride[part]=part==1u ? parts[part]->output+24u : parts[part]->output;
		output[part]=Output(stride[part],f32[part]);
		memset(&targets[part],0,sizeof(targets[part]));
		targets[part].weight=parts[part]->device_weight;
		targets[part].output_bf16=f32[part] ? 0 : (uint16_t *)output[part];
		targets[part].output_f32=f32[part] ? (float *)output[part] : 0;
		targets[part].output_dimension=parts[part]->output;
		targets[part].output_row_stride=part==1u ? stride[part] : 0u;
		targets[part].output_column_offset=offset[part];
		targets[part].scale=Scale(parts[part]);
	}
	for (rows=1u; rows<=LM_SKINNY_ROWS_WIDE; rows++)
	{
		const int32_t status=fp8 ? LmSkinnyDenseMulti<LmFp8>(targets,count,activation,rows,parts[0]->input,stream) : LmSkinnyDenseMulti<LmBf16Format>(targets,count,activation,rows,parts[0]->input,stream);
		REQUIRE(status==LM_LAUNCH_OK);
		CUDA(cudaStreamSynchronize(stream));
		for (part=0u; part<count; part++)
		{
			Check(fp8 ? "multi-fp8" : "multi-bf16",parts[part],rows,stride[part],offset[part],f32[part],output[part],value[part],magnitude[part]);
			CUDA(cudaMemset(output[part],0x5a,(size_t)REF_ALLOC_ROWS*stride[part]*(f32[part] ? 4u : 2u)));
		}
	}
	for (part=0u; part<count; part++) CUDA(cudaFree(output[part]));
}

static void Input(uint32_t input, uint32_t fp8, cudaStream_t stream)
{
	static const uint32_t outputs[]={4u,6u,8u,64u,255u,256u,257u,512u,4096u};
	const uint32_t count=sizeof(outputs)/sizeof(outputs[0]);
	std::vector<uint16_t> host((size_t)REF_MAX_ROWS*input);
	std::vector<double> activation(host.size());
	std::vector<Matrix> matrices(count);
	std::vector<std::vector<double>> value(count), magnitude(count);
	uint16_t *device;
	uint32_t i;
	for (size_t j=0u; j<host.size(); j++) { host[j]=Bf16(4.0f*(((int32_t)(Random()%2049u)-1024)/1024.0f)); activation[j]=Bf16Value(host[j]); }
	CUDA(cudaMalloc(&device,host.size()*2u));
	CUDA(cudaMemcpy(device,host.data(),host.size()*2u,cudaMemcpyHostToDevice));
	for (i=0u; i<count; i++)
	{
		if (fp8) BuildFp8(&matrices[i],input,outputs[i]); else BuildBf16(&matrices[i],input,outputs[i]);
		Reference(&matrices[i],activation,&value[i],&magnitude[i]);
		Dense(&matrices[i],device,value[i],magnitude[i],stream);
	}
	{
		const Matrix *parts[3]={&matrices[0],&matrices[6],&matrices[8]};
		const std::vector<double> v[3]={value[0],value[6],value[8]}, a[3]={magnitude[0],magnitude[6],magnitude[8]};
		Multi(parts,3u,device,v,a,fp8,stream);
	}
	{
		const Matrix *parts[4]={&matrices[1],&matrices[5],&matrices[3],&matrices[7]};
		const std::vector<double> v[4]={value[1],value[5],value[3],value[7]}, a[4]={magnitude[1],magnitude[5],magnitude[3],magnitude[7]};
		Multi(parts,4u,device,v,a,fp8,stream);
	}
	for (i=0u; i<count; i++) Release(&matrices[i]);
	CUDA(cudaFree(device));
	fprintf(stderr,"input=%u fp8=%u compared=%llu failures=%llu sentinel=%llu\n",input,fp8,(unsigned long long)compared,(unsigned long long)failures,(unsigned long long)sentinel_failures);
}

int main(int argc, char **argv)
{
	static const uint32_t inputs[]={64u,128u,256u,512u,1024u,2048u,2816u,3072u,4096u,5120u,6144u};
	cudaStream_t stream;
	uint32_t i, rows;
	if (argc<2 || strcmp(argv[1],"--run")!=0) { printf("SKIP test_skinny_rows_reference needs --run on a CUDA host\n"); return 0; }
	CUDA(cudaStreamCreate(&stream));
	for (i=0u; i<sizeof(inputs)/sizeof(inputs[0]); i++)
	{
		Input(inputs[i],0u,stream);
		if (inputs[i]%16u==0u) Input(inputs[i],1u,stream);
	}
	for (rows=1u; rows<=LM_SKINNY_ROWS_WIDE; rows++)
		fprintf(stderr,"rows=%u compared=%llu failures=%llu\n",rows,(unsigned long long)compared_by_rows[rows],(unsigned long long)failures_by_rows[rows]);
	CUDA(cudaStreamDestroy(stream));
	if (failures!=0u || sentinel_failures!=0u) { fprintf(stderr,"FAIL skinny rows reference failures=%llu sentinel=%llu\n",(unsigned long long)failures,(unsigned long long)sentinel_failures); return 1; }
	printf("OK skinny rows reference compared=%llu\n",(unsigned long long)compared);
	return 0;
}
