#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/linear_attn.cuh"

#define K_DIM 128u
#define V_DIM 128u
#define RT 256u
#define CK(call) do { cudaError_t e_ = (call); if ( e_ != cudaSuccess ) { printf("FAIL cuda %s: %s\n",#call,cudaGetErrorString(e_)); exit(1); } } while (0)

static uint64_t Mix(uint64_t z) { z = z * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull; z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull; z = (z ^ (z >> 27u)) * 0x94d049bb133111ebull; return z ^ (z >> 31u); }
static float Uni(uint64_t i, uint64_t s) { return (float)((int32_t)((Mix(i ^ (s << 40)) >> 40u) & 0xfffu) - 2048) / 2048.0f; }
static uint16_t Bf(float v) { uint32_t b; memcpy(&b,&v,4); return (uint16_t)((b + 0x7fffu + ((b >> 16u) & 1u)) >> 16u); }
static float FromBf(uint16_t v) { uint32_t b = (uint32_t)v << 16; float f; memcpy(&f,&b,4); return f; }
template<class T> static T *Dev(const std::vector<T> &h) { T *d; CK(cudaMalloc((void **)&d,h.size() * sizeof(T) + 16)); CK(cudaMemcpy(d,h.data(),h.size() * sizeof(T),cudaMemcpyHostToDevice)); return d; }

static int failures;

static void Case(const char *label, uint32_t sequences, uint32_t length, uint32_t heads, float min_log_decay)
{
	const uint32_t rows = sequences * length;
	const uint64_t slot_bytes = (uint64_t)heads * K_DIM * V_DIM * 4u;
	std::vector<uint32_t> begin(sequences + 1u),slots(sequences);
	for (uint32_t s = 0u; s <= sequences; s++) begin[s] = s * length;
	for (uint32_t s = 0u; s < sequences; s++) slots[s] = s;
	std::vector<uint16_t> q((uint64_t)rows * heads * K_DIM),k(q.size()),v((uint64_t)rows * heads * V_DIM);
	std::vector<float> f(q.size()),b((uint64_t)rows * heads),st((uint64_t)sequences * slot_bytes / 4u);
	for (uint64_t i = 0u; i < q.size(); i++) { q[i] = Bf(Uni(i,1)); k[i] = Bf(Uni(i,2)); f[i] = expf(min_log_decay * (0.5f + 0.5f * Uni(i,3)) * (0.5f + 0.5f * Uni(i,9))); }
	for (uint64_t i = 0u; i < v.size(); i++) v[i] = Bf(Uni(i,4));
	for (uint64_t i = 0u; i < b.size(); i++) b[i] = 0.5f + 0.5f * Uni(i,5);
	for (uint64_t i = 0u; i < st.size(); i++) st[i] = 0.25f * Uni(i,6);
	uint32_t *bd = Dev(begin),*sd = Dev(slots);
	uint16_t *qd = Dev(q),*kd = Dev(k),*vd = Dev(v),*o1,*o2;
	float *fd = Dev(f),*btd = Dev(b),*s1 = Dev(st),*s2 = Dev(st);
	const uint64_t oe = (uint64_t)rows * heads * V_DIM;
	CK(cudaMalloc((void **)&o1,oe * 2)); CK(cudaMalloc((void **)&o2,oe * 2));
	cudaEvent_t e[3]; for (int i = 0; i < 3; i++) cudaEventCreate(&e[i]);
	cudaEventRecord(e[0]);
	LmDeltaRuleColumnKernel<RT,K_DIM,V_DIM,float><<<dim3(sequences,heads,V_DIM / 32u),LM_DELTA_COLUMN_THREADS>>>((uint8_t *)s1,(uint32_t)slot_bytes,sd,bd,0,qd,kd,vd,fd,btd,o1,heads,1u,sequences,1u,0);
	cudaEventRecord(e[1]);
	CK((LmDeltaRuleChunkLaunch<RT,K_DIM,V_DIM,float>((uint8_t *)s2,(uint32_t)slot_bytes,sd,bd,0,qd,kd,vd,fd,btd,o2,heads,1u,sequences,1u,0,0)));
	cudaEventRecord(e[2]);
	CK(cudaDeviceSynchronize());
	float ms_col, ms_chunk; cudaEventElapsedTime(&ms_col,e[0],e[1]); cudaEventElapsedTime(&ms_chunk,e[1],e[2]);
	std::vector<uint16_t> h1(oe),h2(oe); std::vector<float> t1(st.size()),t2(st.size());
	CK(cudaMemcpy(h1.data(),o1,oe * 2,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(h2.data(),o2,oe * 2,cudaMemcpyDeviceToHost));
	CK(cudaMemcpy(t1.data(),s1,st.size() * 4,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(t2.data(),s2,st.size() * 4,cudaMemcpyDeviceToHost));
	double omax = 0, odiff = 0, osq = 0, odsq = 0, smax = 0, sdiff = 0; uint64_t bad = 0;
	for (uint64_t i = 0; i < oe; i++) { double a = FromBf(h1[i]), c = FromBf(h2[i]); omax = fmax(omax,fabs(a)); odiff = fmax(odiff,fabs(a - c)); osq += a * a; odsq += (a - c) * (a - c); bad += !isfinite(c); }
	for (uint64_t i = 0; i < st.size(); i++) { smax = fmax(smax,fabs(t1[i])); sdiff = fmax(sdiff,fabs((double)t1[i] - t2[i])); }
	const double rel = sqrt(odsq / (osq + 1e-30));
	const int ok = bad == 0 && rel < 2e-3 && sdiff <= 1e-3 * smax;
	failures += !ok;
	printf("%s %s seq=%u len=%u heads=%u min_log_decay=%.1f: output rel rms %.2e worst %.3g of %.3g, state worst %.3g of %.3g, nonfinite %llu | column %.2f ms chunk %.2f ms\n",
		ok ? "PASS" : "FAIL",label,sequences,length,heads,min_log_decay,rel,odiff,omax,sdiff,smax,(unsigned long long)bad,ms_col,ms_chunk);
	cudaFree(bd); cudaFree(sd); cudaFree(qd); cudaFree(kd); cudaFree(vd); cudaFree(o1); cudaFree(o2); cudaFree(fd); cudaFree(btd); cudaFree(s1); cudaFree(s2);
	}

int main(void)
{
	Case("tiny",1u,5u,6u,-0.3f);
	Case("partial-chunk",2u,37u,6u,-1.0f);
	Case("strong-decay",2u,300u,6u,-5.0f);
	Case("long-slice",8u,4096u,6u,-0.3f);
	Case("long-slice-strong-decay",8u,4096u,6u,-5.0f);
	printf("%s chunked delta rule matches the column kernel\n",failures == 0 ? "PASS" : "FAIL");
	return failures != 0;
}
