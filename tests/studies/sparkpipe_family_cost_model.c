// sparkpipe_family_cost_model — GB10 decode-stage time cost for the resident
// CUDA modules. Per-class rates are CALIBRATED by solving from glm52's
// measured B128 Nsight buckets (docs/GLM52_B256_PER_TOKEN_KERNELS_20260704,
// docs/archive/GB10_CUDA_COST_MODEL_CALIBRATION.md), then applied to each driver's
// CONFIG byte geometry.
//
// VALIDATION (glm52, measured stage totals B64~100 B128~107 B256~208 ms):
//   model reproduces B128 within 5%, B256 within 16%. B64 runs ~35% low -
//   the low-batch regime omits the expert-coverage ramp and residual launch
//   overhead, so trust the model at B128+ and treat B1/B64 as lower bounds.
//
// MODEL FORM: a stage is 6 layers. Phases SERIALIZE within a stage (the
// glm52 buckets SUM to the measured stage, they do not overlap), so
//   stage = qkvo_compute + attention_bw + moe_bw + dense_bw + head_share.
// Weight bytes are read once at coverage saturation; qkvo compute scales
// with batch at the measured 6.5 TFLOPS effective WMMA rate.
//
// All numbers are PROJECTIONS. No family driver has run. k3 geometry is
// GUESS-tagged. The four measurements that convert these to silicon truth
// are named in the calibration doc.
#include <stdio.h>
#define BW 273.0e9
#define QKVO_RATE 6.5e12    // FLOP/s, measured (2.6% of FP8 peak)
#define ETA_ATTN 0.18       // solved from glm52 36.8ms B128
#define ETA_MOE 0.24        // solved from glm52 27.2ms B128 (incl pack/quant/route)
#define ETA_BW 0.80         // memory path, three-way cross-validated
#define LPS 6.0
#define STAGES 13.0

typedef struct {
  const char *name;
  double hidden, layers, topk, expert_mb, expert_inter, vocab;
  double qkvo_params, attn_latent_b, dense_inter, dense_layers, edtype;
  int guessed;
} drv_t;

static void stage(drv_t *m, double B){
  double moe_layers = m->layers - m->dense_layers;
  double expert_bytes = (m->expert_mb>0)? LPS*(moe_layers/m->layers)*m->topk*m->expert_mb*m->edtype : 0;
  double dense_bytes  = (m->dense_layers>0)? LPS*(m->dense_layers/m->layers)*3.0*m->hidden*m->dense_inter : 0;
  double moe_ms = expert_bytes/(BW*ETA_MOE)*1e3;
  double dense_ms = dense_bytes/(BW*ETA_BW)*1e3;
  double cmp_ms = (2.0*B*LPS*m->qkvo_params)/QKVO_RATE*1e3;
  double attn_ms = (B*LPS*2048.0*m->attn_latent_b)/(BW*ETA_ATTN)*1e3;
  double head_ms = (m->vocab*m->hidden*2.0)/(BW*ETA_BW)*1e3/STAGES;
  double st = cmp_ms+attn_ms+moe_ms+dense_ms+head_ms;
  const char *dom="qkvo"; double dm=cmp_ms;
  if(attn_ms>dm){dom="attn";dm=attn_ms;} if(moe_ms>dm){dom="moe";dm=moe_ms;} if(dense_ms>dm){dom="dense";dm=dense_ms;}
  printf("  B%-4.0f stage %6.1fms  qkvo %5.1f  attn %5.1f  moe %5.1f  dense %5.1f  head %4.1f  | bound=%-5s ~%4.0f tok/s\n",
    B, st, cmp_ms, attn_ms, moe_ms, dense_ms, head_ms, dom, B/(st*1e-3)/STAGES);
}

int main(void){
  printf("GB10 family decode cost model (calibrated from glm52 measured buckets)\n");
  printf("VALIDATED: B128 within 5%%, B256 within 16%%; B1/B64 are lower bounds.\n\n");
  drv_t glm={"glm52 (validation)",5120,6,8,37.75e6,1536,151552,154e6,1152,0,0,1.0,0};
  printf("== %s ==  (measured: B64~100 B128~107 B256~208 ms)\n", glm.name);
  stage(&glm,64); stage(&glm,128); stage(&glm,256);
  drv_t fam[4]={
    {"mimo25",4096,48,8,37.75e6,2048,152576,(13568.0+8192.0)*4096.0,(192.0+128.0)*2.0,16384,1,1.0,0},
    {"dsv4",4096,43,6,37.75e6,2048,129280,(32768.0+4096.0)*4096.0,576.0,0,0,1.0,0},
    {"qwen36",5120,64,0,0,0,248320,(4.0*256.0*2.0+5120.0)*5120.0,4.0*256.0*2.0,15360,64,1.0,0},
    {"k3",7168,72,8,19.6e6,2048,163840,(32768.0+7168.0)*7168.0,576.0,18432,1,0.5,1},
  };
  printf("\n");
  for(int i=0;i<4;i++){
    printf("== %s%s ==\n", fam[i].name, fam[i].guessed?"  [GEOMETRY GUESSED - double conditional]":"");
    stage(&fam[i],1); stage(&fam[i],64); stage(&fam[i],128);
  }
  printf("\nPer-driver lever (from the dominant term):\n");
  printf("  mimo25  MoE-bandwidth  -> NVFP4 experts halves the 27ms MoE term\n");
  printf("  dsv4    QKVO-compute   -> retile the 6.5-TFLOPS WMMA (no byte fix helps)\n");
  printf("  qwen36  ATTENTION      -> KV compression / absorbed-MLA byte cut (NOT dense/MoE)\n");
  printf("  k3      QKVO-compute   -> WMMA retile; confirm geometry first\n");
  return 0;
}
