#include "modules/glm52_resident_decode_stage/source/cuda/config.h"
#include <stdio.h>
int main(void){
  /* every derived value cross-checked against the old header's constants */
  int fail=0;
  #define CK(a,b,l) do{ if((a)!=(b)){printf("  FAIL %s: %u != %u\n",l,(unsigned)(a),(unsigned)(b));fail++;} else printf("  ok   %s = %u\n",l,(unsigned)(a)); }while(0)
  CK(GLM52_ROUTED_LAYERS,75u,"routed layers");
  CK(GLM52_GATE_UP_DIM,4096u,"gate+up dim");
  CK(GLM52_LATENT_ROW,576u,"latent row elements");
  CK(GLM52_WEIGHT_LAYERS,79u,"weight layers incl MTP");
  CK(Glm52RowsPerExpert(128u),4u,"rows/expert at B128");
  CK(Glm52RowsPerExpert(1024u),32u,"rows/expert at B1024");
  CK(Glm52RowsPerExpert(1u),1u,"rows/expert at B1");
  printf("\n%s\n",fail?"FAIL":"PASS");
  return fail?1:0;}
