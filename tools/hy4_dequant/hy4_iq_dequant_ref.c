/* hy4 lane: CPU reference dequant of real rank-pack tensors (type 8/16/18/29). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "hy4_iq_dequant_vendor.h"

int main(int argc, char **argv) {
    if (argc != 5) { fprintf(stderr, "usage: %s pack offset type nelems\n", argv[0]); return 2; }
    const char *path = argv[1];
    long offset = atol(argv[2]);
    int type = atoi(argv[3]);
    long nelems = atol(argv[4]);
    long nblocks = nelems / 256;
    int bs = type == 8 ? 34 : type == 16 ? 66 : type == 18 ? 98 : type == 29 ? 56 : 0;
    if (!bs) { fprintf(stderr, "unsupported type %d\n", type); return 2; }
    long nbytes = nblocks * bs;
    uint8_t *src = malloc(nbytes);
    FILE *f = fopen(path, "rb");
    if (!f || fseek(f, offset, SEEK_SET) || (long)fread(src, 1, nbytes, f) != nbytes) {
        fprintf(stderr, "read failed\n"); return 3;
    }
    fclose(f);
    float *y = malloc((size_t)nelems * 4);
    if (type == 8) {
        for (long b = 0; b < nblocks; ++b) {
            const uint8_t *blk = src + b * 34;
            uint16_t bits = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
            float s = hy4_fp16_to_fp32(bits);
            for (int i = 0; i < 32; ++i) y[b * 32 + i] = s * (float)(int8_t)blk[2 + i];
        }
    } else if (type == 16) hy4_dequant_iq2_xxs(src, y, nblocks);
    else if (type == 18) hy4_dequant_iq3_xxs(src, y, nblocks);
    else hy4_dequant_iq1_m(src, y, nblocks);
    double sum = 0; float amax = 0; long nan = 0;
    unsigned long h = 1469598103934665603UL;
    const unsigned char *p = (const unsigned char *)y;
    for (long i = 0; i < nelems; ++i) {
        float v = y[i];
        if (v != v) nan++;
        sum += v; if (fabsf(v) > amax) amax = fabsf(v);
        for (int k = 0; k < 4; ++k) { h ^= p[i * 4 + k]; h *= 1099511628211UL; }
    }
    printf("type=%d nblocks=%ld elems=%ld nan=%ld sum=%.4f amax=%.6f fnv1a=%016lx\n",
           type, nblocks, nelems, nan, sum, amax, h);
    printf("first8: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
           y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]);
    return nan ? 1 : 0;
}
