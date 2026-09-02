/* hy4 lane: apply the o_proj (split0) repair payload to one deployed rank
 * bundle, then refresh its digest records.
 *
 * Usage: hy4_wo_fleet_apply <rank.gguf> <blob.bin> <offsets.txt>
 *
 * offsets.txt lines: <file offset> <length>, one per blk.N.attn_output.weight
 * region (identical layout on every rank). Regions whose current bytes equal
 * the payload are skipped; the rest are rewritten. Afterwards the whole file
 * digest is recomputed, the <rank.gguf>.sha256 sidecar is rewritten and the
 * manifest's gguf_sha256 hex is replaced in place (same byte length).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

typedef struct {
    uint32_t state[8];
    uint64_t bitlen;
    uint8_t buf[64];
    size_t buflen;
} Sha256;

static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

static void sha256_update(Sha256 *s, const uint8_t *data, size_t len) {
    static const uint32_t k[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,
        0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
        0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,
        0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,
        0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
        0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,
        0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,
        0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
        0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    for (size_t i = 0; i < len; ++i) {
        s->buf[s->buflen++] = data[i];
        if (s->buflen == 64) {
            uint32_t w[64];
            for (int j = 0; j < 16; ++j)
                w[j] = (uint32_t)s->buf[j*4] << 24 | (uint32_t)s->buf[j*4+1] << 16
                     | (uint32_t)s->buf[j*4+2] << 8 | s->buf[j*4+3];
            for (int j = 16; j < 64; ++j) {
                uint32_t s0 = rotr(w[j-15],7) ^ rotr(w[j-15],18) ^ (w[j-15] >> 3);
                uint32_t s1 = rotr(w[j-2],17) ^ rotr(w[j-2],19) ^ (w[j-2] >> 10);
                w[j] = w[j-16] + s0 + w[j-7] + s1;
            }
            uint32_t a = s->state[0], b = s->state[1], c = s->state[2];
            uint32_t d = s->state[3], e = s->state[4], f = s->state[5];
            uint32_t g = s->state[6], h = s->state[7];
            for (int j = 0; j < 64; ++j) {
                uint32_t S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
                uint32_t ch = (e & f) ^ (~e & g);
                uint32_t t1 = h + S1 + ch + k[j] + w[j];
                uint32_t S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
                uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
                uint32_t t2 = S0 + maj;
                h = g; g = f; f = e; e = d + t1;
                d = c; c = b; b = a; a = t1 + t2;
            }
            s->state[0]+=a; s->state[1]+=b; s->state[2]+=c; s->state[3]+=d;
            s->state[4]+=e; s->state[5]+=f; s->state[6]+=g; s->state[7]+=h;
            s->buflen = 0;
        }
    }
    s->bitlen += (uint64_t)len * 8;
}

static void sha256_final(Sha256 *s, uint8_t *out) {
    uint64_t bits = s->bitlen;
    size_t padlen = (s->buflen < 56) ? 56 - s->buflen : 120 - s->buflen;
    uint8_t pad[72];
    memset(pad, 0, sizeof(pad));
    pad[0] = 0x80;
    for (int i = 0; i < 8; ++i)
        pad[padlen + i] = (uint8_t)(bits >> (56 - 8 * i));
    sha256_update(s, pad, padlen + 8);
    for (int i = 0; i < 8; ++i) {
        out[i*4] = (uint8_t)(s->state[i] >> 24);
        out[i*4+1] = (uint8_t)(s->state[i] >> 16);
        out[i*4+2] = (uint8_t)(s->state[i] >> 8);
        out[i*4+3] = (uint8_t)s->state[i];
    }
}

static long read_regions(const char *path, long **offs, long **lens) {
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); exit(1); }
    int cap = 128, n = 0;
    *offs = malloc(cap * sizeof(long));
    *lens = malloc(cap * sizeof(long));
    long o, l;
    while (fscanf(f, "%ld %ld", &o, &l) == 2) {
        if (n == cap) {
            cap *= 2;
            *offs = realloc(*offs, cap * sizeof(long));
            *lens = realloc(*lens, cap * sizeof(long));
        }
        (*offs)[n] = o;
        (*lens)[n] = l;
        n++;
    }
    fclose(f);
    return n;
}

static const char *base_name(const char *path) {
    const char *s = strrchr(path, '/');
    return s ? s + 1 : path;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <rank.gguf> <blob.bin> <offsets.txt>\n",
                argv[0]);
        return 2;
    }
    long *offs, *lens;
    int n = read_regions(argv[3], &offs, &lens);
    if (n <= 0) { fprintf(stderr, "no regions\n"); return 1; }

    FILE *blob = fopen(argv[2], "rb");
    if (!blob) { perror(argv[2]); return 1; }
    fseek(blob, 0, SEEK_END);
    long blobsz = ftell(blob);
    fseek(blob, 0, SEEK_SET);
    long total = 0;
    for (int i = 0; i < n; ++i) total += lens[i];
    if (blobsz != total) {
        fprintf(stderr, "blob %ld != regions %ld\n", blobsz, total);
        return 1;
    }
    uint8_t *want = malloc((size_t)blobsz);
    if (fread(want, 1, (size_t)blobsz, blob) != (size_t)blobsz) {
        fprintf(stderr, "blob short read\n"); return 1;
    }
    fclose(blob);

    FILE *gguf = fopen(argv[1], "r+b");
    if (!gguf) { perror(argv[1]); return 1; }
    long applied = 0, skipped = 0;
    long cursor = 0;
    for (int i = 0; i < n; ++i) {
        uint8_t *cur = malloc((size_t)lens[i]);
        if (fseek(gguf, offs[i], SEEK_SET) ||
            fread(cur, 1, (size_t)lens[i], gguf) != (size_t)lens[i]) {
            fprintf(stderr, "region %d read fail\n", i); return 1;
        }
        if (memcmp(cur, want + cursor, (size_t)lens[i]) != 0) {
            if (fseek(gguf, offs[i], SEEK_SET) ||
                fwrite(want + cursor, 1, (size_t)lens[i], gguf)
                    != (size_t)lens[i]) {
                fprintf(stderr, "region %d write fail\n", i); return 1;
            }
            applied += lens[i];
        } else {
            skipped += lens[i];
        }
        free(cur);
        cursor += lens[i];
    }
    fflush(gguf);
    fseek(gguf, 0, SEEK_END);
    long fsize = ftell(gguf);
    fflush(gguf);
    rewind(gguf);

    Sha256 sh;
    sh.state[0]=0x6a09e667; sh.state[1]=0xbb67ae85; sh.state[2]=0x3c6ef372;
    sh.state[3]=0xa54ff53a; sh.state[4]=0x510e527f; sh.state[5]=0x9b05688c;
    sh.state[6]=0x1f83d9ab; sh.state[7]=0x5be0cd19;
    sh.bitlen = 0; sh.buflen = 0;
    uint8_t buf[1 << 20];
    size_t got;
    long done = 0;
    while ((got = fread(buf, 1, sizeof(buf), gguf)) > 0) {
        sha256_update(&sh, buf, got);
        done += (long)got;
    }
    if (done != fsize) {
        fprintf(stderr, "digest pass %ld != size %ld\n", done, fsize);
        return 1;
    }
    uint8_t dig[32];
    sha256_final(&sh, dig);
    char hex[65];
    for (int i = 0; i < 32; ++i)
        sprintf(hex + i*2, "%02x", dig[i]);

    rewind(gguf);
    fflush(gguf);
    fsync(fileno(gguf));
    fclose(gguf);

    char sidecar[1024];
    snprintf(sidecar, sizeof(sidecar), "%s.sha256", argv[1]);
    FILE *sc = fopen(sidecar, "w");
    if (!sc) { perror(sidecar); return 1; }
    fprintf(sc, "%s  %s\n", hex, base_name(argv[1]));
    fclose(sc);

    char manpath[1024];
    {
        char *copy = strdup(argv[1]);
        char *dir = strrchr(copy, '/');
        if (dir) *(dir + 1) = 0; else strcpy(copy, "./");
        snprintf(manpath, sizeof(manpath), "%smanifest.json", copy);
        free(copy);
    }
    FILE *mf = fopen(manpath, "rb");
    if (!mf) { perror(manpath); return 1; }
    fseek(mf, 0, SEEK_END);
    long msz = ftell(mf);
    fseek(mf, 0, SEEK_SET);
    char *mtxt = malloc((size_t)msz + 1);
    if (fread(mtxt, 1, (size_t)msz, mf) != (size_t)msz) {
        fprintf(stderr, "manifest read fail\n"); return 1;
    }
    mtxt[msz] = 0;
    fclose(mf);
    const char *key = "\"gguf_sha256\": \"";
    char *pos = strstr(mtxt, key);
    if (!pos) { fprintf(stderr, "manifest key missing\n"); return 1; }
    char *hexpos = pos + strlen(key);
    if (strlen(hexpos) < 64) { fprintf(stderr, "manifest hex short\n"); return 1; }
    memcpy(hexpos, hex, 64);
    FILE *mf2 = fopen(manpath, "wb");
    if (!mf2) { perror(manpath); return 1; }
    fwrite(mtxt, 1, (size_t)msz, mf2);
    fclose(mf2);

    printf("applied %ld bytes, skipped %ld bytes, sha256 %s\n",
           applied, skipped, hex);
    free(want); free(offs); free(lens); free(mtxt);
    return 0;
}
