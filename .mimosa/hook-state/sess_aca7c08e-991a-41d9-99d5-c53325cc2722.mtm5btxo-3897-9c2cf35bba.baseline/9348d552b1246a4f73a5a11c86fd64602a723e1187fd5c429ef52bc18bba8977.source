#!/usr/bin/env python3
# Function-length gate for Spark-prefixed functions: exits NONZERO on any
# body over fifty lines, so shell && chains actually stop. Usage:
#   python3 tools/length_gate.py <file> [<file>...]
import re,sys
bad=0
for path in sys.argv[1:]:
    src=open(path).read().split('\n');i=0
    while i<len(src):
        m=re.match(r'^(?:static\s+)?(?:inline\s+)?(?:extern "C" )?(?:template|__device__|__global__|__forceinline__|SparkStatus|void|cudaError_t|float|int|uint32_t)[^\n]*\b(Spark\w+)\s*\(',src[i])
        if m and i+1<len(src):
            j=i
            while j<len(src) and src[j].strip()!='{': j+=1
            if j<len(src) and j-i<6:
                depth=0;k=j
                while k<len(src):
                    depth+=src[k].count('{')-src[k].count('}')
                    if depth==0:break
                    k+=1
                if k-j-1>50:
                    print(f"LONG {path.split('/')[-1]} {m.group(1)} {k-j-1}")
                    bad+=1
                i=k
        i+=1
print('LEN_OK' if bad==0 else 'LEN_FAIL')
sys.exit(1 if bad else 0)
