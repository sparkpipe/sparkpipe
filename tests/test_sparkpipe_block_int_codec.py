import numpy as np, sys
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'tools'))
import sparkpipe_block_int_codec as I8
rng=np.random.default_rng(0); fails=0
def check(name,cond):
    global fails
    print("  %-52s %s"%(name,"PASS" if cond else "FAIL"))
    if not cond: fails+=1
def bf(x): return I8.f32_to_bf16(np.asarray(x,dtype=np.float32))
# 1 round trip + idempotency on random gaussian
for blk in (32,128,512):
    u=bf(rng.standard_normal(blk*64))
    c,s,st=I8.encode(u,blk); I8.verify(u,c,s,blk)
    check("verify() passes, block=%d"%blk,True)
    check("  bits/weight == 8+16/%d"%blk,abs(st["bits_per_weight"]-(8+16/blk))<1e-9)
# 2 all-zero block must not divide by zero
u=bf(np.zeros(256)); c,s,_=I8.encode(u); I8.verify(u,c,s)
check("all-zero block handled, decodes to zero",np.all(I8.decode(c,s)==0))
# 3 block maximum must map to +-127 exactly
u=bf(np.concatenate([[3.0],rng.standard_normal(127)*0.1]))
c,s,_=I8.encode(u)
check("block max encodes to code 127",int(np.abs(c).max())==127)
# 4 decode is exactly representable in bf16 (no hidden f32 precision)
d=I8.decode(c,s)
check("decode output is valid bf16 uint16",d.dtype==np.uint16)
# 5 sign symmetry: negating input negates codes
u2=bf(-I8.bf16_to_f32(u)); c2,s2,_=I8.encode(u2)
check("symmetric: negation flips codes, scales identical",np.array_equal(c2,-c) and np.array_equal(s2,s))
# 6 non-multiple of block must raise
try:
    I8.encode(bf(np.zeros(130)),128); check("rejects ragged element count",False)
except ValueError: check("rejects ragged element count",True)
# 7 nan/inf must raise
try:
    I8.encode(bf(np.array([np.inf]*128)),128); check("rejects inf/nan",False)
except ValueError: check("rejects inf/nan",True)

# 9 parametrised widths: 6/7/8 bits at two block sizes
for _bits in (6,7,8):
    for _blk in (32,128):
        _u=bf(rng.standard_normal(_blk*64))
        _c,_s,_st=I8.encode(_u,_blk,_bits); I8.verify(_u,_c,_s,_blk,_bits)
        check("verify() passes bits=%d blk=%d"%(_bits,_blk),True)
        check("  bits/weight == %d+16/%d"%(_bits,_blk),abs(_st["bits_per_weight"]-(_bits+16/_blk))<1e-9)
        check("  codes fit in %d bits"%_bits,int(abs(_c).max())<=(1<<(_bits-1))-1)
try:
    I8.encode(bf(rng.standard_normal(128)),128,9); check("rejects bits>8",False)
except ValueError: check("rejects bits>8",True)
try:
    I8.encode(bf(rng.standard_normal(128)),128,1); check("rejects bits<2",False)
except ValueError: check("rejects bits<2",True)

print("\n%d failures"%fails); sys.exit(1 if fails else 0)
