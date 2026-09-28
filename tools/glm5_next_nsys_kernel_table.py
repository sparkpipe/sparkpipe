import csv,re,collections,sys
def load(f,per):
    rows=[r for r in csv.DictReader(open(f)) if r['GrdX']]
    tail=rows[-per*30:]
    reps=[tail[i*per:(i+1)*per] for i in range(30)]
    span=lambda rep:(int(rep[-1]['Start (ns)'])+int(rep[-1]['Duration (ns)'])-int(rep[0]['Start (ns)']))/1e6
    reps.sort(key=span)
    return reps[15],span(reps[15])
def cls(r):
    n=r['Name']; g=int(r['GrdX']); gz=r['GrdZ']
    if 'LmSkinnyKernel' in n:
        fp8='LmFp8' in n
        m={(False,193):('bf16 kda_qkv_beta 4096x1540',12.62),(False,192):('bf16 q_a / dense_gate_up 4096x1536',12.58),(False,256):('bf16 index_q 1536x4096 / attn_out 1024x4096',10.49),
           (False,128):('bf16 kda_out 512x4096 / dense_down',4.38),(False,64):('bf16 kv_a 4096x576 / q_b 1536x1024',3.94),(False,36):('bf16 router 4096x288 (f32 out)',2.36),
           (False,32):('bf16 K=4096 N=256 (decay_gate_down, shared_gate_up) / shared_down',None),(False,16):('bf16 index_k / index_gate 4096x128',1.05),(False,4):('bf16 small (decay_up, gate_up, index_head)',None),
           (True,256):('fp8 experts up 8x(4096x256)',8.65),(True,128):('fp8 experts down 8x(128x4096)',4.20)}
        k=m.get((fp8,g),('skinny other grid=%d'%g,None))
        if not fp8 and g==32 and int(r['Reg/Trd'])==64: k=('bf16 shared_down 128x4096',1.05)
        elif not fp8 and g==32: k=('bf16 K=4096 N=256 (decay_gate_down, shared_gate_up)',2.10)
        return k
    short=re.match(r'(?:void )?([A-Za-z0-9_]+)',n).group(1)
    b={'LmHeadCandidateKernel':79.3,'Glm5NextHcSiteKernel':1.64,'LmLatentAttentionDecodeSplitKernel':1.05,'LmDeltaRuleKernel':0.52}.get(short)
    if short in ('LmCausalConvKernel','LmCausalConvStreamsKernel'): short='LmCausalConv (3 per KDA layer -> 1)'
    return (short,b)
def agg(f,per):
    rep,span=load(f,per)
    d=collections.OrderedDict()
    for r in rep:
        k=cls(r); a=d.setdefault(k[0],[0,0.0,k[1]]); a[0]+=1; a[1]+=int(r['Duration (ns)'])/1e3
    return d,span,len(rep)
b,bs,bn=agg('base/b1_cuda_gpu_trace.csv',1457)
f,fs,fn=agg('final/b1_cuda_gpu_trace.csv',1389)
print("B1, context 1024, graph replay under nsys (cuda-graph-trace=node), median replay of 30. GB10 streaming read peak 243 GB/s (stream_read_probe).")
print("step span ms: base %.2f  final %.2f ; kernels per step: base %d  final %d"%(bs,fs,bn,fn))
print("%-58s %6s %9s %9s %8s %8s %7s %7s"%("kernel class","n","base_us","final_us","base/lnch","fin/lnch","base_GBs","fin_GBs"))
keys=list(b.keys())+[k for k in f.keys() if k not in b]
rows=[]
for k in keys:
    bb=b.get(k,[0,0.0,None]); ff=f.get(k,[0,0.0,None]); mb=bb[2] or ff[2]
    pb=bb[1]/bb[0] if bb[0] else 0; pf=ff[1]/ff[0] if ff[0] else 0
    gb=mb/pb*1e3 if mb and pb else None
    gf=mb/pf*1e3 if mb and pf else None
    rows.append((bb[1],k,bb[0] or ff[0],bb[1],ff[1],pb,pf,gb,gf))
rows.sort(key=lambda x:-x[0])
for _,k,n,t1,t2,p1,p2,g1,g2 in rows[:28]:
    print("%-58s %6d %9.1f %9.1f %8.2f %8.2f %7s %7s"%(k[:58],n,t1,t2,p1,p2,"%.0f"%g1 if g1 else "-","%.0f"%g2 if g2 else "-"))
print("sum of kernel time ms: base %.2f final %.2f"%(sum(v[1] for v in b.values())/1e3,sum(v[1] for v in f.values())/1e3))
