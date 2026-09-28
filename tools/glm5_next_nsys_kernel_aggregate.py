import csv,sys,re,collections
f=sys.argv[1]; per=int(sys.argv[2]) if len(sys.argv)>2 else 1457
rows=list(csv.DictReader(open(f)))
k=[r for r in rows if r['GrdX']]
tail=k[-per*30:]
reps=[tail[i*per:(i+1)*per] for i in range(30)]
def span(rep): return (int(rep[-1]['Start (ns)'])+int(rep[-1]['Duration (ns)'])-int(rep[0]['Start (ns)']))/1e6
best=min(reps,key=span)
print('check names equal', all(a['Name']==b['Name'] for a,b in zip(reps[0],reps[-1])), 'span',span(best),'busy',sum(int(r['Duration (ns)']) for r in best)/1e6)
def short(n):
    n=n.replace('void ','')
    m=re.match(r"([A-Za-z0-9_]+)(<[^(]*>)?",n)
    return m.group(1)+(m.group(2) or '')
agg=collections.OrderedDict()
for r in best:
    key=(short(r['Name'])[:90],r['GrdX'],r['GrdY'],r['BlkX'],r['Reg/Trd'])
    a=agg.setdefault(key,[0,0.0,[]]); a[0]+=1; a[1]+=int(r['Duration (ns)'])/1e3
    a[2].append(int(r['Duration (ns)'])/1e3)
tot=sum(v[1] for v in agg.values())
for key,v in sorted(agg.items(),key=lambda x:-x[1][1]):
    v[2].sort()
    print('%8.1f us %5.1f%% n=%4d med=%7.2f  %s grid=%s,%s blk=%s reg=%s'%(v[1],100*v[1]/tot,v[0],v[2][len(v[2])//2],key[0],key[1],key[2],key[3],key[4]))
