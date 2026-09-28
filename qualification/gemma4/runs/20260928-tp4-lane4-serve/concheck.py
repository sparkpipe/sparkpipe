import json,sys,urllib.request,concurrent.futures as cf
ref=json.load(open(sys.argv[1])); url=sys.argv[2]; reps=int(sys.argv[3])
jobs=[r for r in ref['results'] for _ in range(reps)]
def run(r):
    body=json.dumps({"prompt_token_ids":r['prompt_token_ids'],"max_tokens":32,"temperature":0}).encode()
    try:
        d=json.load(urllib.request.urlopen(urllib.request.Request(url+"/v1/completions",body,{"Content-Type":"application/json"}),timeout=600))
        return r['name'], d.get('tokens'), None
    except Exception as e:
        return r['name'], None, str(e)
with cf.ThreadPoolExecutor(len(jobs)) as ex:
    res=list(ex.map(run,jobs))
refs={r['name']:r['greedy_token_ids'] for r in ref['results']}
ok=0
for n,t,e in res:
    exact = t is not None and t[:len(refs[n])]==refs[n]
    ok+=exact
    print(json.dumps({"name":n,"exact":exact,"error":e,"first_div":None if t is None else next((i for i,(a,b) in enumerate(zip(t,refs[n])) if a!=b),None)}))
print(json.dumps({"streams":len(jobs),"exact":ok}))
