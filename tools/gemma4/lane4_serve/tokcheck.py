import json,sys,urllib.request,time
ref=json.load(open(sys.argv[1])); url=sys.argv[2]
out=[]
for r in ref['results']:
    body=json.dumps({"prompt_token_ids":r['prompt_token_ids'],"max_tokens":32,"temperature":0}).encode()
    t=time.time()
    resp=json.load(urllib.request.urlopen(urllib.request.Request(url+"/v1/completions",body,{"Content-Type":"application/json"}),timeout=300))
    dt=time.time()-t
    toks=resp.get('tokens') or resp['choices'][0].get('token_ids')
    g=r['greedy_token_ids']
    n=min(len(toks),len(g)); first=next((i for i in range(n) if toks[i]!=g[i]),None)
    exact = first is None and len(toks)>=len(g)
    print(json.dumps({"name":r['name'],"exact":exact,"first_divergence":first,"got":toks,"ref":g,"text":resp['choices'][0]['text'],"seconds":round(dt,2)}))
