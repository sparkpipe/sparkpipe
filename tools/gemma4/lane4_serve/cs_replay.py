import json, sys, urllib.request, concurrent.futures as cf
sys.path.insert(0, "/home/spec/g4compsec/tools")
import compsec17 as c
from pathlib import Path
dec = c.load_decoder(Path("/home/spec/g4compsec/qualification/ds4_eval/tokenizer/glm-5.3-flash-tokenizer.json"))
fx = json.load(open("/home/spec/g4compsec/qualification/ds4_eval/quality-fixtures-glm5.3-flash.json"))
cases = {x["id"]: x for x in fx["cases"]}
ids = sys.argv[1].split(","); conc = int(sys.argv[2]); mt = int(sys.argv[3])
def go(i):
    p = c.build_prompt(dec(cases[i]["prompt_token_ids"]), "off", "gemma4")
    d = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8436/v1/completions", json.dumps({"prompt": p, "max_tokens": mt, "temperature": 0}).encode(), {"Content-Type": "application/json"}), timeout=900))
    return i, d["usage"], d["tokens"][:12], d["choices"][0]["text"][-160:]
with cf.ThreadPoolExecutor(conc) as ex:
    for r in ex.map(go, ids): print(json.dumps(r))
