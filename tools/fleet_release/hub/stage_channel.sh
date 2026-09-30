set -euo pipefail
W="$HOME/release-assemble-$S7"
C="$HOME/$RELEASE_API_CHANNEL"
D="$HOME/api-build-$S7"
mkdir -p "$W/channel"
[ "$(sha256sum < "$C/model_resident.json" | cut -c1-16)" = "$CHANNEL_DEPLOYMENT" ] || { echo live-channel-deployment-is-not-production; exit 1; }
[ ! -e "$W/channel/model_resident.json" ] || { echo staged-channel-deployment-exists; exit 1; }
python3 - "$C" "$D" "$W/channel/model_resident.json" "$CHANNEL_ADD" <<'PY'
import hashlib, json, os, sys
channel, build, out, spec = sys.argv[1:5]
src = open(channel + "/model_resident.json").read()
current = json.loads(src)
assert json.dumps(current, indent=1) == src, "live channel file is not json indent=1 without a trailing newline"
tokenizer = json.load(open(channel + "/runtime/" + current["tokenizer"]["path"]))
new = dict(current)
for item in ([] if spec == "none" else spec.split()):
    key, _, rest = item.partition("=")
    path, _, sha = rest.rpartition("@")
    assert key and path and len(sha) == 64, f"CHANNEL_ADD entry {item!r}: want key=path@sha256"
    assert key not in current, f"live channel already declares {key}"
    data = open(os.path.join(build, path), "rb").read()
    assert hashlib.sha256(data).hexdigest() == sha, f"{path}: sha256 differs from {sha[:16]}"
    block = json.loads(data)
    for marker in block.get("stop_markers", []) if isinstance(block, dict) else []:
        hits = [t["id"] for t in tokenizer.get("added_tokens", []) if t.get("special") and t.get("content") == marker]
        assert len(hits) == 1, (marker, hits)
        print(f"{key}: stop marker {marker} -> special token id {hits[0]}")
    new[key] = block
text = json.dumps(new, indent=1)
back = json.loads(text)
for key in list(back):
    if key not in current:
        back.pop(key)
assert back == current and list(back) == list(current)
open(out + ".tmp", "w").write(text)
os.replace(out + ".tmp", out)
print("staged channel deployment = live channel + " + (spec if spec != "none" else "nothing"))
PY
echo "staged $W/channel/model_resident.json $(sha256sum < "$W/channel/model_resident.json" | cut -d' ' -f1)"
