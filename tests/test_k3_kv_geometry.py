"""The serving tier and the kernel tier must agree about the model. The K3
kv geometry header supplies the cache machinery's numbers; the kernel
config is the ground truth those numbers describe. Drift between them
builds a cache of the wrong shape, which is a corruption engine, so the
gate reads both headers and refuses the disagreement."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def defines(path):
    text = (ROOT / path).read_text().replace("\\\n", " ")
    values = {}
    for name, value in re.findall(
            r"#define (K3_\w+|SPARK_K3_KV_\w+)[ \t]+([^\n]+)", text):
        expression = re.sub(r"\(u?int\d+_t\)", "", value.strip())
        expression = re.sub(r"(?<=\d)ull|(?<=\d)u", "", expression)
        expression = re.sub(r"K3_\w+|SPARK_K3_KV_\w+",
                            lambda m: str(values.get(m.group(0), 0)),
                            expression)
        try:
            values[name] = int(eval(expression))
        except Exception:
            pass
    return values


def main():
    geometry = defines("model-families/k3/include/sparkpipe/spark_k3_kv_geometry.h")
    config = defines("inference/llms/kimi_k3/config.h")
    failures = 0
    pairs = [
        ("SPARK_K3_KV_LATENT_DIMENSION", "K3_KV_LORA_RANK"),
        ("SPARK_K3_KV_ROPE_DIMENSION", "K3_QK_UNROTATED_DIM"),
        ("SPARK_K3_KV_KDA_HEADS", "K3_KDA_HEADS"),
        ("SPARK_K3_KV_KDA_KEY_DIM", "K3_KDA_KEY_DIM"),
        ("SPARK_K3_KV_KDA_VALUE_DIM", "K3_KDA_VALUE_DIM"),
        ("SPARK_K3_KV_KDA_CONV_KERNEL", "K3_KDA_CONV_KERNEL"),
    ]
    for serving, kernel in pairs:
        if serving not in geometry or kernel not in config:
            print(f"  FAIL {serving} or {kernel} missing from its header")
            failures += 1
        elif geometry[serving] != config[kernel]:
            print(f"  FAIL {serving}={geometry[serving]} but "
                  f"{kernel}={config[kernel]}; the tiers disagree")
            failures += 1
    layers = config.get("K3_LAYERS", 0)
    kda = geometry.get("SPARK_K3_KV_KDA_LAYER_COUNT", 0)
    mla = geometry.get("SPARK_K3_KV_MLA_LAYER_COUNT", 0)
    if kda + mla != layers:
        print(f"  FAIL {kda} KDA + {mla} MLA != {layers} layers")
        failures += 1
    period = 0  # the schedule is every-4th-plus-last, checked by the split
    if mla != (layers // 4) + 1:
        print(f"  FAIL MLA count {mla} != every-4th-plus-last of {layers}")
        failures += 1
    state = geometry.get("SPARK_K3_KV_KDA_STATE_BYTES_PER_LAYER", 0)
    conv = geometry.get("SPARK_K3_KV_KDA_CONV_BYTES_PER_LAYER", 0)
    if state != config.get("K3_KDA_STATE_SLOT_BYTES", -1):
        print(f"  FAIL state slab {state} != kernel K3_KDA_STATE_BYTES "
              f"{config.get('K3_KDA_STATE_SLOT_BYTES')}")
        failures += 1
    if conv != config.get("K3_KDA_CONV_WINDOW_BYTES", -1):
        print(f"  FAIL conv slab {conv} != kernel K3_KDA_CONV_BYTES "
              f"{config.get('K3_KDA_CONV_WINDOW_BYTES')}")
        failures += 1
    print(f"checked {len(pairs)} dimension pairs, the layer split, "
          f"and both slab formulas")
    if failures:
        print(f"\nFAIL ({failures})")
        return 1
    print("\nthe serving tier and the kernel tier describe the same model")
    return 0


if __name__ == "__main__":
    sys.exit(main())
