"""Gate for tools/generate_recipe.py and the DATAFILE_NAMING grammar.

Four properties, all host-checkable:

1. determinism - the same contracts and topology produce the same bytes and
   the same datafile names, twice in a row, for every model/strategy/degree.
2. geometry-hash invalidation - mutating a KV-content field (a latent
   width) mints a new geometry-hash; mutating a non-KV field (expert count)
   leaves it standing but moves the recipe's content-hash; TP16 and PP16 of
   one model share a geometry-hash, which is the TP16<->PP16 NVMe-resume
   safety property docs/DATAFILE_NAMING.md promises.
3. TP shard coverage - every class the k3 slicer (tools/k3_shard.py) knows
   is classified by the recipe exactly once and identically; match lists
   are pairwise disjoint; a sharded class's rows partition exactly
   (per_rank * degree == extent, no gap, no overlap); anything
   indivisible is marked replicated with a reason, never silently split.
4. PP stage placement - stages are contiguous and cover every layer, the
   dense prefix stays whole in stage zero, no stage exceeds
   MAX_ROUTED_PER_STAGE routed layers, the final-token stage is last, and
   the DP is optimal: on small synthetic geometries it matches a
   brute-force search over every legal composition, and on the real
   recipes the reported balance equals the DP optimum.
"""
import itertools
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import generate_recipe as gr  # noqa: E402
import k3_shard  # noqa: E402

FAILURES = []


def check(condition, message):
    if not condition:
        FAILURES.append(message)
        print(f"FAIL {message}")


def load_contract(tag):
    path = gr.CONTRACTS / gr.MODELS[tag]["contract"]
    return json.loads(path.read_text(encoding="utf-8"))


def build(tag, strategy, degree, contract=None):
    contract = contract if contract is not None else load_contract(tag)
    spec = gr.MODELS[tag]
    geometry = spec["adapter"](contract)
    topology = {"source": "test", "name": None, "mode": None,
                "nodes": degree, "node_names": [None] * degree}
    return gr.build_recipe(tag, f"model_contracts/{spec['contract']}",
                           "0" * 64, contract, geometry, strategy, degree,
                           topology)


def test_determinism():
    for tag in sorted(gr.MODELS):
        for strategy in ("TP", "PP"):
            for degree in (16, 13):
                first = gr.render(build(tag, strategy, degree))
                second = gr.render(build(tag, strategy, degree))
                check(first == second,
                      f"{tag}.{strategy}{degree}: output is not deterministic")
                recipe = json.loads(first)
                check(recipe["datafile"] ==
                      f"{tag}.{strategy}{degree}.{recipe['content_hash']}.json",
                      f"{tag}.{strategy}{degree}: datafile name does not "
                      f"follow <model>.<strategy><n>.<content-hash>.json")
                check(gr.DATAFILE_RE.match(recipe["datafile"]) is not None,
                      f"{recipe['datafile']}: outside the naming grammar")


def test_geometry_hash_invalidation():
    base = load_contract("k3")
    tp16 = build("k3", "TP", 16)
    pp16 = build("k3", "PP", 16)
    tp13 = build("k3", "TP", 13)
    check(tp16["geometry_hash"] == pp16["geometry_hash"],
          "TP16 and PP16 must share a geometry-hash (NVMe resume across the "
          "strategy switch depends on it)")
    check(tp16["geometry_hash"] == tp13["geometry_hash"],
          "geometry-hash must be degree-free")
    check(tp16["content_hash"] != pp16["content_hash"],
          "TP16 and PP16 are different content and must hash differently")
    check(tp16["kv_entry_prefix"] ==
          f"k3.TP16.{tp16['geometry_hash']}/",
          "kv_entry_prefix must be <model>.<strategy><n>.<geometry-hash>/")

    moved_latent = json.loads(json.dumps(base))
    moved_latent["mla"]["kv_lora_rank"] = 256
    check(build("k3", "TP", 16, moved_latent)["geometry_hash"] !=
          tp16["geometry_hash"],
          "a latent-width change must invalidate the geometry-hash")
    moved_experts = json.loads(json.dumps(base))
    moved_experts["moe"]["routed_expert_count"] = 512
    rebuilt = build("k3", "TP", 16, moved_experts)
    check(rebuilt["geometry_hash"] == tp16["geometry_hash"],
          "expert count does not change KV content; the geometry-hash must "
          "survive it (otherwise every MoE tune would orphan live KV)")
    check(rebuilt["content_hash"] != tp16["content_hash"],
          "expert count changes the recipe; the content-hash must move")
    glm52 = build("glm52","PP",13)
    check(glm52["kv_geometry"]["layout"] == "mla_compressed_bf16" and
          glm52["kv_geometry"]["kv_element_bits"] == 16,
          "GLM 5.2 recipes must carry the package BF16 KV contract")
    check(glm52["pp"]["cut_rules"]["source"] ==
          "tools/generate_recipe.py",
          "PP recipes must name their live placement implementation")


def test_tp_shard_coverage():
    # k3 parity: the recipe table is built from k3_shard's sets, and this
    # walks every name those sets know to prove it stayed that way.
    recipe = build("k3", "TP", 16)
    classes = recipe["tp"]["shard_classes"]
    by_name = {}
    for cls in classes:
        for pattern in cls["match"]:
            check(pattern not in by_name,
                  f"k3: {pattern} classified twice ({by_name.get(pattern)} "
                  f"and {cls['name']})")
            by_name[pattern] = cls
    known = (set(k3_shard.REPLICATED) | set(k3_shard.MODEL_REPLICATED) |
             set(k3_shard.OUTPUT_HEADS) | set(k3_shard.INPUT_HEADS) |
             set(k3_shard.OUTPUT_DIM) | set(k3_shard.INPUT_DIM) |
             set(k3_shard.CONCAT_OUTPUT) | set(k3_shard.INPUT_DIM_PLAIN) |
             set(k3_shard.EXPERT_CONCAT) | set(k3_shard.EXPERT_INPUT) |
             {"model.embed_tokens.weight", "lm_head.weight"})
    missing = sorted(known - set(by_name))
    check(not missing, f"k3: slicer-known tensors missing from the recipe "
                       f"table: {missing}")
    for name in sorted(set(k3_shard.REPLICATED) | set(k3_shard.MODEL_REPLICATED)):
        check(by_name[name]["shard_class"] == "REPLICATED",
              f"k3: {name} must stay REPLICATED")
    check(recipe["tp"]["unclassified_policy"] == "refuse",
          "k3: unclassified tensors must fail closed")

    # every model, every degree: disjoint matches, exact partitions, and an
    # explicit reason wherever the recipe replicates a splittable class.
    for tag in sorted(gr.MODELS):
        for degree in (16, 13):
            recipe = build(tag, "TP", degree)
            seen = set()
            for cls in recipe["tp"]["shard_classes"]:
                for pattern in cls["match"]:
                    check(pattern not in seen,
                          f"{tag}.TP{degree}: {pattern} matched twice")
                    seen.add(pattern)
                resolution = cls["degree_resolution"]
                extent = cls["split_extent"]
                if resolution["status"] == "sharded":
                    check(extent is not None and
                          resolution["rows_per_rank"] * degree == extent,
                          f"{tag}.TP{degree}:{cls['name']}: sharded rows do "
                          f"not partition exactly")
                    if cls["split"] in ("output_heads", "input_heads"):
                        check(cls["head_count"] is not None and
                              cls["head_count"] % degree == 0,
                              f"{tag}.TP{degree}:{cls['name']}: sharded on "
                              f"fractional head blocks")
                else:
                    check(resolution["reason"],
                          f"{tag}.TP{degree}:{cls['name']}: replicated "
                          f"without a recorded reason")
            if tag == "qwen38_27b" and degree == 16:
                # 24 query / 4 KV heads cannot split 16 ways; the recipe
                # must say so, not shard on fractional heads (the bug this
                # regression pins: row divisibility used to masquerade as
                # head divisibility)
                status = {c["name"]: c["degree_resolution"]["status"]
                          for c in recipe["tp"]["shard_classes"]}
                check(status["heads_out:full_q"] == "replicated" and
                      status["heads_out:full_kv"] == "replicated" and
                      status["heads_in:full_o"] == "replicated",
                      "qwen38_27b.TP16: full attention must replicate")
                check(status["heads_out:gdn_qk"] == "sharded" and
                      status["heads_out:gdn_v"] == "sharded",
                      "qwen38_27b.TP16: GDN projections divide 16 and must "
                      "shard")
            # the per-rank table tiles the extent with no overlap or gap
            for cls in recipe["tp"]["shard_classes"]:
                if cls["degree_resolution"]["status"] != "sharded":
                    continue
                covered = sorted(
                    tuple(rank["slices"][cls["name"]])
                    for rank in recipe["tp"]["ranks"])
                expected = [(r * cls["degree_resolution"]["rows_per_rank"],
                             cls["degree_resolution"]["rows_per_rank"])
                            for r in range(degree)]
                check(covered == expected,
                      f"{tag}.TP{degree}:{cls['name']}: rank slices do not "
                      f"tile the extent")
                check(len(recipe["tp"]["ranks"]) == degree,
                      f"{tag}.TP{degree}: rank table incomplete")


def brute_force_optimum(layer_costs, first_routed, stage_count, final_extra):
    layers = len(layer_costs)
    best = None
    for cuts in itertools.combinations(range(1, layers), stage_count - 1):
        bounds = (0,) + cuts + (layers,)
        spans = list(zip(bounds, bounds[1:]))
        if not all(gr.range_valid(first, end - first, first_routed, layers)
                   for first, end in spans):
            continue
        worst = 0
        for index, (first, end) in enumerate(spans):
            segment = sum(layer_costs[first:end])
            if index + 1 == stage_count:
                segment += final_extra
            worst = max(worst, segment)
        if best is None or worst < best:
            best = worst
    return best


def test_pp_stage_rules_and_balance():
    # DP == brute force on small geometries, including one where the dense
    # prefix rule and the routed cap both bind.
    cases = [
        ([10] * 9, 0, 3, 25),
        ([10, 10, 10, 40, 10, 10, 10, 40, 10], 3, 3, 15),
        ([5, 5, 5, 5, 5, 5, 5, 5, 5, 5], 5, 4, 50),
        ([7, 3, 9, 2, 8, 4, 6, 1], 8, 2, 11),
    ]
    for costs, first_routed, stages, final_extra in cases:
        plan, optimum = gr.build_balanced_stages(costs, first_routed, stages,
                                                 final_extra)
        check(optimum == brute_force_optimum(costs, first_routed, stages,
                                             final_extra),
              f"DP optimum mismatch on {costs} s={stages}")
        total = sum(stage["layer_count"] for stage in plan)
        check(total == len(costs), "DP plan does not cover every layer")
    # uniform costs balance by count: stage sizes differ by at most one
    plan, _ = gr.build_balanced_stages([1] * 10, 10, 3, 0)
    sizes = [stage["layer_count"] for stage in plan]
    check(max(sizes) - min(sizes) <= 1,
          f"uniform costs must balance by count, got {sizes}")
    # infeasible placements refuse instead of emitting a plan
    try:
        gr.build_balanced_stages([1] * 20, 0, 2, 0)  # 20 routed > 2*8
        check(False, "an over-cap routed placement must refuse")
    except gr.RecipeFailure:
        pass

    for tag in sorted(gr.MODELS):
        for degree in (16, 13):
            recipe = build(tag, "PP", degree)
            contract = load_contract(tag)
            geometry = gr.MODELS[tag]["adapter"](contract)
            stages = recipe["pp"]["stages"]
            check(len(stages) == degree,
                  f"{tag}.PP{degree}: stage count != degree")
            first = 0
            final_flags = 0
            for index, stage in enumerate(stages):
                check(stage["first_layer_index"] == first,
                      f"{tag}.PP{degree}: stages are not contiguous")
                first += stage["layer_count"]
                check(stage["routed_layer_count"] <= gr.MAX_ROUTED_PER_STAGE,
                      f"{tag}.PP{degree}: stage {index} exceeds the routed "
                      f"cap")
                check(gr.range_valid(stage["first_layer_index"],
                                     stage["layer_count"],
                                     geometry["first_routed_layer"],
                                     geometry["layer_count"]),
                      f"{tag}.PP{degree}: stage {index} violates the cut "
                      f"rules")
                if "FINAL_TOKEN" in stage["flags"]:
                    final_flags += 1
                    check(index + 1 == degree,
                          f"{tag}.PP{degree}: final-token stage is not "
                          f"last")
                elif "OUTPUT_HIDDEN" not in stage["flags"]:
                    check(False, f"{tag}.PP{degree}: non-final stage "
                                 f"{index} must emit hidden state")
            check(first == geometry["layer_count"],
                  f"{tag}.PP{degree}: stages do not cover all layers")
            check(final_flags == 1,
                  f"{tag}.PP{degree}: exactly one final-token stage")
            if geometry["first_routed_layer"] < geometry["layer_count"]:
                check(stages[0]["first_layer_index"] == 0 and
                      stages[0]["layer_count"] >=
                      geometry["first_routed_layer"],
                      f"{tag}.PP{degree}: the dense prefix left stage zero")
            balance = recipe["pp"]["balance"]
            check(balance["max_stage_cost"] == balance["optimum_max_cost"],
                  f"{tag}.PP{degree}: reported balance is not the DP "
                  f"optimum")
            check(balance["max_stage_cost"] == max(s["cost"] for s in stages),
                  f"{tag}.PP{degree}: balance table disagrees with the "
                  f"stages")


def test_weighted_pp_placement():
    stages, optimum = gr.build_balanced_stages(
        [10] * 12, 12, 4, 0, [1.0, 2.0, 1.0, 1.0])
    check([(stage["first_layer_index"], stage["layer_count"])
           for stage in stages] == [(0, 2), (2, 5), (7, 2), (9, 3)],
          "weighted PP placement must give extra work to the faster stage")
    check(optimum == 30.0,
          "weighted PP placement must report normalized minimax cost")
    try:
        gr.build_balanced_stages([1] * 4, 4, 2, 0, [1.0])
    except gr.RecipeFailure:
        pass
    else:
        check(False, "weighted PP placement must reject a short profile")
    capacity = [1.0] * 13
    for index in (4, 5, 6, 7):
        capacity[index] = 1.3
    capacity[-1] = 2.0 / 3.0
    stages, _ = gr.build_balanced_stages([10] * 43, 0, 13, 15, capacity)
    sizes = [stage["layer_count"] for stage in stages]
    check(sizes[-1] == 1,
          "a slow terminal stage must retain only the final layer")
    check(sizes[6] == 5 and sizes[7] == 5,
          "faster interior stages must absorb the shifted work")


def test_check_pattern_and_stale_detection():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        expected = gr.generate_set(["k3"], ["TP", "PP"], [16], None)
        for name, text in expected.items():
            (out_dir / name).write_text(text, encoding="utf-8")
        check(gr.managed_files(out_dir, ["k3"]) == expected,
              "written recipes must read back byte-identical")
        # a stale file from an older content-hash is flagged by name set
        stale = dict(expected)
        stale_name = "k3.TP16." + "0" * 16 + ".json"
        stale[stale_name] = "{}"
        for name in stale:
            if name not in expected:
                (out_dir / name).write_text(stale[name], encoding="utf-8")
        actual = gr.managed_files(out_dir, ["k3"])
        check(stale_name in actual and stale_name not in expected,
              "a stale datafile name must be visible to --check")
        # and an edited recipe is flagged by content
        edited = json.loads(expected[sorted(expected)[0]])
        edited["degree"] = 99
        (out_dir / sorted(expected)[0]).write_text(
            json.dumps(edited, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        actual = gr.managed_files(out_dir, ["k3"])
        name = sorted(expected)[0]
        check(actual[name] != expected[name],
              "an edited recipe must differ from the regenerated bytes")


def main():
    test_determinism()
    test_geometry_hash_invalidation()
    test_tp_shard_coverage()
    test_pp_stage_rules_and_balance()
    test_weighted_pp_placement()
    test_check_pattern_and_stale_detection()
    if FAILURES:
        print(f"\n{len(FAILURES)} recipe-generation checks failed")
        return 1
    print("PASS recipe generation, naming, shard coverage, stage balance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
