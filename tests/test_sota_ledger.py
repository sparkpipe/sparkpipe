#!/usr/bin/env python3
"""Tests for the daily primary-source SOTA scanner and comparable-target ledger.

Deterministic by construction: the pipeline runs offline against the captured
envelopes under examples/sota_observations/, never touching the network.

Usage:
    python3 tests/test_sota_ledger.py                    # run assertions
    python3 tests/test_sota_ledger.py --update-committed # regenerate
                                                          # performance/sota_ledger.jsonl
                                                          # from fixtures, then verify
"""

import copy
import pathlib
import sys
import tempfile
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import update_sota_ledger as usl

SOURCES = ROOT / "performance" / "sota_sources.json"
FIXTURES = ROOT / "examples" / "sota_observations"
LEDGER = ROOT / "performance" / "sota_ledger.jsonl"
SCHEMA = ROOT / "schema" / "sota_observation.schema.json"
ADVERSARIAL = FIXTURES / "adversarial"

EXPECTED_MODELS = {
    "dsv4-flash",
    "dsv4-pro",
    "glm-5.2",
    "k3",
    "minimax-h3",
    "qwen-3.8-max",
    "qwen-3.8-27b",
}

EXPECTED_TARGET_VALUES = {
    "dsv4-flash": [3432.0],
    "dsv4-pro": [1078.0],
    "glm-5.2": [1595.0],
    "k3": [792.0],
    "minimax-h3": [1826.0],
    "qwen-3.8-max": [2310.0],
    "qwen-3.8-27b": [1045.0, 1100.0, 2640.0],
}


def assert_value_error(callback, fragment):
    try:
        callback()
    except ValueError as error:
        assert fragment in str(error), (fragment, str(error))
    else:
        raise AssertionError(f"expected ValueError containing {fragment!r}")


def make_obs():
    return {
        "schema_version": 1,
        "observation_id": "0123456789abcdef",
        "model_id": "dsv4-flash",
        "checkpoint": {
            "name": "deepseek-ai/DeepSeek-V4-Flash",
            "revision": "3f9c2ab7d1e40552c8bb0f6a19d4e77a2b91c630",
        },
        "quality_mode": "fp8_e4m3_weights_bf16_activations_fp8_kv_cache",
        "precision": {
            "weight_dtype": "fp8_e4m3",
            "weight_codec": "native_fp8_e4m3",
            "activation_dtype": "bf16",
            "accumulator_dtype": "fp32",
            "kv_cache_dtype": "fp8_e4m3",
            "kv_cache_codec": "native_fp8_e4m3",
        },
        "quality_gate": {
            "method": "token_exact",
            "status": "passed",
            "output_parity": True,
            "atol": 0.0,
            "rtol": 0.0,
        },
        "execution": {
            "compute_route": "cuda_native",
            "kv_route": "device_hbm_contiguous",
        },
        "hardware": {"accelerator_name": "NVIDIA H200 SXM", "accelerator_count": 8},
        "fabric": {
            "interconnect": "nvlink4_nvswitch",
            "collective_backend": "nccl",
            "bandwidth_gbps_per_device": 900.0,
            "rdma_enabled": False,
        },
        "power_state": {
            "state": "locked",
            "power_limit_watts": 700.0,
            "graphics_clock_mhz": 1980,
            "memory_clock_mhz": 2619,
            "thermal_state": "steady",
        },
        "topology": {"kind": "tp", "tp_size": 8, "pp_size": 1, "ep_size": 1},
        "batch": {"batch_size": 16},
        "workload": {
            "prompt_tokens": 1024,
            "output_tokens": 512,
            "context_window_tokens": 163840,
            "context_occupancy_pct": 0.6,
        },
        "request_distribution": {
            "kind": "fixed",
            "batch_size_p50": 16,
            "batch_size_p95": 16,
            "prompt_tokens_p50": 1024,
            "prompt_tokens_p95": 1024,
            "output_tokens_p50": 512,
            "output_tokens_p95": 512,
            "context_occupancy_pct_p50": 0.6,
            "context_occupancy_pct_p95": 0.6,
        },
        "prefix_cache": {"enabled": False, "state": "disabled", "matched_tokens": 0},
        "speculation": {"enabled": False, "provider": None, "draft_length": None},
        "service_level_objectives": {"ttft_ms_max": None, "itl_ms_max": None},
        "statistics": {
            "randomization": "none",
            "paired": False,
            "warmup_iterations": 2,
            "sample_count": 10,
            "summary_statistic": "mean",
            "confidence_interval": {
                "method": "bootstrap",
                "level": 0.95,
                "lower": 3090.0,
                "upper": 3150.0,
            },
        },
        "metric": {
            "name": "output_tokens_per_second",
            "value": 3120.0,
            "unit": "tok/s",
            "direction": "higher_is_better",
            "timing_boundary": "steady_state_decode",
        },
        "source": {
            "url": "https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V4/main/benchmarks/v4_flash_decode_throughput_20260601.json",
            "retrieved_utc": "2026-08-20T09:00:00Z",
            "evidence_class": "MEASURED_BENCHMARK_ARTIFACT",
            "publisher": "DeepSeek",
            "document_kind": "github_repository_artifact",
            "publication_date": "2026-06-01T00:00:00Z",
        },
    }


def t_sources_config():
    config = usl.load_sources(SOURCES)
    policy = config["policy"]
    assert abs(policy["higher_is_better_target_factor"] - 1.10) < 1e-12
    assert abs(policy["lower_is_better_target_factor"] - 0.90) < 1e-12
    assert policy["target_selection"] == usl.TARGET_SELECTION
    assert policy["equal_value_tiebreaker"] == usl.TARGET_TIE_BREAKER
    model_ids = {m["model_id"] for m in config["models"]}
    assert model_ids == EXPECTED_MODELS == set(usl.SUPPORTED_MODELS)
    seen = set()
    for model in config["models"]:
        assert model["display_name"] == usl.SUPPORTED_MODELS[model["model_id"]]
        assert len(model["sources"]) >= 1
        eligible = [
            s for s in model["sources"]
            if s["evidence_class"] in usl.TARGET_ELIGIBLE_EVIDENCE_CLASSES
        ]
        assert eligible, f"{model['model_id']} has no target-eligible primary source"
        for source in model["sources"]:
            sid = source["source_id"]
            assert sid not in seen
            seen.add(sid)
            assert source["url"].startswith("https://"), sid
            assert source["extraction"]["format"] in usl.EXTRACTION_FORMATS


def t_schema_and_validator():
    schema = usl.load_schema(SCHEMA)
    assert usl.validate_against_schema(make_obs(), schema) == []

    def rejects(mutate, fragment):
        instance = make_obs()
        mutate(instance)
        errors = usl.validate_against_schema(instance, schema)
        assert errors, f"expected rejection for {fragment}"
        assert any(fragment in error for error in errors), (fragment, errors)

    rejects(lambda o: o.pop("checkpoint"), "required property 'checkpoint'")
    rejects(lambda o: o.pop("source"), "required property 'source'")
    for required_binding in (
        "precision",
        "quality_gate",
        "execution",
        "fabric",
        "power_state",
        "request_distribution",
        "prefix_cache",
        "service_level_objectives",
        "statistics",
    ):
        rejects(
            lambda o, field=required_binding: o.pop(field),
            f"required property '{required_binding}'",
        )
    rejects(lambda o: o.update(model_id="gpt-9"), "is not one of")
    rejects(lambda o: o["workload"].update(context_occupancy_pct=150), "above maximum")
    rejects(lambda o: o["workload"].pop("context_window_tokens"), "context_window_tokens")
    rejects(lambda o: o["speculation"].update(enabled="yes"), "expected type")
    rejects(lambda o: o.update(vendor_claim="fastest ever"), "additional property")
    rejects(lambda o: o["metric"].update(value=0), "exclusiveMinimum")
    rejects(lambda o: o["hardware"].update(accelerator_count=1.5), "expected type")
    for nonfinite in (float("nan"), float("inf"), float("-inf")):
        rejects(
            lambda o, value=nonfinite: o["metric"].update(value=value),
            "non-finite number",
        )

    def drop_dates(o):
        o["source"].pop("publication_date")
    rejects(drop_dates, "anyOf")


def t_comparator_verdicts():
    incumbent = make_obs()
    declared_paths = {
        "precision.weight_codec",
        "quality_gate.output_parity",
        "execution.compute_route",
        "execution.kv_route",
        "fabric.interconnect",
        "fabric.collective_backend",
        "power_state.power_limit_watts",
        "power_state.graphics_clock_mhz",
        "power_state.memory_clock_mhz",
        "power_state.thermal_state",
        "topology.ep_size",
        "request_distribution.batch_size_p50",
        "request_distribution.context_occupancy_pct_p95",
        "prefix_cache.state",
        "speculation",
        "service_level_objectives.ttft_ms_max",
        "service_level_objectives.itl_ms_max",
        "statistics.sample_count",
        "statistics.confidence_interval.method",
        "statistics.confidence_interval.level",
        "metric.timing_boundary",
    }
    assert declared_paths <= set(usl.COMPARE_PATHS)

    comparable = usl.compare_observations(make_obs(), incumbent)
    assert comparable["verdict"] == "COMPARABLE"
    assert comparable["reasons"] == []

    def variant(**changes):
        candidate = make_obs()
        for dotted, value in changes.items():
            usl.set_path(candidate, dotted.replace("__", "."), value)
        return candidate

    partial = usl.compare_observations(
        variant(checkpoint__revision="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"),
        incumbent,
    )
    assert partial["verdict"] == "PARTIAL"
    assert any("checkpoint.revision" in r for r in partial["reasons"])

    partial_batch = usl.compare_observations(
        variant(**{"batch.batch_size": 64}), incumbent
    )
    assert partial_batch["verdict"] == "PARTIAL"
    assert any("batch.batch_size" in r for r in partial_batch["reasons"])

    partial_context = usl.compare_observations(
        variant(**{"workload.context_window_tokens": 327680}), incumbent
    )
    assert partial_context["verdict"] == "PARTIAL"
    assert any("context_window_tokens" in r for r in partial_context["reasons"])

    boundary = usl.compare_observations(
        variant(**{"metric.timing_boundary": "end_to_end_request"}), incumbent
    )
    assert boundary["verdict"] == "INCOMPARABLE"
    assert any("metric.timing_boundary" in r for r in boundary["reasons"])

    metric = usl.compare_observations(
        variant(**{"metric.name": "requests_per_second"}), incumbent
    )
    assert metric["verdict"] == "INCOMPARABLE"

    missing_occ = make_obs()
    del missing_occ["workload"]["context_occupancy_pct"]
    result = usl.compare_observations(missing_occ, incumbent)
    assert result["verdict"] == "INCOMPARABLE"
    assert any(
        "context_occupancy_pct" in r and "refusing to invent" in r
        for r in result["reasons"]
    )

    missing_spec = make_obs()
    del missing_spec["speculation"]
    result = usl.compare_observations(missing_spec, incumbent)
    assert result["verdict"] == "INCOMPARABLE"
    assert any("absent in candidate" in r for r in result["reasons"])

    incumbent_missing = make_obs()
    del incumbent_missing["batch"]
    result = usl.compare_observations(make_obs(), incumbent_missing)
    assert result["verdict"] == "INCOMPARABLE"
    assert any("absent in incumbent" in r for r in result["reasons"])

    base_cell = usl.benchmark_cell_id(incumbent)
    for path, value in (
        ("batch.batch_size", 64),
        ("workload.prompt_tokens", 2048),
        ("workload.context_window_tokens", 327680),
        ("topology.tp_size", 4),
        ("topology.ep_size", 2),
        ("fabric.interconnect", "roce_v2_200gbe"),
        ("fabric.bandwidth_gbps_per_device", 200.0),
        ("power_state.power_limit_watts", 650.0),
        ("power_state.graphics_clock_mhz", 1800),
        ("power_state.memory_clock_mhz", 2500),
        ("power_state.thermal_state", "throttled"),
        ("precision.weight_dtype", "bf16"),
        ("precision.weight_codec", "native_bf16"),
        ("precision.activation_dtype", "fp16"),
        ("precision.accumulator_dtype", "bf16"),
        ("precision.kv_cache_dtype", "bf16"),
        ("precision.kv_cache_codec", "native_bf16"),
        ("execution.compute_route", "cuda_graph"),
        ("execution.kv_route", "host_paged"),
        ("prefix_cache.enabled", True),
        ("prefix_cache.state", "warm"),
        ("prefix_cache.matched_tokens", 512),
        ("quality_gate.atol", 0.01),
        ("request_distribution.batch_size_p95", 32),
        ("service_level_objectives.ttft_ms_max", 100.0),
        ("service_level_objectives.itl_ms_max", 10.0),
        ("statistics.sample_count", 20),
        ("statistics.confidence_interval.method", "student_t"),
        ("statistics.confidence_interval.level", 0.99),
        ("metric.timing_boundary", "end_to_end_request"),
        ("quality_mode", "bf16"),
    ):
        assert usl.benchmark_cell_id(variant(**{path: value})) != base_cell, path


def t_target_math():
    target = usl.compute_target(3120.0, "higher_is_better")
    assert target["improvement_factor"] == 1.10
    assert target["target_relation"] == ">="
    assert abs(target["target_value"] - 3120.0 * 1.10) < 1e-6
    assert target["target_value"] == 3432.0

    lower = usl.compute_target(100.0, "lower_is_better")
    assert lower["improvement_factor"] == 0.90
    assert lower["target_relation"] == "<="
    assert lower["target_value"] == 90.0

    assert usl.better_value(101.0, 100.0, "higher_is_better") is True
    assert usl.better_value(99.0, 100.0, "higher_is_better") is False
    assert usl.better_value(99.0, 100.0, "lower_is_better") is True

    for nonfinite in (float("nan"), float("inf"), float("-inf")):
        assert_value_error(
            lambda value=nonfinite: usl.compute_target(value, "higher_is_better"),
            "finite",
        )
        assert_value_error(
            lambda value=nonfinite: usl.better_value(value, 1.0, "higher_is_better"),
            "finite",
        )
    assert_value_error(
        lambda: usl.compute_target(1.7e308, "higher_is_better"),
        "finite",
    )


def t_fixture_parity():
    config = usl.load_sources(SOURCES)
    configured = {
        source["source_id"]
        for model in config["models"]
        for source in model["sources"]
    }
    envelopes = usl.load_envelopes(FIXTURES)
    captured = {e["source_id"] for e in envelopes}
    assert captured == configured
    files = {p.stem for p in FIXTURES.glob("*.json")}
    assert files == configured


def load_adversarial_cell_fixture():
    fixture_path = ADVERSARIAL / "cell-maxima.json"
    fixture = usl.strict_json_loads(
        fixture_path.read_text(encoding="utf-8"),
        f"fixture {fixture_path.name}",
    )
    envelopes = copy.deepcopy(fixture["envelopes"])
    by_source = {envelope["source_id"]: envelope for envelope in envelopes}
    for derived in fixture["derived_envelopes"]:
        envelope = copy.deepcopy(by_source[derived["from_source_id"]])
        envelope["source_id"] = derived["source_id"]
        for path, value in derived["body_overrides"].items():
            usl.set_path(envelope["body"], path, value)
        envelopes.append(envelope)
        by_source[envelope["source_id"]] = envelope
    return fixture["model_section"], envelopes


def t_adversarial_exact_cell_maxima():
    model_section, envelopes = load_adversarial_cell_fixture()
    envelope_index = {
        envelope["source_id"]: envelope for envelope in envelopes
    }
    records, targets = usl._build_model_records(
        model_section,
        envelope_index,
        "2026-08-25T01:00:00Z",
    )
    assert len(targets) == 3
    assert sorted(target["baseline_value"] for target in targets) == [6.0, 3200.0, 9000.0]
    assert sorted(target["target_value"] for target in targets) == [5.4, 3520.0, 9900.0]

    observations = [record for record in records if record["record_type"] == "observation"]
    throughput = [
        record for record in observations
        if record["observation"]["metric"]["name"] == "output_tokens_per_second"
    ]
    batch16 = [
        record for record in throughput
        if record["observation"]["batch"]["batch_size"] == 16
    ]
    batch64 = next(
        record for record in throughput
        if record["observation"]["batch"]["batch_size"] == 64
    )
    tied = [
        record for record in batch16
        if record["observation"]["metric"]["value"] == 3200.0
    ]
    expected_tie_winner = min(
        record["observation"]["observation_id"] for record in tied
    )
    throughput_target = next(
        target for target in targets
        if target["baseline_value"] == 3200.0
    )
    assert throughput_target["baseline_observation_id"] == expected_tie_winner
    assert len({record["benchmark_cell_id"] for record in batch16}) == 1
    assert batch64["benchmark_cell_id"] != batch16[0]["benchmark_cell_id"]

    latency_target = next(
        target for target in targets
        if target["metric"]["direction"] == "lower_is_better"
    )
    assert latency_target["baseline_value"] == 6.0
    assert latency_target["target_relation"] == "<="
    assert latency_target["target_value"] == 5.4

    reversed_section = copy.deepcopy(model_section)
    reversed_section["sources"].reverse()
    reversed_records, reversed_targets = usl._build_model_records(
        reversed_section,
        envelope_index,
        "2026-08-25T01:00:00Z",
    )
    assert usl.canonical(records) == usl.canonical(reversed_records)
    assert usl.canonical(targets) == usl.canonical(reversed_targets)


def t_full_pipeline_tie_and_lower_is_better():
    config = usl.strict_json_loads(SOURCES.read_text(encoding="utf-8"), "test config")
    model_section, adversarial_envelopes = load_adversarial_cell_fixture()
    flash = next(model for model in config["models"] if model["model_id"] == "dsv4-flash")
    replaced_ids = {source["source_id"] for source in flash["sources"]}
    flash["sources"] = model_section["sources"]
    canonical_envelopes = [
        envelope for envelope in usl.load_envelopes(FIXTURES)
        if envelope["source_id"] not in replaced_ids
    ]
    with tempfile.TemporaryDirectory() as directory:
        directory = pathlib.Path(directory)
        config_path = directory / "sources.json"
        envelope_dir = directory / "envelopes"
        envelope_dir.mkdir()
        config_path.write_text(
            usl.strict_json_dumps(config, context="test config", indent=2) + "\n",
            encoding="utf-8",
        )
        for envelope in canonical_envelopes + adversarial_envelopes:
            path = envelope_dir / f"{envelope['source_id']}.json"
            path.write_text(
                usl.strict_json_dumps(envelope, context=path.name, indent=2) + "\n",
                encoding="utf-8",
            )
        records_one = usl.build_ledger(config_path, envelope_dir)
        records_two = usl.build_ledger(config_path, envelope_dir)
    assert usl.render_ledger(records_one) == usl.render_ledger(records_two)
    targets = [record for record in records_one if record["record_type"] == "target"]
    flash_targets = [target for target in targets if target["model_id"] == "dsv4-flash"]
    assert len(targets) == 11
    assert len(flash_targets) == 3
    assert sorted(target["baseline_value"] for target in flash_targets) == [6.0, 3200.0, 9000.0]
    tied = [
        record for record in records_one
        if record["record_type"] == "observation"
        and record["observation"]["model_id"] == "dsv4-flash"
        and record["observation"]["metric"]["value"] == 3200.0
    ]
    expected_tie_winner = min(record["observation"]["observation_id"] for record in tied)
    tie_target = next(target for target in flash_targets if target["baseline_value"] == 3200.0)
    assert tie_target["baseline_observation_id"] == expected_tie_winner
    lower_target = next(
        target for target in flash_targets
        if target["metric"]["direction"] == "lower_is_better"
    )
    assert lower_target["baseline_value"] == 6.0
    assert lower_target["target_value"] == 5.4


def t_nonfinite_boundaries():
    for fixture_path in sorted(ADVERSARIAL.glob("nonfinite-*.json")):
        with tempfile.TemporaryDirectory() as directory:
            copied = pathlib.Path(directory) / fixture_path.name
            copied.write_text(fixture_path.read_text(encoding="utf-8"), encoding="utf-8")
            assert_value_error(
                lambda path=copied.parent: usl.load_envelopes(path),
                "non-finite JSON number",
            )

    for nonfinite in (float("nan"), float("inf"), float("-inf")):
        assert_value_error(
            lambda value=nonfinite: usl.canonical({"value": value}),
            "serialization failed",
        )
        assert_value_error(
            lambda value=nonfinite: usl.render_ledger([{"value": value}]),
            "serialization failed",
        )


def t_unknown_regex_type_validation():
    base_config = usl.strict_json_loads(SOURCES.read_text(encoding="utf-8"), "test config")
    bad_source_path = ADVERSARIAL / "unknown-regex-type-source.json"
    bad_source = usl.strict_json_loads(
        bad_source_path.read_text(encoding="utf-8"),
        f"fixture {bad_source_path.name}",
    )
    config = copy.deepcopy(base_config)
    k3 = next(model for model in config["models"] if model["model_id"] == "k3")
    k3["sources"] = [bad_source]
    with tempfile.TemporaryDirectory() as directory:
        config_path = pathlib.Path(directory) / "sota_sources.json"
        config_path.write_text(
            usl.strict_json_dumps(config, context="test config", indent=2) + "\n",
            encoding="utf-8",
        )
        assert_value_error(
            lambda: usl.load_sources(config_path),
            "unknown regex extraction type 'decimal128'",
        )

    for malformed in (["float"], {"name": "float"}, 7, True, None):
        config = copy.deepcopy(base_config)
        k3 = next(model for model in config["models"] if model["model_id"] == "k3")
        k3["sources"][0]["extraction"]["types"]["metric.value"] = malformed
        with tempfile.TemporaryDirectory() as directory:
            config_path = pathlib.Path(directory) / "sota_sources.json"
            config_path.write_text(
                usl.strict_json_dumps(config, context="test config", indent=2) + "\n",
                encoding="utf-8",
            )
            try:
                usl.load_sources(config_path)
            except ValueError as error:
                message = str(error)
                assert "model[3]" in message, message
                assert "source[0]" in message, message
                assert "source_id='k3-paper-table-excerpt'" in message, message
                assert "regex extraction type for 'metric.value' must be a string" in message
            else:
                raise AssertionError(f"load_sources accepted malformed regex type {malformed!r}")

    observation, reasons = usl._apply_regex_text(
        {"text": "reached 720 tok/s"},
        bad_source["extraction"],
    )
    assert observation is None
    assert any("unknown regex extraction type 'decimal128'" in reason for reason in reasons)


def t_source_config_errors_are_actionable():
    base = usl.strict_json_loads(SOURCES.read_text(encoding="utf-8"), "test config")

    def load_mutated(config):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "sources.json"
            path.write_text(
                usl.strict_json_dumps(config, context="test config", indent=2) + "\n",
                encoding="utf-8",
            )
            usl.load_sources(path)

    for field in sorted(usl.SOURCE_FIELDS):
        config = copy.deepcopy(base)
        source = config["models"][0]["sources"][0]
        source.pop(field)
        assert_value_error(
            lambda value=config: load_mutated(value),
            f"missing required field '{field}'",
        )

    config = copy.deepcopy(base)
    config["models"][0]["sources"][0]["url"] = None
    assert_value_error(
        lambda: load_mutated(config),
        "model[0] 'dsv4-flash' source[0]",
    )
    assert_value_error(
        lambda: load_mutated(config),
        "field 'url' must be a non-empty string",
    )

    config = copy.deepcopy(base)
    config["models"][0]["sources"][0]["extraction"].pop("format")
    assert_value_error(
        lambda: load_mutated(config),
        "extraction: missing required field 'format'",
    )

    config = copy.deepcopy(base)
    config["models"][0]["sources"][0]["unexpected"] = "typo"
    assert_value_error(
        lambda: load_mutated(config),
        "unknown source fields: ['unexpected']",
    )


def t_qwen_max_naming():
    forbidden = "Qwen 3.8 " + "Pro"
    paths = [
        ROOT / "tools" / "update_sota_ledger.py",
        ROOT / "tests" / "test_sota_ledger.py",
        ROOT / "performance" / "sota_sources.json",
        ROOT / "performance" / "sota_ledger.jsonl",
        ROOT / "docs" / "SOTA_TRACKING.md",
        ROOT / "schema" / "sota_observation.schema.json",
        *sorted(FIXTURES.rglob("*")),
    ]
    for path in paths:
        if path.is_file():
            assert forbidden not in path.read_text(encoding="utf-8"), path
    config = usl.load_sources(SOURCES)
    qwen = next(model for model in config["models"] if model["model_id"] == "qwen-3.8-max")
    assert qwen["display_name"] == "Qwen 3.8 Max"


def t_pipeline_deterministic_and_matches_committed():
    with_network_guard = getattr(usl.urllib.request, "urlopen", None)

    def forbidden(*args, **kwargs):
        raise AssertionError("offline rebuild must not touch the network")

    try:
        usl.urllib.request.urlopen = forbidden
        records_one = usl.build_ledger(SOURCES, FIXTURES)
        records_two = usl.build_ledger(SOURCES, FIXTURES)
    finally:
        usl.urllib.request.urlopen = with_network_guard
    text_one = usl.render_ledger(records_one)
    text_two = usl.render_ledger(records_two)
    assert text_one == text_two
    committed = LEDGER.read_text(encoding="utf-8")
    assert text_one == committed, (
        "committed performance/sota_ledger.jsonl is stale; run "
        "`python3 tests/test_sota_ledger.py --update-committed` after changing "
        "fixtures, sources, or the tool"
    )


def parse_ledger():
    lines = [line for line in LEDGER.read_text(encoding="utf-8").splitlines() if line]
    return [
        usl.strict_json_loads(line, f"committed ledger line {index}")
        for index, line in enumerate(lines, 1)
    ]


def t_ledger_semantics():
    records = parse_ledger()
    assert records[0]["record_type"] == "ledger_header"
    assert records[-1]["record_type"] == "ledger_summary"

    observations = [r for r in records if r["record_type"] == "observation"]
    targets = [r for r in records if r["record_type"] == "target"]
    rejected = [r for r in records if r["record_type"] == "ingestion_rejected"]

    assert len(targets) == sum(len(values) for values in EXPECTED_TARGET_VALUES.values())
    got_targets = {}
    for target in targets:
        got_targets.setdefault(target["model_id"], []).append(target["target_value"])
    got_targets = {model_id: sorted(values) for model_id, values in got_targets.items()}
    assert got_targets == EXPECTED_TARGET_VALUES

    summary = records[-1]
    assert summary["models_with_targets"] == len(EXPECTED_MODELS)
    assert summary["models_without_targets"] == []
    assert summary["target_count"] == len(targets)
    assert summary["ingestion_rejected_count"] == len(rejected) == 1
    assert all(isinstance(entries, list) for entries in summary["targets"].values())

    verdicts = {o["comparison"]["verdict"] for o in observations}
    assert verdicts == {"BASELINE", "COMPARABLE"}

    by_model = {}
    for record in observations:
        by_model.setdefault(record["observation"]["model_id"], []).append(record)

    qwen_cells = {
        observation["benchmark_cell_id"]
        for observation in by_model["qwen-3.8-27b"]
    }
    assert len(qwen_cells) == 3
    qwen_setters = [
        observation for observation in by_model["qwen-3.8-27b"]
        if observation["sets_target"]
    ]
    assert len(qwen_setters) == 3
    assert sorted(
        observation["observation"]["metric"]["value"]
        for observation in qwen_setters
    ) == [950.0, 1000.0, 2400.0]

    assert rejected[0]["source_id"] == "qwen-3.8-27b-truncated-teaser"
    joined = " ".join(rejected[0]["reasons"])
    assert "context_occupancy_pct" in joined
    assert "speculation.provider" in joined

    quarantined = [
        o for o in by_model["dsv4-flash"] if not o["target_eligible"]
    ]
    assert len(quarantined) == 1
    assert quarantined[0]["observation"]["source"]["evidence_class"] == (
        "AGGREGATOR_SECONDARY"
    )
    assert quarantined[0]["sets_target"] is False

    flash_setter = [
        o for o in by_model["dsv4-flash"] if o["sets_target"]
    ]
    assert len(flash_setter) == 1
    assert flash_setter[0]["observation"]["metric"]["value"] == 3120.0
    assert "DeepSeek-V4" in flash_setter[0]["observation"]["source"]["url"]

    for record in observations:
        observation = record["observation"]
        for path in usl.COMPARE_PATHS + (
            "observation_id",
            "metric.value",
            "statistics.confidence_interval.lower",
            "statistics.confidence_interval.upper",
            "source.url",
            "source.retrieved_utc",
            "source.evidence_class",
        ):
            assert usl.get_path(observation, path) is not usl.MISSING, path
        has_date = usl.get_path(observation, "source.publication_date") is not usl.MISSING
        has_revision = usl.get_path(observation, "source.revision") is not usl.MISSING
        assert has_date or has_revision, observation["observation_id"]

    for target in targets:
        assert target["improvement_factor"] == 1.10
        assert target["target_relation"] == ">="
        assert target["metric"]["direction"] == "higher_is_better"
        assert abs(target["target_value"] - target["baseline_value"] * 1.10) < 1e-6
        setter = [
            o for o in observations
            if o["observation"]["observation_id"] == target["baseline_observation_id"]
        ]
        assert len(setter) == 1
        assert setter[0]["comparison"]["verdict"] == "BASELINE"
        assert setter[0]["target_eligible"] is True
        assert setter[0]["benchmark_cell_id"] == target["benchmark_cell_id"]

    by_id = {
        record["observation"]["observation_id"]: record for record in observations
    }
    for record in observations:
        compared_id = record["comparison"]["compared_against_observation_id"]
        if compared_id is None:
            continue
        assert by_id[compared_id]["benchmark_cell_id"] == record["benchmark_cell_id"]
        assert all(
            field["status"] == "match"
            for field in record["comparison"]["fields"].values()
        )

    eligible_cells = {}
    for record in observations:
        if record["target_eligible"]:
            eligible_cells.setdefault(
                (record["observation"]["model_id"], record["benchmark_cell_id"]),
                [],
            ).append(record["observation"])
    targets_by_cell = {
        (target["model_id"], target["benchmark_cell_id"]): target
        for target in targets
    }
    assert set(targets_by_cell) == set(eligible_cells)
    for cell, cell_observations in eligible_cells.items():
        target = targets_by_cell[cell]
        direction = cell_observations[0]["metric"]["direction"]
        values = [observation["metric"]["value"] for observation in cell_observations]
        expected = max(values) if direction == "higher_is_better" else min(values)
        assert target["baseline_value"] == expected


def t_docs():
    doc = (ROOT / "docs" / "SOTA_TRACKING.md").read_text(encoding="utf-8")
    for token in (
        "110%",
        "exact benchmark cell",
        "finite",
        "COMPARABLE",
        "PARTIAL",
        "INCOMPARABLE",
        "quarantine",
        "update_sota_ledger.py",
        "sota_observation.schema.json",
        "offline",
    ):
        assert token.lower() in doc.lower(), token


def main(argv):
    if "--update-committed" in argv:
        records = usl.build_ledger(SOURCES, FIXTURES)
        LEDGER.write_text(usl.render_ledger(records), encoding="utf-8")
        print("regenerated performance/sota_ledger.jsonl from offline fixtures")
    elif LEDGER.exists() is False:
        print(
            "performance/sota_ledger.jsonl missing; run "
            "`python3 tests/test_sota_ledger.py --update-committed`",
            file=sys.stderr,
        )
        return 1

    checks = [
        t_sources_config,
        t_schema_and_validator,
        t_comparator_verdicts,
        t_target_math,
        t_fixture_parity,
        t_adversarial_exact_cell_maxima,
        t_full_pipeline_tie_and_lower_is_better,
        t_nonfinite_boundaries,
        t_unknown_regex_type_validation,
        t_source_config_errors_are_actionable,
        t_qwen_max_naming,
        t_pipeline_deterministic_and_matches_committed,
        t_ledger_semantics,
        t_docs,
    ]
    for check in checks:
        check()
        print(f"ok {check.__name__}")
    print(f"PASS ({len(checks)} groups)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
