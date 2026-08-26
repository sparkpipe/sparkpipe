#!/usr/bin/env python3
"""Offline semantic invariants for the full SparkPipe program PERT graph."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import program_pert
import oxalpha_fleet


class ProgramPertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = program_pert.validate_and_schedule(program_pert.build_tasks())
        cls.tasks = cls.graph["tasks"]
        cls.by_id = {task["id"]: task for task in cls.tasks}

    def test_graph_is_large_complete_and_acyclic(self) -> None:
        self.assertEqual(len(self.tasks), 423)
        self.assertEqual(len(self.by_id), len(self.tasks))
        self.assertEqual(self.graph["summary"]["workstream_count"], 25)
        self.assertEqual(self.graph["summary"]["pairable_task_count"], 394)
        self.assertGreaterEqual(self.graph["summary"]["unconstrained_peak_pairs"], 32)
        self.assertGreater(self.graph["summary"]["unconstrained_critical_path_days"], 0)
        for task in self.tasks:
            self.assertTrue(task["title"].strip(), task["id"])
            self.assertTrue(task["acceptance"].strip(), task["id"])
            estimate = task["estimate_days"]
            self.assertLessEqual(estimate["optimistic"], estimate["most_likely"])
            self.assertLessEqual(estimate["most_likely"], estimate["pessimistic"])
            self.assertGreaterEqual(task["variance_days2"], 0)
            for dependency in task["dependencies"]:
                self.assertIn(dependency, self.by_id, task["id"])

    def test_every_runner_task_has_pert_provenance(self) -> None:
        platform = json.loads(
            (ROOT / "orchestration" / "platform_tasks.json").read_text()
        )
        for task in platform["tasks"]:
            parent = task.get("source_task_id", task["id"])
            self.assertIn(parent, self.by_id, task["id"])

    def test_confirmed_decisions_and_agent_redundancy_are_exact(self) -> None:
        decisions = self.graph["decisions"]
        self.assertEqual(decisions["large_qwen_name"], "Qwen 3.8 Max")
        self.assertIn("BF16/FP16", decisions["compute_precision"])
        self.assertIn("FP32 accumulation", decisions["compute_precision"])
        self.assertIn("MiniMax H3", decisions["initial_models"])
        self.assertIn("Apple Silicon Metal", decisions["hardware_backends"])
        self.assertIn("capacity actually sold", decisions["provider_fee"])
        self.assertIn("rather than cash", decisions["provider_fee"])
        self.assertEqual(decisions["agent_redundancy"], {"implementer": 2, "auditor": 2})
        self.assertIn("different failure-domain backup", decisions["agent_continuation"])
        driver_policy = decisions["model_driver_agent_policy"]
        self.assertTrue(driver_policy["dedicated_logical_pair_per_model"])
        self.assertTrue(driver_policy["persistent_provider_neutral_context"])
        self.assertTrue(driver_policy["independent_auditor_context"])
        self.assertEqual(len(decisions["model_driver_lanes"]), 7)
        self.assertEqual(decisions["sota_release_policy"]["maximum_age_hours"], 24)
        self.assertTrue(decisions["sota_release_policy"]["parity_required"])
        self.assertEqual(decisions["sota_release_policy"]["economic_target_ratio"], 1.10)

    def test_required_product_backend_ui_and_model_programs_exist(self) -> None:
        required_workstreams = {
            "agents", "amd", "api", "artifacts", "business", "collectives",
            "cuda", "hardware", "kv", "metal", "metering", "models",
            "observability", "performance", "providers", "recipes",
            "reliability", "runtime", "scheduler", "security", "topology", "ui",
        }
        self.assertTrue(required_workstreams.issubset(self.graph["workstreams"]))
        self.assertEqual(len([task for task in self.tasks if task["workstream"] == "metal"]), 15)
        for task_id in ("UI-C06", "UI-P06", "UI-O02", "SDK-001", "SDK-002"):
            self.assertIn(task_id, self.by_id)
        for prefix in ("Q27", "D4F", "GLM", "K3", "D4P", "QMAX", "H3"):
            for package in range(1, 18):
                task_id = f"MOD-{prefix}-{package:03d}"
                self.assertIn(task_id, self.by_id)
                self.assertEqual(
                    self.by_id[task_id]["agent_lane"],
                    f"model-driver:{prefix.lower()}",
                )
        lanes = {lane["task_prefix"]: lane for lane in self.graph["decisions"]["model_driver_lanes"]}
        self.assertEqual(set(lanes), {f"MOD-{prefix}" for prefix in ("Q27", "D4F", "GLM", "K3", "D4P", "QMAX", "H3")})
        self.assertTrue(all(lane["task_count"] == 17 for lane in lanes.values()))

    def test_platform_model_source_lanes_have_disjoint_write_sets(self) -> None:
        graph = oxalpha_fleet.load_task_graph(ROOT / "orchestration" / "platform_tasks.json")
        source_ids = [
            "MOD-Q27-001", "MOD-D4F-001", "MOD-GLM-001", "MOD-K3-001",
            "MOD-D4P-001", "MOD-QMAX-001", "MOD-H3-001",
        ]
        tasks = {task["id"]: task for task in graph["tasks"]}
        for offset, left_id in enumerate(source_ids):
            for right_id in source_ids[offset + 1:]:
                self.assertFalse(
                    oxalpha_fleet.write_sets_overlap(
                        tasks[left_id]["write_set"], tasks[right_id]["write_set"]
                    ),
                    f"{left_id} collides with {right_id}",
                )

    def test_topology_storage_and_selectable_kv_are_concrete(self) -> None:
        topology = self.by_id["RCP-019"]["acceptance"]
        self.assertIn("TP/PP/EP", topology)
        self.assertIn("1/2/4/8/16", topology)
        self.assertIn("deterministically builds", topology)
        kv_schema = self.by_id["KV-017"]["acceptance"]
        for format_name in ("BF16", "FP16", "FP8 E4M3", "FP8 E5M2", "INT8"):
            self.assertIn(format_name, kv_schema)
        kv_cells = self.by_id["KV-020"]["acceptance"]
        self.assertIn("B1-B1024", kv_cells)
        self.assertIn("256K", kv_cells)
        self.assertIn("no Cartesian", kv_cells)
        kv_gate = self.by_id["KV-016"]["acceptance"]
        self.assertIn("2.5 TB rank-local", kv_gate)
        self.assertIn("95%", kv_gate)
        self.assertIn("no active-token backing reads", kv_gate)
        budget = self.by_id["ART-017"]
        self.assertIn("Before any fleet placement move", budget["acceptance"])
        self.assertIn("ART-017", self.by_id["ART-003"]["dependencies"])
        self.assertNotIn("ART-010", budget["dependencies"])
        self.assertIn("ART-017", self.by_id["ART-014"]["dependencies"])
        self.assertIn("Ceph 14+2", self.by_id["ART-011"]["title"])

    def test_capacity_fee_is_in_kind_and_excludes_idle_capacity(self) -> None:
        mint = self.by_id["MTR-009"]["acceptance"]
        sold = self.by_id["PRV-006"]["acceptance"]
        legal = self.by_id["BUS-001"]["acceptance"]
        self.assertIn("10%", mint)
        self.assertIn("in kind", mint)
        self.assertIn("idle", mint)
        self.assertIn("excluding idle", sold)
        self.assertIn("ten-percent in-kind/no-idle/no-cash", legal)

    def test_production_super_sink_contains_every_required_task(self) -> None:
        self.assertEqual(self.graph["sinks"], ["MS-009"])
        closure = {"MS-009"}
        pending = ["MS-009"]
        while pending:
            current = pending.pop()
            for dependency in self.by_id[current]["dependencies"]:
                if dependency not in closure:
                    closure.add(dependency)
                    pending.append(dependency)
        required = {task["id"] for task in self.tasks if task["required_for_release"]}
        self.assertEqual(required, closure)
        self.assertEqual(self.graph["summary"]["required_release_closure_count"], len(self.tasks))

    def test_phases_follow_dependencies(self) -> None:
        for task in self.tasks:
            for dependency in task["dependencies"]:
                self.assertLessEqual(self.by_id[dependency]["phase"], task["phase"], task["id"])

    def test_dispatch_is_gated_and_uses_live_state_overlay(self) -> None:
        policy = self.graph["dispatch_policy"]
        self.assertEqual(policy["broad_pair_gate"], "OXA-012")
        self.assertTrue(policy["live_state_overlay_required"])
        self.assertEqual(policy["provider_request_slots_per_pairable_task"], 4)
        self.assertEqual(policy["minimum_independent_provider_failure_domains"], 2)
        self.assertEqual(policy["provider_supply_freshness_hours"], 24)
        self.assertTrue(policy["model_driver_lane_affinity_required"])
        self.assertIn("independently qualified provider failure domains", policy["dispatchable_when"])
        self.assertIn("patch_sealed", policy["states"])
        self.assertIn("auditing", policy["states"])
        self.assertIn("integrated", policy["states"])
        for task in self.tasks:
            if task["pairable"] and task["dispatch_class"] == "paired_after_oxa":
                self.assertIn("OXA-012", task["dispatch_prerequisites"], task["id"])
                self.assertTrue(task["dispatch_contract_required"], task["id"])
            if task["pairable"]:
                self.assertTrue(task["write_locks"], task["id"])
                self.assertNotEqual(task["resource"], "coordinator", task["id"])
                self.assertGreaterEqual(task["provider_request_slots"], 4, task["id"])
                self.assertEqual(task["provider_failure_domains_required"], 2, task["id"])
            else:
                self.assertEqual(task["provider_request_slots"], 0, task["id"])
                self.assertEqual(task["provider_failure_domains_required"], 0, task["id"])
        self.assertEqual(self.by_id["OXA-011"]["provider_request_slots"], 128)
        self.assertEqual(program_pert.PLANNING_CAPACITY["api_provider_request"], 128)
        points = sorted(
            {task["resource_start_day"] for task in self.tasks}
            | {task["resource_finish_day"] for task in self.tasks}
        )
        peak_provider_slots = max(
            sum(
                task["provider_request_slots"]
                for task in self.tasks
                if task["resource_start_day"] <= point < task["resource_finish_day"]
            )
            for point in points
        )
        self.assertEqual(peak_provider_slots, 128)
        self.assertEqual(self.graph["planning_roots"], ["FND-004", "OXA-001", "PERF-001"])

    def test_hardware_requirements_are_typed_and_resource_forecast_exists(self) -> None:
        for task in self.tasks:
            self.assertTrue(task["hardware_requirements"], task["id"])
            for requirement in task["hardware_requirements"]:
                self.assertIn(requirement["pool"], program_pert.PLANNING_CAPACITY)
                self.assertGreaterEqual(requirement["quantity"], 1)
                self.assertLessEqual(
                    requirement["quantity"],
                    program_pert.PLANNING_CAPACITY[requirement["pool"]],
                )
            self.assertGreaterEqual(task["resource_start_day"], task["earliest_start_day"])
            self.assertGreaterEqual(task["resource_finish_day"], task["resource_start_day"])
        self.assertGreater(
            self.graph["summary"]["resource_constrained_forecast_days"],
            self.graph["summary"]["unconstrained_critical_path_days"],
        )

    def test_zero_provider_slots_do_not_delay_non_pairable_work(self) -> None:
        by_id = {
            "PAIR": {
                "dependencies": [], "dispatch_prerequisites": [],
                "resource": "oxalpha_pair", "hardware_requirements": [],
                "write_locks": ["model:pair"], "provider_request_slots": 128,
            },
            "COORD": {
                "dependencies": [], "dispatch_prerequisites": [],
                "resource": "coordinator", "hardware_requirements": [],
                "write_locks": ["shared-coord"], "provider_request_slots": 0,
            },
        }
        starts, finishes = program_pert.resource_constrained_schedule(
            ["PAIR", "COORD"], by_id, {"PAIR": 5.0, "COORD": 1.0}
        )
        self.assertEqual(starts["PAIR"], 0.0)
        self.assertEqual(starts["COORD"], 0.0)
        self.assertEqual(finishes["COORD"], 1.0)

    def test_model_topologies_are_feasible_and_performance_is_fresh(self) -> None:
        expected_minimum = {
            "Q27": "cuda1", "D4F": "cuda4", "GLM": "capacity_selected",
            "K3": "cuda16", "D4P": "cuda16", "QMAX": "cuda16",
            "H3": "capacity_selected",
        }
        for prefix, hardware in expected_minimum.items():
            minimum = self.by_id[f"MOD-{prefix}-005"]
            self.assertIn("minimum-legal-topology", minimum["title"])
            self.assertEqual(minimum["hardware"], hardware)
            release = self.by_id[f"MOD-{prefix}-008"]
            self.assertIn(f"MOD-{prefix}-017", release["dependencies"])
            self.assertEqual(release["freshness_hours"], 24)
            closure = set(release["dependencies"])
            pending = list(closure)
            while pending:
                current = pending.pop()
                for dependency in self.by_id[current]["dependencies"]:
                    if dependency not in closure:
                        closure.add(dependency)
                        pending.append(dependency)
            self.assertNotIn("PERF-009", closure, prefix)
            sota = self.by_id[f"MOD-{prefix}-017"]["acceptance"]
            self.assertIn("<=24-hour", sota)
            self.assertIn("at least matches SOTA", sota)
            self.assertIn("separate sold-capacity economics gate", sota)
        self.assertNotEqual(self.by_id["MOD-Q27-006"]["hardware"], "cuda1")
        self.assertIn("110%", self.by_id["PERF-009"]["acceptance"])

    def test_critical_metadata_is_a_real_dependency_path(self) -> None:
        path = self.graph["representative_critical_path"]
        self.assertGreater(len(path), 10)
        self.assertIn(path[0], self.graph["roots"])
        self.assertEqual(path[-1], "MS-009")
        for predecessor, successor in zip(path, path[1:]):
            self.assertIn(predecessor, self.by_id[successor]["dependencies"])
            self.assertAlmostEqual(
                self.by_id[predecessor]["earliest_finish_day"],
                self.by_id[successor]["earliest_start_day"],
                places=3,
            )
        self.assertAlmostEqual(
            self.by_id[path[-1]]["earliest_finish_day"],
            self.graph["summary"]["unconstrained_critical_path_days"],
            places=1,
        )
        self.assertEqual(
            set(self.graph["critical_tasks"]),
            {task["id"] for task in self.tasks if task["critical"]},
        )

    def test_current_state_and_recurring_work_do_not_claim_completion(self) -> None:
        states = {task["planning_state"] for task in self.tasks}
        self.assertNotIn("complete", states)
        self.assertNotIn("completed", states)
        self.assertEqual(self.by_id["PERF-001"]["recurring_days"], 1)
        self.assertEqual(self.by_id["PERF-001"]["freshness_hours"], 24)
        self.assertEqual(self.by_id["PERF-007"]["recurring_days"], 1)
        self.assertEqual(self.by_id["OXA-015"]["recurring_days"], 1)
        architecture = (ROOT / "docs" / "PLATFORM_ARCHITECTURE.md").read_text()
        pert_doc = (ROOT / "docs" / "PLATFORM_PERT.md").read_text()
        self.assertNotIn("one-rank full model", architecture)
        self.assertNotIn("one-rank model", pert_doc)

    def test_committed_json_is_byte_exact_generated_output(self) -> None:
        expected = json.dumps(self.graph, indent=2, sort_keys=True) + "\n"
        committed = (ROOT / "orchestration" / "program_pert.json").read_text()
        self.assertEqual(committed, expected)

    def test_visualization_fragment_is_self_contained_and_bounded(self) -> None:
        fragment = program_pert.render_visualization(self.graph)
        self.assertIn('id="sp-program-pert"', fragment)
        self.assertNotIn("__PERT_DATA__", fragment)
        self.assertNotIn("<html", fragment.lower())
        self.assertNotIn("<body", fragment.lower())
        self.assertNotIn("fetch(", fragment)
        self.assertLess(len(fragment.encode()), 2_000_000)
        marker = '<script type="application/json" id="sp-pert-data">'
        encoded = fragment.split(marker, 1)[1].split("</script>", 1)[0]
        embedded = json.loads(encoded)
        self.assertEqual(embedded["summary"], self.graph["summary"])
        self.assertEqual(len(embedded["tasks"]), 423)
        sample = embedded["tasks"][0]
        for field in (
            "slack_days", "resource", "write_locks", "hardware_requirements",
            "successors", "recurring_days", "pairable", "dispatch_prerequisites",
            "resource_start_day", "resource_finish_day", "provider_request_slots",
            "provider_failure_domains_required",
        ):
            self.assertIn(field, sample)


if __name__ == "__main__":
    unittest.main()
