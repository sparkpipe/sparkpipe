#!/usr/bin/env python3
import json
import random
import re
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import long_prompt_suite as suite

HAYSTACK = "".join(f"Line {index} of the haystack says nothing in particular about anything.\n" for index in range(4000))


def encode(text):
    return [ord(character) for character in text]


def decode(ids):
    return "".join(chr(value) for value in ids)


def count(text):
    return len(text)


def built(family, length=60000, seed=7):
    rng = random.Random(seed)
    if family in ("mk", "mv"):
        return suite.build_mk(rng, length, count, encode(HAYSTACK), decode, family == "mv")
    if family == "mk3":
        return suite.build_mk3(rng, length, count)
    return suite.build_vt(rng, length, count)


def perfect(family, items):
    lines = []
    for item in items:
        if family == "mk":
            lines.append(f"{item['n']}. {item['key']}: {item['gold'][0]}")
        elif family == "mv":
            lines.append(f"{item['n']}. {item['key']}: {', '.join(item['gold'])}")
        elif family == "mk3":
            lines.append(f"{item['n']}. {item['gold'][0]}")
        else:
            lines.append(f"{item['n']}. {', '.join(item['gold'])}")
    return "\n".join(lines)


class BuildTest(unittest.TestCase):
    def test_every_family_holds_its_answers_and_fills_its_length(self):
        for family in suite.FAMILIES:
            intro, context, question, items = built(family)
            content = intro + context + question
            self.assertLess(abs(len(content) - 60000), 0.01 * 60000, family)
            self.assertEqual(len(items), suite.VT_CHAINS if family == "vt" else suite.QUESTIONS, family)
            for item in items:
                self.assertIn(f"{item['n']}. {item['key']}", question, family)
                for value in item["gold"]:
                    self.assertIn(value, context, family)
                self.assertTrue(0.0 <= item["depth"] <= 1.0, family)

    def test_multikey_needles_name_each_value_once_and_add_distractors(self):
        _, context, _, items = built("mk")
        needles = re.findall(r"One of the special magic numbers for (\w+) is: (\d+)\.", context)
        self.assertEqual(len(needles), suite.MK_NEEDLES)
        self.assertEqual(len({key for key, _ in needles}), suite.MK_NEEDLES)
        for item in items:
            self.assertEqual([value for key, value in needles if key == item["key"]], item["gold"])

    def test_multivalue_keys_have_four_values(self):
        _, context, _, items = built("mv")
        for item in items:
            values = re.findall(rf"One of the special magic numbers for {item['key']} is: (\d+)\.", context)
            self.assertEqual(sorted(values), sorted(item["gold"]))
            self.assertEqual(len(values), suite.MV_VALUES)

    def test_variable_chains_assign_in_order(self):
        _, context, _, items = built("vt")
        for item in items:
            names = item["gold"]
            first = context.index(f"VAR {names[0]} = {item['key']}.")
            positions = [first] + [context.index(f"VAR {names[index]} = VAR {names[index - 1]}.") for index in range(1, len(names))]
            self.assertEqual(positions, sorted(positions))

    def test_build_is_deterministic(self):
        for family in suite.FAMILIES:
            self.assertEqual(built(family), built(family), family)
            self.assertNotEqual(built(family)[1], built(family, seed=8)[1], family)

    def test_pattern_is_read_from_a_tokenizer_source(self):
        source = 'class T:\n    pat_str = "|".join([\n        r"""[a-z]+""",\n        r"""\\s+""",\n    ])\n'
        self.assertEqual(suite.tiktoken_pattern(source), "[a-z]+|\\s+")
        self.assertEqual(suite.tiktoken_pattern('pat_str = "x+"\n'), "x+")


class ScoreTest(unittest.TestCase):
    def setUp(self):
        self.records = {}
        for family in suite.FAMILIES:
            _, _, _, items = built(family, length=20000)
            self.records[f"{family}-20k-0"] = {"id": f"{family}-20k-0", "family": family, "length": 20000, "items": items}

    def run_rows(self, edit=None):
        rows = {}
        for prompt_id, record in self.records.items():
            content = perfect(record["family"], record["items"])
            if edit:
                content = edit(record, content)
            rows[prompt_id] = {"id": prompt_id, "content": content, "failure": None, "tokens": [ord(c) for c in content],
                               "token_logprobs": [[[ord(c), -0.1], [0, -3.0]] for c in content], "wall_s": 1.0, "prompt_tokens": 20000}
        return rows

    def test_perfect_answers_score_every_question(self):
        rows = suite.score_rows(self.records, self.run_rows())
        self.assertEqual(len(rows), 3 * suite.QUESTIONS + suite.VT_CHAINS)
        self.assertTrue(all(row["right"] for row in rows))

    def test_numbering_styles_and_partial_credit(self):
        record = self.records["mv-20k-0"]
        item = record["items"][0]
        self.assertEqual(suite.answer_lines("**1.** a\n2) b\n- 3: c"), {1: "a", 2: "b", 3: "c"})
        self.assertEqual(suite.item_fraction("mv", item, f"{item['key']}: {item['gold'][0]}, {item['gold'][1]}"), 0.5)
        self.assertEqual(suite.item_fraction("mk", {"gold": ["1234567"]}, "x: 11234567"), 0.0)
        vt = self.records["vt-20k-0"]["items"][0]
        self.assertEqual(suite.item_fraction("vt", vt, ", ".join(vt["gold"][:3])), 0.6)
        uuid_item = self.records["mk3-20k-0"]["items"][0]
        self.assertEqual(suite.item_fraction("mk3", uuid_item, uuid_item["gold"][0].upper()), 1.0)

    def test_a_failed_request_scores_nothing(self):
        rows = self.run_rows()
        rows["mk-20k-0"]["failure"] = "HTTPError: 500"
        scored = suite.score_rows(self.records, rows)
        self.assertFalse(any(row["right"] for row in scored if row["prompt"] == "mk-20k-0"))

    def test_compare_counts_questions_only_one_side_gets(self):
        def wrong_first(record, content):
            if record["family"] != "mk":
                return content
            lines = content.splitlines()
            lines[0] = re.sub(r"\d{7}", "0000000", lines[0])
            return "\n".join(lines)

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            suite_path, reference, arm = folder / "suite.jsonl", folder / "a.jsonl", folder / "b.jsonl"
            suite_path.write_text("".join(json.dumps(record) + "\n" for record in self.records.values()))
            reference.write_text("".join(json.dumps(row) + "\n" for row in self.run_rows().values()))
            arm.write_text("".join(json.dumps(row) + "\n" for row in self.run_rows(wrong_first).values()))
            out = folder / "cmp.json"
            suite.compare(Namespace(suite=str(suite_path), reference=str(reference), arm=str(arm), out=str(out),
                                    seed=suite.SEED, replicates=200))
            report = json.loads(out.read_text())
        whole = report["all"]
        self.assertEqual((whole["only_reference"], whole["only_arm"]), (1, 0))
        self.assertEqual(whole["mcnemar_p"], 1.0)
        self.assertEqual(report["by_family"]["mk"]["arm_right"], suite.QUESTIONS - 1)
        self.assertEqual(sum(1 for row in report["prompts"].values() if row["identical"]), 3)
        self.assertFalse(report["prompts"]["mk-20k-0"]["identical"])
        self.assertLess(report["prompts"]["mk-20k-0"]["shared_tokens"], report["prompts"]["mk-20k-0"]["reference_tokens"])
        self.assertEqual([row["item"] for row in report["items"]], ["mk-20k-0:1"])


if __name__ == "__main__":
    unittest.main()
