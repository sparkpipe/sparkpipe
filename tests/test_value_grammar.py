#!/usr/bin/env python3
"""The value grammar reads numbers, quantities with units, counts, durations, times and dates out of text with a tokenizer and a typed
grammar, never a regular expression: word numbers compose ("one hundred and five", "two and a half"), compound durations add up
("4h 22min", "an hour and a half"), counts keep their noun, and dates resolve against an anchor date, absolute, partial, relative or by
weekday, past unless the sentence speaks of the future, with a named weekday choosing the year. Words that only look like values
("May I", "we march", "the blue one") stay text."""
import ast
import datetime
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "tools" / "serving" / "value_grammar.py"
spec = importlib.util.spec_from_file_location("value_grammar", SOURCE)
vg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vg)


def values(text, anchor=None):
    return [{key: value for key, value in found.items() if key != "span"} for found in vg.read(text, anchor)]


def one(source, anchor=None, **expected):
    found = values(source, anchor)
    assert len(found) == 1 and all(found[0].get(key) == value for key, value in expected.items()), (source, found)
    return found[0]


imports = {alias.name for node in ast.walk(ast.parse(SOURCE.read_text())) if isinstance(node, ast.Import) for alias in node.names}
imports |= {node.module for node in ast.walk(ast.parse(SOURCE.read_text())) if isinstance(node, ast.ImportFrom)}
assert "re" not in imports and "regex" not in imports, f"the value grammar parses without regular expressions: {imports}"

assert [token["low"] for token in vg.tokenize("I've paid 1,234.5 for 4h 22min.")] == ["i've", "paid", "1,234.5", "for", "4", "h", "22", "min", "."]

one("one hundred and five people", kind="count", amount=105, unit="person")
one("twenty-one guests", kind="count", amount=21, unit="guest")
one("a dozen eggs", kind="count", amount=12, unit="egg")
one("I watched 4 MCU films", kind="count", amount=4, unit="film", noun="MCU films")
one("it got 21 comments", kind="count", amount=21, unit="comment")
one("I rode it three times in a row", kind="count", amount=3, unit="time")
one("we meet twice", kind="count", amount=2, unit="time")
one("two cups in the morning", kind="count", amount=2, unit="cup")

one("in 4h 22min", kind="duration", amount=262, unit="minute")
one("a target of 4 hours and 10 minutes", kind="duration", amount=250, unit="minute")
one("an hour and a half", kind="duration", amount=90, unit="minute")
one("half an hour", kind="duration", amount=30, unit="minute")
one("two and a half hours", kind="duration", amount=150, unit="minute")
one("for 2 weeks", kind="duration", amount=14, unit="day")
one("in the last 3 months", kind="duration", amount=3, unit="month")

one("I paid $12.50", kind="quantity", amount=12.5, unit="USD", dimension="money")
one("about twenty dollars", kind="quantity", amount=20, unit="USD", dimension="money")
one("I lost 5 pounds", kind="quantity", amount=5, unit="pound", dimension="mass")
one("ran 10 km", kind="quantity", amount=10, unit="kilometer", dimension="length")
one("a 15% discount", kind="quantity", amount=15, unit="percent")
one("at 10:30 am", kind="time", clock="10:30")
one("at 7pm", kind="time", clock="19:00")

autumn = datetime.date(2023, 11, 4)
one("I rode it on October 15th", autumn, kind="date", first="2023-10-15", grain="day", weekday="Sun")
one("at Disneyland on September 24th", autumn, kind="date", first="2023-09-24")
one("at SeaWorld in July", autumn, kind="date", first="2023-07-01", last="2023-07-31", grain="month")
one("I'm going there on October 15th", datetime.date(2023, 10, 1), kind="date", first="2023-10-15")
one("I'm going there on October 15th", datetime.date(2023, 10, 20), kind="date", first="2024-10-15")
one("we left on October 15th", datetime.date(2023, 10, 1), kind="date", first="2022-10-15")

spring = datetime.date(2023, 5, 10)
one("Two weeks ago I started", spring, kind="date", first="2023-04-26", grain="week")
one("a couple of days ago", spring, kind="date", first="2023-05-08")
one("the day before yesterday", spring, kind="date", first="2023-05-08")
one("yesterday", spring, kind="date", first="2023-05-09")
one("last week", spring, kind="date", first="2023-05-01", last="2023-05-07", grain="week")
one("last month", spring, kind="date", first="2023-04-01", last="2023-04-30", grain="month")
one("a month ago", spring, kind="date", first="2023-04-10", grain="month")
one("three years ago", spring, kind="date", first="2020-05-10", grain="year")
one("next Friday", spring, kind="date", first="2023-05-12")
one("last Tuesday", spring, kind="date", first="2023-05-09")
one("I left on Monday", spring, kind="date", first="2023-05-08")
one("I will see them in 3 days", spring, kind="date", first="2023-05-13")

one("[Session date: 2023/05/20 (Sat)]", None, kind="date", first="2023-05-20", text="2023/05/20 (Sat)")
one("on 05/20/2023", None, kind="date", first="2023-05-20")
one("moved here on 3 March 2015", None, kind="date", first="2015-03-03")
one("in May 2023", None, kind="date", first="2023-05-01", last="2023-05-31", grain="month")
one("born in 1990", None, kind="date", first="1990-01-01", grain="year")
one("Saturday, May 20 was great", datetime.date(2023, 5, 18), kind="date", first="2023-05-20", weekday="Sat")
assert values("on October 15th") == [], "a date without its year needs an anchor"

assert values("May I ask about the blue one? We march on.", spring) == [], "words that only look like values stay text"
summer = datetime.date(2023, 7, 10)
one("the festival I attended on July 2nd, and I'm hoping the next one will be as good", summer, kind="date", first="2023-07-02")
one("I'm going to the show, which is on July 20th.", summer, kind="date", first="2023-07-20")
assert values("I have a commitment every Saturday morning, so I'll adjust the Sunday plan.", summer) == [], "a habitual or attributive weekday is no date"
assert values("see https://docs.example.com/d/1RwzOFeg6iIRS/edit, the DSL100HR amp, 25m 39s, id b71f1f92-90eb-42c8", summer) == [], \
    "numbers inside links, model names, timestamps and ids are no values"
one("owed £285 million in loans", kind="quantity", amount=285000000, unit="GBP")
assert [found["kind"] for found in values("[Session date: 2023/05/20 (Sat) 04:54] I watched 4 MCU films in the last 3 months.")] == \
    ["date", "time", "count", "duration"]

print("PASS test_value_grammar: word numbers, compound durations, counts with their nouns, money, units, times and dates (absolute, partial, "
      "relative, by weekday, past unless the sentence speaks of the future, a named weekday choosing the year) parse without regular expressions, "
      "and words that only look like values stay text")
