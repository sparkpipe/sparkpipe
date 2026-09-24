import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INCLUDE = re.compile(r'^#include "(sparkpipe/family/[^"]+)"\n', re.M)


def expand(text, camel, upper, lower):
    forms = {"SPARK_FAMILY_CAMEL": camel, "SPARK_FAMILY_UPPER": upper, "SPARK_FAMILY_LOWER": lower}
    text = re.sub(r"SPARK_FAMILY_STRING\((SPARK_FAMILY_\w+)\)", lambda m: '"' + forms[m.group(1)] + '"', text)
    text = re.sub(r"SPARK_FAMILY_CONST\((\w+)\)", lambda m: "SPARK_" + upper + "_" + m.group(1), text)
    text = re.sub(r"SPARK_FAMILY\((\w+)\)", lambda m: "Spark" + camel + m.group(1), text)
    return re.sub(r"SPARK_FAMILY_CAT\((\w*),(SPARK_FAMILY_\w+),(\w*)\)", lambda m: m.group(1) + forms[m.group(2)] + m.group(3), text)


def read_source(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    names = [re.search(r"#define SPARK_FAMILY_%s (\w+)" % key, text) for key in ("CAMEL", "UPPER", "LOWER")]
    if not all(names):
        return text
    camel, upper, lower = (m.group(1) for m in names)

    def include(match):
        if match.group(1) == "sparkpipe/family/spark_family.h":
            return match.group(0)
        return expand((ROOT / "include" / match.group(1)).read_text(encoding="utf-8"), camel, upper, lower) + "\n"

    return INCLUDE.sub(include, text)
