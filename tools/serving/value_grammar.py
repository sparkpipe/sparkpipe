import datetime
import unicodedata

ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
        "nineteen": 19}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
SCALES = {"hundred": 100, "thousand": 1000, "million": 1000000, "billion": 1000000000}
ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
            "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15, "sixteenth": 16, "seventeenth": 17,
            "eighteenth": 18, "nineteenth": 19, "twentieth": 20, "thirtieth": 30}
ORDINAL_SUFFIXES = ("st", "nd", "rd", "th")
GROUPS = {"dozen": 12, "couple": 2, "pair": 2}
ONCE = {"once": 1, "twice": 2, "thrice": 3}
MONTHS = {"january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6,
          "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10, "oct": 10, "november": 11,
          "nov": 11, "december": 12, "dec": 12}
WEEKDAYS = {"monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1, "wednesday": 2, "wed": 2, "thursday": 3, "thu": 3, "thur": 3,
            "thurs": 3, "friday": 4, "fri": 4, "saturday": 5, "sat": 5, "sunday": 6, "sun": 6}
CALENDAR = {"day": ("day", 1), "days": ("day", 1), "week": ("day", 7), "weeks": ("day", 7), "fortnight": ("day", 14),
            "fortnights": ("day", 14), "month": ("month", 1), "months": ("month", 1), "year": ("month", 12), "years": ("month", 12),
            "decade": ("month", 120), "decades": ("month", 120)}
CLOCK = {"second": 1 / 60, "seconds": 1 / 60, "sec": 1 / 60, "secs": 1 / 60, "minute": 1, "minutes": 1, "min": 1, "mins": 1,
         "hour": 60, "hours": 60, "hr": 60, "hrs": 60, "h": 60}
ABBREVIATIONS = {"h", "min", "sec", "hr", "g", "mi", "ft", "oz", "ml", "lb", "lbs", "kg", "kgs", "km"}
UNITS = {"mile": ("length", "mile", 1), "miles": ("length", "mile", 1), "mi": ("length", "mile", 1),
         "kilometer": ("length", "kilometer", 1), "kilometers": ("length", "kilometer", 1), "kilometre": ("length", "kilometer", 1),
         "kilometres": ("length", "kilometer", 1), "km": ("length", "kilometer", 1),
         "meter": ("length", "meter", 1), "meters": ("length", "meter", 1), "metre": ("length", "meter", 1), "metres": ("length", "meter", 1),
         "foot": ("length", "foot", 1), "feet": ("length", "foot", 1), "ft": ("length", "foot", 1),
         "inch": ("length", "inch", 1), "inches": ("length", "inch", 1),
         "pound": ("mass", "pound", 1), "pounds": ("mass", "pound", 1), "lb": ("mass", "pound", 1), "lbs": ("mass", "pound", 1),
         "kilogram": ("mass", "kilogram", 1), "kilograms": ("mass", "kilogram", 1), "kg": ("mass", "kilogram", 1), "kgs": ("mass", "kilogram", 1),
         "gram": ("mass", "gram", 1), "grams": ("mass", "gram", 1), "g": ("mass", "gram", 1),
         "ounce": ("mass", "ounce", 1), "ounces": ("mass", "ounce", 1), "oz": ("mass", "ounce", 1),
         "liter": ("volume", "liter", 1), "liters": ("volume", "liter", 1), "litre": ("volume", "liter", 1), "litres": ("volume", "liter", 1),
         "milliliter": ("volume", "milliliter", 1), "milliliters": ("volume", "milliliter", 1), "ml": ("volume", "milliliter", 1),
         "gallon": ("volume", "gallon", 1), "gallons": ("volume", "gallon", 1),
         "degree": ("temperature", "degree", 1), "degrees": ("temperature", "degree", 1),
         "percent": ("percent", "percent", 1),
         "dollar": ("money", "USD", 1), "dollars": ("money", "USD", 1), "buck": ("money", "USD", 1), "bucks": ("money", "USD", 1),
         "cent": ("money", "USD", 0.01), "cents": ("money", "USD", 0.01), "euro": ("money", "EUR", 1), "euros": ("money", "EUR", 1),
         "yen": ("money", "JPY", 1), "won": ("money", "KRW", 1)}
CURRENCY_SIGNS = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY", "₩": "KRW"}
FUTURE = {"will", "going", "plan", "plans", "planning", "upcoming", "coming", "soon", "scheduled", "next", "tomorrow", "gonna"}
RATE_WORDS = {"once", "twice", "thrice", "times", "per", "every", "each"}
YEAR_WORDS = {"in", "since", "during", "from", "of", "by", "until", "before", "after"}
MONTH_WORDS = {"in", "since", "during", "of", "early", "late", "mid", "until", "by", "last", "next", "this"}
SENTENCE_ENDS = {".", "!", "?"}
FUNCTION_WORDS = {
    "a", "an", "the", "this", "that", "these", "those", "my", "your", "his", "her", "its", "our", "their", "some", "any", "no", "every",
    "each", "all", "both", "either", "neither", "other", "another", "such", "what", "which", "who", "whom", "whose", "where", "when", "why",
    "how", "i", "me", "you", "he", "him", "she", "it", "we", "us", "they", "them", "myself", "yourself", "ourselves", "themselves",
    "of", "in", "on", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "out", "off", "over", "under", "again", "further", "then", "once", "here", "there", "than", "per",
    "and", "or", "but", "nor", "so", "yet", "if", "because", "as", "until", "while", "since", "though", "although", "unless", "whether",
    "is", "am", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "can", "could",
    "will", "would", "shall", "should", "may", "might", "must", "not", "very", "too", "just", "also", "only", "really", "now", "ago",
    "last", "next", "more", "most", "less", "least", "much", "many", "few", "lot", "lots", "like", "around", "almost", "nearly",
    "approximately", "roughly", "total", "row", "already", "still", "ever", "even", "or", "each", "earlier", "later", "back", "away"}
IRREGULAR_PLURALS = {"people": "person", "children": "child", "men": "man", "women": "woman", "feet": "foot", "teeth": "tooth",
                     "mice": "mouse", "geese": "goose", "lives": "life", "knives": "knife", "wives": "wife", "leaves": "leaf"}
PAST_FORMS = {"was", "were", "had", "did", "went", "got", "saw", "made", "took", "came", "ate", "bought", "rode", "ran", "met", "left", "felt",
              "found", "gave", "told", "wrote", "spent", "paid", "sold", "won", "lost", "heard", "kept", "brought", "began", "flew", "drove",
              "wore", "chose", "thought", "taught", "caught", "fought", "sought", "built", "sent", "lent", "slept", "swam", "sang", "drank",
              "stood", "understood", "held", "led", "read", "said", "knew", "grew", "threw", "drew", "broke", "spoke", "woke", "froze",
              "forgot", "got", "sat", "set", "put", "cut", "hit", "hurt", "let", "shut", "quit", "fell", "became", "began", "bit", "hid",
              "shook", "struck", "stuck", "swung", "hung", "dug", "fed", "fled", "bled", "bred", "sped", "lit", "meant", "dealt", "felt",
              "ago", "yesterday", "last", "used"}
NOT_PAST = {"need", "seed", "feed", "speed", "bed", "red", "indeed", "hundred", "proceed", "succeed", "exceed", "breed", "bleed", "greed",
            "shred", "sled", "wed", "embed", "shed", "weed", "deed", "reed", "heed", "steed", "tweed"}
CLAUSE_BREAKS = {",", ";", ":", ".", "!", "?", "and", "but", "so", "because", "while", "whereas", "although", "though", "which", "who",
                 "where", "when"}
HABITUAL = {"every", "each", "a", "the", "any", "some", "most", "all"}
ATTACHED = set(ORDINAL_SUFFIXES) | ABBREVIATIONS | {"am", "pm", "k", "hrs", "mins", "secs"}
LINK_STARTS = {"http", "https", "www", "ftp", "mailto"}
NOUN_WORDS = 3


def past_form(word):
    if word in PAST_FORMS:
        return True
    return len(word) > 3 and word.endswith("ed") and word not in NOT_PAST and not word.endswith("eed")


def char_kind(char):
    if char in "%°":
        return "symbol"
    category = unicodedata.category(char)
    if category[0] == "L" or category == "Mn":
        return "letter"
    if category == "Nd":
        return "digit"
    if category[0] == "Z" or char in "\t\n\r\f\v":
        return "space"
    if category == "Sc":
        return "symbol"
    return "punct"


def thousands_group(text, index):
    return index + 3 <= len(text) and all(char_kind(char) == "digit" for char in text[index:index + 3]) and \
        (index + 3 == len(text) or char_kind(text[index + 3]) != "digit")


def tokenize(text):
    found, index, size = [], 0, len(text)
    while index < size:
        kind, start = char_kind(text[index]), index
        if kind == "space":
            index += 1
            continue
        if kind == "letter":
            while index < size and (char_kind(text[index]) == "letter" or text[index] in "'’" and index + 1 < size and char_kind(text[index + 1]) == "letter"):
                index += 1
        elif kind == "digit":
            point = False
            while index < size and (char_kind(text[index]) == "digit" or not point and text[index] == "." and index + 1 < size and char_kind(text[index + 1]) == "digit"
                                    or not point and text[index] == "," and thousands_group(text, index + 1)):
                point = point or text[index] == "."
                index += 1
        else:
            index += 1
        raw = text[start:index]
        found.append({"kind": "word" if kind == "letter" else "number" if kind == "digit" else kind, "low": raw.lower().replace("’", "'"),
                      "start": start, "end": index, "raw": raw})
    return found


def singular(word):
    if word in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[word]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("ches", "shes", "sses", "xes", "zes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def add_months(day, months):
    year, month = divmod(day.year * 12 + day.month - 1 + months, 12)
    following = datetime.date(year + (month + 1) // 12, (month + 1) % 12 + 1, 1)
    return datetime.date(year, month + 1, min(day.day, (following - datetime.timedelta(days=1)).day))


def month_span(year, month):
    first = datetime.date(year, month, 1)
    return first, add_months(first, 1) - datetime.timedelta(days=1)


def valid_day(year, month, day):
    try:
        return datetime.date(year, month, day)
    except ValueError:
        return None


def number_value(raw):
    text = raw.replace(",", "")
    return float(text) if "." in text else int(text)


class Reader:
    def __init__(self, text, anchor=None):
        self.text, self.anchor = text, anchor
        self.tokens = tokenize(text)

    def low(self, index):
        return self.tokens[index]["low"] if 0 <= index < len(self.tokens) else None

    def raw(self, index):
        return self.tokens[index]["raw"] if 0 <= index < len(self.tokens) else ""

    def kind(self, index):
        return self.tokens[index]["kind"] if 0 <= index < len(self.tokens) else None

    def touching(self, index):
        return 0 < index < len(self.tokens) and self.tokens[index - 1]["end"] == self.tokens[index]["start"]

    def digits(self, index, size=None):
        return self.kind(index) == "number" and self.raw(index).isdigit() and (size is None or len(self.raw(index)) == size)

    def future(self, index):
        for breaks in (CLAUSE_BREAKS, SENTENCE_ENDS):
            start = index
            while start > 0 and self.low(start - 1) not in breaks:
                start -= 1
            end = index
            while end < len(self.tokens) and self.low(end) not in breaks:
                end += 1
            marks = [(abs(at - index), self.low(at) in FUTURE or self.low(at).endswith("'ll")) for at in range(start, end)
                     if self.kind(at) == "word" and (self.low(at) in FUTURE or self.low(at).endswith("'ll") or past_form(self.low(at)))]
            if marks:
                return min(marks)[1]
        return False

    def glued(self, index):
        return self.touching(index) and self.kind(index - 1) in ("word", "number") or \
            self.touching(index + 1) and self.kind(index + 1) == "word" and self.low(index + 1) not in ATTACHED

    def link_end(self, index):
        if self.low(index) not in LINK_STARTS and not (self.kind(index) == "word" and self.low(index + 1) == "@" and self.touching(index + 1)):
            return None
        at = index + 1
        while at < len(self.tokens) and self.touching(at):
            at += 1
        return at

    def value(self, kind, start, end, **fields):
        span = (self.tokens[start]["start"], self.tokens[end - 1]["end"])
        return dict(kind=kind, text=self.text[span[0]:span[1]], span=span, **fields), end

    def ordinal_suffix(self, index):
        return self.touching(index + 1) and self.low(index + 1) in ORDINAL_SUFFIXES

    def cardinal(self, index):
        if self.kind(index) == "number":
            if self.ordinal_suffix(index) or self.glued(index):
                return None
            amount, at = number_value(self.raw(index)), index + 1
            if self.low(at) in SCALES and SCALES[self.low(at)] >= 1000:
                amount, at = amount * SCALES[self.low(at)], at + 1
            if self.low(at) == "and" and self.low(at + 1) == "a" and self.low(at + 2) == "half":
                return amount + 0.5, at + 3
            return amount, at
        total, current, at, seen = 0, 0, index, False
        while at < len(self.tokens):
            word = self.low(at)
            if word in ONES and not (seen and current % 10) and not (seen and 10 <= current % 100 < 20):
                current, at, seen = current + ONES[word], at + 1, True
            elif word in TENS and not (seen and current % 100):
                current, at, seen = current + TENS[word], at + 1, True
                if self.low(at) == "-" and self.low(at + 1) in ONES and ONES[self.low(at + 1)] < 10 and self.touching(at) and self.touching(at + 1):
                    current, at = current + ONES[self.low(at + 1)], at + 2
            elif word in SCALES and (seen or self.low(at - 1) == "a" and at == index + 1):
                if SCALES[word] == 100:
                    current = max(current, 1) * 100
                else:
                    total, current = total + max(current, 1) * SCALES[word], 0
                at, seen = at + 1, True
            elif word == "a" and not seen and self.low(at + 1) in SCALES:
                at += 1
            elif word == "and" and seen and current >= 100 and (self.low(at + 1) in ONES or self.low(at + 1) in TENS):
                at += 1
            else:
                break
        if not seen:
            return None
        if self.low(at) == "and" and self.low(at + 1) == "a" and self.low(at + 2) == "half":
            return total + current + 0.5, at + 3
        return total + current, at

    def ordinal(self, index):
        if self.kind(index) == "number" and self.raw(index).isdigit() and self.ordinal_suffix(index):
            return int(self.raw(index)), index + 2
        word = self.low(index)
        if word in ORDINALS:
            return ORDINALS[word], index + 1
        if word in TENS and self.low(index + 1) == "-" and self.low(index + 2) in ORDINALS and ORDINALS[self.low(index + 2)] < 10:
            return TENS[word] + ORDINALS[self.low(index + 2)], index + 3
        return None

    def amount(self, index):
        word = self.low(index)
        if word in ("a", "an"):
            if self.low(index + 1) in GROUPS:
                return GROUPS[self.low(index + 1)], index + 2 + (self.low(index + 2) == "of")
            if self.low(index + 1) == "few":
                return 3, index + 2
            if self.low(index + 1) in SCALES:
                return self.cardinal(index)
            return 1, index + 1
        if word == "half" and self.low(index + 1) in ("a", "an"):
            return 0.5, index + 2
        found = self.cardinal(index)
        if found and self.low(found[1]) in GROUPS:
            return found[0] * GROUPS[self.low(found[1])], found[1] + 1 + (self.low(found[1] + 1) == "of")
        return found

    def unit_allowed(self, index, at):
        return self.low(at) not in ABBREVIATIONS or self.touching(at) or self.kind(index) == "number"

    def clock(self, index):
        found = self.amount(index)
        if not found or self.low(found[1]) not in CLOCK or not self.unit_allowed(index, found[1]):
            return None
        amount, at = found
        minutes, at = amount * CLOCK[self.low(at)], at + 1
        if self.low(at) == "and" and self.low(at + 1) == "a" and self.low(at + 2) == "half":
            minutes, at = minutes + 0.5 * CLOCK[self.low(at - 1)], at + 3
        more = self.clock(at + (self.low(at) == "and"))
        if more and more[0] < minutes:
            return minutes + more[0], more[1]
        return minutes, at

    def duration(self, index):
        found = self.clock(index)
        if found:
            return {"unit": "minute", "amount": found[0]}, found[1]
        found = self.amount(index)
        if found and self.low(found[1]) in CALENDAR:
            grain, size = CALENDAR[self.low(found[1])]
            return {"unit": grain, "amount": found[0] * size}, found[1] + 1
        return None

    def money(self, index):
        if self.kind(index) == "symbol" and self.low(index) in CURRENCY_SIGNS and self.kind(index + 1) == "number":
            amount, at = self.cardinal(index + 1)
            if self.low(at) == "k" and self.touching(at):
                amount, at = amount * 1000, at + 1
            return self.value("quantity", index, at, amount=amount, unit=CURRENCY_SIGNS[self.low(index)], dimension="money")
        return None

    def quantity(self, index):
        word = self.low(index)
        if word in ONCE:
            return self.value("count", index, index + 1, amount=ONCE[word], unit="time", noun=self.raw(index))
        if word in ("a", "an") and self.low(index + 1) not in GROUPS:
            return None
        found = self.amount(index)
        if not found:
            return None
        amount, at = found
        unit = self.low(at)
        if unit in ("%", "percent"):
            return self.value("quantity", index, at + 1, amount=amount, unit="percent", dimension="percent")
        if unit in UNITS and self.unit_allowed(index, at):
            dimension, name, factor = UNITS[unit]
            return self.value("quantity", index, at + 1, amount=amount * factor, unit=name, dimension=dimension)
        if unit == "times":
            return self.value("count", index, at + 1, amount=amount, unit="time", noun="times")
        nouns, last = [], at
        while last < len(self.tokens) and len(nouns) < NOUN_WORDS and self.kind(last) == "word" and self.low(last) not in FUNCTION_WORDS \
                and self.low(last) not in MONTHS and self.low(last) not in CALENDAR and self.low(last) not in CLOCK and self.low(last) not in WEEKDAYS:
            nouns.append(last)
            last += 1
        if nouns:
            return self.value("count", index, nouns[-1] + 1, amount=amount, unit=singular(self.low(nouns[-1])),
                              noun=self.text[self.tokens[nouns[0]]["start"]:self.tokens[nouns[-1]]["end"]])
        if self.kind(index) == "number":
            return self.value("number", index, at, amount=amount)
        return None

    def time_of_day(self, index):
        if self.digits(index) and self.low(index + 1) == ":" and self.digits(index + 2, 2) and self.touching(index + 1) and self.touching(index + 2):
            hour, minute, at = int(self.raw(index)), int(self.raw(index + 2)), index + 3
            if self.low(at) in ("am", "pm"):
                hour, at = hour % 12 + (12 if self.low(at) == "pm" else 0), at + 1
            if hour < 24 and minute < 60:
                return self.value("time", index, at, clock=f"{hour:02d}:{minute:02d}")
        if self.digits(index) and self.low(index + 1) in ("am", "pm") and 1 <= int(self.raw(index)) <= 12:
            hour = int(self.raw(index)) % 12 + (12 if self.low(index + 1) == "pm" else 0)
            return self.value("time", index, index + 2, clock=f"{hour:02d}:00")
        return None

    def weekday_tail(self, at):
        if self.low(at) == "(" and self.low(at + 1) in WEEKDAYS and self.low(at + 2) == ")":
            return at + 3
        return at

    def numeric_date(self, index):
        if self.digits(index, 4) and self.low(index + 1) in ("/", "-") and self.digits(index + 2) and self.low(index + 3) == self.low(index + 1) \
                and self.digits(index + 4) and all(self.touching(at) for at in range(index + 1, index + 5)):
            day = valid_day(int(self.raw(index)), int(self.raw(index + 2)), int(self.raw(index + 4)))
            return (day, day, "day", self.weekday_tail(index + 5)) if day else None
        if self.digits(index) and self.low(index + 1) == "/" and self.digits(index + 2) and self.low(index + 3) == "/" and self.digits(index + 4, 4) \
                and all(self.touching(at) for at in range(index + 1, index + 5)):
            day = valid_day(int(self.raw(index + 4)), int(self.raw(index)), int(self.raw(index + 2)))
            return (day, day, "day", self.weekday_tail(index + 5)) if day else None
        return None

    def year(self, index):
        if self.digits(index, 4) and 1900 <= int(self.raw(index)) <= 2100:
            return int(self.raw(index)), index + 1
        return None

    def month_name(self, index):
        month = MONTHS.get(self.low(index))
        if not month:
            return None
        if self.raw(index)[:1].isupper() or self.ordinal(index + 1) or self.digits(index + 1) or self.low(index - 1) in MONTH_WORDS:
            return month
        return None

    def resolve(self, month, day, start, end):
        if not self.anchor:
            return None
        future = self.future(start)
        for year in ((self.anchor.year, self.anchor.year + 1) if future else (self.anchor.year, self.anchor.year - 1)):
            found = valid_day(year, month, day)
            if found and (found >= self.anchor if future else found <= self.anchor):
                return found, found, "day", end
        return None

    def day_number(self, index):
        found = self.ordinal(index) or (self.cardinal(index) if self.digits(index) else None)
        return found if found and isinstance(found[0], int) and 1 <= found[0] <= 31 else None

    def calendar_date(self, index):
        found = self.numeric_date(index)
        if found:
            return found
        if self.low(index) in WEEKDAYS and self.low(index + 1) in (",", None) or self.low(index) in WEEKDAYS and self.month_name(index + 1):
            found = self.calendar_date(index + 1 + (self.low(index + 1) == ","))
            if found and found[0] == found[1] and found[0].weekday() != WEEKDAYS[self.low(index)] and self.anchor:
                named = [day for day in (valid_day(found[0].year + step, found[0].month, found[0].day) for step in (-1, 1, -2, 2))
                         if day and day.weekday() == WEEKDAYS[self.low(index)]]
                if named:
                    day = min(named, key=lambda day: abs((day - self.anchor).days))
                    return day, day, "day", found[3]
            if found:
                return found
        month = self.month_name(index)
        if month:
            at = index + 1 + (self.low(index + 1) == ".")
            day = self.day_number(at)
            if day:
                at = day[1]
                year = self.year(at + (self.low(at) == ","))
                if year:
                    found = valid_day(year[0], month, day[0])
                    return (found, found, "day", year[1]) if found else None
                return self.resolve(month, day[0], index, at)
            year = self.year(at + (self.low(at) == ","))
            if year:
                first, last = month_span(year[0], month)
                return first, last, "month", year[1]
            if self.anchor and (len(self.low(index)) > 3 or self.low(index - 1) in MONTH_WORDS):
                step = {"last": -1, "next": 1}.get(self.low(index - 1))
                if step == -1 or step is None and not self.future(index):
                    year = self.anchor.year if month <= self.anchor.month else self.anchor.year - 1
                else:
                    year = self.anchor.year if month >= self.anchor.month else self.anchor.year + 1
                first, last = month_span(year, month)
                return first, last, "month", index + 1
            return None
        day = self.day_number(index)
        if day:
            at = day[1] + (self.low(day[1]) == "of")
            month = MONTHS.get(self.low(at)) if len(self.low(at) or "") > 2 or self.raw(at)[:1].isupper() else None
            if month:
                year = self.year(at + 1 + (self.low(at + 1) == ","))
                if year:
                    found = valid_day(year[0], month, day[0])
                    return (found, found, "day", year[1]) if found else None
                return self.resolve(month, day[0], index, at + 1)
        if self.low(index - 1) in YEAR_WORDS:
            year = self.year(index)
            if year:
                return datetime.date(year[0], 1, 1), datetime.date(year[0], 12, 31), "year", year[1]
        return None

    def weekday(self, index):
        if not self.anchor:
            return None
        mode, at = (self.low(index), index + 1) if self.low(index) in ("last", "next", "this", "past", "coming") else (None, index)
        day = WEEKDAYS.get(self.low(at))
        if day is None or len(self.low(at)) <= 4 and not self.raw(at)[:1].isupper() or mode is None and self.low(index - 1) in HABITUAL:
            return None
        if mode in ("next", "coming") or mode is None and self.future(at):
            found = self.anchor + datetime.timedelta(days=(day - self.anchor.weekday()) % 7 or 7)
        elif mode == "this":
            found = self.anchor + datetime.timedelta(days=day - self.anchor.weekday())
        else:
            found = self.anchor - datetime.timedelta(days=(self.anchor.weekday() - day) % 7 or 7)
        return found, found, "day", at + 1

    def relative(self, index):
        if not self.anchor:
            return None
        anchor, word = self.anchor, self.low(index)
        days = {"today": 0, "tonight": 0, "yesterday": -1, "tomorrow": 1}
        if word in days:
            found = anchor + datetime.timedelta(days=days[word])
            return found, found, "day", index + 1
        if word == "the" and self.low(index + 1) == "day" and self.low(index + 2) in ("before", "after") and self.low(index + 3) in ("yesterday", "tomorrow"):
            found = anchor + datetime.timedelta(days=-2 if self.low(index + 3) == "yesterday" else 2)
            return found, found, "day", index + 4
        if word in ("last", "this", "next", "past") and self.low(index + 1) in ("week", "month", "year", "weekend"):
            unit, step = self.low(index + 1), {"last": -1, "past": -1, "this": 0, "next": 1}[word]
            if unit == "week":
                monday = anchor - datetime.timedelta(days=anchor.weekday() - 7 * step)
                return monday, monday + datetime.timedelta(days=6), "week", index + 2
            if unit == "weekend":
                saturday = anchor - datetime.timedelta(days=anchor.weekday() - 5 - 7 * step)
                if step == -1 and anchor.weekday() >= 5:
                    saturday += datetime.timedelta(days=7)
                return saturday, saturday + datetime.timedelta(days=1), "weekend", index + 2
            if unit == "month":
                first = add_months(datetime.date(anchor.year, anchor.month, 1), step)
                return first, add_months(first, 1) - datetime.timedelta(days=1), "month", index + 2
            return datetime.date(anchor.year + step, 1, 1), datetime.date(anchor.year + step, 12, 31), "year", index + 2
        span = self.duration(index)
        if not span:
            return None
        past = self.low(span[1]) in ("ago", "earlier", "back")
        if not past and not (self.low(index - 1) == "in" and self.future(index)):
            return None
        amount, unit, sign = span[0]["amount"], span[0]["unit"], -1 if past else 1
        end = span[1] + past
        if unit == "month":
            found = add_months(anchor, sign * int(round(amount)))
            return found, found, "year" if amount % 12 == 0 else "month", end
        if unit == "day":
            found = anchor + datetime.timedelta(days=sign * amount)
            return found, found, "week" if amount >= 7 and amount % 7 == 0 else "day", end
        return anchor, anchor, "day", end

    def date(self, index):
        found = self.calendar_date(index) or self.relative(index) or self.weekday(index)
        if not found:
            return None
        first, last, grain, end = found
        return self.value("date", index, end, first=first.isoformat(), last=last.isoformat(), grain=grain,
                          weekday=first.strftime("%a") if first == last else None)

    def span_value(self, index):
        if self.low(index - 1) in RATE_WORDS and self.low(index) in ("a", "an"):
            return None
        found = self.duration(index)
        if not found:
            return None
        return self.value("duration", index, found[1], amount=found[0]["amount"], unit=found[0]["unit"])

    def read(self):
        found, index = [], 0
        while index < len(self.tokens):
            link = self.link_end(index)
            if link:
                index = link
                continue
            best = None
            for parse in (self.date, self.time_of_day, self.money, self.span_value, self.quantity):
                candidate = parse(index)
                if candidate and (best is None or candidate[1] > best[1]):
                    best = candidate
            if best:
                found.append(best[0])
                index = best[1]
            else:
                index += 1
        return found


def read(text, anchor=None):
    return Reader(text, anchor).read()
