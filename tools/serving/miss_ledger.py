import time
from collections import OrderedDict

SESSION_CAPACITY = 65536
FAMILY_CAPACITY = 4096
SIGNATURE_CAPACITY = 8
BOOT_CAPACITY = 16
OPENING_BLOCKS = ("tools", "attribution", "system", "project_context")
OPENING_FLOOR_TOKENS = 256
FIRST_OPENING = "first_opening"
OPENING_CHANGED = "opening_changed"
RESTORED = "restored"
LOST = "lost"
COMPACTION = "compaction"
HISTORY_CHANGED = "history_changed"
PACK_PROFILE = "pack_profile"
MODEL_NOT_RESIDENT = "model_not_resident"
RESUME_POINT = "resume_point"
CAUSES = (FIRST_OPENING, OPENING_CHANGED, RESTORED, LOST, COMPACTION, HISTORY_CHANGED, PACK_PROFILE, MODEL_NOT_RESIDENT, RESUME_POINT)
LOST_REASONS = ("wiped", "release", "swap", "restart", "evicted")
COMPACTION_CLASS = "compaction"
TEMPLATE_BLOCK = "template"
REPLY_BLOCK = "reply"
HISTORY_BLOCK = "history"


def whole(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def milliseconds(value):
    return round(value, 1) if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0 else None


def engine_block(block):
    block = block if isinstance(block, dict) else {}
    return {"matched": whole(block.get("matched_tokens")), "resumed": whole(block.get("resumed_tokens")), "restore_ms": milliseconds(block.get("restore_ms")),
            "background_wait_ms": milliseconds(block.get("background_wait_ms")), "boot": block.get("boot") if isinstance(block.get("boot"), str) else None,
            "artifact": block.get("artifact") if isinstance(block.get("artifact"), str) else None}


class Ledger:
    def __init__(self, clock=time.time):
        self.clock = clock
        self.sessions = OrderedDict()
        self.families = OrderedDict()
        self.boots = {}
        self.absent = {}

    def model_absent(self, model_id, now=None):
        self.absent[model_id] = self.clock() if now is None else now

    def observe_boot(self, model_id, boot, artifact, now):
        if boot is None:
            return
        known = self.boots.setdefault(model_id, [])
        if not known or known[-1][1] != boot:
            known.append((now, boot, artifact))
            del known[:-BOOT_CAPACITY]

    def compacting(self, keys):
        for key in keys:
            if key in self.sessions:
                self.sessions[key]["compacting"] = True

    def boot_at(self, model_id, moment):
        found = None
        for first_seen, boot, artifact in self.boots.get(model_id, []):
            if first_seen <= moment:
                found = (boot, artifact)
        return found

    def lost_reason(self, model_id, carried_at, boot, artifact, wiped):
        if carried_at is None:
            return None
        if wiped is not None and carried_at < wiped:
            return "wiped"
        earlier = self.boot_at(model_id, carried_at)
        if earlier is not None and boot is not None and artifact is not None and earlier[1] is not None and earlier[1] != artifact:
            return "release"
        if self.absent.get(model_id, 0) >= carried_at:
            return "swap"
        if earlier is None or boot is None:
            return None
        return "restart" if earlier[0] != boot else "evicted"

    @staticmethod
    def family(facts):
        return (facts.get("scope"), facts["model"], facts.get("shape"), facts.get("request_class"), facts.get("agent_type"))

    def opening_cause(self, family, opening):
        known = self.families.get(family)
        if opening is None or not known:
            return FIRST_OPENING, None
        signature = {block: opening.get(block) for block in OPENING_BLOCKS}
        if signature in known:
            return OPENING_CHANGED, TEMPLATE_BLOCK
        latest = known[0]
        return OPENING_CHANGED, next(block for block in OPENING_BLOCKS if latest.get(block) != signature[block])

    def entry(self, facts):
        now = facts.get("now", self.clock())
        model_id, block_tokens = facts["model"], facts["block_tokens"]
        prompt = facts["prompt_tokens"]
        engine = engine_block(facts.get("engine"))
        resumed = engine["resumed"] if engine["resumed"] is not None else facts["cached_tokens"]
        matched = engine["matched"]
        seen, seen_of, seen_at = facts["seen_tokens"], facts["seen_of"], facts.get("seen_at")
        opening = facts.get("opening")
        session = next((self.sessions[key] for key in [facts.get("session")] + list(facts.get("session_fallbacks") or ())
                        if key is not None and key in self.sessions and self.sessions[key]["model"] == model_id), None)
        self.observe_boot(model_id, engine["boot"], engine["artifact"], now)
        slack = block_tokens if matched is not None else max(block_tokens, facts.get("resume_tokens") or block_tokens)
        extends = session is not None and seen >= session["total"] // block_tokens * block_tokens
        compacted = bool(facts.get("compacted")) or (session is not None and bool(session.get("compacting")) and not extends)
        first_request = session is None and opening is not None and opening.get("first_request") is True
        first_prompt = facts.get("first_prompt_tokens")
        carried, carried_at = (session["total"], session["at"]) if extends else (seen, seen_at)
        if extends:
            tail = max(0, prompt - carried)
        elif session is not None:
            tail = 0
        elif first_request:
            tail = min(prompt - carried, first_prompt) if first_prompt is not None else prompt - carried
        else:
            tail = None
        lost_tokens = max(0, min(carried, prompt - 1) - (matched if matched is not None else resumed))
        lost_tokens = lost_tokens if lost_tokens > slack else 0
        recomputed = max(0, matched - resumed) if matched is not None else None
        miss = max(0, prompt - resumed - tail) if tail is not None else None
        opening_tokens = max(0, prompt - carried - tail) if first_request else 0
        cause, block, position, lost = None, None, None, None
        if facts.get("model_wait_ms"):
            cause = MODEL_NOT_RESIDENT
        elif compacted:
            cause, position = (COMPACTION, seen) if miss is not None and miss > slack else (None, None)
        elif session is not None and not extends and miss > slack:
            cause, position = HISTORY_CHANGED, seen
            block = REPLY_BLOCK if seen >= session["prompt"] // block_tokens * block_tokens else HISTORY_BLOCK
        elif engine["restore_ms"]:
            cause = RESTORED
        elif lost_tokens:
            cause, lost = LOST, self.lost_reason(model_id, carried_at, engine["boot"], engine["artifact"], facts.get("wiped"))
        elif facts.get("pack_profile_new") and miss is not None and miss > slack:
            cause = PACK_PROFILE
        elif opening_tokens >= OPENING_FLOOR_TOKENS:
            cause, block = self.opening_cause(self.family(facts), opening)
        elif recomputed is not None and recomputed > block_tokens:
            cause = RESUME_POINT
        return {"cause": cause, "cause_block": block, "cause_position": position, "lost_reason": lost, "matched_tokens": matched, "resumed_tokens": resumed,
                "recomputed_tokens": recomputed, "tail_tokens": tail, "miss_tokens": miss, "opening_tokens": opening_tokens, "lost_tokens": lost_tokens,
                "seen_tokens": seen, "seen_of": seen_of, "seen_age_s": round(now - seen_at) if seen_at is not None and seen else None,
                "session_tokens": session["total"] if session is not None else None, "session_turn": (session["turns"] if session is not None else 0) + 1,
                "restore_ms": engine["restore_ms"], "background_wait_ms": engine["background_wait_ms"], "model_wait_ms": milliseconds(facts.get("model_wait_ms")),
                "request_class": facts.get("request_class"), "agent_type": facts.get("agent_type"), "compaction": facts.get("compaction"),
                "compacted": bool(facts.get("compacted")), "engine_reported": matched is not None}

    def settle(self, facts, total_tokens):
        now = facts.get("now", self.clock())
        key, model_id = facts.get("session"), facts["model"]
        if key is not None:
            earlier = self.sessions.pop(key, None)
            turns = earlier["turns"] if earlier is not None and earlier["model"] == model_id else 0
            self.sessions[key] = {"model": model_id, "total": total_tokens, "prompt": facts["prompt_tokens"], "at": now, "turns": turns + 1, "compacting": False}
            while len(self.sessions) > SESSION_CAPACITY:
                self.sessions.popitem(last=False)
        opening = facts.get("opening")
        if opening is not None:
            family = self.family(facts)
            signature = {block: opening.get(block) for block in OPENING_BLOCKS}
            known = self.families.pop(family, [])
            self.families[family] = [signature] + [entry for entry in known if entry != signature][:SIGNATURE_CAPACITY - 1]
            while len(self.families) > FAMILY_CAPACITY:
                self.families.popitem(last=False)
