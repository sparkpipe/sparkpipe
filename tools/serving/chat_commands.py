"""Chat commands: per-chat memory settings, a context report, relevance compaction and the chat timer's reply.

A user message that starts with one of these commands never reaches the model as a turn:

  /context                       what the next prompt holds
  /memory [on|off]               memory for this chat (the Full context toggle overrides it per request)
  /target [N|NK|N%|off]          the context target memory keeps the prompt near
  /compact [N|NK|N%|full|undo]   keep the most relevant exchanges in full, summarize the rest
  /timer [every INTERVAL PROMPT|off]

The chat layer answers each command itself, in Korean when the chat's own messages are Korean; the
command and its answer are left out of every later prompt. Settings and compactions are records in the
memory store of the chat.
"""

import json
import re

COMMAND = re.compile(r"^\s*/(context|memory|target|compact|timer)(?:\s+(.*))?\s*$", re.S | re.I)
SIZE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(k|m|%)?\s*$", re.I)
TIMER = re.compile(r"^\s*every\s+(\d+)\s*(s|m|h|d)\s+(.+)$", re.S | re.I)
SUMMARY_OPEN = ("<summary>\nWhat the compacted exchanges of this conversation established, each marked (eN) in conversation order. "
                "The exchanges after this summary labeled [eN] were kept in full and share that numbering: a higher number is later, "
                "so a line marked (e7) supersedes a kept [e3].\n")
SUMMARY_CLOSE = "</summary>\n\n"
EXCERPT_USER = 600
EXCERPT_ASSISTANT = 400
DEFAULT_SHARE = 0.5
HANGUL = re.compile("[\u1100-\u11ff\u3130-\u318f\uac00-\ud7a3]")


def size(text, whole):
    found = SIZE.match(text or "")
    if not found:
        return None
    value, unit = float(found.group(1)), (found.group(2) or "").lower()
    if unit == "%":
        return int(whole * value / 100) if 0 < value <= 100 else None
    tokens = int(value * {"k": 1024, "m": 1048576}.get(unit, 1))
    return tokens if tokens > 0 else None


def label(tokens):
    if tokens >= 1048576 and tokens % 1048576 == 0:
        return f"{tokens // 1048576}M"
    if tokens >= 1024:
        return f"{tokens / 1024:.1f}K".replace(".0K", "K")
    return str(tokens)


def command_of(message):
    if message.get("role") != "user":
        return None
    content = message.get("content")
    text = content if isinstance(content, str) else "".join(part.get("text", "") for part in content or [] if isinstance(part, dict))
    found = COMMAND.match(text)
    return (found.group(1).lower(), (found.group(2) or "").strip()) if found else None


def korean(messages):
    for message in reversed(messages):
        if message.get("role") != "user" or command_of(message) is not None:
            continue
        content = message.get("content")
        text = content if isinstance(content, str) else "".join(part.get("text", "") for part in content or [] if isinstance(part, dict))
        if text.strip():
            return HANGUL.search(text) is not None
    return False


def strip(messages):
    kept, index = [], 0
    while index < len(messages):
        message = messages[index]
        if command_of(message) is not None and index + 1 < len(messages):
            index += 2 if messages[index + 1].get("role") == "assistant" else 1
            continue
        kept.append(message)
        index += 1
    return kept


def parse_choice(text, ids):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or not isinstance(value.get("keep"), list) or not isinstance(value.get("summary"), str):
        return None
    return [item for item in value["keep"] if item in ids], value["summary"].strip()


class Commands:
    def __init__(self, memory, module, frontend):
        self.memory = memory
        self.module = module
        self.frontend = frontend

    def identity(self, headers):
        if headers is None:
            return None
        claims = self.module.verify_jwt(headers.get("X-OpenWebUI-User-Jwt") or "", self.memory.secret)
        chat = self.module.header(headers, "X-OpenWebUI-Chat-Id")
        if not claims or claims.get("iss") != self.memory.issuer or not isinstance(claims.get("sub"), str) or not chat:
            return None
        return claims["sub"], chat

    def settings(self, identity):
        return dict(self.memory.store.load(*identity)["settings"]) if identity else {}

    def match(self, messages):
        return command_of(messages[-1]) if messages else None

    def compacted(self, identity, messages):
        compaction = self.memory.store.load(*identity)["compaction"] if identity else None
        parts = self.module.split(messages) if compaction else None
        if not parts:
            return messages
        head, exchanges, current = parts
        number, keep = {key: index + 1 for index, key in enumerate(compaction["covered"])}, set(compaction["keep"])
        kept = []
        for exchange in exchanges:
            key = self.module.exchange_key(exchange)
            if key not in number:
                kept.append(exchange)
            elif key in keep:
                kept.append([self.module.prefix_text(exchange[0], f"[e{number[key]}] ")] + exchange[1:])
        first = (kept[0] if kept else current)
        first[0] = self.module.prefix_text(first[0], SUMMARY_OPEN + compaction["summary"] + "\n" + SUMMARY_CLOSE) if compaction["summary"] else first[0]
        return head + [message for exchange in kept for message in exchange] + current

    async def run(self, model, identity, messages, name, argument, memory_off):
        ko = korean(messages[:-1])
        if identity is None:
            return ("명령은 로그인한 Open WebUI 채팅에서만 쓸 수 있습니다. 이 요청에는 채팅 식별 정보가 없습니다." if ko else
                    "Commands need a signed-in Open WebUI chat: this request carries no chat identity.")
        if name == "timer":
            return self.timer(argument, ko)
        if name == "memory":
            return self.memory_setting(identity, argument, memory_off, ko)
        if name == "target":
            return self.target(model, identity, argument, ko)
        history = strip(messages[:-1])
        if name == "context":
            return self.context_state(model, identity, history, memory_off) if argument.lower() == "json" else self.context(model, identity, history, memory_off, ko)
        return await self.compact(model, identity, history, argument, memory_off, ko)

    def memory_setting(self, identity, argument, memory_off, ko=False):
        value = argument.lower()
        if value in ("on", "off"):
            self.memory.store.append(*identity, {"type": "settings", "memory": value})
            if ko:
                note = " 이 요청은 전체 컨텍스트 토글이 켜져 있어, 토글을 끌 때까지 메모리는 꺼진 상태로 유지됩니다." if value == "on" and memory_off else ""
                return f"이 채팅의 메모리를 {'켰습니다' if value == 'on' else '껐습니다'}.{note}"
            note = " The Full context toggle is on for this request, so memory stays off until you turn it off." if value == "on" and memory_off else ""
            return f"Memory is {value} for this chat.{note}"
        state = "off" if memory_off or self.settings(identity).get("memory") == "off" else "on"
        if ko:
            return f"이 채팅의 메모리는 {'켜져' if state == 'on' else '꺼져'} 있습니다. `/memory off`는 채팅 전체를 보내고, `/memory on`은 메모리를 다시 켭니다."
        return f"Memory is {state} for this chat. `/memory off` sends the full chat; `/memory on` turns memory back on."

    def target(self, model, identity, argument, ko=False):
        if argument.lower() in ("off", "none", "0"):
            self.memory.store.append(*identity, {"type": "settings", "target": None})
            return "컨텍스트 목표가 없습니다. 메모리는 기본 예산을 씁니다." if ko else "No context target: memory uses its default budgets."
        tokens = size(argument, model.context_tokens)
        if tokens is None:
            current = self.settings(identity).get("target")
            if ko:
                return (f"컨텍스트 목표는 {label(current)} 토큰입니다." if current else "설정된 컨텍스트 목표가 없습니다.") + \
                    f" `/target 64K`, `/target 25%`({label(model.context_tokens)} 기준) 또는 `/target off`."
            return (f"The context target is {label(current)} tokens." if current else "No context target is set.") + \
                f" `/target 64K`, `/target 25%` (of {label(model.context_tokens)}) or `/target off`."
        tokens = min(tokens, model.context_tokens)
        self.memory.store.append(*identity, {"type": "settings", "target": tokens})
        if ko:
            return f"컨텍스트 목표 {label(tokens)} 토큰: 메모리는 프롬프트를 이 값의 25% 안으로 유지하며, {label(model.context_tokens)}는 넘을 수 없는 한도입니다."
        return f"Context target {label(tokens)} tokens: memory keeps the prompt within 25% of it, and at {label(model.context_tokens)} it is a hard limit."

    def timer(self, argument, ko=False):
        if argument.lower() in ("off", "stop"):
            return "채팅 타이머를 껐습니다." if ko else "Chat timer off."
        found = TIMER.match(argument)
        if not found:
            return ("`/timer every 10m <프롬프트>`로 채팅 타이머를 설정합니다. 타이머 상자가 체크되어 있는 동안 실행되며, `/timer off`로 멈춥니다." if ko else
                    "`/timer every 10m <prompt>` sets the chat timer; it runs while the Timer box is ticked. `/timer off` stops it.")
        if ko:
            return (f"채팅 타이머: {found.group(1)}{found.group(2).lower()}마다 다음을 보냅니다: {found.group(3).strip()}\n"
                    "타이머 상자가 체크되어 있는 동안 실행되며, 모델이 set_chat_timer로 바꿀 수 있습니다.")
        return f"Chat timer: every {found.group(1)}{found.group(2).lower()}, it sends: {found.group(3).strip()}\nIt runs while the Timer box is ticked; the model can change it with set_chat_timer."

    def sizes(self, model, exchanges):
        return [self.memory.count(model, self.module.messages_text(exchange)) for exchange in exchanges]

    def context_state(self, model, identity, history, memory_off):
        settings = self.settings(identity)
        parts = self.module.split(history + [{"role": "user", "content": ""}])
        exchanges = parts[1] if parts else []
        compaction = self.memory.store.load(*identity)["compaction"]
        return json.dumps({"model": model.id, "context_tokens": model.context_tokens, "exchanges": len(exchanges),
                           "history_tokens": sum(self.sizes(model, exchanges)), "memory": "off" if memory_off or settings.get("memory") == "off" else "on",
                           "target": settings.get("target"), "compaction": {"kept": len(compaction["keep"]), "of": len(compaction["covered"])} if compaction else None})

    def context(self, model, identity, history, memory_off, ko=False):
        settings = self.settings(identity)
        parts = self.module.split(history + [{"role": "user", "content": ""}])
        exchanges = parts[1] if parts else []
        total = sum(self.sizes(model, exchanges))
        off = memory_off or settings.get("memory") == "off"
        compaction = self.memory.store.load(*identity)["compaction"]
        if ko:
            lines = [f"이 채팅에는 대화 {len(exchanges)}개, {total:,} 토큰이 있으며, {model.id}의 컨텍스트는 {model.context_tokens:,} 토큰입니다."]
            lines.append("메모리가 꺼져 있어 요청마다 채팅 전체를 보냅니다." if off else
                         "메모리가 켜져 있어 프롬프트는 최신 대화와 이전 대화마다 요약 한 줄을 담습니다" +
                         (f". 목표는 약 {label(settings['target'])} 토큰입니다." if settings.get("target") else "."))
            if compaction:
                lines.append(f"압축으로 이전 대화 {len(compaction['covered'])}개 중 {len(compaction['keep'])}개를 그대로 두고 나머지는 요약했습니다. `/compact undo`로 되돌릴 수 있습니다.")
            lines.append("`/compact 50%`는 가장 관련 있는 절반을 남기고, `/compact full`은 요약만 남깁니다. `/memory on|off`, `/target 64K`.")
            return "\n".join(lines)
        lines = [f"The chat holds {len(exchanges)} exchange{'' if len(exchanges) == 1 else 's'}, {total:,} tokens, of the {model.context_tokens:,} {model.id} serves."]
        lines.append("Memory is off: every request sends the whole chat." if off else
                     f"Memory is on: the prompt keeps the newest exchanges and a summary line for each older one" +
                     (f", near a target of {label(settings['target'])} tokens." if settings.get("target") else "."))
        if compaction:
            lines.append(f"A compaction keeps {len(compaction['keep'])} of {len(compaction['covered'])} earlier exchanges in full and summarizes the rest; `/compact undo` restores them.")
        lines.append("`/compact 50%` keeps the most relevant half, `/compact full` keeps only a summary, `/memory on|off`, `/target 64K`.")
        return "\n".join(lines)

    async def compact(self, model, identity, history, argument, memory_off, ko=False):
        if argument.lower() in ("undo", "off"):
            self.memory.store.append(*identity, {"type": "compaction", "undo": True})
            return ("압축을 되돌렸습니다. 다음 프롬프트에는 모든 대화가 다시 들어갑니다." if ko else
                    "Compaction undone: the next prompt carries every exchange again.")
        if not (memory_off or self.settings(identity).get("memory") == "off"):
            if ko:
                return ("메모리가 켜져 있어 프롬프트가 이미 최신 대화를 담고 나머지를 요약합니다. 크기는 `/target`으로 정합니다. "
                        "압축은 메모리가 꺼진 채팅용입니다: `/memory off` 다음에 `/compact 50%`.")
            return ("Memory is on, so the prompt already keeps the newest exchanges and summarizes the rest; `/target` sets how large it gets. "
                    "Compaction is for chats with memory off: `/memory off`, then `/compact 50%`.")
        parts = self.module.split(history + [{"role": "user", "content": ""}])
        exchanges = parts[1] if parts else []
        if len(exchanges) < 2:
            return "아직 압축할 내용이 없습니다." if ko else "There is nothing to compact yet."
        sizes = self.sizes(model, exchanges)
        total = sum(sizes)
        full = argument.lower() == "full"
        budget = 0 if full else size(argument or f"{int(DEFAULT_SHARE * 100)}%", total)
        if budget is None:
            return ("`/compact 50%`, `/compact 40K`, `/compact full` 또는 `/compact undo`." if ko else
                    "`/compact 50%`, `/compact 40K`, `/compact full` or `/compact undo`.")
        ids = [f"e{index + 1}" for index in range(len(exchanges))]
        newest = ids[-1]
        listing = []
        for index, exchange in enumerate(exchanges):
            user, answer, _ = self.module.reading_parts(exchange)
            listing.append(f"[{ids[index]}] {sizes[index]:,} tokens\nUser: {user[:EXCERPT_USER]}\nAssistant: {answer[:EXCERPT_ASSISTANT]}")
        instruction = ("Compact this conversation for its next turns. " +
                       (f"Keep in full the exchanges most relevant to continuing it, within {budget:,} tokens in total; always keep {newest}. "
                        if not full else "Keep no exchange in full. ") +
                       "Write a summary that covers every exchange you do not keep, each marked with its id in parentheses like (e2): "
                       "facts, decisions, names, numbers, file paths, code identifiers and open questions, in reported speech, "
                       "so the conversation can continue without them. "
                       'Reply with one JSON object and nothing else: {"keep": ["e3", "e9"], "summary": "..."}')
        messages = [{"role": "system", "content": "You compact conversations. Answer with JSON only."},
                    {"role": "user", "content": "\n\n".join(listing) + "\n\n" + instruction}]
        profile = self.memory.profile(model)
        prompt_ids, reasoning_open = self.frontend.memory_curator_ids(model, messages, None, {}, profile.curator_suffix)
        answer_tokens = max(1024, min(8192, total // 8))
        if len(prompt_ids) + answer_tokens > model.context_tokens:
            return (f"압축할 목록이 {len(prompt_ids):,} 토큰이라 이 모델의 {model.context_tokens:,} 토큰에 답과 함께 들어가지 않습니다. 아무것도 바뀌지 않았습니다." if ko else
                    f"The compaction listing is {len(prompt_ids):,} tokens, which with its answer does not fit {model.id}'s {model.context_tokens:,}; nothing changed.")
        if not await self.frontend.memory_model_ready(model):
            return (f"{model.id}가 아직 로드 중이거나 다른 모델로 바뀌는 중이라 압축하지 않았습니다. 모델이 답하면 다시 시도하세요." if ko else
                    f"{model.id} is loading or the fleet is switching models, so nothing was compacted; try again once it answers.")
        text = await self.frontend.memory_complete(model, prompt_ids, answer_tokens, reasoning_open, "compact", {"chat": identity[1][:12], "user_message": "compact"})
        choice = parse_choice(text, ids)
        if choice is None:
            return ("모델의 압축 응답이 요청한 JSON이 아니어서 아무것도 바뀌지 않았습니다. 다시 시도하세요." if ko else
                    "The model's compaction answer was not the JSON it was asked for; nothing changed. Try again.")
        keep, summary = choice
        keep = [] if full else sorted(set(keep) | {newest}, key=ids.index)
        while not full and len(keep) > 1 and sum(sizes[ids.index(item)] for item in keep) > budget:
            keep.remove(min((item for item in keep if item != newest), key=lambda item: ids.index(item)))
        missing = [item for item in ids if item not in keep and f"({item})" not in summary]
        if missing:
            summary = "\n".join([summary] + [f"({item}) " + self.module.fallback_line(exchanges[ids.index(item)]) for item in missing]).strip()
        kept_tokens = sum(sizes[ids.index(item)] for item in keep)
        self.memory.store.append(*identity, {"type": "compaction", "covered": [self.module.exchange_key(exchange) for exchange in exchanges],
                                             "keep": [self.module.exchange_key(exchanges[ids.index(item)]) for item in keep], "summary": summary,
                                             "target": budget})
        summary_tokens = self.memory.count(model, summary)
        if ko:
            return (f"{total:,} 토큰을 {kept_tokens + summary_tokens:,} 토큰으로 압축했습니다: " +
                    (f"{', '.join(keep)}는 그대로 두고({kept_tokens:,} 토큰) " if keep else "") +
                    f"나머지 대화 {len(exchanges) - len(keep)}개는 {summary_tokens:,} 토큰으로 요약했습니다. `/compact undo`로 되돌릴 수 있습니다.")
        return (f"Compacted {total:,} tokens to {kept_tokens + summary_tokens:,}: " +
                (f"kept {', '.join(keep)} in full ({kept_tokens:,} tokens) and " if keep else "") +
                f"summarized the other {len(exchanges) - len(keep)} exchanges in {summary_tokens:,} tokens. `/compact undo` restores them.")
