(function () {
  var P = window.SP_API_PRICES, S = window.SP_PRICING;
  var ko = document.documentElement.lang === "ko";
  var T = ko ? {
    reading: "읽는 중", none: "읽을 수 있는 Claude Code나 Codex 요청을 찾지 못했습니다. ~/.claude/projects 또는 ~/.codex/sessions 안의 .jsonl 파일을 고르세요.",
    model: "모델", uncached: "캐시 안 된 입력", writes: "캐시 쓰기", reads: "캐시 읽기", output: "출력", cost: "비용", total: "합계",
    unknown: "공개 가격 없음", period: "기간", active: "작업한 날", perDay: "작업한 날 하루 평균", perWeek: "일주일로 환산",
    rewrites: "멈춘 뒤 컨텍스트를 다시 쓴 요청", rewriteTokens: "다시 쓴 토큰", rewriteCost: "캐시 읽기였다면 아꼈을 비용",
    fleet: "SparkPipe 플릿 1주", team: "엔지니어 3~5명이 함께 쓰면 한 명당", skipped: "가격을 찾지 못한 모델의 요청은 비용 합계에서 뺐습니다: "
  } : {
    reading: "Reading", none: "No Claude Code or Codex requests found. Choose .jsonl files from ~/.claude/projects or ~/.codex/sessions.",
    model: "Model", uncached: "Uncached input", writes: "Cache writes", reads: "Cache reads", output: "Output", cost: "Cost", total: "Total",
    unknown: "no published price", period: "Period", active: "Days worked", perDay: "Per day worked", perWeek: "Scaled to a week",
    rewrites: "Requests that wrote their context again after a pause", rewriteTokens: "Tokens written again", rewriteCost: "Saved had they been cache reads",
    fleet: "A week of a SparkPipe fleet", team: "Shared by three to five engineers, each", skipped: "Requests on models without a published price are left out of the total: "
  };
  var M = 1e6, MIN_PROMPT = 10000, OPENAI_TTL = 300;
  var el = function (id) { return document.getElementById(id); };
  var usd = function (n) { var d = n >= 100 ? 0 : 2; return "$" + n.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }); };
  var num = function (n) { return Math.round(n).toLocaleString("en-US"); };
  var dayCount = function (n) { return ko ? n + "일" : n + (n === 1 ? " day" : " days"); };
  var tally = function (f, t, s) {
    if (ko) return "파일 " + num(f) + "개 · 요청 " + num(t) + "건 · 세션 " + num(s) + "개";
    var one = function (n, w) { return num(n) + " " + w + (n === 1 ? "" : "s"); };
    return one(f, "file") + " · " + one(t, "request") + " · " + one(s, "session");
  };
  var date = function (t) { var d = new Date(t); return d.getFullYear() + "-" + String(d.getMonth() + 1).padStart(2, "0") + "-" + String(d.getDate()).padStart(2, "0"); };

  function norm(id) {
    var s = String(id || "").toLowerCase();
    var i = s.indexOf("claude-");
    if (i > 0) s = s.slice(i);
    return s.split(/[\[@:]/)[0];
  }
  function find(list, id) {
    var best = null;
    list.forEach(function (m) {
      if (id.indexOf(m.id) !== 0) return;
      if (!/^(-\d{8}|-\d{4}-\d{2}-\d{2})?(-v\d+)?$/.test(id.slice(m.id.length))) return;
      if (!best || m.id.length > best.id.length) best = m;
    });
    return best;
  }

  var claude = {}, codex = [], fileCount = 0;

  function readClaude(o) {
    var m = o.message, u = m && m.usage;
    if (!u || !m.model || m.model.charAt(0) === "<") return false;
    var cc = u.cache_creation || {};
    var written = u.cache_creation_input_tokens || 0;
    var w1 = cc.ephemeral_1h_input_tokens || 0;
    var w5 = cc.ephemeral_5m_input_tokens != null ? cc.ephemeral_5m_input_tokens : Math.max(0, written - w1);
    var key = (m.id || o.uuid) + "|" + (o.requestId || "");
    var t = { kind: "anthropic", model: norm(m.model), time: Date.parse(o.timestamp) || 0,
      session: (o.sessionId || "") + (o.isSidechain ? ":" + (o.agentId || "side") : ""),
      input: u.input_tokens || 0, w5: w5, w1: w1, read: u.cache_read_input_tokens || 0, output: u.output_tokens || 0 };
    var old = claude[key];
    if (!old || t.output >= old.output) claude[key] = t;
    return true;
  }

  function readCodex(lines, name) {
    var model = "", session = name, last = "", found = false;
    lines.forEach(function (o) {
      var p = o.payload || {};
      if (o.type === "session_meta" && p.id) session = p.id;
      if (o.type === "turn_context" && p.model) model = String(p.model).toLowerCase();
      if (p.type !== "token_count" || !p.info || !p.info.last_token_usage) return;
      var sig = JSON.stringify(p.info.total_token_usage || p.info.last_token_usage);
      if (sig === last) return;
      last = sig;
      var u = p.info.last_token_usage;
      codex.push({ kind: "openai", model: model || "unknown", time: Date.parse(o.timestamp) || 0, session: session,
        input: u.input_tokens || 0, cached: u.cached_input_tokens || 0, output: u.output_tokens || 0 });
      found = true;
    });
    return found;
  }

  function parse(text, name) {
    var rows = [];
    text.split("\n").forEach(function (line) {
      if (!line) return;
      try { rows.push(JSON.parse(line)); } catch (e) {}
    });
    var any = false;
    rows.forEach(function (o) { if (o && o.type === "assistant" && readClaude(o)) any = true; });
    if (readCodex(rows, name)) any = true;
    if (any) fileCount++;
  }

  function priceOf(t) {
    if (t.kind === "anthropic") {
      var a = find(P.anthropic, t.model);
      if (!a) return null;
      var prompt = t.input + t.w5 + t.w1 + t.read;
      return { m: a, r: a.long && prompt > a.long.over ? a.long : a };
    }
    var o = find(P.openai, t.model);
    if (!o) return null;
    return { m: o, r: o.long && t.input > o.long.over ? o.long : o };
  }

  function cost(t, p) {
    var r = p.r;
    if (t.kind === "anthropic") return (t.input * r.input + t.w5 * r.write5m + t.w1 * r.write1h + t.read * r.read + t.output * r.output) / M;
    var cached = r.cached == null ? 0 : Math.min(t.cached, t.input);
    return ((t.input - cached) * r.input + cached * (r.cached || 0) + t.output * r.output) / M;
  }

  function analyse() {
    var all = Object.keys(claude).map(function (k) { return claude[k]; }).concat(codex);
    if (!all.length) return null;
    var by = {}, sessions = {}, days = {}, first = Infinity, last = 0, total = 0, skipped = {};
    var rw = { count: 0, tokens: 0, saved: 0 };
    all.forEach(function (t) {
      var p = priceOf(t);
      var row = by[t.model] || (by[t.model] = { model: p ? p.m.name : t.model, turns: 0, uncached: 0, writes: 0, reads: 0, output: 0, cost: p ? 0 : null });
      row.turns++;
      row.output += t.output;
      if (t.kind === "anthropic") { row.uncached += t.input; row.writes += t.w5 + t.w1; row.reads += t.read; }
      else { row.uncached += t.input - t.cached; row.reads += t.cached; }
      if (p) { var c = cost(t, p); row.cost += c; total += c; } else skipped[t.model] = true;
      (sessions[t.session] || (sessions[t.session] = [])).push(t);
      if (t.time) { first = Math.min(first, t.time); last = Math.max(last, t.time); days[date(t.time)] = true; }
    });
    Object.keys(sessions).forEach(function (k) {
      var list = sessions[k].sort(function (a, b) { return a.time - b.time; });
      for (var i = 1; i < list.length; i++) {
        var t = list[i], gap = (t.time - list[i - 1].time) / 1000, p = priceOf(t);
        if (!t.time || !list[i - 1].time) continue;
        if (t.kind === "anthropic") {
          var written = t.w5 + t.w1, prompt = t.input + written + t.read, ttl = t.w1 > 0 ? 3600 : 300;
          if (gap > ttl && prompt >= MIN_PROMPT && written * 2 >= prompt) {
            rw.count++; rw.tokens += written;
            if (p) rw.saved += (t.w5 * (p.r.write5m - p.r.read) + t.w1 * (p.r.write1h - p.r.read)) / M;
          }
        } else {
          var uncached = t.input - t.cached;
          if (gap > OPENAI_TTL && t.input >= MIN_PROMPT && uncached * 2 >= t.input) {
            rw.count++; rw.tokens += uncached;
            if (p && p.r.cached != null) rw.saved += uncached * (p.r.input - p.r.cached) / M;
          }
        }
      }
    });
    var span = last > first ? (last - first) / 86400000 : 0;
    return { rows: Object.keys(by).map(function (k) { return by[k]; }).sort(function (a, b) { return (b.cost || 0) - (a.cost || 0); }),
      total: total, turns: all.length, sessions: Object.keys(sessions).length, first: first, last: last, span: span,
      active: Object.keys(days).length, rw: rw, skipped: Object.keys(skipped) };
  }

  function cell(tag, text, cls) { var c = document.createElement(tag); c.textContent = text; if (cls) c.className = cls; return c; }

  function render() {
    var a = analyse(), out = el("results"), status = el("status");
    out.hidden = !a;
    if (!a) { status.textContent = T.none; return; }
    status.textContent = tally(fileCount, a.turns, a.sessions);
    var facts = el("facts");
    facts.textContent = "";
    var week = S ? S.bookings.weekHC * S.hc.prices[S.hc.prices.length - 1].usd : null;
    var items = [
      [T.period, isFinite(a.first) ? date(a.first) + " – " + date(a.last) + " (" + dayCount(Math.max(1, Math.round(a.span))) + ")" : "–"],
      [T.active, String(a.active)],
      [T.total, usd(a.total)],
      [T.perDay, a.active ? usd(a.total / a.active) : "–"],
      [T.perWeek, a.span >= 1 ? usd(a.total / a.span * 7) : "–"],
      [T.rewrites, num(a.rw.count)],
      [T.rewriteTokens, num(a.rw.tokens)],
      [T.rewriteCost, usd(a.rw.saved)]
    ];
    if (week) items.push([T.fleet, usd(week)], [T.team, usd(week / 5) + " – " + usd(week / 3)]);
    items.forEach(function (it) {
      var d = document.createElement("div");
      d.appendChild(cell("small", it[0]));
      d.appendChild(cell("strong", it[1]));
      facts.appendChild(d);
    });
    var body = el("rows");
    body.textContent = "";
    a.rows.forEach(function (r) {
      var tr = document.createElement("tr");
      tr.appendChild(cell("td", r.model));
      [r.turns, r.uncached, r.writes, r.reads, r.output].forEach(function (n) { tr.appendChild(cell("td", num(n), "num")); });
      tr.appendChild(cell("td", r.cost == null ? T.unknown : usd(r.cost), "num"));
      body.appendChild(tr);
    });
    el("skipped").textContent = a.skipped.length ? T.skipped + a.skipped.join(", ") : "";
  }

  function readFiles(files) {
    var list = Array.prototype.filter.call(files, function (f) { return /\.jsonl?$/i.test(f.name); });
    var i = 0;
    function next() {
      if (i >= list.length) { render(); return; }
      var f = list[i++];
      el("status").textContent = T.reading + " " + i + " / " + list.length;
      f.text().then(function (text) { parse(text, f.name); next(); }, next);
    }
    next();
  }

  function walk(entries, done) {
    var files = [], pending = 0;
    function finish() { if (--pending === 0) done(files); }
    function visit(entry) {
      pending++;
      if (entry.isFile) { entry.file(function (f) { files.push(f); finish(); }, finish); return; }
      var reader = entry.createReader();
      (function more() {
        reader.readEntries(function (batch) {
          if (!batch.length) { finish(); return; }
          batch.forEach(visit);
          more();
        }, finish);
      })();
    }
    pending++;
    entries.forEach(visit);
    finish();
  }

  el("files").addEventListener("change", function (e) { readFiles(e.target.files); });
  el("folder").addEventListener("change", function (e) { readFiles(e.target.files); });
  var zone = el("zone");
  ["dragenter", "dragover"].forEach(function (n) { zone.addEventListener(n, function (e) { e.preventDefault(); zone.classList.add("over"); }); });
  ["dragleave", "drop"].forEach(function (n) { zone.addEventListener(n, function () { zone.classList.remove("over"); }); });
  zone.addEventListener("drop", function (e) {
    e.preventDefault();
    var items = e.dataTransfer.items, entries = [];
    if (items && items.length && items[0].webkitGetAsEntry) {
      Array.prototype.forEach.call(items, function (it) { var en = it.webkitGetAsEntry(); if (en) entries.push(en); });
      walk(entries, readFiles);
    } else readFiles(e.dataTransfer.files);
  });
  el("clear").addEventListener("click", function () {
    claude = {}; codex = []; fileCount = 0;
    el("results").hidden = true; el("status").textContent = ""; el("files").value = ""; el("folder").value = "";
  });
  el("checked").textContent = P.checked;
})();
