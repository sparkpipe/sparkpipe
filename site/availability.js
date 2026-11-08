(() => {
  const SANDBOX = new URLSearchParams(location.search).has('sandbox');
  const API = SANDBOX ? '/pay/sandbox/api' : '/pay/api';
  const KO = document.documentElement.lang === 'ko';
  const HOUR = 3600000;
  const DAY = 24 * HOUR;
  const WEEK = 7 * DAY;
  const KST = 9 * HOUR;
  const NS = 'http://www.w3.org/2000/svg';
  const ROW = 30;
  const GAP = 14;
  const HEAD = 34;
  const HOUR_W = 7;
  const WEEK_W = 20;
  const $ = (id) => document.getElementById(id);
  const S = { state: null, view: 'hours', grid: null, drawn: null };

  const days = KO ? ['일', '월', '화', '수', '목', '금', '토'] : ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  const months = KO ? null : ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const k = (ms) => new Date(ms + KST);
  const pad = (n) => String(n).padStart(2, '0');
  const dayText = (ms) => {
    const d = k(ms);
    return KO ? `${d.getUTCMonth() + 1}월 ${d.getUTCDate()}일 (${days[d.getUTCDay()]})` : `${days[d.getUTCDay()]} ${d.getUTCDate()} ${months[d.getUTCMonth()]}`;
  };
  const shortDay = (ms) => {
    const d = k(ms);
    return KO ? `${d.getUTCMonth() + 1}/${d.getUTCDate()} ${days[d.getUTCDay()]}` : `${days[d.getUTCDay()]} ${d.getUTCDate()}`;
  };
  const timeText = (ms) => `${dayText(ms)} ${pad(k(ms).getUTCHours())}:00`;
  const midnight = (ms) => Math.floor((ms + KST) / DAY) * DAY - KST;
  const monday = (ms) => {
    const d = Math.floor((ms + KST) / DAY);
    return (d - ((d + 3) % 7)) * DAY - KST;
  };
  const spans = (list) => (list || []).map(([a, b]) => [Date.parse(a), Date.parse(b)]);
  const fleetName = (f) => (KO ? f.name.replace(/^Fleet /, '플릿 ') : f.name);
  const runEnd = (list, span) => list.filter(([a]) => a >= span[0]).sort((x, y) => x[0] - y[0]).reduce((end, [a, z]) => (a <= end ? Math.max(end, z) : end), span[1]);
  const within = (list, t) => list.find(([a, b]) => a <= t && t < b);
  const overlaps = (list, a, b) => list.some(([x, y]) => x < b && y > a);
  const left = (ms) => {
    const h = Math.max(0, Math.round(ms / HOUR));
    return KO ? `${Math.floor(h / 24)}일 ${h % 24}시간 남음` : `in ${Math.floor(h / 24)} d ${h % 24} h`;
  };

  const T = KO ? {
    loading: '예약 현황을 불러오는 중…',
    failed: '예약 현황을 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.',
    asOf: (when) => `${when} KST 기준, 30초마다 갱신`,
    sandbox: '샌드박스: 테스트 시계와 테스트 계정의 예약입니다. 계정 페이지의 샌드박스에서 직접 해 볼 수 있습니다.',
    pinnedA: (n, m, of, when) => `주 단위 플릿 ${of}개 중 지금 ${n}개, ${when}부터 ${m}개 고정`,
    noneServing: '아직 서비스 중인 주 단위 플릿이 없습니다',
    locks: (n) => n ? `대기 명단 잠금 ${n}개` : '대기 명단이 비어 있음',
    pairs: (o, l) => l ? `다음 주 고정 가능한 쌍: ${l}개 중 ${o}개` : '아직 고정할 수 있는 쌍이 없습니다',
    sunday: (when, rest) => `다음 마감과 배정: ${when} 00:00 KST (${rest})`,
    pair: (n) => `쌍 ${n}`,
    weekly: '주 단위',
    hourly: '시간 단위',
    now: '지금',
    deadline: '일 00:00 마감·배정',
    names: { past: '지남', building: '구축 중', maint: '점검', pinned: '고정됨', booked: '시간 예약됨', open: '지금 예약 가능', pin: '월요일부터 고정 가능', hold: '보유자 우선', wait: '대기 명단에 배정', later: '아직 열리지 않음' },
    labels: { pinned: (end) => `${dayText(end - DAY)}까지 고정`, pin: '월요일부터 고정 가능', hold: '보유자가 일 00:00까지 연장 가능', wait: '일요일에 대기 명단에 배정', building: '구축 중' },
    detail: {
      past: '지난 시간입니다.',
      building: '이 플릿은 아직 구축 중이며, 서비스를 시작하면 예약할 수 있습니다.',
      maint: '공지된 점검 시간입니다. 이 시간을 보유한 고객은 HC로 적립받습니다.',
      pinned: (a, b, pair) => `${dayText(a)}부터 ${dayText(b - DAY)}까지 한 고객에게 고정되어 있습니다${pair ? ' (A 반쪽과 함께 4 TB 쌍으로)' : ''}. 다른 누구도 이 시간을 예약할 수 없습니다. 보유자는 ${dayText(b - DAY)} 00:00 KST 전에 다음 주를 예약하면 계속 보유합니다.`,
      booked: '다른 계정이 시간 단위로 예약한 시간입니다. 예약은 밀려나지 않습니다.',
      open: (hour) => hour ? '지금 시간 단위로 예약할 수 있습니다. 계정당 1주에 시간 플릿 24시간까지, 7일 앞까지.' : '아무도 보유하지 않아 이번 주 끝까지 시간 단위로 예약할 수 있습니다.',
      pin: (when) => `비어 있습니다. 먼저 예약하는 계정이 ${dayText(when)}부터 첫 주로 고정합니다.`,
      hold: (when, locks) => `현재 보유자가 ${dayText(when - DAY)} 00:00 KST까지 이 주를 예약할 우선권을 가집니다. 예약하지 않으면 ${locks ? `대기 명단(잠금 ${locks}개) 중 가장 큰 잠금에` : '먼저 예약하는 계정에'} 넘어갑니다.`,
      wait: (locks) => `비어 있지만 대기 명단에 잠금 ${locks}개가 있어, 일요일 00:00 KST에 가장 큰 잠금에 배정됩니다.`,
      later: (hour) => hour ? '시간 예약은 7일 앞까지만 열립니다.' : '비어 있지만 아직 예약할 수 없습니다. 첫 주는 항상 다음 월요일부터 시작합니다.',
    },
    summary: {
      building: '구축 중',
      pinned: (from, end) => `${from ? `${dayText(from)}부터 ` : ''}${dayText(end - DAY)}까지 고정 (보유자 마감 ${dayText(end - DAY)} 00:00 KST)`,
      pin: (when) => `비어 있음: ${dayText(when)}부터 고정 가능`,
      hold: '이번 주 고정, 다음 주는 아직 예약되지 않음',
      wait: '비어 있음: 일요일에 대기 명단에 배정',
      hours: (n) => `시간 플릿: 앞으로 7일 중 ${n}시간 예약됨`,
      free: '비어 있음',
    },
    textView: '보드를 글로 읽기',
  } : {
    loading: 'Loading the bookings…',
    failed: 'We could not load the bookings. Try again in a minute.',
    asOf: (when) => `As of ${when} KST, refreshed every 30 seconds`,
    sandbox: 'Sandbox: bookings by test accounts on a test clock. Play it yourself in the sandbox on your account page.',
    pinnedA: (n, m, of, when) => `Weekly fleets pinned: ${n} of ${of} now, ${m} from ${when}`,
    noneServing: 'No weekly fleet is serving yet',
    locks: (n) => n ? `${n} waitlist lock${n === 1 ? '' : 's'}` : 'Nobody on the waitlist',
    pairs: (o, l) => l ? `Pairs open to pin next week: ${o} of ${l}` : 'No pair can be pinned yet',
    sunday: (when, rest) => `Next deadline and allocation: ${when} 00:00 KST (${rest})`,
    pair: (n) => `Pair ${n}`,
    weekly: 'weekly',
    hourly: 'hourly',
    now: 'now',
    deadline: 'Sun 00:00 deadline and allocation',
    names: { past: 'Past', building: 'Being built', maint: 'Maintenance', pinned: 'Pinned', booked: 'Booked by the hour', open: 'Open to book now', pin: 'Free to pin from Monday', hold: "Holder's option", wait: 'Goes to the waitlist', later: 'Not open yet' },
    labels: { pinned: (end) => `pinned through ${dayText(end - DAY)}`, pin: 'free to pin from Monday', hold: 'holder may extend until Sun 00:00', wait: 'goes to the waitlist on Sunday', building: 'being built' },
    detail: {
      past: 'This hour has passed.',
      building: 'This fleet is being built. It can be booked once it serves.',
      maint: 'Announced maintenance. Whoever holds these hours is credited in HC.',
      pinned: (a, b, pair) => `Pinned to one customer from ${dayText(a)} through ${dayText(b - DAY)}${pair ? ', with its A half as a 4 TB pair' : ''}. Nobody else can book these hours. The holder keeps it by booking the next week before ${dayText(b - DAY)} 00:00 KST.`,
      booked: 'Booked by the hour by another account. No booking is ever bumped.',
      open: (hour) => hour ? 'Open: book it by the hour now, up to 7 days ahead and 24 hours of Hour Fleets a week per account.' : 'Nobody holds it, so it can be booked by the hour until the end of this week.',
      pin: (when) => `Free. The first account to book it pins it from ${dayText(when)} as its first week.`,
      hold: (when, locks) => `Its holder may book this week until ${dayText(when - DAY)} 00:00 KST. If they do not, it goes to ${locks ? `the biggest of the ${locks} waitlist lock${locks === 1 ? '' : 's'}` : 'whoever books it first'}.`,
      wait: (locks) => `Free, but ${locks} waitlist lock${locks === 1 ? ' is' : 's are'} waiting, so on Sunday at 00:00 KST it goes to the biggest lock.`,
      later: (hour) => hour ? 'Hours open at most 7 days ahead.' : 'Free, but not bookable yet: a first week always starts on the next Monday.',
    },
    summary: {
      building: 'being built',
      pinned: (from, end) => `pinned ${from ? `from ${dayText(from)} ` : ''}through ${dayText(end - DAY)} (holder's deadline ${dayText(end - DAY)} 00:00 KST)`,
      pin: (when) => `free: pin it from ${dayText(when)}`,
      hold: 'pinned this week; next week not booked yet',
      wait: 'free: goes to the waitlist on Sunday',
      hours: (n) => `Hour Fleet: ${n} hour${n === 1 ? '' : 's'} booked in the next 7 days`,
      free: 'free',
    },
    textView: 'Read the board as text',
  };

  const context = (st) => {
    const now = Date.parse(st.now);
    const weekStart = monday(now);
    const weekEnd = weekStart + WEEK;
    return {
      now,
      weekStart,
      weekEnd,
      decide: now < weekEnd - DAY ? weekEnd : weekEnd + WEEK,
      locks: (st.waitlist && st.waitlist.locks) || 0,
      hourDays: (st.bookings && st.bookings.hourFleetDaysAhead) || 7,
      hoursFrom: st.sandbox || !st.hoursFrom ? 0 : Date.parse(st.hoursFrom + 'T00:00:00+09:00'),
    };
  };

  const fleetData = (f) => ({ f, pinned: spans(f.pinned), booked: spans(f.booked), maint: spans(f.maintenance), first: f.firstWeek ? Date.parse(f.firstWeek) : 0 });

  const nextWeek = (d, c) => {
    if (within(d.pinned, c.decide + HOUR)) return 'pinned';
    if (d.first) return 'pin';
    if (within(d.pinned, c.decide - HOUR)) return 'hold';
    if (d.f.hourFleet) return null;
    return c.locks ? 'wait' : 'later';
  };

  const hourState = (d, t, c) => {
    if (t + HOUR <= c.now) return 'past';
    if (d.f.serving === false) return 'building';
    if (within(d.maint, t)) return 'maint';
    if (within(d.pinned, t)) return 'pinned';
    if (within(d.booked, t)) return 'booked';
    if (c.hoursFrom && t < c.hoursFrom) return 'later';
    if (d.f.hourFleet) {
      if (t >= c.decide && t < c.decide + WEEK && within(d.pinned, c.decide - HOUR)) return 'hold';
      return t < c.now + c.hourDays * DAY ? 'open' : 'later';
    }
    if (t < c.weekEnd) return 'open';
    if (t < c.decide) return d.first ? 'pin' : 'later';
    if (t < c.decide + WEEK) return nextWeek(d, c);
    return 'later';
  };

  const weekState = (d, w, c) => {
    if (d.f.serving === false) return 'building';
    if (within(d.pinned, w + HOUR)) return 'pinned';
    if (c.hoursFrom && w + WEEK <= c.hoursFrom) return 'later';
    if (w === c.weekStart) return overlaps(d.booked, w, w + WEEK) ? 'booked' : 'open';
    if (w === c.decide) {
      if (d.f.hourFleet) return within(d.pinned, c.decide - HOUR) ? 'hold' : 'open';
      return nextWeek(d, c);
    }
    if (w < c.decide && !d.f.hourFleet) return d.first ? 'pin' : 'later';
    if (d.f.hourFleet && w < c.now + c.hourDays * DAY) return overlaps(d.booked, w, w + WEEK) ? 'booked' : 'open';
    return 'later';
  };

  const el = (name, attrs, text) => {
    const node = document.createElementNS(NS, name);
    Object.entries(attrs || {}).forEach(([key, value]) => node.setAttribute(key, value));
    if (text !== undefined) node.textContent = text;
    return node;
  };

  const defs = () => {
    const d = el('defs');
    const pattern = (id, size, draw) => {
      const p = el('pattern', { id, width: size, height: size, patternUnits: 'userSpaceOnUse' });
      draw.forEach((node) => p.append(node));
      d.append(p);
    };
    pattern('av-hatch', 6, [el('path', { d: 'M-1,1 l2,-2 M0,6 l6,-6 M5,7 l2,-2', class: 'av-line' })]);
    pattern('av-back', 8, [el('path', { d: 'M-1,7 l2,2 M0,0 l8,8 M7,-1 l2,2', class: 'av-line av-thin' })]);
    pattern('av-cross', 6, [el('path', { d: 'M0,0 l6,6 M6,0 l-6,6', class: 'av-line' })]);
    pattern('av-dots', 6, [el('circle', { cx: 3, cy: 3, r: 1.1, class: 'av-dot' })]);
    pattern('av-rows', 5, [el('path', { d: 'M0,2.5 h5', class: 'av-line av-thin' })]);
    return d;
  };

  const runs = (states) => {
    const found = [];
    states.forEach((s, i) => {
      const last = found[found.length - 1];
      if (last && last.s === s) last.n += 1;
      else found.push({ s, i, n: 1 });
    });
    return found;
  };

  function draw() {
    const st = S.state;
    const c = context(st);
    const fleets = st.fleets.map(fleetData);
    const hours = S.view === 'hours';
    const start = hours ? midnight(c.now) : c.weekStart;
    const step = hours ? HOUR : WEEK;
    const count = hours ? 14 * 24 : 52;
    const cw = hours ? HOUR_W : WEEK_W;
    const width = count * cw;
    const rows = [];
    let y = HEAD;
    fleets.forEach((d, index) => {
      if (index && fleets[index - 1].f.pair !== d.f.id) y += GAP;
      rows.push({ d, y });
      y += ROW;
    });
    const height = y + 6;
    const svg = el('svg', { width, height, viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': hours ? T.names.open : T.names.pinned, class: 'av-svg' });
    svg.append(defs());
    for (let i = 0; i < count; i += 1) {
      const t = start + i * step;
      const x = i * cw;
      const kd = k(t);
      if (hours) {
        if (kd.getUTCHours() === 0) {
          svg.append(el('line', { x1: x, x2: x, y1: 14, y2: height, class: kd.getUTCDay() === 1 ? 'av-week' : 'av-day' }));
          svg.append(el('text', { x: x + 3, y: 12, class: 'av-head' }, shortDay(t)));
        } else if (kd.getUTCHours() % 6 === 0) {
          svg.append(el('text', { x: x + 1, y: 28, class: 'av-tick' }, pad(kd.getUTCHours())));
        }
      } else {
        if (k(t).getUTCDate() <= 7) {
          svg.append(el('line', { x1: x, x2: x, y1: 14, y2: height, class: 'av-week' }));
          svg.append(el('text', { x: x + 3, y: 12, class: 'av-head' }, KO ? `${kd.getUTCMonth() + 1}월` : months[kd.getUTCMonth()]));
        }
        if (i % 4 === 0) svg.append(el('text', { x: x + 1, y: 28, class: 'av-tick' }, String(kd.getUTCDate())));
      }
    }
    const grid = [];
    rows.forEach(({ d, y: top }) => {
      const states = [];
      for (let i = 0; i < count; i += 1) states.push(hours ? hourState(d, start + i * step, c) : weekState(d, start + i * step, c));
      grid.push({ d, top, states });
      runs(states).forEach((r) => {
        svg.append(el('rect', { x: r.i * cw, y: top + 3, width: r.n * cw, height: ROW - 6, class: `av-cell st-${r.s}` }));
        const t0 = start + r.i * step;
        let label = '';
        if (r.s === 'pinned') {
          const span = within(d.pinned, t0 + HOUR) || within(d.pinned, t0);
          if (span) label = T.labels.pinned(span[1]);
        } else if (T.labels[r.s]) {
          label = T.labels[r.s];
        }
        if (label && r.n * cw > label.length * 6.4 + 10) svg.append(el('text', { x: r.i * cw + 6, y: top + ROW / 2 + 4, class: `av-label on-${r.s}` }, label));
      });
    });
    const sunday = c.decide - DAY;
    const sx = ((sunday - start) / step) * cw;
    if (sx > 0 && sx < width) {
      svg.append(el('line', { x1: sx, x2: sx, y1: HEAD - 4, y2: height, class: 'av-sunday' }));
    }
    const nx = ((c.now - start) / step) * cw;
    if (nx >= 0 && nx <= width) {
      svg.append(el('line', { x1: nx, x2: nx, y1: HEAD - 6, y2: height, class: 'av-now' }));
      svg.append(el('text', { x: nx + 3, y: HEAD - 7, class: 'av-nowtext' }, T.now));
    }
    const host = $('av-board');
    const kept = host.scrollLeft;
    host.replaceChildren(svg);
    host.scrollLeft = S.drawn === S.view ? kept : Math.max(0, Math.min(nx, width) - 24);
    S.drawn = S.view;
    const labels = $('av-rows');
    labels.style.height = `${height}px`;
    labels.replaceChildren(...rows.map(({ d, y: top }, index) => {
      const box = document.createElement('div');
      box.className = 'av-row';
      box.style.top = `${top}px`;
      box.style.height = `${ROW}px`;
      const first = !index || rows[index - 1].d.f.pair !== d.f.id;
      if (first && d.f.pair) {
        const tag = document.createElement('span');
        tag.className = 'av-pair';
        tag.textContent = T.pair(d.f.id.replace(/\D/g, ''));
        box.append(tag);
      }
      const name = document.createElement('b');
      name.textContent = fleetName(d.f);
      const kind = document.createElement('span');
      kind.className = 'av-kind';
      kind.textContent = d.f.hourFleet ? T.hourly : T.weekly;
      box.append(name, ' ', kind);
      return box;
    }));
    S.grid = { grid, start, step, cw, count, c };
    legend(hours);
    summaries(fleets, c);
  }

  function legend(hours) {
    const keep = hours ? ['pinned', 'booked', 'open', 'pin', 'hold', 'wait', 'later', 'maint', 'building', 'past'] : ['pinned', 'booked', 'open', 'pin', 'hold', 'wait', 'later', 'building'];
    $('av-legend').replaceChildren(...keep.map((s) => {
      const item = document.createElement('li');
      const swatch = el('svg', { width: 26, height: 14, 'aria-hidden': 'true', class: 'av-swatch' });
      swatch.append(defs(), el('rect', { x: 0.5, y: 0.5, width: 25, height: 13, class: `av-cell st-${s}` }));
      item.append(swatch, document.createTextNode(T.names[s]));
      return item;
    }));
  }

  function summaries(fleets, c) {
    const list = $('av-text');
    list.replaceChildren(...fleets.map((d) => {
      const item = document.createElement('li');
      const b = document.createElement('b');
      b.textContent = fleetName(d.f);
      let text;
      if (d.f.serving === false) text = T.summary.building;
      else {
        const span = d.pinned.find(([a, z]) => z > c.now && a < c.decide + WEEK);
        if (span) {
          text = T.summary.pinned(span[0] > c.now ? span[0] : 0, runEnd(d.pinned, span));
        } else if (d.f.hourFleet) {
          let booked = 0;
          for (let h = Math.floor(c.now / HOUR) * HOUR; h < c.now + c.hourDays * DAY; h += HOUR) {
            if (within(d.booked, h) && !within(d.pinned, h)) booked += 1;
          }
          text = T.summary.hours(booked);
        } else {
          const n = nextWeek(d, c);
          text = n === 'pin' ? T.summary.pin(d.first) : n === 'wait' ? T.summary.wait : n === 'hold' ? T.summary.hold : T.summary.free;
        }
      }
      item.append(b, ': ' + text);
      return item;
    }));
  }

  function chips() {
    const st = S.state;
    const c = context(st);
    const weekly = st.fleets.filter((f) => !f.hourFleet && f.serving !== false);
    const pinned = weekly.filter((f) => within(spans(f.pinned), c.now)).length;
    const next = weekly.filter((f) => within(spans(f.pinned), c.weekEnd + HOUR)).length;
    const items = [
      weekly.length ? T.pinnedA(pinned, next, weekly.length, dayText(c.weekEnd)) : T.noneServing,
      T.locks(c.locks),
      T.pairs((st.pairs && st.pairs.open) || 0, (st.pairs && st.pairs.limit) || 0),
      T.sunday(dayText(c.decide - DAY), left(c.decide - DAY - c.now)),
    ];
    $('av-chips').replaceChildren(...items.map((text) => {
      const li = document.createElement('li');
      li.textContent = text;
      return li;
    }));
    $('av-now').textContent = T.asOf(timeText(c.now).replace(/:00$/, `:${pad(k(c.now).getUTCMinutes())}`));
  }

  function describe(event) {
    if (!S.grid) return;
    const { grid, start, step, cw, count, c } = S.grid;
    const box = $('av-board').firstChild.getBoundingClientRect();
    const x = event.clientX - box.left;
    const yy = event.clientY - box.top;
    const col = Math.floor(x / cw);
    const row = grid.find((r) => yy >= r.top && yy < r.top + ROW);
    if (!row || col < 0 || col >= count) return;
    const t = start + col * step;
    const s = row.states[col];
    const d = row.d;
    const when = step === HOUR ? `${timeText(t)}–${pad((k(t).getUTCHours() + 1) % 24)}:00` : `${dayText(t)} – ${dayText(t + WEEK - DAY)}`;
    let text;
    if (s === 'pinned') {
      const span = within(d.pinned, t + (step === HOUR ? 0 : HOUR)) || [t, t + step];
      text = T.detail.pinned(span[0], span[1], d.f.hourFleet);
    } else if (s === 'open' || s === 'later') text = T.detail[s](d.f.hourFleet);
    else if (s === 'pin') text = T.detail.pin(d.first || c.weekEnd);
    else if (s === 'hold') text = T.detail.hold(c.decide, c.locks);
    else if (s === 'wait') text = T.detail.wait(c.locks);
    else text = T.detail[s];
    $('av-detail').textContent = `${fleetName(d.f)} · ${when} KST · ${T.names[s]}. ${text}`;
  }

  async function refresh() {
    try {
      const response = await fetch(API + '/market/state', { credentials: 'same-origin' });
      if (!response.ok) throw new Error(String(response.status));
      S.state = await response.json();
      chips();
      draw();
    } catch (error) {
      $('av-now').textContent = T.failed;
    }
  }

  document.body.classList.toggle('sandbox', SANDBOX);
  $(SANDBOX ? 'av-mode-sandbox' : 'av-mode-live').setAttribute('aria-current', 'true');
  if (SANDBOX) $('av-sandbox').textContent = T.sandbox;
  $('av-now').textContent = T.loading;
  document.querySelectorAll('[data-av-view]').forEach((button) => button.addEventListener('click', () => {
    S.view = button.dataset.avView;
    document.querySelectorAll('[data-av-view]').forEach((b) => b.setAttribute('aria-pressed', String(b === button)));
    if (S.state) draw();
  }));
  $('av-board').addEventListener('pointermove', describe);
  $('av-board').addEventListener('click', describe);
  refresh();
  setInterval(refresh, 30000);
})();
