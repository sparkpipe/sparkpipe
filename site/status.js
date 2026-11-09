(() => {
  const SANDBOX = new URLSearchParams(location.search).has('sandbox');
  const API = SANDBOX ? '/pay/sandbox/api' : '/pay/api';
  const KO = document.documentElement.lang === 'ko';
  const T = KO ? {
    loading: '기록을 불러오는 중…',
    failed: '상태를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.',
    now: (when) => `${when} 기준, 30초마다 갱신`,
    status: { up: '정상 운영', down: '장애', maintenance: '점검 중', not_serving: '아직 서비스 전', unknown: '아직 기록 없음' },
    since: (when) => `${when}부터`,
    uptime: '가동률',
    windows: { '24h': '최근 24시간', '7d': '최근 7일', '30d': '최근 30일', '90d': '최근 90일' },
    noRecord: '기록 없음',
    incidents: '장애 기간 (최근 90일)',
    none: '최근 90일 동안 장애 기간이 없습니다.',
    head: ['시작 (KST)', '종료 (KST)', '분', '적립'],
    ongoing: '진행 중',
    over: (n) => `${n}분 초과: 예약된 시간마다 적립`,
    under: (n) => `${n}분 이하`,
    gateway: ' (게이트웨이 미관측)',
    maintenance: '예정된 점검',
    noMaintenance: '예정된 점검이 없습니다.',
    records: (when) => `${when}부터의 기록`,
  } : {
    loading: 'Loading our records…',
    failed: 'We could not load the status. Try again in a minute.',
    now: (when) => `As of ${when}, refreshed every 30 seconds`,
    status: { up: 'Serving', down: 'Down', maintenance: 'Maintenance', not_serving: 'Not serving yet', unknown: 'No records yet' },
    since: (when) => `since ${when}`,
    uptime: 'Uptime',
    windows: { '24h': 'Last 24 hours', '7d': 'Last 7 days', '30d': 'Last 30 days', '90d': 'Last 90 days' },
    noRecord: 'No records',
    incidents: 'Periods down (last 90 days)',
    none: 'No period down in the last 90 days.',
    head: ['From (KST)', 'To (KST)', 'Minutes', 'Credit'],
    ongoing: 'ongoing',
    over: (n) => `Over ${n} minutes: every booked hour credited`,
    under: (n) => `${n} minutes or less`,
    gateway: ' (gateway could not ask)',
    maintenance: 'Announced maintenance',
    noMaintenance: 'No maintenance announced.',
    records: (when) => `Records since ${when}`,
  };
  const $ = (id) => document.getElementById(id);
  const parse = (text) => Date.parse(text.length === 17 ? text.replace('Z', ':00Z') : text);
  const kst = (text) => (text ? new Date(parse(text) + 9 * 3600000).toISOString().slice(0, 16).replace('T', ' ') + ' KST' : '-');
  const el = (tag, text, cls) => { const node = document.createElement(tag); if (text !== undefined) node.textContent = text; if (cls) node.className = cls; return node; };
  const table = (head, rows) => {
    const wrap = el('div', undefined, 'scroll');
    const t = el('table');
    const tr = el('tr');
    head.forEach((h, i) => tr.append(el('th', h, i === 2 ? 'num' : '')));
    t.append(tr);
    rows.forEach((cells) => { const r = el('tr'); cells.forEach((c, i) => r.append(el('td', c, i === 2 ? 'num' : ''))); t.append(r); });
    wrap.append(t);
    return wrap;
  };

  const fleet = (f, outage) => {
    const box = el('article', undefined, 'status-fleet');
    const head = el('div', undefined, 'status-head');
    const word = el('span', T.status[f.status] || f.status, 'status-word');
    word.dataset.status = f.status;
    head.append(el('h3', f.name), word);
    if (f.since) head.append(el('span', T.since(kst(f.since)), 'effective'));
    box.append(head);
    if (f.recordsSince) box.append(el('p', T.records(kst(f.recordsSince)), 'effective'));
    box.append(table([T.uptime, ''], Object.keys(T.windows).map((w) => [T.windows[w], f.uptime && f.uptime[w] !== null && f.uptime[w] !== undefined ? f.uptime[w].toFixed(3) + '%' : T.noRecord])));
    box.append(el('p', T.incidents));
    box.append(f.incidents.length ? table(T.head, f.incidents.map((i) => [kst(i.start), i.end ? kst(i.end) : T.ongoing, String(i.minutes), (i.creditable ? T.over(outage) : T.under(outage)) + (i.cause === 'gateway' ? T.gateway : '')])) : el('p', T.none));
    box.append(el('p', T.maintenance));
    box.append(f.maintenance.length ? table(T.head.slice(0, 2), f.maintenance.map(([a, b]) => [kst(a), kst(b)])) : el('p', T.noMaintenance));
    return box;
  };

  const load = async () => {
    try {
      const response = await fetch(API + '/status', { credentials: 'omit' });
      if (!response.ok) throw new Error(String(response.status));
      const data = await response.json();
      $('status-now').textContent = T.now(kst(data.now));
      document.querySelectorAll('[data-outage]').forEach((node) => { node.textContent = String(data.outageMinutes); });
      $('status-fleets').replaceChildren(...data.fleets.map((f) => fleet(f, data.outageMinutes)));
    } catch (error) {
      $('status-now').textContent = T.failed;
    }
  };
  $('status-now').textContent = T.loading;
  load();
  setInterval(load, 30000);
})();
