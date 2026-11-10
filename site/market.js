(() => {
  const SANDBOX = new URLSearchParams(location.search).has('sandbox');
  const API = SANDBOX ? '/pay/sandbox/api' : '/pay/api';
  const HOUR = 3600000;
  const KST = 9 * HOUR;
  const P = window.SP_PRICING;
  const $ = (id) => document.getElementById(id);
  const S = { state: null, codeSent: false, quotes: {} };
  document.body.classList.toggle('sandbox', SANDBOX);
  $(SANDBOX ? 'mode-sandbox' : 'mode-live').setAttribute('aria-current', 'true');
  $('api-base').textContent = location.origin + API + '/v1';

  const call = async (method, path, body) => {
    const response = await fetch(API + path, {
      method,
      credentials: 'same-origin',
      headers: body === undefined ? {} : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const error = new Error(data.error || 'The service did not answer (' + response.status + ').');
      error.reason = data.reason;
      throw error;
    }
    return data;
  };

  const spend = async (path, body, method = 'POST') => {
    try {
      return await call(method, path, body);
    } catch (error) {
      if (error.reason !== 'terms') throw error;
      if (!confirm('The SparkPipe Customer Agreement or its Data Processing Agreement has changed since you accepted them. Accept the current texts (the Korean texts prevail) and continue?')) throw error;
      return call(method, path, Object.assign({}, body, { accept: true }));
    }
  };

  const say = (id, text, kind) => {
    const el = $(id);
    el.textContent = text || '';
    el.className = 'flash' + (kind ? ' ' + kind : '');
    el.hidden = !text;
  };

  const parseUtc = (text) => Date.parse(text.length === 17 ? text.replace('Z', ':00Z') : text);
  const kst = (text) => (text ? new Date(parseUtc(text) + KST).toISOString().slice(0, 16).replace('T', ' ') : '-');
  const kstDay = (text) => (text ? new Date(parseUtc(text) + KST).toISOString().slice(0, 10) : '-');
  const slotFromLocal = (value) => (value ? new Date(Date.parse(value.slice(0, 13) + ':00:00Z') - KST).toISOString().slice(0, 13) + ':00Z' : '');
  const localFromMs = (ms) => new Date(ms + KST).toISOString().slice(0, 13) + ':00';
  const lastDay = (text) => (text ? new Date(parseUtc(text) - 1 + KST).toISOString().slice(0, 10) : '-');
  const hc = (text) => {
    const n = Number(text);
    return Number.isFinite(n) ? n.toLocaleString('en-US', { maximumFractionDigits: 6 }) : '-';
  };
  const cell = (tag, text, cls) => {
    const el = document.createElement(tag);
    el.textContent = text;
    if (cls) el.className = cls;
    return el;
  };
  const row = (values) => {
    const tr = document.createElement('tr');
    values.forEach((v) => tr.append(v instanceof Node ? (() => { const td = document.createElement('td'); td.append(v); return td; })() : cell('td', v)));
    return tr;
  };
  const button = (text, onClick, cls = 'btn small') => {
    const b = cell('button', text, cls);
    b.type = 'button';
    b.addEventListener('click', onClick);
    return b;
  };
  const fill = (id, rows, empty) => {
    const body = $(id);
    body.replaceChildren(...rows);
    if (!rows.length) {
      const tr = document.createElement('tr');
      const td = cell('td', empty, 'muted');
      td.colSpan = 8;
      tr.append(td);
      body.append(tr);
    }
  };
  const fillSelect = (id, options) => {
    const el = $(id);
    const before = el.value;
    el.replaceChildren(...options.map(([value, label]) => { const o = cell('option', label); o.value = value; return o; }));
    if (options.some(([v]) => v === before)) el.value = before;
  };
  const nowMs = () => (S.state ? parseUtc(S.state.now) : Date.now());
  const pct = (x) => Math.round(Number(x) * 100) + '%';

  const busy = new Set();
  const act = async (flashId, work, done) => {
    if (busy.has(flashId)) return;
    busy.add(flashId);
    const form = $(flashId).closest('form');
    const buttons = form ? Array.from(form.querySelectorAll('button')) : [];
    buttons.forEach((b) => { b.disabled = true; });
    say(flashId, 'Working…');
    try {
      const result = await work();
      say(flashId, done ? done(result) : '', 'ok');
      await refresh();
    } catch (error) {
      say(flashId, error.message, 'error');
    } finally {
      buttons.forEach((b) => { b.disabled = false; });
      busy.delete(flashId);
    }
  };

  const statusText = { 'hour-fleet': 'Hour Fleet', pinned: 'Pinned', free: 'Available' };
  const waitingText = { terms: 'waiting for you to accept the current Customer Agreement and Data Processing Agreement', insufficient_hc: 'waiting for enough unlocked HC to book the next week' };

  const setAutoExtend = (fleet, weeks) => act('m-weeks', () => spend('/market/auto-extend', { fleet, weeks_ahead: weeks }, 'PUT'),
    (r) => (r.weeks_ahead ? `Auto-extend on ${fleet}: ${r.weeks_ahead} week${r.weeks_ahead === 1 ? '' : 's'} kept booked ahead${r.through ? `, booked through ${lastDay(r.through)}` : ''}.` : `Auto-extend on ${fleet} is off. Book the next week yourself to keep the fleet.`));

  function autoExtendControl(own) {
    const box = document.createElement('div');
    box.className = 'auto-extend';
    const label = cell('span', '');
    if (own.autoExtend) {
      label.textContent = `Auto-extend on: ${own.autoExtend} week${own.autoExtend === 1 ? '' : 's'} kept booked ahead` + (own.autoExtendWaiting ? `, ${waitingText[own.autoExtendWaiting] || own.autoExtendWaiting}. ` : '. ');
      const ahead = document.createElement('select');
      ahead.setAttribute('aria-label', 'Weeks kept booked ahead on ' + own.fleet);
      for (let n = 1; n <= P.bookings.maxWeeksAhead; n += 1) {
        const o = cell('option', `${n} week${n === 1 ? '' : 's'} ahead`);
        o.value = String(n);
        ahead.append(o);
      }
      ahead.value = String(own.autoExtend);
      ahead.addEventListener('change', () => setAutoExtend(own.fleet, Number(ahead.value)));
      box.append(label, ahead, ' ', button('Turn auto-extend off', () => setAutoExtend(own.fleet, 0)));
    } else {
      label.textContent = 'Auto-extend off: book the next week yourself by the deadline. ';
      box.append(label, button('Turn auto-extend on', () => setAutoExtend(own.fleet, 1), 'btn small solid'));
    }
    return box;
  }

  function renderTop() {
    const st = S.state;
    const me = st.account;
    $('now').textContent = kst(st.now) + ' KST';
    const pinned = (me && me.pinned) || [];
    const mine = (id) => pinned.find((p) => p.fleet === id);
    fill('fleets', st.fleets.map((f) => {
      const own = mine(f.id);
      const yours = own ? `Booked through ${lastDay(own.through)} (${own.weeks} week${own.weeks === 1 ? '' : 's'}); book the next week by ${kst(own.deadline)} KST` : '';
      const open = !own && f.firstWeek ? `available from ${kstDay(f.firstWeek)}` : '';
      const status = own ? 'Pinned to you' : [statusText[f.status] || f.status, open].filter(Boolean).join(', ');
      const detail = own ? (() => { const box = document.createElement('div'); box.append(cell('p', yours), autoExtendControl(own)); return box; })() : yours;
      return row([f.name, (f.kind || '-') + (f.pair ? ` · pairs with ${(st.fleets.find((x) => x.id === f.pair) || { name: f.pair }).name}` : ''), String(f.rate), f.serving === false ? 'Not serving yet' : status, detail]);
    }), 'No fleet is listed yet.');
    const nameOf = (id) => (st.fleets.find((f) => f.id === id) || { name: id }).name;
    const pairable = (f) => !mine(f.id) && f.hourFleet && f.pair && mine(f.pair) && st.pairs && st.pairs.open > 0;
    fillSelect('weeks-fleet', st.fleets.filter((f) => (mine(f.id) && mine(f.id).canExtend) || (!mine(f.id) && f.firstWeek) || pairable(f))
      .map((f) => [f.id, f.name + (mine(f.id) ? ' (extend)' : pairable(f) ? ' (pin as the pair of ' + nameOf(f.pair) + ')' : ' (pin from ' + kstDay(f.firstWeek) + ')')]));
    $('pairs-open').textContent = st.pairs && st.pairs.limit ? `Pairs open to pin for next week: ${st.pairs.open} of ${st.pairs.limit}.` : '';
    fillSelect('hours-fleet', st.fleets.filter((f) => f.serving !== false && (f.hourFleet || f.status === 'free')).map((f) => [f.id, f.name + (f.hourFleet ? ' (Hour Fleet)' : ' (available this week)')]));
    fillSelect('try-model', (st.models || P.models).map((m) => [m.id, m.name]));
    fill('discounts', P.bookings.discounts.map((d) => row([`${d.weeks} or more`, pct(d.off)])), '');
    const waiting = st.waitlist && Number(st.waitlist.locks);
    $('waitlist-count').textContent = waiting ? `${waiting} lock${waiting === 1 ? '' : 's'} waiting.` : 'Nobody is waiting.';
    renderBuy();
  }

  function renderBuy() {
    const count = Math.max(0, Math.floor(Number($('buy-hc').value) || 0));
    const tier = P.hc.prices.filter((t) => count >= t.from).pop() || P.hc.prices[0];
    $('buy-sum').textContent = count ? `${hc(count)} HC × $${tier.usd} = $${(count * tier.usd).toFixed(2)} plus $${(count * tier.usd * P.vatRate).toFixed(2)} VAT, or ₩${(count * tier.krw).toLocaleString('en-US')} plus VAT in won` : '';
  }

  function renderAccount() {
    const me = S.state.account;
    $('signin').hidden = !!me;
    $('me').hidden = !me;
    $('signout').hidden = !me;
    if (!me) {
      ['lots', 'bookings', 'keys'].forEach((id) => fill(id, [], 'Sign in to see yours.'));
      $('waitlist-status').textContent = 'Sign in to join the waitlist.';
      $('hour-discount').textContent = '';
      return;
    }
    $('me-company').textContent = me.company;
    $('me-email').textContent = me.email;
    $('balance').textContent = hc(me.balance);
    $('locked').textContent = hc(me.locked || 0) + ' HC';
    fill('lots', (me.lots || []).map((l, i) => row([String(i + 1), hc(l.hc), kstDay(l.expires)])), 'No HC yet.');
    fill('bookings', (me.bookings || []).map((b) => row([kst(b.start), kst(b.end), b.fleet, b.kind === 'week' ? `${b.weeks} week${b.weeks === 1 ? '' : 's'}` : `${b.hours} hour${b.hours === 1 ? '' : 's'}`, hc(b.hc)])), 'No bookings yet.');
    $('waitlist-status').textContent = me.position ? `On the waitlist: ${hc(me.locked)} HC locked, position ${me.position}.` : 'Not on the waitlist.';
    const first = P.bookings.discounts[0];
    $('hour-discount').textContent = Number(me.discount.discount) ? `Your discount on hours: ${pct(me.discount.discount)}, for ${discountFor(me.discount)}.`
      : `No discount on hours yet: hold ${first.weeks} weeks on a fleet, or wait on the waitlist with a lock that books them, for ${pct(first.off)} off.`;
    fill('keys', (me.keys || []).filter((k) => !k.revoked).map((k) => row(['…' + k.hint, k.note || '', kstDay(k.created), button('Revoke', () => act('m-try', () => call('POST', '/market/keys/revoke', { id: k.key }), () => 'Key revoked.'))])), 'No keys yet.');
  }

  function discountFor(d) {
    return d.basis === 'waitlist' ? `the ${d.weeks} weeks your waitlist lock would book` : `${d.weeks} week${d.weeks === 1 ? '' : 's'} held${d.fleet ? ' on ' + d.fleet : ''}`;
  }

  async function quoteInto(id, path, body, text) {
    const key = JSON.stringify(body);
    S.quotes[id] = key;
    if (!S.state || !S.state.account || !body) { $(id).textContent = ''; return; }
    try {
      const q = await call('POST', path, body);
      if (S.quotes[id] === key) $(id).textContent = text(q);
    } catch (error) {
      if (S.quotes[id] === key) $(id).textContent = error.message;
    }
  }

  function quoteWeeks() {
    const fleet = $('weeks-fleet').value;
    const weeks = Math.floor(Number($('weeks-count').value) || 0);
    S.weeksFirst = null;
    quoteInto('weeks-quote', '/market/quote-weeks', fleet && weeks >= 1 ? { fleet, weeks } : null,
      (q) => (S.weeksFirst = { fleet, weeks, first: q.first }) && `${weeks} week${weeks === 1 ? '' : 's'} from ${kstDay(q.first)} to ${lastDay(q.end)}: ${hc(q.price)} HC${Number(q.discount) ? `, ${pct(q.discount)} off with ${q.count} weeks held` : ''}.`);
  }

  function quoteHours() {
    const fleet = $('hours-fleet').value;
    const start = $('hours-start').value ? slotFromLocal($('hours-start').value) : '';
    const hours = Math.floor(Number($('hours-count').value) || 0);
    quoteInto('hours-quote', '/market/quote-hours', fleet && start && hours >= 1 ? { fleet, start, hours } : null,
      (q) => `${q.hours} hour${q.hours === 1 ? '' : 's'} on ${q.fleet} from ${kst(q.start)} KST: ${hc(q.price)} HC${Number(q.discount) ? `, ${pct(q.discount)} off for ${discountFor({ basis: q.basis, weeks: q.weeks })}` : ''}.`
        + (Number(q.waitlist_saving) ? ` If you leave the waitlist before it gives you a fleet, the ${hc(q.waitlist_saving)} HC saved is repaid from your lock.` : ''));
  }

  async function refresh() {
    S.state = await call('GET', '/market/state');
    renderTop();
    renderAccount();
    quoteWeeks();
    quoteHours();
  }

  $('signin').addEventListener('submit', (e) => {
    e.preventDefault();
    const email = $('email').value.trim();
    act('m-signin', async () => {
      const body = { email, accept: $('accept').checked };
      if ($('company').value.trim()) body.company = $('company').value.trim();
      if (S.codeSent) {
        body.code = $('code').value.trim();
      } else {
        const sent = await call('POST', '/code', { email });
        if (!sent.code) {
          S.codeSent = true;
          $('code-row').hidden = false;
          $('signin-go').textContent = 'Sign in with the code';
          $('code').focus();
          return `We emailed a 6-digit code to ${email}.`;
        }
        body.code = sent.code;
      }
      try {
        const done = await call('POST', '/verify', body);
        S.codeSent = false;
        $('code-row').hidden = true;
        $('code').value = '';
        $('signin-go').textContent = 'Sign in';
        return done;
      } catch (error) {
        if (!SANDBOX && /expired|new one/.test(error.message)) {
          S.codeSent = false;
          $('code-row').hidden = true;
          $('signin-go').textContent = 'Sign in';
        }
        throw error;
      }
    }, (r) => (typeof r === 'string' ? r : ''));
  });
  $('signout').addEventListener('click', () => act('m-signin', () => call('POST', '/signout', {})));
  $('buy-hc').addEventListener('input', renderBuy);
  $('buy').addEventListener('submit', (e) => {
    e.preventDefault();
    const count = Math.floor(Number($('buy-hc').value) || 0);
    if (!SANDBOX) {
      location.href = 'buy.html?hc=' + encodeURIComponent(count);
      return;
    }
    act('m-buy', () => spend('/market/buy', { hc: String(count) }), (r) => `Added ${hc(r.hc)} HC.`);
  });
  $('weeks-fleet').addEventListener('change', quoteWeeks);
  $('weeks-count').addEventListener('input', quoteWeeks);
  $('hours-fleet').addEventListener('change', quoteHours);
  $('hours-start').addEventListener('input', quoteHours);
  $('hours-count').addEventListener('input', quoteHours);
  $('weeks').addEventListener('submit', (e) => {
    e.preventDefault();
    const fleet = $('weeks-fleet').value;
    const weeks = Math.floor(Number($('weeks-count').value));
    const quoted = S.weeksFirst && S.weeksFirst.fleet === fleet && S.weeksFirst.weeks === weeks ? { first: S.weeksFirst.first } : {};
    act('m-weeks', () => spend('/market/book-weeks', Object.assign({ fleet, weeks }, quoted)),
      (r) => `Booked ${r.weeks} week${r.weeks === 1 ? '' : 's'} on ${r.fleet}, ${kstDay(r.first)} to ${lastDay(r.end)}, for ${hc(r.price)} HC.`);
  });
  $('hours-form').addEventListener('submit', (e) => {
    e.preventDefault();
    act('m-hours', () => spend('/market/book-hours', { fleet: $('hours-fleet').value, start: slotFromLocal($('hours-start').value), hours: Math.floor(Number($('hours-count').value)) }),
      (r) => `Booked ${r.hours} hour${r.hours === 1 ? '' : 's'} on ${r.fleet} from ${kst(r.start)} KST, for ${hc(r.hc)} HC.`);
  });
  $('waitlist').addEventListener('submit', (e) => {
    e.preventDefault();
    act('m-waitlist', () => spend('/market/waitlist', { lock: String(Math.floor(Number($('lock-hc').value) || 0)) }), (r) => `${hc(r.locked)} HC locked. Your position: ${r.position}.`);
  });
  $('leave').addEventListener('click', () => {
    const owed = Number((S.state && S.state.account && S.state.account.waitlistRepay) || 0);
    if (owed && !confirm(`Leaving now repays ${hc(owed)} HC, what the waitlist discount saved on hours you booked while waiting, from your lock. Leave the waitlist?`)) return;
    act('m-waitlist', () => call('POST', '/market/waitlist/leave', {}),
      (r) => (Number(r.repaid) ? `You left the waitlist: ${hc(r.repaid)} HC of waitlist discount repaid, the rest of your HC unlocked.` : 'You left the waitlist, and your HC is unlocked.'));
  });
  document.querySelectorAll('[data-clock]').forEach((b) => b.addEventListener('click', () => act('m-clock', () => call('POST', '/market/sandbox/clock', { hours: Number(b.dataset.clock) }), (r) => `Service time is now ${kst(r.now)} KST.`)));
  $('reset').addEventListener('click', () => {
    if (!confirm('Reset the sandbox for everyone? All sandbox accounts, HC and bookings are cleared.')) return;
    act('m-clock', () => call('POST', '/market/sandbox/reset', {}), () => 'The sandbox is empty again. Sign in to start over.');
  });
  $('key-create').addEventListener('click', () => act('m-try', async () => {
    const made = await call('POST', '/market/keys', { note: $('key-note').value });
    $('try-key').value = made.key;
    $('key-once').textContent = made.key + '  (shown once: copy it now)';
    $('key-once').hidden = false;
    return made;
  }, () => 'Key created and filled in below.'));
  $('try').addEventListener('submit', async (e) => {
    e.preventDefault();
    const reply = $('try-reply');
    reply.textContent = '';
    say('m-try', 'Sending…');
    const stream = $('try-stream').checked;
    try {
      const response = await fetch(API + '/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + $('try-key').value.trim() },
        body: JSON.stringify({ model: $('try-model').value, stream, max_tokens: 64, messages: [{ role: 'user', content: $('try-prompt').value }] }),
      });
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error((data.error && (data.error.message || data.error)) || 'The fleet answered ' + response.status);
      }
      let meta = null;
      if (stream) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const parts = buffer.split('\n\n');
          buffer = parts.pop();
          parts.forEach((part) => {
            const line = part.replace(/^data: /, '');
            if (line === '[DONE]') return;
            const chunk = JSON.parse(line);
            const delta = chunk.choices && chunk.choices[0] && chunk.choices[0].delta;
            if (delta && delta.content) reply.textContent += delta.content;
            if (chunk.sparkpipe) meta = chunk.sparkpipe;
          });
        }
      } else {
        const data = await response.json();
        reply.textContent = data.choices[0].message.content;
        meta = data.sparkpipe;
      }
      say('m-try', `Answered on ${meta && meta.fleet ? meta.fleet : 'your fleet'}.`, 'ok');
      await refresh();
    } catch (error) {
      say('m-try', error.message, 'error');
    }
  });
  $('ledger-load').addEventListener('click', async () => {
    try {
      const data = await call('GET', '/market/ledger');
      $('ledger-head').textContent = `${data.count} lines · head ${data.head.slice(0, 16)}…${data.complete === false ? ' · partial' : ''}`;
      $('ledger-partial').hidden = data.complete !== false;
      fill('ledger', data.lines.slice().reverse().map((l) => row([String(l.seq), kst(l.at), l.type, JSON.stringify(l.data).slice(0, 120), l.hash.slice(0, 12) + '…'])), 'No lines yet.');
    } catch (error) {
      $('ledger-head').textContent = error.message;
    }
  });

  const tabs = [...document.querySelectorAll('#tabs [data-tab]')];
  const show = (name) => {
    tabs.forEach((t) => t.setAttribute('aria-selected', String(t.dataset.tab === name)));
    document.querySelectorAll('[data-panel]').forEach((p) => { p.hidden = p.dataset.panel !== name; });
    try { localStorage.setItem('sp-tab', name); } catch (error) {}
  };
  tabs.forEach((t) => t.addEventListener('click', () => { show(t.dataset.tab); history.replaceState(null, '', location.search + '#' + t.dataset.tab); }));
  let saved = null;
  try { saved = localStorage.getItem('sp-tab'); } catch (error) {}
  const wanted = location.hash.slice(1) || saved;
  show(tabs.some((t) => t.dataset.tab === wanted) ? wanted : 'fleets');

  refresh().then(() => {
    $('hours-start').value = localFromMs(Math.floor(nowMs() / HOUR) * HOUR + HOUR);
    quoteHours();
  }).catch((error) => say('m-clock', error.message, 'error'));
  setInterval(() => { refresh().catch(() => {}); }, 20000);
})();
