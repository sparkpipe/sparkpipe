(() => {
  const SANDBOX = new URLSearchParams(location.search).has('sandbox');
  const API = SANDBOX ? '/pay/sandbox/api' : '/pay/api';
  const HOUR = 3600000;
  const KST = 9 * HOUR;
  const $ = (id) => document.getElementById(id);
  const S = { state: null, cal: null, book: null, pick: null, days: 7, lastKey: '', codeSent: false };
  document.body.classList.toggle('sandbox', SANDBOX);
  $(SANDBOX ? 'mode-sandbox' : 'mode-live').setAttribute('aria-current', 'true');
  $('pool-base').textContent = location.origin + API + '/pool/v1';
  const first = new Date(window.SP_PRICING.hourlyEarliestStart + 'T00:00:00Z');
  document.querySelectorAll('[data-start]').forEach((el) => { el.textContent = first.toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric', timeZone: 'UTC' }); });

  const call = async (method, path, body) => {
    const response = await fetch(API + path, {
      method,
      credentials: 'same-origin',
      headers: body === undefined ? {} : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || 'The service did not answer (' + response.status + ').');
    return data;
  };

  const say = (id, text, kind) => {
    const el = $(id);
    el.textContent = text || '';
    el.className = 'flash' + (kind ? ' ' + kind : '');
    el.hidden = !text;
  };

  const parseUtc = (text) => Date.parse(text.length === 17 ? text.replace('Z', ':00Z') : text);
  const kst = (text) => {
    const d = new Date(parseUtc(text) + KST);
    return d.toISOString().slice(5, 16).replace('T', ' ');
  };
  const kstDate = (text) => new Date(parseUtc(text) + KST).toISOString().slice(0, 10);
  const kstDay = (ms) => new Date(ms + KST).toISOString().slice(0, 10);
  const kstHour = (ms) => new Date(ms + KST).getUTCHours();
  const slotFromLocal = (value) => {
    if (!value) return '';
    const ms = Date.parse(value.slice(0, 13) + ':00:00Z') - KST;
    return new Date(ms).toISOString().slice(0, 13) + ':00Z';
  };
  const localFromMs = (ms) => new Date(ms + KST).toISOString().slice(0, 13) + ':00';
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
  const nowMs = () => (S.state ? parseUtc(S.state.now) : Date.now());

  const act = async (flashId, work, done) => {
    say(flashId, 'Working…');
    try {
      const result = await work();
      say(flashId, done ? done(result) : '', 'ok');
      await refresh();
    } catch (error) {
      say(flashId, error.message, 'error');
    }
  };

  const models = () => (S.state ? S.state.pool.models : []);
  const fillSelect = (id, options, keep) => {
    const el = $(id);
    const before = el.value;
    el.replaceChildren(...options.map(([value, label]) => { const o = cell('option', label); o.value = value; return o; }));
    if (keep && options.some(([v]) => v === before)) el.value = before;
  };

  function renderTop() {
    const st = S.state;
    $('now').textContent = kst(st.now) + ' KST';
    fillSelect('pick-model', models().map((m) => [m.id, m.name]), true);
    fillSelect('bid-model', models().map((m) => [m.id, m.name]), true);
    fillSelect('try-model', models().map((m) => [m.id, m.name]), true);
    fillSelect('bid-fleet', [['', 'Any fleet']].concat(st.fleets.map((f) => [f.id, f.name])), true);
    const price = st.market.hcPrice;
    const amount = Number($('amount').value) || 0;
    $('buy-sum').textContent = amount ? `${hc(amount / price)} HC for $${amount.toFixed(2)} plus $${(amount * st.vatRate).toFixed(2)} VAT, lasting ${st.market.validityMonths} months` : '';
    $('on-sale').textContent = Number(st.onSale) > 0 ? `On sale now: up to ${hc(st.onSale)} HC. We sell only the HC our free fleet-hours in the next 12 months can deliver.` : 'Sold out for now: every free fleet-hour in the next 12 months already backs HC sold.';
    const pool = st.pool;
    $('pool-status').textContent = `Price ${pool.hcPerSeatSecond} HC per seat-second ($${pool.usdPerSeatHour} per seat-hour${pool.usdPerMillionTokens ? `, about $${pool.usdPerMillionTokens} per million tokens` : ''}) · ${pool.active} of ${pool.seats} seats busy · pooled fleets: ${pool.fleets.join(', ') || 'none this hour'}`;
    $('feed-market').href = API + '/market/market.txt';
    $('feed-pool').href = API + '/market/pool.txt';
  }

  function renderAccount() {
    const me = S.state.account;
    $('signin').hidden = !!me;
    $('me').hidden = !me;
    $('signout').hidden = !me;
    if (!me) {
      ['lots', 'hours', 'bids', 'keys', 'charges', 'tranches'].forEach((id) => fill(id, [], 'Sign in first.'));
      return;
    }
    $('me-company').textContent = me.company;
    $('me-email').textContent = me.email;
    $('balance').textContent = hc(me.balance);
    $('held').textContent = hc(me.held) + ' HC';
    fill('lots', me.lots.filter((l) => Number(l.hc) > 0 || !l.expired).map((l) => row([hc(l.hc), hc(l.first), l.source, (l.expired ? 'lapsed ' : '') + kstDate(l.expires)])), 'No HC yet. Buy some to start.');
    fill('tranches', me.tranche.map((t) => row([t.month, String(t.granted), String(t.used), String(t.left)])), 'No Tranche hours.');
    $('tranche-box').hidden = !SANDBOX && !me.tranche.length;
    fill('hours', me.reservations.map(hourRow), 'No reservations. Pick an hour on the calendar.');
    fill('bids', me.bids.map((b) => row([`${kst(b.from)} → ${kst(b.to)}${b.standing ? ' weekly' : ''}${b.consecutive ? ', consecutive' : ''}`, b.fleet || 'any', `${b.filled} / ${b.hours}`, hc(b.max), hc(b.held), b.state, button('Cancel', () => act('m-bid', () => call('POST', '/market/cancel', { bid: b.bid }), () => 'Bid cancelled; its held HC is back.'))])), 'No open bids.');
    fill('keys', me.keys.map((k) => row([`…${k.hint}`, k.note || '', kst(k.created), k.revoked ? 'revoked' : button('Revoke', () => act('m-try', () => call('POST', '/market/keys/revoke', { id: k.key }), () => 'Key revoked.'))])), 'No keys yet.');
    fill('charges', me.charges.map((c) => row([kst(c.now), c.model, c.seconds, `${c.tokensIn} / ${c.tokensOut}`, c.hc])), 'No pool requests yet.');
  }

  function hourRow(r) {
    const paid = r.source === 'tranche' ? 'Tranche hour' : `${hc(r.price)} HC${r.source === 'release' ? ', released hour' : ''}`;
    const rel = r.release;
    let state = r.state;
    let action = '';
    if (r.state === 'held') {
      const wrap = document.createElement('span');
      const min = document.createElement('input');
      min.type = 'number'; min.step = '0.01'; min.min = '0.000001'; min.placeholder = 'list'; min.title = 'Lowest price in HC; empty means list';
      wrap.append(min, ' ', button('Release', () => act('m-hours', () => call('POST', '/market/release', { reservation: r.reservation, min: min.value || null }), (x) => x.state === 'sold' ? 'Released and sold at once to a waiting bid.' : 'Released. It sells before our own hours.')));
      action = wrap;
    } else if (r.state === 'released') {
      state = `released at ${hc(rel.min)} HC`;
      action = button('Cancel release', () => act('m-hours', () => call('POST', '/market/cancel', { ask: rel.ask }), () => 'Release cancelled; the hour is yours again.'));
    } else if (r.state === 'sold') {
      state = `sold at ${hc(rel.price)} HC`;
    } else if (r.state === 'pooled') {
      state = rel.earned === null ? 'in the shared pool' : `pool earned ${hc(rel.earned)} HC`;
    } else if (r.state === 'live') {
      state = 'running now: your keys use it free';
    }
    return row([kst(r.start), r.fleet, r.model, paid, state, action]);
  }

  function renderCalendar() {
    const cal = S.cal;
    const box = $('calendar');
    box.replaceChildren();
    const first = parseUtc(cal.from);
    cal.fleets.forEach((fleet) => {
      const wrap = cell('div', '', 'cal-fleet');
      wrap.append(cell('h3', fleet.name));
      const grid = cell('div', '', 'cal-grid');
      grid.append(cell('span', ''));
      for (let h = 0; h < 24; h++) grid.append(cell('span', h % 3 === 0 ? String(h).padStart(2, '0') : '', 'h'));
      const days = new Map();
      fleet.cells.forEach(([kind, price], n) => {
        const ms = first + n * HOUR;
        const day = kstDay(ms);
        if (!days.has(day)) days.set(day, new Array(24).fill(null));
        days.get(day)[kstHour(ms)] = { kind, price, ms };
      });
      days.forEach((hours, day) => {
        grid.append(cell('span', day.slice(5), 'd'));
        hours.forEach((slot) => {
          const b = cell('button', '', 'cell ' + (slot ? slot.kind : 'blank'));
          b.type = 'button';
          if (slot) {
            const start = new Date(slot.ms).toISOString().slice(0, 13) + ':00Z';
            if (slot.kind === 'release') b.style.setProperty('--depth', String(Math.max(0.25, Math.min(1, 1.25 - Number(slot.price)))));
            b.title = `${fleet.name} · ${kst(start)} KST · ${slot.kind}${slot.price ? ' · ' + hc(slot.price) + ' HC' : ''}`;
            if (S.pick && S.pick.fleet === fleet.id && S.pick.start === start) b.classList.add('picked');
            if (['hyor', 'release', 'mine', 'mine-release', 'closed'].includes(slot.kind)) b.addEventListener('click', () => pick(fleet, start, slot));
          } else {
            b.disabled = true;
          }
          grid.append(b);
        });
      });
      wrap.append(grid);
      box.append(wrap);
    });
  }

  function pick(fleet, start, slot) {
    S.pick = { fleet: fleet.id, start, kind: slot.kind, price: slot.price };
    $('pick').hidden = false;
    $('pick-title').textContent = `${fleet.name}, ${kst(start)} KST`;
    const notes = {
      hyor: `Our hour at ${hc(slot.price)} HC. Tranche hours can pay for it too.`,
      release: `A released hour at ${hc(slot.price)} HC: the seller gets the price less the fee.`,
      mine: 'This hour is yours. Release it from Your hours.',
      'mine-release': 'You released this hour. Cancel the release from Your hours to keep it.',
      closed: 'Not for sale now: beyond 31 days, kept for plan holders until 6 hours before it starts, on a fleet locked to a plan, or kept for HC that lapses sooner. Tranche hours can reserve any hour in their month that no other plan keeps.',
    };
    $('pick-note').textContent = notes[slot.kind];
    $('reserve').disabled = slot.kind === 'mine' || slot.kind === 'mine-release';
    say('m-pick', '');
    renderCalendar();
  }

  function renderBook() {
    const book = S.book;
    fill('asks', book.asks.map((a) => row([kst(a.start), a.fleet, hc(a.min) + ' HC'])), 'No released hours for sale.');
    fill('openbids', book.bids.map((b) => row([`${kst(b.from)} → ${kst(b.to)}${b.standing ? ' weekly' : ''}`, b.fleet || 'any', String(b.left), hc(b.max) + ' HC'])), 'No open bids.');
    fill('trades', book.trades.map((t) => row([kst(t.start), t.fleet, hc(t.price) + ' HC', { hyor: 'our capacity', release: 'released hour', tranche: 'Tranche hour' }[t.kind] || t.kind])), 'No trades yet.');
  }

  async function refresh() {
    const [state, cal, book] = await Promise.all([call('GET', '/market/state'), call('GET', '/market/calendar?days=' + S.days), call('GET', '/market/book')]);
    Object.assign(S, { state, cal, book });
    renderTop();
    renderAccount();
    renderCalendar();
    renderBook();
  }

  $('signin').addEventListener('submit', (e) => {
    e.preventDefault();
    act('m-signin', async () => {
      const email = $('email').value.trim();
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
  $('amount').addEventListener('input', renderTop);
  $('buy').addEventListener('submit', (e) => {
    e.preventDefault();
    if (!SANDBOX) {
      location.href = 'buy.html?amount=' + encodeURIComponent($('amount').value);
      return;
    }
    act('m-buy', () => call('POST', '/market/buy', { amount: $('amount').value }), (r) => `Added ${hc(r.hc)} HC.`);
  });

  $('tranche').addEventListener('click', () => act('m-tranche', () => call('POST', '/market/sandbox/tranche', {}), (r) => `${r.hours} Tranche hours for each of ${r.months.join(' and ')}.`));
  $('days').addEventListener('change', () => { S.days = Number($('days').value); refresh().catch((error) => say('m-pick', error.message, 'error')); });
  $('reserve').addEventListener('click', () => {
    const p = S.pick;
    if (!p) return;
    const pay = $('pick-pay').value;
    act('m-pick', () => call('POST', '/market/reserve', { fleet: p.fleet, start: p.start, model: $('pick-model').value, pay, max: pay === 'hc' ? p.price : undefined }), (r) => `Reserved for ${pay === 'tranche' ? 'one Tranche hour' : hc(r.price) + ' HC'}.`);
  });
  document.querySelectorAll('[data-clock]').forEach((b) => b.addEventListener('click', () => act('m-clock', () => call('POST', '/market/sandbox/clock', { hours: Number(b.dataset.clock) }), (r) => `Market time is now ${kst(r.now)} KST.`)));
  $('reset').addEventListener('click', () => {
    if (!confirm('Reset the sandbox for everyone? All sandbox accounts, HC and orders are cleared.')) return;
    act('m-clock', () => call('POST', '/market/sandbox/reset', {}), () => 'The sandbox is empty again. Sign in to start over.');
  });
  $('bid').addEventListener('submit', (e) => {
    e.preventDefault();
    act('m-bid', () => call('POST', '/market/bid', {
      fleet: $('bid-fleet').value || null, model: $('bid-model').value,
      from: slotFromLocal($('bid-from').value), to: slotFromLocal($('bid-to').value),
      hours: Number($('bid-hours').value), max: $('bid-max').value,
      consecutive: $('bid-consecutive').checked, standing: $('bid-standing').checked,
    }), (r) => r.state === 'filled' ? 'Filled at once.' : `Bid placed; ${r.filled} hours filled so far.`);
  });
  const nextKstDay = (weekday) => {
    const base = nowMs() + KST;
    const today = new Date(base).getUTCDay();
    const ahead = ((weekday - today + 7) % 7) || 7;
    return Date.UTC(new Date(base).getUTCFullYear(), new Date(base).getUTCMonth(), new Date(base).getUTCDate() + ahead) - KST;
  };
  $('preset-saturday').addEventListener('click', () => {
    const day = nextKstDay(6);
    $('bid-from').value = localFromMs(day + 10 * HOUR);
    $('bid-to').value = localFromMs(day + 18 * HOUR);
    $('bid-hours').value = 8;
    $('bid-consecutive').checked = true;
    $('bid-standing').checked = false;
  });
  $('preset-weekdays').addEventListener('click', () => act('m-bid', async () => {
    let placed = 0;
    for (let weekday = 1; weekday <= 5; weekday++) {
      const day = nextKstDay(weekday);
      await call('POST', '/market/bid', { fleet: $('bid-fleet').value || null, model: $('bid-model').value, from: new Date(day + 9 * HOUR).toISOString().slice(0, 13) + ':00Z', to: new Date(day + 18 * HOUR).toISOString().slice(0, 13) + ':00Z', hours: 9, max: $('bid-max').value, consecutive: true, standing: true });
      placed++;
    }
    return placed;
  }, (n) => `${n} standing bids placed, one per weekday.`));
  $('key-create').addEventListener('click', () => act('m-try', async () => {
    const made = await call('POST', '/market/keys', { note: $('key-note').value });
    S.lastKey = made.key;
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
      const response = await fetch(API + '/pool/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + $('try-key').value.trim() },
        body: JSON.stringify({ model: $('try-model').value, stream, max_tokens: 64, messages: [{ role: 'user', content: $('try-prompt').value }] }),
      });
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error((data.error && data.error.message) || 'The pool answered ' + response.status);
      }
      let charge = null;
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
            if (chunk.sparkpipe) charge = chunk.sparkpipe;
          });
        }
      } else {
        const data = await response.json();
        reply.textContent = data.choices[0].message.content;
        charge = data.sparkpipe;
      }
      say('m-try', charge && charge.dedicated ? `Ran on your reserved hour: no HC charged (${charge.seconds} s).` : `Charged ${charge ? charge.hc : '?'} HC for ${charge ? charge.seconds : '?'} seat-seconds.`, 'ok');
      await refresh();
    } catch (error) {
      say('m-try', error.message, 'error');
    }
  });
  $('ledger-load').addEventListener('click', async () => {
    try {
      const data = await call('GET', '/market/ledger');
      $('ledger-head').textContent = `${data.count} lines · head ${data.head.slice(0, 16)}…`;
      fill('ledger', data.lines.slice().reverse().map((l) => row([String(l.seq), kst(l.at), l.type, JSON.stringify(l.data).slice(0, 120), l.hash.slice(0, 12) + '…'])), 'No lines yet.');
    } catch (error) {
      $('ledger-head').textContent = error.message;
    }
  });

  const start = nowMs();
  $('bid-from').value = localFromMs(start + 24 * HOUR);
  $('bid-to').value = localFromMs(start + 48 * HOUR);
  refresh().then(() => {
    $('bid-from').value = localFromMs(Math.floor(nowMs() / HOUR) * HOUR + 24 * HOUR);
    $('bid-to').value = localFromMs(Math.floor(nowMs() / HOUR) * HOUR + 48 * HOUR);
  }).catch((error) => say('m-clock', error.message, 'error'));
  setInterval(() => { refresh().catch(() => {}); }, 20000);
})();
