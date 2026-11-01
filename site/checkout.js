(() => {
  const SANDBOX = new URLSearchParams(window.location.search).has('sandbox');
  const API = SANDBOX ? '/pay/sandbox/api' : '/pay/api';
  const call = async (path, body) => {
    const options = body === undefined
      ? { cache: 'no-store', credentials: 'same-origin' }
      : { method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
    let response;
    try { response = await fetch(API + path, options); } catch (_) { throw new Error('Could not reach sparkpipe.ai. Check your connection and try again.'); }
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const error = new Error(data.error || 'The payment service did not answer (' + response.status + '). Try again in a minute.');
      error.status = response.status;
      error.reason = data.reason;
      throw error;
    }
    return data;
  };
  const me = async () => {
    try { return await call('/account'); } catch (error) { if (error.status === 401) return null; throw error; }
  };
  const usd = (n) => '$' + (+n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const krw = (n) => '₩' + Math.round(+n).toLocaleString('en-US');
  const priced = (n, currency) => (currency === 'KRW' ? krw(n) : usd(n));
  const vatOf = (net, currency, rate) => {
    const unit = currency === 'KRW' ? 1 : 100, mille = Math.round(rate * 1000), base = Math.round(net * unit);
    return Math.floor((base * mille + 500) / 1000) / unit;
  };
  const whole = (n) => (+n).toLocaleString('en-US', { maximumFractionDigits: 4 });
  const say = (el, text, kind) => { el.textContent = text || ''; el.className = 'flash' + (kind ? ' ' + kind : ''); el.hidden = !text; };
  const COINS = {
    USDC: [
      { id: 'USDC.ERC20', name: 'Ethereum', note: 'ERC20' },
      { id: 'USDC.BASE', name: 'Base', note: 'Base' },
      { id: 'USDC.SOL', name: 'Solana', note: 'SOL' },
      { id: 'USDC.ARB', name: 'Arbitrum', note: 'Arbitrum One' },
      { id: 'USDC.POL', name: 'Polygon', note: 'Polygon PoS' },
      { id: 'USDC.BEP20', name: 'BNB Chain', note: 'BEP20' },
    ],
    USDT: [
      { id: 'USDT.ERC20', name: 'Ethereum', note: 'ERC20' },
      { id: 'USDT.TRC20', name: 'Tron', note: 'TRC20' },
      { id: 'USDT.SOL', name: 'Solana', note: 'SOL' },
      { id: 'USDT.BEP20', name: 'BNB Chain', note: 'BEP20' },
      { id: 'USDT.ARB', name: 'Arbitrum', note: 'Arbitrum One' },
      { id: 'USDT.POL', name: 'Polygon', note: 'Polygon PoS' },
    ],
  };
  const STATES = {
    created: 'Creating the invoice', unpaid: 'Awaiting payment', pending: 'Payment seen', paid: 'Confirming',
    completed: 'Paid, hours issued', cancelled: 'Cancelled', timedout: 'Expired unpaid', failed: 'No invoice created', mismatch: 'Under review',
    waiting: 'Awaiting deposit', refunded: 'Refunded',
  };
  const METHODS = { '카드': 'Card', '가상계좌': 'Virtual account', '계좌이체': 'Bank transfer', '간편결제': 'Easy pay', '휴대폰': 'Mobile phone' };
  const coinOf = (id) => String(id || '').split('.')[0];
  const coinName = (id) => {
    const found = (COINS[coinOf(id)] || []).find((n) => n.id === id);
    return found ? coinOf(id) + ' on ' + found.name : id || '-';
  };
  const paidWith = (o) => (o.processor === 'toss' ? (METHODS[o.method] || 'Card or bank') + ', Toss Payments' : coinName(o.coin));
  const stateOf = (o) => (o.processor === 'toss' && o.state === 'failed' ? 'Not paid' : o.processor === 'toss' && o.state === 'created' ? 'Awaiting payment' : STATES[o.state] || o.state);
  const tossStatus = (o) => {
    const total = krw(o.total), d = o.deposit || {};
    if (o.state === 'completed' && o.issued) return null;
    if (o.state === 'waiting') return { title: 'Waiting for your deposit.', text: 'Deposit ' + total + ' to ' + d.bank + ' account ' + d.accountNumber + ', the virtual account Toss Payments issued for this order, by ' + new Date(o.expires).toLocaleString('en-GB', { dateStyle: 'medium', timeStyle: 'short', timeZone: 'Asia/Seoul' }) + ' KST. This page updates by itself once it arrives; if nothing arrives by then, the account expires with the order.', step: 1 };
    if (o.state === 'created') return { title: 'Confirming your payment…', text: 'Toss Payments is confirming the payment of ' + total + '. This page updates by itself.', step: 2 };
    if (o.state === 'refunded') return { title: 'This payment was refunded.', text: krw(o.refunded) + ' went back by the method you paid with, through Toss Payments.', kind: 'ok', step: 4, retry: false };
    if (o.state === 'mismatch' || (o.state === 'cancelled' && o.reason === 'amount')) return { title: 'This payment did not match its order.', text: 'Toss Payments reported a different amount, so nothing was issued' + (o.state === 'cancelled' ? ' and the payment was cancelled.' : '. We will email you; you can also write to hyor@sparkpipe.ai with order ' + o.order + '.'), kind: 'error', step: 0, retry: true };
    const why = o.state === 'timedout' ? 'The payment was not completed in time.' : o.state === 'cancelled' ? 'The payment was cancelled.' : 'Toss Payments did not complete the payment' + (o.reason && o.reason !== 'amount' ? ' (' + o.reason + ')' : '') + '.';
    return { title: 'Not paid.', text: why + ' Nothing was charged and nothing was issued.', kind: o.state === 'failed' ? 'error' : undefined, step: 0, retry: true };
  };
  const TOSS = { config: null, widgets: null, box: null, parts: null, agreed: null, chain: Promise.resolve() };
  const tossConfig = () => TOSS.config || (TOSS.config = call('/toss/config').catch(() => ({ enabled: false })));
  const tossSdk = () => new Promise((resolve, reject) => {
    const fail = () => reject(new Error('Toss Payments did not load. Check your connection, or pay in USDC or USDT.'));
    if (window.TossPayments) return resolve(window.TossPayments);
    const tag = document.getElementById('toss-sdk');
    if (!tag) return fail();
    tag.addEventListener('load', () => (window.TossPayments ? resolve(window.TossPayments) : fail()));
    tag.addEventListener('error', fail);
    setTimeout(() => (window.TossPayments ? resolve(window.TossPayments) : fail()), 15000);
  });
  let mounts = 0;
  const tossMount = (methods, agreement, amount) => (TOSS.chain = TOSS.chain.catch(() => {}).then(async () => {
    const config = await tossConfig();
    if (!config.enabled) throw new Error('Card and bank payments are not open yet. Pay in USDC or USDT.');
    const Toss = await tossSdk();
    if (!TOSS.widgets) TOSS.widgets = Toss(config.clientKey).widgets({ customerKey: Toss.ANONYMOUS });
    await TOSS.widgets.setAmount({ currency: 'KRW', value: amount });
    if (TOSS.box === methods) return;
    if (TOSS.parts) await Promise.all(TOSS.parts.map((part) => part.destroy()));
    mounts += 1;
    methods.id = 'toss-methods-' + mounts;
    agreement.id = 'toss-agreement-' + mounts;
    TOSS.agreed = null;
    TOSS.parts = await Promise.all([
      TOSS.widgets.renderPaymentMethods({ selector: '#' + methods.id, variantKey: config.variantKey }),
      TOSS.widgets.renderAgreement({ selector: '#' + agreement.id }),
    ]);
    TOSS.parts[1].on('agreementStatusChange', (status) => { TOSS.agreed = status.agreedRequiredTerms; });
    TOSS.box = methods;
  }));
  const tossAmount = (amount) => (TOSS.widgets ? (TOSS.chain = TOSS.chain.catch(() => {}).then(() => TOSS.widgets.setAmount({ currency: 'KRW', value: amount }))) : Promise.resolve());
  const tossReady = () => {
    if (!TOSS.widgets || !TOSS.box) throw new Error('The card and bank payment form has not loaded yet. Wait a moment and try again.');
    if (TOSS.agreed === false) throw new Error('Accept the Toss Payments terms below the payment methods to continue.');
  };
  const tossPay = async (request) => {
    await TOSS.chain;
    await TOSS.widgets.setAmount({ currency: 'KRW', value: request.amount });
    const asked = { orderId: request.orderId, orderName: request.orderName, successUrl: request.successUrl, failUrl: request.failUrl, customerName: request.customerName };
    if (request.customerEmail) asked.customerEmail = request.customerEmail;
    try {
      await TOSS.widgets.requestPayment(asked);
    } catch (error) {
      const code = error && /^[A-Z0-9_]{1,100}$/.test(error.code || '') ? error.code : 'PAY_PROCESS_CANCELED';
      await fetch(request.failUrl + '?code=' + code, { credentials: 'same-origin', redirect: 'manual' }).catch(() => {});
      throw error;
    }
  };
  const radio = (name, value, checked) => { const input = document.createElement('input'); Object.assign(input, { type: 'radio', name, value, checked }); return input; };
  const coinPicker = (box, name, chosen, onChange) => {
    let current = (COINS[coinOf(chosen)] || []).some((n) => n.id === chosen) ? chosen : COINS.USDC[0].id;
    const coins = document.createElement('div');
    coins.className = 'coins';
    coins.setAttribute('role', 'radiogroup');
    coins.setAttribute('aria-label', 'Stablecoin');
    Object.keys(COINS).forEach((coin) => {
      const label = document.createElement('label');
      label.className = 'coin';
      label.append(radio(name + '-coin', coin, coin === coinOf(current)), document.createTextNode(coin));
      coins.append(label);
    });
    const tiles = document.createElement('div');
    tiles.className = 'tiles';
    tiles.setAttribute('role', 'radiogroup');
    tiles.setAttribute('aria-label', 'Network');
    const draw = () => {
      tiles.replaceChildren(...COINS[coinOf(current)].map((n) => {
        const label = document.createElement('label');
        label.className = 'tile net';
        const b = document.createElement('b');
        b.textContent = n.name;
        const span = document.createElement('span');
        span.textContent = coinOf(n.id) + ' · ' + n.note;
        label.append(radio(name, n.id, n.id === current), b, span);
        if (n.tag) { const em = document.createElement('em'); em.textContent = n.tag; label.append(em); }
        return label;
      }));
    };
    box.addEventListener('change', (event) => {
      if (event.target.name === name + '-coin') current = COINS[event.target.value][0].id;
      else if (event.target.name === name) current = event.target.value;
      else return;
      draw();
      if (onChange) onChange(current);
    });
    draw();
    box.replaceChildren(coins, tiles);
  };
  const chosen = (form, name) => { const input = form.querySelector('input[name="' + name + '"]:checked'); return input ? input.value : null; };
  const store = {
    get: (key) => { try { return JSON.parse(sessionStorage.getItem(key) || 'null'); } catch (_) { return null; } },
    set: (key, value) => { try { sessionStorage.setItem(key, JSON.stringify(value)); } catch (_) {} },
  };
  const codeFlow = (parts) => {
    let email = '';
    const send = async (address) => {
      email = address;
      const sent = await call('/code', { email });
      parts.codeForm.hidden = false;
      parts.sentTo.textContent = email;
      parts.code.value = SANDBOX && sent.code ? sent.code : '';
      parts.code.focus();
    };
    parts.code.addEventListener('input', () => {
      parts.code.value = parts.code.value.replace(/\D/g, '').slice(0, 6);
      if (parts.code.value.length === 6 && parts.autoSubmit) parts.codeForm.requestSubmit();
    });
    return { send, email: () => email };
  };
  const methodPicker = (box, name, onChange) => {
    const tiles = document.createElement('div');
    tiles.className = 'tiles methods';
    tiles.setAttribute('role', 'radiogroup');
    tiles.setAttribute('aria-label', 'How you pay');
    const tile = (value, title, note) => {
      const label = document.createElement('label');
      label.className = 'tile method';
      const b = document.createElement('b');
      b.textContent = title;
      const span = document.createElement('span');
      span.textContent = note;
      label.append(radio(name, value, value === 'coin'), b, span);
      return label;
    };
    const toss = tile('toss', 'Card or Korean bank transfer', 'Toss Payments, in won');
    toss.hidden = true;
    tiles.append(toss, tile('coin', 'USDC or USDT', 'CoinPayments, in US dollars'));
    box.prepend(tiles);
    tiles.addEventListener('change', (event) => { if (event.target.name === name) onChange(event.target.value); });
    return {
      offer: (on) => { toss.hidden = !on; if (on && !tiles.querySelector('input:checked')) toss.querySelector('input').checked = true; },
      choose: (value) => { const input = tiles.querySelector('input[value="' + value + '"]'); if (input) input.checked = true; onChange(value); },
      note: (value, text) => { tiles.querySelector('input[value="' + value + '"]').parentElement.querySelector('span').textContent = text; },
    };
  };
  const planPay = (form, a, P) => {
    const k = (name) => form.querySelector('[data-k="' + name + '"]');
    const amount = form.elements.amount, note = form.querySelector('.flash'), button = form.querySelector('button[type=submit]');
    const box = k('pay'), coins = document.createElement('div'), toss = document.createElement('div'), methods = document.createElement('div'), agreement = document.createElement('div');
    toss.className = 'toss-box';
    toss.hidden = true;
    toss.append(methods, agreement);
    box.append(toss, coins);
    let method = 'coin';
    const net = () => (method === 'toss' ? Math.round(+amount.value || 0) : Math.round((+amount.value || 0) * 100) / 100);
    const update = () => {
      const currency = method === 'toss' ? 'KRW' : 'USD', value = net(), vat = vatOf(value, currency, P.vatRate), price = method === 'toss' ? +a.priceKRW : +a.price;
      const smallest = method === 'toss' ? 1000 : 1, left = method === 'toss' ? +a.remainingKRW : +a.remaining;
      const how = method === 'toss' ? 'by card or bank transfer' : 'in ' + coinOf(chosen(form, 'coin-' + a.plan) || 'USDC.ERC20');
      k('issues').textContent = value >= smallest ? 'Issues ' + whole(Math.floor(value / price * 10000) / 10000) + ' hours. Total with VAT: ' + priced(value + vat, currency) + ' ' + how + '.' : '';
      button.textContent = value >= smallest ? 'Pay ' + priced(value + vat, currency) + ' ' + how : 'Pay ' + how;
      button.disabled = Boolean(a.ended) || value < smallest || value > left;
      if (method === 'toss' && value >= smallest) tossAmount(value + vat);
    };
    const switchTo = (value) => {
      method = value;
      const won = method === 'toss';
      k('unit').textContent = won ? 'Amount, before VAT (KRW)' : 'Amount, before VAT (USD)';
      amount.step = won ? '1' : '0.01';
      amount.min = won ? '1000' : '1';
      amount.value = won ? a.nextGateKRW || '' : a.nextGate ? a.nextGate.amount : '';
      toss.hidden = !won;
      coins.hidden = won;
      update();
      if (won) tossMount(methods, agreement, Math.max(net() + vatOf(net(), 'KRW', P.vatRate), 1100)).catch((error) => say(note, error.message, 'error'));
    };
    const picker = methodPicker(box, 'method-' + a.plan, switchTo);
    coinPicker(coins, 'coin-' + a.plan, 'USDC.ERC20', () => update());
    amount.addEventListener('input', update);
    switchTo('coin');
    if (a.priceKRW && !a.ended) tossConfig().then((config) => { if (config.enabled) { picker.offer(true); picker.choose('toss'); } });
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      button.disabled = true;
      say(note, '');
      try {
        if (method === 'toss') {
          tossReady();
          const placed = await call('/order', { product: 'plan', plan: a.plan, amount: String(net()), processor: 'toss' });
          await tossPay(placed.toss);
          return;
        }
        const placed = await call('/order', { product: 'plan', plan: a.plan, amount: net().toFixed(2), coin: chosen(form, 'coin-' + a.plan) });
        if (!/^https:\/\//.test(placed.checkout)) throw new Error('The checkout link is not secure. Your order is listed below.');
        window.location.assign(placed.checkout);
      } catch (error) { say(note, error.message, 'error'); update(); }
    });
  };
  window.SP = { call, me, usd, krw, priced, vatOf, whole, say, COINS, STATES, METHODS, coinOf, coinName, paidWith, stateOf, tossStatus, coinPicker, methodPicker, planPay, chosen, store, codeFlow, tossConfig, tossMount, tossAmount, tossReady, tossPay, sandbox: SANDBOX };
})();
