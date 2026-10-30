(() => {
  const API = '/pay/api';
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
  };
  const coinOf = (id) => String(id || '').split('.')[0];
  const coinName = (id) => {
    const found = (COINS[coinOf(id)] || []).find((n) => n.id === id);
    return found ? coinOf(id) + ' on ' + found.name : id || '-';
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
      await call('/code', { email });
      parts.codeForm.hidden = false;
      parts.sentTo.textContent = email;
      parts.code.value = '';
      parts.code.focus();
    };
    parts.code.addEventListener('input', () => {
      parts.code.value = parts.code.value.replace(/\D/g, '').slice(0, 6);
      if (parts.code.value.length === 6 && parts.autoSubmit) parts.codeForm.requestSubmit();
    });
    return { send, email: () => email };
  };
  window.SP = { call, me, usd, whole, say, COINS, STATES, coinOf, coinName, coinPicker, chosen, store, codeFlow };
})();
