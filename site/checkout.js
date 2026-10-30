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
  const NETWORKS = [
    { id: 'base', name: 'Base', note: 'Lowest fees', tag: 'Cheapest' },
    { id: 'solana', name: 'Solana', note: 'Low fees, fast' },
    { id: 'arbitrum', name: 'Arbitrum', note: 'Low fees' },
    { id: 'polygon', name: 'Polygon', note: 'Low fees' },
    { id: 'bnb', name: 'BNB Chain', note: 'Low fees' },
    { id: 'ethereum', name: 'Ethereum', note: 'Highest network fees' },
  ];
  const STATES = {
    created: 'Creating the invoice', unpaid: 'Awaiting payment', pending: 'Payment seen', paid: 'Confirming',
    completed: 'Paid, hours issued', cancelled: 'Cancelled', timedout: 'Expired unpaid', failed: 'No invoice created', mismatch: 'Under review',
  };
  const networkName = (id) => (NETWORKS.find((n) => n.id === id) || { name: id || '-' }).name;
  const networkTiles = (box, name, chosen) => {
    box.replaceChildren(...NETWORKS.map((n) => {
      const label = document.createElement('label');
      label.className = 'tile net';
      const input = document.createElement('input');
      Object.assign(input, { type: 'radio', name, value: n.id, checked: n.id === chosen });
      const b = document.createElement('b');
      b.textContent = n.name;
      const span = document.createElement('span');
      span.textContent = 'USDC · ' + n.note;
      label.append(input, b, span);
      if (n.tag) { const em = document.createElement('em'); em.textContent = n.tag; label.append(em); }
      return label;
    }));
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
  window.SP = { call, me, usd, whole, say, NETWORKS, STATES, networkName, networkTiles, chosen, store, codeFlow };
})();
