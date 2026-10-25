(function () {
  if (window.sparkpipeControls) return;
  window.sparkpipeControls = true;
  var TEXT = {
    en: {
      language: 'Language', memory: 'Memory', on: 'on', off: 'off', keep: 'Keep in full', summary: 'summary only',
      compact: 'Compact', undo: 'Undo', context: 'Context', local: 'Local mode',
      memoryTitle: 'Memory keeps long chats short: it summarizes older turns and resumes from the cached prefix. Off sends the whole chat every time.',
      compactTitle: 'Keep the most relevant earlier exchanges in full, up to this share, and summarize the rest. 0% keeps only a summary.',
      undoTitle: 'Restore every exchange the last compaction summarized.',
      contextTitle: 'Show what the next prompt holds.',
      localTitle: 'Chat with nothing stored on our servers: your chats stay in this browser.'
    },
    ko: {
      language: '언어', memory: '메모리', on: '켜짐', off: '꺼짐', keep: '원문 유지', summary: '요약만',
      compact: '압축', undo: '되돌리기', context: '컨텍스트', local: '로컬 모드',
      memoryTitle: '메모리는 긴 대화를 짧게 유지합니다. 오래된 대화를 요약하고 캐시된 앞부분부터 이어 갑니다. 끄면 매번 대화 전체를 보냅니다.',
      compactTitle: '관련성이 높은 이전 대화를 이 비율까지 원문으로 남기고 나머지는 요약합니다. 0%는 요약만 남깁니다.',
      undoTitle: '마지막 압축으로 요약된 대화를 모두 되돌립니다.',
      contextTitle: '다음 프롬프트에 무엇이 들어가는지 보여 줍니다.',
      localTitle: '서버에 아무것도 저장하지 않는 채팅입니다. 대화는 이 브라우저에만 남습니다.'
    }
  };
  var read = function (key) { try { return localStorage.getItem(key); } catch (e) { return null; } };
  var write = function (key, value) { try { localStorage.setItem(key, value); } catch (e) {} };
  var korean = function () { return (read('locale') || navigator.language || '').toLowerCase().indexOf('ko') === 0; };
  var t = function (key) { return TEXT[korean() ? 'ko' : 'en'][key]; };
  var chatId = function () { var found = location.pathname.match(/^\/c\/([^/?#]+)/); return found ? found[1] : ''; };
  var memoryKey = function () { return 'sparkpipe-memory-' + (chatId() || 'new'); };

  var style = document.createElement('style');
  style.textContent =
    '#sparkpipe-bar{display:flex;flex-wrap:wrap;align-items:center;gap:6px 14px;margin:0 auto 6px;padding:0 10px;font-size:12px;line-height:1.4;opacity:.9;max-width:100%}' +
    '#sparkpipe-bar button,#sparkpipe-lang button{font:inherit;color:inherit;background:transparent;border:1px solid currentColor;border-radius:999px;padding:2px 9px;cursor:pointer;opacity:.75}' +
    '#sparkpipe-bar button:hover,#sparkpipe-lang button:hover{opacity:1}' +
    '#sparkpipe-bar button[aria-pressed="true"],#sparkpipe-lang button[aria-pressed="true"]{opacity:1;font-weight:600}' +
    '#sparkpipe-bar label{display:inline-flex;align-items:center;gap:6px}' +
    '#sparkpipe-bar input[type=range]{width:110px}' +
    '#sparkpipe-bar a{color:inherit;text-decoration:underline;opacity:.85}' +
    '.sparkpipe-group{display:inline-flex;align-items:center;gap:6px}' +
    '#sparkpipe-lang{position:fixed;top:10px;right:12px;z-index:9999;display:flex;gap:4px;font-size:12px}';
  document.head.appendChild(style);

  var switchLanguage = function (code) {
    if ((code === 'ko-KR') === korean()) return;
    write('locale', code);
    location.reload();
  };
  var languageButtons = function () {
    var group = document.createElement('span');
    group.className = 'sparkpipe-group';
    group.setAttribute('role', 'group');
    group.setAttribute('aria-label', 'Language / 언어');
    [['en-US', 'EN'], ['ko-KR', '한국어']].forEach(function (pair) {
      var button = document.createElement('button');
      button.type = 'button';
      button.textContent = pair[1];
      button.setAttribute('aria-pressed', String((pair[0] === 'ko-KR') === korean()));
      button.addEventListener('click', function () { switchLanguage(pair[0]); });
      group.appendChild(button);
    });
    return group;
  };

  var send = function (text) {
    var box = document.querySelector('#chat-input');
    if (!box) return;
    box.focus();
    if (box.isContentEditable) {
      document.execCommand('selectAll', false);
      document.execCommand('insertText', false, text);
    } else {
      var setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
      setter.call(box, text);
      box.dispatchEvent(new Event('input', { bubbles: true }));
    }
    setTimeout(function () {
      var button = document.querySelector('#send-message-button');
      if (button && !button.disabled) button.click();
    }, 200);
  };

  var bar = function () {
    var root = document.createElement('div'), memory = document.createElement('button'), range = document.createElement('input');
    var share = document.createElement('span'), keep = document.createElement('label'), compact = document.createElement('button');
    var undo = document.createElement('button'), context = document.createElement('button'), local = document.createElement('a');
    root.id = 'sparkpipe-bar';
    var paint = function () {
      var on = read(memoryKey()) !== 'off';
      memory.textContent = t('memory') + ': ' + t(on ? 'on' : 'off');
      memory.setAttribute('aria-pressed', String(on));
    };
    memory.type = 'button';
    memory.title = t('memoryTitle');
    memory.addEventListener('click', function () {
      var on = read(memoryKey()) !== 'off';
      write(memoryKey(), on ? 'off' : 'on');
      paint();
      send(on ? '/memory off' : '/memory on');
    });
    range.type = 'range';
    range.min = '0';
    range.max = '90';
    range.step = '10';
    range.value = read('sparkpipe-compact-share') || '50';
    var label = function () { share.textContent = range.value === '0' ? t('summary') : range.value + '%'; };
    range.addEventListener('input', function () { write('sparkpipe-compact-share', range.value); label(); });
    keep.title = t('compactTitle');
    keep.append(document.createTextNode(t('keep')), range, share);
    compact.type = 'button';
    compact.textContent = t('compact');
    compact.title = t('compactTitle');
    compact.addEventListener('click', function () { send(range.value === '0' ? '/compact full' : '/compact ' + range.value + '%'); });
    undo.type = 'button';
    undo.textContent = t('undo');
    undo.title = t('undoTitle');
    undo.addEventListener('click', function () { send('/compact undo'); });
    context.type = 'button';
    context.textContent = t('context');
    context.title = t('contextTitle');
    context.addEventListener('click', function () { send('/context'); });
    local.href = '/local/';
    local.textContent = t('local');
    local.title = t('localTitle');
    var compactGroup = document.createElement('span');
    compactGroup.className = 'sparkpipe-group';
    compactGroup.append(keep, compact, undo);
    root.append(languageButtons(), memory, compactGroup, context, local);
    label();
    paint();
    root.repaint = paint;
    return root;
  };

  var forgetNew = function () { try { localStorage.removeItem('sparkpipe-memory-new'); } catch (e) {} };
  if (!chatId()) forgetNew();
  var lastChat = null;
  var mount = function () {
    var input = document.querySelector('#chat-input-container'), existing = document.getElementById('sparkpipe-bar'), pill = document.getElementById('sparkpipe-lang');
    if (input) {
      if (pill) pill.remove();
      if (!existing || existing.nextElementSibling !== input) {
        if (existing) existing.remove();
        input.parentElement.insertBefore(bar(), input);
      } else if (lastChat !== chatId()) {
        if (lastChat === '' && chatId() && read('sparkpipe-memory-new')) write(memoryKey(), read('sparkpipe-memory-new'));
        if (lastChat !== '') forgetNew();
        if (chatId()) forgetNew();
        existing.repaint();
      }
      lastChat = chatId();
      return;
    }
    if (existing) existing.remove();
    if (!pill) {
      pill = document.createElement('div');
      pill.id = 'sparkpipe-lang';
      pill.appendChild(languageButtons());
      document.body.appendChild(pill);
    }
  };
  var queued = false;
  new MutationObserver(function () {
    if (queued) return;
    queued = true;
    requestAnimationFrame(function () { queued = false; mount(); });
  }).observe(document.documentElement, { childList: true, subtree: true });
  if (document.body) mount();
  else document.addEventListener('DOMContentLoaded', mount);
})();
