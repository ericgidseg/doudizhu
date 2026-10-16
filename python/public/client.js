'use strict';
/* ============ 连接 ============ */
const ws = new WebSocket('ws://' + location.host + '/ws');
let S = { roomId: null, mySeat: -1, state: null, selected: new Set(), phase: 'lobby' };

const $ = id => document.getElementById(id);
const QUICK = ['大家好，很高兴见到各位', '快点吧，我等得花儿都谢了', '你的牌打得也太好了', '不要走，决战到天亮', '大家好厉害啊'];

function send(o) { ws.readyState === 1 && ws.send(JSON.stringify(o)); }
function toast(t, ms = 2200) {
  const el = $('toast'); el.textContent = t; el.classList.add('show');
  clearTimeout(toast._t); toast._t = setTimeout(() => el.classList.remove('show'), ms);
}
function logLine(t) {
  const el = $('log'); const d = document.createElement('div');
  d.textContent = t; el.prepend(d);
  while (el.children.length > 60) el.lastChild.remove();
}

ws.onopen = () => {
  $('lobbyMsg').textContent = '';
  /* 刷新/重连后自动收回原座位 (令牌存 sessionStorage, 关闭标签页即失效) */
  const rid = sessionStorage.getItem('ddzRoom'), tok = sessionStorage.getItem('ddzToken');
  if (rid && tok) { send({ type: 'join', roomId: rid, token: tok }); return; }
  send({ type: 'rooms' });
};
ws.onclose = () => { toast('连接已断开, 请刷新页面重连'); };
ws.onerror = () => { $('lobbyMsg').textContent = '无法连接服务器'; };
ws.onmessage = e => {
  let d; try { d = JSON.parse(e.data); } catch { return; }
  switch (d.type) {
    case 'joined':
      S.roomId = d.roomId; S.mySeat = d.seat;
      if (d.token) { sessionStorage.setItem('ddzRoom', d.roomId); sessionStorage.setItem('ddzToken', d.token); }
      $('roomIdTxt').textContent = d.roomId;
      show('room'); break;
    case 'rooms': renderRoomList(d.rooms); break;
    case 'state':
      S.state = d; S.phase = d.phase; S.selected.clear();
      /* 新一局开始 (bidding/playing): 自动关闭上一局的结算弹窗.
         否则房主点"再来一局"后, 其他玩家的弹窗永远留在屏幕上, 无法参与下一局 */
      if (d.phase === 'bidding' || d.phase === 'playing') $('overModal').classList.add('hidden');
      render(); break;
    case 'event': logLine(d.event); break;
    case 'chat': chatBubble(d.seat, d.name, d.text); logLine(d.name + ': ' + d.text); break;
    case 'hint': applyHint(d.ids); break;
    case 'gameOver': showGameOver(d); break;
    case 'error':
      toast(d.msg);
      /* 重连凭证失效 (房间已解散/服务重启): 清掉, 避免每次刷新都重试 */
      if (d.msg === '重连失败' || d.msg === '房间不存在') {
        sessionStorage.removeItem('ddzRoom'); sessionStorage.removeItem('ddzToken');
        send({ type: 'rooms' });
      }
      break;
  }
};

function show(id) {
  for (const s of ['lobby', 'room', 'table']) $(s).classList.toggle('hidden', s !== id);
}

/* ============ 大厅 ============ */
$('createBtn').onclick = () => {
  const name = $('nameInput').value.trim();
  if (!name) return $('lobbyMsg').textContent = '请先输入昵称';
  send({ type: 'create', name });
};
$('soloLegendBtn').onclick = () => {
  const name = $('nameInput').value.trim() || '玩家' + Math.floor(Math.random() * 100);
  send({ type: 'solo', name, diff: 'legend' });
};
$('joinBtn').onclick = () => {
  const name = $('nameInput').value.trim();
  const rid = $('roomInput').value.trim();
  if (!name) return $('lobbyMsg').textContent = '请先输入昵称';
  if (!/^\d{4}$/.test(rid)) return $('lobbyMsg').textContent = '房间号是 4 位数字';
  send({ type: 'join', name, roomId: rid });
};
$('roomInput').onkeydown = e => { if (e.key === 'Enter') $('joinBtn').click(); };
$('nameInput').onkeydown = e => { if (e.key === 'Enter') $('createBtn').click(); };
$('lanTip').textContent = '内网同事直接访问本页面地址即可一起游戏: ' + location.origin;

/* 规则弹窗 */
$('rulesBtn').onclick = () => $('rulesModal').classList.remove('hidden');
$('rulesClose').onclick = () => $('rulesModal').classList.add('hidden');
$('rulesModal').onclick = e => { if (e.target === $('rulesModal')) $('rulesModal').classList.add('hidden'); };

/* 房间列表 */
let roomListTimer = null;
function renderRoomList(rooms) {
  const ul = $('roomList'); ul.innerHTML = '';
  if (!rooms || !rooms.length) {
    ul.innerHTML = '<li class="empty">暂无空房间, 创建一个吧</li>';
    return;
  }
  for (const r of rooms) {
    const li = document.createElement('li');
    li.innerHTML = `<span>房间 ${r.id} · ${r.names.join('、')}</span><span class="cnt">${r.count}/3</span>`;
    li.onclick = () => {
      $('roomInput').value = r.id;
      $('joinBtn').click();
    };
    ul.appendChild(li);
  }
}
function refreshRooms() { if (S.phase === 'lobby') send({ type: 'rooms' }); }
$('refreshRooms').onclick = refreshRooms;
roomListTimer = setInterval(refreshRooms, 5000);

$('startBtn').onclick = () => send({ type: 'start' });
$('addBotBtn').onclick = () => send({ type: 'addbot' });
$('againBtn').onclick = () => { $('overModal').classList.add('hidden'); send({ type: 'start' }); };
$('backBtn').onclick = () => {
  /* 主动返回大厅 = 离开房间, 清掉重连凭证后再刷新 */
  sessionStorage.removeItem('ddzRoom'); sessionStorage.removeItem('ddzToken');
  location.reload();
};

/* ============ 渲染 ============ */
function render() {
  const st = S.state; if (!st) return;
  if (st.phase === 'lobby') { renderRoomWait(st); show('room'); return; }
  show('table');
  renderPlayers(st);
  renderTop(st);
  renderHand(st);
  renderActions(st);
  renderPlayed(st);
  renderScores(st);
  renderCounter(st);
}

/* ---------- 记分牌 ---------- */
function renderScores(st) {
  const el = $('scoreBody');
  el.innerHTML = st.players.map(p => {
    const sc = p.score || 0;
    const cls = sc > 0 ? 'pos' : sc < 0 ? 'neg' : '';
    return `<div class="score-row"><span class="nm">${p.name}${p.bot ? '🤖' : ''}</span><span class="sc ${cls}">${sc > 0 ? '+' : ''}${sc}</span></div>`;
  }).join('');
}

/* ---------- 记牌器: 显示已出的数量 ---------- */
const COUNTER_RANKS = ['BJ', 'SJ', '2', 'A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3'];
function renderCounter(st) {
  const el = $('counterBody');
  if (st.phase !== 'playing') { el.innerHTML = '<div style="color:#7d9b84;font-size:11px;padding:6px;text-align:center">开局后显示</div>'; return; }
  // 已出 = 服务端累计的本局所有已出牌 (公开信息) + 当前桌面 lastPlay 兜底
  const out = {};
  for (const r of COUNTER_RANKS) out[r] = 0;
  for (const c of (st.played || [])) if (out[c.rank] !== undefined) out[c.rank]++;
  if (st.lastPlay && st.lastPlay.cards) {
    for (const c of st.lastPlay.cards) {
      // lastPlay 可能还没进 played (同一回合快照), 防重复: 只补未计的
      if (out[c.rank] !== undefined && !(st.played || []).some(p => p.id === c.id)) out[c.rank]++;
    }
  }
  const total = r => (r === 'BJ' || r === 'SJ') ? 1 : 4;
  el.innerHTML = COUNTER_RANKS.map(r => {
    const n = out[r];
    const done = n >= total(r);          // 已全部出现
    const jk = r === 'BJ' || r === 'SJ';
    const label = r === 'BJ' ? '大王' : r === 'SJ' ? '小王' : r;
    return `<div class="cnt-cell ${done ? 'gone' : ''} ${jk ? 'jk' : ''} ${r === '2' ? 'red' : ''}">
      <span class="r">${label}</span><span class="n">${n}</span></div>`;
  }).join('');
}
/* 面板折叠 */
document.querySelectorAll('.panel-head').forEach(h => {
  h.onclick = () => h.parentElement.classList.toggle('collapsed');
});
$('counterToggle').onclick = () => $('counterStrip').classList.toggle('collapsed');

function renderRoomWait(st) {
  const ul = $('waitList'); ul.innerHTML = '';
  for (let i = 0; i < 3; i++) {
    const li = document.createElement('li');
    const p = st.players[i];
    if (p) {
      li.textContent = p.name + (p.bot ? ' 🤖' : '') + (i === st.hostSeat ? ' 👑' : '') + (p.online ? '' : ' (断线)');
      if (i === S.mySeat) li.classList.add('me');
    } else { li.textContent = '等待玩家加入…'; li.classList.add('empty'); }
    ul.appendChild(li);
  }
  const amHost = S.mySeat === st.hostSeat;
  $('addBotBtn').classList.toggle('hidden', !(amHost && st.players.length < 3));
  $('startBtn').classList.toggle('hidden', !(amHost && st.players.length === 3));
  $('roomMsg').textContent = amHost
    ? (st.players.length === 3 ? '人齐了, 开始吧!' : `已有 ${st.players.length}/3 人`)
    : `已有 ${st.players.length}/3 人, 等待房主开始`;
}

function seatView(st, seat) {
  return st.players.find(p => p.seat === seat);
}
/* 更新常驻 .role-tag 元素 (不能用 outerHTML 删除它, 否则下次渲染找不到节点) */
function setRoleTag(tagEl, seat, st) {
  if (!tagEl) return;
  const labeled = st.phase === 'playing' || st.phase === 'done';
  const lord = labeled && seat === st.landlordSeat;
  tagEl.textContent = labeled ? (lord ? '地主' : '农民') : '';
  tagEl.className = 'role-tag' + (lord ? ' lord' : '');
}
function renderPlayers(st) {
  const left = (S.mySeat + 1) % 3, right = (S.mySeat + 2) % 3;
  for (const [el, seat] of [[$('oppL'), left], [$('oppR'), right]]) {
    const p = seatView(st, seat);
    el.querySelector('.pname').textContent = p ? p.name : '—';
    setRoleTag(el.querySelector('.role-tag'), seat, st);
    el.querySelector('.ccount').textContent = p ? p.cardCount + ' 张' : '';
    el.querySelector('.opp-info').classList.toggle('turn', st.turn === seat);
    const backs = el.querySelector('.opp-backs'); backs.innerHTML = '';
    const n = p ? Math.min(p.cardCount, 17) : 0;
    for (let i = 0; i < n; i++) backs.appendChild(document.createElement('i'));
  }
  const me = seatView(st, S.mySeat);
  $('meInfo').querySelector('.pname').textContent = me ? me.name + ` (${me.cardCount}张)` : '';
  setRoleTag($('meInfo').querySelector('.role-tag'), S.mySeat, st);
  $('meInfo').classList.toggle('turn', st.turn === S.mySeat);
  const isLead = st.phase === 'playing' && st.turn === S.mySeat && !st.lastPlay;
  const isBeat = st.phase === 'playing' && st.turn === S.mySeat && st.lastPlay && st.lastPlay.seat !== S.mySeat;
  /* 倒计时 interval 会补具体秒数; 这里只管清空非 playing 阶段 */
  if (st.phase !== 'playing') $('turnBanner').textContent = '';
}
function renderTop(st) {
  const bc = $('bottomCards'); bc.innerHTML = '';
  /* 服务端只在叫地主结束 (playing/done) 后才发 bottom → 之前不显示底牌 */
  if (st.bottom && st.bottom.length) {
    for (const c of st.bottom) bc.appendChild(cardEl(c, { small: true, passive: true }));
  }
  $('multiTxt').textContent = st.phase !== 'lobby' ? '倍数 ×' + st.multiplier : '';
}
function cardEl(c, { small = false, passive = false } = {}) {
  const el = document.createElement('div');
  const joker = c.rank === 'SJ' || c.rank === 'BJ';
  const red = c.suit === '♥' || c.suit === '♦';
  el.className = 'card' + (small ? ' small' : '')
    + (joker ? (c.rank === 'BJ' ? ' joker joker-b' : ' joker joker-s') : (red ? ' red' : ' black'))
    + (passive ? ' passive' : '');
  if (joker) {
    // 竖排 JOKER 字样, 仿真实扑克王牌
    el.innerHTML = `<span class="joker-col">${'JOKER'.split('').map(ch => `<i>${ch}</i>`).join('')}</span>`;
  } else {
    el.innerHTML =
      `<span class="corner"><span class="rank">${c.rank}</span><span class="suit-sm">${c.suit}</span></span>` +
      `<span class="suit-big">${c.suit}</span>`;
  }
  el.dataset.id = c.id;
  return el;
}
/* 拖选状态 */
const drag = { active: false, target: true, lastId: null };

function renderHand(st) {
  const h = $('hand'); h.innerHTML = '';
  const my = st.turn === S.mySeat && st.phase === 'playing';
  h.style.touchAction = my ? 'pan-y' : 'auto';
  for (const c of (st.hand || [])) {
    const el = cardEl(c, { passive: !my });
    if (S.selected.has(c.id)) el.classList.add('selected');
    if (my) attachDrag(el, c.id);
    h.appendChild(el);
  }
}

/* 拖选 + 单击兼容: 按下开始; 拖动经过的牌统一选中/取消; 松开结束 */
function attachDrag(el, id) {
  el.style.touchAction = 'none';
  el.addEventListener('pointerdown', e => {
    e.preventDefault();
    drag.active = true;
    drag.target = !S.selected.has(id);   // 点的是未选 → 整段选中; 已选 → 整段取消
    drag.lastId = id;
    applySelect(el, id, drag.target);
    try { el.setPointerCapture(e.pointerId); } catch (_) {}
  });
  el.addEventListener('pointermove', e => {
    if (!drag.active) return;
    const t = document.elementFromPoint(e.clientX, e.clientY);
    const card = t && t.closest ? t.closest('.card') : null;
    if (card && card.dataset.id && drag.lastId !== card.dataset.id && card.parentElement === $('hand')) {
      drag.lastId = card.dataset.id;
      const cid = Number(card.dataset.id);
      applySelect(card, cid, drag.target);
    }
  });
  el.addEventListener('pointerup', () => { endDrag(); });
}
function applySelect(el, id, sel) {
  id = Number(id);
  if (sel) S.selected.add(id); else S.selected.delete(id);
  el.classList.toggle('selected', sel);
}
function endDrag() { drag.active = false; drag.lastId = null; }
document.addEventListener('pointerup', endDrag);
document.addEventListener('pointercancel', endDrag);
function renderActions(st) {
  const bar = $('actions'); bar.innerHTML = '';
  const mk = (txt, cls, fn) => {
    const b = document.createElement('button');
    b.className = 'btn ' + (cls || ''); b.textContent = txt; b.onclick = fn;
    bar.appendChild(b); return b;
  };
  if (st.phase === 'bidding' && st.turn === S.mySeat) {
    const opts = [[0, '不叫'], [1, '1 分'], [2, '2 分'], [3, '3 分']];
    for (const [v, t] of opts) {
      if (v !== 0 && v <= st.currentBid) continue;
      mk(t, v === 0 ? 'ghost' : 'gold', () => send({ type: 'bid', bid: v }));
    }
    return;
  }
  if (st.phase === 'playing' && st.turn === S.mySeat) {
    const freeLead = !st.lastPlay || st.lastPlay.seat === S.mySeat;
    if (!freeLead) mk('不出', 'ghost', () => send({ type: 'pass' }));
    mk('提示', '', () => send({ type: 'hint' }));
    mk('出牌', 'warn', () => {
      if (!S.selected.size) return toast('先点击选择要出的牌');
      send({ type: 'play', ids: [...S.selected] });
    });
  }
  if (st.phase === 'playing' && st.turn !== S.mySeat) {
    const who = seatView(st, st.turn);
    if (who) mk(`等待 ${who.name} 出牌…`, 'ghost', () => {});
  }
}
function renderPlayed(st) {
  /* 出牌显示在出牌者位置靠中央方向: 我在下方、左对手在左中、右对手在右中 */
  const cp = $('centerPlay');
  cp.innerHTML = '';
  cp.className = '';
  const lp = st.lastPlay;
  const mySeat = S.mySeat;
  const posClass = seat => seat === mySeat ? 'pos-me'
    : seat === (mySeat + 1) % 3 ? 'pos-left' : 'pos-right';
  if (!lp) {
    return;   // 领出提示走 turnBanner, 不占出牌区
  }
  cp.className = posClass(lp.seat);
  const p = st.players.find(x => x.seat === lp.seat);
  const name = p ? p.name : '玩家' + lp.seat;
  const tag = document.createElement('span');
  tag.className = 'cp-tag';
  tag.textContent = `${name} · ${lp.typeName}`;
  const box = document.createElement('div');
  box.className = 'cp-cards';
  for (const c of lp.cards) box.appendChild(cardEl(c, { small: false, passive: true }));
  cp.appendChild(tag); cp.appendChild(box);
}
function findCard(st, seat, id) {
  return (st.hand || []).find(c => c.id === id) || null;
}

/* ============ 提示 ============ */
let hintCycle = [];
function applyHint(ids) {
  if (!ids || !ids.length) { toast('没有能压过的牌'); return; }
  hintCycle = hintCycle.length && JSON.stringify(hintCycle) === JSON.stringify(ids) ? [] : ids;
  S.selected = new Set(ids);
  renderHand(S.state);
}

/* ============ 聊天 ============ */
for (const q of QUICK) {
  const s = document.createElement('span'); s.textContent = q;
  s.onclick = () => send({ type: 'chat', text: q });
  $('quickChat').appendChild(s);
}
$('chatInput').onkeydown = e => {
  if (e.key === 'Enter' && e.target.value.trim()) {
    send({ type: 'chat', text: e.target.value.trim() });
    e.target.value = '';
  }
};
function chatBubble(seat, name, text) {
  const b = document.createElement('div');
  b.className = 'chat-bubble';
  b.textContent = `${name}: ${text}`;
  const pos = seat === S.mySeat ? { bottom: '170px', left: '50%', transform: 'translateX(-50%)' }
    : seat === (S.mySeat + 1) % 3 ? { top: '150px', left: '16px' }
    : { top: '150px', right: '16px' };
  Object.assign(b.style, pos);
  document.body.appendChild(b);
  setTimeout(() => b.remove(), 3000);
}

/* ============ 结算 ============ */
function showGameOver(d) {
  $('overTitle').textContent = d.landlordWin ? '地主获胜!' : '农民获胜!';
  $('overDetail').innerHTML =
    `胜者: <b>${d.winnerName}</b> · 底分 ${d.baseBid} × 倍数 ${d.multiplier} = ${d.baseBid * d.multiplier}`;
  // 分数表
  if (d.scores) {
    $('overScores').innerHTML = '<tr><th>玩家</th><th>身份</th><th>本局</th><th>总分</th></tr>' +
      d.scores.map(s => `<tr>
        <td>${s.name}</td><td>${s.isLord ? '地主' : '农民'}</td>
        <td class="${s.delta >= 0 ? 'up' : 'down'}">${s.delta >= 0 ? '+' : ''}${s.delta}</td>
        <td>${s.total}</td></tr>`).join('');
  }
  // 各家手牌
  if (d.hands) {
    $('overHands').innerHTML = '';
    for (const h of d.hands) {
      const row = document.createElement('div');
      row.className = 'hand-row' + (h.cards.length ? '' : ' empty');
      row.innerHTML = `<div class="hn">${h.name}${h.isLord ? '<span class="role">地主</span>' : '<span class="role">农民</span>'}${h.seat === S.mySeat ? ' (你)' : ''}</div>`;
      const box = document.createElement('div');
      box.className = 'hcards';
      for (const c of h.cards) box.appendChild(cardEl(c, { small: true, passive: true }));
      row.appendChild(box);
      $('overHands').appendChild(row);
    }
  }
  /* 结算后任意玩家都可以开下一局 (服务端 done 阶段放开权限) */
  $('againBtn').classList.remove('hidden');
  $('overModal').classList.remove('hidden');
}

/* ============ 回合倒计时 ============ */
setInterval(() => {
  const st = S.state;
  const el = $('turnBanner');
  if (!el) return;
  if (!st || (st.phase !== 'playing' && st.phase !== 'bidding')
      || typeof st.turnDeadline !== 'number' || !isFinite(st.turnDeadline)
      || st.turn < 0) {
    if (st && st.phase !== 'playing' && st.phase !== 'bidding') el.textContent = '';
    return;
  }
  const left = Math.max(0, Math.ceil(st.turnDeadline - Date.now() / 1000));
  const isMe = st.turn === S.mySeat;
  const base = !isMe ? '' :
    st.phase === 'bidding' ? '轮到你叫分' :
    (st.lastPlay && st.lastPlay.seat !== S.mySeat ? '轮到你出牌' : '轮到你领出');
  const color = left <= 5 ? '#ff8b6e' : left <= 10 ? '#ffe9b0' : '#a8e6b0';
  el.innerHTML = base
    ? `${base} <span style="color:${color};font-weight:700">${left}s</span>`
    : `<span style="color:${color};font-size:14px;opacity:.85">${left}s</span>`;
}, 200);
