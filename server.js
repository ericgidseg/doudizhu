'use strict';
/**
 * 斗地主服务端 —— 零依赖 (node >= 16)
 * 用法: node server.js [port]     默认端口 8000
 * 局域网内其他玩家浏览器访问: http://<本机IP>:<port>
 */
const http = require('http');
const fs = require('fs');
const path = require('path');
const os = require('os');
const C = require('./cards.js');
const AI = require('./ai.js');
const { attach } = require('./ws.js');

/* SEA 单文件模式下静态资源内嵌; 普通模式读磁盘 public/ */
let sea = null;
try { sea = require('node:sea'); } catch (e) { /* Node < 20 无此模块 */ }
const isSea = !!(sea && sea.isSea && sea.isSea());
function readAsset(rel) {
  if (isSea) {
    try { return Buffer.from(sea.getAsset('public/' + rel)); } catch (e) { return null; }
  }
  try { return fs.readFileSync(path.join(PUBLIC, rel)); } catch (e) { return null; }
}

const PORT = parseInt(process.argv[2] || '8000', 10);
const PUBLIC = path.join(__dirname, 'public');
const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8', '.png': 'image/png', '.jpg': 'image/jpeg',
  '.svg': 'image/svg+xml', '.ico': 'image/x-icon', '.json': 'application/json',
};

const httpServer = http.createServer((req, res) => {
  const urlPath = decodeURIComponent((req.url || '/').split('?')[0]);
  /* ---- 复盘 API: /api/replays 列表, /api/replay?f=xxx.json 详情 ---- */
  if (urlPath === '/api/replays') {
    try {
      const list = fs.readdirSync(REPLAY_DIR).filter(f => f.endsWith('.json'))
        .map(f => {
          try {
            const r = JSON.parse(fs.readFileSync(path.join(REPLAY_DIR, f), 'utf8'));
            return {
              f,
              roomId: r.roomId, seq: r.seq, difficulty: r.difficulty,
              startedAt: r.startedAt, finishedAt: r.finishedAt,
              landlordSeat: r.landlordSeat, winnerSeat: r.winnerSeat, landlordWin: r.landlordWin,
              players: r.players.map(p => ({ seat: p.seat, name: p.name, bot: p.bot })),
              actions: r.actions.length,
            };
          } catch (e) { return null; }
        }).filter(Boolean)
        .sort((a, b) => (b.startedAt || '').localeCompare(a.startedAt || ''));
      res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
      return res.end(JSON.stringify(list));
    } catch (e) { res.writeHead(200, { 'Content-Type': 'application/json' }); return res.end('[]'); }
  }
  if (urlPath === '/api/replay') {
    const q = (req.url.split('?')[1] || '');
    const file = new URLSearchParams(q).get('f') || '';
    if (!/^[\w.-]+\.json$/.test(file)) { res.writeHead(400); return res.end('bad file'); }
    try {
      const data = fs.readFileSync(path.join(REPLAY_DIR, file), 'utf8');
      res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
      return res.end(data);
    } catch (e) { res.writeHead(404); return res.end('not found'); }
  }
  /* ---- 静态资源 ---- */
  const rel = urlPath === '/' ? 'index.html' : urlPath.replace(/^\/+/, '');
  if (rel.includes('..')) { res.writeHead(403); return res.end(); }
  const data = readAsset(rel);
  if (!data) { res.writeHead(404); return res.end('not found'); }
  res.writeHead(200, { 'Content-Type': MIME[path.extname(rel)] || 'application/octet-stream' });
  res.end(data);
});

/* ---------------- 房间 ---------------- */
const rooms = new Map();  // id -> room
const sanitize = s => String(s || '').slice(0, 12).replace(/[<>&"']/g, '') || '玩家';

function genRoomId() {
  let id;
  do { id = String(1000 + Math.floor(Math.random() * 9000)); } while (rooms.has(id));
  return id;
}
function newRoom() {
  return {
    id: genRoomId(),
    players: [],            // [{name, conn|null, seat, hand:[], wins, lastIds, lastName}]
    hostSeat: -1,
    phase: 'lobby',         // lobby | bidding | playing | done
    dealerIdx: 0,
    bottom: [],
    currentBid: 0,
    currentBidder: -1,
    bids: [],
    turn: -1,
    passCount: 0,
    lastPlay: null,         // { ids, ana, seat }
    multiplier: 1,
    landlordSeat: -1,
    difficulty: 'normal',   // AI 难度: normal | hard | expert | legend
    played: [],             // 本局已出的所有牌 (公开信息, 供记牌器统计)
    replay: null,           // 本局录制数据 (见 recordInit)
  };
}

/* ---------------- 对局录制 ---------------- */
/* 打包(.command/SEA)模式下 __dirname 是临时解压根目录, 重启就丢;
   写到用户目录持久化. 可用环境变量 DDZ_REPLAY_DIR 覆盖. */
const REPLAY_DIR = process.env.DDZ_REPLAY_DIR ||
  (isSea ? path.join(os.homedir(), 'Library', 'Application Support', 'doudizhu', 'replays')
         : path.join(__dirname, 'replays'));
const REPLAY_KEEP = 100;                    // 每局一文件, 保留最近 100
const replayFile = room => path.join(REPLAY_DIR, `r${room.id}-${room._seq || 0}.json`);

function recordInit(room) {
  room._seq = (room._seq || 0) + 1;         // 同一房间连打几局
  room.replay = {
    roomId: room.id,
    seq: room._seq,
    difficulty: room.difficulty,
    startedAt: new Date().toISOString(),
    players: room.players.map(p => ({
      seat: p.seat, name: p.name, bot: !!p.bot,
      hand: p.hand.map(c => ({ id: c.id, rank: c.rank, suit: c.suit })),
    })),
    actions: [],                            // {type:'bid',seat,bid} | {type:'play',seat,ids,cards,typeName,ana:{type,main,len}} | {type:'pass',seat}
    bottom: [],                             // 亮地主后填
    landlordSeat: -1,
    winnerSeat: -1,
    landlordWin: false,
    finalMultiplier: 0,
    finishedAt: null,
  };
}
function replaySnapshot(room) {
  return Object.assign({}, room.replay, {
    currentHands: room.players.map(p => p.hand.map(c => ({ id: c.id, rank: c.rank, suit: c.suit }))),
  });
}
function saveReplay(room) {
  if (!room.replay) return;
  try {
    fs.mkdirSync(REPLAY_DIR, { recursive: true });
    fs.writeFileSync(replayFile(room), JSON.stringify(replaySnapshot(room), null, 1));
    // 滚动清理
    const files = fs.readdirSync(REPLAY_DIR).filter(f => f.endsWith('.json')).sort();
    while (files.length > REPLAY_KEEP) {
      fs.unlinkSync(path.join(REPLAY_DIR, files.shift()));
    }
  } catch (e) { console.error('save replay failed', e); }
}
function seatOf(room, conn) {
  const p = room.players.find(p => p.conn === conn);
  return p ? p.seat : -1;
}
function playerView(p) {
  return { seat: p.seat, name: p.name, cardCount: p.hand.length, online: !!p.conn || !!p.bot, wins: p.wins, bot: !!p.bot, score: p.score || 0 };
}
function broadcast(room, msg, data = {}) {
  for (const p of room.players) if (p.conn) p.conn.send(Object.assign({ type: msg }, data));
}
/** 全量状态: 每个人只看到自己的手牌 */
function sendState(room) {
  const base = {
    id: room.id, phase: room.phase, hostSeat: room.hostSeat,
    players: room.players.map(playerView),
    turn: room.turn, currentBid: room.currentBid, currentBidder: room.currentBidder,
    bids: room.bids, multiplier: room.multiplier, difficulty: room.difficulty,
    landlordSeat: room.landlordSeat, bottomCount: room.bottom.length,
  };
  for (const p of room.players) {
    if (!p.conn) continue;
    p.conn.send({
      type: 'state', ...base,
      mySeat: p.seat,
      hand: p.conn ? C.sortCards(p.hand) : [],
      /* 底牌对所有人公开显示 (斗地主惯例: 亮底牌) */
      bottom: room.phase !== 'lobby' ? C.sortCards(room.bottom) : [],
      lastPlay: room.lastPlay && {
        seat: room.lastPlay.seat,
        ids: room.lastPlay.ids.slice(),
        cards: room.lastPlay.cards.map(c => ({ id: c.id, rank: c.rank, suit: c.suit })),
        typeName: room.lastPlay.ana.typeName,
      },
      played: room.played.map(c => ({ id: c.id, rank: c.rank })),   // 记牌器: 已出牌面
    });
  }
}
function cleanupRoom(room) {
  // 房间内没有任何真人连接才删 (纯 AI 房不会出现)
  if (room.players.every(p => p.bot || !p.conn)) rooms.delete(room.id);
}

/* AI 调度: 轮到机器人时延迟后执行决策 */
function scheduleBot(room) {
  if (room.phase !== 'bidding' && room.phase !== 'playing') return;
  const p = room.players[room.turn];
  if (!p || !p.bot) return;
  clearTimeout(room._botTimer);
  room._botTimer = setTimeout(() => {
    if (!rooms.has(room.id)) return;
    try {
      const act = AI.decide(room, room.turn);
      if (act.type === 'bid') applyBid(room, room.turn, act.bid);
      else if (act.type === 'play') applyPlay(room, room.turn, act.ids);
      else if (act.type === 'pass') applyPass(room, room.turn);
    } catch (e) { console.error('bot error', e); }
  }, 500 + Math.random() * 600);
}
function addBotToRoom(room) {
  if (room.players.length >= 3) return null;
  const seat = room.players.length;
  const p = { name: 'AI·' + '甲乙丙'[seat], conn: null, seat, hand: [], wins: 0, lastIds: [], bot: true };
  room.players.push(p);
  return p;
}

/* ---------------- 游戏流程 ---------------- */
function startGame(room) {
  const deck = C.shuffle(C.createDeck());
  for (const p of room.players) { p.hand = deck.splice(0, 17); p.lastIds = []; }
  room.bottom = deck;                       // 3 张底牌
  room.phase = 'bidding';
  room.dealerIdx = Math.floor(Math.random() * 3);
  room.turn = room.dealerIdx;
  room.currentBid = 0; room.currentBidder = -1;
  room.bids = new Array(3).fill(-1);        // -1 未叫
  room.multiplier = 1; room.landlordSeat = -1;
  room.passCount = 0; room.lastPlay = null;
  room.played = [];
  recordInit(room);
  broadcast(room, 'event', { event: `第 ${room.id} 房间开局, ${room.players[room.dealerIdx].name} 先叫分` });
  sendState(room);
  scheduleBot(room);
}
function applyBid(room, seat, bid) {
  room.bids[seat] = bid;
  if (room.replay) room.replay.actions.push({ type: 'bid', seat, bid });
  if (bid > room.currentBid) { room.currentBid = bid; room.currentBidder = seat; }
  const name = room.players[seat].name;
  broadcast(room, 'event', { event: `${name} ${bid === 0 ? '不叫' : '叫 ' + bid + ' 分'}` });
  if (bid === 3 || room.bids.every(b => b !== -1)) {
    if (room.currentBidder === -1) {        // 全不叫 -> 重新发牌
      broadcast(room, 'event', { event: '无人叫地主, 重新发牌' });
      return setTimeout(() => { if (rooms.has(room.id)) startGame(room); }, 1500);
    }
    room.landlordSeat = room.currentBidder;
    room.multiplier = room.currentBid;
    room.players[room.landlordSeat].hand.push(...room.bottom);
    if (room.replay) {
      room.replay.landlordSeat = room.landlordSeat;
      room.replay.bottom = room.bottom.map(c => ({ id: c.id, rank: c.rank, suit: c.suit }));
    }
    room.phase = 'playing';
    room.turn = room.landlordSeat;
    broadcast(room, 'event', {
      event: `${room.players[room.landlordSeat].name} 成为地主 (底分 ${room.currentBid}), 底牌: ` +
        C.sortCards(room.bottom).map(c => (c.suit || '') + c.rank).join(' '),
    });
    sendState(room);
    scheduleBot(room);
    return;
  }
  room.turn = (seat + 1) % 3;
  sendState(room);
  scheduleBot(room);
}
function removeCards(hand, ids) {
  const set = new Set(ids);
  const played = hand.filter(c => set.has(c.id));
  if (played.length !== ids.length) return null;
  return { played, rest: hand.filter(c => !set.has(c.id)) };
}
function applyPlay(room, seat, ids) {
  const p = room.players[seat];
  const cards = ids.map(id => p.hand.find(c => c.id === id)).filter(Boolean);
  const ana = C.analyze(cards);
  const r = removeCards(p.hand, ids);
  p.hand = r.rest;
  p.lastIds = ids.slice();
  room.lastPlay = { ids: ids.slice(), cards, ana, seat };
  room.played.push(...cards);          // 累计本局已出的牌
  if (room.replay) {
    room.replay.actions.push({
      type: 'play', seat,
      ids: ids.slice(),
      cards: cards.map(c => ({ id: c.id, rank: c.rank, suit: c.suit })),
      typeName: ana.typeName,
      ana: { type: ana.type, main: ana.main === Infinity ? 'Infinity' : ana.main, len: ana.len },
      left: p.hand.length,
    });
  }
  room.passCount = 0;
  broadcast(room, 'event', { event: `${p.name} 出 ${ana.typeName}: ${C.sortCards(cards).map(c => (c.suit || '') + c.rank).join(' ')}` });
  if (ana.type === C.TYPE.BOMB || ana.type === C.TYPE.ROCKET) {
    room.multiplier *= 2;
    broadcast(room, 'event', { event: `💥 ${ana.typeName}! 倍数 ×2 → ${room.multiplier}` });
  }
  if (p.hand.length === 0) return endGame(room, seat);
  room.turn = (seat + 1) % 3;
  sendState(room);
  scheduleBot(room);
}
function applyPass(room, seat) {
  const p = room.players[seat];
  p.lastIds = [];
  if (room.replay) room.replay.actions.push({ type: 'pass', seat });
  room.passCount++;
  broadcast(room, 'event', { event: `${p.name} 不出` });
  room.turn = (seat + 1) % 3;
  if (room.passCount >= 2) {                // 两家 pass -> 最后出牌者重新领出
    room.turn = room.lastPlay.seat;
    room.lastPlay = null;
    room.passCount = 0;
    broadcast(room, 'event', { event: `${room.players[room.turn].name} 领出` });
  }
  sendState(room);
  scheduleBot(room);
}
function endGame(room, winnerSeat) {
  room.phase = 'done';
  const landlordWin = winnerSeat === room.landlordSeat;
  if (room.replay) {
    room.replay.winnerSeat = winnerSeat;
    room.replay.landlordWin = landlordWin;
    room.replay.finalMultiplier = room.multiplier;
    room.replay.finishedAt = new Date().toISOString();
    saveReplay(room);
  }
  const finalWinner = landlordWin ? room.landlordSeat : winnerSeat;
  room.players[finalWinner].wins++;
  // 积分: 地主赢则 +2*base*mult, 输则 -2*base*mult; 农民对称 ±base*mult
  const base = room.currentBid || 1;
  const stake = base * room.multiplier;
  const scores = room.players.map((p, s) => {
    const isLord = s === room.landlordSeat;
    const delta = landlordWin ? (isLord ? 2 * stake : -stake) : (isLord ? -2 * stake : stake);
    p.score = (p.score || 0) + delta;
    return { seat: s, name: p.name, delta, total: p.score, isLord };
  });
  // 结算公开所有人手牌
  const hands = room.players.map((p, s) => ({
    seat: s, name: p.name, isLord: s === room.landlordSeat,
    cards: C.sortCards(p.hand).map(c => ({ id: c.id, rank: c.rank, suit: c.suit })),
  }));
  broadcast(room, 'gameOver', {
    winnerSeat: finalWinner, winnerName: room.players[finalWinner].name,
    landlordSeat: room.landlordSeat, landlordWin,
    multiplier: room.multiplier, baseBid: base, scores, hands,
    event: `${room.players[finalWinner].name} (${landlordWin ? '地主' : '农民'}) 获胜! 倍数 ${room.multiplier}`,
  });
  room.turn = -1;
  sendState(room);
}

/* ---------------- 消息处理 ---------------- */
function err(conn, msg) { conn.send({ type: 'error', msg }); }

function onCreate(conn, d) {
  const room = newRoom();
  const p = { name: sanitize(d.name), conn, seat: 0, hand: [], wins: 0, lastIds: [] };
  room.players.push(p); room.hostSeat = 0;
  rooms.set(room.id, room);
  conn._room = room;
  conn.send({ type: 'joined', roomId: room.id, seat: 0 });
  sendState(room);
}
function onJoin(conn, d) {
  const room = rooms.get(String(d.roomId || '').trim());
  if (!room) return err(conn, '房间不存在');
  if (room.players.length >= 3) return err(conn, '房间已满');
  const seat = room.players.length;
  const p = { name: sanitize(d.name), conn, seat, hand: [], wins: 0, lastIds: [] };
  room.players.push(p);
  conn._room = room;
  conn.send({ type: 'joined', roomId: room.id, seat });
  broadcast(room, 'event', { event: `${p.name} 加入了房间` });
  sendState(room);
}
function onStart(conn) {
  const room = conn._room; if (!room) return;
  if (seatOf(room, conn) !== room.hostSeat) return err(conn, '只有房主可以开始');
  if (room.players.length !== 3) return err(conn, '需要 3 名玩家');
  if (room.phase === 'playing' || room.phase === 'bidding') return;
  startGame(room);
}
function onBid(conn, d) {
  const room = conn._room; if (!room || room.phase !== 'bidding') return;
  const seat = seatOf(room, conn);
  if (seat !== room.turn) return;
  const bid = parseInt(d.bid, 10);
  if (!(bid === 0 || bid > room.currentBid) || bid > 3) return err(conn, '无效叫分');
  applyBid(room, seat, bid);
}
function onPlay(conn, d) {
  const room = conn._room; if (!room || room.phase !== 'playing') return;
  const seat = seatOf(room, conn);
  if (seat !== room.turn) return err(conn, '还没轮到你');
  const p = room.players[seat];
  const ids = Array.isArray(d.ids) ? d.ids.slice(0, 54) : [];
  const handIds = new Set(p.hand.map(c => c.id));
  if (!ids.length || ids.some(id => !handIds.has(id))) return err(conn, '你没有这些牌');
  const cards = ids.map(id => p.hand.find(c => c.id === id));
  const ana = C.analyze(cards);
  if (ana.type === C.TYPE.INVALID) return err(conn, '无效牌型: ' + ana.typeName);
  const base = room.lastPlay && room.lastPlay.ana;
  if (!C.canBeat(base, ana)) return err(conn, `压不过上家的${base ? base.typeName : ''}`);
  applyPlay(room, seat, ids);
}
function onPass(conn) {
  const room = conn._room; if (!room || room.phase !== 'playing') return;
  const seat = seatOf(room, conn);
  if (seat !== room.turn) return;
  if (!room.lastPlay || room.lastPlay.seat === seat) return err(conn, '你必须出牌');
  applyPass(room, seat);
}
function onHint(conn) {
  const room = conn._room; if (!room || room.phase !== 'playing') return;
  const seat = seatOf(room, conn);
  if (seat !== room.turn) return;
  const p = room.players[seat];
  const ids = C.hintPlay(p.hand, room.lastPlay && room.lastPlay.seat !== seat ? room.lastPlay.ana : null);
  conn.send({ type: 'hint', ids: ids || [] });
}
function onChat(conn, d) {
  const room = conn._room; if (!room) return;
  const seat = seatOf(room, conn);
  broadcast(room, 'chat', { seat, name: room.players[seat].name, text: String(d.text || '').slice(0, 60) });
}

function onAddBot(conn) {
  const room = conn._room; if (!room) return;
  if (seatOf(room, conn) !== room.hostSeat) return err(conn, '只有房主可以添加 AI');
  if (room.phase !== 'lobby') return err(conn, '对局已开始');
  if (room.players.length >= 3) return err(conn, '房间已满');
  const p = addBotToRoom(room);
  broadcast(room, 'event', { event: `${p.name} 加入了房间` });
  sendState(room);
}
function onSolo(conn, d) {
  const room = newRoom();
  room.difficulty = ['normal', 'hard', 'expert', 'legend'].includes(d.diff) ? d.diff : 'normal';
  const p = { name: sanitize(d.name), conn, seat: 0, hand: [], wins: 0, lastIds: [] };
  room.players.push(p); room.hostSeat = 0;
  addBotToRoom(room); addBotToRoom(room);          // 座位 1、2 是两个 AI
  rooms.set(room.id, room);
  conn._room = room;
  conn.send({ type: 'joined', roomId: room.id, seat: 0 });
  broadcast(room, 'event', { event: '单人模式: 你 + 2 个 AI' });
  sendState(room);
  setTimeout(() => { if (rooms.has(room.id)) startGame(room); }, 800);
}
function onSetDiff(conn, d) {
  const room = conn._room; if (!room) return;
  if (seatOf(room, conn) !== room.hostSeat) return;
  if (['normal', 'hard', 'expert', 'legend'].includes(d.diff)) {
    room.difficulty = d.diff;
    const label = { normal: '普通', hard: '困难', expert: '大师', legend: '传说' }[d.diff];
    broadcast(room, 'event', { event: `AI 难度已设为 ${label}` });
    sendState(room);
  }
}
function onRooms(conn) {
  const list = [];
  for (const r of rooms.values()) {
    if (r.phase === 'lobby' && r.players.length < 3) {
      list.push({ id: r.id, count: r.players.length, names: r.players.map(p => p.name) });
    }
  }
  conn.send({ type: 'rooms', rooms: list });
}

const HANDLERS = { create: onCreate, join: onJoin, start: onStart, bid: onBid, play: onPlay, pass: onPass, hint: onHint, chat: onChat, rooms: onRooms, addbot: onAddBot, solo: onSolo, setdiff: onSetDiff };

attach(httpServer, (conn) => {
  attach.conns.add(conn);
  conn.on('message', (raw) => {
    let d;
    try { d = JSON.parse(raw); } catch (e) { return; }
    const h = HANDLERS[d && d.type];
    if (h) { try { h(conn, d); } catch (e) { console.error(e); } }
  });
  conn.on('close', () => {
    attach.conns.delete(conn);
    const room = conn._room;
    if (!room) return;
    const seat = seatOf(room, conn);
    if (seat >= 0) {
      room.players[seat].conn = null;
      broadcast(room, 'event', { event: `${room.players[seat].name} 断线` });
      sendState(room);
    }
    cleanupRoom(room);
  });
});

httpServer.listen(PORT, '0.0.0.0', () => {
  const nets = os.networkInterfaces();
  const ips = [];
  for (const list of Object.values(nets)) {
    for (const n of list || []) if (n.family === 'IPv4' && !n.internal) ips.push(n.address);
  }
  console.log(`斗地主服务端已启动 (零依赖)`);
  console.log(`  本机访问:   http://localhost:${PORT}`);
  console.log(`  主机名访问: http://${os.hostname()}:${PORT}  <- 公司内网一般有内部 DNS 可解析`);
  for (const ip of ips) console.log(`  内网IP访问: http://${ip}:${PORT}  <- 发给同事直接玩`);
});
