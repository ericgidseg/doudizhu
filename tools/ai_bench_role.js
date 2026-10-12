'use strict';
/**
 * 分角色基准: 把"地主控场"和"农民配合"分开测, 信噪比远高于混合指标.
 *
 *   node tools/ai_bench_role.js [deals] [farmer|lord] [aiA] [aiB] [opponent]
 *
 * farmer: 地主固定为 opponent, 农民分别用 A / B 打同一副牌, 比较农民胜局.
 * lord:   两个农民固定为 opponent, 地主分别用 A / B 打同一副牌, 比较地主胜局.
 * 同一副牌配对比较, 消除发牌方差; awins/bwins 是"A 胜 / B 胜"的局数.
 */
const path = require('path');
const fs = require('fs');
const vm = require('vm');
const Module = require('module');
const C = require(path.join(__dirname, '..', 'cards.js'));
const AI = {
  new: require(path.join(__dirname, '..', 'ai.js')),
  baseline: require(path.join(__dirname, '..', 'ai_baseline.js')),
};
function loadAI(spec) {
  if (AI[spec]) return AI[spec];
  if (/[\\/]/.test(spec) || spec.endsWith('.js')) {
    const p = path.isAbsolute(spec) ? spec : path.resolve(process.cwd(), spec);
    const src = fs.readFileSync(p, 'utf8');
    const mod = new Module(p, null);
    mod.filename = p;
    mod.paths = Module._nodeModulePaths(path.dirname(p));
    const customRequire = (id) => {
      if (id === './cards.js' || id === '../cards.js') return C;
      return Module.createRequire(p)(id);
    };
    const fn = vm.runInThisContext(Module.wrap(src), { filename: p });
    fn.call(mod.exports, mod.exports, customRequire, mod, p, path.dirname(p));
    return mod.exports;
  }
  return null;
}

function mkRoom(hands, lord, bottom) {
  return {
    id: 1, phase: 'playing',
    players: hands.map((h, i) => ({ name: 'p' + i, seat: i, hand: h.slice(), wins: 0, lastIds: [], bot: true, conn: null })),
    landlordSeat: lord, turn: lord, currentBid: 3, multiplier: 3,
    bids: [3, -1, -1], passCount: 0, lastPlay: null, played: [], bottom: bottom || [],
  };
}

function playDeal(deck, lord, ais) {
  const hands = [deck.slice(0, 17), deck.slice(17, 34), deck.slice(34, 51)];
  const bottom = deck.slice(51, 54);
  hands[lord] = hands[lord].concat(bottom);
  const room = mkRoom(hands, lord, bottom);
  let steps = 0;
  while (steps++ < 300) {
    const seat = room.turn, p = room.players[seat];
    const act = ais[seat].decide(room, seat);
    if (act.type === 'play') {
      const ids = act.ids || [];
      const cards = p.hand.filter(c => ids.includes(c.id));
      const ana = C.analyze(cards);
      const legal = cards.length === ids.length && ana.type !== C.TYPE.INVALID &&
        (!room.lastPlay || room.lastPlay.seat === seat || C.canBeat(room.lastPlay.ana, ana));
      if (!legal) return { illegal: true };
      p.hand = p.hand.filter(c => !ids.includes(c.id));
      room.lastPlay = { ids: ids.slice(), cards, ana, seat };
      room.played.push(...cards);
      room.passCount = 0;
      if (!p.hand.length) return { lordWon: seat === lord };
    } else if (act.type === 'pass') {
      if (!room.lastPlay || room.lastPlay.seat === seat) return { illegal: true };
      room.passCount++;
      if (room.passCount >= 2) { room.turn = room.lastPlay.seat; room.lastPlay = null; room.passCount = 0; continue; }
    } else return { illegal: true };
    room.turn = (seat + 1) % 3;
  }
  return { timeout: true };
}

function detShuffle(deck, seed) {
  let a = seed;
  for (let i = deck.length - 1; i > 0; i--) {
    a = (a * 1103515245 + 12345) & 0x7fffffff;
    const j = a % (i + 1);
    [deck[i], deck[j]] = [deck[j], deck[i]];
  }
  return deck;
}

const N = parseInt(process.argv[2] || '200', 10);
const ROLE = process.argv[3] || 'farmer';
const SEED_OFF = parseInt(process.env.SEEDOFF || '0', 10);
const A = loadAI(process.argv[4] || 'new');
const B = loadAI(process.argv[5] || 'baseline');
const OPP = loadAI(process.argv[6] || path.join(__dirname, 'archive', 'ai_preveto.js'));
if (!A || !B || !OPP) { console.error('usage: node tools/ai_bench_role.js [deals] [farmer|lord] [aiA] [aiB] [opponent]'); process.exit(1); }

let aw = 0, bw = 0, same = 0, illegal = 0, timeout = 0;
const t0 = Date.now();
for (let i = 1; i <= N; i++) {
  for (const lord of [0, 1, 2]) {
    const base = C.createDeck().map((c, j) => Object.assign({}, c, { id: j + 1000 }));
    const seed = i * 104729 + lord * 3 + SEED_OFF;
    const r1 = ROLE === 'lord'
      ? playDeal(detShuffle(base.slice(), seed), lord, [0, 1, 2].map(s => (s === lord ? A : OPP)))
      : playDeal(detShuffle(base.slice(), seed), lord, [0, 1, 2].map(s => (s === lord ? OPP : A)));
    const r2 = ROLE === 'lord'
      ? playDeal(detShuffle(base.slice(), seed), lord, [0, 1, 2].map(s => (s === lord ? B : OPP)))
      : playDeal(detShuffle(base.slice(), seed), lord, [0, 1, 2].map(s => (s === lord ? OPP : B)));
    if (r1.illegal || r2.illegal) { illegal++; continue; }
    if (r1.timeout || r2.timeout) { timeout++; continue; }
    /* 统一换算成"A 所在阵营是否赢": lord 模式 A 是地主, farmer 模式 A 是两个农民 */
    const aWin1 = ROLE === 'lord' ? r1.lordWon : !r1.lordWon;
    const aWin2 = ROLE === 'lord' ? r2.lordWon : !r2.lordWon;
    if (aWin1 && !aWin2) aw++;
    else if (!aWin1 && aWin2) bw++;
    else same++;   // 两边同胜/同负: 这副牌不区分两者, 不计入胜负
  }
}
console.log(JSON.stringify({ deals: N, role: ROLE, games: N * 3 * 2, awins: aw, bwins: bw, same, illegal, timeout, ms: Date.now() - t0 }));
