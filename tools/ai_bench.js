'use strict';
/**
 * AI 对局基准: 同一副牌让两套 AI 各当一次地主, 配对比较消除发牌方差.
 *
 *   node tools/ai_bench.js [deals] [aiA] [aiB]
 *     deals 每副牌打两局(各当地主一次), 默认 200
 *     aiA/aiB 取 new | baseline | <任意 ai 文件路径>, 默认 new baseline
 *
 * 输出: awins(aiA 双胜) / bwins(aiB 双胜) / split(各胜一局) / illegal.
 * 合法性与 300 步上限都会检查, illegal 必须为 0.
 */
const path = require('path');
const C = require(path.join(__dirname, '..', 'cards.js'));
/* 归档/快照文件可能放在任意目录, _load_ai 统一把它们的 require('./cards.js')
   指向项目根, 让任意位置的 AI 文件都能直接加载. */
const { loadAI } = require(path.join(__dirname, '_load_ai.js'));

function mkRoom(hands, lord, bottom) {
  return {
    id: 1, phase: 'playing',
    players: hands.map((h, i) => ({ name: 'p' + i, seat: i, hand: h.slice(), wins: 0, lastIds: [], bot: true, conn: null })),
    landlordSeat: lord, turn: lord, currentBid: 3, multiplier: 3,
    bids: [3, -1, -1], passCount: 0, lastPlay: null, played: [], bottom: bottom || [],
  };
}

function playDeal(deck, lord, aiA, aiB) {
  /* 与真实发牌一致: 每人 17 张 + 3 张底牌归地主 (底牌是公开信息) */
  const hands = [deck.slice(0, 17), deck.slice(17, 34), deck.slice(34, 51)];
  const bottom = deck.slice(51, 54);
  hands[lord] = hands[lord].concat(bottom);
  const ais = [0, 1, 2].map(i => (i === lord ? aiA : aiB));
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

/* 确定性洗牌: 同一 seed 得到同一副牌, 便于复现与回归对比 */
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
/* 换一批牌局复核时用 SEEDOFF 错开 (例如 SEEDOFF=5000000), 避免只对
   固定的一批牌过拟合. */
const SEED_OFF = parseInt(process.env.SEEDOFF || '0', 10);
const A = loadAI(process.argv[3] || 'new');
const B = loadAI(process.argv[4] || 'baseline');
if (!A || !B) { console.error('usage: node tools/ai_bench.js [deals] [new|baseline|<path>] [new|baseline|<path>]'); process.exit(1); }

let aw = 0, bw = 0, split = 0, illegal = 0, timeout = 0;
const t0 = Date.now();
for (let i = 1; i <= N; i++) {
  for (const lord of [0, 1, 2]) {
    const base = C.createDeck().map((c, j) => Object.assign({}, c, { id: j + 1000 }));
    const seed = i * 104729 + lord * 3 + SEED_OFF;
    const r1 = playDeal(detShuffle(base.slice(), seed), lord, A, B);
    const r2 = playDeal(detShuffle(base.slice(), seed), lord, B, A);
    if (r1.illegal || r2.illegal) { illegal++; continue; }
    if (r1.timeout || r2.timeout) { timeout++; continue; }
    // r1: A 当 landlord; r2: B 当 landlord. 统一换算成"A 所在阵营是否赢"
    const aWin1 = r1.lordWon, aWin2 = !r2.lordWon;
    if (aWin1 && aWin2) aw++;
    else if (!aWin1 && !aWin2) bw++;
    else split++;
  }
}
console.log(JSON.stringify({ deals: N, games: N * 3, awins: aw, bwins: bw, split, illegal, timeout, ms: Date.now() - t0 }));
