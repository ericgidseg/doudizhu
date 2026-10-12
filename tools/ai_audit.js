'use strict';
/**
 * 出牌审计: 跑 N 副牌, 统计"明显违背牌理"的决策次数 (不是胜率).
 * 用于验证规则类修复是否真的把某类错误打法压下去.
 *
 *   node tools/ai_audit.js [deals] [ai文件] [seedoff]
 *
 * 目前统计:
 *   leadBombWaste  领出时甩炸/王炸, 但炸完还剩 >2 手且对手不临门 (空炸)
 *   leadBombFine   领出甩炸但属于正常甩牌 (炸完剩 ≤2 手 / 对手临门 / 炸完走完)
 *   leadBombHist   领炸时"炸完剩余张数 -> 剩余手数"直方图 (观察行为变化)
 *   leadControl    手牌还多 (余 >6 张且分解 >4 手) 时领出单张 2/小王/大王
 */
const path = require('path');
const C = require(path.join(__dirname, '..', 'cards.js'));
const { loadAI } = require(path.join(__dirname, '_load_ai.js'));
const AI = loadAI(process.argv[3] || 'new');
if (!AI) { console.error('usage: node tools/ai_audit.js [deals] [new|baseline|<ai路径>] [seedoff]'); process.exit(1); }

const V = r => C.RANK_VALUE[r];

function mkRoom(hands, lord, bottom) {
  return {
    id: 1, phase: 'playing',
    players: hands.map((h, i) => ({ name: 'p' + i, seat: i, hand: h.slice(), wins: 0, lastIds: [], bot: true, conn: null })),
    landlordSeat: lord, turn: lord, currentBid: 3, multiplier: 3,
    bids: [3, -1, -1], passCount: 0, lastPlay: null, played: [], bottom: bottom || [],
  };
}

/** 炸完剩几手 (与 ai.js 领炸 veto 同一判据) */
function handsLeft(hand) {
  return AI.bestDecomp(hand).count;
}

const N = parseInt(process.argv[2] || '200', 10);
const SEED_OFF = parseInt(process.argv[4] || '0', 10);
const stats = {
  deals: 0, decisions: 0, leadBombWaste: 0, leadBombFine: 0,
  leadControl: 0, samples: [], leadBombHist: {}, leadControlSamples: [],
};

for (let i = 1; i <= N; i++) {
  for (const lord of [0, 1, 2]) {
    const deck = C.createDeck().map((c, j) => Object.assign({}, c, { id: j + 1000 }));
    /* 与 ai_bench 相同的确定性洗牌, 保证可复现 */
    let a = i * 104729 + lord * 3 + SEED_OFF;
    for (let k = deck.length - 1; k > 0; k--) {
      a = (a * 1103515245 + 12345) & 0x7fffffff;
      const j = a % (k + 1);
      [deck[k], deck[j]] = [deck[j], deck[k]];
    }
    const hands = [deck.slice(0, 17), deck.slice(17, 34), deck.slice(34, 51)];
    const bottom = deck.slice(51, 54);
    hands[lord] = hands[lord].concat(bottom);
    const room = mkRoom(hands, lord, bottom);
    let steps = 0, ok = true;
    while (steps++ < 300) {
      const seat = room.turn, p = room.players[seat];
      const leading = !room.lastPlay || room.lastPlay.seat === seat;
      const act = AI.decide(room, seat);
      stats.decisions++;
      if (act.type === 'play') {
        const ids = act.ids || [];
        const cards = p.hand.filter(c => ids.includes(c.id));
        const ana = C.analyze(cards);
        if (ana.type === C.TYPE.INVALID || cards.length !== ids.length ||
            (!leading && !C.canBeat(room.lastPlay.ana, ana))) { ok = false; break; }
        const isBomb = ana.type === C.TYPE.BOMB || ana.type === C.TYPE.ROCKET;
        /* 领出单张 2/王: 手牌还多时属于白烧控场牌 */
        if (leading && ana.type === C.TYPE.SINGLE && ana.main >= V('2') &&
            p.hand.length > 6 && handsLeft(p.hand) > 4) {
          stats.leadControl++;
          if (stats.leadControlSamples.length < 8) {
            stats.leadControlSamples.push(`deal${i}-${lord} 步${steps} seat${seat}` +
              `${seat === lord ? '(地主)' : '(农民)'} 手牌${p.hand.length} ` +
              `${C.sortCards(p.hand).map(c => c.rank).join(' ')}`);
          }
        }
        if (leading && isBomb) {
          const rest = p.hand.filter(c => !ids.includes(c.id));
          const oppNear = room.players.some((q, s) => s !== seat && q.hand.length > 0 && q.hand.length <= 4);
          const left = handsLeft(rest);
          const hk = `rest${rest.length}:left${left}`;
          stats.leadBombHist[hk] = (stats.leadBombHist[hk] || 0) + 1;
          if (rest.length <= 2 || oppNear || left > 2) stats.leadBombFine++;
          else {
            stats.leadBombWaste++;
            if (stats.samples.length < 8) {
              stats.samples.push(`deal${i}-${lord} 步${steps} seat${seat} ${ana.typeName} ${ana.main} ` +
                `手牌${p.hand.length} 余[${C.sortCards(rest).map(c => c.rank).join(' ')}]`);
            }
          }
        }
        p.hand = p.hand.filter(c => !ids.includes(c.id));
        room.lastPlay = { ids: ids.slice(), cards, ana, seat };
        room.played.push(...cards);
        room.passCount = 0;
        if (!p.hand.length) break;
      } else if (act.type === 'pass') {
        if (leading) { ok = false; break; }
        room.passCount++;
        if (room.passCount >= 2) { room.turn = room.lastPlay.seat; room.lastPlay = null; room.passCount = 0; continue; }
      } else { ok = false; break; }
      room.turn = (seat + 1) % 3;
    }
    if (ok) stats.deals++;
  }
}
console.log(JSON.stringify(stats, null, 2));
