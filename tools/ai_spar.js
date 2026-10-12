'use strict';
/**
 * 同局面对拍: 从真实复盘 (replays/*.json) 里逐个提取"人类/AI 实际打过的
 * 决策点", 让两版 AI 在同一局面上各决策一次, 统计分歧率与"谁更接近人类
 * 实战选择". 相比对局级基准, 这个指标噪声小得多, 适合判断规则改动方向.
 *
 *   node tools/ai_spar.js [dir...] [aiA] [aiB]
 *     默认 dir = replays, A=new, B=上一个归档版本
 *   环境变量: SPAR_GROUP=human|bot 只统计/输出该组; SPAR_MAX=样例上限(默认 25)
 *
 * 输出 (决策点按出牌者身份分成 human / bot 两组):
 *   points   决策点总数
 *   differ   两版选择不同的比例
 *   aSame/bSame 各自与复盘里实际选择一致的比例
 *     - human 组: 真实人类玩家的选择 (弱标签, 参考 AI 是否"像高手")
 *     - bot 组: 上一版 AI 的选择 (回归指标, 大幅偏离说明行为漂移)
 *   examples 分歧样例 (局面 + 双方选择 + 实战选择)
 */
const path = require('path');
const fs = require('fs');
const C = require(path.join(__dirname, '..', 'cards.js'));
const { loadAI } = require(path.join(__dirname, '_load_ai.js'));

const V = r => C.RANK_VALUE[r];

function buildRoom(replay, upTo) {
  const hands = replay.players.map(p => p.hand.slice());
  hands[replay.landlordSeat] = hands[replay.landlordSeat].concat(replay.bottom || []);
  const room = {
    id: replay.roomId || 'r', phase: 'playing',
    players: replay.players.map((p, i) => ({
      name: p.name, seat: i, hand: hands[i].slice(), wins: 0, lastIds: [], bot: true, conn: null,
    })),
    landlordSeat: replay.landlordSeat, turn: 0, currentBid: 3,
    multiplier: replay.multiplier || 3, bids: [3, -1, -1],
    passCount: 0, lastPlay: null, played: [], bottom: (replay.bottom || []).slice(),
  };
  let pass = 0;
  for (let i = 0; i < upTo; i++) {
    const a = replay.actions[i];
    if (!a || a.type === 'bid') continue;
    if (a.type === 'play') {
      room.players[a.seat].hand = room.players[a.seat].hand.filter(c => !a.ids.includes(c.id));
      room.played.push(...a.cards);
      room.lastPlay = { seat: a.seat, cards: a.cards, ana: C.analyze(a.cards), ids: a.ids };
      pass = 0;
    } else {
      pass++;
      if (pass >= 2) { room.lastPlay = null; pass = 0; }
    }
  }
  return room;
}

function describe(hand, ids) {
  if (!ids || !ids.length) return 'pass';
  const cards = hand.filter(c => ids.includes(c.id));
  const a = C.analyze(cards);
  return `${a.typeName}(${C.sortCards(cards).map(c => c.rank).join('')})`;
}

const dirs = [];
const args = process.argv.slice(2);
const onlyGroup = process.env.SPAR_GROUP || '';
const maxExamples = parseInt(process.env.SPAR_MAX || '25', 10);
let aiSpecs = [];
for (const arg of args) {
  if (/\.js$/.test(arg) || arg === 'new' || arg === 'baseline') aiSpecs.push(arg);
  else dirs.push(arg);
}
if (!dirs.length) dirs.push(path.join(__dirname, '..', 'replays'));
if (!aiSpecs.length) aiSpecs = ['new', path.join(__dirname, 'archive', 'ai_preveto.js')];
const A = loadAI(aiSpecs[0]);
const B = loadAI(aiSpecs[1]);
if (!A || !B) { console.error('usage: node tools/ai_spar.js [dir...] [aiA] [aiB]'); process.exit(1); }

const files = dirs.flatMap(d => {
  try { return fs.readdirSync(d).filter(f => f.endsWith('.json')).map(f => path.join(d, f)); }
  catch (e) { return []; }
});

const st = {
  files: files.length,
  points: 0, differ: 0, aPlay: 0, bPlay: 0, examples: [],
  human: { points: 0, differ: 0, aSame: 0, bSame: 0 },
  bot: { points: 0, differ: 0, aSame: 0, bSame: 0 },
};
for (const f of files) {
  let replay;
  try { replay = JSON.parse(fs.readFileSync(f, 'utf8')); } catch (e) { continue; }
  if (!replay.actions) continue;
  for (let i = 0; i < replay.actions.length; i++) {
    const act = replay.actions[i];
    if (act.type !== 'play' && act.type !== 'pass') continue;
    const room = buildRoom(replay, i);
    const seat = act.seat;
    const isHuman = !(replay.players[seat] && replay.players[seat].bot);
    const group = isHuman ? st.human : st.bot;
    /* 实战选择 (人类玩家 或 上一版 AI) */
    const humanIds = act.type === 'play' ? act.ids : null;
    const ka = A.decide(room, seat);
    const kb = B.decide(room, seat);
    const aIds = ka.type === 'play' ? ka.ids : null;
    const bIds = kb.type === 'play' ? kb.ids : null;
    const key = ids => (ids ? ids.slice().sort((x, y) => x - y).join(',') : 'pass');
    st.points++;
    group.points++;
    const differ = key(aIds) !== key(bIds);
    if (differ) { st.differ++; group.differ++; }
    if (aIds) st.aPlay++;
    if (bIds) st.bPlay++;
    const hk = key(humanIds);
    const aHit = key(aIds) === hk, bHit = key(bIds) === hk;
    if (aHit) group.aSame++;
    if (bHit) group.bSame++;
    if (differ && st.examples.length < maxExamples && (!onlyGroup || onlyGroup === (isHuman ? 'human' : 'bot'))) {
      const hand = room.players[seat].hand;
      const lead = !room.lastPlay || room.lastPlay.seat === seat;
      st.examples.push({
        file: path.basename(f), step: i, seat,
        role: seat === replay.landlordSeat ? 'lord' : 'farmer',
        lead: lead ? 'lead' : `beat ${room.lastPlay.ana.typeName}`,
        left: hand.length,
        hand: C.sortCards(hand).map(c => c.rank).join(' '),
        by: isHuman ? 'human' : 'oldAI',
        a: describe(hand, aIds), b: describe(hand, bIds), actual: describe(hand, humanIds),
        aHit, bHit,
      });
    }
  }
}
console.log(JSON.stringify(st, null, 2));
