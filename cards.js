'use strict';
/**
 * 斗地主牌型判断 / 比较 / 提示
 * 牌: { id, rank, suit }  rank ∈ 3..2, SJ(小王), BJ(大王)
 */

const RANK_ORDER = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2', 'SJ', 'BJ'];
const RANK_VALUE = {};
RANK_ORDER.forEach((r, i) => { RANK_VALUE[r] = i; });
const V2 = RANK_VALUE['2'];
const SUITS = ['♠', '♥', '♣', '♦'];

const TYPE = {
  INVALID: 0, SINGLE: 1, PAIR: 2, TRIPLE: 3, TRIPLE_1: 4, TRIPLE_2: 5,
  CHAIN: 6, PAIR_CHAIN: 7, AIRPLANE: 8, AIRPLANE_1: 9, AIRPLANE_2: 10,
  FOUR_2: 11, FOUR_2_2: 12, BOMB: 13, ROCKET: 14,
};
const TYPE_NAMES = {
  1: '单张', 2: '对子', 3: '三张', 4: '三带一', 5: '三带二',
  6: '顺子', 7: '连对', 8: '飞机', 9: '飞机带单', 10: '飞机带对',
  11: '四带二', 12: '四带二对', 13: '炸弹', 14: '王炸', 0: '无效牌型',
};

function createDeck() {
  const deck = [];
  let id = 0;
  for (const suit of SUITS) {
    for (const rank of RANK_ORDER) {
      if (rank === 'SJ' || rank === 'BJ') continue;
      deck.push({ id: id++, rank, suit });
    }
  }
  deck.push({ id: id++, rank: 'SJ', suit: '' });   // 小王
  deck.push({ id: id++, rank: 'BJ', suit: '' });   // 大王
  return deck;
}

function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function val(card) { return RANK_VALUE[card.rank]; }

function sortCards(cards) {
  return cards.slice().sort((a, b) => val(b) - val(a) || a.id - b.id);
}

/** 把 id 数组转为按点数分组 {value: [card,...]} */
function groupByVal(cards) {
  const g = new Map();
  for (const c of cards) {
    const v = val(c);
    if (!g.has(v)) g.set(v, []);
    g.get(v).push(c);
  }
  return g;
}

/** 判断 vals(升序)是否连续 */
function isConsecutive(vals) {
  for (let i = 1; i < vals.length; i++) {
    if (vals[i] !== vals[i - 1] + 1) return false;
  }
  return true;
}

/**
 * 分析一组牌 -> { type, typeName, len, main }
 * main: 用于比较大小的主牌点数(火箭为 Infinity)
 */
function analyze(cards) {
  const bad = { type: TYPE.INVALID, typeName: TYPE_NAMES[0], len: cards ? cards.length : 0, main: -1 };
  if (!cards || cards.length === 0) return bad;

  const n = cards.length;
  const g = groupByVal(cards);
  const keys = [...g.keys()].sort((a, b) => a - b);   // 升序点数
  const counts = keys.map(k => g.get(k).length);
  const ok = (type, main) => ({ type, typeName: TYPE_NAMES[type], len: n, main });

  // 王炸
  if (n === 2 && g.has(RANK_VALUE['SJ']) && g.has(RANK_VALUE['BJ'])) {
    return ok(TYPE.ROCKET, Infinity);
  }
  if (n === 1) return ok(TYPE.SINGLE, keys[0]);
  // 炸弹
  if (n === 4 && keys.length === 1) return ok(TYPE.BOMB, keys[0]);

  // 同一点数
  if (keys.length === 1) {
    if (n === 2 && keys[0] < RANK_VALUE['SJ']) return ok(TYPE.PAIR, keys[0]);
    if (n === 3) return ok(TYPE.TRIPLE, keys[0]);
    return bad;
  }

  const tripleVals = keys.filter(k => g.get(k).length === 3);
  const fourVals = keys.filter(k => g.get(k).length === 4);

  // 三带一 / 三带二(一对)
  if (tripleVals.length === 1 && fourVals.length === 0) {
    const t = tripleVals[0];
    const rest = keys.filter(k => k !== t);
    if (n === 4) return ok(TYPE.TRIPLE_1, t);
    if (n === 5 && rest.length === 1 && g.get(rest[0]).length === 2) return ok(TYPE.TRIPLE_2, t);
  }

  // 四带二 / 四带二对
  if (fourVals.length === 1) {
    const f = fourVals[0];
    const rest = keys.filter(k => k !== f);
    if (n === 6 && rest.reduce((s, k) => s + g.get(k).length, 0) === 2) return ok(TYPE.FOUR_2, f);
    if (n === 8 && rest.length === 2 && rest.every(k => g.get(k).length === 2)) return ok(TYPE.FOUR_2_2, f);
    return bad;
  }

  // 顺子: >=5 单连续, 不含 2 和王
  if (n >= 5 && keys.every(k => g.get(k).length === 1) && keys[keys.length - 1] < V2 && isConsecutive(keys)) {
    return ok(TYPE.CHAIN, keys[keys.length - 1]);
  }
  // 连对: >=3 对连续
  if (n >= 6 && keys.every(k => g.get(k).length === 2) && keys.length >= 3 && keys[keys.length - 1] < V2 && isConsecutive(keys)) {
    return ok(TYPE.PAIR_CHAIN, keys[keys.length - 1]);
  }
  // 飞机(不带)
  if (n >= 6 && keys.every(k => g.get(k).length === 3) && keys.length >= 2 && keys[keys.length - 1] < V2 && isConsecutive(keys)) {
    return ok(TYPE.AIRPLANE, keys[keys.length - 1]);
  }
  // 飞机带翅膀: 连续三张 m 组 + m 张单 / m 对
  /* 翅膀自身可能是三张同点 (如 333777888999 = 飞机 777888999 带 333),
     所以不能只认"全部三张都连续". 枚举连续三张段作为主体, 剩下的
     牌只要凑够 m 张单 (且不含王炸歧义) 或 m 对就算飞机带牌. */
  {
    const cand = tripleVals.filter(v => v < V2);
    for (let end = 0; end < cand.length; end++) {
      for (let start = end - 1; start >= 0; start--) {
        const seg = cand.slice(start, end + 1);
        if (!isConsecutive(seg)) break;            // 段不连续, 再往前也不会连续
        const m = seg.length;
        if (m < 2) continue;
        const rest = keys.filter(k => !seg.includes(k));
        const restCount = rest.reduce((s, k) => s + g.get(k).length, 0);
        if (restCount === m) return ok(TYPE.AIRPLANE_1, seg[m - 1]);
        if (rest.length === m && rest.every(k => g.get(k).length === 2)) return ok(TYPE.AIRPLANE_2, seg[m - 1]);
      }
    }
  }
  return bad;
}

/** prev 为 analyze 结果(上家出的牌); next 能否压过 */
function canBeat(prev, next) {
  if (next.type === TYPE.INVALID) return false;
  if (!prev) return true;
  if (next.type === TYPE.ROCKET) return true;
  if (prev.type === TYPE.ROCKET) return false;
  if (next.type === TYPE.BOMB && prev.type !== TYPE.BOMB) return true;
  if (next.type !== prev.type) return false;
  if (next.len !== prev.len) return false;
  return next.main > prev.main;
}

/* ---------------- 提示 (找能压过 base 的最小牌) ---------------- */

function lowestCardsOf(g, v, count, exclude) {
  const cs = g.get(v) || [];
  const avail = cs.filter(c => !exclude || !exclude.has(c.id));
  return avail.length >= count ? avail.slice(0, count) : null;
}

/** 在手牌中找能压过 base 的牌, 返回 id 数组或 null */
function hintPlay(hand, base) {
  const g = groupByVal(hand);
  const allVals = [...g.keys()].sort((a, b) => a - b);        // 含 2 和王
  const vals = allVals.filter(v => v < V2);                    // 普通牌点(升序, 顺子等用)
  const min = base ? base.main : -1;

  const find = (pred, take) => {
    for (const v of vals) {
      if (v <= min) continue;
      if (!pred(v)) continue;
      const r = take(v);
      if (r) return r;
    }
    return null;
  };
  /**
   * 按"张数优先"找能压的点: 先不拆结构 (exact), 再拆更大的结构.
   * prefer: [1,2,3] 单张依次找 只有1张的点 → 对子拆 → 三张拆;
   * 炸弹点(4张)默认不拆, 除非 allowBombBreak
   */
  const pickByCount = (need, prefs, allowBombBreak) => {
    for (const cnt of prefs) {
      for (const v of allVals) {
        if (v <= min) continue;
        if (need >= 2 && v >= RANK_VALUE['SJ']) break;   // 对子以上不含王
        if (g.get(v).length !== cnt) continue;
        const r = g.get(v).slice(0, need).map(c => c.id);
        if (r.length === need) return r;
      }
    }
    if (allowBombBreak) {
      for (const v of allVals) {
        if (v <= min) continue;
        if (need >= 2 && v >= RANK_VALUE['SJ']) break;
        if (g.get(v).length === 4) {
          const r = g.get(v).slice(0, need).map(c => c.id);
          if (r.length === need) return r;
        }
      }
    }
    return null;
  };
  const lowSingles = (n, exclude, excludeVals) => {
    const out = [];
    for (const v of allVals) {
      if (excludeVals && excludeVals.has(v)) continue;
      for (const c of g.get(v)) {
        if (exclude && exclude.has(c.id)) continue;
        out.push(c.id);
        if (out.length === n) return out;
      }
    }
    return null;
  };
  const lowPair = (exclude, excludeVals) => {
    for (const v of vals) {
      if (excludeVals && excludeVals.has(v)) continue;
      const cs = lowestCardsOf(g, v, 2, exclude);
      if (cs) return cs.map(c => c.id);
    }
    return null;
  };
  const chainWindow = (needCount, len) => {
    const elig = vals.filter(v => (g.get(v) || []).length >= needCount);
    for (let i = 0; i + len <= elig.length; i++) {
      const win = elig.slice(i, i + len);
      if (!isConsecutive(win)) continue;
      if (win[len - 1] <= min) continue;
      const ids = [];
      for (const v of win) for (const c of g.get(v).slice(0, needCount)) ids.push(c.id);
      return ids;
    }
    return null;
  };

  let res = null;
  const t = base ? base.type : TYPE.SINGLE;
  switch (t) {
    case TYPE.SINGLE:
      /* 散牌 → 拆对 → 拆三张; 2/王也作为单张参与 */
      res = pickByCount(1, [1, 2, 3], true); break;
    case TYPE.PAIR:
      /* 对子 → 拆三张; 含对2 (v===V2) */
      res = pickByCount(2, [2, 3], true); break;
    case TYPE.TRIPLE:
      /* 三张 → 拆四张; 含 222 */
      res = pickByCount(3, [3], true); break;
    case TYPE.TRIPLE_1: {
      for (const cnt of [3, 4]) {
        for (const v of allVals) {
          if (v <= min || v >= RANK_VALUE['SJ'] || g.get(v).length !== cnt) continue;
          const tri = g.get(v).slice(0, 3).map(c => c.id);
          const s = lowSingles(1, new Set(tri), new Set([v]));
          if (s) { res = tri.concat(s); break; }
        }
        if (res) break;
      }
      break;
    }
    case TYPE.TRIPLE_2: {
      for (const cnt of [3, 4]) {
        for (const v of allVals) {
          if (v <= min || v >= RANK_VALUE['SJ'] || g.get(v).length !== cnt) continue;
          const tri = g.get(v).slice(0, 3).map(c => c.id);
          const p = lowPair(new Set(tri), new Set([v]));
          if (p) { res = tri.concat(p); break; }
        }
        if (res) break;
      }
      break;
    }
    case TYPE.CHAIN:
      res = chainWindow(1, base.len); break;
    case TYPE.PAIR_CHAIN:
      res = chainWindow(2, base.len / 2); break;
    case TYPE.AIRPLANE:
      res = chainWindow(3, base.len / 3); break;
    case TYPE.AIRPLANE_1: {
      const m = base.len / 4;
      res = chainWindow(3, m);
      if (res) {
        /* 翅膀避开主体点数, 否则牌型非法 */
        const bodyVals = new Set();
        for (const id of res) bodyVals.add(RANK_VALUE[hand.find(c => c.id === id).rank]);
        const s = lowSingles(m, new Set(res), bodyVals);
        res = s ? res.concat(s) : null;
      }
      break;
    }
    case TYPE.AIRPLANE_2: {
      const m = base.len / 5;
      res = chainWindow(3, m);
      if (res) {
        const ex = new Set(res);
        const exv = new Set();
        for (const id of res) exv.add(RANK_VALUE[hand.find(c => c.id === id).rank]);
        const wings = [];
        for (const v of vals) {
          if (wings.length === m * 2) break;
          if (exv.has(v)) continue;
          const cs = lowestCardsOf(g, v, 2, ex);
          if (cs) { for (const c of cs) { wings.push(c.id); ex.add(c.id); } }
        }
        res = wings.length === m * 2 ? res.concat(wings) : null;
      }
      break;
    }
    case TYPE.FOUR_2:
      res = find(v => g.get(v).length === 4, v => {
        const four = g.get(v).map(c => c.id);
        const s = lowSingles(2, new Set(four), new Set([v]));
        return s ? four.concat(s) : null;
      });
      break;
    case TYPE.FOUR_2_2:
      res = find(v => g.get(v).length === 4, v => {
        const four = g.get(v).map(c => c.id);
        const ex = new Set(four);
        const wings = [];
        for (const vv of vals) {
          if (vv === v) continue;
          const cs = lowestCardsOf(g, vv, 2, ex);
          if (cs) { for (const c of cs) { wings.push(c.id); ex.add(c.id); } }
          if (wings.length === 4) break;
        }
        return wings.length === 4 ? four.concat(wings) : null;
      });
      break;
    case TYPE.BOMB:
      for (const v of allVals) {           // 含 2222 炸弹
        if (v > min && g.get(v).length === 4) { res = g.get(v).map(c => c.id); break; }
      }
      break;
    default:
      res = null;
  }
  // 找不到同型 -> 找炸弹 (含 2222) -> 王炸
  if (!res && (!base || (base.type !== TYPE.BOMB && base.type !== TYPE.ROCKET))) {
    for (const v of allVals) if (g.get(v).length === 4) { res = g.get(v).map(c => c.id); break; }
  }
  if (!res && base && base.type === TYPE.BOMB) {
    for (const v of allVals) if (v > min && g.get(v).length === 4) { res = g.get(v).map(c => c.id); break; }
  }
  if (!res && g.has(RANK_VALUE['SJ']) && g.has(RANK_VALUE['BJ'])) {
    res = [g.get(RANK_VALUE['SJ'])[0].id, g.get(RANK_VALUE['BJ'])[0].id];
  }
  if (!res && !base) res = [sortCards(hand)[hand.length - 1].id]; // 自由出牌提示最小单张
  return res;
}

module.exports = {
  RANK_VALUE, SUITS, TYPE, TYPE_NAMES,
  createDeck, shuffle, sortCards, analyze, canBeat, hintPlay,
};
