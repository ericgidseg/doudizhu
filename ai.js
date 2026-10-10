'use strict';
/**
 * AI 决策 —— 三难度
 *   normal: 基础最小压牌 + 农民让牌 (弱)
 *   hard:   残局抢出、必压地主、队友送牌、威胁评估
 *   expert: 记牌 + 结构规划 (整手牌最少手数分解) + 深度领牌/跟牌策略
 */
const C = require('./cards.js');
const V = r => C.RANK_VALUE[r];
const V2 = V('2');

const DIFFS = ['normal', 'hard', 'expert', 'legend'];
const diffAtLeast = (a, b) => DIFFS.indexOf(a) >= DIFFS.indexOf(b);

/* ---------- 分组工具 ---------- */
function groupByVal(hand) {
  const g = new Map();
  for (const c of hand) {
    const v = V(c.rank);
    if (!g.has(v)) g.set(v, []);
    g.get(v).push(c);
  }
  return g;
}
function lowSingles(g, n, exclude, excludeVals) {
  const out = [];
  for (const v of [...g.keys()].sort((a, b) => a - b)) {
    if (excludeVals && excludeVals.has(v)) continue;   // 翅膀不能撞主体点数
    for (const c of g.get(v)) {
      if (exclude && exclude.has(c.id)) continue;
      out.push(c.id);
      if (out.length === n) return out;
    }
  }
  return null;
}
function lowPairs(g, n, exclude, excludeVals) {
  const out = [];
  for (const v of [...g.keys()].sort((a, b) => a - b)) {
    if (out.length === n * 2) break;
    if (excludeVals && excludeVals.has(v)) continue;
    const cs = (g.get(v) || []).filter(c => !exclude || !exclude.has(c.id));
    if (cs.length >= 2) out.push(cs[0].id, cs[1].id);
  }
  return out.length === n * 2 ? out : null;
}
function longestRun(g, need, minLen) {
  const vals = [...g.keys()].filter(v => v < V2 && g.get(v).length >= need).sort((a, b) => a - b);
  let best = null, cur = [];
  for (let i = 0; i < vals.length; i++) {
    if (cur.length && vals[i] !== cur[cur.length - 1] + 1) cur = [];
    cur.push(vals[i]);
    if (cur.length >= minLen && (!best || cur.length > best.length)) best = cur.slice();
  }
  return best;
}
function idsOfRun(g, run, need) {
  const ids = [];
  for (const v of run) for (const c of g.get(v).slice(0, need)) ids.push(c.id);
  return ids;
}
function bombs(hand) {
  const g = groupByVal(hand), out = [];
  for (const v of g.keys()) if (g.get(v).length === 4) out.push(g.get(v).map(c => c.id));
  const sj = g.get(V('SJ')), bj = g.get(V('BJ'));
  if (sj && bj) out.push([sj[0].id, bj[0].id]);
  return out;
}
/** 粗算还需几次才能出完 (出牌单元) */
function playUnits(hand) {
  if (!hand.length) return 0;
  const g = groupByVal(hand);
  let u = 0;
  for (const [v, cs] of g) u += Math.ceil(cs.length / 4); // 同点多张算1次+散牌
  return Math.max(1, u + Math.floor(hand.length / 10));
}
/** 我牌里还剩几张"大牌" (A/2/王) */
function bigCardsLeft(hand) {
  return hand.filter(c => ['A', '2', 'SJ', 'BJ'].includes(c.rank)).length;
}

/* ============ 结构规划 (expert) ============ */
/**
 * 枚举手牌中所有结构化牌型 combo:
 *   { kind: 'chain'|'pairchain'|'plane'|'plane1'|'plane2'|'triple'|'triple1'|'triple2'|'pair',
 *     len, main, slots: [{v, take}], kickers: {singles:n, pairs:n} | null }
 * slots 是主体占用的点数位; kickers 是出牌时需要额外带的散牌。
 */
function enumCombos(hand) {
  const g = groupByVal(hand);
  const vals = [...g.keys()].sort((a, b) => a - b);
  const count = v => g.get(v).length;
  const combos = [];

  // 连续段上的滑动窗口: need=每张数量, minLen=最小连续长度
  const runsFor = (need, minLen) => {
    const elig = vals.filter(v => v < V2 && count(v) >= need);
    const segs = [];
    let cur = [];
    for (const v of elig) {
      if (cur.length && v !== cur[cur.length - 1] + 1) { if (cur.length) segs.push(cur); cur = []; }
      cur.push(v);
    }
    if (cur.length) segs.push(cur);
    const out = [];
    for (const seg of segs) {
      for (let i = 0; i < seg.length; i++) {
        for (let j = i + minLen - 1; j < seg.length; j++) {
          out.push(seg.slice(i, j + 1));
        }
      }
    }
    return out;
  };

  for (const run of runsFor(1, 5))
    combos.push({ kind: 'chain', len: run.length, main: run[run.length - 1], slots: run.map(v => ({ v, take: 1 })), kickers: null });
  for (const run of runsFor(2, 3))
    combos.push({ kind: 'pairchain', len: run.length * 2, main: run[run.length - 1], slots: run.map(v => ({ v, take: 2 })), kickers: null });
  for (const run of runsFor(3, 2)) {
    const m = run.length;
    combos.push({ kind: 'plane', len: m * 3, main: run[run.length - 1], slots: run.map(v => ({ v, take: 3 })), kickers: null });
    combos.push({ kind: 'plane1', len: m * 4, main: run[run.length - 1], slots: run.map(v => ({ v, take: 3 })), kickers: { singles: m, pairs: 0 } });
    combos.push({ kind: 'plane2', len: m * 5, main: run[run.length - 1], slots: run.map(v => ({ v, take: 3 })), kickers: { singles: 0, pairs: m } });
  }
  // 三张系列
  for (const v of vals) {
    if (count(v) >= 3) {
      combos.push({ kind: 'triple', len: 3, main: v, slots: [{ v, take: 3 }], kickers: null });
      combos.push({ kind: 'triple1', len: 4, main: v, slots: [{ v, take: 3 }], kickers: { singles: 1, pairs: 0 } });
      combos.push({ kind: 'triple2', len: 5, main: v, slots: [{ v, take: 3 }], kickers: { singles: 0, pairs: 1 } });
    }
    if (count(v) >= 2 && v < V('SJ'))
      combos.push({ kind: 'pair', len: 2, main: v, slots: [{ v, take: 2 }], kickers: null });
  }
  return combos;
}

/**
 * 最优分解: 返回把整手牌拆成合法出牌单元的方案, 最小化:
 *   手数*16 + Σ每张牌位置权重(顺子/三张内位置+1, 散牌 15-v) - 保留炸加成
 * 结果: { moves: [{typeName, ids, isBomb}], cost }
 * DP: cnt[15] 每点剩余张数, bombFlag 是否已拆炸弹点数.
 */
function bestDecomp(hand) {
  const g = groupByVal(hand);
  const cnt = new Array(15).fill(0);
  for (const v of g.keys()) cnt[v] = g.get(v).length;
  const cardPool = new Map();  // v -> [card...]
  for (const [v, cs] of g) cardPool.set(v, cs.slice());
  const rocket = cnt[V('SJ')] > 0 && cnt[V('BJ')] > 0;

  const combos = enumCombos(hand).map(co => ({
    ...co,
    // 每个 slot 占用的数量数组, 用于扣减 cnt
    takes: co.slots.map(s => ({ v: s.v, take: s.take })),
  }));

  // 散牌权重: 越小越便宜(鼓励大散牌被结构吃掉)
  const w = v => 15 - v;
  const memo = new Map();
  const keyOf = (cntArr, bf) => cntArr.join(',') + '|' + bf;
  const deadline = Date.now() + 120;           // 单局决策 ≤120ms
  let states = 0;
  const MAX_STATES = 40000;
  let aborted = false;

  function dp(cntArr, bf) {
    if (!aborted && (Date.now() > deadline || ++states > MAX_STATES)) aborted = true;
    if (aborted) return { cost: cntArr.reduce((s, c) => s + c, 0) * 20, moves: null }; // 启发式兜底
    const key = keyOf(cntArr, bf);
    if (memo.has(key)) return memo.get(key);
    if (cntArr.every(c => c === 0)) { const r = { cost: 0, moves: [] }; memo.set(key, r); return r; }

    let best = null;
    const tryMove = (cost, nextCnt, nextBf, move) => {
      const sub = dp(nextCnt, nextBf);
      if (sub.moves === null) return;          // 中断分支不还原
      const total = cost + sub.cost;
      if (!best || total < best.cost) best = { cost: total, moves: [move].concat(sub.moves) };
    };

    for (const co of combos) {
      // 主体可用性
      let ok = true;
      for (const t of co.takes) if (cntArr[t.v] < t.take) { ok = false; break; }
      if (!ok) continue;
      // 拆弹惩罚: 主体用到某点数 4 张但会留下余牌(变非整炸)
      let bombPenalty = 0;
      for (const t of co.takes) {
        const total = cardPool.get(t.v).length;
        if (total === 4 && cntArr[t.v] === 4 && t.take < 4) bombPenalty += 30;
      }
      // 翅膀占用: 顺序取最小可用点 —— 在 DP 里枚举翅膀代价太高,
      // 近似: 用"当前剩余中最小的合法散牌"占位扣减
      const next = cntArr.slice();
      for (const t of co.takes) next[t.v] -= t.take;
      let bodyCost = 16;
      for (const t of co.takes) bodyCost += t.take * 1; // 结构内每张 +1
      let kickCost = 0;
      if (co.kickers) {
        const need = co.kickers;
        const bodyVals = new Set(co.takes.map(t => t.v));
        if (need.singles) {
          // 翅膀必须避开主体点数 (否则 analyze 合并成更大三张, 非法)
          let left = need.singles;
          for (let v = 0; v < 15 && left > 0; v++) {
            if (bodyVals.has(v)) continue;
            while (next[v] > 0 && left > 0) { next[v]--; left--; kickCost += w(v); }
          }
          if (left > 0) continue; // 带不动 (宁可放弃, 不拆主体)
        } else if (need.pairs) {
          let left = need.pairs;
          for (let v = 0; v < V('SJ') && left > 0; v++) {
            if (bodyVals.has(v)) continue;
            if (next[v] >= 2) { next[v] -= 2; left--; kickCost += 2 * w(v); }
          }
          if (left > 0) continue;
        }
      }
      tryMove(bodyCost + kickCost + bombPenalty, next, bf, { co });
    }

    // 炸弹保留: 若有 4 张点数, 提供"当炸弹"的走法(仅在 bf 允许时优于拆散)
    for (let v = 0; v < 15; v++) {
      if (cntArr[v] === 4) {
        const next = cntArr.slice(); next[v] = 0;
        tryMove(16 + 4, next, bf, { bomb: v });
      }
    }
    // 王炸
    if (!bf && rocket && cntArr[V('SJ')] > 0 && cntArr[V('BJ')] > 0) {
      const next = cntArr.slice(); next[V('SJ')]--; next[V('BJ')]--;
      tryMove(16 + 2, next, bf, { rocket: true });
    }
    // 散牌: 每点拆出 单张/对子/三张(三张已在 combos), 这里兜底单张
    // (枚举所有非空点的单张/对子, 防止 combos 覆盖不到: 如 SJ/BJ 单张)
    for (let v = 0; v < 15; v++) {
      if (cntArr[v] > 0) {
        const next = cntArr.slice(); next[v]--;
        tryMove(16 + w(v), next, bf, { single: v });
      }
    }
    memo.set(key, best || { cost: 1e9, moves: [] });
    return memo.get(key);
  }

  const res = dp(cnt.slice(), 0);
  if (!res.moves || !res.moves.length) {
    // 超时兜底: 贪心 —— 每张按单张/对子/三张/炸弹粗分
    res.moves = [];
    const left = cnt.slice();
    for (let v = 0; v < 15; v++) {
      while (left[v] >= 4) { res.moves.push({ bomb: v }); left[v] -= 4; }
      while (left[v] >= 3) { res.moves.push({ co: { takes: [{ v, take: 3 }], kickers: null } }); left[v] -= 3; }
      while (left[v] >= 2 && v < V('SJ')) { res.moves.push({ co: { takes: [{ v, take: 2 }], kickers: null } }); left[v] -= 2; }
      while (left[v] > 0) { res.moves.push({ single: v }); left[v]--; }
    }
  }
  // 落地为实际 id 列表
  const useCard = v => (cardPool.get(v) || []).shift();
  const moves = [];
  const moveOf = co => {
    const bodyIds = [];
    for (const t of co.takes) for (let i = 0; i < t.take; i++) bodyIds.push(useCard(t.v).id);
    const kickIds = [];
    if (co.kickers) {
      const bodyVals = new Set(co.takes.map(t => t.v));
      if (co.kickers.singles) {
        // 翅膀避开主体点数 (与 DP 一致, 否则 analyze 会合并成非法牌型)
        let left = co.kickers.singles;
        for (let v = 0; v < 15 && left > 0; v++) {
          if (bodyVals.has(v)) continue;
          const pool = cardPool.get(v) || [];
          while (pool.length > 0 && left > 0) { kickIds.push(useCard(v).id); left--; }
        }
      } else if (co.kickers.pairs) {
        let left = co.kickers.pairs;
        for (let v = 0; v < V('SJ') && left > 0; v++) {
          if (bodyVals.has(v)) continue;
          const pool = cardPool.get(v) || [];
          if (pool.length >= 2) { kickIds.push(useCard(v).id, useCard(v).id); left--; }
        }
      }
    }
    const all = bodyIds.concat(kickIds);
    const cards = all.map(id => hand.find(c => c.id === id));
    const ana = C.analyze(cards);
    return { ids: all, ana, kind: co.kind, isBomb: false };
  };
  for (const m of res.moves) {
    if (m.single != null) {
      const c = useCard(m.single);
      moves.push({ ids: [c.id], ana: C.analyze([c]), kind: 'single', isBomb: false });
    } else if (m.bomb != null) {
      const ids = cardPool.get(m.bomb).splice(0, 4).map(c => c.id);
      moves.push({ ids, ana: C.analyze(ids.map(id => hand.find(c => c.id === id))), kind: 'bomb', isBomb: true });
    } else if (m.rocket) {
      const ids = [useCard(V('SJ')).id, useCard(V('BJ')).id];
      moves.push({ ids, ana: { type: C.TYPE.ROCKET, typeName: '王炸', len: 2, main: Infinity }, kind: 'rocket', isBomb: true });
    } else if (m.co) {
      moves.push(moveOf(m.co));
    }
  }
  return { moves, count: moves.length };
}

/* ---------- 记牌 (expert) ---------- */
function unseenCounts(room, seat) {
  const g = groupByVal(room.players[seat].hand);
  const un = new Array(15).fill(0);
  for (let v = 0; v < 15; v++) un[v] = 4;
  un[V('SJ')] = 1; un[V('BJ')] = 1;
  for (const c of room.played) un[V(c.rank)]--;
  for (const [v, cs] of g) un[v] -= cs.length;
  return un;  // 其余两家合计的剩余张数
}
function maxUnseenBelow(room, seat, thresholdV, excludeSeat) {
  const un = unseenCounts(room, seat);
  let max = -1;
  // 计算 excludeSeat 自己贡献了多少 (粗略: 平分给两家也行, 这里用 played+我的牌推断另外两家)
  // 保守: 全部 unseen 都可能集中在某一家
  for (let v = 0; v < thresholdV; v++) if (un[v] > 0) max = v;
  return max;
}
/** 两家对手(含队友)某点合计还剩几张在外 */
function unseenAt(room, seat, v) { return unseenCounts(room, seat)[v]; }
/** 我能否断言 "牌型 X 我的是全场最大" (顺子/连对/飞机) */
function comboIsMax(room, seat, ana) {
  const un = unseenCounts(room, seat);
  // 对 chain/pairchain/plane: 若有任意 unseen 点 > main 且能组成同型更长/更高的窗口就危险
  // 简化: 只看是否有 unseen 点数 > main
  for (let v = ana.main + 1; v < V2; v++) {
    if (un[v] > 0) return false;
  }
  return true;
}
/** 外面还有没有炸弹/王炸 (我手上的之外) */
function bombThreatOut(room, seat) {
  const un = unseenCounts(room, seat);
  const myG = groupByVal(room.players[seat].hand);
  let threat = 0;
  for (let v = 0; v < V2; v++) {
    // 某点 4 张全在别人那里 -> 潜在炸弹
    if (un[v] === 4) threat++;
    // 也可能: 别人凑齐 4 张 (un[v]===4 是必要条件)
  }
  if (un[V('SJ')] === 1 && un[V('BJ')] === 1) threat++; // 王炸可能在外
  return threat;
}

/* ---------- 叫分 ---------- */
function handScore(hand) {
  const cnt = {};
  for (const c of hand) cnt[c.rank] = (cnt[c.rank] || 0) + 1;
  let s = 0;
  if (cnt['BJ']) s += 4;
  if (cnt['SJ']) s += 2.5;
  if (cnt['BJ'] && cnt['SJ']) s += 3.5;
  s += (cnt['2'] || 0) * 1.3;
  s += (cnt['A'] || 0) * 0.8;
  s += (cnt['K'] || 0) * 0.35;
  for (const r in cnt) if (cnt[r] === 4) s += 4.5;
  const run = longestRun(groupByVal(hand), 1, 3);
  if (run && run.length >= 5) s += 1.5;
  const plane = longestRun(groupByVal(hand), 3, 2);
  if (plane) s += 2;
  return s;
}
function chooseBid(hand, currentBid, diff) {
  const s = handScore(hand);
  if (diff === 'expert') {
    // 结合分解手数: 手数少也是叫分资本
    const d = bestDecomp(hand);
    const eff = s + Math.max(0, 8 - d.count) * 0.6;
    const want = eff >= 6 ? 3 : eff >= 4.2 ? 2 : eff >= 2.6 ? 1 : 0;
    return want > currentBid ? want : 0;
  }
  const want = s >= 6 ? 3 : s >= 4 ? 2 : s >= 2.5 ? 1 : 0;
  return want > currentBid ? want : 0;
}

/* ---------- 候选出牌 ---------- */
function cardById(hand, id) { return hand.find(c => c.id === id); }
function anaOfIds(hand, ids) { return C.analyze(ids.map(id => cardById(hand, id)).filter(Boolean)); }

/** 生成所有能领出的候选 (非炸) + 可选炸弹 */
function leadCandidates(hand, withBombs) {
  const g = groupByVal(hand);
  const cand = [];
  const bombSet = new Set(bombs(hand).flat());
  const run1 = longestRun(g, 1, 5), run2 = longestRun(g, 2, 3), run3 = longestRun(g, 3, 2);
  const seen = new Set();
  const push = ids => {
    if (!ids || !ids.length) return;
    const k = ids.slice().sort((a, b) => a - b).join(',');
    if (seen.has(k)) return;
    seen.add(k); cand.push(ids);
  };
  if (run1) push(idsOfRun(g, run1, 1));
  if (run2) push(idsOfRun(g, run2, 2));
  if (run3) {
    const body = idsOfRun(g, run3, 3);
    const ex = new Set(body);
    const exv = new Set(run3);          // 翅膀避开主体点数
    push(body);  // 裸飞机
    const s = lowSingles(g, run3.length, ex, exv);
    if (s) push(body.concat(s));
    const p = lowPairs(g, run3.length, ex, exv);
    if (p) push(body.concat(p));
  }
  for (const v of [...g.keys()].sort((a, b) => a - b)) {
    if (g.get(v).length >= 3 && g.get(v).length < 4) {
      const tri = g.get(v).slice(0, 3).map(c => c.id);
      const ex = new Set([...tri, ...bombSet]);
      const exv = new Set([v]);
      push(tri);
      const s = lowSingles(g, 1, ex, exv);
      if (s) push(tri.concat(s));
      const p = lowPairs(g, 1, ex, exv);
      if (p) push(tri.concat(p));
    }
  }
  for (const v of [...g.keys()].sort((a, b) => a - b)) {
    const cs = g.get(v).filter(c => !bombSet.has(c.id));
    if (cs.length === 2 || cs.length === 3) push(cs.slice(0, 2).map(c => c.id));
  }
  const sorted = C.sortCards(hand);
  for (let i = sorted.length - 1; i >= 0; i--) {
    if (!bombSet.has(sorted[i].id)) { push([sorted[i].id]); break; }
  }
  if (withBombs) for (const b of bombs(hand)) push(b);
  if (!cand.length) push([sorted[sorted.length - 1].id]);
  return cand;
}

/** 生成所有能压过 lastAna 的候选 (含拆三张出的单张等), 按"代价"升序 */
function beatCandidates(hand, lastAna) {
  const g = groupByVal(hand);
  const vals = [...g.keys()].sort((a, b) => a - b);
  const bombSet = new Set(bombs(hand).flat());
  const out = [];
  const seen = new Set();
  const push = ids => {
    if (!ids) return;
    const k = ids.slice().sort((a, b) => a - b).join(',');
    if (seen.has(k)) return;
    seen.add(k); out.push(ids);
  };
  const min = lastAna.main;
  const takeIds = (v, n) => g.get(v).slice(0, n).map(c => c.id);
  const windows = (per, len) => {
    const elig = vals.filter(v => v < V2 && v > min - len + 1 && g.get(v).length >= per);
    // 需要连续 len 个点且最大点 > min
    const all = vals.filter(v => v < V2 && g.get(v).length >= per);
    for (let i = 0; i + len <= all.length; i++) {
      const win = all.slice(i, i + len);
      if (win[len - 1] <= min) continue;
      let ok = true;
      for (let j = 1; j < len; j++) if (win[j] !== win[j - 1] + 1) { ok = false; break; }
      if (!ok) continue;
      push(win.flatMap(v => takeIds(v, per)));
    }
  };

  switch (lastAna.type) {
    case C.TYPE.SINGLE:
      for (const v of vals) if (v > min) push([g.get(v)[0].id]);
      break;
    case C.TYPE.PAIR:
      for (const v of vals) if (v > min && g.get(v).length >= 2) push(takeIds(v, 2));
      break;
    case C.TYPE.TRIPLE:
      for (const v of vals) if (v > min && g.get(v).length >= 3) push(takeIds(v, 3));
      break;
    case C.TYPE.TRIPLE_1:
      for (const v of vals) if (v > min && g.get(v).length >= 3) {
        const tri = takeIds(v, 3);
        const s = lowSingles(g, 1, new Set(tri), new Set([v]));
        if (s) push(tri.concat(s));
      }
      break;
    case C.TYPE.TRIPLE_2:
      for (const v of vals) if (v > min && g.get(v).length >= 3) {
        const tri = takeIds(v, 3);
        const ex = new Set(tri);
        const p = lowPairs(g, 1, ex, new Set([v]));
        if (p) push(tri.concat(p));
      }
      break;
    case C.TYPE.CHAIN: windows(1, lastAna.len); break;
    case C.TYPE.PAIR_CHAIN: windows(2, lastAna.len / 2); break;
    case C.TYPE.AIRPLANE: windows(3, lastAna.len / 3); break;
    case C.TYPE.AIRPLANE_1: {
      const m = lastAna.len / 4;
      const all = vals.filter(v => v < V2 && g.get(v).length >= 3);
      for (let i = 0; i + m <= all.length; i++) {
        const win = all.slice(i, i + m);
        if (win[m - 1] <= min) continue;
        let ok = true;
        for (let j = 1; j < m; j++) if (win[j] !== win[j - 1] + 1) { ok = false; break; }
        if (!ok) continue;
        const body = win.flatMap(v => takeIds(v, 3));
        const s = lowSingles(g, m, new Set(body), new Set(win));
        if (s) push(body.concat(s));
      }
      break;
    }
    case C.TYPE.AIRPLANE_2: {
      const m = lastAna.len / 5;
      const all = vals.filter(v => v < V2 && g.get(v).length >= 3);
      for (let i = 0; i + m <= all.length; i++) {
        const win = all.slice(i, i + m);
        if (win[m - 1] <= min) continue;
        let ok = true;
        for (let j = 1; j < m; j++) if (win[j] !== win[j - 1] + 1) { ok = false; break; }
        if (!ok) continue;
        const body = win.flatMap(v => takeIds(v, 3));
        const ex = new Set(body);
        const exv = new Set(win);
        const wings = [];
        for (const v of vals) {
          if (wings.length === m * 2) break;
          if (exv.has(v)) continue;
          const cs = (g.get(v) || []).filter(c => !ex.has(c.id));
          if (cs.length >= 2) { wings.push(cs[0].id, cs[1].id); }
        }
        if (wings.length === m * 2) push(body.concat(wings));
      }
      break;
    }
    case C.TYPE.FOUR_2:
      for (const v of vals) if (v > min && g.get(v).length === 4) {
        const four = takeIds(v, 4);
        const s = lowSingles(g, 2, new Set(four));
        if (s) push(four.concat(s));
      }
      break;
    case C.TYPE.FOUR_2_2:
      for (const v of vals) if (v > min && g.get(v).length === 4) {
        const four = takeIds(v, 4);
        const ex = new Set(four);
        const wings = lowPairs(g, 2, ex);
        if (wings) push(four.concat(wings));
      }
      break;
    case C.TYPE.BOMB:
      for (const v of vals) if (v > min && g.get(v).length === 4) push(takeIds(v, 4));
      break;
    default: break;
  }
  // 炸弹 / 王炸兜底
  if (lastAna.type !== C.TYPE.ROCKET) {
    for (const v of vals) {
      if (g.get(v).length === 4) {
        if (lastAna.type === C.TYPE.BOMB && v <= min) continue;
        push(takeIds(v, 4));
      }
    }
    const sj = g.get(V('SJ')), bj = g.get(V('BJ'));
    if (sj && bj) push([sj[0].id, bj[0].id]);
  }
  return out;
}

/* ---------- 领牌 ---------- */
function leadPlay(hand, room, seat, diff) {
  const g = groupByVal(hand);
  const cand = [];
  const push = ids => { if (ids && ids.length) cand.push(ids); };
  const bombSet = new Set(bombs(hand).flat());

  const run1 = longestRun(g, 1, 5), run2 = longestRun(g, 2, 3), run3 = longestRun(g, 3, 2);
  if (run1) push(idsOfRun(g, run1, 1));
  if (run2) push(idsOfRun(g, run2, 2));
  if (run3) {
    const body = idsOfRun(g, run3, 3);
    const ex = new Set(body);
    const exv = new Set(run3);
    push(body.concat(lowSingles(g, run3.length, ex, exv) || lowPairs(g, run3.length, ex, exv) || []));
  }
  // 三带一/二 (带最小散牌)
  for (const v of [...g.keys()].sort((a, b) => a - b)) {
    if (g.get(v).length >= 3) {
      const tri = g.get(v).slice(0, 3).map(c => c.id);
      const ex = new Set([...tri, ...bombSet]);
      const exv = new Set([v]);
      const s = lowSingles(g, 1, ex, exv) || (lowPairs(g, 1, ex, exv) || []).slice(0, 2);
      push(tri.concat(s || []));
      break;
    }
  }
  for (const v of [...g.keys()].sort((a, b) => a - b)) {
    const cs = g.get(v).filter(c => !bombSet.has(c.id));
    if (cs.length === 2) { push(cs.map(c => c.id)); break; }
  }
  const sorted = C.sortCards(hand);
  const low = [...sorted].reverse().find(c => !bombSet.has(c.id));
  if (low) push([low.id]);
  if (!cand.length) push([sorted[sorted.length - 1].id]);

  if (diffAtLeast(diff, 'hard')) {
    /* 残局: 只剩 ≤3 张能一次出完的牌型优先 */
    const finish = cand.find(c => c.length === hand.length);
    if (finish) return finish;
    /* 队友 (农民) 只剩 1-2 张: 出小单送队友接牌走 */
    if (seat !== room.landlordSeat) {
      const mates = room.players.filter(p => p.seat !== room.landlordSeat && p.seat !== seat);
      const mate = mates[0];
      if (mate && mate.hand.length <= 2) {
        const single = cand.find(c => c.length === 1 && V(cardById(hand, c[0]).rank) < V('Q'));
        if (single) return single;
      }
    }
    /* 地主: 优先甩长牌型清手牌, 剩牌少则先走大单 */
    if (seat === room.landlordSeat && hand.length <= 5) {
      const big = [...sorted].find(c => !bombSet.has(c.id));
      if (big) return [big.id];
    }
    /* 地主领出策略: 手数还很多时先出"被压损失小"的牌型(顺子/三带/连对),
       且优先出小点数的对子/单张而不是把大牌型先甩完 */
    if (seat === room.landlordSeat) {
      const plan = bestDecomp(hand);
      const moves = plan.moves.filter(m => !m.isBomb);
      /* 农民临门: 避开对手张数刚好能接的牌型
         剩1张→别出单; 剩2张→别出对; 剩3张→别出三张/三带; 剩5张→别出顺5 */
      const dangerType = new Set();
      for (const p of room.players) {
        if (p.seat === seat) continue;
        const n = p.hand.length;
        if (n === 1) dangerType.add(C.TYPE.SINGLE);
        if (n === 2) dangerType.add(C.TYPE.PAIR);
        if (n === 3) { dangerType.add(C.TYPE.TRIPLE); dangerType.add(C.TYPE.PAIR); }
        if (n === 4) { dangerType.add(C.TYPE.TRIPLE_1); dangerType.add(C.TYPE.SINGLE); }
        if (n === 5) { dangerType.add(C.TYPE.TRIPLE_2); dangerType.add(C.TYPE.CHAIN); }
      }
      const safe = moves.filter(m => !dangerType.has(m.ana.type));
      const pool = safe.length ? safe : moves;
      // 只有 2 手时: 先走小的一手, 留大的收尾
      if (pool.length === 2) {
        return (pool[0].ana.main <= pool[1].ana.main ? pool[0] : pool[1]).ids;
      }
      // 有长牌型且我牌还多: 先甩 (被压概率低)
      const long = pool.filter(m =>
        [C.TYPE.CHAIN, C.TYPE.PAIR_CHAIN, C.TYPE.AIRPLANE, C.TYPE.AIRPLANE_1, C.TYPE.AIRPLANE_2].includes(m.ana.type));
      if (long.length && hand.length > 10) {
        return long.sort((a, b) => b.ids.length - a.ids.length)[0].ids;
      }
      // 三带优先 (带掉小牌)
      const tri = pool.filter(m => m.kind && m.kind.startsWith('triple'))
        .sort((a, b) => b.ids.length - a.ids.length || a.ana.main - b.ana.main);
      if (tri.length && hand.length > 8) return tri[0].ids;
      // 否则最小对子 > 最小单张 (对子更难被农民接力)
      const pairs = pool.filter(m => m.ana.type === C.TYPE.PAIR).sort((a, b) => a.ana.main - b.ana.main);
      if (pairs.length) return pairs[0].ids;
      if (pool.length) return pool.sort((a, b) => a.ana.main - b.ana.main)[0].ids;
    }
    cand.sort((a, b) => b.length - a.length);
    return cand[0];
  }
  cand.sort((a, b) => b.length - a.length);
  return cand[0];
}

/* ---------- expert 领牌: 按分解方案 + 记牌 ---------- */
function expertLead(hand, room, seat) {
  const plan = bestDecomp(hand);
  const moves = plan.moves;
  if (!moves.length) return leadPlay(hand, room, seat, 'hard');
  const sorted = C.sortCards(hand);
  const isLord = seat === room.landlordSeat;
  const mates = room.players.filter(p => p.seat !== room.landlordSeat && p.seat !== seat);

  /* 一把能出完 */
  const fin = moves.find(m => m.ids.length === hand.length);
  if (fin) return fin.ids;

  /* 农民队友只剩 1 张: 喂最小单 (别喂对子, 防止卡住) */
  if (!isLord) {
    const mate = mates[0];
    if (mate) {
      if (mate.hand.length === 1) {
        const sg = moves.filter(m => m.ana.type === C.TYPE.SINGLE && !m.isBomb)
          .sort((a, b) => a.ana.main - b.ana.main);
        if (sg.length) {
          // 确认地主可能拦截也要喂 —— 喂最小
          return sg[0].ids;
        }
      }
      if (mate.hand.length === 2) {
        // 队友剩 2 张: 优先小对子(他可能对子), 其次小单
        const pr = moves.filter(m => m.ana.type === C.TYPE.PAIR && m.ana.main < V('Q'))
          .sort((a, b) => a.ana.main - b.ana.main);
        if (pr.length) return pr[0].ids;
        const sg = moves.filter(m => m.ana.type === C.TYPE.SINGLE && !m.isBomb)
          .sort((a, b) => a.ana.main - b.ana.main);
        if (sg.length) return sg[0].ids;
      }
    }
  }

  /* 剩 2 手以内: 先出"对手拦不住"的那手, 保证收尾 */
  if (moves.length === 2) {
    const [m1, m2] = moves;
    // 若 m1 是炸弹/王炸, 先走非炸的
    if (m1.isBomb && !m2.isBomb) return m2.ids;
    if (m2.isBomb && !m1.isBomb) return m1.ids;
    // 两手都非炸: 先出较难被压的; 若一手是"全场最大"则另一手先走? 保守: 先小后大留住控场
    // 但若先走小的会被地主接走反杀 —— 判断对手剩牌
    const lordCnt = room.players[room.landlordSeat].hand.length;
    if (!isLord && lordCnt <= 3) {
      // 地主快赢: 先甩最稳的(大)牌
      return (m1.ana.main >= m2.ana.main ? m1 : m2).ids;
    }
    return (m1.ana.main <= m2.ana.main ? m1 : m2).ids;
  }

  /* 常规: 优先出"出牌权难被抢"的长牌型 (顺子/连对/飞机), 且我用记牌确认是最大 */
  const longMoves = moves.filter(m =>
    !m.isBomb && [C.TYPE.CHAIN, C.TYPE.PAIR_CHAIN, C.TYPE.AIRPLANE, C.TYPE.AIRPLANE_1, C.TYPE.AIRPLANE_2].includes(m.ana.type));
  for (const m of longMoves.sort((a, b) => b.ids.length - a.ids.length)) {
    if (comboIsMax(room, seat, m.ana)) return m.ids;
  }
  // 长牌型即便不是最大, 也优先清 (除非地主临门)
  const lordCnt = room.players[room.landlordSeat].hand.length;
  if (longMoves.length && !(isLord && bombThreatOut(room, seat) >= 2 && hand.length <= 8)) {
    if (isLord || lordCnt > 4) return longMoves[0].ids;
  }

  /* 三带: 带小牌清散张 */
  const tri = moves.filter(m => !m.isBomb && (m.kind === 'triple1' || m.kind === 'triple2' || m.kind === 'triple'))
    .sort((a, b) => b.ids.length - a.ids.length || a.ana.main - b.ana.main);
  if (tri.length) {
    const t = tri[0];
    // 若三张主体够大(K/A)或外面 2 已尽, 可当控场牌先留; 否则早出
    if (t.ana.main < V('K') || hand.length <= 8) return t.ids;
  }

  /* 对子策略: 地主/农民都先用小对子试牌权, 但下家若是队友且牌少就避免 */
  const pairs = moves.filter(m => !m.isBomb && m.ana.type === C.TYPE.PAIR)
    .sort((a, b) => a.ana.main - b.ana.main);
  const singles = moves.filter(m => !m.isBomb && m.ana.type === C.TYPE.SINGLE)
    .sort((a, b) => a.ana.main - b.ana.main);

  if (!isLord && mates[0] && mates[0].hand.length <= 3 && singles.length) {
    return singles[0].ids; // 队友快走完, 出单让他接
  }

  /* 地主残局: 大单压场 */
  if (isLord && hand.length <= 5) {
    const bigSg = singles.filter(m => m.ana.main >= V('A'));
    if (bigSg.length) return bigSg[0].ids;
  }
  if (pairs.length && hand.length > 6) {
    // 早中期: 小对子先出 (通常能过一圈)
    const small = pairs.filter(m => m.ana.main < V('J'));
    if (small.length) return small[0].ids;
  }
  if (singles.length) {
    // 保留最大单牌控场: 出"倒数第二小"附近; 简单起见出最小非炸单
    return singles[0].ids;
  }
  if (pairs.length) return pairs[0].ids;
  // 兜底
  const nonBomb = moves.filter(m => !m.isBomb);
  return (nonBomb[0] || moves[0]).ids;
}

/* ---------- 是否压牌 ---------- */
function shouldBeat(room, seat, hand, lastAna, ids, diff) {
  const meFarmer = seat !== room.landlordSeat;
  const lastSeat = room.lastPlay.seat;
  const lastFarmer = lastSeat !== room.landlordSeat;
  const lastCnt = room.players[lastSeat].hand.length;
  const lordCnt = room.players[room.landlordSeat].hand.length;
  const ana2 = ids ? anaOfIds(hand, ids) : null;
  const isBomb = ana2 && (ana2.type === C.TYPE.BOMB || ana2.type === C.TYPE.ROCKET);

  if (!ids) return false;

  /* 队友出牌: 原则上不压 —— 我压了牌权到我这, 牌还是得出, 白浪费大牌.
   * 例外: 我能一把走完 / 队友牌还很多而地主只剩1-2张等着收 */
  if (meFarmer && lastFarmer) {
    if (hand.length === ids.length) return true;                       // 能出完
    if (lastAna.type === C.TYPE.ROCKET || lastAna.type === C.TYPE.BOMB) return false;
    if (diffAtLeast(diff, 'hard') && lastCnt <= 2) return false;       // 队友临门, 让他走
    // 队友出大牌(A/2/王 或 长牌型): 让, 地主都不一定接得住
    if (lastAna.type === C.TYPE.SINGLE && lastAna.main >= V('A')) {
      if (diffAtLeast(diff, 'hard') && lastCnt <= 3) return false;
      if (Math.random() < (diffAtLeast(diff, 'hard') ? 0.85 : 0.7)) return false;
    }
    if ([C.TYPE.CHAIN, C.TYPE.PAIR_CHAIN, C.TYPE.AIRPLANE, C.TYPE.AIRPLANE_1, C.TYPE.AIRPLANE_2].includes(lastAna.type)) {
      if (Math.random() < (diffAtLeast(diff, 'hard') ? 0.8 : 0.5)) return false;
    }
    /* 队友出小牌: 我可以用"不太贵的牌"接力, 让牌权留在农民方;
       但要动大牌或拆结构就让队友继续 */
    if (diffAtLeast(diff, 'hard')) {
      const myMain = ana2.main;
      const g = groupByVal(hand);
      const used = {};
      for (const id of ids) { const v = V(cardById(hand, id).rank); used[v] = (used[v] || 0) + 1; }
      let breakStructure = false;
      for (const v in used) {
        const tot = g.get(+v).length;
        if (tot >= 3 && used[v] < tot) { breakStructure = true; break; }  // 拆了三张/四张
      }
      if (isBomb) return false;                            // 不炸队友
      if (lordCnt <= 1) return true;                       // 地主只剩1张, 挡死
      // 接力条件: 我快走完 / 压牌不太大 / 队友出了小牌我顺势接手
      if (breakStructure) {
        // 拆三张/炸弹为队友接力, 只有我出完这手就赢才值
        return hand.length <= ids.length;
      }
      if (hand.length <= 6) return true;                   // 我快走完, 抢主动权
      if (myMain <= V('K')) return true;                   // 小牌接力很划算
      if (myMain <= V('2') && lastAna.main >= V('J')) return true;  // 队友中牌, 我用大牌盖 (防地主接)
      return false;                                      // 队友小牌我要动 2/王 -> 让
    }
    /* normal: 大概率让 */
    return Math.random() < 0.25;
  }

  /* 地主只剩 ≤2 张: 农民坚决不能放 */
  if (meFarmer && !lastFarmer && lordCnt <= 2) {
    return diffAtLeast(diff, 'hard') ? true : Math.random() < 0.9;
  }
  if (diffAtLeast(diff, 'hard') && meFarmer && !lastFarmer && lordCnt === 1 && lastAna.type === C.TYPE.SINGLE) {
    return true;
  }
  /* 炸弹使用决策 */
  if (isBomb) {
    if (hand.length <= ids.length + 1) return true;          // 能出完
    if (!lastFarmer && lordCnt <= 4) return true;            // 地主快赢
    if (diffAtLeast(diff, 'hard') && playUnits(hand) <= 2) return true;
    if (diffAtLeast(diff, 'hard') && bigCardsLeft(hand) >= 3) return true;
    return Math.random() < (diffAtLeast(diff, 'hard') ? 0.45 : 0.3);
  }
  /* 农民压地主: 我能走完必压; 用大牌压小牌时看地主剩牌决定值不值 */
  if (diffAtLeast(diff, 'hard') && meFarmer && !lastFarmer && !isBomb) {
    if (hand.length <= 4) return true;                        // 我快出完
    const myMain = ana2.main;
    if (lastAna.type === C.TYPE.SINGLE) {
      // 地主出小单, 我要用 2/王 压: 地主牌还多时让给队友顶 (队友后出)
      if (myMain >= V('2') && lordCnt > 6) {
        const mate = room.players.find(p => p.seat !== room.landlordSeat && p.seat !== seat);
        if (mate && mate.hand.length <= hand.length) return false;  // 队友牌更少, 他更适合顶
        return Math.random() < 0.4;                          // 40% 忍一手
      }
    }
    if (lastAna.type === C.TYPE.PAIR && myMain >= V('A') && lordCnt > 6) {
      return Math.random() < 0.55;                           // 对 A/对2 压小对, 一半概率忍
    }
  }
  /* normal: 农民压地主有点钝 —— 30% 会放过不该放的 */
  if (diff === 'normal' && meFarmer && !lastFarmer && !isBomb && lordCnt > 3) {
    return Math.random() < 0.75;
  }
  if (diffAtLeast(diff, 'hard') && hand.length <= 4 && !isBomb) return true;
  return true;
}

/* ---------- expert 压牌决策 ---------- */
function expertShouldBeat(room, seat, hand, lastAna, ids) {
  const ana2 = anaOfIds(hand, ids);
  const isBomb = ana2.type === C.TYPE.BOMB || ana2.type === C.TYPE.ROCKET;
  const meFarmer = seat !== room.landlordSeat;
  const lastSeat = room.lastPlay.seat;
  const lastFarmer = lastSeat !== room.landlordSeat;
  const lordCnt = room.players[room.landlordSeat].hand.length;
  const mateCnts = room.players.filter(p => p.seat !== room.landlordSeat && p.seat !== seat).map(p => p.hand.length);

  /* 能一把出完: 必出 */
  if (hand.length === ids.length) return true;

  /* 队友出的牌 */
  if (meFarmer && lastFarmer) {
    const lastCnt = room.players[lastSeat].hand.length;
    if (lastCnt <= 2) return false;                            // 让队友走
    if (lastAna.type === C.TYPE.ROCKET || lastAna.type === C.TYPE.BOMB) return false;
    // 队友已经压过地主的大牌, 我接力意义不大; 除非我几乎走完且这是免费窗口
    if (lastAna.main >= V('A') || lastAna.type === C.TYPE.CHAIN || lastAna.type === C.TYPE.PAIR_CHAIN) {
      return hand.length <= 5 && !isBomb;
    }
    // 队友出小牌, 我可以接手控场: 用手头富裕牌型接, 别用炸
    if (isBomb) return false;
    // 接牌成本: 若我的 ids 含 2/王级大牌 且地主牌还多, 可以忍一手? 简化: 出
    return true;
  }

  /* 地主出的牌 (我是农民) 或 农民出的牌 (我是地主) */
  if (meFarmer && !lastFarmer) {
    /* 地主临门: 必压 */
    if (lordCnt <= 2) return true;
    if (lordCnt <= 4 && !isBomb) return true;
    if (isBomb) {
      if (lordCnt <= 5) return true;
      // 外面威胁: 若地主很可能有更大炸, 忍
      const threat = bombThreatOut(room, seat);
      if (threat > 0 && hand.length > 8) return false;
      // 出完炸后手数 ≤2 才值得
      const rest = hand.filter(c => !ids.includes(c.id));
      return bestDecomp(rest).count <= 2;
    }
    /* 地主出大牌 (2/王): 用最小能压的接, 别拆炸 */
    if (lastAna.type === C.TYPE.SINGLE && lastAna.main >= V('2') && ana2.main >= V('SJ')) {
      // 用王压 2: 若地主牌还多且我有小些的应对空间, 可让一手等队友
      const mate = mateCnts[0];
      if (mate != null && mate <= 6) return false;   // 队友牌少, 让他先顶
    }
    return true;
  }

  /* 我是地主, 农民出牌 */
  if (!meFarmer) {
    // 农民快出完 -> 必压
    const lastCnt = room.players[lastSeat].hand.length;
    if (lastCnt <= 2) return true;
    if (isBomb) {
      if (lastCnt <= 4) return true;
      const rest = hand.filter(c => !ids.includes(c.id));
      if (bestDecomp(rest).count <= 2) return true;
      // 倍数与节奏: 手牌 ≤ 6 手才炸
      return playUnits(hand) <= 5 && hand.length <= 14;
    }
    // 顶门板: 用大牌压制, 但保留绝对大牌? 简化: 出
    return true;
  }
  return true;
}

/* ---------- 候选排序: expert 选"最优"压牌 ---------- */
function pickBeat(hand, room, seat, lastAna) {
  const cands = beatCandidates(hand, lastAna);
  if (!cands.length) return null;
  const bombSet = new Set(bombs(hand).flat());
  const score = ids => {
    const ana = anaOfIds(hand, ids);
    let s = 0;
    // 越大越差
    s += Math.min(ana.main, 30) * 2;
    // 拆弹惩罚
    for (const id of ids) if (bombSet.has(id) && !(ana.type === C.TYPE.BOMB || ana.type === C.TYPE.ROCKET)) s += 60;
    if (ana.type === C.TYPE.BOMB) s += 40;
    if (ana.type === C.TYPE.ROCKET) s += 80;
    // 拆顺子/三带结构惩罚: 若拿出的散牌破坏了 combos —— 近似: 用了 2 张同点(拆三张)罚
    const g = groupByVal(hand);
    const used = {};
    for (const id of ids) { const v = V(cardById(hand, id).rank); used[v] = (used[v] || 0) + 1; }
    for (const v in used) {
      const total = g.get(+v) ? g.get(+v).length : 0;
      if (total >= 3 && used[v] < total && total !== 4) s += 12; // 拆三张
      if (total === 4 && used[v] < 4) s += 25;
    }
    // 多带牌(三带)让翅膀带小牌加分
    return s;
  };
  cands.sort((a, b) => score(a) - score(b));
  return cands[0];
}

/* ============ 传说 (legend): 蒙特卡洛模拟选牌 ============
 * 对每个候选出牌, 抽样 N 种"对手暗牌分布"并把整局 rollout 到结束,
 * 用我方阵营实际胜率代替启发式打分。
 * rollout 内的快棋策略只用公开信息(各位置剩余张数/已出牌), 不偷看暗牌。
 */

const RANK_BY_V = ['3','4','5','6','7','8','9','10','J','Q','K','A','2','SJ','BJ'];
/* 手工构建牌面对象 ({id,rank,suit}), id 仅为区分个体 */
function mkCard(v, k) {
  return { id: 1000 + v * 8 + k, rank: RANK_BY_V[v], suit: '' };
}
/** 暗牌池: 全部 54 张 - 我的手牌 - 已出的牌 */
function unseenPool(room, seat) {
  const cnt = new Array(15).fill(4);
  cnt[V('SJ')] = 1; cnt[V('BJ')] = 1;
  for (const c of room.players[seat].hand) cnt[V(c.rank)]--;
  for (const c of room.played) cnt[V(c.rank)]--;
  const pool = [];
  for (let v = 0; v < 15; v++) for (let k = 0; k < cnt[v]; k++) pool.push(mkCard(v, k));
  return pool;
}

/* ---- rollout 用轻量手牌结构: {cnt:[15], size} ---- */
function handToSim(cards) {
  const cnt = new Array(15).fill(0);
  for (const c of cards) cnt[V(c.rank)]++;
  return { cnt, size: cards.length };
}
function simTake(sim, v, n) {
  sim.cnt[v] -= n; sim.size -= n;
  return { v, n };          // 返回 {点数, 张数} 供 ana 重建
}
/**
 * rollout 内领出: 快速启发式 —— 最长顺子/连对/三张带/对子/最小单
 * 返回 { ana: {type,main,len}, took: [{v,n}] }
 */
function simLead(sim) {
  const vals = [];
  for (let v = 0; v < 15; v++) if (sim.cnt[v] > 0) vals.push(v);
  const took = [];
  const take = (v, n) => { took.push({ v, n }); sim.cnt[v] -= n; sim.size -= n; };
  // 顺子
  let cur = [];
  for (let i = 0; i <= vals.length; i++) {
    const v = vals[i];
    if (i < vals.length && v < V2 && sim.cnt[v] >= 1 && (cur.length === 0 || v === cur[cur.length - 1] + 1)) { cur.push(v); continue; }
    if (cur.length >= 5) {
      for (const x of cur) take(x, 1);
      return { ana: { type: C.TYPE.CHAIN, main: cur[cur.length - 1], len: cur.length }, took };
    }
    cur = (i < vals.length && v < V2 && sim.cnt[v] >= 1) ? [v] : [];
  }
  // 连对
  cur = [];
  for (let i = 0; i <= vals.length; i++) {
    const v = vals[i];
    if (i < vals.length && v < V2 && sim.cnt[v] >= 2 && (cur.length === 0 || v === cur[cur.length - 1] + 1)) { cur.push(v); continue; }
    if (cur.length >= 3) {
      for (const x of cur) take(x, 2);
      return { ana: { type: C.TYPE.PAIR_CHAIN, main: cur[cur.length - 1], len: cur.length * 2 }, took };
    }
    cur = (i < vals.length && v < V2 && sim.cnt[v] >= 2) ? [v] : [];
  }
  // 三张 + 翅膀
  for (const v of vals) {
    if (sim.cnt[v] >= 3 && v < V2) {
      take(v, 3);
      for (let k = 0; k < 15; k++) if (sim.cnt[k] > 0) { take(k, 1); break; }  // 带最小单
      return { ana: { type: C.TYPE.TRIPLE_1, main: v, len: 4 }, took };
    }
  }
  // 对子
  for (const v of vals) if (sim.cnt[v] >= 2 && v < V('SJ')) { take(v, 2); return { ana: { type: C.TYPE.PAIR, main: v, len: 2 }, took }; }
  // 最小单
  for (const v of vals) { take(v, 1); return { ana: { type: C.TYPE.SINGLE, main: v, len: 1 }, took }; }
  return null;
}
/** rollout 内压牌: 返回 {ana, took} 或 null (不出) */
function simBeat(sim, ana) {
  const vals = [];
  for (let v = 0; v < 15; v++) if (sim.cnt[v] > 0) vals.push(v);
  const min = ana.main;
  const took = [];
  const take = (v, n) => { took.push({ v, n }); sim.cnt[v] -= n; sim.size -= n; };
  const undo = () => { for (const t of took) { sim.cnt[t.v] += t.n; sim.size += t.n; } took.length = 0; };
  const run = (per, len) => {
    for (let lo = 0; lo + len <= V2; lo++) {
      if (lo + len - 1 <= min) continue;
      let ok = true;
      for (let j = 0; j < len; j++) if (sim.cnt[lo + j] < per) { ok = false; break; }
      if (!ok) continue;
      for (let j = 0; j < len; j++) take(lo + j, per);
      return { type: ana.type, main: lo + len - 1, len: ana.len };
    }
    return null;
  };
  let res = null;
  switch (ana.type) {
    case C.TYPE.SINGLE:
      for (const v of vals) if (v > min) { take(v, 1); res = { type: ana.type, main: v, len: 1 }; break; }
      break;
    case C.TYPE.PAIR:
      for (const v of vals) if (v > min && sim.cnt[v] >= 2 && v < V('SJ')) { take(v, 2); res = { type: ana.type, main: v, len: 2 }; break; }
      break;
    case C.TYPE.TRIPLE:
      for (const v of vals) if (v > min && sim.cnt[v] >= 3 && v < V2) { take(v, 3); res = { type: ana.type, main: v, len: 3 }; break; }
      break;
    case C.TYPE.TRIPLE_1: case C.TYPE.TRIPLE_2: {
      const needPair = ana.type === C.TYPE.TRIPLE_2;
      for (const v of vals) {
        if (v <= min || sim.cnt[v] < 3 || v >= V2) continue;
        take(v, 3);
        let got = false;
        if (needPair) {
          for (let k = 0; k < V('SJ'); k++) if (sim.cnt[k] >= 2) { take(k, 2); got = true; break; }
        } else {
          for (let k = 0; k < 15; k++) if (sim.cnt[k] > 0) { take(k, 1); got = true; break; }
        }
        if (got) { res = { type: ana.type, main: v, len: ana.len }; break; }
        undo();  // 带不动, 还原
      }
      break;
    }
    case C.TYPE.CHAIN: res = run(1, ana.len); break;
    case C.TYPE.PAIR_CHAIN: res = run(2, ana.len / 2); break;
    case C.TYPE.AIRPLANE: res = run(3, ana.len / 3); break;
    case C.TYPE.AIRPLANE_1: case C.TYPE.AIRPLANE_2: {
      const m = ana.type === C.TYPE.AIRPLANE_1 ? ana.len / 4 : ana.len / 5;
      const perK = ana.type === C.TYPE.AIRPLANE_1 ? 1 : 2;
      const body = run(3, m);
      if (body) {
        let ok = true, left = m;
        if (perK === 1) {
          for (let k = 0; k < 15 && left > 0; k++) while (sim.cnt[k] > 0 && left > 0) { take(k, 1); left--; }
        } else {
          for (let k = 0; k < V('SJ') && left > 0; k++) if (sim.cnt[k] >= 2) { take(k, 2); left--; }
        }
        if (left === 0) res = { type: ana.type, main: body.main, len: ana.len };
        else undo();
      }
      break;
    }
    case C.TYPE.BOMB:
      for (const v of vals) if (v > min && sim.cnt[v] === 4 && v < V2) { take(v, 4); res = { type: C.TYPE.BOMB, main: v, len: 4 }; break; }
      if (!res && sim.cnt[V('SJ')] > 0 && sim.cnt[V('BJ')] > 0) {
        take(V('SJ'), 1); take(V('BJ'), 1);
        res = { type: C.TYPE.ROCKET, main: Infinity, len: 2 };
      }
      return res ? { ana: res, took } : null;
    default: break;
  }
  if (!res && ana.type !== C.TYPE.BOMB && ana.type !== C.TYPE.ROCKET) {
    for (const v of vals) if (sim.cnt[v] === 4 && v < V2) { take(v, 4); res = { type: C.TYPE.BOMB, main: v, len: 4 }; break; }
    if (!res && sim.cnt[V('SJ')] > 0 && sim.cnt[V('BJ')] > 0) {
      take(V('SJ'), 1); take(V('BJ'), 1);
      res = { type: C.TYPE.ROCKET, main: Infinity, len: 2 };
    }
  }
  if (!res) { undo(); return null; }
  return { ana: res, took };
}

/**
 * rollout 到结束.
 * hands: [sim0,sim1,sim2]; cur: 轮到谁; lastAna/lastSeat: 当前桌面;
 * 返回 'lord' | 'farmer' | null(超限)
 */
function rollout(hands, landlordSeat, cur, lastAna, lastSeat, maxSteps) {
  let pass = 0, steps = 0;
  while (steps++ < (maxSteps || 300)) {
    const sim = hands[cur];
    if (sim.size === 0) {
      return cur === landlordSeat ? 'lord' : 'farmer';
    }
    if (!lastAna || lastSeat === cur) {
      /* 农民领出: 队友只剩1-2张时优先喂小单/小对子让他接牌走完 */
      let r = null;
      if (cur !== landlordSeat) {
        const mateSeat = 3 - cur - landlordSeat;
        const mateLeft = hands[mateSeat].size;
        if (mateLeft === 1 || mateLeft === 2) {
          // 找最小单张(队友剩1)或最小对子(队友剩2)
          const want = mateLeft === 1 ? 1 : 2;
          const vals = [];
          for (let v = 0; v < 15; v++) if (sim.cnt[v] > 0) vals.push(v);
          for (const v of vals) {
            if (want === 1 && sim.cnt[v] >= 1) {
              sim.cnt[v]--; sim.size--;
              r = { ana: { type: C.TYPE.SINGLE, main: v, len: 1 }, took: [{ v, n: 1 }] };
              break;
            }
            if (want === 2 && sim.cnt[v] >= 2 && v < V('SJ')) {
              sim.cnt[v] -= 2; sim.size -= 2;
              r = { ana: { type: C.TYPE.PAIR, main: v, len: 2 }, took: [{ v, n: 2 }] };
              break;
            }
          }
          // 队友剩2但我没对子可喂 -> 退回喂单 (至少别出长牌卡他)
          if (!r && mateLeft === 2) {
            for (const v of vals) {
              if (sim.cnt[v] >= 1) {
                sim.cnt[v]--; sim.size--;
                r = { ana: { type: C.TYPE.SINGLE, main: v, len: 1 }, took: [{ v, n: 1 }] };
                break;
              }
            }
          }
        }
      }
      if (!r) r = simLead(sim);
      if (!r) return cur === landlordSeat ? 'lord' : 'farmer';
      lastAna = r.ana; lastSeat = cur; pass = 0;
      if (sim.size === 0) return cur === landlordSeat ? 'lord' : 'farmer';
    } else {
      /* rollout 内简化决策: 队友大牌让一让; 敌人临门必压 */
      const meLord = cur === landlordSeat;
      const lastLord = lastSeat === landlordSeat;
      const lastLeft = hands[lastSeat].size;
      const lordLeft = hands[landlordSeat].size;
      let tryBeat = true;
      if (!meLord && !lastLord) {
        /* 农民队友临门(≤3张)或已出大牌/炸: 让队友走, 不抢牌权.
           原来 lastLeft<=2, 队友剩3张(可能是三带/对+单)时也压就抢跑了. */
        if (lastLeft <= 3 || (lastAna.type === C.TYPE.SINGLE && lastAna.main >= V('A')) ||
            lastAna.type === C.TYPE.ROCKET || lastAna.type === C.TYPE.BOMB) tryBeat = false;
      }
      if (!meLord && lastLord && lordLeft <= 2) tryBeat = true;   // 地主临门必压
      if (meLord && !lastLord && lastLeft <= 2) tryBeat = true;   // 农民临门必压
      if (!tryBeat) {
        pass++;
        if (pass >= 2) { cur = lastSeat; lastAna = null; pass = 0; continue; }
        cur = (cur + 1) % 3; continue;
      }
      const r = simBeat(sim, lastAna);
      if (r) {
        // 用炸弹很谨慎: 对方牌还多且我不是只差一手就不炸
        const isBomb = r.ana.type === C.TYPE.BOMB || r.ana.type === C.TYPE.ROCKET;
        const threatLeft = meLord ? Math.min(hands[(cur + 1) % 3].size, hands[(cur + 2) % 3].size) : lordLeft;
        if (isBomb && sim.size > 2 && threatLeft > 4) {
          // 把牌退回去当作 pass
          for (const t of r.took) { sim.cnt[t.v] += t.n; sim.size += t.n; }
          pass++;
          if (pass >= 2) { cur = lastSeat; lastAna = null; pass = 0; continue; }
          cur = (cur + 1) % 3; continue;
        }
        lastAna = r.ana; lastSeat = cur; pass = 0;
        if (sim.size === 0) return cur === landlordSeat ? 'lord' : 'farmer';
      } else {
        pass++;
        if (pass >= 2) { cur = lastSeat; lastAna = null; pass = 0; continue; }
      }
    }
    cur = (cur + 1) % 3;
  }
  return null;
}

/** 评估一次候选出牌: 在 S 个抽样世界里 rollout, 返回胜率 [0,1]; ids=null 表示 pass */
function mcEval(room, seat, ids, lastAna, S) {
  const mySeat = seat;
  const lordSeat = room.landlordSeat;
  const sizes = room.players.map(p => p.hand.length);
  const pool = unseenPool(room, seat);
  const iAmLord = mySeat === lordSeat;
  let win = 0, total = 0;
  for (let s = 0; s < S; s++) {
    // 洗牌抽分布
    const sh = pool.slice();
    for (let i = sh.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [sh[i], sh[j]] = [sh[j], sh[i]]; }
    const hands = [null, null, null];
    hands[mySeat] = handToSim(ids ? room.players[mySeat].hand.filter(c => !ids.includes(c.id)) : room.players[mySeat].hand);
    let idx = 0;
    for (let p = 0; p < 3; p++) {
      if (p === mySeat) continue;
      const n = sizes[p];
      hands[p] = handToSim(sh.slice(idx, idx + n));
      idx += n;
    }
    const ana2 = ids && ids.length ? C.analyze(ids.map(id => room.players[seat].hand.find(c => c.id === id)).filter(Boolean)) : lastAna;
    // pass 时桌面保留 lastAna, lastSeat 仍为原出牌者; 出牌时桌面变成 ana2, lastSeat=me
    const r = rollout(hands, lordSeat, (mySeat + 1) % 3,
      ids ? ana2 : lastAna,
      ids ? mySeat : (room.lastPlay ? room.lastPlay.seat : mySeat));
    total++;
    if (r && ((r === 'lord') === iAmLord)) win++;
  }
  return total ? win / total : 0;
}

/** legend 决策: 枚举候选 + pass, MC 评估选胜率最高 */
function legendDecide(room, seat) {
  const hand = room.players[seat].hand;
  const lp = room.lastPlay;
  const leading = !lp || lp.seat === seat;
  let cands;
  if (leading) {
    // 领牌: 分解方案中的每手 + 基础领牌候选
    const plan = bestDecomp(hand);
    cands = plan.moves.map(m => m.ids);
    for (const ids of leadCandidates(hand, false)) {
      const k = ids.slice().sort((a, b) => a - b).join(',');
      if (!cands.some(x => x.slice().sort((a, b) => a - b).join(',') === k)) cands.push(ids);
    }
    // 出完校验
    const fin = cands.find(ids => ids.length === hand.length);
    if (fin) return { type: 'play', ids: fin };
    /* 2/王是"每张都能稳拿一次牌权"的控制牌. 手牌还很散(分解手数多,
       后面必然要靠 2/王 收回牌权)时, 一手牌里塞 >=2 张 2 属于把多张
       保险一次烧掉 (r5389-3: 地主开局 2224, 之后满手散牌再没拿过
       牌权). 手数 ≤4(接近走完, 需要快速倒牌)或这是唯一候选时豁免. */
    if (plan.count > 4) {
      const kept = cands.filter(ids => {
        const g = groupByVal(hand);
        let twos = 0, jokers = 0;
        for (const id of ids) {
          const v = V(cardById(hand, id).rank);
          if (v === V2) twos++;
          if (v === V('SJ') || v === V('BJ')) jokers++;
        }
        return twos < 2 && jokers < 2;
      });
      if (kept.length) cands = kept;
    }
  } else {
    cands = beatCandidates(hand, lp.ana);
    const fin = cands.find(ids => ids.length === hand.length);
    if (fin) return { type: 'play', ids: fin };
    cands.push(null);   // pass 也是候选
    if (!cands.length || (cands.length === 1 && cands[0] === null)) return { type: 'pass' };
    /* 农民队友剩 1 张: 让他走完. MC 不知道队友手里是什么 (抽样按 unseen 均分),
       会算出"我压大牌能赢"的假高胜率; 但队友这手出完就赢, 规则直接 veto.
       剩 2-3 张时交给 MC 判断 (他可能还得再出几手). */
    const lastSeat = lp.seat;
    const meFarmer = seat !== room.landlordSeat;
    const lastFarmer = lastSeat !== room.landlordSeat;
    if (meFarmer && lastFarmer && room.players[lastSeat].hand.length <= 1) {
      return { type: 'pass' };
    }
    if (meFarmer && lastFarmer) {
      const lordLeft = room.players[room.landlordSeat].hand.length;

      /* 队友的牌没人接得住, 让他站住(r6432 步12: 队友对KK 没人能压时
         我对22 去压纯属内耗; 步24: 队友单2, 全场只剩大小王, 队友这张 2
         已经封顶). 用记牌判断: lp.main 之上同点数在外还剩几张 + 更高
         main 的同型组合是否还能拼出来. */
      {
        const un = unseenCounts(room, seat);
        const la = lp.ana;
        let canBeBeaten = false;
        /* 同型更高组合是否可能存在于外面 (队友+地主):
           单张/对子/三张/三带/四带 看 main 以上的点是否还够同型张数.
           注意 un 是"除我以外"的牌, 队友自己的剩余手牌也在其中——他可能
           持有更大的同型, 但那仍是"农民方胜", 地主压不了就不抢. */
        const bodyNeed = { [C.TYPE.SINGLE]: 1, [C.TYPE.PAIR]: 2, [C.TYPE.TRIPLE]: 3, [C.TYPE.TRIPLE_1]: 3, [C.TYPE.TRIPLE_2]: 3, [C.TYPE.BOMB]: 4 }[la.type];
        if (bodyNeed) {
          /* 地主已 pass 过这手 -> 队友必赢, 不用算牌 */
          if (room.passCount >= 1) {
            canBeBeaten = false;
          } else {
            for (let v = la.main + 1; v < 15; v++) {
              if (un[v] >= bodyNeed) { canBeBeaten = true; break; }
            }
          }
        } else {
          /* 顺子/连对/飞机/四带: 保守认为可能被压, 不用本规则 */
          canBeBeaten = true;
        }
        /* 外面没有更高的同型, 也没有炸弹/王炸能压 -> 队友这手必胜, 绝不抢 */
        if (!canBeBeaten) {
          let bombOut = un[V('SJ')] === 1 && un[V('BJ')] === 1;
          if (!bombOut) for (let v = 0; v < V2; v++) if (un[v] >= 4) { bombOut = true; break; }
          if (!bombOut) return { type: 'pass' };
        }
      }

      /* 队友出中小牌: 牌权已在农民方. 比较"压牌后我剩余手牌的完整结构数",
         若最小压牌都要拆散结构(r3894: 唯一的对QQ 压队友对77)或动 ≥A 大牌,
         就让队友继续. 例外: 我能走完或地主临门. */
      if (lordLeft > 3) {
        const countStructs = h => {
          const g = groupByVal(h);
          let n = 0;
          for (const [, cs] of g) if (cs.length >= 2) n++;
          return n;
        };
        const myStructs = countStructs(hand);
        const nonBomb = cands.filter(ids => {
          if (!ids) return false;
          const a = anaOfIds(hand, ids);
          return a.type !== C.TYPE.BOMB && a.type !== C.TYPE.ROCKET;
        });
        /* 弹药牌(main ≥ A: 单A/2/王, 对A/对2, 三张A…)是留着压地主大牌的,
           花在队友身上内耗. 从候选中剔除(压这手能一次倒 ≥6 张的保留——
           接近走完, 抢牌权连出更值), 剔完只剩 pass 就直接 pass. */
        const culled = nonBomb.filter(ids => {
          const a = anaOfIds(hand, ids);
          if (a.main >= V('A') && ids.length <= 5) return false;
          /* 同型大跳压(≤3 张的净牌: 对子/三张): 队友小三带/对子, 我高出
             ≥3 级的同型留着压地主同型才值钱 (r6432 步3 QQQ 压 333). */
          if (ids.length <= 3 && a.type === lp.ana.type && a.main - lp.ana.main >= 3) return false;
          return true;
        });
        const kept = culled.concat(cands.filter(ids => ids === null));
        if (!culled.length) return { type: 'pass' };
        cands = kept;   /* 炸弹也一并拿掉: 炸队友几乎永远是浪费
                           (要一把走完时 fin 已提前返回) */
        /* 剩余最小压牌仍要拆散结构(失去 ≥2 个对子/三张)压中小牌 -> 让 */
        let bestCheap = null;
        for (const ids of culled) {
          const a = anaOfIds(hand, ids);
          const rest = hand.filter(c => !ids.includes(c.id));
          const lostStructs = myStructs - countStructs(rest);
          const cost = a.main + lostStructs * 15;
          if (!bestCheap || cost < bestCheap.cost) bestCheap = { cost, main: a.main, lostStructs };
        }
        if (bestCheap && lp.ana.main < V('K') && bestCheap.lostStructs >= 2) {
          return { type: 'pass' };
        }
      }
    }

    /* 农民压地主的"过牌试探": 地主出最小的三带/对子(333/33x 之类),
       我用 top 级同型(QQQ/AA 以上, 差 ≥6 级)去压, 是在浪费压地主大牌
       的弹药——地主牌还多, 他大概率有 AAA/22 收走. r6432 步3: AI·乙
       出 QQQ4 压地主 3334, 下轮地主 AAA5 直接收走, QQQ 白死.
       例外: 我这手能走完 / 地主剩 ≤5 张(他快赢了必压) / 我牌多到
       可以挥霍. */
    if (meFarmer && !lastFarmer) {
      const lordLeft = room.players[room.landlordSeat].hand.length;
      const la = lp.ana;
      const smallProbe = (la.type === C.TYPE.TRIPLE || la.type === C.TYPE.TRIPLE_1 || la.type === C.TYPE.TRIPLE_2 || la.type === C.TYPE.PAIR) && la.main <= V('7');
      if (smallProbe && lordLeft > 5 && hand.length > 8) {
        /* 我的最小同型压牌 main 是多少? 若最小压牌本身已经是"大牌
           (≥Q, 差≥6 级)", 说明我手里没有便宜的对子/三张可出——硬压
           等于烧弹药. 有便宜压牌时交给 MC 选择. */
        let minSame = null;
        for (const ids of cands) {
          if (!ids) continue;
          const a = anaOfIds(hand, ids);
          if (a.type === C.TYPE.BOMB || a.type === C.TYPE.ROCKET) continue;
          if (a.type !== la.type || a.main <= la.main) continue;
          if (minSame === null || a.main < minSame) minSame = a.main;
        }
        if (minSame !== null && minSame - la.main >= 6 && minSame >= V('Q')) {
          return { type: 'pass' };
        }
      }
    }

    /* 地主用炸/王炸压农民非临门牌: MC 的 rollout 快棋看不出"炸留到
       残局才值钱", 会把炸当普通大牌算胜率. r9662-3 步4: 地主王炸压
       农民 JJJ5(他还剩13张), 下轮自己只出个 9, 炸白烧.
       规则: 被压的农民 > 4 张(不会下轮走完)且这手 main < 2(不是 2/王
       级必压大牌)时, 把炸/王炸从候选剔除——除非我炸完手里 ≤2 张
       (即将走完)或农民已经 ≤4 张必须拦. */
    if (!meFarmer) {
      const lastLeft = room.players[lastSeat].hand.length;
      const la = lp.ana;
      if (lastLeft > 4 && la.main < V2 && la.type !== C.TYPE.BOMB && la.type !== C.TYPE.ROCKET) {
        const nonBombKept = cands.filter(ids => {
          if (!ids) return true;
          const a = anaOfIds(hand, ids);
          if (a.type === C.TYPE.BOMB || a.type === C.TYPE.ROCKET) {
            /* 炸完剩 ≤2 张(这手出完只余一手牌)才值得炸 */
            return hand.length - ids.length <= 2;
          }
          return true;
        });
        /* 剔完还有非炸候选或 pass 就用剔后的; 全剔光就 pass */
        if (nonBombKept.length !== cands.length) {
          cands = nonBombKept.some(ids => ids !== null) ? nonBombKept : [null];
        }
      }
    }
  }
  // 剪枝: 候选太多时只保留有代表性的 (每种 type 取最小 + 最大 + 分解方案内)
  if (cands.length > 10) {
    const byType = new Map();
    for (const ids of cands) {
      if (!ids) continue;
      const ana = anaOfIds(hand, ids);
      const key = ana.type + ':' + ana.len;
      if (!byType.has(key)) byType.set(key, []);
      byType.get(key).push(ids);
    }
    const keep = [];
    for (const arr of byType.values()) { keep.push(arr[0]); if (arr.length > 1) keep.push(arr[arr.length - 1]); }
    if (cands.includes(null)) keep.push(null);
    cands = keep;
  }
  const deadline = Date.now() + 280;
  const S = 14;
  let bestIds = cands[0], bestWin = -1;
  for (const ids of cands) {
    const w = mcEval(room, seat, ids, leading ? null : lp.ana, S);
    if (w > bestWin) { bestWin = w; bestIds = ids; }
    if (Date.now() > deadline) break;
  }
  if (bestIds === null) return { type: 'pass' };
  return { type: 'play', ids: bestIds };
}

/* ---------- 总决策 ---------- */
function decide(room, seat) {
  const diff = DIFFS.includes(room.difficulty) ? room.difficulty : 'normal';
  const hand = room.players[seat].hand;
  if (room.phase === 'bidding') {
    return { type: 'bid', bid: chooseBid(hand, room.currentBid, diff) };
  }
  if (diff === 'legend') return legendDecide(room, seat);

  const lp = room.lastPlay;
  if (!lp || lp.seat === seat) {
    /* 剩一手能出完直接走 (三带二/连对/飞机等非整炸牌型, cand 里可能没有) */
    if (diffAtLeast(diff, 'hard')) {
      const plan = bestDecomp(hand);
      const fin = plan.moves.find(m => m.ids.length === hand.length);
      if (fin) return { type: 'play', ids: fin.ids };
    }
    const ids = diff === 'expert' ? expertLead(hand, room, seat) : leadPlay(hand, room, seat, diff);
    return { type: 'play', ids };
  }

  const lastAna = lp.ana;
  let ids;
  if (diff === 'expert') {
    ids = pickBeat(hand, room, seat, lastAna);
    if (ids && !expertShouldBeat(room, seat, hand, lastAna, ids)) return { type: 'pass' };
    if (!ids) return { type: 'pass' };
    return { type: 'play', ids };
  }

  ids = C.hintPlay(hand, lastAna);

  /* 一把出完 (hint 只找同型最小牌, 可能漏了三带二/连对等收尾牌型) */
  if (ids && ids.length === hand.length) return { type: 'play', ids };
  if (!ids) {
    const anaAll = C.analyze(hand);
    if (anaAll.type !== C.TYPE.INVALID && C.canBeat(lastAna, anaAll)) {
      return { type: 'play', ids: hand.map(c => c.id) };
    }
  }

  /* hard: 若只能拆炸弹压, 但牌面不值得, 则保留炸弹让一手 */
  if (ids && diffAtLeast(diff, 'hard')) {
    const ana2 = anaOfIds(hand, ids);
    const wasteBomb = (ana2.type === C.TYPE.BOMB || ana2.type === C.TYPE.ROCKET)
      && lastAna.type !== C.TYPE.BOMB && lastAna.type !== C.TYPE.ROCKET
      && hand.length > ids.length + 3 && playUnits(hand) > 2;
    const lastFarmer = lp.seat !== room.landlordSeat;
    const meFarmer = seat !== room.landlordSeat;
    if (wasteBomb && !(meFarmer && !lastFarmer && room.players[room.landlordSeat].hand.length <= 3)) {
      return { type: 'pass' };
    }
  }
  if (!ids || !shouldBeat(room, seat, hand, lastAna, ids, diff)) return { type: 'pass' };
  return { type: 'play', ids };
}

module.exports = { decide, handScore, leadPlay, bestDecomp };
