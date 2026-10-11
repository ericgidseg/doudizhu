'use strict';
/**
 * AI 决策 —— 单难度 (传说)
 *   候选生成 + 蒙特卡洛模拟选牌, 上面再叠一层规则 veto 修正
 *   MC 在确定性局面的误判 (农民配合 / 炸弹与 2 的使用时机).
 */
const C = require('./cards.js');
const V = r => C.RANK_VALUE[r];
const V2 = V('2');

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
/* ============ 结构规划 ============ */
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
  /* memo 键必须含"当前还剩几个整炸"——否则"顺子拆炸到 cnt[v]=0"
     和"整炸出掉 cnt[v]=0"被当成同一状态, 后者偷用前者的缓存
     (缓存里没有炸弹). r4446-1: 顺子 34567 先拆 6, DP 到达
     cnt[6]=0 后缓存了"无炸"最优解, 整炸 6666 分支随后到达同一
     状态被 memo 短路, 炸弹白丢. */
  const origBombs = [];
  for (const v of g.keys()) if (cardPool.get(v).length === 4) origBombs.push(v);
  const keyOf = (cntArr, bf) => {
    let bl = 0;
    for (const v of origBombs) if (cntArr[v] === 4) bl++;
    return cntArr.join(',') + '|' + bf + '|' + bl;
  };
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

    /* 炸弹优先探: 整炸分支放在 combos 之前——它是"保留保险"的走法,
       先递归它能把"留炸"的子树最优解先算出来缓存; 放最后的话前面
       combo 递归吃掉大量状态数, bomb 子树容易撞上 abort 阈值被丢
       (r4446-1: bomb v=3 的 sub 返回 moves:null 被 tryMove 丢弃). */
    for (let v = 0; v < 15; v++) {
      if (cntArr[v] === 4) {
        const next = cntArr.slice(); next[v] = 0;
        tryMove(-30, next, bf, { bomb: v });   // 炸弹白赚 30 分
      }
    }

    for (const co of combos) {
      // 主体可用性
      let ok = true;
      for (const t of co.takes) if (cntArr[t.v] < t.take) { ok = false; break; }
      if (!ok) continue;
      /* 拆弹惩罚: 任何 combo 只要"部分使用"了某个原本是 4 张的
         同点(看 cardPool 不看当前剩余——顺子先拆走一张后, 后面的
         combo 用剩余 3 张也算拆炸), 就吃重罚. 炸弹 ×2+压一切的
         价值远超多凑一手散牌. r4446-1: 四个 6 被拆成 34567 顺 +
         666 三带一, 顺子先用一张 6, 三带一再用 3 张——两层都该罚.
         整炸走 bomb 分支 (cost -30) 不受此罚. */
      let bombPenalty = 0;
      for (const t of co.takes) {
        if (cardPool.get(t.v).length === 4 && t.take < 4) {
          bombPenalty += 120;
        }
      }
      /* 封顶牌惩罚: 主体每消耗一张 >=2 的牌, 等于烧掉一次未来的
         "必拿牌权"——222/AAA 是三张级天花板, 单出/拆对都是牌权.
         但惩罚不能太重: 三张 2 合起来还能压任何三张结构, 拆成单张
         反而丧失功能. r1446-2 步11: 222带BJ 一手甩出去, BJ 白进
         坟场; 正确是 222 带小牌当三带一 (主体保留) + BJ 单出. */
      let anchorPenalty = 0;
      for (const t of co.takes) if (t.v >= V2) anchorPenalty += t.take * 8;
      // 翅膀占用: 顺序取最小可用点 —— 在 DP 里枚举翅膀代价太高,
      // 近似: 用"当前剩余中最小的合法散牌"占位扣减.
      // 翅膀禁止用 >=2 的牌: 2/王单独都是牌权, 塞进翅膀是纯浪费.
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
          // 且不碰 >=2 (那是牌权不是散牌)
          let left = need.singles;
          for (let v = 0; v < V2 && left > 0; v++) {
            if (bodyVals.has(v)) continue;
            while (next[v] > 0 && left > 0) { next[v]--; left--; kickCost += w(v); }
          }
          if (left > 0) continue; // 带不动 (宁可放弃, 不拆主体)
        } else if (need.pairs) {
          let left = need.pairs;
          for (let v = 0; v < V2 && left > 0; v++) {
            if (bodyVals.has(v)) continue;
            if (next[v] >= 2) { next[v] -= 2; left--; kickCost += 2 * w(v); }
          }
          if (left > 0) continue;
        }
      }
      tryMove(bodyCost + kickCost + bombPenalty + anchorPenalty, next, bf, { co });
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

/* ---------- 记牌 ---------- */
function unseenCounts(room, seat) {
  const g = groupByVal(room.players[seat].hand);
  const un = new Array(15).fill(0);
  for (let v = 0; v < 15; v++) un[v] = 4;
  un[V('SJ')] = 1; un[V('BJ')] = 1;
  for (const c of room.played) un[V(c.rank)]--;
  for (const [v, cs] of g) un[v] -= cs.length;
  return un;  // 其余两家合计的剩余张数
}

/** 这手牌在外面的两家手里是否已经无人能压 (含炸弹/王炸检查) */
function unbeatableOutside(room, seat, ana) {
  const un = unseenCounts(room, seat);
  const rocketOut = un[V('SJ')] === 1 && un[V('BJ')] === 1;
  let bombOut = rocketOut;
  if (!bombOut) for (let v = 0; v < V2; v++) if (un[v] >= 4) { bombOut = true; break; }
  if (ana.type === C.TYPE.ROCKET) return true;
  if (ana.type === C.TYPE.BOMB) {
    if (rocketOut) return false;
    for (let v = ana.main + 1; v < V2; v++) if (un[v] >= 4) return false;
    return true;
  }
  if (bombOut) return false;
  const need = {
    [C.TYPE.SINGLE]: 1, [C.TYPE.PAIR]: 2, [C.TYPE.TRIPLE]: 3,
    [C.TYPE.TRIPLE_1]: 3, [C.TYPE.TRIPLE_2]: 3,
    [C.TYPE.FOUR_2]: 4, [C.TYPE.FOUR_2_2]: 4,
  }[ana.type];
  if (need) {
    for (let v = ana.main + 1; v < 15; v++) if (un[v] >= need) return false;
    return true;
  }
  if (ana.type === C.TYPE.CHAIN || ana.type === C.TYPE.PAIR_CHAIN ||
      ana.type === C.TYPE.AIRPLANE || ana.type === C.TYPE.AIRPLANE_1 || ana.type === C.TYPE.AIRPLANE_2) {
    const per = ana.type === C.TYPE.CHAIN ? 1 : ana.type === C.TYPE.PAIR_CHAIN ? 2 : 3;
    const width = ana.type === C.TYPE.CHAIN ? ana.len :
      ana.type === C.TYPE.PAIR_CHAIN ? ana.len / 2 :
      ana.type === C.TYPE.AIRPLANE ? ana.len / 3 :
      ana.type === C.TYPE.AIRPLANE_1 ? ana.len / 4 : ana.len / 5;
    for (let hi = ana.main + 1; hi + width - 1 < V2; hi++) {
      let ok = true;
      for (let v = hi; v < hi + width; v++) if (un[v] < per) { ok = false; break; }
      if (ok) return false;
    }
    return true;
  }
  return false;
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
function chooseBid(hand, currentBid) {
  // 牌力分 + 分解手数加成 (手数少也是叫分资本)
  const s = handScore(hand);
  const d = bestDecomp(hand);
  const eff = s + Math.max(0, 8 - d.count) * 0.6;
  const want = eff >= 6 ? 3 : eff >= 4.2 ? 2 : eff >= 2.6 ? 1 : 0;
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
    const s = lowSingles(g, run3.length, ex, exv);
    const p = lowPairs(g, run3.length, ex, exv);
    if (!s && !p) push(body);  // 仅在无翅膀可带时才裸飞机
    if (s) push(body.concat(s));
    if (p) push(body.concat(p));
  }
  for (const v of [...g.keys()].sort((a, b) => a - b)) {
    if (g.get(v).length >= 3 && g.get(v).length < 4) {
      const tri = g.get(v).slice(0, 3).map(c => c.id);
      const ex = new Set([...tri, ...bombSet]);
      const exv = new Set([v]);
      const s = lowSingles(g, 1, ex, exv);
      const p = lowPairs(g, 1, ex, exv);
      if (!s && !p) push(tri); // 仅在无杂牌可带时才裸三张
      if (s) push(tri.concat(s));
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
/** 手牌能否一手走完; lastAna=null 表示领出, 否则须压过 lastAna. 返回 ana 或 null */
function simOneShot(sim, lastAna) {
  if (sim.size === 0) return null;
  const cards = [];
  for (let v = 0; v < 15; v++) {
    for (let k = 0; k < sim.cnt[v]; k++) cards.push({ id: v * 8 + k, rank: RANK_BY_V[v], suit: '' });
  }
  const ana = C.analyze(cards);
  if (ana.type === C.TYPE.INVALID) return null;
  if (lastAna && !C.canBeat(lastAna, ana)) return null;
  return ana;
}

/** 某家剩余手牌能否压过 ana (rollout 内做农民配合判断用, 不改变 sim) */
function handCanBeat(sim, ana) {
  const probe = { cnt: sim.cnt.slice(), size: sim.size };
  return !!simBeat(probe, ana);
}

/**
 * 农民喂牌: 队友临门 (剩 1~2 张) 且这手轮到我领出时, 挑一张队友
 * 压得住的牌送出去 —— 用队友的真实剩余手牌判断, 避免"送了他却接不住"
 * (例如队友剩单 3, 我却领单 4, 队友反而被卡住). 返回 {ana,took} 或 null.
 */
function simFeedMate(sim, mateSim, lordSim, mateLeft) {
  const vals = [];
  for (let v = 0; v < 15; v++) if (sim.cnt[v] > 0) vals.push(v);
  if (!vals.length) return null;
  const lordVals = [];
  for (let v = 0; v < 15; v++) if (lordSim.cnt[v] > 0) lordVals.push(v);
  const lordMax = lordVals.length ? lordVals[lordVals.length - 1] : -1;
  const lordRocket = lordSim.cnt[V('SJ')] > 0 && lordSim.cnt[V('BJ')] > 0;
  const mateVals = [];
  for (let v = 0; v < 15; v++) if (mateSim.cnt[v] > 0) mateVals.push(v);
  if (!mateVals.length) return null;
  const mateMax = mateVals[mateVals.length - 1];
  const take = (v, n) => {
    sim.cnt[v] -= n; sim.size -= n;
    return { v, n };
  };
  /* 队友剩 2 张且成对: 优先送比它小的对子, 让他直接对子走完 */
  if (mateLeft === 2 && mateSim.cnt[mateMax] === 2) {
    let best = -1;
    for (let v = 0; v < mateMax; v++) {
      if (sim.cnt[v] >= 2 && (v > lordMax || lordRocket)) best = v;   // 地主压不住的对子最理想
    }
    if (best < 0) for (let v = mateMax - 1; v >= 0; v--) if (sim.cnt[v] >= 2) { best = v; break; }
    if (best >= 0) {
      return { ana: { type: C.TYPE.PAIR, main: best, len: 2 }, took: [take(best, 2)] };
    }
  }
  /* 送单张: 必须比队友最大的一张牌小, 他才能压住; 优先地主压不住的 */
  const cands = vals.filter(v => v < mateMax);
  if (cands.length) {
    const safe = cands.filter(v => v > lordMax || lordRocket);
    const pick = safe.length ? safe[safe.length - 1] : cands[cands.length - 1];
    return { ana: { type: C.TYPE.SINGLE, main: pick, len: 1 }, took: [take(pick, 1)] };
  }
  return null;
}

function rollout(hands, landlordSeat, cur, lastAna, lastSeat, maxSteps) {
  let pass = 0, steps = 0;
  while (steps++ < (maxSteps || 300)) {
    const sim = hands[cur];
    if (sim.size === 0) {
      return cur === landlordSeat ? 'lord' : 'farmer';
    }
    if (!lastAna || lastSeat === cur) {
      /* 能一手走完就立刻走完 (残局精确性: 否则 rollout 会多算好几步,
         低估"报单/报双"的必胜性). */
      if (simOneShot(sim, null)) return cur === landlordSeat ? 'lord' : 'farmer';
      /* 农民领出: 队友只剩1-2张时优先喂小单/小对子让他接牌走完 */
      let r = null;
      if (cur !== landlordSeat) {
        const mateSeat = 3 - cur - landlordSeat;
        const mateLeft = hands[mateSeat].size;
        /* 队友只剩 1-2 张且这手牌权在我: 优先喂他. 用队友的剩余手牌
           真实判断他压不压得住, 而不是盲猜 (盲喂单 4 给剩单 3 的队友
           等于卡死他). 地主报单时不喂单, 改出对子/长牌逼地主. */
        if (mateLeft === 1 || mateLeft === 2) {
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
            lastAna.type === C.TYPE.ROCKET || lastAna.type === C.TYPE.BOMB) {
          tryBeat = false;
        }
      }
      if (!meLord && lastLord && lordLeft <= 2) tryBeat = true;   // 地主临门必压
      if (meLord && !lastLord && lastLeft <= 2) tryBeat = true;   // 农民临门必压
      if (!tryBeat) {
        pass++;
        if (pass >= 2) { cur = lastSeat; lastAna = null; pass = 0; continue; }
        cur = (cur + 1) % 3; continue;
      }
      /* 能一手压完就压完 (残局精确性, 同领出). */
      if (simOneShot(sim, lastAna)) return cur === landlordSeat ? 'lord' : 'farmer';
      const r = simBeat(sim, lastAna);
      if (r) {
        // 用炸弹很谨慎: 对方牌还多且我不是只差一手就不炸
        const isBomb = r.ana.type === C.TYPE.BOMB || r.ana.type === C.TYPE.ROCKET;
        const threatLeft = meLord ? Math.min(hands[(cur + 1) % 3].size, hands[(cur + 2) % 3].size) : lordLeft;
        /* 临门长牌必接: 农民眼看地主只剩一手就要跑完时, 炸弹也得交;
           "留炸"不能变成"陪跑". */
        const urgent = !meLord && lastLord && lordLeft <= 5 && lastAna.len >= 5;
        if (isBomb && sim.size > 2 && threatLeft > 4 && !urgent) {
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

/* 预生成 K 个"对手暗牌世界": 除我之外的暗牌随机分给另外两家.
   同一批世界用于评估所有候选 (共同随机数 CRN) —— 候选之间的比较
   变成配对比较, 比各候选独立抽样噪声低得多. */
function genWorlds(room, seat, K) {
  const pool = unseenPool(room, seat);
  const sizes = room.players.map(p => p.hand.length);
  const worlds = [];
  for (let s = 0; s < K; s++) {
    const sh = pool.slice();
    for (let i = sh.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [sh[i], sh[j]] = [sh[j], sh[i]]; }
    const hands = [null, null, null];
    let idx = 0;
    for (let p = 0; p < 3; p++) {
      if (p === seat) continue;
      const n = sizes[p];
      hands[p] = handToSim(sh.slice(idx, idx + n));
      idx += n;
    }
    worlds.push(hands);
  }
  return worlds;
}

/** 在给定世界批上评估一个候选出牌, 返回胜率 [0,1]; ids=null 表示 pass */
function mcEvalWorlds(room, seat, ids, lastAna, worlds) {
  const mySeat = seat;
  const lordSeat = room.landlordSeat;
  const iAmLord = mySeat === lordSeat;
  const mySim = handToSim(ids ? room.players[mySeat].hand.filter(c => !ids.includes(c.id)) : room.players[mySeat].hand);
  const ana2 = ids && ids.length ? C.analyze(ids.map(id => room.players[seat].hand.find(c => c.id === id)).filter(Boolean)) : lastAna;
  const lastSeat = ids ? mySeat : (room.lastPlay ? room.lastPlay.seat : mySeat);
  let win = 0;
  for (const w of worlds) {
    const hands = [null, null, null];
    hands[mySeat] = { cnt: mySim.cnt.slice(), size: mySim.size };
    for (let p = 0; p < 3; p++) if (p !== mySeat) hands[p] = { cnt: w[p].cnt.slice(), size: w[p].size };
    // pass 时桌面保留 lastAna, lastSeat 仍为原出牌者; 出牌时桌面变成 ana2, lastSeat=me
    const r = rollout(hands, lordSeat, (mySeat + 1) % 3, ids ? ana2 : lastAna, lastSeat);
    if (r && ((r === 'lord') === iAmLord)) win++;
  }
  return worlds.length ? win / worlds.length : 0;
}

/** legend 决策: 枚举候选 + pass, MC 评估选胜率最高 */
function legendDecide(room, seat) {
  const hand = room.players[seat].hand;
  const lp = room.lastPlay;
  const leading = !lp || lp.seat === seat;
  /* 位置角色: 地主下家(顺位)负责溜小牌跑得快; 地主上家(顶家)要拿
     中大单/对顶住地主, 别把小单送到地主手里白捡牌权 (职业打法里
     顶家的核心职责就是消耗地主的 A/2). */
  const iAmLord = seat === room.landlordSeat;
  const isGuard = !iAmLord && seat === (room.landlordSeat + 2) % 3;
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
    /* 两手必胜 (先手必胜): 手牌恰好拆成两手, 其中一手全场无人能压.
       先出那手必赢的 -> 无人能接, 牌权还在我手里 -> 剩下那手直接走完.
       这是确定性收益, MC 抽样看不到, 直接短路. */
    if (plan.count === 2 && plan.moves.length === 2) {
      for (const m of plan.moves) {
        if (unbeatableOutside(room, seat, m.ana)) return { type: 'play', ids: m.ids };
      }
    }
    /* 顶家领单张 (阈值 9): 没到残局时不溜 <9 的小单, 优先从 9~K 里挑;
       这类牌既能领走又真的消耗地主的 A/2. 全是小单时领最大那张,
       也不能白送. */
    if (isGuard && hand.length > 6) {
      let maxSmall = null;
      const kept = cands.filter(ids => {
        if (!ids || ids.length !== 1) return true;
        const a = anaOfIds(hand, ids);
        if (a.type !== C.TYPE.SINGLE || a.main >= V('2')) return true;
        if (a.main >= V('9')) return true;
        if (!maxSmall || a.main > maxSmall.main) maxSmall = { ids, main: a.main };
        return false;
      });
      if (kept.length) cands = kept;
      else if (maxSmall) return { type: 'play', ids: maxSmall.ids };
    }
    /* 有"必赢的长牌型"时, 单领 2/SJ/BJ 是浪费: 长牌型一次甩 ≥5 张且
       同样稳拿牌权 (r5389-1 步15: 地主 11 张里有 678910JQKA 顺子
       没人接得住, MC 却领单张 BJ——王只甩 1 张, 顺子甩 8 张).
       把单张 2/王 领出剔除 (单张 3-K 仍可做试探保留). */
    {
      const un = unseenCounts(room, seat);
      /* 炸弹/王炸可能在外 -> 没有"必赢"牌型 */
      let bombOut = un[V('SJ')] === 1 && un[V('BJ')] === 1;
      if (!bombOut) for (let v = 0; v < V2; v++) if (un[v] >= 4) { bombOut = true; break; }
      const unbeat = !bombOut && cands.find(ids => {
        if (ids.length < 5) return false;
        const a = anaOfIds(hand, ids);
        if (a.type === C.TYPE.BOMB || a.type === C.TYPE.ROCKET) return false;
        /* 同型更高组合是否还可能拼出来:
           顺子(len张连续, 需 main 之上的等长窗口各点>=1)
           连对(len/2 个连续对, 窗内各点>=2)
           飞机(连续三张组, 窗内各点>=3)
           三张/四带: main 之上任意点 >=3/4 张即可 */
        if (a.type === C.TYPE.CHAIN || a.type === C.TYPE.PAIR_CHAIN || a.type === C.TYPE.AIRPLANE || a.type === C.TYPE.AIRPLANE_1 || a.type === C.TYPE.AIRPLANE_2) {
          const need = a.type === C.TYPE.CHAIN ? 1 : a.type === C.TYPE.PAIR_CHAIN ? 2 : 3;
          const width = a.type === C.TYPE.CHAIN ? a.len : a.type === C.TYPE.PAIR_CHAIN ? a.len / 2 : a.len / (a.type === C.TYPE.AIRPLANE ? 3 : a.type === C.TYPE.AIRPLANE_1 ? 4 : 5);
          for (let hi = a.main + 1; hi + width - 1 < V2; hi++) {
            let ok = true;
            for (let v = hi; v < hi + width; v++) if (un[v] < need) { ok = false; break; }
            if (ok) return false;   // 外面能拼出更高的同型
          }
          return true;
        }
        if (a.type === C.TYPE.TRIPLE || a.type === C.TYPE.TRIPLE_1 || a.type === C.TYPE.TRIPLE_2) {
          for (let v = a.main + 1; v < V2; v++) if (un[v] >= 3) return false;
          return true;
        }
        if (a.type === C.TYPE.FOUR_2 || a.type === C.TYPE.FOUR_2_2) {
          for (let v = a.main + 1; v < V2; v++) if (un[v] >= 4) return false;
          return true;
        }
        return false;   // 单张/对子走不到这里 (len<5)
      });
      if (unbeat) {
        const kept = cands.filter(ids => {
          if (ids.length !== 1) return true;
          const v = V(cardById(hand, ids[0]).rank);
          return v < V2;   // 剔掉单张 2/小王/大王 领出
        });
        if (kept.length) cands = kept;
      }
    }
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
    /* 小牌先行, 大牌护航 (回马枪战术):
       手牌手数较多 (plan.count >= 3) 且非残局临门时, 若手里同时有同类型 (对子/三带) 的
       小牌 (main < 10) 与顶门大牌 (main >= A), 领牌时禁止先打该类型的大牌.
       大牌是用来收回牌权或掩护小牌的, 盲目先出大牌会导致后续小对/小三带沦为累赘,
       缺乏收牌手段 (r8500-2: 地主出对7溜过之后握有绝对牌权, 却先打对A导致对3被顶死).
       仅当手里该牌型全是 >= A 的牌 (无小牌可溜) 时豁免. */
    if (plan.count >= 3) {
      const types = [C.TYPE.PAIR, C.TYPE.TRIPLE, C.TYPE.TRIPLE_1, C.TYPE.TRIPLE_2];
      for (const t of types) {
        let hasSmall = false;
        for (const ids of cands) {
          if (!ids) continue;
          const a = anaOfIds(hand, ids);
          if (a.type === t && a.main < V('10')) { hasSmall = true; break; }
        }
        if (hasSmall) {
          const kept = cands.filter(ids => {
            if (!ids) return true;
            const a = anaOfIds(hand, ids);
            if (a.type === t && a.main >= V('A')) return false;
            return true;
          });
          if (kept.length) cands = kept;
        }
      }
    }

    /* 残局全单张 + 手里捏着 >=2 级封顶牌: 领最小的那张.
       农民要压你得交大牌, 你的大牌留着下一轮收牌权; 先甩大牌
       (K/A) 反而是"帮农民消耗你"——他 2/王 接走后你连小的都出不掉.
       r4424-3 步17: 地主剩 {10,J,A,2} 领 A, seat1 用 2 抢走牌权;
       若领 10, 农民得交 Q/K/A 接, 地主的 A+2 双保险留着稳收. */
    if (seat === room.landlordSeat && hand.length <= 5) {
      const sorted = C.sortCards(hand);
      const isAllSingles = sorted.every((c, i, arr) =>
        i === 0 || C.RANK_VALUE[c.rank] !== C.RANK_VALUE[arr[i - 1].rank]);
      const hasAnchor = sorted.some(c => C.RANK_VALUE[c.rank] >= V2);
      const minSingle = cands.find(ids =>
        ids && ids.length === 1 && ids[0] === sorted[sorted.length - 1].id);
      if (isAllSingles && hasAnchor && minSingle) {
        return { type: 'play', ids: minSingle };
      }
    }

    /* 农民领牌面对地主报单/残局 (地主剩 1~2 张) 的防守:
       1) 地主报单 (lordLeft === 1):
          - 地主只有1张牌, 绝不能单领小单牌送死!
          - 只要手里有非单张牌型 (对子/三带/顺子/连对...), 绝不领单张;
          - 若手牌全散只剩单张: 必须顶出手里最大的单张 (防止送地主小牌过关, 如 r4816-1 步60).
       2) 地主剩 2 张 (lordLeft === 2):
          - 若领单张, 剔除 < 10 的小单牌 (防止送地主白溜走小单张并保留大牌绝杀, 如 r4816-2 步33);
            优先领对子等非单张, 或领 >= 10 的大单牌逼地主交大牌/拆对.
    */
    if (seat !== room.landlordSeat) {
      const lordLeft = room.players[room.landlordSeat].hand.length;
      if (lordLeft === 1) {
        const nonSingle = cands.filter(ids => {
          if (!ids) return false;
          const a = anaOfIds(hand, ids);
          return a.type !== C.TYPE.SINGLE;
        });
        if (nonSingle.length) {
          cands = nonSingle;
        } else {
          let maxSingle = null, maxV = -1;
          for (const ids of cands) {
            if (!ids || ids.length !== 1) continue;
            const v = V(cardById(hand, ids[0]).rank);
            if (v > maxV) { maxV = v; maxSingle = ids; }
          }
          if (maxSingle) return { type: 'play', ids: maxSingle };
        }
      } else if (lordLeft === 2) {
        const safer = cands.filter(ids => {
          if (!ids) return false;
          const a = anaOfIds(hand, ids);
          if (a.type === C.TYPE.SINGLE && a.main < V('10')) return false;
          return true;
        });
        if (safer.length) {
          cands = safer;
        } else {
          let maxSingle = null, maxV = -1;
          for (const ids of cands) {
            if (!ids || ids.length !== 1) continue;
            const v = V(cardById(hand, ids[0]).rank);
            if (v > maxV) { maxV = v; maxSingle = ids; }
          }
          if (maxSingle) cands = [maxSingle];
        }
      }
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

      /* 弹药牌(main ≥ A: 单A/2/王, 对A/对2, 三张A…)是留着压地主大牌的,
         花在队友身上内耗. 从候选中剔除(压这手能一次倒 ≥6 张的保留——
         接近走完, 抢牌权连出更值), 剔完只剩 pass 就直接 pass.
         无论地主剩几张, 动 A/2/王 砸队友都是自毁长城 (r4816-1 步57:
         地主剩1张, AI 拿小王砸队友的 10, 随后送小3给地主赢). */
      const nonBomb = cands.filter(ids => {
        if (!ids) return false;
        const a = anaOfIds(hand, ids);
        return a.type !== C.TYPE.BOMB && a.type !== C.TYPE.ROCKET;
      });
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
      cands = kept;   /* 炸弹也一并拿掉: 炸队友几乎永远是浪费 (要一把走完时 fin 已提前返回) */

      /* 队友出中小牌: 牌权已在农民方. 地主牌还多时, 比较"压牌后我剩余手牌的完整结构数",
         若最小压牌都要拆散结构(r3894: 唯一的对QQ 压队友对77), 就让队友继续. */
      if (lordLeft > 3) {
        const countStructs = h => {
          const g = groupByVal(h);
          let n = 0;
          for (const [, cs] of g) if (cs.length >= 2) n++;
          return n;
        };
        const myStructs = countStructs(hand);
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
      /* 农民 >6 张时连他出的 2/王 也不值得炸——牌权很快会回来;
         他 ≤6 张时 2/王 算临门必压, 炸的候选保留.
         main >= K 的大牌(对A/三K/单A/单2/顺顶K...)本身就是农民顶门牌,
         炸了它对面同型线就断——不是"浪费", 是"收割", 此时炸的候选
         保留不剔. r8233-4 步12: 地主有 5555+王炸 却 pass 农民的对A,
         白送一轮牌权 (MC 显示炸 95.4% > pass 88.7%). */
      const veto = la.main < V('K');
      if (lastLeft > 4 && veto && la.type !== C.TYPE.BOMB && la.type !== C.TYPE.ROCKET) {
        const nonBombKept = cands.filter(ids => {
          if (!ids) return true;
          const a = anaOfIds(hand, ids);
          if (a.type === C.TYPE.BOMB || a.type === C.TYPE.ROCKET) {
            /* 炸完剩 ≤2 张, 或剩牌一手就能走完(如 炸+三带) 才值得炸 */
            const rest = hand.filter(c => !ids.includes(c.id));
            return rest.length <= 2 || bestDecomp(rest).count <= 1;
          }
          return true;
        });
        /* 剔完还有非炸候选或 pass 就用剔后的; 全剔光就 pass */
        if (nonBombKept.length !== cands.length) {
          cands = nonBombKept.some(ids => ids !== null) ? nonBombKept : [null];
        }
      }

      /* 王炸不压小单: 单 4/单 7 这种小牌, 手里只要有能压的普通牌,
         王炸就该留着 (拆开是两次牌权, 合着是残局保险). r5947-1 步21:
         地主拿王炸压农民的单 4, 明明单 2 就能压, 白白烧了倍数和保险.
         例外: 炸完一手直接走完 (剩 ≤2 张), 或除了王炸没有任何牌压得住. */
      if (la.type === C.TYPE.SINGLE && la.main < V('A')) {
        const rkIdx = cands.findIndex(ids => {
          if (!ids) return false;
          const a = anaOfIds(hand, ids);
          return a.type === C.TYPE.ROCKET;
        });
        if (rkIdx >= 0) {
          const rk = cands[rkIdx];
          const rest = hand.filter(c => !rk.includes(c.id));
          const hasCheap = cands.some(ids => {
            if (!ids || ids === rk) return false;
            const a = anaOfIds(hand, ids);
            return a.type !== C.TYPE.ROCKET && a.type !== C.TYPE.BOMB;
          });
          /* 有便宜压牌 且 炸完不能一手走完 -> 剔掉王炸 */
          if (hasCheap && rest.length > 2) cands.splice(rkIdx, 1);
        }
      }

      /* 压小单的分级: 不强制要压——对面甩小牌时地主该权衡"这张值不
         值得我管", 牌还多就让队友方向/下轮再说.
         规则分三层:
         1) 王 (SJ/BJ) 是王炸组件, 手里有 2 或 >=A 的普通单牌时绝不出
            (压小单拆王炸是白烧). r7031-2 步5: 地主拿 BJ 压单 4.
         2) 对面甩 <J 的小单, 我牌还多 (>5 张) 且对面牌也多 (>4 张):
            用 2/王 压是杀鸡用牛刀——大牌留着压对面真正的大牌.
            r7031-2 步5/步11: BJ 压 4, SJ 压 K, 双王两手烧光.
         3) 手里有 2 时 2 是最优封顶牌 (压了对面要么交炸要么让),
            有 2 就不用王; 没 2 时 Q/K/A 也可以压 (不必强行 pass).
         例外: 对面剩 <=2 张临门, 或我 <=3 张急需跑牌, 或除了大牌
         没任何牌压得住——这些情况交给 MC 全候选评估. */
      if (la.type === C.TYPE.SINGLE && la.main < V('2')) {
        const lastLeft = room.players[lastSeat].hand.length;
        const myLeft = hand.length;
        if (lastLeft > 2 && myLeft > 3) {
          const has2 = cands.some(ids =>
            ids && ids.length === 1 &&
            V(cardById(hand, ids[0]).rank) === V('2'));
          const hasAceSingle = cands.some(ids =>
            ids && ids.length === 1 &&
            V(cardById(hand, ids[0]).rank) === V('A'));
          /* 临门小单必压: 对面 <=4 张出单牌是"甩最小留大", 必须用
             >=2 压死, 不能让他再领 (r5947-1 步21: seat0 剩 4 出单 4,
             K 压过去对面 2 收走牌权, 正确是直接用 2 封顶). */
          const mustSeal = lastLeft <= 4;
          const kept = cands.filter(ids => {
            if (!ids) return true;
            const a = anaOfIds(hand, ids);
            if (a.type !== C.TYPE.SINGLE) return true;
            const v = a.main;
            /* 王单出: 有 2 或有 A 都别用 (拆王炸/浪费顶张);
               但临门必压时王也是武器 */
            if (!mustSeal && v >= V('SJ') && (has2 || hasAceSingle)) return false;
            if (mustSeal) {
              /* 临门: 手里有 2 只留 2 (王留作王炸组件),
                 没 2 留王 (最大的单牌兜底) */
              if (v < V('2')) return false;
              if (has2) return v === V('2');
              return true;
            }
            /* 非临门小单 (<J): 用 >=2 压是杀鸡用牛刀 */
            if (la.main < V('J') && v >= V('2') && myLeft > 5 && lastLeft > 6) return false;
            return true;
          });
          if (kept.length) cands = kept;
        }
      }
    }

    /* 农民拿炸/王炸压地主非临门牌同样浪费: r6513-2 步2 农民 10101010
       炸地主开局的 56789 顺子(地主还剩15张), 后面牌权全丢.
       规则: 地主 >4 张且这手 main < 2 且非炸弹时, 剔掉炸/王炸候选——
       除非我炸完 ≤2 张快走完. (地主 ≤4 张不剔: 必须拦.) */
    if (meFarmer && !lastFarmer) {
      const lordLeft = room.players[room.landlordSeat].hand.length;
      const la = lp.ana;

      /* 地主 ≤3 张临门: 能压必压 (除非只能拆炸且这手很小).
         MC 抽样队友暗牌后可能算出"队友会拦"的假安全——队友暗牌
         未必接得住, 我手里有便宜压牌就该我上. r7975-2 步25: 地主
         剩 3 张出 K, 队友有 A/2 却 pass 了.
         注意这里没有"让地主憋大牌"的例外: 地主剩 ≤3 张出大单,
         让他再领牌等于赌他手里没 2——赌输了他小牌+2 连甩直接走完
         (r4424-3: 地主剩 {10,2} 出 A, 农民若 pass 地主必赢; 用 2
         压掉抢牌权才是唯一活路). */
      if (lordLeft <= 3) {
        /* 找最便宜的非炸压牌 */
        let cheap = null;
        for (const ids of cands) {
          if (!ids) continue;
          const a = anaOfIds(hand, ids);
          if (a.type === C.TYPE.BOMB || a.type === C.TYPE.ROCKET) continue;
          const rest = hand.filter(c => !ids.includes(c.id));
          const g0 = groupByVal(hand), g1 = groupByVal(rest);
          let lost = 0;
          for (const [v, cs] of g0) { const left = (g1.get(v) || []).length; if (cs.length >= 2 && left < cs.length && left < 2) lost++; }
          const cost = a.main + lost * 15;
          if (!cheap || cost < cheap.cost) cheap = { cost, ids, a };
        }
        if (cheap) return { type: 'play', ids: cheap.ids };
        /* 只有炸能压: 不拦就输, 炸! (fin 已先检查过一把走完) */
        if (cands.some(Boolean)) return { type: 'play', ids: cands.find(Boolean) };
      }

      /* 地主 >6 张时他的 2/王 也不值得炸——压了这张他还有一堆牌;
         ≤6 张时保留炸 (必须拦) */
      const veto = la.main < V2 || lordLeft > 6;
      if (lordLeft > 4 && veto && la.type !== C.TYPE.BOMB && la.type !== C.TYPE.ROCKET) {
        const kept = cands.filter(ids => {
          if (!ids) return true;
          const a = anaOfIds(hand, ids);
          if (a.type === C.TYPE.BOMB || a.type === C.TYPE.ROCKET) {
            const rest = hand.filter(c => !ids.includes(c.id));
            return rest.length <= 2 || bestDecomp(rest).count <= 1;
          }
          return true;
        });
        if (kept.length !== cands.length) {
          cands = kept.some(ids => ids !== null) ? kept : [null];
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
  /* 三轮筛选 (候选 ≤ 24, 单步预算 ≤300ms):
     1) 全部候选共用一批 24 个世界粗筛;
     2) 前 8 名再共用 64 个世界;
     3) 前 4 名再共用 192 个世界, 并按各轮成绩加权定胜负.
     CRN 保证同一批世界里候选胜负直接可比, 比独立抽样稳定.
     实测决策均值 ~10ms / p99 ~60ms, 远低于 300ms 预算 (人类玩家
     感知不到延迟, 而样本量是选牌精度的主要来源). */
  const deadline = Date.now() + 300;
  let ranked = cands.map(ids => ({ ids, w: 0 }));
  const stages = [[24, 8], [64, 4], [192, 1]];
  for (const [K, keep] of stages) {
    const worlds = genWorlds(room, seat, K);
    const scored = [];
    for (const item of ranked) {
      if (Date.now() > deadline) break;
      const w = mcEvalWorlds(room, seat, item.ids, leading ? null : lp.ana, worlds);
      scored.push({ ids: item.ids, w, prev: item.w });
    }
    if (!scored.length) break;
    for (const it of scored) it.w = it.prev ? (it.prev + it.w) / 2 : it.w;
    scored.sort((a, b) => b.w - a.w);
    ranked = scored.slice(0, keep);
    if (ranked.length === 1) break;
  }
  const bestIds = ranked.length ? ranked[0].ids : cands[0];
  if (bestIds === null) return { type: 'pass' };
  return { type: 'play', ids: bestIds };
}

/* ---------- 总决策 ---------- */
function decide(room, seat) {
  const hand = room.players[seat].hand;
  if (room.phase === 'bidding') {
    return { type: 'bid', bid: chooseBid(hand, room.currentBid) };
  }
  return legendDecide(room, seat);
}

module.exports = { decide, handScore, bestDecomp };
