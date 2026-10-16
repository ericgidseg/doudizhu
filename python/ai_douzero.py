"""DouZero AI 接口 —— 对齐 ai.js 的 decide(room, seat) 接口.

内部把 {id,rank,suit} 牌转成 DouZero 点数, 构造 InfoSet,
ONNX 推理出最优动作, 再转回 {id} 列表.
"""
import os
import sys
import time
from collections import Counter
import numpy as np
import onnxruntime as ort
from cards import hand_to_dz, dz_to_cards, analyze, can_beat, RANK_TO_DZ, DZ_TO_RANK
from env.env import get_obs
from env.game import InfoSet
from env.move_generator import MovesGener
from env.move_detector import get_move_type
import env.move_detector as md
import env.move_selector as ms

_BASE_DIR = getattr(sys, '_MEIPASS', os.path.dirname(__file__))
MODEL_DIR = os.path.join(_BASE_DIR, 'models')          # ADP
SL_MODEL_DIR = os.path.join(_BASE_DIR, 'models_sl')    # SL 监督学习模型
WP_MODEL_DIR = os.path.join(_BASE_DIR, 'models_wp_backup')  # WP 留作投票

# AI_STYLE 控制出牌策略:
#   argmax  - 永远选 Q 值最大动作 (最强, 但套路可被摸清)
#   sample  - 按 softmax(Q/T) 采样, 有变化, 难被读牌
#   mix     - 每步以 MIX_SL_PROB 概率改用 SL 模型, 打法更像人
#   vote    - ADP+WP+SL 三模型 Q 值投票, 最稳 (推荐硬实力)
AI_STYLE = os.environ.get('AI_STYLE', 'vote')
SAMPLE_TEMP = float(os.environ.get('AI_TEMP', '0.5'))
MIX_SL_PROB = float(os.environ.get('AI_MIX_SL', '0.25'))
# 是否启用 ISMCTS rollout 增强 (vote 模式下生效). 设 1 可打开.
#
# 默认关闭: 修完所有已知 bug、权重调到最优之后, 它依然显著更差.
# 配对 A/B (同一副牌, 地主用 decide(), 200 局):
#   修复前 (blend 尺度不匹配 + 权重 0.7):  关 0.450 / 开 0.300
#   修复后 (尺度映射 + 权重 0.3):          关 0.405 / 开 0.350  (z=-2.22)
# 修复确实让危害变小 (-0.15 → -0.055), 说明诊断是对的, 但没能转正.
# 瓶颈不是采样数量 (提速 21 倍后差距没缩小), 而是 rollout 这个信号本身
# 携带的增量信息太少 —— blend 尺度不匹配只是把它放大成了灾难.
# 详见 MCTS_增强方案.md Bug 3 / 0.7 节.
USE_MCTS = os.environ.get('AI_USE_MCTS', '0') != '0'
# MCTS 每步时间预算上限 (秒). 服务器每步给 30s, 用满会让玩家等太久.
# 启发式 rollout 约 0.45ms/次, 2.5s 能给每个候选动作采样 700+ 次.
MCTS_BUDGET = float(os.environ.get('AI_MCTS_BUDGET', '2.5'))
# rollout 里的队友感知: 农民不压队友的牌 (除非能一把走完).
# 原实现里三个位置各自独立决策, 完全没有配合意识.
TEAMMATE_AWARE = os.environ.get('AI_TEAMMATE_AWARE', '1') != '0'
# MCTS 结果与 Q 值的混合权重 (0=纯Q值, 1=纯rollout).
# 扫描结果 (100 局配对, 基线 MCTS 关 = 0.390): 0.7→-0.100, 0.5→-0.075,
# 0.3→+0.010, 0.2→-0.040. 趋势是「权重越低越好」.
# 但 200 局复测把 0.3 打回原形: 关 0.4050 / 开 0.3500, 差 -0.055 (z=-2.22).
# 即 0.3 的 +0.010 是噪声 —— 权重调优只能减轻危害, 不能让 MCTS 转正.
# 保留 0.3 是因为它是扫描里最好的, 不是因为它是好的.
MCTS_BLEND_W = float(os.environ.get('AI_MCTS_BLEND', '0.3'))

_sessions = {}
_sl_sessions = {}
_wp_sessions = {}

# onnxruntime 线程数. 0 = 用 ort 默认 (吃满核心, 单步延迟最低, 适合实际对局).
# 跑 benchmark / 批量评估时设 ORT_NUM_THREADS=2, 否则会占满 CPU.
ORT_NUM_THREADS = int(os.environ.get('ORT_NUM_THREADS', '0'))


def _new_session(path):
    if ORT_NUM_THREADS <= 0:
        return ort.InferenceSession(path)
    so = ort.SessionOptions()
    so.intra_op_num_threads = ORT_NUM_THREADS
    so.inter_op_num_threads = 1
    return ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])


def _sess(position):
    if position not in _sessions:
        _sessions[position] = _new_session(
            os.path.join(MODEL_DIR, f'{position}.onnx'))
    return _sessions[position]


def _sess_sl(position):
    if position not in _sl_sessions:
        _sl_sessions[position] = _new_session(
            os.path.join(SL_MODEL_DIR, f'{position}.onnx'))
    return _sl_sessions[position]


def _sess_wp(position):
    if position not in _wp_sessions:
        _wp_sessions[position] = _new_session(
            os.path.join(WP_MODEL_DIR, f'{position}.onnx'))
    return _wp_sessions[position]


def _pick_idx(values):
    """根据 AI_STYLE 从 Q 值数组中选动作下标."""
    if AI_STYLE in ('argmax', 'vote') or len(values) <= 1:
        return int(np.argmax(values))
    # softmax with temperature
    logits = values / max(SAMPLE_TEMP, 1e-6)
    logits = logits - logits.max()
    p = np.exp(logits)
    p = p / p.sum()
    return int(np.random.choice(len(values), p=p))


def _vote_values(obs, pos):
    """三模型 Q 值投票: 各自归一化到 [0,1] 再取均值.

    不同模型的 Q 值量纲不同, 必须 min-max 归一化再平均,
    否则方差大的模型主导投票.
    """
    def _norm(v):
        rng = v.max() - v.min()
        return (v - v.min()) / max(rng, 1e-9)

    inp = {'z': obs['z_batch'], 'x': obs['x_batch']}
    v_adp = _sess(pos).run(['values'], inp)[0].flatten()
    v_wp  = _sess_wp(pos).run(['values'], inp)[0].flatten()
    v_sl  = _sess_sl(pos).run(['values'], inp)[0].flatten()
    return (_norm(v_adp) + _norm(v_wp) + _norm(v_sl)) / 3.0


def _build_legal_moves(hand_dz, last_move_dz):
    """用 DouZero move_generator 枚举合法动作."""
    mg = MovesGener(hand_dz)
    if not last_move_dz:
        return mg.gen_moves()
    rival = get_move_type(last_move_dz)
    t, ln = rival['type'], rival.get('len', 1)
    if t == md.TYPE_0_PASS:
        return mg.gen_moves()
    if t == md.TYPE_5_KING_BOMB:
        return [[]]
    # 同型压
    gen_map = {
        md.TYPE_1_SINGLE: mg.gen_type_1_single,
        md.TYPE_2_PAIR: mg.gen_type_2_pair,
        md.TYPE_3_TRIPLE: mg.gen_type_3_triple,
        md.TYPE_6_3_1: mg.gen_type_6_3_1,
        md.TYPE_7_3_2: mg.gen_type_7_3_2,
        md.TYPE_8_SERIAL_SINGLE: lambda: mg.gen_type_8_serial_single(repeat_num=ln),
        md.TYPE_9_SERIAL_PAIR: lambda: mg.gen_type_9_serial_pair(repeat_num=ln),
        md.TYPE_10_SERIAL_TRIPLE: lambda: mg.gen_type_10_serial_triple(repeat_num=ln),
        md.TYPE_11_SERIAL_3_1: lambda: mg.gen_type_11_serial_3_1(repeat_num=ln),
        md.TYPE_12_SERIAL_3_2: lambda: mg.gen_type_12_serial_3_2(repeat_num=ln),
        md.TYPE_13_4_2: mg.gen_type_13_4_2,
        md.TYPE_14_4_22: mg.gen_type_14_4_22,
    }
    filt_map = {
        md.TYPE_1_SINGLE: ms.filter_type_1_single,
        md.TYPE_2_PAIR: ms.filter_type_2_pair,
        md.TYPE_3_TRIPLE: ms.filter_type_3_triple,
        md.TYPE_6_3_1: ms.filter_type_6_3_1,
        md.TYPE_7_3_2: ms.filter_type_7_3_2,
        md.TYPE_8_SERIAL_SINGLE: ms.filter_type_8_serial_single,
        md.TYPE_9_SERIAL_PAIR: ms.filter_type_9_serial_pair,
        md.TYPE_10_SERIAL_TRIPLE: ms.filter_type_10_serial_triple,
        md.TYPE_11_SERIAL_3_1: ms.filter_type_11_serial_3_1,
        md.TYPE_12_SERIAL_3_2: ms.filter_type_12_serial_3_2,
        md.TYPE_13_4_2: ms.filter_type_13_4_2,
        md.TYPE_14_4_22: ms.filter_type_14_4_22,
    }
    if t == md.TYPE_4_BOMB:
        # 对方出炸弹: 只能炸压炸 (更大炸或王炸)
        moves = ms.filter_type_4_bomb(
            mg.gen_type_4_bomb() + mg.gen_type_5_king_bomb(), last_move_dz)
    else:
        # 同型压
        moves = filt_map[t](gen_map[t](), last_move_dz)
        # 非炸牌型都能被炸压
        moves += mg.gen_type_4_bomb() + mg.gen_type_5_king_bomb()
    moves += [[]]  # pass
    for m in moves:
        m.sort()
    return moves


def decide(room, seat):
    """
    room: 服务器房间对象 (见 server.py)
    seat: 当前座位号 0/1/2
    返回 {'type': 'play', 'ids': [card_id,...]} 或 {'type': 'pass'}
    """
    players = room['players']
    landlord_seat = room['landlordSeat']

    # ---- 叫分阶段: DouZero 没有叫分模型, 用简单启发式 ----
    if room['phase'] == 'bidding':
        score = _hand_score(hand_to_dz(players[seat]['hand']))
        # 分高叫 3, 中等跟注, 低不叫
        if score > 12:
            bid = 3
        elif score > 8:
            bid = 2
        elif score > 5:
            bid = 1
        else:
            bid = 0
        return {'type': 'bid', 'bid': bid}

    # ---- 出牌阶段 ----
    my_hand = players[seat]['hand']
    my_dz = hand_to_dz(my_hand)

    # 位置映射: seat -> DouZero position
    seat2pos = {landlord_seat: 'landlord',
                (landlord_seat + 1) % 3: 'landlord_down',
                (landlord_seat + 2) % 3: 'landlord_up'}
    pos = seat2pos[seat]
    pos2seat = {v: k for k, v in seat2pos.items()}

    # 已出牌序列 (DouZero 格式)
    action_seq = room.get('action_seq_dz', [])
    played_cards_dz = {s: hand_to_dz(played) for s, played in
                       room.get('played_by', {0: [], 1: [], 2: []}).items()}
    played_by_pos = {seat2pos[s]: played_cards_dz.get(s, []) for s in range(3)}

    # 每家剩余张数
    num_left = {seat2pos[s]: len(players[s]['hand']) for s in range(3)}

    # 上一手牌 (DouZero 格式)
    last_dz = room.get('last_move_dz', [])
    last_move_dict_dz = {}
    for s in range(3):
        lm = room.get('last_move_dict_dz', {}).get(s, [])
        last_move_dict_dz[seat2pos[s]] = lm

    # 其他玩家手牌: DouZero 的定义是「另外两家的当前手牌」(见 env/game.py 的
    # other_hand_cards = 另两家 player_hand_cards 之和), 也就是
    #   全副牌 - 我的手牌 - 所有人已出的牌
    # 注意必须减掉「三家」已出的牌. 原实现只减了自己的, 导致对手打过的牌
    # 仍被算作还在他们手里 —— 每次决策的 x 特征都是错的.
    all_cards = [3,4,5,6,7,8,9,10,11,12,13,14]*4 + [17]*4 + [20, 30]
    all_played = [c for lst in played_by_pos.values() for c in lst]
    for c in my_dz + all_played:
        if c in all_cards:
            all_cards.remove(c)
    other_hand = all_cards

    legal = _build_legal_moves(my_dz, last_dz if room.get('passCount', 0) < 2 else [])

    infoset = InfoSet(pos)
    infoset.player_hand_cards = sorted(my_dz)
    infoset.other_hand_cards = sorted(other_hand)
    infoset.legal_actions = legal
    infoset.last_move = last_dz
    infoset.last_move_dict = last_move_dict_dz
    infoset.num_cards_left_dict = num_left
    infoset.played_cards = played_by_pos
    infoset.card_play_action_seq = action_seq
    infoset.bomb_num = room.get('bomb_num', 0)
    infoset.three_landlord_cards = hand_to_dz(room.get('bottom', []))

    obs = get_obs(infoset)
    deadline = room.get('turnDeadline')
    time_left = max(0.0, deadline - time.time()) if deadline else 0.0

    if AI_STYLE == 'vote':
        values = _vote_values(obs, pos)
        # 决策难度判断: 只在"碾压级 Q 差距"或"终局"时跳过 MCTS
        n_legal = len(obs['legal_actions'])
        n_hand = len(my_dz)
        # 注意: _vote_values 输出已 min-max 归一化到 [0,1], 绝对跨度恒 ~1,
        # 不能用绝对值判断"差距大". 改用「最优与次优的差距」占整体跨度的比例:
        # 比例接近 1 说明最优动作碾压次优 -> 容易决策.
        if n_legal > 1:
            sv = np.sort(values)
            margin = float(sv[-1] - sv[-2]) / max(float(sv[-1] - sv[0]), 1e-9)
        else:
            margin = 1.0
        # 严苛条件: 单选项 / 手牌极少 / 最优动作碾压次优
        is_easy = (n_legal <= 1) or (n_hand <= 3) or (margin > 0.5)
        # 其余情况都跑 ISMCTS rollout, 用满时间预算找最优
        if USE_MCTS and not is_easy and time_left > 1.5 and n_legal > 1:
            values = _mcts_pick(infoset, values, pos, played_by_pos,
                                num_left, time_left, min_rollouts=6,
                                room=room, seat=seat, seat2pos=seat2pos)
        best_idx = _pick_idx(values)
    else:
        # mix 模式下这一步用 SL 模型 (更像人类打法)
        use_sl = (AI_STYLE == 'mix' and np.random.random() < MIX_SL_PROB)
        sess = _sess_sl(pos) if use_sl else _sess(pos)
        values = sess.run(['values'], {
            'z': obs['z_batch'], 'x': obs['x_batch']})[0].flatten()
        best_idx = _pick_idx(values)
    best_dz = obs['legal_actions'][best_idx]

    if not best_dz:
        return {'type': 'pass'}
    cards = dz_to_cards(best_dz, my_hand)
    return {'type': 'play', 'ids': [c['id'] for c in cards]}


def _hand_score(hand_dz):
    """简易手牌评分 (叫分用): 炸+王+2+A 加分."""
    score = 0
    cnt = Counter(hand_dz)
    for v, n in cnt.items():
        if v == 30: score += 4       # 大王
        elif v == 20: score += 3     # 小王
        elif v == 17: score += 2 * n # 2
        elif v == 14: score += n     # A
        if n == 4: score += 6        # 炸弹
    if 20 in cnt and 30 in cnt: score += 4  # 王炸
    return score


# ---- 快速 rollout 策略 ----
# rollout 里用神经网络太慢: 每步一次 ONNX 推理, 实测占 rollout 总耗时 65%,
# 导致每个候选动作只能采样 2~3 次 —— 胜率估计标准误 ±0.29,
# 而动作间的真实胜率差只有 0.05~0.15, 噪声淹没信号, MCTS 反而帮倒忙.
# 换成启发式后 rollout 快一个数量级, 同样预算能多采样 10 倍以上.
USE_FAST_ROLLOUT = os.environ.get('AI_FAST_ROLLOUT', '1') != '0'

# 牌面代价: 越大越贵, 鼓励先出小牌
_CARD_COST = {3: 1, 4: 1, 5: 1, 6: 1, 7: 2, 8: 2, 9: 2, 10: 3,
              11: 3, 12: 3, 13: 4, 14: 5, 17: 8, 20: 14, 30: 16}
# 炸弹/王炸的额外代价: 留着炸关键手, 不轻易拆
_BOMB_PENALTY = 60.0


def _is_bomb_move(move):
    if len(move) == 4 and len(set(move)) == 1:
        return True
    return sorted(move) == [20, 30]


def _move_cost(move, hand_cnt):
    """动作代价: 牌越大越贵; 拆对子/三张再加权; 炸弹重罚."""
    cost = sum(_CARD_COST.get(c, 5) for c in move)
    if _is_bomb_move(move):
        cost += _BOMB_PENALTY
    elif len(move) == 1:
        if hand_cnt.get(move[0], 0) >= 2:      # 拆对子出单张
            cost += 2.0
    elif len(move) == 2 and move[0] == move[1]:
        if hand_cnt.get(move[0], 0) >= 3:      # 拆三张出对子
            cost += 2.0
    return cost


def _heuristic_pick(hand, legal, cur_last_pos, turn_pos, hands):
    """不用神经网络的快速 rollout 决策.

    原则 (按优先级):
      1. 能一把走完就走完
      2. 农民队友领先时让路, 不抢队友的出牌权
      3. 否则出代价最小的合法动作 (跟牌即"最小能压")
      4. 只有炸弹能压时, 对手不急着走完就先留着
    """
    # 1. 能直接赢
    for act in legal:
        if act and len(act) == len(hand):
            return list(act)

    i_am_farmer = (turn_pos != 'landlord')
    teammate_leads = (cur_last_pos is not None
                      and cur_last_pos != turn_pos
                      and cur_last_pos != 'landlord')

    # 2. 队友领先 -> 让路 (对手快走完时除外)
    if i_am_farmer and teammate_leads and [] in legal:
        if len(hands.get('landlord', [])) > 2:
            return []

    cnt = Counter(hand)
    best, best_cost = None, None
    for act in legal:
        if not act:
            continue
        c = _move_cost(act, cnt)
        if best_cost is None or c < best_cost:
            best, best_cost = list(act), c

    if best is None:
        return []

    # 3. 只有炸弹能压 -> 对手不急着走完就先留着
    if _is_bomb_move(best) and [] in legal:
        if i_am_farmer:
            if len(hands.get('landlord', [])) > 2:
                return []
        else:
            opp_min = min(len(hands.get(p, []))
                          for p in ('landlord_up', 'landlord_down'))
            if opp_min > 2:
                return []
    return best


def _rollout(hands, turn_pos, last_move, last_mover, action_seq,
             bomb_num, landlord_cards, played, max_steps=60):
    """三家都用 ADP 打完一局, 返回胜方位置字符串.

    hands: {pos: [dz,...]} 三家手牌 (假设分配, 不完全信息)
    turn_pos: 当前该谁出
    last_move: 当前要压的牌 ([]=领出)
    last_mover: 上一手出牌者位置 (用于判断 pass 逻辑)
    action_seq: 历史出牌
    played: {pos: [dz,...]} 已出牌
    """
    hands = {p: sorted(list(h)) for p, h in hands.items()}
    played = {p: list(v) for p, v in played.items()}
    seq = list(action_seq)
    cur_last = sorted(last_move)
    cur_last_pos = last_mover
    consecutive_pass = 0
    order = ['landlord', 'landlord_down', 'landlord_up']
    bombs = bomb_num

    for _ in range(max_steps):
        hand = hands[turn_pos]
        if not hand:
            return turn_pos

        # 有效 last_move: 如果上家就是自己 (刚 pass 完轮到我领出) 就传 []
        effective_last = [] if cur_last_pos == turn_pos else cur_last
        legal = _build_legal_moves(hand, effective_last)
        if not legal:
            legal = [[]]

        if USE_FAST_ROLLOUT:
            # 启发式策略: 省掉每步一次 ONNX 推理 (原来占 rollout 耗时 65%)
            act = _heuristic_pick(hand, legal, cur_last_pos, turn_pos, hands)
        else:
            # 原路径: 用神经网络决策 (慢, 保留用于对比测试)
            other = []
            for p, h in hands.items():
                if p != turn_pos:
                    other += h
            infoset = InfoSet(turn_pos)
            infoset.player_hand_cards = sorted(hand)
            infoset.other_hand_cards = sorted(other)
            infoset.legal_actions = legal
            infoset.last_move = list(effective_last)
            infoset.last_move_dict = {
                p: (list(cur_last) if p == cur_last_pos else []) for p in order}
            infoset.num_cards_left_dict = {p: len(h) for p, h in hands.items()}
            infoset.played_cards = played
            infoset.card_play_action_seq = seq
            infoset.bomb_num = bombs
            infoset.three_landlord_cards = list(landlord_cards)

            try:
                obs = get_obs(infoset)
                v = _sess(turn_pos).run(['values'], {
                    'z': obs['z_batch'], 'x': obs['x_batch']})[0].flatten()
                act = list(legal[int(np.argmax(v))])

                # 队友感知: 农民通常不该压自己队友的牌
                if TEAMMATE_AWARE and act:
                    i_am_farmer = (turn_pos != 'landlord')
                    last_is_teammate = (cur_last_pos != 'landlord'
                                        and cur_last_pos != turn_pos)
                    if i_am_farmer and last_is_teammate:
                        would_empty_hand = (sorted(act) == sorted(hand))
                        if not would_empty_hand and [] in legal:
                            act = []
            except Exception:
                act = []

        # 应用
        if act:
            for c in act:
                if c in hands[turn_pos]:
                    hands[turn_pos].remove(c)
            seq.append(sorted(act))
            played.setdefault(turn_pos, []).extend(act)
            cur_last = sorted(act)
            cur_last_pos = turn_pos
            consecutive_pass = 0
            if len(act) == 4 and len(set(act)) == 1:
                bombs += 1
            elif sorted(act) == [20, 30]:
                bombs += 1
            if not hands[turn_pos]:
                return turn_pos
        else:
            seq.append([])
            consecutive_pass += 1
            if consecutive_pass >= 2:
                # 两家都 pass -> 由「最后出牌的人」领出 (不是最后 pass 的人).
                # cur_last_pos 此时仍指向最后出牌者, 不要覆盖它.
                cur_last = []
                consecutive_pass = 0
                turn_pos = cur_last_pos
                continue

        # 轮下家
        idx = order.index(turn_pos)
        turn_pos = order[(idx + 1) % 3]

    return cur_last_pos


def _build_belief(room, my_seat, seat2pos, played_by_pos, my_dz):
    """根据历史 action_seq 推断对手手牌概率分布.

    返回: {pos: {dz_card: weight}}  weight 越大越可能在 pos 手里.
    基础 weight=1.0, 每个信号乘系数 (<1 表示不太可能在, >1 表示更可能在).
    """
    order = ['landlord', 'landlord_down', 'landlord_up']
    opp_pos = [p for p in order if p != seat2pos[my_seat]]

    # 未知牌池 = 全副 - 我手 - 已出
    # 底牌已归地主: 出了的在 played_by 里, 没出的在地主手牌里(即未知)
    # 所以底牌不额外减
    full = [3,4,5,6,7,8,9,10,11,12,13,14]*4 + [17]*4 + [20, 30]
    pool = list(full)
    for c in my_dz:
        if c in pool: pool.remove(c)
    for lst in played_by_pos.values():
        for c in lst:
            if c in pool: pool.remove(c)

    # 每张未知牌对两家对手的初始 weight
    beliefs = {p: {c: 1.0 for c in pool} for p in opp_pos}

    # ---- 从历史 action_seq 推断 ----
    # room['action_seq_dz'] 是出牌序列, 但不知道是谁出的
    # 我们用 room['played_by'] 已经知道每家出了什么
    # 但 pass 信息在 replay 里没单独存, 从 action_seq_dz 里 [] 推断

    # 先重建每家的 pass 历史:
    # action_seq_dz 是按出牌顺序的所有动作 (含 pass=[]), 顺序跟 turn 对齐
    action_seq = room.get('action_seq_dz', [])
    landlord_seat = room['landlordSeat']
    # 重新推演 turn 序列, 把每个 action 归属到正确玩家
    # 假设地主先出, turn 按 landlord -> landlord_down -> landlord_up -> landlord
    turn_order = [landlord_seat, (landlord_seat+1)%3, (landlord_seat+2)%3]
    cur_turn_idx = 0
    last_play_seat = -1
    last_play_dz = []

    pass_history = {seat2pos[s]: [] for s in range(3)}  # pos -> [被 pass 的牌]

    for act in action_seq:
        seat = turn_order[cur_turn_idx % 3]
        pos = seat2pos[seat]
        if act:  # 出牌
            last_play_dz = sorted(act)
            last_play_seat = seat
        else:    # pass
            if last_play_dz:
                pass_history[pos].append(list(last_play_dz))
        cur_turn_idx += 1

    # ---- 规则: 根据 pass 历史调整 belief ----
    # 规则1: pass 过单 X -> 该玩家大概率没 X+1 .. BJ 的单 (他不想/不能压)
    # 注意这是"软"信号, 只是降权不为 0
    for pos in opp_pos:
        for passed_move in pass_history[pos]:
            t = get_move_type(passed_move)
            if t['type'] == md.TYPE_1_SINGLE:
                thresh = passed_move[0]
                # 对大于 thresh 的单张降权 (他没出更大的单)
                for c in beliefs[pos]:
                    if c > thresh and c >= 17:  # 只对大牌降权
                        beliefs[pos][c] *= 0.6
                    elif c > thresh:
                        beliefs[pos][c] *= 0.85
            elif t['type'] == md.TYPE_2_PAIR:
                thresh = passed_move[0]
                # 他有 pair 能压就该出, pass 说明没 pair >= thresh
                for c in beliefs[pos]:
                    if c > thresh:
                        beliefs[pos][c] *= 0.8
            elif t['type'] == md.TYPE_4_BOMB:
                # pass 炸弹说明他没炸/王炸 -> 大幅降低他有4张/王炸的概率
                # 这个不好直接对单张降权, 改用"4张同点牌"的集体降权
                # 简化: 对每张 c, 如果它在 pool 里有 >=3 张同点就降权
                pool_cnt = Counter(pool)
                for c in beliefs[pos]:
                    if pool_cnt.get(c, 0) >= 3:
                        beliefs[pos][c] *= 0.4
            elif t['type'] == md.TYPE_3_TRIPLE:
                # pass 三张说明他没大 triple / 没炸
                thresh = passed_move[2] if len(passed_move) >= 3 else passed_move[0]
                for c in beliefs[pos]:
                    if c > thresh:
                        beliefs[pos][c] *= 0.8

    # 规则2: 没出过炸 -> 炸弹概率下降 (但不为 0, 可能留着)
    # 统计各家出了多少炸
    bombs_played = {p: 0 for p in order}
    for pos, cards in played_by_pos.items():
        cnt = Counter(cards)
        for v, n in cnt.items():
            if n == 4:
                bombs_played[pos] += 1
        # 王炸
        if 20 in cnt and 30 in cnt:
            bombs_played[pos] += 1
    pool_cnt = Counter(pool)
    for pos in opp_pos:
        # 对每张可能成为炸的牌降权
        for c in beliefs[pos]:
            if pool_cnt.get(c, 0) >= 3:
                beliefs[pos][c] *= 0.85
        # 王炸概率: 20 和 30 都在 pool 时
        if 20 in beliefs[pos] and 30 in beliefs[pos]:
            beliefs[pos][20] *= 0.9
            beliefs[pos][30] *= 0.9

    # 规则3: 地主叫了高分 -> 地主手牌偏好高 (底牌质量好)
    if room.get('currentBid', 0) >= 2:
        ll_pos = 'landlord'
        if ll_pos in beliefs:
            for c in beliefs[ll_pos]:
                if c >= 14:  # A 以上
                    beliefs[ll_pos][c] *= 1.15
                elif c >= 11:
                    beliefs[ll_pos][c] *= 1.05

    # 规则4: 手牌数越少 -> 剩余牌分布越集中 (该家更可能"凑成"现有组合)
    # 跳过: 已由 num_left 控制

    return beliefs, pool, pass_history


def _sample_hands(pool, beliefs, opp_pos, opp_sizes, my_pos, my_hand_after_act):
    """按 belief weight 采样对手手牌分配 (Gumbel top-k).

    约束: opp_pos[i] 恰好 opp_sizes[i] 张牌.

    做法: 对每张牌算 log-odds = log(w0) - log(w1), 加上 Gumbel 噪声后排序,
    取前 n0 张给 opp0, 其余给 opp1. 这样「每张牌归谁」的随机性和「恰好 n0 张」
    的硬约束是解耦的 —— 排序天然保证张数精确.

    旧实现是「先 shuffle, 再逐张按 w0/(w0+w1) 抽」, 并把剩余张数
    max(remaining, 0.1) 乘进权重里. 实测这条曲线被压平:
    某张牌几乎确定在某家时 (比值 100:1), 旧实现只给 0.93 概率, 应为 1.00;
    比值 1:20 时给 0.105, 应为 0.036. 即 belief 信号最强的地方最不听话.
    """
    p0, p1 = opp_pos[0], opp_pos[1]
    n0, n1 = opp_sizes[0], opp_sizes[1]
    pool = list(pool)
    if n0 <= 0:
        return {p0: [], p1: sorted(pool[:max(n1, 0)])}
    if n1 <= 0:
        return {p0: sorted(pool[:n0]), p1: []}

    b0 = np.array([max(beliefs[p0].get(c, 1.0), 1e-12) for c in pool])
    b1 = np.array([max(beliefs[p1].get(c, 1.0), 1e-12) for c in pool])
    key = np.log(b0) - np.log(b1) + np.random.gumbel(size=len(pool))
    order = np.argsort(key)[::-1]

    hands = {p0: [], p1: []}
    for i in range(min(n0, len(order))):
        hands[p0].append(pool[order[i]])
    for i in range(n0, min(n0 + n1, len(order))):
        hands[p1].append(pool[order[i]])
    for p in opp_pos:
        hands[p] = sorted(hands[p])
    return hands


def _sample_hands_old(pool, beliefs, opp_pos, opp_sizes, my_pos, my_hand_after_act):
    """旧实现, 保留用于 A/B 对比 (AI_OLD_SAMPLER=1 时启用).

    缺陷: 把剩余张数 max(remaining, 0.1) 乘进权重, 与 belief 权重互相干扰,
    导致 belief 信号被稀释 (见 _sample_hands 的说明).
    """
    hands = {p: [] for p in opp_pos}
    pool_shuffled = list(pool)
    np.random.shuffle(pool_shuffled)
    remaining = {opp_pos[0]: opp_sizes[0], opp_pos[1]: opp_sizes[1]}

    for c in pool_shuffled:
        if remaining[opp_pos[0]] <= 0 and remaining[opp_pos[1]] <= 0:
            break
        w0 = beliefs[opp_pos[0]].get(c, 1.0) * max(remaining[opp_pos[0]], 0.1)
        w1 = beliefs[opp_pos[1]].get(c, 1.0) * max(remaining[opp_pos[1]], 0.1)
        total = w0 + w1
        p0 = w0 / total if total > 0 else 0.5
        if remaining[opp_pos[0]] <= 0:
            p0 = 0.0
        elif remaining[opp_pos[1]] <= 0:
            p0 = 1.0
        if np.random.random() < p0:
            hands[opp_pos[0]].append(c)
            remaining[opp_pos[0]] -= 1
        else:
            hands[opp_pos[1]].append(c)
            remaining[opp_pos[1]] -= 1

    # 若分配不均 (剩余 >0), 从对方手里按 weight 反向移
    for p in opp_pos:
        while remaining[p] > 0:
            other = opp_pos[1] if p == opp_pos[0] else opp_pos[0]
            if not hands[other]:
                break
            # 从 other 手里挑"最不可能在 other 那"的牌移过来
            move = min(hands[other],
                       key=lambda c: beliefs[other].get(c, 1.0) /
                                     max(beliefs[p].get(c, 1.0), 0.01))
            hands[other].remove(move)
            hands[p].append(move)
            remaining[p] -= 1

    for p in opp_pos:
        hands[p] = sorted(hands[p])
    return hands


# belief 采样器: 默认用 Gumbel top-k (见 _sample_hands 说明).
# 设 AI_OLD_SAMPLER=1 可切回旧实现做 A/B 对比.
USE_OLD_SAMPLER = os.environ.get('AI_OLD_SAMPLER', '0') != '0'


def _sample(pool, beliefs, opp_pos, opp_sizes, my_pos, my_hand_after_act):
    """按开关选择采样器."""
    if USE_OLD_SAMPLER:
        return _sample_hands_old(pool, beliefs, opp_pos, opp_sizes,
                                 my_pos, my_hand_after_act)
    return _sample_hands(pool, beliefs, opp_pos, opp_sizes,
                         my_pos, my_hand_after_act)


def _mcts_pick(infoset, base_values, my_pos, played_by_pos, num_left,
               time_left, min_rollouts=6, room=None, seat=None, seat2pos=None):
    """ISMCTS: belief-state 加权 determinized rollout."""
    n = len(base_values)
    if n <= 1:
        return base_values

    TOP_K = 6
    top_idx = list(np.argsort(base_values)[-TOP_K:][::-1])

    order = ['landlord', 'landlord_down', 'landlord_up']
    opp_pos = [p for p in order if p != my_pos]
    opp_sizes = [num_left[p] for p in opp_pos]

    # ---- belief state ----
    if room is not None and seat is not None and seat2pos is not None:
        beliefs, pool, _ = _build_belief(
            room, seat, seat2pos, played_by_pos, infoset.player_hand_cards)
    else:
        # fallback: 均匀采样
        full = [3,4,5,6,7,8,9,10,11,12,13,14]*4 + [17]*4 + [20, 30]
        pool = list(full)
        for c in list(infoset.player_hand_cards):
            if c in pool: pool.remove(c)
        for lst in played_by_pos.values():
            for c in lst:
                if c in pool: pool.remove(c)
        beliefs = {p: {c: 1.0 for c in pool} for p in opp_pos}

    win_count = {i: 0 for i in top_idx}
    tries = {i: 0 for i in top_idx}

    start = time.time()
    budget = min(MCTS_BUDGET, max(0.3, time_left - 0.5))
    roll = 0
    while True:
        elapsed = time.time() - start
        if elapsed > budget:
            break
        if roll >= min_rollouts and elapsed > budget * 0.7:
            break
        roll += 1

        for i in top_idx:
            if time.time() - start > budget:
                break
            act = infoset.legal_actions[i]

            # 加权采样对手手牌
            my_hand = sorted(c for c in infoset.player_hand_cards if c not in act)
            hands = _sample(pool, beliefs, opp_pos, opp_sizes,
                            my_pos, my_hand)
            hands[my_pos] = my_hand

            # 我出 act (或 pass) 后轮到下家
            next_idx = (order.index(my_pos) + 1) % 3
            next_pos = order[next_idx]
            new_seq = list(infoset.card_play_action_seq)
            new_played = {p: list(v) for p, v in played_by_pos.items()}
            if act:
                new_seq.append(sorted(act))
                new_played[my_pos] = new_played.get(my_pos, []) + list(act)
                new_last = sorted(act)
                new_last_pos = my_pos
            else:
                new_seq.append([])
                new_last = list(infoset.last_move)
                new_last_pos = None
                for p, lm in infoset.last_move_dict.items():
                    if lm:
                        new_last_pos = p
                if new_last_pos is None:
                    new_last_pos = my_pos

            winner = _rollout(hands, next_pos, new_last, new_last_pos,
                              new_seq, infoset.bomb_num,
                              infoset.three_landlord_cards, new_played)

            if my_pos == 'landlord':
                won = (winner == 'landlord')
            else:
                won = (winner != 'landlord')
            if won:
                win_count[i] += 1
            tries[i] += 1

    # blend: 把 rollout 胜率映射到「与 Q 值可比的尺度」后再混合.
    #
    # 关键: rollout 胜率的「绝对值」没有意义 —— 斗地主地主整体胜率就只有
    # 30~40%, 所以 wr 集中在 0.2~0.6. 而 base_values (vote 归一化后) 集中在
    # 0.85~1.0. 直接做 0.3*Q + 0.7*wr, 平均只有 0.3*0.9+0.7*0.35 = 0.52,
    # 反而低于「未参与评估、保持原值」的普通动作 (0.84) —— 于是 top-K 里
    # 大部分动作被自己压低, 输给压根没评估过的动作. 这就是 MCTS 帮倒忙的原因.
    #
    # 有意义的不是 wr 的绝对值, 而是它相对其他候选动作的「排序」.
    # 所以把 top-K 的 wr 线性映射到「这些动作 Q 值的 [min,max] 区间」,
    # 保证 rollout 只在 Q 值给出的范围内做调整, 不会把动作踢出竞争.
    vals = [win_count[i] / tries[i] for i in top_idx if tries[i] > 0]
    cand_q = [base_values[i] for i in top_idx]
    q_lo, q_hi = min(cand_q), max(cand_q)
    wr_lo, wr_hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
    wr_span = wr_hi - wr_lo

    refined = base_values.copy()
    for i in top_idx:
        if tries[i] > 0:
            wr = win_count[i] / tries[i]
            if wr_span > 1e-9:
                # 映射到 top-K 的 Q 值区间, 保持排序信息
                wr_scaled = q_lo + (wr - wr_lo) / wr_span * (q_hi - q_lo)
            else:
                wr_scaled = (q_lo + q_hi) / 2.0   # 所有动作胜率相同 -> 无信息
            w = MCTS_BLEND_W
            refined[i] = (1.0 - w) * base_values[i] + w * wr_scaled
        else:
            refined[i] = base_values[i] * 0.5
    # pass 惩罚: 残局挣扎
    for i, act in enumerate(infoset.legal_actions):
        if not act:
            refined[i] -= 0.15
    return refined
