"""DouZero AI 接口 —— decide(room, seat), 与 server.py 的房间状态对接.

内部把 {id,rank,suit} 牌转成 DouZero 点数, 构造 InfoSet, ONNX 推理出
最优动作, 再转回 {id} 列表.

模型分配: 地主用 models_best/ (hero-vs-baseline 微调, 配对胜率 0.525),
农民用 models/ (原始). 三个模型 vote 集成是默认策略.

AI_STYLE 环境变量:
  vote    - ADP+WP+SL 三模型 Q 值 min-max 归一化后平均, 取 argmax (默认)
  human   - 同上投票, 但按 softmax(Q/HUMAN_TEMP) 采样, 打法更像人
  argmax  - 只用 ADP 模型, 永远取 Q 值最大动作
  sample  - 只用 ADP 模型, 按 softmax(Q/SAMPLE_TEMP) 采样
  mix     - 只用 ADP, 但每步以 MIX_SL_PROB 概率改用 SL 模型
"""
import os
import sys
import time
from collections import Counter
import numpy as np
import onnxruntime as ort

from cards import hand_to_dz, dz_to_cards
from encoding import get_obs
from env.game import InfoSet
from env.move_generator import MovesGener
from env.move_detector import get_type as get_move_type, TYPE_NAMES
import env.move_detector as md
import env.move_selector as ms

_BASE_DIR = getattr(sys, '_MEIPASS', os.path.dirname(__file__))
MODEL_DIR = os.path.join(_BASE_DIR, 'models')          # ADP (农民)
LL_MODEL_DIR = os.path.join(_BASE_DIR, 'models_best')  # 地主微调模型
if not os.path.isfile(os.path.join(LL_MODEL_DIR, 'landlord.onnx')):
    LL_MODEL_DIR = MODEL_DIR
SL_MODEL_DIR = os.path.join(_BASE_DIR, 'models_sl')
WP_MODEL_DIR = os.path.join(_BASE_DIR, 'models_wp_backup')

AI_STYLE = os.environ.get('AI_STYLE', 'vote')
SAMPLE_TEMP = float(os.environ.get('AI_TEMP', '0.5'))
MIX_SL_PROB = float(os.environ.get('AI_MIX_SL', '0.25'))
HUMAN_TEMP = float(os.environ.get('AI_HUMAN_TEMP', '0.3'))

# onnxruntime 线程数. 0 = ort 默认 (单步延迟最低); 批量评估时设 2 免得吃满 CPU.
ORT_NUM_THREADS = int(os.environ.get('ORT_NUM_THREADS', '0'))

_sessions = {}


def _sess(position):
    """按位置取模型: 地主 models_best, 农民 models."""
    if position not in _sessions:
        d = LL_MODEL_DIR if position == 'landlord' else MODEL_DIR
        path = os.path.join(d, f'{position}.onnx')
        if ORT_NUM_THREADS > 0:
            so = ort.SessionOptions()
            so.intra_op_num_threads = ORT_NUM_THREADS
            so.inter_op_num_threads = 1
            _sessions[position] = ort.InferenceSession(
                path, so, providers=['CPUExecutionProvider'])
        else:
            _sessions[position] = ort.InferenceSession(path)
    return _sessions[position]


def _sess_in(model_dir, position):
    key = (model_dir, position)
    if key not in _sessions:
        _sessions[key] = ort.InferenceSession(
            os.path.join(model_dir, f'{position}.onnx'))
    return _sessions[key]


def _pick_idx(values, temperature=None):
    """argmax / vote 用 argmax; sample / human 走 softmax 采样."""
    if AI_STYLE in ('argmax', 'vote') or len(values) <= 1:
        return int(np.argmax(values))
    T = temperature if temperature is not None else SAMPLE_TEMP
    logits = values / max(T, 1e-6)
    p = np.exp(logits - logits.max())
    return int(np.random.choice(len(values), p=p / p.sum()))


def _vote_values(obs, pos):
    """三模型 Q 值投票: 各自 min-max 归一化到 [0,1] 再取均值.

    不同模型 Q 值量纲不同, 不归一化的话方差大的模型会主导投票.
    """
    def _norm(v):
        return (v - v.min()) / max(v.max() - v.min(), 1e-9)

    inp = {'z': obs['z_batch'], 'x': obs['x_batch']}
    v_adp = _sess(pos).run(['values'], inp)[0].flatten()
    v_wp = _sess_in(WP_MODEL_DIR, pos).run(['values'], inp)[0].flatten()
    v_sl = _sess_in(SL_MODEL_DIR, pos).run(['values'], inp)[0].flatten()
    return (_norm(v_adp) + _norm(v_wp) + _norm(v_sl)) / 3.0


def _build_legal_moves(hand_dz, last_move_dz):
    """用 DouZero move_generator 枚举合法动作 (与 env/game.py 同口径)."""
    mg = MovesGener(hand_dz)
    if not last_move_dz:
        return mg.gen_moves()

    t_info = get_move_type(last_move_dz)
    t = t_info['type']
    if t == md.TYPE_0_PASS:
        return mg.gen_moves()
    if t == md.TYPE_5_KING_BOMB:
        return [[]]

    # 序列类牌型的 len 是「组数」(连对 3 对 = 3), 不是牌张数
    ln = t_info.get('len', 1)
    gen = {
        md.TYPE_1_SINGLE: mg.gen_type_1_single,
        md.TYPE_2_PAIR: mg.gen_type_2_pair,
        md.TYPE_3_TRIPLE: mg.gen_type_3_triple,
        md.TYPE_6_3_1: mg.gen_type_6_3_1,
        md.TYPE_7_3_2: mg.gen_type_7_3_2,
        md.TYPE_8_SERIAL_SINGLE: lambda: mg.gen_type_8_serial_single(ln),
        md.TYPE_9_SERIAL_PAIR: lambda: mg.gen_type_9_serial_pair(ln),
        md.TYPE_10_SERIAL_TRIPLE: lambda: mg.gen_type_10_serial_triple(ln),
        md.TYPE_11_SERIAL_3_1: lambda: mg.gen_type_11_serial_3_1(ln),
        md.TYPE_12_SERIAL_3_2: lambda: mg.gen_type_12_serial_3_2(ln),
        md.TYPE_13_4_2: mg.gen_type_13_4_2,
        md.TYPE_14_4_22: mg.gen_type_14_4_22,
    }
    filt = {
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
        moves = ms.filter_type_4_bomb(
            mg.gen_type_4_bomb() + mg.gen_type_5_king_bomb(), last_move_dz)
    else:
        moves = filt[t](gen[t](), last_move_dz)
        moves += mg.gen_type_4_bomb() + mg.gen_type_5_king_bomb()
    moves += [[]]
    for m in moves:
        m.sort()
    return moves


def _hand_score(hand_dz):
    """简易手牌评分 (叫分用): 王/2/A/炸弹加分."""
    score = 0
    cnt = Counter(hand_dz)
    for v, n in cnt.items():
        if v == 30:
            score += 4
        elif v == 20:
            score += 3
        elif v == 17:
            score += 2 * n
        elif v == 14:
            score += n
        if n == 4:
            score += 6
    if 20 in cnt and 30 in cnt:
        score += 4
    return score


def decide(room, seat):
    """room: server.py 的房间字典; seat: 0/1/2.
    返回 {'type': 'bid', 'bid': n} 或 {'type': 'play', 'ids': [...]} / {'type': 'pass'}."""
    players = room['players']

    # ---- 叫分阶段: DouZero 没有叫分模型, 用启发式 ----
    if room['phase'] == 'bidding':
        score = _hand_score(hand_to_dz(players[seat]['hand']))
        bid = 3 if score > 12 else 2 if score > 8 else 1 if score > 5 else 0
        return {'type': 'bid', 'bid': bid}

    # ---- 出牌阶段 ----
    landlord_seat = room['landlordSeat']
    seat2pos = {landlord_seat: 'landlord',
                (landlord_seat + 1) % 3: 'landlord_down',
                (landlord_seat + 2) % 3: 'landlord_up'}
    pos = seat2pos[seat]
    my_hand = players[seat]['hand']
    my_dz = hand_to_dz(my_hand)

    played_by_pos = {seat2pos[s]: hand_to_dz(room['played_by'][s]) for s in range(3)}

    # other_hand_cards = 另外两家的当前手牌 = 全副牌 - 我的手牌 - 三家已出的牌
    all_cards = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] * 4 + [17] * 4 + [20, 30]
    for c in my_dz + [c for lst in played_by_pos.values() for c in lst]:
        all_cards.remove(c)

    # 两家都 pass 后由最后出牌者领出, 此时 last_move 作废
    last_dz = room['last_move_dz'] if room['passCount'] < 2 else []
    legal = _build_legal_moves(my_dz, last_dz)

    infoset = InfoSet(pos)
    infoset.player_hand_cards = sorted(my_dz)
    infoset.other_hand_cards = sorted(all_cards)
    infoset.legal_actions = legal
    infoset.last_move = last_dz
    infoset.last_move_dict = {seat2pos[s]: room['last_move_dict_dz'][s] for s in range(3)}
    infoset.num_cards_left_dict = {seat2pos[s]: len(players[s]['hand']) for s in range(3)}
    infoset.played_cards = played_by_pos
    infoset.card_play_action_seq = room['action_seq_dz']
    infoset.bomb_num = room['bomb_num']
    infoset.three_landlord_cards = hand_to_dz(room['bottom'])

    obs = get_obs(infoset)

    if AI_STYLE in ('vote', 'human'):
        values = _vote_values(obs, pos)
        temp = None if AI_STYLE == 'vote' else HUMAN_TEMP
        best_idx = _pick_idx(values, temperature=temp)
    else:
        use_sl = (AI_STYLE == 'mix' and np.random.random() < MIX_SL_PROB)
        sess = _sess_in(SL_MODEL_DIR, pos) if use_sl else _sess(pos)
        values = sess.run(['values'], {
            'z': obs['z_batch'], 'x': obs['x_batch']})[0].flatten()
        best_idx = _pick_idx(values)

    best_dz = obs['legal_actions'][best_idx]
    if not best_dz:
        return {'type': 'pass'}
    return {'type': 'play', 'ids': [c['id'] for c in dz_to_cards(best_dz, my_hand)]}
