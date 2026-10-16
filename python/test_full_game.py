"""模拟完整一局: DouZero 三家互打, 验证管线端到端."""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env.game import InfoSet
from env.move_generator import MovesGener
from env.move_detector import get_move_type
import env.move_selector as ms
import env.move_detector as md
from encoding import get_obs
from ai import decide

R2N = {'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,
       'J':11,'Q':12,'K':13,'A':14,'2':17,'SJ':20,'BJ':30}
N2R = {v:k for k,v in R2N.items()}

def deal():
    deck = [3,4,5,6,7,8,9,10,11,12,13,14]*4 + [17]*4 + [20,30]
    random.shuffle(deck)
    hands = [sorted(deck[0:17]), sorted(deck[17:34]), sorted(deck[34:51])]
    bottom = sorted(deck[51:54])
    return hands, bottom

def fmt(nums): return ' '.join(N2R.get(n,str(n)) for n in sorted(nums))

def play_game():
    hands, bottom = deal()
    # Simple bidding: first player bids 3
    landlord = random.randint(0,2)
    hands[landlord] = sorted(hands[landlord] + bottom)
    seat2pos = {landlord:'landlord', (landlord+1)%3:'landlord_down', (landlord+2)%3:'landlord_up'}
    pos2seat = {v:k for k,v in seat2pos.items()}

    played_by = {0:[],1:[],2:[]}
    action_seq = []
    last_move, last_seat = [], -1
    pass_count = 0
    bomb_num = 0
    step = 0

    cur = landlord
    while True:
        step += 1
        if step > 200: return None
        my_hand = hands[cur]
        all_other = [c for s in range(3) if s != cur for c in hands[s]]
        num_left = {seat2pos[s]: len(hands[s]) for s in range(3)}
        last_move_dict = {}
        played_dict = {}
        for s in range(3):
            played_dict[seat2pos[s]] = played_by[s]
            # find last move per seat
            lm = []
            for a in reversed(action_seq):
                pass  # simplification
            last_move_dict[seat2pos[s]] = lm

        # legal moves — 复用 ai_douzero 的构造逻辑, 保证与线上口径一致
        if not last_move or last_seat == cur or pass_count >= 2:
            legal = MovesGener(my_hand).gen_moves()
            lm_for_obs = []
        else:
            mt = get_move_type(last_move)
            t, ln = mt['type'], mt.get('len', 1)
            mg = MovesGener(my_hand)
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
                legal = ms.filter_type_4_bomb(
                    mg.gen_type_4_bomb() + mg.gen_type_5_king_bomb(), last_move)
            elif t == md.TYPE_5_KING_BOMB:
                legal = []
            else:
                legal = filt_map[t](gen_map[t](), last_move)
                legal += mg.gen_type_4_bomb() + mg.gen_type_5_king_bomb()
            legal += [[]]
            for m in legal:
                m.sort()
            lm_for_obs = last_move

        infoset = InfoSet(seat2pos[cur])
        infoset.player_hand_cards = my_hand
        infoset.other_hand_cards = sorted(all_other)
        infoset.legal_actions = legal
        infoset.last_move = lm_for_obs
        infoset.last_move_dict = last_move_dict
        infoset.num_cards_left_dict = num_left
        infoset.played_cards = played_dict
        infoset.card_play_action_seq = action_seq
        infoset.bomb_num = bomb_num
        infoset.three_landlord_cards = bottom
        act = decide(infoset)
        if not act:
            pass_count += 1
            action_seq.append([])
            print(f'{step:2}. seat{cur}({seat2pos[cur]}) pass')
        else:
            t = get_move_type(act)['type']
            tn = {md.TYPE_4_BOMB:'炸弹', md.TYPE_5_KING_BOMB:'王炸',
                  md.TYPE_1_SINGLE:'单张', md.TYPE_2_PAIR:'对子',
                  md.TYPE_3_TRIPLE:'三张', md.TYPE_6_3_1:'三带一',
                  md.TYPE_7_3_2:'三带二', md.TYPE_8_SERIAL_SINGLE:'顺子',
                  md.TYPE_9_SERIAL_PAIR:'连对', md.TYPE_10_SERIAL_TRIPLE:'飞机',
                  md.TYPE_11_SERIAL_3_1:'飞机带单', md.TYPE_12_SERIAL_3_2:'飞机带对',
                  md.TYPE_13_4_2:'四带二', md.TYPE_14_4_22:'四带二对'}.get(t, str(t))
            if t in (md.TYPE_4_BOMB, md.TYPE_5_KING_BOMB):
                bomb_num += 1
            for c in act:
                hands[cur].remove(c)
                played_by[cur].append(c)
            action_seq.append(act)
            last_move, last_seat = act, cur
            pass_count = 0
            print(f'{step:2}. seat{cur}({seat2pos[cur]}) {tn}: {fmt(act)} (剩{len(hands[cur])})')
            if not hands[cur]:
                return 'landlord' if cur==landlord else 'farmer'
        cur = (cur+1) % 3

if __name__ == '__main__':
    random.seed(42)
    print('=== DouZero 三家自打 ===')
    w = play_game()
    print('胜方:', w)
