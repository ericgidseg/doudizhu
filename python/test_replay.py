"""用真实 replay 验证 DouZero ONNX 模型的决策是否合理."""
import json
import sys
sys.path.insert(0, '/Users/yahaha/Desktop/Game/doudizhu-py')
from encoding import InfoSet, get_obs, gen_legal_moves, gen_beat_moves, analyze, cards2array
from ai import decide

# DouZero 牌编码: 3-14, 17=2, 20=小王, 30=大王
# 你的 replay 是 {id, rank, suit} — 需要转换
R2N = {'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,
       'J':11,'Q':12,'K':13,'A':14,'2':17,'SJ':20,'BJ':30}

def hand2nums(hand):
    return sorted(R2N[c['rank']] for c in hand)

def build_infoset(r, upto_step, seat):
    """重建 upto_step 之前的状态, 构造 seat 的 InfoSet."""
    played = []           # 已出的牌 (所有家)
    played_by = {0: [], 1: [], 2: []}  # 每家各自出过什么
    action_seq = []       # 出牌序列 (用于 LSTM)
    last_move = []
    last_move_dict = {0: [], 1: [], 2: []}
    last_seat = -1
    bomb_num = 0
    pass_count = 0

    for i, a in enumerate(r['actions'][:upto_step]):
        if a['type'] == 'bid':
            continue
        if a['type'] == 'pass':
            action_seq.append([])
            pass_count += 1
            continue
        cards = [R2N[c['rank']] for c in a['cards']]
        played.extend(cards)
        played_by[a['seat']].extend(cards)
        action_seq.append(cards)
        last_move = cards
        last_move_dict[a['seat']] = cards
        last_seat = a['seat']
        pass_count = 0
        if a['typeName'] in ('炸弹', '王炸'):
            bomb_num += 1

    # 当前 seat 的手牌 (初始手牌 - 已出的)
    my_hand_all = r['players'][seat]['hand'][:]
    if seat == r['landlordSeat']:
        my_hand_all += r['bottom']
    my_hand = [R2N[c['rank']] for c in my_hand_all]
    for c in played_by[seat]:
        my_hand.remove(c)

    # other_hand = 所有不在我手里/未打出的牌
    all_cards = [3,4,5,6,7,8,9,10,11,12,13,14]*4 + [17]*4 + [20,30]
    for c in my_hand + played:
        all_cards.remove(c)

    # 位置映射
    landlord_seat = r['landlordSeat']
    if seat == landlord_seat:
        pos = 'landlord'
    elif seat == (landlord_seat + 1) % 3:
        pos = 'landlord_down'   # 地主下家
    else:
        pos = 'landlord_up'     # 地主上家

    seat2pos = {landlord_seat: 'landlord',
                (landlord_seat+1)%3: 'landlord_down',
                (landlord_seat+2)%3: 'landlord_up'}
    pos2seat = {v:k for k,v in seat2pos.items()}

    num_left = {seat2pos[s]: len(r['players'][s]['hand']) - len(played_by[s]) +
                (3 if s==landlord_seat else 0)
                for s in range(3)}

    # legal actions
    if not last_move or last_seat == seat or pass_count >= 2:
        legal = gen_legal_moves(my_hand)
    else:
        lt, lr, ll = analyze(last_move)
        legal = gen_beat_moves(my_hand, lt, lr, ll)

    infoset = InfoSet(
        player_position=pos,
        player_hand_cards=sorted(my_hand),
        other_hand_cards=sorted(all_cards),
        legal_actions=legal,
        last_move=last_move if pass_count < 2 else [],
        last_move_dict={seat2pos[s]: last_move_dict[s] for s in range(3)},
        num_cards_left_dict=num_left,
        played_cards={seat2pos[s]: played_by[s] for s in range(3)},
        card_play_action_seq=action_seq,
        bomb_num=bomb_num,
        three_landlord_cards=[R2N[c['rank']] for c in r['bottom']],
    )
    return infoset, legal


def nums2cards(nums, hand_map):
    """DouZero 点数 -> 你的牌面对象."""
    N2R = {v:k for k,v in R2N.items()}
    return [N2R.get(n, str(n)) for n in nums]


if __name__ == '__main__':
    # Test on r5947-1 step 21: landlord (seat1) facing single 4
    r = json.load(open('/Users/yahaha/Desktop/Game/doudizhu/replays/r5947-1.json'))
    infoset, legal = build_infoset(r, 21, 1)
    print('seat1 (landlord) hand:', nums2cards(infoset.player_hand_cards, None))
    print('legal actions:', len(legal))
    print('DouZero decides...')
    act = decide(infoset)
    print('  ->', nums2cards(act, None) if act else 'pass')
