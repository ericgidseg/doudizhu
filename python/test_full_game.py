"""模拟完整一局: DouZero 三家互打, 验证管线端到端."""
import sys, json, random
sys.path.insert(0, '/Users/yahaha/Desktop/Game/doudizhu-py')
from encoding import InfoSet, gen_legal_moves, gen_beat_moves, analyze, cards2array
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

        # legal moves
        if not last_move or last_seat == cur or pass_count >= 2:
            legal = gen_legal_moves(my_hand)
            lm_for_obs = []
        else:
            lt, lr, ll = analyze(last_move)
            legal = gen_beat_moves(my_hand, lt, lr, ll)
            lm_for_obs = last_move

        infoset = InfoSet(
            player_position=seat2pos[cur],
            player_hand_cards=my_hand,
            other_hand_cards=sorted(all_other),
            legal_actions=legal,
            last_move=lm_for_obs,
            last_move_dict=last_move_dict,
            num_cards_left_dict=num_left,
            played_cards=played_dict,
            card_play_action_seq=action_seq,
            bomb_num=bomb_num,
            three_landlord_cards=bottom,
        )
        act = decide(infoset)
        if not act:
            pass_count += 1
            action_seq.append([])
            print(f'{step:2}. seat{cur}({seat2pos[cur]}) pass')
        else:
            t, r, l = analyze(act)
            if t in ('bomb','rocket'): bomb_num += 1
            for c in act:
                hands[cur].remove(c)
                played_by[cur].append(c)
            action_seq.append(act)
            last_move, last_seat = act, cur
            pass_count = 0
            print(f'{step:2}. seat{cur}({seat2pos[cur]}) {t}: {fmt(act)} (剩{len(hands[cur])})')
            if not hands[cur]:
                return 'landlord' if cur==landlord else 'farmer'
        cur = (cur+1) % 3

if __name__ == '__main__':
    random.seed(42)
    print('=== DouZero 三家自打 ===')
    w = play_game()
    print('胜方:', w)
