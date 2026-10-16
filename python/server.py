"""斗地主服务端 —— aiohttp 单端口, 对齐 server.js 协议.

用法: python3 server.py [port]     默认端口 8000
  HTTP :8000  静态资源 + /api/replays + /api/replay
  WS   :8000/ws  游戏消息
前端 public/ 无需改动, 仅 client.js 的 ws URL 加 '/ws' 后缀.
"""
import asyncio
import json
import os
import random
import re
import secrets
import socket
import sys
from datetime import datetime, timezone

from aiohttp import web, WSMsgType

import cards as C
import ai_douzero as AI

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
# PyInstaller 打包后 __file__ 在临时解压目录; sys._MEIPASS 是 bundle 资源根
HERE = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
PUBLIC = os.path.join(HERE, 'public')
MIME = {
    '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8', '.png': 'image/png', '.jpg': 'image/jpeg',
    '.svg': 'image/svg+xml', '.ico': 'image/x-icon', '.json': 'application/json',
}
REPLAY_DIR = os.environ.get('DDZ_REPLAY_DIR') or os.path.join(
    os.path.expanduser('~'), 'Library', 'Application Support', 'doudizhu', 'replays')
REPLAY_KEEP = 100

rooms = {}              # id -> room
ws_room = {}            # id(ws) -> room


def sanitize(s):
    return re.sub(r'[<>&"\']', '', str(s or ''))[:12] or '玩家'


def new_room():
    while True:
        rid = str(1000 + random.randint(0, 8999))
        if rid not in rooms:
            break
    return {'id': rid, 'players': [], 'hostSeat': -1,
            'phase': 'lobby', 'dealerIdx': 0,
            'bottom': [], 'currentBid': 0, 'currentBidder': -1,
            'bids': [], 'turn': -1, 'passCount': 0,
            'lastPlay': None, 'multiplier': 1, 'landlordSeat': -1,
            'played': [], 'action_seq_dz': [],
            'played_by': {0: [], 1: [], 2: []},
            'last_move_dz': [], 'last_move_dict_dz': {0: [], 1: [], 2: []},
            'bomb_num': 0, 'replay': None, '_seq': 0,
            '_bot_task': None, '_cleanup_task': None}


# ---------------- 录制 ----------------
def record_init(room):
    room['_seq'] += 1
    room['replay'] = {
        'roomId': room['id'], 'seq': room['_seq'], 'difficulty': 'douzero',
        'startedAt': datetime.now(timezone.utc).isoformat(),
        'players': [{'seat': p['seat'], 'name': p['name'], 'bot': bool(p.get('bot')),
                     'hand': [{'id': c['id'], 'rank': c['rank'], 'suit': c['suit']}
                              for c in p['hand']]} for p in room['players']],
        'actions': [], 'bottom': [], 'landlordSeat': -1,
        'winnerSeat': -1, 'landlordWin': False, 'finalMultiplier': 0,
        'finishedAt': None}


def save_replay(room):
    rep = room.get('replay')
    if not rep:
        return
    try:
        os.makedirs(REPLAY_DIR, exist_ok=True)
        snap = dict(rep)
        snap['currentHands'] = [[{'id': c['id'], 'rank': c['rank'], 'suit': c['suit']}
                                 for c in p['hand']] for p in room['players']]
        with open(os.path.join(REPLAY_DIR, f"r{room['id']}-{room['_seq']}.json"), 'w') as f:
            json.dump(snap, f, ensure_ascii=False, indent=1)
        files = sorted(f for f in os.listdir(REPLAY_DIR) if f.endswith('.json'))
        while len(files) > REPLAY_KEEP:
            os.unlink(os.path.join(REPLAY_DIR, files.pop(0)))
    except Exception as e:
        print('save replay failed', e)


def seat_of(room, ws):
    for p in room['players']:
        if p.get('ws') is ws:
            return p['seat']
    return -1


def player_view(p):
    return {'seat': p['seat'], 'name': p['name'], 'cardCount': len(p['hand']),
            'online': bool(p.get('ws')) or bool(p.get('bot')),
            'wins': p.get('wins', 0), 'bot': bool(p.get('bot')),
            'score': p.get('score', 0)}


def send_to(ws, msg):
    if ws.closed:
        return
    asyncio.ensure_future(ws.send_str(json.dumps(msg, ensure_ascii=False)))


def broadcast(room, msg, **data):
    payload = json.dumps({'type': msg, **data}, ensure_ascii=False)
    for p in room['players']:
        ws = p.get('ws')
        if ws and not ws.closed:
            asyncio.ensure_future(ws.send_str(payload))


TURN_SECONDS = 30  # 每步思考时间, 超时自动 pass


def _arm_turn_timer(room):
    """每次轮到新玩家时重置计时. 到点未动作则:
       - 叫分阶段: 自动叫 0 分 (不叫)
       - 出牌阶段: 自动 pass; 若必须领出则强制出最小单张"""
    deadline = datetime.now(timezone.utc).timestamp() + TURN_SECONDS
    room['turnDeadline'] = deadline
    if room.get('_turn_timer'):
        room['_turn_timer'].cancel()
    async def _timeout():
        await asyncio.sleep(TURN_SECONDS + 0.1)
        if room['id'] not in rooms:
            return
        # 期间 turn 又变了(意味着 deadline 也被重置), 旧的 timer 就该失效
        if room.get('turnDeadline') != deadline:
            return
        seat = room['turn']
        if seat < 0:
            return
        name = room['players'][seat]['name']
        if room['phase'] == 'bidding':
            broadcast(room, 'event', event=f"{name} 超时未叫分, 视为不叫")
            apply_bid(room, seat, 0)
            return
        if room['phase'] != 'playing':
            return
        broadcast(room, 'event', event=f"{name} 超时未出牌, 自动过")
        # 领出阶段不能 pass, 强制出最小单张
        if not room['lastPlay'] or room['lastPlay']['seat'] == seat:
            hand = room['players'][seat]['hand']
            if hand:
                smallest = min(hand, key=lambda c: C.RANK_VALUE.get(c['rank'], 99))
                apply_play(room, seat, [smallest['id']])
        else:
            apply_pass(room, seat)
    room['_turn_timer'] = asyncio.ensure_future(_timeout())


def send_state(room):
    base = {'id': room['id'], 'phase': room['phase'], 'hostSeat': room['hostSeat'],
            'players': [player_view(p) for p in room['players']],
            'turn': room['turn'], 'currentBid': room['currentBid'],
            'currentBidder': room['currentBidder'], 'bids': room['bids'],
            'multiplier': room['multiplier'], 'difficulty': 'douzero',
            'landlordSeat': room['landlordSeat'], 'bottomCount': len(room['bottom']),
            'turnDeadline': room.get('turnDeadline'),
            'turnSeconds': TURN_SECONDS}
    lp = room.get('lastPlay')
    last_play = None
    if lp:
        last_play = {'seat': lp['seat'], 'ids': list(lp['ids']),
                     'cards': [{'id': c['id'], 'rank': c['rank'], 'suit': c['suit']}
                               for c in lp['cards']],
                     'typeName': lp['ana']['typeName']}
    show_bottom = room['phase'] in ('playing', 'done')
    for p in room['players']:
        ws = p.get('ws')
        if not ws or ws.closed:
            continue
        send_to(ws, {'type': 'state', **base, 'mySeat': p['seat'],
                     'hand': C.sort_cards(p['hand']),
                     'bottom': C.sort_cards(room['bottom']) if show_bottom else [],
                     'lastPlay': last_play,
                     'played': [{'id': c['id'], 'rank': c['rank']} for c in room['played']]})


def cleanup_room(room):
    if not all(p.get('bot') or not p.get('ws') for p in room['players']):
        return
    async def _gc():
        await asyncio.sleep(60)
        if all(p.get('bot') or not p.get('ws') for p in room['players']):
            rooms.pop(room['id'], None)
    if room.get('_cleanup_task'):
        room['_cleanup_task'].cancel()
    room['_cleanup_task'] = asyncio.ensure_future(_gc())


# ---------------- AI ----------------
def schedule_bot(room):
    if room['phase'] not in ('bidding', 'playing'):
        return
    p = room['players'][room['turn']] if 0 <= room['turn'] < 3 else None
    if not p:
        return
    disconnected = not p.get('bot') and not p.get('ws')
    if not p.get('bot') and not disconnected:
        return
    if room.get('_bot_task'):
        room['_bot_task'].cancel()

    async def _run():
        await asyncio.sleep(0.5 + random.random() * 0.6)
        if room['id'] not in rooms or room['phase'] not in ('bidding', 'playing'):
            return
        cur = room['players'][room['turn']] if 0 <= room['turn'] < 3 else None
        if not cur or (not cur.get('bot') and cur.get('ws')):
            return
        try:
            # ONNX 推理 ~10ms, 放 executor 避免阻塞 WS 收发
            act = await asyncio.get_event_loop().run_in_executor(
                None, AI.decide, room, room['turn'])
            if act['type'] == 'bid':
                apply_bid(room, room['turn'], act['bid'])
            elif act['type'] == 'play':
                apply_play(room, room['turn'], act['ids'])
            elif act['type'] == 'pass':
                apply_pass(room, room['turn'])
        except Exception:
            import traceback; traceback.print_exc()
    room['_bot_task'] = asyncio.ensure_future(_run())


def add_bot(room):
    if len(room['players']) >= 3:
        return None
    seat = len(room['players'])
    p = {'name': 'AI·' + '甲乙丙'[seat], 'ws': None, 'seat': seat,
         'hand': [], 'wins': 0, 'score': 0, 'bot': True, 'token': None}
    room['players'].append(p)
    return p


# ---------------- 流程 ----------------
def start_game(room):
    deck = C.shuffle(C.create_deck())
    for p in room['players']:
        p['hand'] = deck[:17]; deck = deck[17:]
    room['bottom'] = deck
    room['phase'] = 'bidding'
    room['dealerIdx'] = random.randint(0, 2)
    room['turn'] = room['dealerIdx']
    room['currentBid'] = 0; room['currentBidder'] = -1
    room['bids'] = [-1] * 3
    room['multiplier'] = 1; room['landlordSeat'] = -1
    room['passCount'] = 0; room['lastPlay'] = None
    room['played'] = []; room['action_seq_dz'] = []
    room['played_by'] = {0: [], 1: [], 2: []}
    room['last_move_dz'] = []; room['last_move_dict_dz'] = {0: [], 1: [], 2: []}
    room['bomb_num'] = 0
    record_init(room)
    broadcast(room, 'event',
              event=f"第 {room['id']} 房间开局, {room['players'][room['dealerIdx']]['name']} 先叫分")
    _arm_turn_timer(room)
    send_state(room)
    schedule_bot(room)


def apply_bid(room, seat, bid):
    room['bids'][seat] = bid
    if room['replay']:
        room['replay']['actions'].append({'type': 'bid', 'seat': seat, 'bid': bid})
    if bid > room['currentBid']:
        room['currentBid'] = bid; room['currentBidder'] = seat
    name = room['players'][seat]['name']
    broadcast(room, 'event', event=f"{name} {'不叫' if bid == 0 else '叫 ' + str(bid) + ' 分'}")
    if bid == 3 or all(b != -1 for b in room['bids']):
        if room['currentBidder'] == -1:
            broadcast(room, 'event', event='无人叫地主, 重新发牌')
            async def _again():
                await asyncio.sleep(1.5)
                if room['id'] in rooms:
                    start_game(room)
            asyncio.ensure_future(_again())
            return
        room['landlordSeat'] = room['currentBidder']
        room['multiplier'] = room['currentBid']
        room['players'][room['landlordSeat']]['hand'].extend(room['bottom'])
        if room['replay']:
            room['replay']['landlordSeat'] = room['landlordSeat']
            room['replay']['bottom'] = [{'id': c['id'], 'rank': c['rank'], 'suit': c['suit']}
                                        for c in room['bottom']]
        room['phase'] = 'playing'
        room['turn'] = room['landlordSeat']
        _arm_turn_timer(room)
        bs = ' '.join((c['suit'] or '') + c['rank'] for c in C.sort_cards(room['bottom']))
        broadcast(room, 'event',
                  event=f"{room['players'][room['landlordSeat']]['name']} 成为地主 (底分 {room['currentBid']}), 底牌: {bs}")
        send_state(room); schedule_bot(room); return
    room['turn'] = (seat + 1) % 3
    _arm_turn_timer(room)
    send_state(room); schedule_bot(room)


def apply_play(room, seat, ids):
    p = room['players'][seat]
    idset = set(ids)
    cards_ = [c for c in p['hand'] if c['id'] in idset]
    ana = C.analyze(cards_)
    if ana['type'] == C.TYPE.INVALID:
        # 兜底: 人类路径已拦, AI/超时强制路径可能漏进来. 降级出最小单.
        broadcast(room, 'event',
                  event=f"{p['name']} 出的牌型无效, 自动出最小单张")
        smallest = min(p['hand'], key=lambda c: C.RANK_VALUE.get(c['rank'], 99))
        return apply_play(room, seat, [smallest['id']])
    p['hand'] = [c for c in p['hand'] if c['id'] not in idset]
    room['lastPlay'] = {'ids': list(ids), 'cards': cards_, 'ana': ana, 'seat': seat}
    room['played'].extend(cards_)
    dz = C.hand_to_dz(cards_)
    room['action_seq_dz'].append(dz)
    room['played_by'][seat].extend(cards_)
    room['last_move_dz'] = dz
    room['last_move_dict_dz'][seat] = dz
    if ana['type'] in (C.TYPE.BOMB, C.TYPE.ROCKET):
        room['bomb_num'] += 1
    if room['replay']:
        room['replay']['actions'].append({
            'type': 'play', 'seat': seat, 'ids': list(ids),
            'cards': [{'id': c['id'], 'rank': c['rank'], 'suit': c['suit']} for c in cards_],
            'typeName': ana['typeName'],
            'ana': {'type': ana['type'],
                    'main': 'Infinity' if ana['main'] == float('inf') else ana['main'],
                    'len': ana['len']},
            'left': len(p['hand'])})
    room['passCount'] = 0
    cs = ' '.join((c['suit'] or '') + c['rank'] for c in C.sort_cards(cards_))
    broadcast(room, 'event', event=f"{p['name']} 出 {ana['typeName']}: {cs}")
    if ana['type'] in (C.TYPE.BOMB, C.TYPE.ROCKET):
        room['multiplier'] *= 2
        broadcast(room, 'event', event=f"💥 {ana['typeName']}! 倍数 ×2 → {room['multiplier']}")
    if not p['hand']:
        end_game(room, seat); return
    room['turn'] = (seat + 1) % 3
    _arm_turn_timer(room)
    send_state(room); schedule_bot(room)


def apply_pass(room, seat):
    p = room['players'][seat]
    if room['replay']:
        room['replay']['actions'].append({'type': 'pass', 'seat': seat})
    room['action_seq_dz'].append([])
    room['passCount'] += 1
    broadcast(room, 'event', event=f"{p['name']} 不出")
    room['turn'] = (seat + 1) % 3
    if room['passCount'] >= 2:
        room['turn'] = room['lastPlay']['seat']
        room['lastPlay'] = None; room['last_move_dz'] = []; room['passCount'] = 0
        broadcast(room, 'event', event=f"{room['players'][room['turn']]['name']} 领出")
    _arm_turn_timer(room)
    send_state(room); schedule_bot(room)


def end_game(room, winner_seat):
    room['phase'] = 'done'
    if room.get('_turn_timer'):
        room['_turn_timer'].cancel()
        room['_turn_timer'] = None
    room.pop('turnDeadline', None)
    landlord_win = winner_seat == room['landlordSeat']
    if room['replay']:
        room['replay'].update(winnerSeat=winner_seat, landlordWin=landlord_win,
                              finalMultiplier=room['multiplier'],
                              finishedAt=datetime.now(timezone.utc).isoformat())
        save_replay(room)
    fw = room['landlordSeat'] if landlord_win else winner_seat
    room['players'][fw]['wins'] = room['players'][fw].get('wins', 0) + 1
    base = room['currentBid'] or 1
    stake = base * room['multiplier']
    scores = []
    for s, p in enumerate(room['players']):
        lord = s == room['landlordSeat']
        delta = (2 * stake if lord else -stake) if landlord_win else (-2 * stake if lord else stake)
        p['score'] = p.get('score', 0) + delta
        scores.append({'seat': s, 'name': p['name'], 'delta': delta,
                       'total': p['score'], 'isLord': lord})
    hands = [{'seat': s, 'name': p['name'], 'isLord': s == room['landlordSeat'],
              'cards': [{'id': c['id'], 'rank': c['rank'], 'suit': c['suit']}
                        for c in C.sort_cards(p['hand'])]}
             for s, p in enumerate(room['players'])]
    broadcast(room, 'gameOver', winnerSeat=fw,
              winnerName=room['players'][fw]['name'],
              landlordSeat=room['landlordSeat'], landlordWin=landlord_win,
              multiplier=room['multiplier'], baseBid=base,
              scores=scores, hands=hands,
              event=f"{room['players'][fw]['name']} ({'地主' if landlord_win else '农民'}) 获胜! 倍数 {room['multiplier']}")
    room['turn'] = -1
    send_state(room)


# ---------------- 消息 ----------------
def err(ws, msg):
    send_to(ws, {'type': 'error', 'msg': msg})


def room_of(ws):
    return ws_room.get(id(ws))


def on_create(ws, d):
    room = new_room()
    p = {'name': sanitize(d.get('name')), 'ws': ws, 'seat': 0, 'hand': [],
         'wins': 0, 'score': 0, 'bot': False, 'token': secrets.token_hex(16)}
    room['players'].append(p); room['hostSeat'] = 0
    rooms[room['id']] = room
    ws_room[id(ws)] = room
    send_to(ws, {'type': 'joined', 'roomId': room['id'], 'seat': 0, 'token': p['token']})
    send_state(room)


def on_join(ws, d):
    room = rooms.get(str(d.get('roomId') or '').strip())
    if not room:
        return err(ws, '房间不存在')
    token = str(d.get('token') or '')
    if token:
        p = next((x for x in room['players'] if x.get('token') == token), None)
        if not p:
            return err(ws, '重连失败')
        if room.get('_cleanup_task'):
            room['_cleanup_task'].cancel()
        old = p.get('ws')
        p['ws'] = ws; p['bot'] = False
        ws_room[id(ws)] = room
        send_to(ws, {'type': 'joined', 'roomId': room['id'], 'seat': p['seat'], 'token': p['token']})
        if old and old is not ws and not old.closed:
            asyncio.ensure_future(old.close())
        broadcast(room, 'event', event=f"{p['name']} 重新连接")
        send_state(room); return
    if len(room['players']) >= 3:
        return err(ws, '房间已满')
    seat = len(room['players'])
    p = {'name': sanitize(d.get('name')), 'ws': ws, 'seat': seat, 'hand': [],
         'wins': 0, 'score': 0, 'bot': False, 'token': secrets.token_hex(16)}
    room['players'].append(p)
    ws_room[id(ws)] = room
    send_to(ws, {'type': 'joined', 'roomId': room['id'], 'seat': seat, 'token': p['token']})
    broadcast(room, 'event', event=f"{p['name']} 加入了房间")
    send_state(room)


def on_start(ws, d):
    room = room_of(ws)
    if not room:
        return
    seat = seat_of(room, ws)
    if seat < 0 or room['phase'] in ('playing', 'bidding'):
        return
    if room['phase'] != 'done' and seat != room['hostSeat']:
        return err(ws, '只有房主可以开始')
    if len(room['players']) != 3:
        return err(ws, '需要 3 名玩家')
    start_game(room)


def on_bid(ws, d):
    room = room_of(ws)
    if not room or room['phase'] != 'bidding':
        return
    seat = seat_of(room, ws)
    if seat != room['turn']:
        return
    try:
        bid = int(d.get('bid'))
    except (TypeError, ValueError):
        return err(ws, '无效叫分')
    if not (bid == 0 or bid > room['currentBid']) or bid > 3:
        return err(ws, '无效叫分')
    apply_bid(room, seat, bid)


def on_play(ws, d):
    room = room_of(ws)
    if not room or room['phase'] != 'playing':
        return
    seat = seat_of(room, ws)
    if seat != room['turn']:
        return err(ws, '还没轮到你')
    p = room['players'][seat]
    ids = d.get('ids') if isinstance(d.get('ids'), list) else []
    ids = ids[:54]
    hand_ids = {c['id'] for c in p['hand']}
    if not ids or any(i not in hand_ids for i in ids):
        return err(ws, '你没有这些牌')
    cards_ = [c for c in p['hand'] if c['id'] in set(ids)]
    ana = C.analyze(cards_)
    if ana['type'] == C.TYPE.INVALID:
        return err(ws, '无效牌型: ' + ana['typeName'])
    base = room['lastPlay']['ana'] if room['lastPlay'] else None
    if base and not C.can_beat(base, ana):
        return err(ws, '压不过上家的' + base['typeName'])
    apply_play(room, seat, ids)


def on_pass(ws, d):
    room = room_of(ws)
    if not room or room['phase'] != 'playing':
        return
    seat = seat_of(room, ws)
    if seat != room['turn']:
        return
    if not room['lastPlay'] or room['lastPlay']['seat'] == seat:
        return err(ws, '你必须出牌')
    apply_pass(room, seat)


def on_hint(ws, d):
    """提示: 用 DouZero AI 给玩家建议 (而非贪心最小压).
    ONNX 推理 ~10ms, 放到 executor 避免阻塞 event loop."""
    room = room_of(ws)
    if not room or room['phase'] != 'playing':
        return
    seat = seat_of(room, ws)
    if seat != room['turn']:
        return

    async def _run():
        try:
            act = await asyncio.get_event_loop().run_in_executor(
                None, AI.decide, room, seat)
            ids = act['ids'] if act['type'] == 'play' else []
        except Exception:
            ids = []
        if not ws.closed:
            send_to(ws, {'type': 'hint', 'ids': ids})
    asyncio.ensure_future(_run())


def on_chat(ws, d):
    room = room_of(ws)
    if not room:
        return
    seat = seat_of(room, ws)
    if seat >= 0:
        broadcast(room, 'chat', seat=seat, name=room['players'][seat]['name'],
                  text=str(d.get('text') or '')[:60])


def on_addbot(ws, d):
    room = room_of(ws)
    if not room:
        return
    if seat_of(room, ws) != room['hostSeat']:
        return err(ws, '只有房主可以添加 AI')
    if room['phase'] != 'lobby':
        return err(ws, '对局已开始')
    p = add_bot(room)
    if not p:
        return err(ws, '房间已满')
    broadcast(room, 'event', event=f"{p['name']} 加入了房间")
    send_state(room)


def on_solo(ws, d):
    room = new_room()
    p = {'name': sanitize(d.get('name')), 'ws': ws, 'seat': 0, 'hand': [],
         'wins': 0, 'score': 0, 'bot': False, 'token': None}
    room['players'].append(p); room['hostSeat'] = 0
    add_bot(room); add_bot(room)
    rooms[room['id']] = room
    ws_room[id(ws)] = room
    send_to(ws, {'type': 'joined', 'roomId': room['id'], 'seat': 0})
    broadcast(room, 'event', event='单人模式: 你 + 2 个 AI')
    send_state(room)
    async def _go():
        await asyncio.sleep(0.8)
        if room['id'] in rooms:
            start_game(room)
    asyncio.ensure_future(_go())


def on_rooms(ws, d):
    lst = [{'id': r['id'], 'count': len(r['players']),
            'names': [p['name'] for p in r['players']]}
           for r in rooms.values() if r['phase'] == 'lobby' and len(r['players']) < 3]
    send_to(ws, {'type': 'rooms', 'rooms': lst})


def on_setdiff(ws, d):
    room = room_of(ws)
    if room and seat_of(room, ws) == room['hostSeat']:
        broadcast(room, 'event', event='AI 难度: DouZero 传说')
        send_state(room)


HANDLERS = {'create': on_create, 'join': on_join, 'start': on_start, 'bid': on_bid,
            'play': on_play, 'pass': on_pass, 'hint': on_hint, 'chat': on_chat,
            'rooms': on_rooms, 'addbot': on_addbot, 'solo': on_solo, 'setdiff': on_setdiff}


# ---------------- HTTP + WS ----------------
async def handle_ws(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    async for msg in ws:
        if msg.type != WSMsgType.TEXT:
            continue
        try:
            d = json.loads(msg.data)
        except Exception:
            continue
        h = HANDLERS.get(d.get('type'))
        if h:
            try:
                h(ws, d)
            except Exception:
                import traceback; traceback.print_exc()
    room = ws_room.pop(id(ws), None)
    if room:
        seat = seat_of(room, ws)
        if seat >= 0:
            room['players'][seat]['ws'] = None
            broadcast(room, 'event', event=f"{room['players'][seat]['name']} 断线")
            send_state(room)
            schedule_bot(room)
        cleanup_room(room)
    return ws


async def handle_http(request):
    path = request.path
    if path == '/api/replays':
        out = []
        try:
            for f in os.listdir(REPLAY_DIR):
                if not f.endswith('.json'):
                    continue
                try:
                    with open(os.path.join(REPLAY_DIR, f)) as fp:
                        r = json.load(fp)
                    out.append({'f': f, 'roomId': r.get('roomId'), 'seq': r.get('seq'),
                                'difficulty': r.get('difficulty'),
                                'startedAt': r.get('startedAt'), 'finishedAt': r.get('finishedAt'),
                                'landlordSeat': r.get('landlordSeat'),
                                'winnerSeat': r.get('winnerSeat'), 'landlordWin': r.get('landlordWin'),
                                'players': [{'seat': p['seat'], 'name': p['name'], 'bot': p['bot']}
                                            for p in r.get('players', [])],
                                'actions': len(r.get('actions', []))})
                except Exception:
                    pass
            out.sort(key=lambda x: x.get('startedAt') or '', reverse=True)
        except Exception:
            pass
        return web.json_response(out)
    if path == '/api/replay':
        f = request.query.get('f', '')
        if not re.match(r'^[\w.-]+\.json$', f):
            return web.Response(status=400, text='bad file')
        fp = os.path.join(REPLAY_DIR, f)
        if not os.path.isfile(fp):
            return web.Response(status=404, text='not found')
        return web.FileResponse(fp, headers={'Content-Type': 'application/json; charset=utf-8'})
    rel = 'index.html' if path == '/' else path.lstrip('/')
    if '..' in rel:
        return web.Response(status=403)
    fp = os.path.join(PUBLIC, rel)
    if not os.path.isfile(fp):
        return web.Response(status=404, text='not found')
    return web.FileResponse(fp)


def lan_ips():
    ips = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            ip = info[4][0]
            if ':' not in ip and not ip.startswith('127.') and ip not in ips:
                ips.append(ip)
    except Exception:
        pass
    return ips


def main():
    app = web.Application()
    app.router.add_get('/ws', handle_ws)
    app.router.add_get('/{tail:.*}', handle_http)
    print('斗地主服务端已启动 (DouZero AI)')
    print(f'  本机访问:   http://localhost:{PORT}')
    for ip in lan_ips():
        print(f'  内网IP访问: http://{ip}:{PORT}')
    web.run_app(app, host='0.0.0.0', port=PORT, print=None)


if __name__ == '__main__':
    main()
