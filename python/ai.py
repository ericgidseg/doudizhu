"""DouZero ONNX 推理 —— 给 infoset 打分选最优动作."""
import os
import numpy as np
import onnxruntime as ort

from encoding import get_obs

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

_sessions = {}


def _sess(position):
    if position not in _sessions:
        path = os.path.join(MODEL_DIR, f'{position}.onnx')
        _sessions[position] = ort.InferenceSession(path)
    return _sessions[position]


def decide(infoset):
    """
    输入 InfoSet, 返回选中的动作 (牌列表, [] 表示 pass).
    对每个合法动作编码一行, ONNX 打分, argmax.
    """
    obs = get_obs(infoset)
    sess = _sess(infoset.player_position)
    out = sess.run(['values'], {
        'z': obs['z_batch'],
        'x': obs['x_batch'],
    })[0].flatten()
    best_idx = int(np.argmax(out))
    return obs['legal_actions'][best_idx]
