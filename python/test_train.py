"""DMC 训练管线的健全性检查.

用法: python3 test_train.py   (需要 torch, 约 1 分钟)
"""
import os
import sys
import tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from model import (DouZeroModel, DMCModel, load_onnx_weights,
                   export_to_onnx, verify_export, predict, X_DIMS,
                   MODEL_DIR, POSITIONS, SEQ_LEN, LSTM_INPUT)
from env.env import Env, get_obs
from selfplay import run_episode


def test_env_basic():
    env = Env(objective='adp')
    env.reset()
    assert env.infoset is not None
    assert env.infoset.legal_actions
    assert env._acting_player_position == 'landlord'
    print('  env_basic: PASS')


def test_obs_shape():
    env = Env(objective='adp')
    env.reset()
    obs = get_obs(env.infoset)
    assert obs['z_batch'].shape[1:] == (SEQ_LEN, LSTM_INPUT)
    pos = env.infoset.player_position
    assert obs['x_batch'].shape[1] == X_DIMS[pos]
    print(f'  obs_shape: PASS (x_dim={X_DIMS[pos]})')


def test_model_forward():
    for pos in POSITIONS:
        model = DouZeroModel(pos)
        out = model(torch.randn(4, SEQ_LEN, LSTM_INPUT),
                    torch.randn(4, X_DIMS[pos]))
        assert out.shape == (4, 1), f'{pos}: {out.shape}'
    print('  model_forward: PASS')


def test_onnx_load():
    """ONNX 权重加载 + 与 onnxruntime 输出一致."""
    import onnxruntime as ort
    for pos in POSITIONS:
        path = os.path.join(MODEL_DIR, f'{pos}.onnx')
        model = load_onnx_weights(DouZeroModel(pos), path).eval()

        z = np.random.randn(1, SEQ_LEN, LSTM_INPUT).astype(np.float32)
        x = np.random.randn(1, X_DIMS[pos]).astype(np.float32)

        pt_out = predict(model, {'z_batch': z, 'x_batch': x}, 'cpu')
        onnx_out = ort.InferenceSession(path).run(
            ['values'], {'z': z, 'x': x})[0].flatten()

        diff = np.abs(pt_out - onnx_out).max()
        assert diff < 1e-2, f'{pos}: diff={diff}'
    print('  onnx_load: PASS')


def test_onnx_export():
    """PyTorch -> ONNX 导出 + 推理一致 (含 models_best/ 的权重名格式)."""
    for src in (MODEL_DIR, os.path.join(os.path.dirname(__file__), 'models_best')):
        model = load_onnx_weights(DouZeroModel('landlord'),
                                  os.path.join(src, 'landlord.onnx')).eval()
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            path = f.name
        export_to_onnx(model, path, opset=14)
        verify_export(model, path)
        os.remove(path)
    print('  onnx_export: PASS')


def test_selfplay_episode():
    traj, rewards, winner, steps = run_episode(models=None, eps=0.0,
                                               model_dir=MODEL_DIR)
    assert steps > 0 and winner in ('landlord', 'farmer')
    for pos in POSITIONS:
        for step in traj[pos]:
            assert step['z'].dtype == np.uint8 and step['x'].dtype == np.uint8
    print(f'  selfplay: PASS (winner={winner}, steps={steps})')


def test_training_step():
    """跑一局 -> 取 trajectory -> 一步梯度更新."""
    models = DMCModel()
    models.load_onnx_all()
    traj, rewards, _, _ = run_episode(models=None, eps=0.0, model_dir=MODEL_DIR)

    for pos in POSITIONS:
        if not traj[pos]:
            continue
        model = models.get_model(pos)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-6)
        step = traj[pos][0]
        values = model(torch.from_numpy(step['z'][None]).float(),
                       torch.from_numpy(step['x'][None]).float()).squeeze(-1)
        loss = (values - rewards[pos]) ** 2
        opt.zero_grad()
        loss.backward()
        opt.step()
    print('  training_step: PASS')


if __name__ == '__main__':
    print('=== DMC Pipeline Tests ===')
    test_env_basic()
    test_obs_shape()
    test_model_forward()
    test_onnx_load()
    test_onnx_export()
    test_selfplay_episode()
    test_training_step()
    print('\nAll tests passed.')
