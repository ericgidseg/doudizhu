"""Sanity checks for the DMC training pipeline."""
import os
import sys
import tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from model import (DouZeroModel, DMCModel, load_onnx_weights,
                   export_to_onnx, predict, X_DIMS, MODEL_DIR,
                   SEQ_LEN, LSTM_INPUT)
from env.env import Env, get_obs
from selfplay import run_episode, predict_onnx

POSITIONS = ['landlord', 'landlord_up', 'landlord_down']


def test_env_basic():
    """Env 基本功能: reset, step, game over."""
    env = Env(objective='adp')
    env.reset()
    assert env.infoset is not None
    assert env.infoset.legal_actions
    assert env._acting_player_position == 'landlord'
    print('  env_basic: PASS')


def test_obs_shape():
    """Obs 特征维度正确."""
    env = Env(objective='adp')
    env.reset()
    obs = get_obs(env.infoset)
    assert obs['z_batch'].shape[1] == SEQ_LEN
    assert obs['z_batch'].shape[2] == LSTM_INPUT
    pos = env.infoset.player_position
    assert obs['x_batch'].shape[1] == X_DIMS[pos]
    print(f'  obs_shape: PASS (x_dim={X_DIMS[pos]})')


def test_model_forward():
    """PyTorch model 前向."""
    for pos in POSITIONS:
        model = DouZeroModel(pos)
        x_dim = X_DIMS[pos]
        z = torch.randn(4, SEQ_LEN, LSTM_INPUT)
        x = torch.randn(4, x_dim)
        out = model(z, x)
        assert out.shape == (4, 1), f'{pos}: {out.shape}'
    print('  model_forward: PASS')


def test_onnx_load():
    """ONNX 权重加载 + 输出一致."""
    for pos in POSITIONS:
        model = DouZeroModel(pos)
        load_onnx_weights(model, os.path.join(MODEL_DIR, f'{pos}.onnx'))
        model.eval()

        x_dim = X_DIMS[pos]
        z = np.random.randn(1, SEQ_LEN, LSTM_INPUT).astype(np.float32)
        x = np.random.randn(1, x_dim).astype(np.float32)

        pt_out = predict(model, {'z_batch': z, 'x_batch': x}, 'cpu')

        import onnxruntime as ort
        sess = ort.InferenceSession(os.path.join(MODEL_DIR, f'{pos}.onnx'))
        onnx_out = sess.run(['values'], {'z': z, 'x': x})[0].flatten()

        diff = np.abs(pt_out - onnx_out).max()
        assert diff < 1e-2, f'{pos}: diff={diff}'
    print('  onnx_load: PASS')


def test_onnx_export():
    """PyTorch -> ONNX 导出 + 推理."""
    model = DouZeroModel('landlord')
    load_onnx_weights(model, os.path.join(MODEL_DIR, 'landlord.onnx'))
    model.eval()

    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        path = f.name
    export_to_onnx(model, path, opset=14)

    import onnxruntime as ort
    sess = ort.InferenceSession(path)
    z = np.random.randn(4, SEQ_LEN, LSTM_INPUT).astype(np.float32)
    x = np.random.randn(4, 373).astype(np.float32)
    out = sess.run(['values'], {'z': z, 'x': x})[0]
    assert out.shape == (4, 1)
    os.remove(path)
    print('  onnx_export: PASS')


def test_selfplay_episode():
    """跑一局自对弈."""
    traj, rewards, winner, steps = run_episode(
        models=None, eps=0.0, model_dir=MODEL_DIR)
    assert steps > 0
    assert winner in ['landlord', 'farmer']
    for pos in POSITIONS:
        assert isinstance(rewards[pos], float)
        if traj[pos]:
            # 轨迹只存选中 action 的特征行, 且为 uint8 (省内存)
            assert 'z' in traj[pos][0], f'{pos}: missing z'
            assert 'x' in traj[pos][0], f'{pos}: missing x'
            assert traj[pos][0]['z'].dtype == np.uint8
            assert traj[pos][0]['x'].dtype == np.uint8
            assert 'action_idx' in traj[pos][0]
    print(f'  selfplay: PASS (winner={winner}, steps={steps})')


def test_dmc_loss():
    """DMC loss 计算."""
    models = DMCModel(device='cpu')
    models.load_onnx_all()
    models.eval()

    traj, rewards, winner, _ = run_episode(
        models=None, eps=0.0, model_dir=MODEL_DIR)

    for pos in POSITIONS:
        if not traj[pos]:
            continue
        model = models.get_model(pos)
        model.train()

        # 取第一个样本 (轨迹已只存选中 action 的特征行)
        item = traj[pos][0]
        z = torch.from_numpy(item['z'][None, ...]).float()
        x = torch.from_numpy(item['x'][None, ...]).float()
        target = torch.tensor(rewards[pos], dtype=torch.float32)

        values = model(z, x).squeeze(-1)
        loss = (values - target) ** 2
        assert loss.item() >= 0
    print('  dmc_loss: PASS')


def test_full_pipeline():
    """端到端: 训练几步 -> 导出 -> 推理."""
    models = DMCModel(device='cpu')
    models.load_onnx_all()

    # 手动一步梯度更新
    for pos in POSITIONS:
        model = models.get_model(pos)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)

        x_dim = X_DIMS[pos]
        z = torch.randn(8, SEQ_LEN, LSTM_INPUT)
        x = torch.randn(8, x_dim)
        target = torch.randn(8)

        values = model(z, x).squeeze(-1)
        loss = ((values - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.eval()

    # 导出
    with tempfile.TemporaryDirectory() as td:
        for pos in POSITIONS:
            m = models.get_model(pos)
            export_to_onnx(m, os.path.join(td, f'{pos}.onnx'), opset=14)

        # 用导出的模型跑一局
        traj, rewards, winner, steps = run_episode(
            models=None, eps=0.0, model_dir=td)
        assert steps > 0
    print('  full_pipeline: PASS')


if __name__ == '__main__':
    print('=== DMC Pipeline Tests ===')
    test_env_basic()
    test_obs_shape()
    test_model_forward()
    test_onnx_load()
    test_onnx_export()
    test_selfplay_episode()
    test_dmc_loss()
    test_full_pipeline()
    print('\nAll tests passed.')
