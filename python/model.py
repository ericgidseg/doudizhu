"""PyTorch 重建 DouZero LSTM 网络, 支持从 ONNX 导入权重 / 导出回 ONNX.

模型结构 (对齐 ONNX):
  z: (batch, 5, 162) -> LSTM(162->128) -> 取最后时间步 -> lstm_out (batch, 128)
  x: (batch, x_dim)  -> concat(lstm_out, x) -> (batch, 128+x_dim)
  -> dense1 -> ReLU -> dense2 -> ReLU -> ... -> dense5 -> ReLU -> dense6 -> values (batch, 1)

输入维度:
  landlord:      x=373 -> dense1 input=501
  landlord_up:   x=484 -> dense1 input=612
  landlord_down: x=484 -> dense1 input=612

ONNX LSTM 门序: [input, output, forget, cell]  (iofc)
PyTorch LSTM 门序: [input, forget, cell, output] (ifgo)
导出时做 ifgo -> iofc 重排.
"""
import os
import numpy as np
import torch
import torch.nn as nn

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

X_DIMS = {
    'landlord': 373,
    'landlord_up': 484,
    'landlord_down': 484,
}

HIDDEN_SIZE = 128
NUM_LAYERS = 1
FC_HIDDEN = 512
SEQ_LEN = 5
LSTM_INPUT = 162


class DouZeroModel(nn.Module):
    """DouZero LSTM-Q 网络."""

    def __init__(self, position):
        super().__init__()
        assert position in X_DIMS
        self.position = position
        x_dim = X_DIMS[position]

        # LSTM: (batch, seq=5, feat=162) -> (batch, 128)
        self.lstm = nn.LSTM(
            input_size=LSTM_INPUT,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
        )

        # FC: concat(128 + x_dim) -> 512 -> 512 -> 512 -> 512 -> 512 -> 1
        dense_in = HIDDEN_SIZE + x_dim
        self.dense1 = nn.Linear(dense_in, FC_HIDDEN)
        self.dense2 = nn.Linear(FC_HIDDEN, FC_HIDDEN)
        self.dense3 = nn.Linear(FC_HIDDEN, FC_HIDDEN)
        self.dense4 = nn.Linear(FC_HIDDEN, FC_HIDDEN)
        self.dense5 = nn.Linear(FC_HIDDEN, FC_HIDDEN)
        self.dense6 = nn.Linear(FC_HIDDEN, 1)

        self.relu = nn.ReLU()

    def forward(self, z, x):
        """
        z: (batch, 5, 162)  float32
        x: (batch, x_dim)   float32
        returns: (batch, 1)  Q values
        """
        lstm_out, _ = self.lstm(z)          # (batch, 5, 128)
        lstm_out = lstm_out[:, -1, :]       # (batch, 128)  取最后时间步

        cat = torch.cat([lstm_out, x], dim=1)  # (batch, 128 + x_dim)

        h = self.relu(self.dense1(cat))
        h = self.relu(self.dense2(h))
        h = self.relu(self.dense3(h))
        h = self.relu(self.dense4(h))
        h = self.relu(self.dense5(h))
        return self.dense6(h)               # (batch, 1)


# ---------------------------------------------------------------------------
# ONNX <-> PyTorch 权重转换
# ---------------------------------------------------------------------------

def _onnx_to_torch_lstm(W, R, B):
    """ONNX LSTM iofc -> PyTorch ifgo.

    ONNX W: (1, 4*H, input)   gate order i,o,f,c
    ONNX R: (1, 4*H, hidden)   gate order i,o,f,c
    ONNX B: (1, 8*H)          [Wb_i,o,f,c | Rb_i,o,f,c]
    PyTorch weight_ih_l0: (4*H, input)  gate order i,f,g,o
    PyTorch weight_hh_l0: (4*H, hidden)  gate order i,f,g,o
    PyTorch bias_ih_l0:   (4*H,)        gate order i,f,g,o
    PyTorch bias_hh_l0:   (4*H,)        gate order i,f,g,o
    """
    H = HIDDEN_SIZE
    W = W.reshape(4, H, -1)   # (4, H, input)   i,o,f,c
    R = R.reshape(4, H, -1)   # (4, H, hidden)  i,o,f,c

    # ONNX -> PyTorch gate order: i,f,g,o  (g = cell in ONNX)
    # ONNX idx: i=0, o=1, f=2, c=3
    # PyTorch idx: i=0, f=1, g=2, o=3
    W_pt = np.concatenate([W[0], W[2], W[3], W[1]], axis=0)  # i,f,c,o
    R_pt = np.concatenate([R[0], R[2], R[3], R[1]], axis=0)

    B = B.reshape(-1)         # (8*H,)
    Wb = B[:4*H].reshape(4, H)  # i,o,f,c
    Rb = B[4*H:].reshape(4, H)  # i,o,f,c
    Wb_pt = np.concatenate([Wb[0], Wb[2], Wb[3], Wb[1]])
    Rb_pt = np.concatenate([Rb[0], Rb[2], Rb[3], Rb[1]])

    return W_pt, R_pt, Wb_pt, Rb_pt


def _torch_to_onnx_lstm(W_pt, R_pt, Wb_pt, Rb_pt):
    """PyTorch LSTM ifgo -> ONNX iofc."""
    H = HIDDEN_SIZE
    W = W_pt.reshape(4, H, -1)   # i,f,g,o
    R = R_pt.reshape(4, H, -1)

    # PyTorch -> ONNX: i=0, o=1, f=2, c=3
    # PyTorch idx: i=0, f=1, g=2, o=3
    W_onnx = np.concatenate([W[0], W[3], W[1], W[2]], axis=0)  # i,o,f,c
    R_onnx = np.concatenate([R[0], R[3], R[1], R[2]], axis=0)

    Wb = Wb_pt.reshape(4, H)
    Rb = Rb_pt.reshape(4, H)
    Wb_onnx = np.concatenate([Wb[0], Wb[3], Wb[1], Wb[2]])
    Rb_onnx = np.concatenate([Rb[0], Rb[3], Rb[1], Rb[2]])
    B_onnx = np.concatenate([Wb_onnx, Rb_onnx])

    return W_onnx.reshape(1, 4*H, -1), R_onnx.reshape(1, 4*H, -1), B_onnx.reshape(1, 8*H)


def load_onnx_weights(model, onnx_path):
    """从 ONNX 模型加载权重到 PyTorch model."""
    import onnx
    m = onnx.load(onnx_path)

    weights = {}
    for init in m.graph.initializer:
        arr = np.frombuffer(init.raw_data, dtype=np.float32).reshape(init.dims)
        weights[init.name] = arr

    with torch.no_grad():
        # LSTM
        W = weights['onnx::LSTM_131']
        R = weights['onnx::LSTM_132']
        B = weights['onnx::LSTM_133']

        W_pt, R_pt, Wb_pt, Rb_pt = _onnx_to_torch_lstm(W, R, B)

        model.lstm.weight_ih_l0.copy_(torch.from_numpy(W_pt))
        model.lstm.weight_hh_l0.copy_(torch.from_numpy(R_pt))
        model.lstm.bias_ih_l0.copy_(torch.from_numpy(Wb_pt))
        model.lstm.bias_hh_l0.copy_(torch.from_numpy(Rb_pt))

        # FC layers: ONNX Gemm weight is (out, in), PyTorch Linear is (out, in) — same
        for i in range(1, 7):
            w = weights[f'base.dense{i}.weight'].copy()
            b = weights[f'base.dense{i}.bias'].copy()
            fc = getattr(model, f'dense{i}')
            fc.weight.copy_(torch.from_numpy(w))
            fc.bias.copy_(torch.from_numpy(b))

    print(f'  [model] loaded ONNX weights from {onnx_path}')
    return model


def load_model(position, onnx_path=None, device='cpu'):
    """创建 DouZeroModel 并加载 ONNX 权重."""
    model = DouZeroModel(position)
    if onnx_path is None:
        onnx_path = os.path.join(MODEL_DIR, f'{position}.onnx')
    model = load_onnx_weights(model, onnx_path)
    model = model.to(device)
    return model


def export_to_onnx(model, save_path, opset=14):
    """导出 PyTorch model 为 ONNX, 匹配原始模型输入输出格式."""
    model.eval()
    x_dim = X_DIMS[model.position]
    batch = 1

    dummy_z = torch.randn(batch, SEQ_LEN, LSTM_INPUT)
    dummy_x = torch.randn(batch, x_dim)

    torch.onnx.export(
        model,
        (dummy_z, dummy_x),
        save_path,
        input_names=['z', 'x'],
        output_names=['values'],
        dynamic_axes={
            'z': {0: 'batch'},
            'x': {0: 'batch'},
            'values': {0: 'batch'},
        },
        dynamo=False,
        opset_version=opset,
    )
    print(f'  [model] exported ONNX to {save_path}')


def verify_export(model, onnx_path, device='cpu'):
    """验证导出的 ONNX 和 PyTorch 输出一致."""
    import onnxruntime as ort
    model.eval()
    x_dim = X_DIMS[model.position]

    # 构造两个不同输入测试
    for batch_size in [1, 4]:
        z = np.random.randn(batch_size, SEQ_LEN, LSTM_INPUT).astype(np.float32)
        x = np.random.randn(batch_size, x_dim).astype(np.float32)

        with torch.no_grad():
            pt_out = model(torch.from_numpy(z).to(device),
                           torch.from_numpy(x).to(device)).cpu().numpy()

        sess = ort.InferenceSession(onnx_path)
        onnx_out = sess.run(['values'], {'z': z, 'x': x})[0]

        max_diff = np.abs(pt_out - onnx_out).max()
        status = 'PASS' if max_diff < 1e-2 else 'FAIL'
        print(f'  [verify] batch={batch_size} max_diff={max_diff:.6f}  {status}')
        assert max_diff < 1e-2, f'ONNX export mismatch: {max_diff}'

    return True


# ---------------------------------------------------------------------------
# 推理接口 (兼容 ai_douzero.py 的调用方式)
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(model, obs, device='cpu'):
    """对 obs dict 做推理, 返回各合法动作的 Q 值."""
    z = torch.from_numpy(obs['z_batch']).float().to(device)
    x = torch.from_numpy(obs['x_batch']).float().to(device)
    values = model(z, x).cpu().numpy().flatten()
    return values


# ---------------------------------------------------------------------------
# 训练辅助
# ---------------------------------------------------------------------------

class DMCModel(nn.Module):
    """包装三个位置的 DouZeroModel, 统一管理."""

    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.models = nn.ModuleDict({
            'landlord': DouZeroModel('landlord'),
            'landlord_up': DouZeroModel('landlord_up'),
            'landlord_down': DouZeroModel('landlord_down'),
        })

    def forward(self, position, z, x):
        return self.models[position](z, x)

    def get_model(self, position):
        return self.models[position]

    def load_onnx_all(self, model_dir=None):
        if model_dir is None:
            model_dir = MODEL_DIR
        for pos in ['landlord', 'landlord_up', 'landlord_down']:
            path = os.path.join(model_dir, f'{pos}.onnx')
            load_onnx_weights(self.models[pos], path)
        return self


if __name__ == '__main__':
    # 测试: 加载 ONNX -> PyTorch -> 导出 -> 验证一致
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'Using device: {device}')

    for pos in ['landlord', 'landlord_up', 'landlord_down']:
        print(f'\n=== {pos} ===')
        # 在 CPU 上验证精度 (MPS 有浮点差异)
        model_cpu = load_model(pos, device='cpu')
        model_cpu.eval()

        import onnxruntime as ort
        obs_dim = X_DIMS[pos]
        z = np.random.randn(1, SEQ_LEN, LSTM_INPUT).astype(np.float32)
        x = np.random.randn(1, obs_dim).astype(np.float32)

        pt_out = predict(model_cpu, {'z_batch': z, 'x_batch': x}, device='cpu')
        sess = ort.InferenceSession(os.path.join(MODEL_DIR, f'{pos}.onnx'))
        onnx_out = sess.run(['values'], {'z': z, 'x': x})[0].flatten()
        diff = np.abs(pt_out - onnx_out).max()
        print(f'  PyTorch vs ONNX max diff: {diff:.6f}')
        assert diff < 1e-2, f'Mismatch: {diff}'

        # 导出再验证
        test_path = f'/tmp/test_{pos}.onnx'
        export_to_onnx(model_cpu, test_path)
        verify_export(model_cpu, test_path, 'cpu')
        os.remove(test_path)

    print('\nAll tests passed.')
