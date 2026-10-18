"""PyTorch 重建 DouZero LSTM 网络, 支持从 ONNX 导入权重 / 导出回 ONNX.

模型结构 (对齐 ONNX):
  z: (batch, 5, 162) -> LSTM(162->128) -> 取最后时间步 -> (batch, 128)
  x: (batch, x_dim)  -> concat -> (batch, 128+x_dim)
  -> 6 层 Dense(512) (前 5 层 ReLU) -> values (batch, 1)

输入维度: landlord x=373, landlord_up/down x=484

ONNX LSTM 门序是 iofc (input/output/forget/cell), PyTorch 是 ifgo,
导入/导出时都要重排. ONNX LSTM bias 布局是 [Wb | Rb], 不交错.

导出必须 dynamo=False, 否则 batch 维度会被固定成 1.
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

        self.lstm = nn.LSTM(
            input_size=LSTM_INPUT,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
        )

        dense_in = HIDDEN_SIZE + x_dim
        self.dense1 = nn.Linear(dense_in, FC_HIDDEN)
        self.dense2 = nn.Linear(FC_HIDDEN, FC_HIDDEN)
        self.dense3 = nn.Linear(FC_HIDDEN, FC_HIDDEN)
        self.dense4 = nn.Linear(FC_HIDDEN, FC_HIDDEN)
        self.dense5 = nn.Linear(FC_HIDDEN, FC_HIDDEN)
        self.dense6 = nn.Linear(FC_HIDDEN, 1)

        self.relu = nn.ReLU()

    def forward(self, z, x):
        """z: (batch, 5, 162)  x: (batch, x_dim)  ->  (batch, 1) Q 值."""
        lstm_out, _ = self.lstm(z)
        lstm_out = lstm_out[:, -1, :]

        h = self.relu(self.dense1(torch.cat([lstm_out, x], dim=1)))
        h = self.relu(self.dense2(h))
        h = self.relu(self.dense3(h))
        h = self.relu(self.dense4(h))
        h = self.relu(self.dense5(h))
        return self.dense6(h)


# ---------------------------------------------------------------------------
# ONNX <-> PyTorch 权重转换
# ---------------------------------------------------------------------------

def _onnx_to_torch_lstm(W, R, B):
    """ONNX LSTM (iofc, bias=[Wb|Rb]) -> PyTorch (ifgo)."""
    H = HIDDEN_SIZE
    W = W.reshape(4, H, -1)
    R = R.reshape(4, H, -1)

    # ONNX 索引 i=0, o=1, f=2, c=3  ->  PyTorch i=0, f=1, g=2, o=3
    W_pt = np.concatenate([W[0], W[2], W[3], W[1]], axis=0)
    R_pt = np.concatenate([R[0], R[2], R[3], R[1]], axis=0)

    B = B.reshape(-1)
    Wb = B[:4 * H].reshape(4, H)
    Rb = B[4 * H:].reshape(4, H)
    Wb_pt = np.concatenate([Wb[0], Wb[2], Wb[3], Wb[1]])
    Rb_pt = np.concatenate([Rb[0], Rb[2], Rb[3], Rb[1]])

    return W_pt, R_pt, Wb_pt, Rb_pt


def load_onnx_weights(model, onnx_path):
    """从 ONNX 模型加载权重到 PyTorch model.

    权重名兼容两种来源: 官方模型是 `base.denseN.*`, PyTorch 重新导出的
    是 `denseN.*`, 所以按后缀匹配.
    """
    import onnx
    m = onnx.load(onnx_path)
    weights = {init.name: np.frombuffer(init.raw_data, dtype=np.float32).reshape(init.dims)
               for init in m.graph.initializer}

    def find(suffix):
        for name, arr in weights.items():
            if name == suffix or name.endswith('.' + suffix):
                return arr
        raise KeyError(f'{suffix} not found in {onnx_path}')

    with torch.no_grad():
        W_pt, R_pt, Wb_pt, Rb_pt = _onnx_to_torch_lstm(
            find('onnx::LSTM_131'), find('onnx::LSTM_132'), find('onnx::LSTM_133'))
        model.lstm.weight_ih_l0.copy_(torch.from_numpy(W_pt))
        model.lstm.weight_hh_l0.copy_(torch.from_numpy(R_pt))
        model.lstm.bias_ih_l0.copy_(torch.from_numpy(Wb_pt))
        model.lstm.bias_hh_l0.copy_(torch.from_numpy(Rb_pt))

        for i in range(1, 7):
            fc = getattr(model, f'dense{i}')
            fc.weight.copy_(torch.from_numpy(find(f'dense{i}.weight').copy()))
            fc.bias.copy_(torch.from_numpy(find(f'dense{i}.bias').copy()))

    return model


def export_to_onnx(model, save_path, opset=14):
    """导出 PyTorch model 为 ONNX, 输入输出名与官方一致."""
    model.eval()
    dummy_z = torch.randn(1, SEQ_LEN, LSTM_INPUT)
    dummy_x = torch.randn(1, X_DIMS[model.position])

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


def verify_export(model, onnx_path, device='cpu'):
    """验证导出的 ONNX 和 PyTorch 输出一致."""
    import onnxruntime as ort
    model.eval()
    x_dim = X_DIMS[model.position]

    for batch_size in [1, 4]:
        z = np.random.randn(batch_size, SEQ_LEN, LSTM_INPUT).astype(np.float32)
        x = np.random.randn(batch_size, x_dim).astype(np.float32)

        with torch.no_grad():
            pt_out = model(torch.from_numpy(z).to(device),
                           torch.from_numpy(x).to(device)).cpu().numpy()

        sess = ort.InferenceSession(onnx_path)
        onnx_out = sess.run(['values'], {'z': z, 'x': x})[0]

        max_diff = np.abs(pt_out - onnx_out).max()
        print(f'  [verify] batch={batch_size} max_diff={max_diff:.6f}  '
              f'{"PASS" if max_diff < 1e-2 else "FAIL"}')
        assert max_diff < 1e-2, f'ONNX export mismatch: {max_diff}'
    return True


@torch.no_grad()
def predict(model, obs, device='cpu'):
    """对 obs dict 做推理, 返回各合法动作的 Q 值."""
    z = torch.from_numpy(obs['z_batch']).float().to(device)
    x = torch.from_numpy(obs['x_batch']).float().to(device)
    return model(z, x).cpu().numpy().flatten()


# ---------------------------------------------------------------------------
# 训练辅助
# ---------------------------------------------------------------------------

POSITIONS = ['landlord', 'landlord_up', 'landlord_down']


class DMCModel(nn.Module):
    """三个位置的 DouZeroModel 集合."""

    def __init__(self):
        super().__init__()
        self.models = nn.ModuleDict({p: DouZeroModel(p) for p in POSITIONS})

    def get_model(self, position):
        return self.models[position]

    def load_onnx_all(self, model_dir=None):
        model_dir = model_dir or MODEL_DIR
        for pos in POSITIONS:
            load_onnx_weights(self.models[pos], os.path.join(model_dir, f'{pos}.onnx'))
        return self
