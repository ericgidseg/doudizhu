"""导出微调后的模型为 ONNX + 验证推理一致.

用法:
  python3 export.py                          # 从 checkpoints/latest.pt 导出
  python3 export.py --ckpt checkpoints/best.pt -o models_v2/
  python3 export.py --ckpt checkpoints/best.pt --verify
"""
import os
import sys
import copy
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import torch
import onnxruntime as ort

from model import (DMCModel, DouZeroModel, load_onnx_weights,
                   export_to_onnx, verify_export, X_DIMS, MODEL_DIR,
                   SEQ_LEN, LSTM_INPUT)

POSITIONS = ['landlord', 'landlord_up', 'landlord_down']
CKPT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')


def export_from_checkpoint(ckpt_path, export_dir):
    """从 PyTorch checkpoint 导出 ONNX 模型."""
    print(f'Loading checkpoint: {ckpt_path}')
    models = DMCModel(device='cpu')
    ckpt = torch.load(ckpt_path, map_location='cpu')

    for pos in POSITIONS:
        models.get_model(pos).load_state_dict(ckpt[f'model_{pos}'])
        models.get_model(pos).eval()

    os.makedirs(export_dir, exist_ok=True)

    for pos in POSITIONS:
        model = models.get_model(pos)
        path = os.path.join(export_dir, f'{pos}.onnx')
        export_to_onnx(model, path, opset=14)

    print(f'\nExported to {export_dir}/')
    return models


def compare_models(models, ref_dir, n_samples=5):
    """对比导出模型和参考 ONNX 的输出."""
    print('\n=== Comparing with reference ONNX ===')
    for pos in POSITIONS:
        ref_path = os.path.join(ref_dir, f'{pos}.onnx')
        if not os.path.exists(ref_path):
            print(f'  {pos}: no reference found')
            continue

        ref_sess = ort.InferenceSession(ref_path)
        model = models.get_model(pos)
        model.eval()

        x_dim = X_DIMS[pos]
        max_diff = 0.0

        for _ in range(n_samples):
            z = np.random.randn(4, SEQ_LEN, LSTM_INPUT).astype(np.float32)
            x = np.random.randn(4, x_dim).astype(np.float32)

            with torch.no_grad():
                pt_out = model(
                    torch.from_numpy(z), torch.from_numpy(x)
                ).numpy().flatten()

            onnx_out = ref_sess.run(['values'], {'z': z, 'x': x})[0].flatten()
            diff = np.abs(pt_out - onnx_out).max()
            max_diff = max(max_diff, diff)

        status = 'SAME' if max_diff < 1e-2 else 'DIFF'
        print(f'  {pos}: max_diff={max_diff:.6f}  [{status}]')


def verify_exported(export_dir):
    """验证导出的 ONNX 可以正常推理."""
    print('\n=== Verifying exported ONNX ===')
    for pos in POSITIONS:
        path = os.path.join(export_dir, f'{pos}.onnx')
        sess = ort.InferenceSession(path)
        x_dim = X_DIMS[pos]

        for batch in [1, 8, 32]:
            z = np.random.randn(batch, SEQ_LEN, LSTM_INPUT).astype(np.float32)
            x = np.random.randn(batch, x_dim).astype(np.float32)
            out = sess.run(['values'], {'z': z, 'x': x})[0]
            assert out.shape == (batch, 1), f'{pos} bad shape: {out.shape}'
            assert np.isfinite(out).all(), f'{pos} non-finite values'

        print(f'  {pos}: OK (z={list(sess.get_inputs()[0].shape)}, '
              f'x={list(sess.get_inputs()[1].shape)})')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                        default=os.path.join(CKPT_DIR, 'latest.pt'))
    parser.add_argument('-o', '--output', type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             'models_v2'))
    parser.add_argument('--verify', action='store_true',
                        help='验证导出模型')
    parser.add_argument('--compare', type=str, default=None,
                        help='对比参考 ONNX 目录')
    args = parser.parse_args()

    models = export_from_checkpoint(args.ckpt, args.output)

    if args.verify:
        verify_exported(args.output)

    if args.compare:
        compare_models(models, args.compare)

    print('\nDone.')


if __name__ == '__main__':
    main()
