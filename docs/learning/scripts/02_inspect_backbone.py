"""
第 02 課示範腳本：用隨機權重跑一次 FocalNet-T backbone，印出每一層的形狀和參數量。

不需要 GPU、不需要下載資料集或權重，也不需要安裝 detectron2。
只需要：pip install torch timm

用法（在 repo 根目錄執行）：
    python docs/learning/scripts/02_inspect_backbone.py
    python docs/learning/scripts/02_inspect_backbone.py --size 512
"""
import argparse
import importlib.util
import os
import sys
import time
import types

import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
FOCAL_PATH = os.path.join(REPO_ROOT, "modeling", "vision", "backbone", "focal_dw.py")


def stub_detectron2():
    """focal_dw.py 只用到 detectron2 的幾個小東西，這裡用假的代替，就不用安裝 detectron2。"""
    d2 = types.ModuleType("detectron2")
    utils = types.ModuleType("detectron2.utils")
    file_io = types.ModuleType("detectron2.utils.file_io")
    file_io.PathManager = None
    modeling = types.ModuleType("detectron2.modeling")

    class Backbone(nn.Module):
        pass

    class ShapeSpec:
        def __init__(self, channels=None, stride=None):
            self.channels, self.stride = channels, stride

    modeling.Backbone = Backbone
    modeling.ShapeSpec = ShapeSpec
    modeling.BACKBONE_REGISTRY = None
    for name, mod in {
        "detectron2": d2,
        "detectron2.utils": utils,
        "detectron2.utils.file_io": file_io,
        "detectron2.modeling": modeling,
    }.items():
        sys.modules[name] = mod


def load_focal_module():
    """直接載入 repo 裡的 focal_dw.py（就是訓練和部署真正在用的那個檔案）。"""
    stub_detectron2()
    pkg = types.ModuleType("lesson_backbone")
    pkg.__path__ = []
    sys.modules["lesson_backbone"] = pkg
    build = types.ModuleType("lesson_backbone.build")
    build.register_backbone = lambda fn: fn
    sys.modules["lesson_backbone.build"] = build

    spec = importlib.util.spec_from_file_location("lesson_backbone.focal_dw", FOCAL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# 和 configs/xdecoder/focalt_unicl_lang_lpcvc25.yaml 的 MODEL.BACKBONE 完全一樣
FOCAL_T_CFG = {
    "BACKBONE": {
        "FOCAL": {
            "PRETRAIN_IMG_SIZE": 224,
            "PATCH_SIZE": 4,
            "EMBED_DIM": 96,
            "DEPTHS": [2, 2, 6, 2],
            "FOCAL_LEVELS": [3, 3, 3, 3],
            "FOCAL_WINDOWS": [3, 3, 3, 3],
            "DROP_PATH_RATE": 0.3,
            "MLP_RATIO": 4.0,
            "DROP_RATE": 0.0,
            "PATCH_NORM": True,
            "USE_CONV_EMBED": True,
            "SCALING_MODULATOR": True,
            "USE_CHECKPOINT": False,
            "USE_POSTLN": True,
            "USE_POSTLN_IN_MODULATION": False,
            "USE_LAYERSCALE": True,
            "OUT_FEATURES": ["res2", "res3", "res4", "res5"],
            "OUT_INDICES": [0, 1, 2, 3],
        }
    }
}


def count(module):
    return sum(p.numel() for p in module.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256, help="送進 backbone 的影像邊長（baseline 是 256）")
    args = parser.parse_args()

    focal = load_focal_module()
    model = focal.D2FocalNet(FOCAL_T_CFG, 224).eval()

    print("=" * 70)
    print(f"FocalNet-T，輸入 (1, 3, {args.size}, {args.size})")
    print("=" * 70)

    # ---- 1. 每個部分的參數量 ----
    print("\n[1] 參數量")
    print(f"  patch_embed (stem)        : {count(model.patch_embed):>10,}")
    for i, layer in enumerate(model.layers):
        blocks = count(layer.blocks)
        down = count(layer.downsample) if layer.downsample is not None else 0
        print(f"  stage {i + 1}  ({len(layer.blocks)} blocks)      : {blocks:>10,}   + downsample {down:,}")
    print(f"  合計                      : {count(model):>10,}")

    # ---- 2. 用 hook 記錄每一步的輸出形狀 ----
    print("\n[2] 資料流程（每一步的輸出形狀）")
    records = []

    def hook(name):
        def fn(_m, _inp, out):
            if isinstance(out, tuple):
                out = out[0]  # BasicLayer 回傳 (x, H, W, x_down, Wh, Ww)，第一個是這個 stage 的輸出
            records.append((name, tuple(out.shape)))
        return fn

    model.patch_embed.register_forward_hook(hook("patch_embed (stem conv 7x7, stride 4)"))
    for i, layer in enumerate(model.layers):
        for j, blk in enumerate(layer.blocks):
            blk.register_forward_hook(hook(f"stage {i + 1} / block {j + 1}  (B, H*W, C)"))
        if layer.downsample is not None:
            layer.downsample.register_forward_hook(hook(f"stage {i + 1} / downsample (conv 3x3, stride 2)"))

    x = torch.randn(1, 3, args.size, args.size)
    with torch.no_grad():
        t0 = time.time()
        outs = model(x)
        dt = time.time() - t0

    for name, shape in records:
        print(f"  {name:<45} {shape}")

    print("\n[3] backbone 最後輸出給 pixel decoder 的 4 種尺度")
    for k, v in outs.items():
        stride = args.size // v.shape[-1]
        print(f"  {k}: {tuple(v.shape)}   stride {stride:>2}  （{v.shape[-1]}x{v.shape[-1]} 個位置，每個 {v.shape[1]} 維）")

    # ---- 3. Focal modulation 內部 ----
    print("\n[4] 一個 Focal Modulation 的內部（以 stage 1 第一個 block 為例）")
    fm = model.layers[0].blocks[0].modulation
    ks = [m[0].kernel_size[0] for m in fm.focal_layers]
    print(f"  focal_level = {fm.focal_level}，每一層 depthwise conv 的 kernel size = {ks}")
    rf = 1
    for k in ks:
        rf += k - 1
        print(f"    疊到 kernel {k}x{k} 之後，每個位置能看到的範圍 = {rf}x{rf}")
    print("    再加上一個 global average（整張圖的平均）→ 全域資訊")
    print(f"  f 層：Linear({fm.dim} → {fm.f.out_features}) = q({fm.dim}) + ctx({fm.dim}) + gates({fm.focal_level + 1})")

    print(f"\n[5] CPU 上跑一次約 {dt * 1000:.0f} ms（只是參考，和手機 NPU 上的延遲不同）")


if __name__ == "__main__":
    main()
