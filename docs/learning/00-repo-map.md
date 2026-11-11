# 00 — Repo 地圖：這個 codebase 怎麼組織

> 一開始會覺得這個 repo 很複雜，主要原因不是模型難，而是 **它是從 X-Decoder / SEEM 研究框架整包複製來的**。
> 原框架支援十幾種任務和資料集，這個比賽只用到其中一小部分。
> 所以第一步不是讀懂每個檔案，而是**分清楚哪些用得到、哪些可以先忽略**。

## 1. 和你之前寫的 CNN 作業比，多了什麼

簡單的 CNN 作業通常是一個檔案搞定：

```python
model = MyCNN()                       # 模型
loader = DataLoader(MNIST(...))       # 資料
for x, y in loader:                   # 訓練迴圈
    loss = F.cross_entropy(model(x), y)
    loss.backward(); opt.step()
```

這個 repo 也是**同樣這四件事**，只是每一件都被拆成一個資料夾，而且用 config 檔來決定要組裝哪些零件：

| 簡單 CNN 作業 | 這個 repo 對應的位置 |
|---|---|
| `MyCNN()` | `modeling/`（模型零件）+ `configs/*.yaml`（決定用哪些零件、多大） |
| `DataLoader(MNIST)` | `datasets/`（資料集註冊、前處理、評估） |
| `for` 訓練迴圈 | `trainer/` + `pipeline/` |
| `python train.py` | `entry.py` + `command.sh` |
| 部署到手機 | `compile_and_profile/`（這個比賽才有的部分） |

## 2. 核心設計模式：Config + Registry（註冊表）

這是看懂這個 repo **最重要的一個概念**。
程式碼裡幾乎看不到 `model = FocalNet()` 這種直接建立物件的寫法，而是：

1. 每個零件檔案在定義時用 `@register_xxx` 把自己登記到一個字典裡，key 通常就是**檔名**。
2. yaml config 裡寫 `NAME: focal_dw`。
3. `build_xxx(config)` 去字典裡查 `focal_dw`，找到對應的函式並建立物件。

好處是換零件時只要改 yaml，不用改 code。壞處是**用 IDE 的「跳到定義」追不到**，你必須自己知道 NAME 對應到哪個檔案。
你之前的修改沒有生效，根本原因就是這個：改到了一個沒有被註冊、也沒被引用的檔案。

### 本比賽用到的 NAME → 檔案對照表

以 `configs/xdecoder/focalt_unicl_lang_finetune.yaml` 為例：

| Config 欄位 | 值 | 實際檔案 | 負責什麼 |
|---|---|---|---|
| `MODEL.NAME` | `xdecoder_model` | `modeling/architectures/xdecoder_model.py` | 訓練用的完整模型外殼（組裝零件、算 loss） |
| `MODEL.HEAD` | `xdecoder_head` | `modeling/body/xdecoder_head.py` | 把 pixel decoder 和 decoder 包在一起 |
| `MODEL.BACKBONE.NAME` | `focal_dw` | `modeling/vision/backbone/focal_dw.py` | 影像特徵抽取（FocalNet-T） |
| `MODEL.ENCODER.NAME` | `transformer_encoder_fpn` | `modeling/vision/encoder/transformer_encoder_fpn.py` | Pixel decoder（FPN + Transformer encoder） |
| `MODEL.DECODER.NAME` | `xdecoder` | `modeling/interface/xdecoder_qnn.py` ⚠️ | 核心 decoder。**注意**：`modeling/interface/__init__.py` 會自動在名稱後面加 `_qnn` |
| `MODEL.TEXT.ARCH` | `vlpencoder` | `modeling/language/vlpencoder.py` | 文字端外殼（tokenizer + 投影到 512 維） |
| `MODEL.TEXT.NAME` | `transformer` | `modeling/language/LangEncoder/transformer.py` | 12 層文字 Transformer 本體 |
| `PIPELINE` | `XDecoderPipeline` | `pipeline/XDecoderPipeline.py` | 一個 training step / evaluation 怎麼跑 |
| `TRAINER` | `xdecoder` | `trainer/xdecoder_trainer.py` | optimizer、lr scheduler、訓練迴圈 |

**要改哪一層，就改這張表右邊那個檔案。**

## 3. 資料夾導覽：🟢 這個比賽會用到 ／ ⚪ 可以先忽略

```
.
├── entry.py                 🟢 訓練 / 評估的入口（python entry.py train|evaluate）
├── command.sh               🟢 範例訓練指令（用 --overrides 覆蓋 yaml 裡的值）
├── configs/
│   ├── xdecoder/            🟢 X-Decoder 的各種 config
│   │   ├── focalt_unicl_lang_finetune.yaml   🟢 command.sh 用的
│   │   ├── focalt_unicl_lang_lpcvc25.yaml    🟢 部署腳本會用到的設定
│   │   └── 其他（davit / focall …）           ⚪ 其他 backbone 或更大模型
│   └── seem/                ⚪ SEEM（另一個模型，比賽不用）
│
├── modeling/                ← 模型都在這裡
│   ├── architectures/
│   │   ├── xdecoder_model.py   🟢 訓練時的完整模型
│   │   └── seem_model_*.py     ⚪
│   ├── vision/
│   │   ├── backbone/focal_dw.py            🟢 影像 backbone
│   │   ├── backbone/其他（davit, vit, focal） ⚪ 可替換的 backbone
│   │   ├── encoder/transformer_encoder_fpn.py 🟢 Pixel decoder
│   │   ├── encoder/transformer_blocks.py     🟢 上面用到的 Transformer 層
│   │   └── encoder/ops/, transformer_encoder_deform.py ⚪ Deformable attention（QNN 不支援）
│   ├── interface/
│   │   ├── xdecoder_qnn.py   🟢🟢 真正在用的 decoder（為 QNN 改寫過的版本）
│   │   ├── modules.py        🟢 decoder 用的 Self/Cross-Attention、FFN、MLP
│   │   ├── xdecoder.py       ⚪ 原版 decoder（沒被用到，被 _qnn 版取代）
│   │   └── seem_*.py, prototype/ ⚪
│   ├── body/
│   │   ├── xdecoder_head.py  🟢 把 encoder + decoder 包起來
│   │   ├── decoder/          ⚪⚠️ 舊版的重複程式碼，沒有被 import（之前改錯的就是這裡）
│   │   ├── encoder/          ⚪⚠️ 同上
│   │   └── transformer_blocks.py ⚪⚠️ 同上
│   ├── backbone/             ⚪⚠️ 舊版的重複程式碼，真正用的是 modeling/vision/backbone/
│   ├── language/             🟢 文字 encoder
│   ├── modules/
│   │   ├── matcher.py        🟢 訓練用：匈牙利匹配
│   │   ├── criterion.py      🟢 訓練用：loss
│   │   └── 其他               🟢 position encoding、後處理等小工具
│   └── utils/                🟢 config 小工具（configurable decorator 等）
│
├── datasets/                ← 資料
│   ├── registration/        🟢 告訴程式「某個資料集名稱對應到哪些檔案」
│   │   ├── register_coco_panoptic_annos_caption_grounding.py 🟢 訓練資料（COCO + grounding 標註）
│   │   ├── register_refcoco_dataset.py                       🟢 評估資料（RefCOCOg）
│   │   └── 其他（ade20k, scannet, bdd …）                      ⚪
│   ├── dataset_mappers/     🟢 一筆資料怎麼讀圖、做 augmentation、轉成 tensor
│   ├── evaluation/grounding_evaluation.py 🟢 算 grounding 的 IoU
│   └── build.py             🟢 組裝 dataloader 和 evaluator
│
├── trainer/                 🟢 訓練迴圈、存讀 checkpoint、多 GPU
├── pipeline/                🟢 單一 step 的 forward / evaluate 流程
├── utils/                   🟢 讀 config（arguments.py）、分散式工具等
│
├── compile_and_profile/     🟢🟢 這個比賽專屬：部署
│   ├── build_baseline_model.py         🟢 部署版模型（只保留推論需要的部分）
│   └── compile_profile_inference_aihub.py 🟢 匯出 ONNX → 上傳 AI Hub 編譯、測速、推論
│
├── demo/, inference/        ⚪ 原 X-Decoder / SEEM 的 demo
└── xdecoder_data/           🟢 資料放這裡（需要自己下載）
```

## 4. 兩條主線：訓練 vs. 部署

這個 repo 有**兩個入口**，會組出兩種不同外殼的模型，但**中間的零件是同一批檔案**。

### 主線 A：訓練 / 評估

```
command.sh
 └─ entry.py train --conf_files xxx.yaml --overrides KEY VALUE ...
     ├─ utils/arguments.py         讀 yaml，再用 --overrides 覆蓋
     ├─ trainer/xdecoder_trainer.py 建 optimizer、scheduler
     │   └─ trainer/default_trainer.py  train()：epoch 迴圈
     └─ pipeline/XDecoderPipeline.py
         ├─ initialize_model() → modeling.build_model() → GeneralizedXdecoder
         │                       （architectures/xdecoder_model.py）
         │    ├─ backbone        vision/backbone/focal_dw.py
         │    ├─ sem_seg_head    body/xdecoder_head.py
         │    │    ├─ pixel_decoder  vision/encoder/transformer_encoder_fpn.py
         │    │    └─ predictor      interface/xdecoder_qnn.py
         │    ├─ lang_encoder    language/vlpencoder.py
         │    └─ criterion       modules/criterion.py + modules/matcher.py
         ├─ get_dataloaders()   → datasets/build.py
         └─ forward_step()      → 算 loss、backward
```

### 主線 B：部署（比賽評分真正跑的東西）

```
python compile_and_profile/compile_profile_inference_aihub.py
 ├─ 準備輸入：圖片 1x3x1024x1024、文字 2x1x77（CLIP tokenizer）
 ├─ build_baseline_model.py
 │   ├─ 個別建立 backbone / encoder / lang_encoder / decoder（同樣那幾個檔案）
 │   ├─ 從 checkpoint 載入權重（backbone / encoder / 文字 encoder 用 strict=True，名稱對不上會直接報錯；
 │   │   decoder 用 strict=False，因為部署時不需要 captioning 等權重，對不上的會被默默丟掉）
 │   └─ 包成一個精簡的 XDecoder(nn.Module)：只有推論，沒有 loss
 ├─ torch → ONNX
 └─ 上傳 Qualcomm AI Hub：compile（轉成 QNN）→ profile（量延遲）→ inference（驗證輸出）
```

**重點**：訓練時用 `GeneralizedXdecoder`，部署時用 `build_baseline_model.py` 裡的 `XDecoder`，
兩個外殼不一樣，但 backbone / encoder / decoder / 文字 encoder 是同一份程式碼、同一份權重。
所以改模型時要確認**兩條主線都會用到你的修改**。

## 5. 讀 code 的建議順序

1. `compile_and_profile/build_baseline_model.py` 的 `XDecoder.forward()`：最短、最乾淨的推論流程（第 01 課）
2. 照著 forward 的順序往下讀零件：backbone → pixel decoder → 文字 encoder → decoder（第 02–05 課）
3. 最後才讀訓練：`xdecoder_model.py` 的 loss 部分、`matcher.py`、`criterion.py`（第 06 課）

`trainer/`、`pipeline/`、`datasets/` 大部分是工程上的通用框架，**知道它們在做什麼就好，不用逐行讀**。

## 6. 自我檢查

- [ ] 我能說出 `MODEL.DECODER.NAME: xdecoder` 最後會建立哪個檔案裡的哪個 class 嗎？
- [ ] 如果要換 FFN 的設計，應該改哪個檔案？（提示：看 `xdecoder_qnn.py` 從哪裡 import `FFNLayer`）
- [ ] 為什麼 `modeling/body/decoder/` 裡的修改不會有任何效果？
- [ ] 訓練和部署兩條主線，共用了哪些檔案？
