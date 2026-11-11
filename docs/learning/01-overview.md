# 01 — 全局地圖：一張圖 + 一句話，怎麼變成一個 mask

> 這一課只看部署版模型 `compile_and_profile/build_baseline_model.py` 的 `XDecoder.forward()`（第 221 行起）。
> 它是整個 repo 最短、最乾淨的推論流程，也是比賽評分時真正跑的東西。
> 目標不是看懂每一行，而是建立一張「資料從哪裡來、變成什麼形狀、到哪裡去」的地圖。

## 1. 這個模型在解決什麼問題

### 從 CNN 分割說起

你寫過的 CNN 最熟悉的大概是**分類**：輸入一張圖，輸出「這是狗」。
CNN 做分割時（FCN、U-Net），做法是**對每個 pixel 分類**：輸出一張 `H × W × N類` 的圖。
這個做法的前提是**類別清單固定**，例如 COCO 的 80 類。

### 這個比賽的任務：開放詞彙 + 文字提示

輸入是一張圖和一句話，例如「左邊那隻戴紅色項圈的狗」，輸出就是那個物體的 mask。
這句話可以是任何東西，沒有固定的類別清單，所以「每個 pixel 分成 N 類」行不通。

X-Decoder 用兩個想法解決：

1. **Query-based 分割**（來自 DETR → MaskFormer → Mask2Former 這條研究脈絡）
   不直接對 pixel 分類，而是先產生一堆**候選物體**，每個候選有自己的 mask。
   每個候選由一個 **query** 負責：query 是一個可學習的 512 維向量。
2. **影像和文字放進同一個向量空間**（來自 CLIP）
   每個候選物體除了 mask，還有一個 512 維的「語意向量」；文字也被編碼成一個 512 維向量。
   **和文字向量最相似的那個候選，就是答案。**

## 2. 資料流程與形狀（內部解析度 = 256 時）

```
輸入圖片 (1, 3, 1024, 1024)，數值 0~255         輸入文字 (2, 1, 77) = [token ids ; attention mask]
   │ pre_processing()：減 mean、除 std，縮到 256        │
   ▼                                                │
(1, 3, 256, 256)                                    │
   │                                                │
① Backbone：FocalNet-T                               ③ 文字 Encoder：12 層 Transformer
   res2 (1,  96, 64, 64)   stride 4                     token_emb (1, 77, 512)  每個 token 一個向量
   res3 (1, 192, 32, 32)   stride 8                     class_emb (1, 512)      整句話一個向量
   res4 (1, 384, 16, 16)   stride 16                         │
   res5 (1, 768,  8,  8)   stride 32                         │
   │                                                         │
② Pixel Decoder：FPN + 6 層 Transformer encoder               │
   multi_scale_features：8×8、16×16、32×32 三層，各 512 通道   │
   mask_features：(1, 512, 64, 64)  ← 最終 mask 的「畫布」     │
   │                                                         │
   └────────────────► ④ X-Decoder（9 層）◄─────────────────────┘
                        101 個 learnable queries（100 個物體 + 1 個 class token）
                        再複製出 100 個「可以看到文字」的 grounding queries
                        每層：masked cross-attention（看圖）→ self-attention（互看 + 看文字）→ FFN
                        │
                        ▼
                 pred_masks    (1, 100, 64, 64)  100 個候選 mask
                 pred_captions (1, 100, 512)     每個候選的語意向量
                        │
⑤ post_processing()：候選向量和 class_emb 算 cosine similarity → 取 top-1
   → 上採樣到 1024×1024 → >= 0 二值化 → 輸出 (1024, 1024)
```

### 對照程式碼

| 步驟 | 程式碼位置 | 做了什麼 |
|---|---|---|
| 前處理 | `build_baseline_model.py:152` `pre_processing()` | 正規化，`F.interpolate` 縮到 `down_sample_size = 256`（第 164 行） |
| ① Backbone | `self.backbone(images)` | 回傳 dict：`{'res2','res3','res4','res5'}` |
| ② Pixel Decoder | `self.feature_extractor(visual_features)` | 回傳 `mask_features, _, multi_scale_features` |
| ③ 文字 Encoder | `self.lang_encoder.get_text_token_embeddings(...)` | 取 `class_emb`，變成 `grounding_tokens` (1, 1, 512) |
| ④ Decoder | `self.mask_decoder(..., task='grounding_eval', extra=extra)` | 回傳 dict，包含 `pred_masks`、`pred_captions` |
| 取 grounding 結果 | 第 244–245 行 `[:, self.num_queries:]` | 丟掉前 101 個一般分割用的 queries，只留 100 個 grounding queries |
| ⑤ 後處理 | `build_baseline_model.py:187` `post_processing()` | 相似度 → top-1 → 上採樣 → 二值化 |

## 3. 用 CNN 的經驗理解每一塊

### ① Backbone ≈ ResNet
- 一樣是 4 個 stage，解析度每次減半（64→32→16→8），通道數每次加倍（96→192→384→768）。
- FocalNet 用 depthwise conv 做「focal modulation」，取代 Vision Transformer 的 self-attention。
  本質上很接近 CNN，所以對手機 NPU 很友善。第 02 課會細看。

### ② Pixel Decoder ≈ FPN
- 深層特徵（res5）語意強、解析度低；淺層特徵（res2）細節多、語意弱。
- FPN 由上往下，把深層特徵上採樣後和淺層特徵相加，兩者的優點都留下來。
- 最小的 8×8 那層先過 6 層 Transformer encoder，讓每個位置都能看到整張圖（全域資訊）。CNN 的 receptive field 做不到這件事。
- 輸出兩種東西：
  - `multi_scale_features`：給 decoder 的 cross-attention 看的三種尺度特徵。
  - `mask_features`：1/4 解析度的「畫布」，最後的 mask 都畫在這上面。

### ③ 文字 Encoder ≈ CLIP 的文字端
- 句子先被 CLIP tokenizer 切成最多 77 個 token（這一步在模型外面做，所以輸入已經是 token ids）。
- 12 層 Transformer 之後，取句尾 token 的向量、投影到 512 維，作為「整句話的意思」= `class_emb`。

### ④ X-Decoder：最不像 CNN 的部分
每個 query 像一個**會主動去圖裡找東西的探針**：

- **Mask 怎麼產生**：query 向量 (512) 和 `mask_features` 上每個 pixel 的向量 (512) 做內積，
  得到一張 64×64 的分數圖，這就是 mask。程式碼在 `modeling/interface/xdecoder_qnn.py:448-455`
  （原本是 `einsum("bqc,bchw->bqhw")`，為了 QNN 改寫成 `matmul`）。
- **Masked cross-attention**：每一層中，query 只看「自己上一層預測的 mask 範圍內」的 pixels，
  一層一層把 mask 修得更準。這就是之後實作 B 要修的效能瓶頸 `attn_mask` 的由來。
- **Self-attention**：queries 之間互相交換資訊；grounding queries 還會看到文字 token，
  這就是「文字資訊進入影像候選」的地方。
- **FFN**：每個 query 各自過一個兩層 MLP（512 → 2048 → 512）。

### ⑤ 後處理：開放詞彙的關鍵
- 100 個候選，每個都有一個 512 維語意向量；文字有一個 512 維向量。
- 兩者都做 L2 正規化後相乘 = cosine similarity，乘上溫度參數。
- 分數最高的候選就是答案，取它的 mask。

## 4. 從這張地圖可以看出的優化方向

把形狀和運算量對應起來：

| 部分 | 處理的東西 | 運算量和解析度的關係 |
|---|---|---|
| ① Backbone | 整張圖，每個 pixel 都要算 | 解析度平方：256 → 512 就是 4 倍 |
| ② Pixel Decoder | 三種尺度的特徵圖 | 解析度平方 |
| ③ 文字 Encoder | 77 個 token | 和影像解析度無關 |
| ④ Decoder 的 FFN / self-attention | 約 200 個 queries | 和解析度無關，而且很小 |
| ④ Decoder 的 cross-attention + mask | queries × pixels | 和解析度成正比 |

所以之前改 decoder 的 FFN，等於在優化運算量最小的地方。
但實際 profile 的結果更出人意料：baseline 約 90% 的時間耗在 `xdecoder_qnn.py:262-264` 那兩行 mask 運算。

**「運算量算出來在哪」和「實際硬體上慢在哪」可能完全不同**，所以優化前一定要 profile。第 07 課會詳細講。

## 5. 關鍵名詞

| 名詞 | 一句話解釋 |
|---|---|
| Backbone | 從圖片抽出多尺度特徵的網路，通常是預訓練好的 |
| FPN（Feature Pyramid Network） | 把不同解析度的特徵由上往下融合 |
| Query | 一個可學習的向量，負責找出一個物體 |
| Cross-attention | query 去「看」影像特徵（Q 來自 query，K/V 來自影像） |
| Self-attention | query 之間互相看（Q、K、V 都來自 queries） |
| Masked attention | 只允許看 mask 範圍內的位置 |
| Open vocabulary | 類別不是固定清單，用文字描述 |
| Grounding | 把文字描述對應到圖中的區域 |
| mIoU | 預測 mask 和正確 mask 的交集 / 聯集，再對所有樣本取平均 |

## 6. 作業

1. `post_processing` 最後用 `mask_pred_results >= 0` 二值化，評分規則卻寫 `sigmoid() > 0.5`。兩者有什麼關係？為什麼部署時要這樣寫？
2. 如果把 `down_sample_size` 從 256 改成 512，`res5` 和 `mask_features` 的形狀分別會變成多少？哪些部分的運算量會變成 4 倍？
3. 用自己的話說：為什麼這個任務不能像 U-Net 那樣「每個 pixel 分類」，而需要 queries？
4. （進階）`forward()` 裡為什麼要丟掉前 101 個 queries 的結果（`[:, self.num_queries:]`）？提示：看 `xdecoder_qnn.py:228-245`，哪些 queries 在 self-attention 時看得到文字 token？

## 7. 作業解答

> 先自己想過再看。

<details>
<summary>點開看解答</summary>

1. sigmoid 是單調遞增函數，而且 `sigmoid(0) = 0.5`，所以 `sigmoid(x) > 0.5` 和 `x > 0` 等價。
   部署時直接比較 logit，可以省掉 1024×1024 次 sigmoid 運算，結果一樣。
2. 512 / 32 = 16，所以 `res5` 變成 `(1, 768, 16, 16)`；512 / 4 = 128，所以 `mask_features` 變成 `(1, 512, 128, 128)`。
   Backbone 和 Pixel Decoder 的運算量大約變成 4 倍；cross-attention 要看的 pixel 數量和 mask 的計算量也會變大；
   文字 encoder 和 decoder 的 FFN 不變。
3. U-Net 的最後一層輸出 `N類` 個通道，N 必須在訓練前就固定。這個任務的文字可以是任何句子，沒辦法事先列出所有類別。
   Query 的做法是先找出「物體」本身（不管它叫什麼），再用向量相似度和文字配對，所以類別可以是開放的。
4. 在 `grounding_eval` 模式下，原本的 101 個 queries 後面會再接上 100 個複製出來的 queries，
   self-attention 的 mask（`pad_tgt_mask`）只允許**後面這 100 個**和文字 token 互相看。
   前 101 個看不到文字，是給一般分割任務用的；只有後 100 個融合了文字資訊，所以只取它們的結果。

</details>
