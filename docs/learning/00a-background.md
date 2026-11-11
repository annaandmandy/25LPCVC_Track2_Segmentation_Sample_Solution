# 00a — 背景知識：為什麼要學、需要哪些基礎、這套架構過時了嗎

> 建議在讀 `00-repo-map.md` 之前先讀這份。
> 看懂這份，你就會知道這個 repo 裡每個零件「為什麼是它」，而不只是「它是什麼」。

---

## Part 1：已經有 Gemini 這種大模型了，為什麼還要學這個？

### 先講結論

**很多產品直接呼叫大模型 API 就是正確答案**，而且這本身就是 AI Engineer 很重要的工作。
但有一大類問題是 API 解決不了的，而這個 repo 練的正是那一類問題需要的能力。

### API 解決不了的情境

| 限制 | 例子 | 為什麼 API 不行 |
|---|---|---|
| **延遲** | 自駕車、機器手臂、AR 眼鏡 | 網路來回就要幾百 ms，這些場景需要幾十 ms 內回應 |
| **連線** | 無人機、工廠產線、偏遠地區的設備 | 沒有網路，或網路不穩 |
| **成本（量大時）** | 一支 30 fps 的監視器 | 每小時 108,000 張畫面。單張很便宜，乘上數量和攝影機數就不便宜了 |
| **隱私與法規** | 醫療影像、家用攝影機、手機相簿 | 資料不能或不該離開裝置 |
| **精確度和可控性** | 工業瑕疵檢測、醫療分割 | 通用模型擅長「描述」，但像素級的 mask 精確度和結果的穩定性通常不如專用模型；而且你沒辦法針對自己的資料 finetune 它 |

這也是 LPCVC 這個比賽存在的原因：**在低功耗裝置上，用有限的時間跑出夠準的結果**。

### 大模型公司自己也需要這種能力

- 大模型公司都在把模型「縮小」放進手機（例如 Google 的 Gemini Nano、Apple 的裝置端模型）。
  做法正是：蒸餾（distillation）、量化（quantization）、找出硬體瓶頸、針對 NPU 改寫運算。
- 這個 repo 的實作 B（profile → 找瓶頸 → 改寫）、實作 C（finetune），就是同一類工作的縮小版。

### 「AI Engineer」其實有兩種

| | LLM 應用工程師 | ML / 邊緣部署工程師 |
|---|---|---|
| 主要工作 | 串 API、RAG、agent、prompt、評估 | 訓練、finetune、壓縮、部署到特定硬體 |
| 核心技能 | 系統設計、評估、資料處理 | 看懂模型架構和研究 code、profiling、量化 |
| 這個 repo 練到的 | 評估方法、讀大型 codebase | **幾乎全部** |

兩種都需要的共同基礎：**Transformer、attention、embedding、對比學習**。
這個 repo 剛好全部都有。學懂這裡的 attention 和 CLIP，回頭看 LLM 的架構會快很多。

**建議**：不用二選一。這個專案的價值不是「學會 X-Decoder」，而是那些可以帶走的能力。

---

## Part 2：需要知道的基礎知識

### 2.1 一張演進地圖

```
影像分類 backbone（從圖片抽特徵）
  CNN ──► ResNet (2015) ──────────────► ConvNeXt (2022)
            │ 殘差連接、4 個 stage
            ▼
  Transformer (2017, NLP) ──► ViT (2020) ──► Swin (2021) / FocalNet (2022)
                                │ 圖片切成 patch     │ 加回「階層」和「局部」，像 CNN
                                ▼                  ▼
                          CLIP (2021)         本 repo 的 backbone = FocalNet-T

任務頭（在 backbone 後面接什麼）
  分類：一個全連接層
  偵測：Faster R-CNN (2015) ──► YOLO (2016~今) ──► DETR (2020, query + 匈牙利匹配)
  分割：FCN / U-Net (2015) ──► Mask R-CNN (2017) ──► MaskFormer / Mask2Former (2021-22, query-based)

把文字接進來（開放詞彙）
  CLIP (2021) ──► X-Decoder (2023) / Grounding DINO (2023) ──► SAM 3 (2025, 文字提示分割)
                    ↑ 本 repo                                  ↑ 2026 年的主流方向之一
  SAM (2023) ──► SAM 2 (2024, 影片) ──┘
```

### 2.2 Backbone：模型的「眼睛」

**是什麼**：從圖片抽出特徵的網路，通常先在大資料集（ImageNet 等）上預訓練好，再拿來接各種任務。
你寫過的 CNN 前面那幾層 conv 就是一個小 backbone。

**重要的設計選擇**

| 類型 | 代表 | 特點 | 適合 |
|---|---|---|---|
| CNN | ResNet、ConvNeXt、MobileNet | 局部運算、有階層（多尺度）、硬體最友善 | 邊緣裝置、偵測、分割 |
| 純 Transformer | ViT | 每個 patch 都看整張圖（全域）、單一尺度、運算量和解析度平方成正比 | 大模型、有大量資料時 |
| 混合 / 階層式 Transformer | Swin、FocalNet、EfficientViT、MobileViT | 保留 CNN 的階層和局部性，加上 Transformer 的長距離關係 | 分割、偵測、邊緣部署 |

**這個 repo 為什麼選 FocalNet-T**

1. **有階層**：輸出 1/4、1/8、1/16、1/32 四種尺度，後面的 FPN 和多尺度 cross-attention 需要這個。純 ViT 只有一種尺度，要另外處理。
2. **運算對 NPU 友善**：主要是 depthwise conv 和逐元素運算，QNN 都支援，也容易量化。
3. **小**：Tiny 版本約 2 千多萬參數，適合手機。
4. **有現成的預訓練權重**：X-Decoder 官方就有 FocalNet-T 版本，不用自己從頭訓練。

> 你之前試過 MobileViT、EfficientFormer 的方向是對的（找更適合邊緣裝置的 backbone），
> 但換 backbone 等於丟掉預訓練好的整套權重，需要大量重新訓練。第 3 名就是試了 DaViT、ViT 之後放棄。

### 2.3 Transformer 與 Attention：所有現代模型的共同語言

**一句話**：attention 讓每個元素根據「和其他元素的相關程度」，從其他元素收集資訊。

- 每個元素產生三個向量：**Q**（我要找什麼）、**K**（我有什麼）、**V**（我的內容）。
- `Q·K` 算相關程度，softmax 變成權重，再對 V 加權平均。
- **Self-attention**：Q、K、V 來自同一組東西（例如 queries 之間互相看）。
- **Cross-attention**：Q 來自一組，K、V 來自另一組（例如 queries 去看影像特徵）。
- **成本**：N 個元素就要算 N² 個配對。這就是為什麼高解析度很貴，也是 linear attention 想解決的問題。

LLM（GPT、Gemini、Claude）的核心也是同一個東西，只是元素從「影像 patch / query」換成「文字 token」。

### 2.4 CLIP：讓圖和文字可以比較

**怎麼訓練**：收集 4 億組（圖片, 說明文字）。圖片 encoder 和文字 encoder 各自把輸入變成向量，
訓練目標是「配對的圖文向量要靠近，不配對的要遠離」。這叫**對比學習（contrastive learning）**。

**得到什麼**：一個圖和文字共用的向量空間。任何文字都能變成向量，和圖片特徵算相似度。
這就是**開放詞彙**的基礎：不需要事先定義類別清單。

**這個 repo 裡**：文字 encoder 是 CLIP 式的 12 層 Transformer，tokenizer 用 CLIP 的
（比賽規定輸入格式是 CLIP tokenizer 的輸出）。最後一步「候選向量和文字向量算 cosine similarity」就是 CLIP 的做法。

### 2.5 YOLO 與 DETR：偵測的兩種思路（理解 decoder 必備）

| | YOLO | DETR |
|---|---|---|
| 怎麼產生預測 | 在特徵圖的每個格子上預測框 | 用 N 個 learnable **queries**，每個 query 預測一個物體 |
| 重複預測怎麼處理 | 後處理 NMS（非極大值抑制）去掉重疊的框 | 訓練時用**匈牙利匹配**，一個物體只配一個 query，不需要 NMS |
| 優點 | 非常快、工程成熟，業界即時偵測的首選 | 架構簡潔、容易延伸到分割和多模態 |
| 和本 repo 的關係 | 沒有直接關係 | **X-Decoder 的 decoder 就是 DETR 這一派** |

**為什麼這個任務不用 YOLO**：YOLO 輸出的是框，不是 mask；而且類別固定。
（YOLO 也有分割版和開放詞彙版，例如 YOLO-World，但精細的 mask 和複雜句子的理解不是它的強項。）

### 2.6 分割的演進：從「每個 pixel 分類」到「每個 query 一個 mask」

| 世代 | 代表 | 做法 | 限制 |
|---|---|---|---|
| 1 | FCN、U-Net | 每個 pixel 分成 N 類 | 類別固定；分不開同類的不同個體 |
| 2 | Mask R-CNN | 先偵測框，再在框裡畫 mask | 兩階段、流程複雜 |
| 3 | MaskFormer、Mask2Former | N 個 queries，每個輸出一個 mask + 一個類別 | **本 repo 的 decoder 就是這一代** |
| 4 | X-Decoder、SEEM | 第 3 代 + CLIP 文字空間 → 開放詞彙、文字提示 | **本 repo** |
| 5 | SAM 系列 | 超大量資料訓練的「分割基礎模型」，用點、框、文字提示 | 大、需要壓縮才能上手機 |

**Mask2Former 的關鍵發明 = masked attention**：每層 query 只看上一層預測的 mask 範圍，
讓 query 專注在自己的物體上、收斂更快。第 05 課會細看，這也是實作 B 要修的效能瓶頸所在。

### 2.7 把它們組起來：這個 repo 的每個零件從哪來

| 零件 | 來源概念 | 為什麼選它 |
|---|---|---|
| FocalNet-T backbone | 階層式 Transformer（2.2） | 多尺度、NPU 友善、小、有預訓練權重 |
| FPN + Transformer encoder（pixel decoder） | FPN + Transformer（2.3） | 融合多尺度，在最小那層加全域資訊 |
| 12 層文字 encoder + CLIP tokenizer | CLIP（2.4） | 把任何句子變成和影像可比較的向量 |
| 101 queries + masked cross-attention decoder | DETR + Mask2Former（2.5、2.6） | 每個 query 產生一個候選 mask，可以和文字配對 |
| 訓練時的匈牙利匹配 | DETR（2.5） | 一個物體只配一個 query |
| 推論時用 cosine similarity 選 top-1 | CLIP（2.4） | 開放詞彙：選和文字最像的候選 |

---

## Part 3：2026 年了，這套架構過時了嗎？

**模型本身：作為最先進的方法，已經被超越了。**

- X-Decoder 是 2023 年的模型。之後有 SAM 2、SAM 3（Meta 在 2025 年底推出，直接用文字概念提示做分割），
  以及各種「VLM + 分割頭」的方法（例如 LISA 這類，讓大型語言模型輸出分割結果）。
- 大型 VLM（包括 Gemini）也能直接輸出分割結果。
- 如果今天要做一個「準確度最高」的文字提示分割產品，不會從 X-Decoder 開始。

**架構的組成方式：沒有被淘汰，反而是現在的標準語言。**

| 這個 repo 裡的概念 | 2026 年仍在哪裡使用 |
|---|---|
| 階層式 backbone + FPN 多尺度特徵 | 幾乎所有偵測和分割模型 |
| Query-based decoder（DETR / Mask2Former） | SAM 系列的 mask decoder、SAM 3 的偵測器、RT-DETR 等即時偵測器 |
| 圖文對比學習的共同向量空間 | CLIP 的後繼者（如 SigLIP）是許多 VLM 的影像 encoder |
| Cross-attention 融合兩種資訊 | 多模態模型的核心 |
| 匈牙利匹配 | DETR 類模型的標準訓練方式 |

**這個專案練到的工程能力：比以前更重要。**

模型越來越大、越來越多公司要把它放進裝置裡。profiling、找硬體瓶頸、量化、
確認運算有沒有被 NPU 支援、針對部署解析度 finetune——這些能力不會因為某個模型過時而失效。

### 一個比喻

學 X-Decoder 像是學一台 2023 年出廠的車：它不是最新款，
但引擎、變速箱、底盤的原理和今天的車是一樣的，而且**修車和改裝的技術完全通用**。
先把一台結構完整、資料齊全的車拆懂，再去看 SAM 3 這類新模型，會發現大部分零件你都認得。

### 學完這個 repo 之後的延伸方向

1. **SAM 系列**（SAM → SAM 2 → SAM 3）：看它的 mask decoder，會發現和第 05 課很像。
2. **邊緣版分割模型**（MobileSAM、EfficientSAM、EdgeSAM 等）：看別人怎麼把大模型縮小。
3. **VLM 的影像端**：看 CLIP / SigLIP 怎麼接進大型語言模型。

---

## 自我檢查

- [ ] 我能舉出兩個「不能用 API、必須在裝置上跑模型」的情境，並說出原因嗎？
- [ ] 我能用自己的話解釋 self-attention 和 cross-attention 的差別嗎？
- [ ] 為什麼 CLIP 是「開放詞彙」的關鍵？
- [ ] DETR 和 YOLO 最大的差別是什麼？X-Decoder 屬於哪一派？
- [ ] 為什麼說 X-Decoder「模型過時了，但架構沒有」？
