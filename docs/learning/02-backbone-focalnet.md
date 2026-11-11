# 02 — 影像 Backbone：FocalNet-T

> 檔案：`modeling/vision/backbone/focal_dw.py`（約 790 行，真正核心只有前 300 行）
> Config：`configs/xdecoder/focalt_unicl_lang_lpcvc25.yaml` 的 `MODEL.BACKBONE`
> 示範腳本：`docs/learning/scripts/02_inspect_backbone.py`（用隨機權重實際跑一次，印出每一層的形狀）

這一課的目標：
1. 看懂 FocalNet 的整體骨架，並發現它和你熟悉的 ResNet 幾乎一樣
2. 看懂一個 block 裡面在做什麼，以及「token mixer + channel mixer」這個所有現代模型共用的設計
3. 看懂 Focal Modulation：它如何用 CNN 的零件做到 attention 的效果
4. 把這些設計連結到「手機 NPU 上跑得快不快」

---

## 1. 整體骨架：其實就是 ResNet 的形狀

### 程式碼裡的四個 class，由外到內

```
get_focal_backbone(cfg)           第 778 行：被 registry 註冊的入口（NAME: focal_dw）
 └─ D2FocalNet                    第 695 行：讀 config、包成 detectron2 的 Backbone 介面
     └─ FocalNet                  第 434 行：真正的網路
         ├─ patch_embed           PatchEmbed：stem，把圖片變成特徵
         ├─ layers[0..3]          4 個 BasicLayer = 4 個 stage
         │   ├─ blocks            每個 stage 有 depth 個 FocalModulationBlock
         │   │   └─ modulation    FocalModulation（核心）
         │   └─ downsample        PatchEmbed：解析度減半、通道加倍
         └─ norm0..norm3          每個輸出尺度各一個 LayerNorm
```

### 和 ResNet-50 對照

| | ResNet-50 | FocalNet-T（本 repo） |
|---|---|---|
| Stem | Conv 7×7 stride 2 + MaxPool stride 2 → stride 4 | Conv 7×7 stride 4（`PatchEmbed`，`is_stem=True`） |
| Stage 數 | 4 | 4 |
| 每個 stage 的 block 數 | [3, 4, 6, 3] | **[2, 2, 6, 2]**（config 的 `DEPTHS`） |
| 通道數 | 256 → 512 → 1024 → 2048 | **96 → 192 → 384 → 768**（`EMBED_DIM: 96`，每個 stage 加倍） |
| Stage 之間怎麼降解析度 | stride-2 conv | Conv 3×3 stride 2（`PatchEmbed`，`is_stem=False`） |
| Block 裡怎麼混合空間資訊 | 3×3 conv | **Focal Modulation**（下面細講） |
| 輸出 | 4 個尺度（stride 4/8/16/32） | 4 個尺度 `res2`～`res5`（stride 4/8/16/32） |

**結論：骨架完全是 CNN 的骨架，差別只在 block 內部。**
這也是為什麼它叫「階層式（hierarchical）」：有多個尺度，和 ViT 那種從頭到尾都同一個解析度不同。

### 為什麼 block 數是 [2, 2, 6, 2]，中間特別多？

- Stage 1（1/4 解析度）：位置最多，每多一個 block 都很貴 → 少放。
- Stage 3（1/16 解析度）：位置數已經少了 16 倍，而語意資訊開始豐富 → **把大部分運算放在這裡**，CP 值最高。
- Stage 4（1/32）：位置很少，但通道數最大，參數量很重 → 少放。

ResNet-50 的 [3, 4, 6, 3]、Swin-T 的 [2, 2, 6, 2] 都是同樣的邏輯。

---

## 2. 一個 Block 在做什麼：`FocalModulationBlock`（第 118–204 行）

### 所有現代模型 block 的共同結構

不管是 ViT、Swin、ConvNeXt、MLP-Mixer 還是 FocalNet，一個 block 幾乎都由兩件事組成：

| 部分 | 作用 | 在 ViT 裡 | 在 FocalNet 裡 |
|---|---|---|---|
| **Token mixer** | 讓**不同位置**之間交換資訊（空間） | Self-attention | **Focal Modulation** |
| **Channel mixer** | 在**每個位置內部**混合不同通道（特徵） | MLP（FFN） | MLP（`Mlp` class，第 24 行） |

每一部分外面都包著 **殘差連接（residual）** 和 **正規化（LayerNorm）**。
你之前改的 FFN / SwiGLU 就是 channel mixer；Linear Attention 想改的是 token mixer。

### 照著 forward（第 168 行）一步一步看

```python
# 輸入 x: (B, H*W, C)，每個位置是一個 C 維的 token
x = x + self.dw1(x)          # ① depthwise conv 3x3 + 殘差：加入局部位置資訊
shortcut = x
x = self.modulation(x)       # ② Focal Modulation：token mixer
x = shortcut + self.gamma_1 * x
x = self.norm1(x)            #    post-LN（config 的 USE_POSTLN: True）
x = x + self.dw2(x)          # ③ 再一次 depthwise conv 3x3 + 殘差
x = x + self.gamma_2 * self.mlp(x)   # ④ MLP：channel mixer（C → 4C → C）
x = self.norm2(x)
```

### 幾個細節，各自是為了什麼

- **depthwise conv（`dw1`、`dw2`）**：`groups=dim` 的 3×3 conv，每個通道只看自己的 3×3 鄰居，非常便宜。
  作用類似 Transformer 的 positional encoding：讓模型知道「誰在誰旁邊」。
- **`gamma_1`、`gamma_2`（LayerScale，`USE_LAYERSCALE: True`）**：初始值 `1e-4` 的可學習縮放。
  剛開始訓練時每個 block 幾乎等於「什麼都不做」（只剩殘差），讓很深的網路比較容易訓練穩定。
- **`DropPath`（`DROP_PATH_RATE: 0.3`）**：訓練時隨機跳過整個 block，是一種正則化；推論時不起作用。
- **形狀一直在 `(B, H*W, C)` 和 `(B, C, H, W)` 之間轉換**：
  `Linear` 和 `LayerNorm` 作用在最後一維，所以用 `(B, H*W, C)`；
  `Conv2d` 需要 `(B, C, H, W)`。程式碼裡大量的 `permute(...).contiguous()` 就是在來回轉換。
  （補充：對最後一維做 `Linear`，數學上就等於 1×1 conv。）

---

## 3. 核心：Focal Modulation（第 44–116 行）

### 先回憶 self-attention 怎麼做（2 步）

對每個位置 i：
1. **算權重**：i 的 Q 和**所有**位置的 K 做內積 → N 個權重（N = H×W）
2. **聚合**：用這 N 個權重對所有位置的 V 加權平均

問題：每個位置都要和所有位置配對，成本是 **N²**。256×256 的輸入在 stage 1 有 64×64 = 4,096 個位置，配對數就是 1,600 萬。

### Focal Modulation 把順序反過來（3 步）

1. **先聚合**：不管 query 是誰，先用 depthwise conv 在每個位置周圍收集「由近到遠」的情境資訊
2. **閘控**：每個位置自己決定要多重視近的、中的、遠的、全域的資訊
3. **調變**：把聚合好的情境，用**逐元素相乘**套到 query 上

成本只和 N 成正比，而且用到的全是 conv、乘法、平均這些便宜又對硬體友善的運算。

### 對照程式碼

```python
# __init__（第 68–87 行）
self.f = nn.Linear(dim, 2*dim + (focal_level+1))     # 一次產生 q、ctx、gates
self.h = nn.Conv2d(dim, dim, kernel_size=1)          # 調變前的 1x1 conv
for k in range(focal_level):                          # focal_level = 3
    kernel_size = focal_factor*k + focal_window       # 2*k + 3 → 3, 5, 7
    focal_layers.append(Conv2d(dim, dim, kernel_size, groups=dim) + GELU)  # depthwise

# forward（第 89–116 行）
q, ctx, gates = split(self.f(x), (C, C, focal_level+1))   # 分成三份

ctx_all = 0
for l in range(focal_level):                  # ① 由近到遠一層一層疊
    ctx = self.focal_layers[l](ctx)           #    注意：ctx 是「接續」上一層的結果
    ctx_all = ctx_all + ctx * gates[:, l:l+1] # ② 每一層乘上自己的 gate 再加總
ctx_global = GELU(ctx.mean(2).mean(3))        #    整張圖的平均 = 全域資訊
ctx_all = ctx_all + ctx_global * gates[:, focal_level:]

x_out = q * self.h(ctx_all)                   # ③ 調變：逐元素相乘
x_out = self.proj(x_out)                      #    最後再一個 Linear
```

### 「由近到遠」是怎麼來的

三層 depthwise conv 是**疊起來**的（每一層的輸入是上一層的輸出），所以看到的範圍會累加：

| 疊到第幾層 | 這層的 kernel | 每個位置能看到的範圍 |
|---|---|---|
| 1 | 3×3 | 3×3 |
| 2 | 5×5 | 7×7 |
| 3 | 7×7 | 13×13 |
| global | 全圖平均 | 整張圖 |

在 stage 3（1/16 解析度），13×13 個位置已經對應到原圖約 208×208 的範圍；再加上 global，等於什麼距離都照顧到了。

### Self-attention vs. Focal Modulation 總整理

| | Self-attention | Focal Modulation |
|---|---|---|
| 順序 | 先算 query 對每個位置的權重，再聚合 | 先聚合情境，再用 gate 和 query 調變 |
| 成本 | N²（和位置數的平方成正比） | N（和位置數成正比） |
| 用到的運算 | 大矩陣乘法、softmax、動態大小的注意力圖 | depthwise conv、逐元素乘法、平均、GELU |
| 對手機 NPU | 高解析度時很吃記憶體 | **很友善**：都是固定形狀、NPU 原生支援的運算 |
| 長距離關係 | 天生全域 | 靠多層疊加 + global 平均 |

這就是第 00a 課說「FocalNet 本質上很接近 CNN、對 NPU 友善」的具體原因。

---

## 4. 實際跑一次看看

在 repo 根目錄執行（只需要 `pip install torch timm`，不需要 GPU、detectron2 或任何權重）：

```bash
python docs/learning/scripts/02_inspect_backbone.py            # 預設 256，和 baseline 一樣
python docs/learning/scripts/02_inspect_backbone.py --size 512 # 看看解析度加倍會怎樣
```

腳本會載入 repo 裡**真正的** `focal_dw.py`，用和 config 一模一樣的設定建模型，印出：
參數量、每個 block 的輸出形狀、最後 4 個尺度、Focal Modulation 的 kernel 和感受野。

輸出結果見下方「附錄：腳本輸出」。

---

## 5. 和部署、優化的關係

| 觀察 | 對優化的意義 |
|---|---|
| 參數量大部分在 stage 3、4，但**運算量**大部分在高解析度的 stage 1、2、3 | 縮小輸入解析度主要省在前面的 stage；砍 stage 4 的 block 主要省參數 |
| 運算量大約和 H×W 成正比 | 256 → 512，backbone 的運算量約 4 倍 |
| `size_divisibility = 32`（第 775 行） | 輸入邊長最好是 32 的倍數，否則每個 stage 下採樣時會出現不整除 |
| 大量 `permute` / `contiguous` | 在 NPU 上會變成 transpose 運算，數量多時也有成本；profile 時會看到 |
| 全部是 conv / Linear / LayerNorm / GELU / 乘法 | QNN 都支援，也容易做 INT8 量化 |
| `load_weights` 會處理 kernel 大小不同的預訓練權重（第 603–616 行） | 換 `FOCAL_WINDOWS` 這類設定還能沿用部分權重；但換掉整個 backbone 就得從頭來 |

---

## 6. 作業

1. 用你自己的話說：FocalNet 的哪些部分和 ResNet 一樣？哪個部分不一樣？
2. `FocalModulation` 裡，如果把三層 depthwise conv 改成「平行」（三層都接收同一個 `ctx` 當輸入），每一層能看到的範圍會變成多少？為什麼原作者要用疊的？
3. 跑一次 `--size 512`，比較和 256 的輸出形狀。哪些數字變了？哪些沒變？為什麼參數量完全不變？
4. （進階）第 1 名用的解析度是 590。跑一次 `--size 590`，看看 `res2`～`res5` 的大小。590 不是 32 的倍數，你覺得這可能會在後面的哪個環節造成問題？

<details>
<summary>點開看解答</summary>

1. **一樣**：stem 先降到 1/4；4 個 stage、每個 stage 解析度減半、通道加倍；輸出 4 個尺度；block 裡有殘差連接。
   **不一樣**：block 內部負責「位置之間交換資訊」的部分。ResNet 用 3×3 conv，FocalNet 用 Focal Modulation
   （多層 depthwise conv 聚合 + gate + 和 query 逐元素相乘）。另外正規化用 LayerNorm 而不是 BatchNorm。
2. 平行的話，三層各自只看到 3×3、5×5、7×7，最大只有 7×7。疊起來可以用同樣的參數量看到 13×13，
   也就是用小 kernel 累積出大的感受野。這和 VGG「兩個 3×3 等於一個 5×5 的感受野」是同一個想法。
3. 變的是每一層的 H、W（全部變成 2 倍，位置數變 4 倍）；不變的是通道數和參數量。
   參數只存在 conv 的 kernel 和 Linear 的權重裡，和輸入有多大無關，同一組權重滑過更大的圖而已。
   這也是為什麼可以「用 256 訓練的模型直接拿去跑 512」，雖然準確度不一定好（第 1 名和第 2 名的經驗）。
4. 590 經過 stem（stride 4）變 148，再來 74、37、19，最後不是剛好的 2 倍關係（37 → 19 是向上取整）。
   後面的 pixel decoder（FPN）要把小的特徵圖上採樣後和大的相加。如果是「固定放大 2 倍」，19×2 = 38 ≠ 37，就會報錯。
   這個 repo 的 FPN 是 `F.interpolate(y, size=cur_fpn.shape[-2:])`（`transformer_encoder_fpn.py:295`），
   **直接對齊到目標大小**，所以不會報錯。
   但代價是：19 → 37 不是整數倍放大，特徵在空間上會有一點點錯位，可能稍微影響準確度。
   這也說明了為什麼第 1 名的 590 是「掃過一輪之後實測出來的」，而不是從理論推導出來的。

</details>
