# LPCVC 2025 Track 2 學習筆記

這個資料夾是一套「從看懂到改對」的學習筆記，目標是：

1. 看懂這個 repo 怎麼組織、每一層模型在做什麼
2. 理解每個設計背後的原因
3. 正確地修改 → finetune → 實際跑起來
4. 寫一份有數據支撐的優化報告，並和 2025 年的排行榜對照

## 這個 branch 是什麼

這個 branch 是從**官方原版 sample solution 的最新 commit**（`lpcvai/25LPCVC_Track2_Segmentation_Sample_Solution` 的 `43d1373`）重新開出來的，
只多了這個 `docs/learning/` 資料夾。

- 之前 final project 的修改（DyT、SwiGLU、Linear Attention、GAU 等）**都還在 `main` branch 上**，沒有被刪掉。
- 這裡是乾淨的起點：看到的每一行 code 都是官方原本的設計，不會和舊的修改混在一起。
- 這個 branch 是學習用的，**不打算 merge 回 `main`**（merge 的話會把舊的修改全部還原掉）。

> 為什麼不用 rebase？rebase 是「把你的 commits 搬到另一個起點上重放」，你的舊修改還是會跟著過來。
> 你要的是「一個完全沒有舊修改的版本」，所以正確的做法是**從原版 commit 開一個新 branch**。

## 課程地圖

| # | 主題 | 主要檔案 | 筆記 |
|---|---|---|---|
| 00 | Repo 地圖：資料夾怎麼組織、哪些用得到、哪些可以忽略 | 全部 | [00-repo-map.md](00-repo-map.md) |
| 01 | 全局地圖：一張圖 + 一句話怎麼變成 mask | `compile_and_profile/build_baseline_model.py` | [01-overview.md](01-overview.md) |
| 02 | 影像 Backbone：FocalNet，從 CNN 接到 Transformer | `modeling/vision/backbone/focal_dw.py` | 待寫 |
| 03 | Transformer 基礎 + Pixel Decoder（FPN + encoder） | `modeling/vision/encoder/transformer_encoder_fpn.py` | 待寫 |
| 04 | 文字 Encoder（CLIP）與影像和文字的對齊 | `modeling/language/` | 待寫 |
| 05 | X-Decoder 本體：queries、masked attention、預測頭 | `modeling/interface/xdecoder_qnn.py` | 待寫 |
| 06 | 訓練：匈牙利匹配、loss、資料流程 | `modeling/modules/`、`datasets/`、`trainer/` | 待寫 |
| 07 | 部署：ONNX → QNN → AI Hub、profiling | `compile_and_profile/` | 待寫 |
| A | 實作：建立評估（本地 mIoU + AI Hub 延遲） | | 待做 |
| B | 實作：修 attn_mask 瓶頸 + 掃解析度 | | 待做 |
| C | 實作：選權重 + finetune | | 待做 |
| D | 實作：寫優化報告、和排行榜對照 | | 待做 |

## 參考：2025 年排行榜（Track 2）

| | mIoU | 延遲 | 做法重點 |
|---|---|---|---|
| Baseline | 0.46 | ~863 ms | 官方 sample solution，內部解析度 256 |
| 第 1 名 SICer | 0.532 | 515.8 ms | 架構幾乎不變；解析度 ~590；LR 1e-6、≤20 epochs finetune |
| 第 2 名 Sailor Moon | 0.527 | 882.1 ms | profile 找到 attn_mask 瓶頸並修掉；解析度 800；改用官方 X-Decoder 權重 |
| 第 3 名 MaXinLab | 未公布 | 未公布 | 解析度 512 finetune |

注意：官方測試集不公開，比賽也已經結束，所以我們只能用公開的 sample data / RefCOCOg 估計，沒辦法得到真正的名次。
