# lbm-nca-thesis

**神經細胞自動機運用於建築設計中 CFD 運算**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

English: [README.md](README.md)

一個神經細胞自動機，預測二維都市建築配置周圍的穩態平均風場；同時附上產生其訓練
資料的完整 lattice-Boltzmann 管線，以及讓訓練好的模型在 Rhino/Grasshopper 裡即時
運作的 WebSocket 橋接。

這個代理模型是**反覆施加於場上的局部更新規則**，不是一次到位的影像對映。給定某個
配置的收斂流場，它把該場精修成鄰近配置的收斂流場——設計者調整量體時，付出的代價
對應於「這次編輯」，而不是重跑一次完整模擬。

本 repo 為國立陽明交通大學碩士論文的附屬程式碼，含程式與訓練權重。**資料集不隨
repo 發佈**，因為可以用這裡的程式完整重新生成（見
[docs/DATA_GENERATION.md](docs/DATA_GENERATION.md)）。

<p align="center">
  <img src="docs/assets/demo-grasshopper-live.png" alt="從 Grasshopper 畫布即時驅動風場推論，於獨立 Taichi GGUI 視窗渲染" width="850">
  <br>
  <em>即時推論：右側 Grasshopper 上編輯的幾何，即時驅動左側 Taichi 渲染的風場，
  維持互動式更新率。</em>
</p>

---

## 管線總覽

```
morphospace          lbm                       nca                    bridge
──────────────       ───────────────────       ─────────────────      ──────────────
建築配置           →  D2Q9 MRT-LES 求解器   →   NCA 代理模型        →   Rhino /
+ G_vec (14D)        HDF5 → Zarr               訓練 + 評測            Grasshopper
```

| 套件 | 職責 |
| --- | --- |
| `lbmnca.morphospace` | 從 PCA 設計空間抽樣建築配置、光柵化成 mask，並計算 14 維幾何描述子 `G_vec` |
| `lbmnca.lbm` | GPU lattice-Boltzmann 求解器（D2Q9 網格、MRT 碰撞、Smagorinsky LES），把每個配置模擬到收斂、寫出 HDF5 軌跡，再轉成 Zarr 訓練格式 |
| `lbmnca.nca` | 神經細胞自動機代理模型：感知、更新規則、損失項、訓練迴圈、評測指標 |
| `lbmnca.bridge` | WebSocket 伺服器 + Taichi GGUI 渲染器，讓 Grasshopper 或瀏覽器畫出的配置即時求解並顯示 |

### 為何用細胞自動機，而非一次到位的網路

<p align="center">
  <img src="docs/assets/concept-iterative-refinement.png" alt="直接映射一次輸出、無中間步；序列迭代把場逐步推進，透過反覆的局部更新" width="700">
</p>

多數學習型 CFD 代理模型是直接映射：輸入幾何，直接輸出收斂場，中間沒有任何狀態。
這個代理模型改成像 lattice-Boltzmann 求解器本身一樣，把局部更新規則反覆施加於
場上。因為模型本身就知道怎麼把場收斂，所以可以把某個配置的收斂流場交給它，讓它
把場精修成**鄰近配置**的收斂流場——也就是暖啟動——而不必每次編輯都從零收斂。

<p align="center">
  <img src="docs/assets/method-landscape.png" alt="以暖啟動能力與取得可用結果的壁鐘時間，比較本方法與 DeepONet、FNO、CNN、CFDNet、Super-fidelity、CoNFiLD" width="650">
</p>

純前饋式代理模型（CNN、FNO、DeepONet）速度快，但永遠只能從幾何冷啟動。包一層
古典求解器的混合式方法（CFDNet、Super-fidelity）雖能暖啟動，但每一步都要付出
求解器呼叫的代價。這個代理模型正好落在「場編輯」這個使用情境真正需要的位置：
暖啟動、且互動式（次秒級）延遲。

## 成果

<p align="center">
  <img src="docs/assets/result-validation-velocity.png" alt="三個配置在訓練域邊界之前、邊界時、邊界後的速度量值真值、模型預測與誤差場" width="850">
</p>

真值（上）、模型預測（中）與誤差場（下），對象為速度量值，評測配置分別取自訓練
域內、邊界處、邊界外——誤差集中在建築邊緣，且不會在訓練範圍外暴衝。

<p align="center">
  <img src="docs/assets/demo-warm-start-editing.png" alt="連續四次建築配置編輯，每一次都從前一次編輯收斂後的流場接續，而非從頭重解" width="850">
</p>

連續四次設計編輯，每次都從前一次編輯收斂後的場暖啟動：代理模型只需精修編輯帶來
的局部擾動，不必重新求解整個域。

## 發表著作

- CAADRIA 2025 — 呂博軒、侯君昊，〈應用於建築空間動態通風的神經細胞自動機模型〉，
  [caadria2025_858](https://papers.cumincad.org/cgi-bin/works/paper/caadria2025_858)
- CAADRIA 2026 — 呂博軒、侯君昊，〈模組化神經細胞自動機（M-NCA）：面向建築設計的
  物理資訊即時動態模擬框架〉，
  [caadria2026_232](https://papers.cumincad.org/cgi-bin/works/paper/caadria2026_232)
- eCAADe 2026 — 呂博軒、侯君昊，〈模擬隨設計行動而動：面向風環境建築設計的動態
  暖啟動 GC-NCA 框架〉，
  [ecaade2026_147](https://papers.cumincad.org/cgi-bin/works/paper/ecaade2026_147)

### 作者

呂博軒¹、侯君昊²
國立陽明交通大學建築研究所
¹apc582nntscott@arch.nycu.edu.tw，²jhou@arch.nycu.edu.tw

---

# 安裝與使用

## 硬體需求

推論與訓練都需要 NVIDIA GPU。求解器尤其吃重：跑在 2560 × 1440 的網格上，預設保留
八成顯存。**這不是能在 CPU 或免費雲端層跑起來的工作量。**

| | 最低 | 產生論文數字時所用 |
| --- | --- | --- |
| GPU | 支援 CUDA，8 GB VRAM | NVIDIA RTX 級，24 GB VRAM |
| Python | 3.10 | 3.13 |
| 作業系統 | Windows 或 Linux | Windows 11 |

`bridge` 另需支援 Vulkan 的顯卡驅動（Taichi GGUI 視窗），要用 Grasshopper 元件則需
Rhino 7 或 8。

## 安裝

```bash
git clone https://github.com/scott-lbx/lbm-nca-thesis
cd lbm-nca-thesis
python -m venv .venv
```

PyTorch 必須從它自己的 index 裝，因為正確的 wheel 取決於你的 CUDA 版本。把 `cu126`
換成你的驅動支援的版本：

```bash
# Windows
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\pip install -e ".[all]"

# Linux / macOS
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.venv/bin/pip install -e ".[all]"
```

只需要部分管線時可裝窄一點：`.[morphospace]`、`.[solver]`、`.[train]`、`.[bridge]`。

訓練權重以 [Git LFS](https://git-lfs.com) 儲存。clone 後若 `checkpoints/main_v4.pth`
是一個很小的文字檔，執行 `git lfs pull`。

### 資料與權重放哪

兩個環境變數決定所有路徑，程式碼裡沒有任何綁定特定機器的路徑。

| 變數 | 預設 | 內容 |
| --- | --- | --- |
| `LBMNCA_DATA_ROOT` | `<repo>/data` | 生成的 mask、per-case config、HDF5 軌跡、Zarr 資料集 |
| `LBMNCA_CHECKPOINT_ROOT` | `<repo>/checkpoints` | 模型權重與正規化統計 |

---

## 快速上手

### 1. 跑訓練好的模型

checkpoint loader 只依架構設定建網路，**完全不開訓練資料集**，所以沒有資料集也能跑：

```bash
python scripts/run_inference.py --checkpoint checkpoints/main_v4.pth --config configs/nca/main.yaml
```

### 2. 在瀏覽器或 Grasshopper 裡即時推論

```bash
python -m lbmnca.bridge.main
```

流場畫在**獨立的 Taichi GGUI 視窗**；瀏覽器頁面與 Grasshopper 元件是控制面板，負責
送出幾何、收回 ack。一鍵啟動腳本與完整的 Grasshopper 元件設定步驟見
[demo/README.md](demo/README.md)。

### 3. 重新生成資料集

六個步驟，從抽樣配置母體、模擬，到 Zarr 訓練格式。這是最花時間的部分，需要 GPU 跑
數天，所以文件另附一組**小規模驗證設定**，讓你先花數小時確認環境正確，再投入完整
重建。見 [docs/DATA_GENERATION.md](docs/DATA_GENERATION.md)。

### 4. 訓練

`LBMNCA_DATA_ROOT` 裡有重生成的資料集之後：

```bash
python scripts/train.py --config configs/nca/main.yaml
```

---

## 目錄結構

```
configs/        訓練設定（canonical + 消融臂）
checkpoints/    訓練權重與凍結正規化統計
demo/           瀏覽器頁面、啟動腳本、Grasshopper 元件
docs/           架構、資料格式、重生成、重現
examples/       不需資料集的小型範例
scripts/        命令列入口，依管線階段編號
src/lbmnca/     上表四個套件
tests/          測試；`-m "not gpu and not dataset"` 純 CPU 可跑
```

## 文件

| 文件 | 內容 |
| --- | --- |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 條件向量、正規化、以及**實際啟用**的更新規則 |
| [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) | HDF5 軌跡結構與 node/edge Zarr schema |
| [docs/DATA_GENERATION.md](docs/DATA_GENERATION.md) | 資料集重生成與算力成本 |
| [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | 重現論文數字 |
| [docs/GVEC_SPEC.md](docs/GVEC_SPEC.md) | 14 個幾何維度各自實際量到什麼 |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | 在 Rhino/Grasshopper 內部署 |

---

## 已知限制

以下是釋出模型的實際性質，寫在這裡以免結果被過度解讀。

- **只做平均流場。** 代理模型預測的是時均場。脈動模組在設定 schema 中有定義，但沒有
  釋出的 checkpoint，不在範圍內。
- **二維。** 求解器是 D2Q9。真實行人風舒適度由垂直方向流動結構主導，此處完全沒有。
- **架構名稱誇大了實際執行的內容。** canonical 架構註冊名為
  `operator_gated_streamcollide`，但在釋出設定下，operator gating 與 semi-Lagrangian
  transport **兩者都被跳過**（`transport.skip: true`）。實際路徑是：多受體感知 →
  帶流函數動量頭的 Markov cell 更新 → coarse/fine 多尺度疊加。
  `docs/ARCHITECTURE.md` 描述的是真正會執行的路徑。
- **無散度保證比字面窄。** 由流函數的 curl 導出動量，保證的是每一次**增量**在流體
  內部、在同一組有限差分核下無散度。它**不移除**初始條件既有的散度，在固/流界面
  也不成立。
- **canonical 臂同時是消融臂。** `main_v4` 是資料條件析因的 A✓B✓ 格位，設定與
  `abl_use_v4` 相同。兩者之差量到的是訓練隨機性，不是效應。
- **舒適度圖層未校準。** 格子速度到物理速度的換算係數是佔位值，即時 demo 的
  Lawson/Beaufort 上色**不構成物理風舒適度評估**。渲染器會在畫面上明示這件事。
- **求解器限制**（LES 模型、守恆、雷諾數口徑）記載於 `docs/ARCHITECTURE.md`；它們會
  移動代理模型所對照的 ground truth。

## 引用

若這份程式碼或重生成管線對你有用，請引用論文。機器可讀的 metadata 在
[CITATION.cff](CITATION.cff)。

## 授權

MIT，見 [LICENSE](LICENSE)。
