# NCA demo — 操作說明

本目錄含**全部使用者需要用到的檔案**。照著這份文件操作，不需要進 `src/`
（那裡是原始碼，開發者才需要進去）。

---

## 這裡有什麼

| 檔案                                 | 用途                                                                                          |
| ------------------------------------ | --------------------------------------------------------------------------------------------- |
| `index.html`                         | demo 頁面本體。可直接雙擊開，或透過 `run_demo.bat` 自動開。                                   |
| `run_demo.bat`                       | **一鍵全自動啟動（網頁用）**：開推論服務 + http.server 兩個視窗、自動開瀏覽器到正式頁。       |
| `run_gh_backend.bat`                 | **一鍵啟動（Grasshopper 專用）**：只開推論服務一個視窗，不開 http.server、不開瀏覽器。        |
| `launcher.bat`                       | 啟動「常駐橋接服務」（監聽 8090）。配合直接雙擊 `index.html` 用（見下，網頁用）。              |
| `install_autostart.ps1`              | 一次性設定：讓 `launcher.bat` 開機自動啟動，之後永久不用手動起任何東西（網頁用）。             |
| `demo.config.json`                   | 使用者可調 config（風力選項、viz 清單、序列腳本…）。一般不用改。                              |
| `gh/analyze_component_standalone.py` | **Grasshopper 元件腳本**。貼進 GH Python 元件即可用 GH 送幾何（見下方「Grasshopper 操作」）。 |

流場**不會顯示在網頁上**。網頁只是控制面板；流場畫在一個獨立彈出的 **Taichi GGUI 視窗**
（640×360），操作時兩邊都要看。

---

## 啟動方式（兩選一）

### 方式 A：一鍵全自動（最簡單，推薦新手）

雙擊 **`run_demo.bat`**。

會自動：開推論服務（`python -m lbmnca.bridge.main`）獨立視窗 → 開 `http.server` 獨立視窗 →
等 8 秒 → 開瀏覽器到 demo 正式頁。等主控台視窗印出模型載入完成（見下方「推論服務視窗該看到
什麼」），按網頁上的 `Connect` 即可開始。

每次要用都要重新雙擊一次這個檔案。

### 方式 B：先設好，之後只要開 html 點按鈕（推薦常用）

1. **只需做一次**：對 `install_autostart.ps1` 按右鍵 → 「使用 PowerShell 執行」；或開
   PowerShell 貼（`<repo>` 換成本 repo 實際路徑）：

   ```
   powershell -ExecutionPolicy Bypass -File "<repo>\demo\install_autostart.ps1"
   ```

   （雙擊 `.ps1` 檔本身不會執行，Windows 預設會拿去開記事本——這是正常現象，不是壞掉，
   要用上面方式跑。）

   這會在「開機啟動」資料夾建一個捷徑，之後每次登入 Windows 都會自動啟動 launcher（監聽
   8090）。若不想等下次登入才生效，設定完後可以先手動雙擊一次 **`launcher.bat`**。

2. 之後只要**雙擊 `index.html`**，會看到「🚀 NCA demo 啟動器」面板（不是紅框錯誤畫面）。
3. 面板上有個 **WS port** 輸入框，預設 `8765`（推論服務實際監聽的 port）。一般不用改；
   只有跟別的服務衝突時才需要換一個數字。改過的值會記住（存 localStorage），下次開頁面
   直接帶入上次填的值。
4. 按 **「🚀 啟動 demo 後端」**：面板會呼叫 launcher 起推論服務 + http.server（各自彈出獨立
   視窗）+ GGUI 視窗，狀態列輪詢直到就緒，然後自動跳轉到正式頁（網址會帶
   `?ws_port=<你填的數字>`，正式頁的 Connect 欄位會自動填成同一個 port——這條鏈路上
   只有這一個地方需要設定，其他地方都跟著它走）。

**若跳出「main.py 已在 port X 執行中」**：代表推論服務已經用另一個 port 跑著。要換 port
必須先關掉那個視窗——launcher **不會**自動幫你關掉正在跑的東西（避免誤殺你還在
用的 session），關掉舊視窗後再按一次啟動鈕即可。

移除自動啟動：到 `Win+R` 輸入 `shell:startup`，刪除裡面的 `NCA_launcher.lnk` 即可，
不影響其他任何東西。

---

## 推論服務視窗該看到什麼

正常啟動 log 依序長這樣（一切正常，無需理會其他雜訊行）：

```
[Taichi] version 1.7.4, ...
[Taichi] Starting on arch=cuda
[main] GPUFields OK (360x640)
[tes_adapter] loaded ckpt step=8000 (main_v4.pth), arch=operator_gated_streamcollide state_dim=21
[tes_adapter] loaded ckpt step=8000 (base_modern.pth), arch=baseline_unet_modern state_dim=13
[main] backend = decompose
[main] WS server ws://localhost:8765
[main] starting GGUI...
```

`loaded ckpt` 那兩行代表兩個模型都正確載入（mean + IC 生成器）。若卡在這之前很久沒動，或印出
`Traceback`，代表出問題了，不要繼續往下操作。

---

## 頁面操作

| 動作           | 怎麼做                                                                         |
| -------------- | ------------------------------------------------------------------------------ |
| 連線           | 按 **Connect**，圓點紅轉綠。若還是紅的，推論服務可能還在載模型（第一次啟動約需 20–40 秒），等一下再按一次。 |
| 畫建物         | 畫布空白處**拖曳**（按下→拖→放開），出現深色矩形。                             |
| 移動建物       | 在建物上拖曳。                                                                 |
| 刪除建物       | 對建物**點右鍵**。                                                             |
| 送出（冷啟）   | **Send (fresh 冷啟)**：從頭算，適合第一次畫完 / 大幅改動後。                   |
| 送出（溫啟）   | **Send (warm 溫啟)**：從目前流場狀態接續，適合小幅調整（移動/加減一棟）。      |
| 切換顯示模式   | `velocity` `vorticity` `pressure` `streamline` `comfort` `ibfv` 六顆按鈕，對應 GGUI 視窗畫面內容。 |
| 固定序列播放   | 下拉選 `basic`（7 個 event，約 32 秒）或 `d2-1`（11 個 event，約 2.5 分鐘），按 **▶ Play** 自動播放，全程盯著 GGUI 視窗看反應。 |

每次操作都去看 **GGUI 視窗**（不是網頁）的畫面變化；網頁 log 區只顯示送出/收到的訊息文字。

---

## Grasshopper 操作

除了網頁，**Rhino/Grasshopper 也能直接送幾何**，走同一個 WebSocket server（同一個
推論服務、同一個 8765 port）。網頁跟 GH 可以**同時連線、同時送**，不會互相干擾——server
會依 `buildings` 的資料格式自動判斷是哪一邊送的（網頁＝矩形 sim px 座標，GH＝任意多邊形
world 座標），兩邊各自處理、各自收 ack。唯一共用的是同一個 Taichi GGUI 視窗——兩邊都看
同一個流場畫面，後送的一方會覆蓋畫面。

### 設定步驟

1. 開 Rhino，在 Grasshopper 畫布上放一個 **Python 元件**（GH1 用 GhPython，GH2/Rhino 8
   用 Script Editor 皆可）。
2. 把 `gh/analyze_component_standalone.py`（本目錄下）**全文複製貼進**元件程式碼區。
3. 依下表建立元件的 Input／Output（名稱需與程式碼一致，大小寫敏感）：

   | Input        | 型別  | Access | 說明                                                                                                               |
   | ------------ | ----- | ------ | ------------------------------------------------------------------------------------------------------------------ |
   | `run`        | bool  | Item   | Button/Toggle，`True` 才送出                                                                                       |
   | `boundary`   | Curve | Item   | 分析域外框（`Rectangle3d` 或封閉曲線）；**矩形最長邊方向＝風向**                                                   |
   | `buildings`  | Curve | List   | 建物封閉外框曲線，可多棟（少於 3 點的會被丟棄）                                                                    |
   | `wind_level` | int   | Item   | 1=微風 2=中風 3=強風；預設 2                                                                                       |
   | `mode`       | str   | Item   | `"fresh"`（冷啟重算）／`"warm"`（沿用前次流場接續，`"adapt"` 同義）；預設 `fresh`                                  |
   | `viz_mode`   | str   | Item   | 可選：`velocity\|vorticity\|pressure\|streamline\|comfort\|ibfv`，留空＝不切換                                     |
   | `auto_stop`  | bool  | Item   | 可選，不接＝沿用 `config.yaml` 現行設定。`True`＝判準宣告收斂即凍結；`False`＝持續推進                              |
   | `url`        | str   | Item   | 可選；預設 `"ws://localhost:8765"`，跟網頁預設 port 相同不用改                                                     |

   | Output    | 說明                                                                                            |
   | --------- | ----------------------------------------------------------------------------------------------- |
   | `status`  | server 回應／錯誤訊息（流場**不會**回傳到 GH，只有這行文字）                                    |
   | `payload` | 實際送出的 JSON，除錯用                                                                         |
   | `metrics` | 送出前自檢：`wind_angle` / `L_char(px)` / `BCR` / `aspect`，落在訓練分布外時 `status` 出 `WARN` |

4. 先啟動後端：雙擊 **`run_gh_backend.bat`**（本目錄下，GH 專用，不開網頁/http.server）。
   等 `NCA main.py (GH)` 視窗印出 `[main] WS server ws://localhost:8765`（模型載入約
   20–40 秒），再把 GH 的 `run` 撥成 `True`。
   （純網頁流程用的 `run_demo.bat`／啟動器面板也會啟動同一個推論服務，GH 跟網頁可同時連，
   兩者擇一啟動即可，不用重複開。）
5. 流場一樣顯示在 **Taichi GGUI 視窗**，GH 面板不顯示流場，`status` 只是 server 的 ack 字串。

### 跟網頁並行使用

想同時展示「網頁畫建物」+「GH 送任意角度風向」時，兩邊各自照自己的啟動流程連上同一個
8765 即可，不用額外設定。唯一要注意：GH 元件的 `url` 欄位**不會**跟著啟動器面板的
WS port 輸入框自動同步（那條自動同步只接到網頁 `Connect` 欄位）——如果曾經在啟動器面板
把 port 從預設 `8765` 改掉，GH 元件的 `url` 也要手動改成同一個值，否則會連不上。

### 注意事項

- boundary bbox 會被 server 正規化到 `[0,1]×[0,1]` 再光柵化成 360×640 mask；若 boundary
  長寬比不是 16:9，幾何會被拉伸——`status` 會出現 `WARN` 提示，建議把 boundary 畫成 16:9。
- 建物尺寸有訓練分布：`L_char`（迎風橫向投影寬度，換算成 1920×1080 sim px）須落在
  `[32, 128]`，即該邊長為 boundary 短邊的 **3%–12%**；`BCR`（建物總面積／boundary 面積）
  須落在 `[0.01, 0.30]`。超出時 `status` 出 `WARN`，推論照跑但條件已在訓練分布外。
- 元件自足：不需要裝 `websocket-client`（內建純 socket RFC6455 client），Rhino 7
  （IronPython 2.7）跟 Rhino 8（CPython 3.9）都能跑。

---

## 疑難排解

| 症狀                                                  | 原因 / 處置                                                                                                                        |
| ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| 雙擊 `.bat` 跳出一堆「不是內部或外部命令」亂碼         | 本目錄下的 `.bat` 都是純 ASCII，正常不會發生。若出現，代表檔案被改動過（例如貼入了中文註解），在 Big5/950 codepage 下會把整個 batch 解析炸掉，回報並還原。 |
| 雙擊 `install_autostart.ps1` 開了記事本，沒反應        | 正常——Windows 對 `.ps1` 預設動作是編輯不是執行。改用上面「方式 B 步驟 1」的指令跑。                                               |
| 按「啟動 demo 後端」，面板顯示「無法連上 launcher」    | launcher 沒在跑。先雙擊 `launcher.bat`（或做過方式 B 步驟 1 的話，重開機讓它自動起）。                                            |
| Connect 一直紅燈                                       | 推論服務還在載模型，等 20–40 秒再按。或檢查 8765 有沒有被別的程式佔用（`netstat -ano \| findstr 8765`）。                          |
| GGUI 視窗沒跳出來 / 全黑                               | 看推論服務視窗有沒有印出 Traceback；可能是 Vulkan/顯卡驅動問題。                                                                   |
| 推論服務視窗印出 `GVecUnavailableError`                | 這是設計上的守門行為（幾何轉 G_vec 失敗時直接報錯，不會偷偷退回假資料），不是要你改設定去關掉它——回報實際錯誤內容。               |
| 想同時開好幾次 demo                                    | 不要。已在跑的話 `launcher /start` 會偵測到 port 已被佔用而不重複啟動；但用 `run_demo.bat` 手動開兩次會搶 port 出錯，關掉舊視窗再開新的。 |

---

## 開發者才需要知道的事（一般使用者可略過）

`launcher.py`（常駐橋接服務本體）、`app.js`（前端邏輯）都在本目錄下；推論服務本體
（`src/lbmnca/bridge/main.py`）、工程參數（`configs/bridge/config.yaml`）在 repo 根的
`src/` 與 `configs/` 底下，細節見 repo 根的說明文件。
