"""param_pca 域 / 取樣常數（自包含，不 import 外部）。

數值對齊 flat_diverse_gen 的規範佈局（inner-box 偏左 + 右側 wake 區 + 大小錯落），
但在此**獨立定義**，使 param_pca 完全不依賴本資料夾外功能。
"""

from __future__ import annotations

# ── Domain（mask 像素座標）─────────────────────────────────────────────
PAD_LEFT: int = 256
PAD_TOP: int = 156
PAD_RIGHT: int = 896
PAD_BOT: int = 156
INNER: int = 768
H_TOTAL: int = PAD_TOP + INNER + PAD_BOT      # 1080
W_TOTAL: int = PAD_LEFT + INNER + PAD_RIGHT    # 1920

# ── 建物取樣範圍（已知 min/max）────────────────────────────────────────
W_MIN: int = 25
W_MAX: int = 220
H_MIN: int = 25
H_MAX: int = 220
N_MIN: int = 3        # 每佈局建物數下界
N_MAX: int = 20       # 上界
MIN_GAP: int = 12     # 建物邊緣最小間距
INNER_MARGIN: int = 6  # 距 inner box 邊界緩衝
MAX_PLACE_TRIES: int = 200

# ── 建物中心可放範圍（inner box 內，含 margin）────────────────────────
CX_LO: int = PAD_LEFT + INNER_MARGIN                 # 262
CX_HI: int = PAD_LEFT + INNER - INNER_MARGIN          # 1018
CY_LO: int = PAD_TOP + INNER_MARGIN                   # 162
CY_HI: int = PAD_TOP + INNER - INNER_MARGIN           # 918

# ── 參數欄位順序 ───────────────────────────────────────────────────────
PARAM_COLS: tuple[str, ...] = ("cx", "cy", "w", "h")   # (n, 4)
