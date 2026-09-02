"""param_pca — 純參數（rectangle）設計空間的 PCA pipeline（完全自包含）。

mask 本質 = N 個 axis-aligned rectangle，每個 (cx, cy, w, h)。本套件**不 render 圖片、不依賴
本資料夾外任何功能**（只用 numpy / matplotlib）：

  generate.py  自包含 rectangle-layout 取樣器（已知 min/max 範圍）→ 母群體 NPZ（凍結）
  features.py  order-stats featurize：(n,4) → 定長 21-d（permutation-invariant、平滑）
  model.py     ParamPCA：StandardScaler + PCA（numpy SVD）fit/transform/save/load（凍結 transform）
  inspect.py   載母群體 → fit → 解釋變異 / PC 意義 / scatter / 小編輯 locality

設計理由：用生成參數本身（非 pixel、非 NN）→ 線性、單調、可解釋、可反解、母群體凍結破循環依賴。
"""

from lbmnca.morphospace.param_pca.features import FEATURE_NAMES, featurize, featurize_population
from lbmnca.morphospace.param_pca.model import ParamPCA

__all__ = ["featurize", "featurize_population", "FEATURE_NAMES", "ParamPCA"]
