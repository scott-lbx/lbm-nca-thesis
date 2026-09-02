"""
公開 API facade -- lbm_mrt_les 的共用前後處理函式。

所有在 pipeline 各環節（mask 生成 -> config -> 模擬 -> H5 -> Zarr）
以及外部專案（lbmnca.nca）之間共用的處理函式都從此處匯出。

使用方式：
    from lbmnca.lbm.lbm_mrt_les.public import mask, static_features, conditioning
    from lbmnca.lbm.lbm_mrt_les.public import geometry  # 需要 morphospace 相依時才 import

★ geometry 刻意不在此 eager import：它相依 lbmnca.morphospace（L_char 單一真相
  委派至 physics_proxy.lchar_api），只需要 mask / static_features 的呼叫端
  不該被迫連帶安裝 morphospace 相依。需要 geometry 者直接
  ``from lbmnca.lbm.lbm_mrt_les.public import geometry`` 或
  ``from lbmnca.lbm.lbm_mrt_les.public.geometry import ...``。
"""

from . import conditioning, mask, static_features
