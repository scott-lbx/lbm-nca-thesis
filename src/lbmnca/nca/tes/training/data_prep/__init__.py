"""訓練資料前處理子模組。

本套件將 TESTrainer 的資料前處理邏輯拆分為三個職責明確的子元件：

- ConditionBuilder：建構正規化後的條件向量（cond_norm / g_vec_norm）。
- DataRenormalizer：將 source-local 正規化空間轉至 dataset-global 空間。
- ICOps：初始條件（IC）相關操作（流體遮罩、訓練雜訊、速度先驗、GT 注入機率）。

每個子元件的 __init__ 只接受 ctx: TrainerContext，方法回傳值均為
torch.Tensor 或 dict[str, float]。
"""

from .condition_builder import ConditionBuilder
from .renorm import DataRenormalizer
from .ic_ops import ICOps

__all__ = ["ConditionBuilder", "DataRenormalizer", "ICOps"]
