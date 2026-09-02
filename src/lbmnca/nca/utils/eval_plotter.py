import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
import logging
from typing import List, Callable, Optional

# 繁中可顯示字型（優先 JhengHei；避免 CJK glyph 缺字方框）
_avail_fonts = {f.name for f in _fm.fontManager.ttflist}
for _zh in ("Microsoft JhengHei", "Microsoft YaHei", "Noto Sans CJK TC", "SimHei"):
    if _zh in _avail_fonts:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [_zh, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        break

def plot_multi_k_convergence(
    eval_func: Callable[[int], float],
    k_list: List[int],
    step: int,
    output_dir: str,
    logger: Optional[logging.Logger] = None
):
    """
    評估並繪製多個 K 步數下的 Eval MAE 收斂曲線，驗證模型是否具有 Attractor 特性。
    
    Args:
        eval_func: 接受 K (int) 並返回 MAE (float) 的評估函式（Callback）。
        k_list: 評估時所使用的 K (NCA Steps) 列表。
        step: 當前的訓練步數。
        output_dir: 圖表儲存目錄。
        logger: 日誌記錄器。
    """
    try:
        mae_list = []
        for k in k_list:
            mae = eval_func(k)
            mae_list.append(mae)

        plt.figure(figsize=(8, 5))
        plt.plot(k_list, mae_list, marker='o', linestyle='-', color='#D32F2F', linewidth=2)
        plt.title(f"Eval MAE vs NCA Steps (K) @ Step {step}", fontsize=14)
        plt.xlabel("NCA Steps (K)", fontsize=12)
        plt.ylabel("Eval MAE", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(k_list)
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f"k_convergence_step{step}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        if logger:
            logger.info(f"[Multi-K Eval@{step}] Saved convergence plot to {os.path.basename(save_path)}")
    except Exception as e:
        if logger:
            logger.warning(f"[Multi-K Eval@{step}] Multi-K Convergence Failed: {e}")
        else:
            print(f"[Multi-K Eval@{step}] Multi-K Convergence Failed: {e}")
