"""VQI 的量測項（terms）定義與項組（term sets）。

本模組只負責**定義**，即每一項在算什麼、哪些項組成一組。刻度所需的參照場在
`fields`，刻度與歸一在 `anchors`，合成與否決在 `score`，
對外唯一入口為 `score.compute_vqi`。

## 項的分群

| 群 | 職責 |
|---|---|
| `struct` | 結構對位。位置錯了、崩壞了 |
| `texture` | 紋理健康。偽影、顆粒、過度平滑 |
| `energy` | 能量與峰值。整體幅值與剪切層強度 |

## 項的三種方向

| 方向 | 意義 | 偏差量的取法 |
|---|---|---|
| `up` | 越大越好 | ``1 − x``，理想值為 1 |
| `down` | 越小越好 | ``x``，理想值為 0 |
| `ratio` | 比值，目標 1.0 | 以 ``\\|ln r\\|`` 度量偏離，雙側對稱 |

偏差量其後由 `anchors.bounded_deviation` 壓成有界，再以該案零場的偏差為單位
換算成分數，故 ``+1`` 為真值、``−1`` 為零場。
"""
from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn.functional as F

from .. import visual_metrics as VM

__all__ = [
    "ERODE_PX", "RAPSD_BINS", "PEAK_Q",
    "TERMS", "TERM_SETS", "CANONICAL_TERM_SET", "EXCLUDED_TERMS",
    "VETO_REF_TERM", "IDEAL_BY_KIND", "BASELINE_TAG", "ZERO_TAG",
    "ARTIFACT_TAGS", "DEAD_TAGS", "CLEAN_TAGS",
    "resolve_term_set", "compute_terms",
    "LATTICE_SOUND_SPEED", "validity",
]


# ─────────────────────────── 物理有效性 ───────────────────────────

#: 晶格聲速。LBM 的低馬赫近似在速度接近此值時失效，資料集本身亦全數遠低於它。
#: 取此值作界不是調出來的門檻，是該數值方法的適用邊界。
LATTICE_SOUND_SPEED: float = 1.0 / math.sqrt(3.0)


def validity(pred_norm: torch.Tensor, fluid_mask: torch.Tensor,
             normalizer) -> list[str]:
    """物理有效性前置檢查。回傳違反的項目，全部通過時為空 list。

    **這不是視覺判斷，是前置條件。** 一個帶負密度的場根本不是流場，
    談它「看起來像不像」沒有意義，故與 VQI 的數值分開報。

    兩條判準皆無可調參數。

    | 判準 | 為何是硬性 |
    |---|---|
    | 流體區出現負密度或非有限值 | 密度為負在物理上不存在；且速度由 $j/
ho$ 得出，跨過零點時該除法發散 |
    | 速度大小超過晶格聲速 | 超過即離開 LBM 低馬赫近似的適用範圍，該場已不在資料生成時的物理設定內 |

    ★ 這兩條是本檢查存在的理由。實測一支模型於長推進段在**固體邊界旁數十個像素**
    產生負密度，速度隨之發散至真值的十萬倍，而五項中有四項完全看不見——
    頻譜兩項的遮罩腐蝕正好把固體邊那一圈切掉、`Peak_Ratio` 取 p99.9 而該事件
    只佔萬分之一像素、結構項上色後被顯示範圍截斷。唯一看得見的是
    `VortSignedCorr`，且它只能報成相關係數歸零，說不出成因。

    Args:
        pred_norm: ``[N,9,H,W]`` 正規化預測場。
        fluid_mask: ``[N,1,H,W]``，流體為 1。
        normalizer: `FieldNormalizer`。

    Returns:
        違反項目的中文描述。空 list 代表通過。
    """
    bad: list[str] = []
    if not bool(torch.isfinite(pred_norm).all()):
        bad.append("出現非有限值")
    fm = fluid_mask[:, 0] > 0.5
    if not bool(fm.any()):
        return bad
    rho = normalizer.denorm_mphy(pred_norm)[:, VM._RHO_CH]
    if float(rho[fm].min()) < 0.0:
        bad.append("流體區出現負密度")
    if float(_vel_mag(pred_norm, normalizer)[fm].max()) > LATTICE_SOUND_SPEED:
        bad.append("速度大小超過晶格聲速")
    return bad


# ─────────────────────────── 共用前處理 ───────────────────────────

def _vel_mag(x_norm: torch.Tensor, norm) -> torch.Tensor:
    """反正規化取速度大小場。★必先反正規化，不可對正規化場直接算速度。"""
    return VM._denorm_vel_mag(x_norm, norm)


def _masked(v: torch.Tensor, fm: torch.Tensor) -> torch.Tensor:
    """固體區歸零。預測與真值施以相同處理，比值中的邊緣效應因而對消。"""
    return v * fm[:, 0]


#: 頻譜類項的遮罩腐蝕半徑（像素）。
#: ★未腐蝕時固體邊界的階梯直接進頻譜，實測使棋盤格項在「模糊 σ=0.7」給 2.68、
#: 在「平移 2px」給 17，兩者都不是棋盤格偽影。
ERODE_PX = 8


def _erode(fm: torch.Tensor, r: int = ERODE_PX) -> torch.Tensor:
    """把流體遮罩自**固體周圍**往內縮 ``r`` 像素（以最大池化對補集膨脹實作）。

    ★兩點行為須知，兩者皆為刻意。

    其一，**計算域的四個邊界不受影響**。`max_pool2d` 以負無限大補邊，
    全流體遮罩的補集恆為零，池化後仍為零。要擋的是固體邊界的階梯，
    計算域邊界沒有固體、不構成該問題。

    其二，腐蝕**把階梯往內移，並未消除它**。真正抵銷邊緣效應的是**比值**，
    預測與真值施以完全相同的遮罩與腐蝕，共同的邊緣能量在相除時對消。
    腐蝕的作用是讓固體周圍那一圈受幾何改動影響最劇烈的格點不進入頻譜。
    """
    k = 2 * r + 1
    return 1.0 - F.max_pool2d(1.0 - fm, k, stride=1, padding=r)


def _spec(v: torch.Tensor, fm: torch.Tensor
          ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """腐蝕後流體內部的二維功率譜與其正規化波數格。

    Returns:
        ``(功率譜, |ky|, |kx|)``，波數以 Nyquist 為 0.5 的循環頻率表示。
    """
    fe = _erode(fm)
    u = _masked(v, fe)
    u = u - u.sum() / fe.sum().clamp_min(1.0)
    u = u * fe[:, 0]
    p = torch.fft.fft2(u).abs().square()
    ky = torch.fft.fftfreq(p.shape[-2], device=p.device).abs().view(-1, 1)
    kx = torch.fft.fftfreq(p.shape[-1], device=p.device).abs().view(1, -1)
    return p, ky, kx


# ─────────────────────────── 紋理群的量 ───────────────────────────

def total_variation(v: torch.Tensor, fm: torch.Tensor) -> float:
    """流體區內的總變差，量的是紋理總量（顆粒與條紋都會抬高它）。"""
    u = _masked(v, fm)
    dx = (u[..., :, 1:] - u[..., :, :-1]).abs().sum()
    dy = (u[..., 1:, :] - u[..., :-1, :]).abs().sum()
    return float((dx + dy).item())


def checker_energy(v: torch.Tensor, fm: torch.Tensor, frac: float = 0.1) -> float:
    """Nyquist 角落的能量佔比。棋盤格偽影的能量幾乎全落在此處。

    取**佔比**而非絕對能量，使其不隨場的整體幅值變動。
    """
    p, ky, kx = _spec(v, fm)
    m = (ky > 0.5 * (1 - frac)) & (kx > 0.5 * (1 - frac))
    return float((p[..., m].sum() / p.sum().clamp_min(1e-20)).item())


def band_prominence(v: torch.Tensor, fm: torch.Tensor, ky_min: float = 0.05) -> float:
    """水平條紋的凸起度，即 ``kx≈0`` 那一行在中高 ``ky`` 上的能量佔比。

    ★取絕對能量比值時水平條紋 25% 只給 1.04、完全沒有鑑別度，因為真值本身在
    ``kx≈0`` 的能量極大（時均流沿 x 近乎均勻）。改取佔比並排除低 ``ky`` 才分得開。
    """
    p, ky, kx = _spec(v, fm)
    hi = ky >= ky_min
    line = (kx <= 1.5 / p.shape[-1]) & hi
    denom = p[..., hi.expand_as(kx * ky)].sum().clamp_min(1e-20)
    return float((p[..., line].sum() / denom).item())


def hf_band_energy(v: torch.Tensor, fm: torch.Tensor,
                   lo: float = 0.15, hi: float = 0.40) -> float:
    """中高頻環帶的能量佔比，**雙側**。

    過度平滑使其下降、顆粒使其上升。其獨有貢獻在平滑側，顆粒側與其他項重疊。
    """
    p, ky, kx = _spec(v, fm)
    kr = (ky.square() + kx.square()).sqrt()
    band = (kr >= lo) & (kr <= hi)
    return float((p[..., band].sum() / p.sum().clamp_min(1e-20)).item())


def diag_prominence(v: torch.Tensor, fm: torch.Tensor, kr_min: float = 0.05,
                    kr_max: float = 0.40, wedge: float = 0.12) -> float:
    """斜向條紋的凸起度，即頻譜對角楔形 ``|ky|≈|kx|`` 在中頻環帶中的能量佔比。

    ``_spec`` 回傳 ``|ky|`` 與 ``|kx|``，兩個對角方向摺疊到同一條線，單一楔形
    即可涵蓋 ``y+x`` 與 ``y−x``。環帶上界 0.40 排除 Nyquist 角落，棋盤格因而
    不落入本項，與 `Checker_Ratio` 不重複計分。
    """
    p, ky, kx = _spec(v, fm)
    kr = (ky.square() + kx.square()).sqrt()
    ann = (kr >= kr_min) & (kr <= kr_max)
    aniso = (ky - kx).abs() / kr.clamp_min(1e-12)
    diag = ann & (aniso <= wedge)
    return float((p[..., diag].sum() / p[..., ann].sum().clamp_min(1e-20)).item())


#: 徑向分箱數。二維譜的最大半徑在角落，為 ``sqrt(0.5)``。
RAPSD_BINS = 24
_KR_MAX = 0.7071067811865476


def rapsd_shape(v: torch.Tensor, fm: torch.Tensor, nbin: int = RAPSD_BINS) -> torch.Tensor:
    """徑向平均功率譜，並正規化為**形狀**（除以總和）。

    正規化是關鍵。未正規化時幅值縮放會讓整條譜均勻平移，實測「幅值 ×0.25」
    在對數尺度上偏離 2.865，恰等於 ``2·|ln 0.25|``，該平移會蓋過真正的偽影訊號。
    除以總和之後只剩形狀，幅值改由能量群承擔。

    分箱涵蓋到角落，最高一環即網格尺度（2Δx），棋盤格的能量全落在該環。
    """
    p, ky, kx = _spec(v, fm)
    kr = (ky.square() + kx.square()).sqrt()
    idx = (kr / _KR_MAX * nbin).long().clamp(0, nbin - 1).flatten()
    pf = p.mean(dim=0).flatten()
    acc = torch.zeros(nbin, device=p.device, dtype=p.dtype)
    cnt = torch.zeros(nbin, device=p.device, dtype=p.dtype)
    acc.scatter_add_(0, idx, pf)
    cnt.scatter_add_(0, idx, torch.ones_like(pf))
    r = acc / cnt.clamp_min(1.0)
    return r / r.sum().clamp_min(1e-20)


def rapsd_distance(vp: torch.Tensor, vt: torch.Tensor, fm: torch.Tensor,
                   nbin: int = RAPSD_BINS) -> float:
    """預測與真值的徑向功率譜形狀距離，取逐環對數比的最大值。

    這是生成模型頻譜偏差的標準診斷（Durall et al., CVPR 2020），在流體側等同於
    比對能量譜。棋盤格在流體力學裡即 odd-even decoupling／2Δx 振盪，其能量落在
    最高一環，本項因而以單一構造涵蓋四類偽影。

    **回報最大值而非平均。** 棋盤格的能量集中在譜的一個小角落，取平均會被稀釋，
    實測全譜平均的分離度僅 0.13、逐環取最大為 10.40。

    binding 的環號另有診斷價值，最高環為網格尺度、中頻環為條紋或顆粒。
    """
    a = rapsd_shape(vp, fm, nbin)
    b = rapsd_shape(vt, fm, nbin)
    return float((torch.log(a + 1e-12) - torch.log(b + 1e-12)).abs().max().item())


# ─────────────────────────── 能量群的量 ───────────────────────────

def kinetic_energy(v: torch.Tensor, fm: torch.Tensor) -> float:
    """流體區平均動能代理 ``mean(|u|²)``。"""
    u = _masked(v, fm)
    return float((u.square().sum() / fm.sum().clamp_min(1.0)).item())


#: 峰值所取的分位數。取 p99.9 而非最大值，避免單一格點主導。
PEAK_Q = 0.999


def peak_speed(v: torch.Tensor, fm: torch.Tensor, q: float = PEAK_Q) -> float:
    """流體區速度大小的高分位數，代表剪切層與狹管噴流的峰值強度。"""
    return float(torch.quantile(v[fm[:, 0] > 0].float(), q).item())


# ─────────────────────────── 項表 ───────────────────────────

#: 已於儀器測試證明失效、**不得進入 VQI** 的項。
EXCLUDED_TERMS: dict[str, str] = {
    "StructVort_RGB_SSIM": "對 25% 棋盤格給 0.9871、對均勻場給 0.9567",
    "HorizontalBand_Score": "對棋盤格、白雜訊、條紋三類偽影皆給 0.0000",
    "HorizontalBand_FFT": "對最糟的均勻場給滿分 0.0000，非單調",
    "Sat_Frac": "超出真值全域最大值的像素比例。極端序統計量當門檻且為單向，"
                "16 個退化中 12 個給滿分，已由 Peak_Ratio 取代",
}

#: 項名 → ``(群, 方向, 取值函式)``。
#: 取值函式簽名統一為 ``f(pred_norm, target_norm, fluid_mask, normalizer) -> float``。
TERMS: dict[str, tuple[str, str, Callable]] = {
    # 結構對位
    "StructVel_RGB_SSIM": ("struct", "up",
                           lambda p, t, m, n: VM.struct_vel_rgb_ssim(p, t, n, m)),
    "StructVort_HighMask_IoU": ("struct", "up",
                                lambda p, t, m, n: VM.struct_vort_highmask_iou(p, t, m, n)),
    "StructWake_Region_Mae": ("struct", "down",
                              lambda p, t, m, n: VM.struct_wake_region_mae(p, t, m, n)),
    "VortSignedCorr": ("struct", "up",
                       lambda p, t, m, n: VM.vort_signed_corr(p, t, m, n)),
    # 紋理健康
    "HF_Excess_Ratio": ("texture", "down",
                        lambda p, t, m, n: VM.hf_excess_ratio(p, t, m, n)),
    "TV_Ratio": ("texture", "ratio",
                 lambda p, t, m, n: total_variation(_vel_mag(p, n), m)
                 / max(total_variation(_vel_mag(t, n), m), 1e-12)),
    "Checker_Ratio": ("texture", "ratio",
                      lambda p, t, m, n: checker_energy(_vel_mag(p, n), m)
                      / max(checker_energy(_vel_mag(t, n), m), 1e-12)),
    "Band_Prom_Ratio": ("texture", "ratio",
                        lambda p, t, m, n: band_prominence(_vel_mag(p, n), m)
                        / max(band_prominence(_vel_mag(t, n), m), 1e-12)),
    "HF_Band_Ratio": ("texture", "ratio",
                      lambda p, t, m, n: hf_band_energy(_vel_mag(p, n), m)
                      / max(hf_band_energy(_vel_mag(t, n), m), 1e-12)),
    "Diag_Prom_Ratio": ("texture", "ratio",
                        lambda p, t, m, n: diag_prominence(_vel_mag(p, n), m)
                        / max(diag_prominence(_vel_mag(t, n), m), 1e-12)),
    "RAPSD_Shape": ("texture", "down",
                    lambda p, t, m, n: rapsd_distance(_vel_mag(p, n), _vel_mag(t, n), m)),
    # 能量與峰值
    "KE_Ratio": ("energy", "ratio",
                 lambda p, t, m, n: kinetic_energy(_vel_mag(p, n), m)
                 / max(kinetic_energy(_vel_mag(t, n), m), 1e-12)),
    "Peak_Ratio": ("energy", "ratio",
                   lambda p, t, m, n: peak_speed(_vel_mag(p, n), m)
                   / max(peak_speed(_vel_mag(t, n), m), 1e-12)),
}

# ─────────────────────────── 刻度的定點 ───────────────────────────

#: 理想值。**由指標定義決定，不是量出來的**——相似度與比值的理想值為 1，
#: 距離型的理想值為 0。真值自身兩段時均窗之間仍有差距，該差距不作為刻度，
#: 改登錄於錨點的 `noise_floor` 欄供對照，其落點即該項的量測噪聲底線。
IDEAL_BY_KIND: dict[str, float] = {"up": 1.0, "down": 0.0, "ratio": 1.0}

#: 暖啟動起點在退化階梯上的級名。**起點不參與刻度定義**，
#: 它只是卷面上量出來的一個點，列於階梯供對照與相減。
BASELINE_TAG: str = "恆等預測（起點）"

#: 零場在退化階梯上的級名。刻度的下端點，分數恆為 $-1$。
#: 論文以「零場」指稱它，與「均勻場（結構全失）」不同——後者仍保有均勻來流。
ZERO_TAG: str = "零場"


#: 逐項的否決參照。**每一項只對它負責偵測的偽影設門檻**，取該類最輕的一級。
#: 判定式為 ``s ≤ τ``，被列為參照的偽影因而必然自我否決——這對偽影是要的行為，
#: 參照的語義就是「這一類裡最輕、但仍必須被擋下的那一級」。
#:
#: ★參照場由該案的真值當場造出，見 `fields.veto_reference_fields`，
#: 其振幅由 `fields.VETO_REF_AMPLITUDE` 定為動量標準差的一成。
#: 類別名不帶百分比即因振幅已由該常數固定，寫進名字會造成兩處真相。
#:
#: ★**過度平滑不列入**。曾把過度平滑列為 `HF_Band_Ratio` 的參照，
#: 等於宣告過度平滑為視覺不可接受，且把門檻抬高而連帶把「幅值 ×0.25」
#: 掃進否決。原生像素裁切顯示這兩者乾淨無偽影，其劣化由 VQI 數值承載。
VETO_REF_TERM: dict[str, tuple[str, ...]] = {
    "Checker_Ratio": ("棋盤格",),
    "HF_Excess_Ratio": ("白雜訊",),
    "TV_Ratio": ("白雜訊",),
    "Band_Prom_Ratio": ("水平條紋",),
    "HF_Band_Ratio": ("白雜訊",),
    "Diag_Prom_Ratio": ("斜向條紋",),
}

#: 精簡組的否決參照。單一頻譜項同時負責四類偽影，故四類的**輕級全部列入**，
#: `veto_thresholds` 取其中最高的子分數，門檻因而落在最容易漏掉的那一類上。
#:
#: ★`HF_Band_Ratio` **不列入**。它在精簡組裡專責過度平滑，而過度平滑不是偽影；
#: 且其門檻會塌成 τ=0，語義變成「只要超出錨點範圍就否決」，實測使兩個範式基線
#: 於部署步 100% 被否決、主模型 22%，過嚴且無鑑別力。
_SLIM_VETO_REF: dict[str, tuple[str, ...]] = {
    "RAPSD_Shape": ("棋盤格", "白雜訊", "水平條紋", "斜向條紋"),
}


def _pick(*names: str) -> dict[str, tuple[str, str, Callable]]:
    """由 `TERMS` 取指定項，維持宣告順序。"""
    return {n: TERMS[n] for n in names}


#: 項組。`full` 為十二項的完整組，其餘為精簡候選。
#: `slim_pos5` 為五項組，一種失效對一項。
TERM_SETS: dict[str, dict[str, tuple[str, str, Callable]]] = {
    "full": {k: v for k, v in TERMS.items() if k != "RAPSD_Shape"},
    "slim_iou": _pick("StructVort_HighMask_IoU", "RAPSD_Shape", "Peak_Ratio"),
    "slim_corr": _pick("VortSignedCorr", "RAPSD_Shape", "Peak_Ratio"),
    "slim_ssim": _pick("StructVel_RGB_SSIM", "RAPSD_Shape", "Peak_Ratio"),
    "slim4": _pick("StructVel_RGB_SSIM", "StructVort_HighMask_IoU",
                   "RAPSD_Shape", "Peak_Ratio"),
    "slim_smooth": _pick("StructVel_RGB_SSIM", "RAPSD_Shape",
                         "HF_Band_Ratio", "Peak_Ratio"),
    "slim_corr4": _pick("VortSignedCorr", "RAPSD_Shape",
                        "HF_Band_Ratio", "Peak_Ratio"),
    # 五項各負責一種失效：
    #   整體結構／位置偏移／網格尺度偽影／過度平滑／峰值強度
    "slim_pos5": _pick("StructVel_RGB_SSIM", "VortSignedCorr", "RAPSD_Shape",
                       "HF_Band_Ratio", "Peak_Ratio"),
}

#: **canonical 項組**。五項各負責一種失效，八關全過。
#: 讀數檔名一律帶項組後綴以資區別；canonical 不帶後綴。
CANONICAL_TERM_SET = "slim_pos5"

#: 視覺校準的三組標籤。**由退化階梯的原生像素裁切逐格目視判定**。
#: 否決旗標必須與本組標籤完全吻合，不吻合時改項組或改參照，不調門檻遷就。
ARTIFACT_TAGS: tuple[str, ...] = (
    "棋盤格 25%", "棋盤格 10%", "白雜訊 25%", "白雜訊 10%",
    "水平條紋 25%", "水平條紋 10%", "斜向條紋 25%", "斜向條紋 10%")
#: 結構全失的兩級。非紋理偽影，但視覺上同樣不可接受。
#: 「均勻場（結構全失）」逐通道取流體區空間平均後廣播回全場，空間變化盡失
#: 而動量守住，仍保有約六成動能；「零場」才是動量全歸零、逐點靜止。
#: 後者即刻度的下端點，其分數恆為 −1。
DEAD_TAGS: tuple[str, ...] = ("均勻場（結構全失）", ZERO_TAG)
#: 乾淨退化。品質有損但無偽影，其劣化由 VQI 數值承載，**不得觸發否決**。
CLEAN_TAGS: tuple[str, ...] = (
    "GT 自己", "模糊 σ=0.7", "模糊 σ=1.5", "模糊 σ=3.0",
    "幅值 ×0.75", "幅值 ×0.5", "幅值 ×0.25", "平移 2px", "平移 8px")


def resolve_term_set(name: str) -> tuple[dict, dict]:
    """依名稱取項表與否決參照表。

    Args:
        name: `TERM_SETS` 的鍵。

    Returns:
        ``(terms, veto_ref_term)``。

    Raises:
        KeyError: 未知的項組名稱。
    """
    if name not in TERM_SETS:
        raise KeyError(f"未知的項組 {name!r}，可選 {'／'.join(TERM_SETS)}")
    terms = TERM_SETS[name]
    return terms, (VETO_REF_TERM if name == "full" else _SLIM_VETO_REF)


def compute_terms(pred: torch.Tensor, target: torch.Tensor, fluid_mask: torch.Tensor,
                  normalizer, term_set: str = "full") -> dict[str, float]:
    """算出指定項組全部項的原始值。

    單一項失敗記 ``nan``，**不靜默**，由呼叫端的日誌承接。

    Args:
        pred: ``[N,9,H,W]`` 正規化預測場。
        target: ``[N,9,H,W]`` 正規化真值場。
        fluid_mask: ``[N,1,H,W]``，流體為 1。
        normalizer: `FieldNormalizer`。
        term_set: 項組名稱。

    Returns:
        項名 → 原始值。
    """
    import logging
    log = logging.getLogger("tes.eval.vqi")
    terms, _veto_ref = resolve_term_set(term_set)
    out: dict[str, float] = {}
    for name, (_grp, _dirn, fn) in terms.items():
        try:
            out[name] = float(fn(pred, target, fluid_mask, normalizer))
        except Exception as exc:                                  # noqa: BLE001
            out[name] = float("nan")
            log.warning("項 %s 計算失敗：%s: %s", name, type(exc).__name__, exc)
    return out
