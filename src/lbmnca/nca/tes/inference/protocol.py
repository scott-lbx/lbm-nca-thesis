"""protocol.py — 三態（train / eval / deploy）推論協定單一真相。

背景：同一顆 ckpt 若在訓練、評估、部署三處各自實作，容易走出三套不同算子——
``ke_anchor`` 三種來源、chain 交接時 hidden 一處歸零兩處保留、``apply_ic_mask``
有處套有處不套、step 與 CTU 的換算有四套常數。結果是「評估的不是部署的那條鏈」，
重跑只會產出不可比的數字。

本模組提供這四件事的實作，mean 線的訓練 / 評估 / 部署三態應一律呼叫本模組，
不得各自複製邏輯：

    * :func:`resolve_ke_anchor` / :func:`resolve_ke_anchor_from_model` —— KE ceiling
      錨點。**純粹是 rollout 自身起始 state 的函式**，不吃監督目標（訓練期 GT），
      也不跨 rollout 累積（部署期 running-min）。``ke_anchor_mode="off"`` 時回 ``None``。
    * :func:`mask_ic_solid` —— ``apply_ic_mask`` 的固體區歸零實作。推論／評估側
      （``src/lbmnca/nca/tes/eval/`` 與 ``src/lbmnca/nca/tes/inference/``）共用這一份：
      ``initial_state`` 與訓練端的 val／vis 路徑都改呼叫它。
      ★界線：訓練層與腳本層**沒有**併進來——``src/lbmnca/nca/tes/training/steps/``
      仍有 **1** 處自寫 ``masked_fill``（``base_step``）。
      （``eval/null_models.py`` 另有一處 ``masked_fill``，但那是零模型**輸出**的後處理、
      不是 stage 0 的 IC 政策，不算在內。）
    * :func:`initial_state` / :func:`initial_state_from_flag` —— stage 0 起始 state
      （政策由 ``cfg.apply_ic_mask`` 直接存取決定，禁 fallback；拿不到 cfg 的泛用
      rollout 函式改用 ``_from_flag`` 版，把布林值當顯式參數傳）。
    * :func:`chain_handoff` —— chain stage 交接（``apply_ic_mask`` +
      ``handoff_reset_mask`` + static 置換，**hidden 在重置區之外保留**；重置區內 hidden
      與 physics 一起歸零，見該函式 docstring）。
    * :data:`NOMINAL_DELTA_CTU_PER_STEP` —— **mean 線**的 step↔CTU 換算。（turb 線是另一
      個常數 :data:`TURB_DELTA_CTU_PER_STEP`，語意不同、不可共用；見本 docstring 末段。）

★ 覆蓋範圍的誠實界線：本模組**不是**全 workspace 唯一入口。

    * ``ke_anchor``：掃描器為 ``tests/nca/tes/test_forward_ke_anchor_coverage.py``，只覆蓋其
      ``SCOPE_DIRS``（``src/lbmnca/nca/tes/eval/``、``src/lbmnca/nca/tes/inference/``、
      ``src/lbmnca/nca/inference/``、前端橋接的 ``adapters/``）加上明列 harness；
      ``missing=0`` 只代表這個範圍內沒有漏接。豁免一律在該檔 ``ALLOWLIST`` 逐筆具名
      並附理由，**筆數、內容與理由分類一律以該檔為準**（此處刻意不複製，避免兩處
      各記一份而漂移）。豁免理由常見兩種：turb 線的 forward 簽章不收 ``ke_anchor``，
      以及尚未遷移到協定的檔案；確切清單請直接開該檔的 ``ALLOWLIST``。
    * ``apply_ic_mask``：掃描器為 ``tests/nca/tes/test_ic_mask_policy_coverage.py``，
      同樣只看上述目錄，且只查「stage 0 是否走 :func:`initial_state`」，不查訓練層與
      ``scripts/``。ke_anchor 掃描器對 IC 政策**一字未證**，兩者的 ``missing=0`` 不可互相
      借用。
    * turb 線與其他架構（gc_nca / dynca）都不在本模組治理範圍。

★ 為何錨點必須是「起始 state 的函式」：協定要求同一 ckpt 對**同一輸入**在三態給
相同輸出。訓練期舊實作取監督目標（GT mean）的 KE，推論期根本拿不到；部署端舊實作
取跨幾何跨鏈永不重置的 running-min，輸出因此取決於此進程先前評過哪些幾何，連
「輸入的函式」都不是。兩者皆無法在三態成立；本次 rollout 起始 state 自己的 KE 則三態
都拿得到、且對同一輸入必然相同，故採用它。（沒有窮舉過所有候選量——這裡主張的是
「舊的兩個不成立、這個成立」，不是「除它以外別無選擇」。）

★ 為何 step↔CTU 只能是 1.0：canonical mean module 的 ``time_conditioned=False``，
三態都以 ``build_cond(17, g_vec, p_vec)`` 建條件向量（cond_17D 無時間維），該路徑
把 ``delta_ctu`` 填為 ``1.0``（``data/normalizer.py::build_cond``）。也就是說，模型每
一步被告知的步長恆為 1.0 CTU。任何其他常數（0.2 / 0.65）與模型自己收到的條件互相
矛盾。實務上 mean module 的一步是不動點迭代而非時間積分，**回報單位一律以 step 為
主**，CTU 僅作標籤。

★ mean 與 turb 是兩種語意，不可共用同一個常數：

    * **mean module**：一步 = 不動點迭代（求 steady mean）。CTU 只是標籤，換算恆為
      :data:`NOMINAL_DELTA_CTU_PER_STEP` = 1.0。
    * **turb module**：一步 = 真實時間積分（瞬態渦街），其步長由訓練資料的取樣間隔
      決定，見 :data:`TURB_DELTA_CTU_PER_STEP`。turb 線報 CTU 有物理意義。

兩者都由本模組具名匯出（見 :data:`__all__`），eval / 腳本檔**不應**再留隱式預設值。
"""
from __future__ import annotations

from typing import Optional

import torch

from ..utils import spatial_ops

__all__ = [
    "NOMINAL_DELTA_CTU_PER_STEP",
    "PROTOCOL_VERSION",
    "TURB_DELTA_CTU_PER_STEP",
    "assert_delta_ctu",
    "chain_handoff",
    "consec_rel_l2",
    "ctu_to_steps",
    "forward_anchor_kwargs",
    "initial_state",
    "initial_state_from_flag",
    "mask_ic_solid",
    "protocol_version_record",
    "resolve_ke_anchor",
    "resolve_ke_anchor_from_model",
    "steps_to_ctu",
]

# ── 協定語意版本 ────────────────────────────────────────────────────────
# ckpt 的 inference_hparams 只記 ke_anchor_mode / gmin / margin，記不到「錨點來源」
# 這個語意版本，載入端因此無法察覺「訓練時的算子」與「現在推論的算子」不同。
#
#   v0（未標記）：訓練期 ke_anchor = 監督目標 GT mean 的 KE；推論期拿不到 → 三態分歧。
#                canonical `main_v4` 屬本版。
#   v1          ：訓練期 ke_anchor = 本次 rollout 起始 state 的 KE（見 resolve_ke_anchor），
#                三態同式。程式碼現走 v1 語意。
#
# ★現況：存檔端已接線。
# `training/checkpoint/identity.py::collect_inference_hparams` 末尾
# `out.update(protocol_version_record())`，而該函式由同檔 `build_identity`
# （寫入 ckpt 的 "identity"."inference_hparams"）與載入端比對函式呼叫。
# 也就是說：**此後新存的 ckpt 會帶 protocol_version，載入端的版本比對分支可達**。
#
# 但辨識力仍有一條硬邊界：本欄位記的是「存檔當時那份程式碼自稱的世代」，不是對權重的
# 反向鑑定。**所有在接線之前存下的 ckpt**（含 canonical `main_v4`）一律無此欄，
# 讀出來是「未記錄」——不得據此推定它是 v0 或 v1。
PROTOCOL_VERSION: str = "v1"

# 三態共同的名目步長（CTU / step）。理由見模組 docstring：這是 build_cond 實際餵給
# 模型的值，不是可調參數。若未來啟用 time_conditioned，必須改為讀 batch 的 delta_ctu，
# 而非在此改常數。
NOMINAL_DELTA_CTU_PER_STEP: float = 1.0

# turb module 的真實時間步長（CTU / step）。與 mean 的名目 1.0 語意不同——turb 一步是
# 時間積分，此值來自訓練資料的取樣間隔，故報 turb 的 CTU 有物理意義。原先散落在
# `eval/turb_long_rollout.py` 當隱式預設（`_DEFAULT_CTU_PER_STEP`），現收於此具名匯出。
TURB_DELTA_CTU_PER_STEP: float = 0.65


def steps_to_ctu(n_steps: int) -> float:
    """step 數 → 名目 CTU（僅供標籤，mean module 一步非時間積分）。

    Args:
        n_steps: rollout 步數。

    Returns:
        名目 CTU 值。
    """
    return float(n_steps) * NOMINAL_DELTA_CTU_PER_STEP


def ctu_to_steps(ctu: float) -> int:
    """名目 CTU → step 數（四捨五入）。

    Args:
        ctu: 名目 CTU 值。

    Returns:
        對應 step 數（至少 1）。
    """
    return max(1, int(round(float(ctu) / NOMINAL_DELTA_CTU_PER_STEP)))


def protocol_version_record() -> dict[str, str]:
    """回傳可併入 ckpt ``inference_hparams`` 的協定版本欄位。

    存檔端已接線——
    ``training/checkpoint/identity.py::collect_inference_hparams`` 會
    ``out.update(protocol_version_record())``。此後新存的 ckpt 帶得到此欄位。

    Returns:
        ``{"protocol_version": PROTOCOL_VERSION}``。由存檔端併入 identity 的
        ``inference_hparams``，載入端比對；**接線前存下的 ckpt** 無此欄位＝無法辨識
        （canonical ``main_v4`` 屬此類），該情形只警告不中止。
    """
    return {"protocol_version": PROTOCOL_VERSION}


def assert_delta_ctu(value: float, *, kind: str = "mean", context: str = "") -> float:
    """核對外部 config 給的 ``delta_ctu`` 與協定常數一致，不一致即報錯。

    部署端（backend / adapter）的 CTU 軸若與訓練/評估用的刻度不同，KE 兩窗這類
    「以 CTU 定窗長」的判準會在不同窗長上做判定——同一顆 ckpt 因此在不同態給出
    不同的收斂步數。

    Args:
        value: config 讀到的 delta_ctu。
        kind: ``"mean"``（比對 :data:`NOMINAL_DELTA_CTU_PER_STEP`）或 ``"turb"``
            （比對 :data:`TURB_DELTA_CTU_PER_STEP`）。
        context: 呼叫端識別字串，僅供錯誤訊息。

    Returns:
        原樣回傳 ``value``（float），方便直接指派。

    Raises:
        ValueError: ``kind`` 不合法，或 ``value`` 與協定常數不一致。
    """
    if kind == "mean":
        expected = NOMINAL_DELTA_CTU_PER_STEP
    elif kind == "turb":
        expected = TURB_DELTA_CTU_PER_STEP
    else:
        raise ValueError(f"未知的 kind={kind!r}；僅支援 'mean' 與 'turb'。")
    v = float(value)
    if abs(v - expected) > 1e-9:
        raise ValueError(
            f"{context or 'delta_ctu'}={v} 與三態協定的 {kind} 常數 {expected} 不一致；"
            f"**同一條線（此處為 {kind}）**的 step<->CTU 換算只能有一套"
            "（mean 與 turb 是兩套、語意不同）。"
        )
    return v


def consec_rel_l2(curr: torch.Tensor, prev: torch.Tensor,
                  fluid_mask: Optional[torch.Tensor] = None) -> float:
    """相鄰兩步 physics 的相對 L2（停步判準候選，**不受 ke_anchor 影響**）。

    部署端現行停步判準 ``_two_window_ke`` 與 ``ke_anchor`` 是同一個泛函（KE），
    修正器一開就同時動到「被判的量」與「判準」。本函式提供一個
    與 KE 正交的量：``||x_t − x_{t−1}||₂ / ||x_t||₂``（fluid 區）。

    ★ 目前只作**記錄**用（evidence 並排比較）。

    Args:
        curr: 本步 physics ``[B, physics_dim, H, W]``（norm space）。
        prev: 上一步 physics，同形狀。
        fluid_mask: 流體遮罩 ``[B, 1, H, W]``；``None`` = 全域（不遮罩）。

    Returns:
        相對 L2 純量（float）。
    """
    d = (curr - prev).float()
    c = curr.float()
    if fluid_mask is not None:
        fm = fluid_mask.float()
        d = d * fm
        c = c * fm
    num = float(torch.linalg.vector_norm(d).item())
    den = float(torch.linalg.vector_norm(c).item())
    return num / max(den, 1e-12)


def _masked_ke(physics: torch.Tensor, fluid_mask: torch.Tensor) -> torch.Tensor:
    """fluid 區 momentum KE 均值 ``[B]``（norm space；與 arch ceiling 定義同式）。

    Args:
        physics: physics 通道 ``[B, physics_dim, H, W]``（norm space）。
        fluid_mask: 流體遮罩 ``[B, 1, H, W]``，1=流體。

    Returns:
        ``[B]`` 每 sample 的 fluid-masked KE。
    """
    fm = fluid_mask.float()
    denom = fm.sum(dim=(1, 2, 3)).clamp_min(1.0)
    jx = physics[:, 3:4].float()
    jy = physics[:, 5:6].float()
    return ((jx ** 2 + jy ** 2) * fm).sum(dim=(1, 2, 3)) / denom


def _anchor_from_model_cfg(
    model_cfg,
    state: torch.Tensor,
    fluid_mask: torch.Tensor,
) -> Optional[torch.Tensor]:
    """本模組內錨點解析的單一實作（吃 ``TESModelConfig``）。

    :func:`resolve_ke_anchor` 與 :func:`resolve_ke_anchor_from_model` 都委派本函式，差別
    只在 ``TESModelConfig`` 從哪裡取得。（模型內部另有自己的 ceiling 計算——
    ``h3t_streamcollide`` 的 ``_ke_of``——那是「被壓的量」不是「錨點」，兩者不同。）

    Args:
        model_cfg: ``TESModelConfig``（需有 ``static_dim`` / ``physics_dim`` /
            ``transport.ke_anchor_mode``；一律直接存取，缺欄位即報錯）。
        state: 本次 rollout 的起始完整 state ``[B, state_dim, H, W]``。
        fluid_mask: 流體遮罩 ``[B, 1, H, W]``。

    Returns:
        ``[B]`` 錨點張量，或 ``None``（``mode="off"``）。

    Raises:
        ValueError: ``ke_anchor_mode`` 不在 ``{"off", "ceiling"}``。
    """
    mode = str(model_cfg.transport.ke_anchor_mode)
    if mode == "off":
        return None
    if mode != "ceiling":
        raise ValueError(
            f"未知的 ke_anchor_mode={mode!r}；僅支援 'off' 與 'ceiling'。"
        )
    sd = int(model_cfg.static_dim)
    pd = int(model_cfg.physics_dim)
    with torch.no_grad():
        return _masked_ke(state[:, sd:sd + pd], fluid_mask).detach()


def resolve_ke_anchor(
    cfg,
    state: torch.Tensor,
    fluid_mask: torch.Tensor,
) -> Optional[torch.Tensor]:
    """依 train config 的 ``ke_anchor_mode`` 產生本次 rollout 的 KE ceiling 錨點。

    ``mode="off"``（協定建議值）回 ``None``，模型端 ceiling 分支完全 bypass；
    ``mode="ceiling"`` 回本次 rollout **起始 state** 的 fluid-masked KE
    ——推論期拿得到、不跨 rollout 累積、是輸入的純函式。

    Args:
        cfg: TES train config（讀 ``cfg.model``）。
        state: 本次 rollout 的起始完整 state ``[B, state_dim, H, W]``。
        fluid_mask: 流體遮罩 ``[B, 1, H, W]``。

    Returns:
        ``[B]`` 錨點張量，或 ``None``（``mode="off"``）。

    Raises:
        ValueError: ``ke_anchor_mode`` 不在 ``{"off", "ceiling"}``。
    """
    return _anchor_from_model_cfg(cfg.model, state, fluid_mask)


def resolve_ke_anchor_from_model(
    model,
    state: torch.Tensor,
    fluid_mask: torch.Tensor,
) -> Optional[torch.Tensor]:
    """同 :func:`resolve_ke_anchor`，但模式改由 **model 自身的 ModelConfig** 取得。

    供拿不到 train config 的呼叫端使用（``eval/`` 底下數支泛用 rollout 函式的簽章
    只收 ``model`` 與張量，硬塞 cfg 會擴散到所有呼叫端）。``model._cfg`` 就是
    ``build_tes_model`` 當初用的那份 ``TESModelConfig``，故與 :func:`resolve_ke_anchor`
    讀到的是同一組欄位，不是另一套預設值。

    Args:
        model: 由 ``models.factory.build_tes_model`` 建出的模型（具 ``_cfg``）。
        state: 本次 rollout 的起始完整 state ``[B, state_dim, H, W]``。
        fluid_mask: 流體遮罩 ``[B, 1, H, W]``。

    Returns:
        ``[B]`` 錨點張量，或 ``None``（``mode="off"``）。
    """
    return _anchor_from_model_cfg(model._cfg, state, fluid_mask)


def forward_anchor_kwargs(
    model,
    state: torch.Tensor,
    fluid_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """回傳可直接 ``**`` 展開進 ``model(...)`` 的錨點關鍵字引數。

    ★為何不直接寫 ``ke_anchor=...``：TES 各 arch 的 ``forward`` 簽章**不一致**——
    ``h3t_streamcollide`` / ``operator_gated_streamcollide`` 收 ``ke_anchor``；
    ``flat``（``assembly/flat_nca.py``）與 ``turb_perception_a`` 既沒有這個參數也沒有
    ``**kwargs``，硬傳會 ``TypeError``。泛用 rollout 函式（``eval/`` 底下數支）跑的
    arch 由 config 決定，因此只能「有錨點才傳」。

    只有 ``ke_anchor_mode='ceiling'`` 會回非空 dict，故此寫法在 ``off`` 下與完全不傳
    逐位元等價。現行設 ``ceiling`` 的 config，其 ``arch`` 皆為
    ``operator_gated_streamcollide``（``models/factory.py::_build_arch`` 把它建成
    ``h3t_streamcollide`` 那支類別，其 ``forward`` 收 ``ke_anchor``）。**但這只是現況、
    不是強制約束**——若有人把 ``ceiling`` 配到 ``flat`` 之類不收此參數的 arch，這裡仍會
    傳出去而 ``TypeError``。

    Args:
        model: 由 ``models.factory.build_tes_model`` 建出的模型（具 ``_cfg``）。
        state: 本次 rollout 的起始完整 state ``[B, state_dim, H, W]``。
        fluid_mask: 流體遮罩 ``[B, 1, H, W]``。

    Returns:
        ``{}``（``mode='off'``）或 ``{"ke_anchor": <[B] 張量>}``。
    """
    anchor = resolve_ke_anchor_from_model(model, state, fluid_mask)
    return {} if anchor is None else {"ke_anchor": anchor}


def mask_ic_solid(
    ic_physics: torch.Tensor,
    fluid_mask: torch.Tensor,
) -> torch.Tensor:
    """把 IC physics 的固體區歸零（``apply_ic_mask`` 的動作本體）。

    推論／評估側的共用實作：:func:`initial_state_from_flag`、
    ``training/trainer`` 的 val／vis 路徑呼叫本函式，
    不再各自寫 ``masked_fill``。**只做動作，不決定政策**——要不要套由呼叫端依
    ``cfg.apply_ic_mask`` 決定。冪等：對已遮罩過的 IC 再套一次結果不變。

    Args:
        ic_physics: 初始 physics ``[B, physics_dim, H, W]``。
        fluid_mask: 流體遮罩 ``[B, 1, H, W]``，>0.5 為流體。

    Returns:
        固體區已歸零的新張量（不就地改動輸入）。
    """
    solid = fluid_mask <= 0.5
    return ic_physics.masked_fill(solid.expand_as(ic_physics), 0.0)


def initial_state_from_flag(
    model,
    apply_ic_mask: bool,
    static: torch.Tensor,
    ic_physics: Optional[torch.Tensor],
    fluid_mask: torch.Tensor,
) -> torch.Tensor:
    """同 :func:`initial_state`，但政策以顯式布林傳入（拿不到 cfg 的呼叫端用）。

    ``eval/`` 底下數支泛用 rollout 函式（如 ``evaluate_long_rollout``）的簽章只收
    ``model`` 與張量、外加一個 ``apply_ic_mask`` 布林，硬塞 cfg 會擴散到所有呼叫端。
    這些呼叫端的布林值本身仍應來自 ``cfg.apply_ic_mask``（trainer 與各 ``main()``
    都是這樣餵的），本函式只是不再要求它們持有 cfg 物件。

    Args:
        model: 具 ``init_state`` 的 TES 模型。
        apply_ic_mask: 是否清固體區 IC。
        static: static 通道 ``[B, static_dim, H, W]``。
        ic_physics: 初始 physics ``[B, physics_dim, H, W]``；``None`` = 冷啟零場。
        fluid_mask: 流體遮罩 ``[B, 1, H, W]``。

    Returns:
        起始 state ``[B, state_dim, H, W]``（hidden 為零，冷啟語意）。
    """
    if ic_physics is not None and apply_ic_mask:
        ic_physics = mask_ic_solid(ic_physics, fluid_mask)
    return model.init_state(static, ic_physics)


def initial_state(
    model,
    cfg,
    static: torch.Tensor,
    ic_physics: Optional[torch.Tensor],
    fluid_mask: torch.Tensor,
) -> torch.Tensor:
    """建 stage 0 起始 state：依 ``cfg.apply_ic_mask`` 決定是否清固體區 IC。

    mean 線三態共用入口。``cfg.apply_ic_mask`` 一律直接存取——以
    ``getattr(cfg, "apply_ic_mask", False)`` 取值會使預設值與 dataclass 預設 ``True``
    相反，欄位改名即靜默切成與訓練相反的 IC 協定。

    Args:
        model: 具 ``init_state`` 的 TES 模型。
        cfg: TES train config（需有 ``apply_ic_mask``）。
        static: static 通道 ``[B, static_dim, H, W]``。
        ic_physics: 初始 physics ``[B, physics_dim, H, W]``；``None`` = 冷啟零場。
        fluid_mask: 流體遮罩 ``[B, 1, H, W]``。

    Returns:
        起始 state ``[B, state_dim, H, W]``（hidden 為零，冷啟語意）。
    """
    return initial_state_from_flag(
        model, bool(cfg.apply_ic_mask), static, ic_physics, fluid_mask
    )


def chain_handoff(
    model,
    cfg,
    prev_state: torch.Tensor,
    static_new: torch.Tensor,
    fluid_new: torch.Tensor,
    fluid_prev: Optional[torch.Tensor],
) -> torch.Tensor:
    """chain stage 交接：上一 stage 完整 state → 新幾何的起始 state。

    mean 線三態共用實作。**hidden 在重置區之外保留**——重置區內 hidden 與 physics 一起
    歸零（``model.apply_ic_mask`` 對 ``static_dim`` 之後的所有動態通道一視同仁，見
    ``models/assembly/h_base.py::apply_ic_mask``）。對比舊式以
    ``model.init_state(static_new, prev_pred)`` 交接的作法，那是把 hidden **全域**歸零，
    而訓練 ``run_chain`` 與部署 adapter 都只在重置區歸零。

    重置區採 ``handoff_reset_mask``（新固體 ∪ 舊固體 ∪ 變動邊界膨脹），
    ``mean_chain_handoff_dilate_px=0`` 時退化為僅新固體。

    Args:
        model: 具 ``apply_ic_mask`` 的 TES 模型。
        cfg: TES train config（需有 ``mean_chain_handoff_dilate_px``）。
        prev_state: 上一 stage 的完整 state ``[B, state_dim, H, W]``。
        static_new: 新幾何 static ``[B, static_dim, H, W]``。
        fluid_new: 新幾何流體遮罩 ``[B, 1, H, W]``。
        fluid_prev: 上一 stage 流體遮罩 ``[B, 1, H, W]``；``None`` = 無有效舊幾何
            （僅以新固體重置）。

    Returns:
        新 stage 的起始 state ``[B, state_dim, H, W]``。
    """
    reset = spatial_ops.handoff_reset_mask(
        fluid_new, fluid_prev, int(cfg.mean_chain_handoff_dilate_px)
    )
    state = model.apply_ic_mask(prev_state, reset).clone()
    state[:, : int(cfg.model.static_dim)] = static_new
    return state
