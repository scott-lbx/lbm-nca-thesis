"""Checkpoint 身分（identity）欄位：建立、讀取、不一致偵測。

動機
------------------------------------------------------------------
原本 checkpoint 只存 ``{step, model, optimizer, scheduler, scaler}`` 五個 key，
既無程式碼版本（commit）、無訓練 config 快照、也無反正規化常數。同一顆 ckpt 若被
以不同的推論敏感超參數（如 ``psi_lowpass_sigma``）重新評估，會得到不同結果，
而載入端沒有任何程式層比對可以攔截這種 config 與 ckpt 不匹配的情形。

``psi_lowpass_sigma`` / ``ke_anchor_mode`` 等是 **inference-time 算子參數**
（``h3t_streamcollide.py::_curl_psi`` / KE ceiling），它們不在 ``state_dict`` 裡，
換一份 yaml 就換一套推論行為，而載入端沒有任何程式層比對——本模組即補上該比對。

設計原則
------------------------------------------------------------------
1. **內嵌而非引用**：反正規化常數（``FieldNormalizer`` 全部 buffer）直接寫進 ckpt，
   不只存 stats 檔路徑。正規化空間是「呼叫端 zarr 清單」的函數（移除單一 1.43%
   佔比來源即使全域尺度改變 32%），只存路徑無法還原當時的尺度。
2. **完整 model config 快照**：``cfg.model`` 整份 asdict 寫入，因為模型完全由
   ``build_tes_model(cfg.model)`` 決定；如此可做逐欄 diff 而非只有一個 hash 通/不通。
3. **區分「沒有記錄」與「記錄為某值」**：舊 ckpt 無 identity → ``extract_identity``
   回傳 ``None``（而非填預設值）；identity 存在但缺某欄（schema 較舊）→ 該欄狀態為
   ``not_recorded``，不與「記錄為預設值」混為一談。
4. **禁 fallback config 存取**：cfg 欄位一律以「無 default」的方式讀——動態欄名走
   :func:`_resolve_dotted` 的 2-arg ``getattr``，固定欄名（``cfg.seed`` / ``cfg.stats_policy``
   等）直接屬性存取；兩者缺欄位皆為 ``AttributeError``（專案鐵則）。
   ⚠ **例外只有 ``config_source`` 一項**（全檔 ``vars(cfg).get``
   共 2 處，分別在 :func:`build_identity` 與 :func:`build_identity_best_effort`，皆讀
   ``_source_yaml``；其餘 ``.get`` 全部作用在**已讀回的 identity dict** 上，不是 cfg）。
   ``_source_yaml`` 是「來源 metadata」不是 config 參數，缺失代表「不知道來源」，
   故記為 ``None`` 而非報錯，理由就地註記於 :func:`build_identity`。

Ckpt 內結構（key = ``"identity"``）::

    {
      # ★此值即 :data:`IDENTITY_SCHEMA_VERSION`；該常數 bump 時本行須同步（已上斷言）。
      "schema_version": 2,
      "created_at": "2026-07-27T12:34:56",
      "git": {"commit": ..., "branch": ..., "dirty": bool, "workspace_commit": ...},
      "seed": {"seed": int},
      "config": {"sha256": ..., "model_sha256": ..., "model": {...}, "config_source": ...},
      "inference_hparams": {"apply_ic_mask": ..., "model.psi_lowpass_sigma": ...,
                            "protocol_version": "v1", ...},
      "denorm": {"sha256": ..., "stats_ref": {...}, "field_normalizer": {...},
                 "cond_normalizer": {...}},
      "runtime": {"torch": ..., "python": ..., "n_params": ...}
    }
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import logging
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from .....paths import REPO_ROOT as _REPO_ROOT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 常數
# ---------------------------------------------------------------------------

#: ckpt dict 內存放身分區塊的 key。
IDENTITY_KEY = "identity"

#: identity 區塊 schema 版本；新增欄位時 +1（載入端據此區分 not_recorded）。
#:
#: * v1 — 初版（git / seed / config / inference_hparams / denorm / runtime）。
#: * v2 — ``inference_hparams`` 新增 ``protocol_version``
#:   （見 :func:`collect_inference_hparams`）。v1 ckpt 讀到此欄為
#:   :data:`NOT_RECORDED`，**不等於** ``"v0"``——只代表「當時沒記」。
IDENTITY_SCHEMA_VERSION = 2

#: ★旁掛（sidecar）身分檔的副檔名：``{ckpt}.identity.json``。
#:
#: 用於「ckpt 已存在且**不得回寫**」的情形（回寫 .pth 會改變其 sha256，使
#: 既有的 ``ckpt_sha8`` 證據對照全數失效）。
#: sidecar 是**次等**身分來源：它與權重可被分開搬移，故讀取端一律顯式出聲
#: （見 :func:`extract_identity` 的 ``allow_sidecar``），⛔ 不得靜默等同於內嵌身分。
IDENTITY_SIDECAR_SUFFIX = ".identity.json"

#: repo 根目錄（由 :mod:`lbmnca.paths` 統一決定，git 溯源以此為 cwd）。
REPO_ROOT = _REPO_ROOT

#: repo 的上一層目錄。若上層本身不是 git repo，``workspace_commit`` 為 None。
WORKSPACE_ROOT = REPO_ROOT.parent

#: ★推論敏感的 **TESTrainConfig 頂層** 欄位。
#:
#: 判準：該欄位若在推論時與訓練時取值不同，模型輸出即改變（或反正規化空間改變），
#: 而 ``model.state_dict()`` 完全看不出差異。
INFERENCE_SENSITIVE_TRAIN_FIELDS: tuple[str, ...] = (
    # ── rollout / IC 協定（eval 路徑直接讀）──
    "apply_ic_mask",          # warm-start IC 是否對 solid 歸零（long_rollout.py:341）
    "time_conditioned",       # 是否傳 per-step delta_ctu_seq
    "per_step_delta_ctu",     # validator/cond 組裝：逐步 delta_ctu
    "use_amp",                # autocast 開關（數值路徑不同）
    # ── 正規化空間來源（決定 denorm 常數）──
    # static 通道變換（x2_/x3_/zero_static_channels）與
    #   `stats_path`／`zarr_path`（單路徑舊欄位）已隨 v3 一併移除，故不再列入。
    "stats_policy",
    "stats_paths",
    "zarr_paths",
    "val_zarr_paths",
    "dataset_names",
    # ── 模組選擇 ──
    "mean_module_enabled",
    "mean_ablation_arm",
    # ── POD 基底 ──
    "pod_basis_path",
    "pod_modes",
)

#: ★推論敏感的 **cfg.model** 欄位（dotted path，相對於 ``cfg.model``）。
#:
#: ``cfg.model`` 整份已快照於 ``config.model``，此清單是「必比對且必單獨列出」的子集：
#: 這些是曾經/可能造成 train-deploy 錯配的 inference-time 算子參數。
INFERENCE_SENSITIVE_MODEL_FIELDS: tuple[str, ...] = (
    "arch",
    "state_dim",
    "static_dim",
    "physics_dim",
    "hidden_dim",
    "cond_dim",
    "g_vec_dim",
    "gvec_mode",
    "nca_channels",
    "update_clip",
    "rate_mode",
    # psi low-pass 是 inference-time 固定 Gaussian（_curl_psi），必須列入敏感清單
    "psi_lowpass_sigma",
    "psi_curl_momentum",
    "d1_surgical",
    "stoch_update_rate",       # deploy 全同步；train-only Bernoulli
    "nca_spectral_update",
    "nca_spectral_modes",
    "nca_msp_scales",
    "mean_film",
    # ★KE anchor ceiling（推論每步壓幅）
    "transport.ke_anchor_mode",
    "transport.ke_anchor_gmin",
    "transport.ke_anchor_margin",
    "transport.ke_rescale",
    "transport.ke_rescale_gmax",
    "transport.ke_rescale_gmin",
    "transport.ke_rescale_start_ctu",
    "transport.skip",
    "transport.warp_mode",
    "transport.warp_ratio",
    "transport.disp_max_px",
    "transport.disp_ctu_cap",
    "transport.noise_sigma",
    "transport.production_weight",
    "transport.prior_mode",
    "transport.gate_mode",
    "transport.use_velocity_disp",
    "transport.use_create",
    "multiscale.upsample_antialias_sigma",
    "mrp.enabled",
)


class _NotRecorded:
    """哨兵：代表「此 ckpt 的 identity 未記錄該欄位」，與「記錄為某值」嚴格區分。"""

    _instance: Optional["_NotRecorded"] = None

    def __new__(cls) -> "_NotRecorded":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<NOT_RECORDED>"

    def __bool__(self) -> bool:
        return False


#: 單例哨兵（見 :class:`_NotRecorded`）。
NOT_RECORDED = _NotRecorded()


# ---------------------------------------------------------------------------
# 內部工具
# ---------------------------------------------------------------------------

def _jsonable(obj: Any) -> Any:
    """將任意物件轉為 JSON 原生型別（供 ``torch.save`` weights_only 安全序列化）。

    ``torch.load(weights_only=True)`` 只允許 dict/list/tuple/str/int/float/bool/None，
    因此 identity 區塊內不得出現 ``Path`` / ``numpy`` / dataclass 物件。

    Args:
        obj: 待轉換物件。

    Returns:
        僅由 JSON 原生型別構成的等價結構。
    """
    return json.loads(json.dumps(obj, default=str, ensure_ascii=False))


def _canonical_json(obj: Any) -> str:
    """產生穩定（key 排序、無空白）的 JSON 字串，供 hash 使用。

    Args:
        obj: 待序列化物件。

    Returns:
        canonical JSON 字串。
    """
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=False
    )


def sha256_of(obj: Any) -> str:
    """對任意可序列化物件計算 canonical JSON 的 SHA-256。

    Args:
        obj: 待雜湊物件。

    Returns:
        64 字元十六進位 SHA-256 字串。
    """
    return hashlib.sha256(_canonical_json(obj).encode("utf-8")).hexdigest()


def sha256_of_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """計算檔案內容的 SHA-256（供 ckpt 檔完整性核對，取代 byte-size 比對）。

    Args:
        path: 檔案路徑。
        chunk: 每次讀取的位元組數。

    Returns:
        64 字元十六進位 SHA-256 字串。
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _git(repo: Path, *args: str) -> Optional[str]:
    """在 ``repo`` 執行 git 子命令並回傳 stdout；失敗回傳 None（不拋例外）。

    Args:
        repo: git repo 根目錄。
        *args: git 子命令參數。

    Returns:
        stdout 去尾空白後的字串；git 不可用或非 repo 時為 None。
    """
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


_GIT_CACHE: Optional[dict[str, Any]] = None


def git_info(refresh: bool = False) -> dict[str, Any]:
    """收集本檔所在 repo（與上層 workspace）當前 git 狀態。

    結果於 process 內快取：訓練期間 HEAD 不應變動，且避免每次存 ckpt 都 fork git。

    Args:
        refresh: True 時忽略快取重新查詢。

    Returns:
        ``{"commit", "branch", "dirty", "dirty_files", "workspace_commit",
        "available"}``；git 不可用時 ``available=False`` 且 commit 為 None
        （**不填假值**，讓載入端知道「查不到」而非「等於某值」）。
    """
    global _GIT_CACHE
    if _GIT_CACHE is not None and not refresh:
        return dict(_GIT_CACHE)

    commit = _git(REPO_ROOT, "rev-parse", "HEAD")
    branch = _git(REPO_ROOT, "rev-parse", "--abbrev-ref", "HEAD")
    status = _git(REPO_ROOT, "status", "--porcelain")
    dirty_files: list[str] = (
        [ln[3:] for ln in status.splitlines() if ln.strip()] if status is not None else []
    )
    info: dict[str, Any] = {
        "available": commit is not None,
        "repo": "lbm-nca-thesis",
        "commit": commit,
        "branch": branch,
        "dirty": bool(dirty_files) if status is not None else None,
        "dirty_files": dirty_files[:50],
        "workspace_commit": _git(WORKSPACE_ROOT, "rev-parse", "HEAD"),
    }
    _GIT_CACHE = info
    return dict(info)


def _resolve_dotted(obj: Any, dotted: str) -> Any:
    """依 dotted path 逐層取屬性（2-arg getattr，**無 default**）。

    刻意不帶 default：缺欄位即 ``AttributeError``，符合專案「禁 config fallback」鐵則
    ——寧可炸掉也不要靜默用預設值假裝有記錄。

    Args:
        obj: 起始物件。
        dotted: 以 ``.`` 分隔的屬性路徑。

    Returns:
        解析出的屬性值。

    Raises:
        AttributeError: 路徑上任一層屬性不存在。
    """
    cur = obj
    for part in dotted.split("."):
        cur = getattr(cur, part)
    return cur


# ---------------------------------------------------------------------------
# 建立 identity
# ---------------------------------------------------------------------------

def collect_inference_hparams(cfg: Any) -> dict[str, Any]:
    """蒐集 :data:`INFERENCE_SENSITIVE_TRAIN_FIELDS` 與
    :data:`INFERENCE_SENSITIVE_MODEL_FIELDS` 兩份清單所列超參數的當前取值，
    外加 ``protocol_version``。

    頂層欄位以原名為 key；``cfg.model`` 下的欄位以 ``model.`` 前綴的 dotted key。

    ★``protocol_version``
    ------------------------------------------------------------------
    三態推論協定的**語意版本**（``lbmnca.nca.tes.inference.protocol.PROTOCOL_VERSION``）
    不是 config 欄位，也不在 ``state_dict`` 裡——它記的是「訓練當時的 ``ke_anchor``
    錨點取自哪裡」這種算子語意。v0（未標記）取監督目標 GT mean 的 KE，v1 取本次
    rollout 起始 state 的 KE；兩者對同一顆權重會給出不同輸出，而舊 identity 完全
    看不出來。此欄接線後，載入端才有辦法把「未記錄」與「記錄為某世代」分開。

    ⚠ **限度**：本欄記的是「存這顆 ckpt 的那份程式碼自稱的協定世代」，不是對權重的
    反向鑑定。既有 ckpt 一律無此欄，讀出來是 :data:`NOT_RECORDED`——**不得**據此
    推定它是 v0 或 v1。

    Args:
        cfg: ``TESTrainConfig``（或具相同欄位的 config 物件）。

    Returns:
        ``{key: value}``，值均為 JSON 原生型別。

    Raises:
        AttributeError: config 缺少清單中的欄位（刻意不 fallback）。
    """
    # 函式內 import：``protocol`` 依賴 torch，而本模組其餘部分刻意保持無 torch 相依
    # （離線盤點工具 ``tools/l00b_*`` 只 import 本模組的常數與純 JSON 函式）。
    from ...inference.protocol import protocol_version_record

    out: dict[str, Any] = {}
    for name in INFERENCE_SENSITIVE_TRAIN_FIELDS:
        out[name] = _jsonable(_resolve_dotted(cfg, name))
    for name in INFERENCE_SENSITIVE_MODEL_FIELDS:
        out[f"model.{name}"] = _jsonable(_resolve_dotted(cfg.model, name))
    out.update(protocol_version_record())
    return out


def field_normalizer_snapshot(field_normalizer: Any) -> dict[str, Any]:
    """把 ``FieldNormalizer`` 的全部 buffer 序列化為反正規化常數快照。

    這是「內嵌反正規化常數」的實作：``FieldNormalizer`` 的 buffer 集合
    （``_mf_mean`` / ``_mf_std`` / ``_up_*`` / ``_sdf_*`` / ``_upstream_*`` /
    ``_px_*`` / ``_py_*`` / ``_cond_*`` / ``_vis_*``）加上 ``is_tanh`` /
    ``is_physics_informed`` 兩個模式旗標，即可完整重建 norm/denorm 映射，
    不需要再回頭讀 zarr 或 stats json。

    Args:
        field_normalizer: ``src.utils.FieldNormalizer.FieldNormalizer`` 實例。

    Returns:
        ``{"is_tanh", "is_physics_informed", "buffers"}``；buffers 為
        ``{buffer_name: float | list[float]}``（已排序，供穩定 hash）。
    """
    sd = field_normalizer.state_dict()
    buffers = {k: sd[k].detach().cpu().tolist() for k in sorted(sd.keys())}
    return {
        "is_tanh": bool(field_normalizer.is_tanh),
        "is_physics_informed": bool(field_normalizer.is_physics_informed),
        "buffers": buffers,
    }


def build_identity(
    cfg: Any,
    *,
    field_normalizer: Any | None,
    cond_normalizer: Any | None,
    model: Any | None = None,
    config_source: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """建立可直接寫入 checkpoint 的 identity 區塊。

    Args:
        cfg: ``TESTrainConfig``（訓練當下的完整設定）。
        field_normalizer: ``FieldNormalizer``；``stats_policy != dataset_global_runtime``
            時可能為 None（會明確記為 ``null`` + ``reason``，非靜默省略）。
        cond_normalizer: ``TESCondNormalizer``；None 時同上處理。
        model: 模型實例，僅用於記參數量；None 則不記。
        config_source: 訓練 yaml 路徑；``None`` 時自動取 ``load_tes_config`` 寫入的
            ``cfg._source_yaml``（非 dataclass 欄位，不影響 config sha256）。
            兩者皆無 → 記為 ``None``（明確表示「未知來源」，非代入假路徑）。
        extra: 額外欄位（合併進 ``extra`` 子區塊）。

    Returns:
        JSON 原生型別構成的 identity dict。
    """
    if config_source is None:
        # vars() dict 查找而非 getattr fallback：這是「來源 metadata」不是 config 參數，
        # 缺失代表「不知道來源」而非「用預設值」。
        _src = vars(cfg).get("_source_yaml")
        config_source = str(_src) if _src else None
    model_dict = _jsonable(dataclasses.asdict(cfg.model))
    full_cfg_dict = _jsonable(dataclasses.asdict(cfg))

    if field_normalizer is not None:
        fn_snap: Any = field_normalizer_snapshot(field_normalizer)
        fn_reason: Any = None
    else:
        fn_snap = None
        fn_reason = (
            f"field_normalizer 未建立（stats_policy={cfg.stats_policy!r}）"
            "；此 ckpt 未內嵌反正規化常數"
        )

    if cond_normalizer is not None:
        cond_snap: Any = _jsonable(cond_normalizer.to_stats_dict())
    else:
        cond_snap = None

    denorm: dict[str, Any] = {
        # 來源標識（stats_ref）：正規化空間是「zarr 清單 + stats 清單」的函數，
        # 單獨存路徑不足以還原，故同時內嵌常數本身（field_normalizer）。
        "stats_ref": {
            "stats_policy": cfg.stats_policy,
            "stats_paths": list(cfg.stats_paths),
            "zarr_paths": [str(z) for z in cfg.zarr_paths],
            "dataset_names": list(cfg.dataset_names),
        },
        "field_normalizer": fn_snap,
        "field_normalizer_absent_reason": fn_reason,
        "cond_normalizer": cond_snap,
    }
    denorm["sha256"] = sha256_of(
        {"fn": fn_snap, "cond": cond_snap, "ref": denorm["stats_ref"]}
    )

    identity: dict[str, Any] = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "git": git_info(),
        "seed": {"seed": int(cfg.seed)},
        "config": {
            "config_source": config_source,
            "output_dir": str(cfg.output_dir),
            "sha256": sha256_of(full_cfg_dict),
            "model_sha256": sha256_of(model_dict),
            "model": model_dict,
        },
        "inference_hparams": collect_inference_hparams(cfg),
        "denorm": denorm,
        "runtime": {
            "torch": _torch_version(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "n_params": (
                int(sum(p.numel() for p in model.parameters() if p.requires_grad))
                if model is not None
                else None
            ),
            # ★RNG 治理快照。缺了它，「同一顆 ckpt 能否重現
            #   同一條 rollout」沒有程式層依據——determinism_mode / 具名串流是否
            #   啟用、cudnn 旗標為何，事後完全無從得知。兩個函式皆唯讀無副作用。
            **_rng_runtime_snapshot(),
        },
    }
    if extra:
        identity["extra"] = _jsonable(extra)
    return identity


def build_identity_best_effort(
    cfg: Any,
    *,
    field_normalizer: Any | None,
    cond_normalizer: Any | None,
    model: Any | None = None,
    config_source: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """對**任意形狀的 config 物件**盡力建立 identity；缺欄位不炸，改記為降級。

    存在理由
    ------------------------------------------------------------------
    :func:`build_identity` 只吃 ``TESTrainConfig``（頂層扁平欄位）。但 turb / seed
    pipeline 用的是 ``TurbTrainConfig``（欄位收在 ``cfg.data`` / ``cfg.optim`` 底下），
    對它呼叫 ``build_identity`` 必然 ``AttributeError``。若直接改那些腳本走
    :func:`~.checkpointer.save_checkpoint`，結果會是「原本能存檔的訓練從此存不出檔」
    —— 這是行為倒退，不是修復。

    因此本函式做的是：**能建完整身分就建完整身分；不能就建一份誠實標記為降級的身分**
    （``degraded=True`` + ``degraded_reason``），供 sidecar 使用。降級身分**不會**通過
    ``_identity_answerability_problems``（它照樣答不出三個問題），所以不會把
    「有 sidecar」誤讀成「身分完備」。

    Args:
        cfg: 任意 config 物件（``TESTrainConfig`` / ``TurbTrainConfig`` / 其他）。
        field_normalizer: ``FieldNormalizer``；可為 None。
        cond_normalizer: ``TESCondNormalizer``；可為 None。
        model: 模型實例（僅記參數量）；可為 None。
        config_source: 訓練 yaml 路徑；None 時同 :func:`build_identity` 的處理。
        extra: 額外欄位。

    Returns:
        identity dict。完整路徑與 :func:`build_identity` 同構；降級路徑另有
        ``degraded=True`` / ``degraded_reason`` / ``config.dump``（整份 config asdict）。
    """
    try:
        identity = build_identity(
            cfg,
            field_normalizer=field_normalizer,
            cond_normalizer=cond_normalizer,
            model=model,
            config_source=config_source,
            extra=extra,
        )
        identity["degraded"] = False
        identity["degraded_reason"] = None
        return identity
    except Exception as exc:  # noqa: BLE001 — 降級路徑刻意吃下並**明說**原因
        reason = (
            f"build_identity 無法套用於此 config（{type(cfg).__name__}）："
            f"{type(exc).__name__}: {exc}"
        )
        logger.warning("[ckpt_identity] 降級身分：%s", reason)

    if dataclasses.is_dataclass(cfg):
        cfg_dump: Any = _jsonable(dataclasses.asdict(cfg))
    else:
        cfg_dump = _jsonable(vars(cfg)) if hasattr(cfg, "__dict__") else str(cfg)

    if config_source is None and hasattr(cfg, "__dict__"):
        _src = vars(cfg).get("_source_yaml")
        config_source = str(_src) if _src else None

    fn_snap = (
        field_normalizer_snapshot(field_normalizer) if field_normalizer is not None else None
    )
    cond_snap = (
        _jsonable(cond_normalizer.to_stats_dict()) if cond_normalizer is not None else None
    )
    denorm: dict[str, Any] = {
        "stats_ref": None,
        "field_normalizer": fn_snap,
        "field_normalizer_absent_reason": (
            None if fn_snap is not None else "降級身分：呼叫端未提供 field_normalizer"
        ),
        "cond_normalizer": cond_snap,
    }
    denorm["sha256"] = sha256_of({"fn": fn_snap, "cond": cond_snap, "ref": None})

    identity = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "degraded": True,
        "degraded_reason": reason,
        "git": git_info(),
        "seed": {},
        "config": {
            "config_source": config_source,
            "sha256": sha256_of(cfg_dump),
            "dump": cfg_dump,
        },
        "inference_hparams": {},
        "denorm": denorm,
        "runtime": {
            "torch": _torch_version(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "n_params": (
                int(sum(p.numel() for p in model.parameters() if p.requires_grad))
                if model is not None
                else None
            ),
            # ★RNG 治理快照。缺了它，「同一顆 ckpt 能否重現
            #   同一條 rollout」沒有程式層依據——determinism_mode / 具名串流是否
            #   啟用、cudnn 旗標為何，事後完全無從得知。兩個函式皆唯讀無副作用。
            **_rng_runtime_snapshot(),
        },
    }
    if extra:
        identity["extra"] = _jsonable(extra)
    return identity


def sidecar_path(ckpt_path: str | Path) -> Path:
    """回傳某 ckpt 的旁掛身分檔路徑（``{ckpt}.identity.json``）。

    Args:
        ckpt_path: ckpt 檔路徑。

    Returns:
        sidecar 路徑（不保證存在）。
    """
    p = Path(ckpt_path)
    return p.with_name(p.name + IDENTITY_SIDECAR_SUFFIX)


def read_identity_sidecar(ckpt_path: str | Path) -> Optional[dict[str, Any]]:
    """讀取 ckpt 的旁掛身分檔；不存在或壞掉回 None（不拋例外）。

    Args:
        ckpt_path: ckpt 檔路徑。

    Returns:
        sidecar 內的 identity 區塊；無檔／格式不符時為 None。
    """
    p = sidecar_path(ckpt_path)
    if not p.is_file():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError) as exc:
        logger.warning("[ckpt_identity] sidecar 讀取失敗：%s（%s）", p, exc)
        return None
    ident = doc.get(IDENTITY_KEY) if isinstance(doc, dict) else None
    if not isinstance(ident, dict):
        logger.warning("[ckpt_identity] sidecar 格式不符（缺 %r）：%s", IDENTITY_KEY, p)
        return None
    return ident


def write_identity_sidecar(
    ckpt_path: str | Path,
    identity: dict[str, Any],
    *,
    overwrite: bool = False,
    ckpt_sha256: str | None = None,
) -> Optional[Path]:
    """把 identity 寫成 ckpt 旁的 ``{ckpt}.identity.json``（⛔ **不改動 .pth 本身**）。

    刻意不回寫 ckpt 的理由：回寫會改變 ``.pth`` 的 sha256，而
    既有證據檔的檔名正是以 ``ckpt_sha8`` 標身分 —— 一旦 sha 變動，
    既有全部證據檔與其 ckpt 的對應關係當場失效。

    Args:
        ckpt_path: ckpt 檔路徑。
        identity: 身分區塊。
        overwrite: 已存在時是否覆寫；False 時直接跳過（回 None）。
        ckpt_sha256: ckpt 檔內容的 SHA-256；None 時自動計算（大檔會花數秒）。

    Returns:
        寫出的 sidecar 路徑；已存在且 ``overwrite=False`` 時為 None。
    """
    out = sidecar_path(ckpt_path)
    if out.exists() and not overwrite:
        logger.info("[ckpt_identity] sidecar 已存在，未覆寫：%s", out)
        return None
    if ckpt_sha256 is None:
        ckpt_sha256 = sha256_of_file(ckpt_path)
    doc = {
        "note": (
            "旁掛身分檔：此 ckpt 的 .pth 內沒有 identity 區塊，且不得回寫"
            "（回寫會改變 ckpt sha256，使既有證據檔的 ckpt_sha8 對照全部失效）。"
            "本檔與權重可被分開搬移，故身分可信度**低於**內嵌 identity。"
        ),
        "ckpt_path": str(ckpt_path),
        "ckpt_sha256": ckpt_sha256,
        "written_at": datetime.datetime.now().isoformat(timespec="seconds"),
        IDENTITY_KEY: _jsonable(identity),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    logger.info("[ckpt_identity] sidecar 已寫出：%s", out)
    return out


def _rng_runtime_snapshot() -> dict[str, Any]:
    """RNG 治理快照：具名串流狀態＋determinism 執行期旗標。

    以函式內 import 取用 ``training.rng_streams``，避免 identity 模組在 import
    階段就相依訓練層（本模組被推論端與離線工具 import）。

    Returns:
        ``{"rng_streams": ..., "determinism_flags": ...}``；模組不可用時
        兩鍵值為 ``None`` 並附 ``rng_snapshot_error``——不靜默省略，否則
        「沒記錄」與「記錄了但沒啟用」在 ckpt 上長得一樣。
    """
    try:
        from ..rng_streams import snapshot_runtime_flags, streams_summary
    except Exception as exc:  # noqa: BLE001 — identity 不得因此寫不出來
        return {"rng_streams": None, "determinism_flags": None,
                "rng_snapshot_error": f"{type(exc).__name__}: {exc}"}
    return {
        "rng_streams": _jsonable(streams_summary()),
        "determinism_flags": _jsonable(snapshot_runtime_flags()),
    }


def _torch_version() -> Optional[str]:
    """回傳 torch 版本字串；未安裝時為 None。

    Returns:
        torch ``__version__`` 或 None。
    """
    mod = sys.modules.get("torch")
    if mod is None:
        return None
    return str(mod.__version__)


# ---------------------------------------------------------------------------
# 讀取 / 比對
# ---------------------------------------------------------------------------

def extract_identity(
    ckpt: dict[str, Any],
    *,
    ckpt_path: str | Path | None = None,
    allow_sidecar: bool = False,
) -> Optional[dict[str, Any]]:
    """從已載入的 ckpt dict 取出 identity 區塊。

    ``allow_sidecar=True`` 時，ckpt 內找不到 identity 才退回讀 ``{ckpt}.identity.json``
    （見 :func:`read_identity_sidecar`）。此退回**一定出聲**（warning）並在回傳值標記
    ``identity_source="sidecar"``，⛔ 不得被讀成「這顆 ckpt 自己答得出三個問題」。

    ★預設 ``False``：落盤斷言（``checkpointer._assert_identity_on_disk``）驗的正是
    「身分有沒有走進 .pth 檔裡」，若預設吃 sidecar，該斷言會被旁邊的 json 蒙混通過。

    Args:
        ckpt: ``torch.load`` 得到的 checkpoint dict。
        ckpt_path: ckpt 路徑；``allow_sidecar=True`` 時必要。
        allow_sidecar: 是否允許退回旁掛身分檔。

    Returns:
        identity dict；**舊 ckpt 無此欄位（且無 sidecar／未允許）時回傳 None**。
    """
    if isinstance(ckpt, dict):
        ident = ckpt.get(IDENTITY_KEY)
        if isinstance(ident, dict):
            return ident
    if not allow_sidecar or ckpt_path is None:
        return None
    side = read_identity_sidecar(ckpt_path)
    if side is None:
        return None
    logger.warning(
        "[ckpt_identity] %s：ckpt 內**無**身分區塊，身分改由旁掛 sidecar 提供"
        "（%s）。sidecar 可與權重分開搬移，可信度低於內嵌身分。",
        Path(ckpt_path).name,
        sidecar_path(ckpt_path).name,
    )
    side = dict(side)
    side["identity_source"] = "sidecar"
    return side


def _recorded(identity: dict[str, Any], key: str) -> Any:
    """從 identity 的 ``inference_hparams`` 取值；未記錄回傳 :data:`NOT_RECORDED`。

    Args:
        identity: identity 區塊。
        key: hparam key。

    Returns:
        記錄值，或 :data:`NOT_RECORDED`。
    """
    hp = identity.get("inference_hparams")
    if not isinstance(hp, dict) or key not in hp:
        return NOT_RECORDED
    return hp[key]


def _deep_diff(
    recorded: Any, current: Any, prefix: str = ""
) -> list[dict[str, Any]]:
    """遞迴比較兩個 dict/值，回傳差異清單。

    Args:
        recorded: ckpt 內記錄的值。
        current: 當前 config 的值。
        prefix: 目前 key 路徑前綴。

    Returns:
        ``[{"key", "recorded", "current"}]``；``recorded`` 為 ``"<NOT_RECORDED>"``
        表示 ckpt 未記錄該欄（新增欄位），與「記錄為某值但不同」可區分。
    """
    diffs: list[dict[str, Any]] = []
    if isinstance(recorded, dict) and isinstance(current, dict):
        for k in sorted(set(recorded) | set(current)):
            sub = f"{prefix}.{k}" if prefix else k
            if k not in recorded:
                diffs.append(
                    {"key": sub, "recorded": repr(NOT_RECORDED), "current": current[k]}
                )
            elif k not in current:
                diffs.append(
                    {"key": sub, "recorded": recorded[k], "current": "<ABSENT_IN_CONFIG>"}
                )
            else:
                diffs.extend(_deep_diff(recorded[k], current[k], sub))
        return diffs
    if recorded != current:
        diffs.append({"key": prefix, "recorded": recorded, "current": current})
    return diffs


def verify_identity(
    ckpt: dict[str, Any],
    cfg: Any,
    *,
    field_normalizer: Any | None = None,
    ckpt_path: str | Path | None = None,
    on_mismatch: str = "warn",
    context: str = "",
    allow_sidecar: bool = False,
) -> dict[str, Any]:
    """比對 ckpt 內記錄的身分與「當前這次要用的 config」，不一致時報警。

    這是防止 ``psi_lowpass 0.8/0.9`` 類事故重演的核心機制：載入端在建模型之後、
    真正推論之前呼叫本函式，即可在程式層攔截「拿 A config 去跑 B ckpt」。

    Args:
        ckpt: ``torch.load`` 得到的 checkpoint dict。
        cfg: 這次推論/續訓要使用的 ``TESTrainConfig``。
        field_normalizer: 這次建立的 ``FieldNormalizer``；提供時額外比對反正規化常數。
        ckpt_path: ckpt 路徑（僅用於訊息）。
        on_mismatch: ``"warn"``（預設，記 warning）或 ``"raise"``（拋 ``ValueError``）。
        context: 呼叫端標識，寫入訊息（如 ``"tes_adapter"``）。
        allow_sidecar: True 時允許以旁掛 ``{ckpt}.identity.json`` 補身分（顯式出聲；
            預設 False，維持「只認 ckpt 內嵌身分」）。

    Returns:
        報告 dict::

            {"has_identity": bool, "mismatches": [...], "not_recorded": [...],
             "n_checked": int, "model_config_diff": [...], "denorm": {...},
             "git": {...}, "summary": str}

    Raises:
        ValueError: ``on_mismatch="raise"`` 且存在推論敏感不一致。
    """
    tagged = f"[ckpt_identity{'/' + context if context else ''}]"
    where = f" ckpt={Path(ckpt_path).name}" if ckpt_path else ""
    identity = extract_identity(ckpt, ckpt_path=ckpt_path, allow_sidecar=allow_sidecar)

    report: dict[str, Any] = {
        "has_identity": identity is not None,
        # "embedded"（ckpt 內）／"sidecar"（旁掛檔）／None（無身分）——三者不可混談。
        "identity_source": (
            None if identity is None else identity.get("identity_source", "embedded")
        ),
        "ckpt_path": str(ckpt_path) if ckpt_path else None,
        "mismatches": [],
        "not_recorded": [],
        "n_checked": 0,
        "model_config_diff": [],
        "denorm": {"status": "unchecked"},
        "git": {},
        "summary": "",
    }

    if identity is None:
        report["summary"] = (
            f"{tagged} [WARN] 此 checkpoint **無身分資訊**（pre-identity ckpt）"
            f"{where}：無法比對 commit / config / 推論敏感超參數 / 反正規化常數。"
            "請以外部記錄（run 目錄的 metrics.json startup event、SSOT 文件）人工確認"
            "使用的 config 正確。"
        )
        logger.warning("%s", report["summary"])
        return report

    # ── 1. 推論敏感超參數逐欄比對 ─────────────────────────────────────
    current = collect_inference_hparams(cfg)
    for key in sorted(current):
        rec = _recorded(identity, key)
        if isinstance(rec, _NotRecorded):
            report["not_recorded"].append(key)
            continue
        report["n_checked"] += 1
        if rec != current[key]:
            report["mismatches"].append(
                {"key": key, "trained_with": rec, "loading_with": current[key]}
            )

    # ── 2. 完整 model config deep diff（捕捉不在 curated 清單上的欄位）──
    rec_model = identity.get("config", {}).get("model")
    if isinstance(rec_model, dict):
        cur_model = _jsonable(dataclasses.asdict(cfg.model))
        report["model_config_diff"] = _deep_diff(rec_model, cur_model)

    # ── 3. 反正規化常數比對 ──────────────────────────────────────────
    rec_denorm = identity.get("denorm")
    if isinstance(rec_denorm, dict):
        rec_fn = rec_denorm.get("field_normalizer")
        if field_normalizer is None:
            report["denorm"] = {
                "status": "unchecked",
                "note": "呼叫端未提供 field_normalizer",
                "recorded_sha256": rec_denorm.get("sha256"),
                "recorded_stats_ref": rec_denorm.get("stats_ref"),
            }
        elif rec_fn is None:
            report["denorm"] = {
                "status": "not_recorded",
                "note": rec_denorm.get("field_normalizer_absent_reason"),
            }
        else:
            cur_fn = field_normalizer_snapshot(field_normalizer)
            same = sha256_of(cur_fn) == sha256_of(rec_fn)
            report["denorm"] = {
                "status": "match" if same else "mismatch",
                "recorded_stats_ref": rec_denorm.get("stats_ref"),
                "diff": [] if same else _deep_diff(rec_fn, cur_fn, "field_normalizer"),
            }

    # ── 4. git 版本（僅提示，不視為錯誤）─────────────────────────────
    rec_git = identity.get("git", {})
    cur_git = git_info()
    report["git"] = {
        "trained_at_commit": rec_git.get("commit"),
        "trained_dirty": rec_git.get("dirty"),
        "current_commit": cur_git.get("commit"),
        "same_commit": (
            rec_git.get("commit") is not None
            and rec_git.get("commit") == cur_git.get("commit")
        ),
    }

    # ── 5. 匯總 + 報警 ───────────────────────────────────────────────
    critical: list[dict[str, Any]] = list(report["mismatches"])
    if report["denorm"].get("status") == "mismatch":
        critical.append(
            {
                "key": "denorm.field_normalizer",
                "trained_with": "<embedded constants>",
                "loading_with": "<rebuilt from current zarr/stats list>",
            }
        )

    if critical:
        lines = [
            f"{tagged} ★★ 推論設定與訓練此 ckpt 時**不一致**{where} — "
            f"{len(critical)} 項不符（這正是 psi_lowpass 0.8/0.9 事故的類型）：",
        ]
        for m in critical:
            lines.append(
                f"    {m['key']}: 訓練值={m['trained_with']!r} ≠ 本次載入值={m['loading_with']!r}"
            )
        if identity.get("config", {}).get("config_source"):
            lines.append(
                f"    訓練此 ckpt 的 config = {identity['config']['config_source']}"
            )
        report["summary"] = "\n".join(lines)
        if on_mismatch == "raise":
            raise ValueError(report["summary"])
        logger.warning("%s", report["summary"])
    else:
        extra_diff = len(report["model_config_diff"])
        report["summary"] = (
            f"{tagged} [OK] 身分比對通過{where}："
            f"{report['n_checked']} 項推論敏感超參數一致，"
            f"denorm={report['denorm'].get('status')}，"
            f"model_config 其餘差異 {extra_diff} 項，"
            f"trained@{(report['git'].get('trained_at_commit') or 'unknown')[:8]}"
        )
        logger.info("%s", report["summary"])
        if report["model_config_diff"]:
            logger.info(
                "%s model_config 非敏感差異：%s",
                tagged,
                [d["key"] for d in report["model_config_diff"]][:20],
            )
    if report["not_recorded"]:
        logger.info(
            "%s 此 ckpt 的 identity schema (v%s) **未記錄** 下列欄位（非「值相同」）：%s",
            tagged,
            identity.get("schema_version"),
            report["not_recorded"],
        )
    return report


def format_identity(identity: Optional[dict[str, Any]], *, verbose: bool = False) -> str:
    """把 identity 區塊排版為人類可讀文字（CLI / log 用）。

    Args:
        identity: identity 區塊；None 表示此 ckpt 無身分資訊。
        verbose: True 時附完整 model config 與反正規化 buffer。

    Returns:
        多行字串。
    """
    if identity is None:
        return (
            "此 checkpoint **無身分資訊**（pre-identity ckpt，只有 "
            "{step, model, optimizer, scheduler, scaler}）。\n"
            "→ 無 commit / config sha / 推論敏感超參數 / 反正規化常數可供比對。"
        )

    git = identity.get("git", {})
    cfgb = identity.get("config", {})
    denorm = identity.get("denorm", {})
    seed = identity.get("seed", {})
    rt = identity.get("runtime", {})

    lines: list[str] = [
        f"schema_version   : {identity.get('schema_version')}",
        f"created_at       : {identity.get('created_at')}",
        f"git.commit       : {git.get('commit')}"
        f"  (branch={git.get('branch')}, dirty={git.get('dirty')})",
        f"git.workspace    : {git.get('workspace_commit')}",
        f"seed             : seed={seed.get('seed')}",
        f"config.source    : {cfgb.get('config_source')}",
        f"config.sha256    : {cfgb.get('sha256')}",
        f"config.model_sha : {cfgb.get('model_sha256')}",
        f"output_dir       : {cfgb.get('output_dir')}",
        f"runtime          : torch={rt.get('torch')} python={rt.get('python')} "
        f"n_params={rt.get('n_params')}",
        f"denorm.sha256    : {denorm.get('sha256')}",
        f"denorm.embedded  : {denorm.get('field_normalizer') is not None}"
        f"  ({denorm.get('field_normalizer_absent_reason') or '常數已內嵌'})",
        "denorm.stats_ref :",
    ]
    for k, v in (denorm.get("stats_ref") or {}).items():
        lines.append(f"    {k} = {v}")
    lines.append("inference_hparams（推論敏感；與訓練不符即改變輸出）：")
    for k, v in sorted((identity.get("inference_hparams") or {}).items()):
        lines.append(f"    {k} = {v!r}")
    if verbose:
        lines.append("config.model（完整快照）：")
        lines.append(json.dumps(cfgb.get("model"), indent=2, ensure_ascii=False))
        lines.append("denorm.field_normalizer（反正規化常數）：")
        lines.append(json.dumps(denorm.get("field_normalizer"), indent=2, ensure_ascii=False))
    return "\n".join(lines)


def log_identity_on_load(
    ckpt: dict[str, Any],
    *,
    ckpt_path: str | Path | None = None,
    context: str = "",
    allow_sidecar: bool = False,
) -> Optional[dict[str, Any]]:
    """載入時把 ckpt 身分摘要寫進 log（無 identity 時明確標示）。

    Args:
        ckpt: checkpoint dict。
        ckpt_path: ckpt 路徑（訊息用）。
        context: 呼叫端標識。
        allow_sidecar: True 時允許以旁掛 ``{ckpt}.identity.json`` 補身分（顯式出聲）。

    Returns:
        identity 區塊或 None。
    """
    tagged = f"[ckpt_identity{'/' + context if context else ''}]"
    name = Path(ckpt_path).name if ckpt_path else "?"
    identity = extract_identity(ckpt, ckpt_path=ckpt_path, allow_sidecar=allow_sidecar)
    if identity is None:
        logger.warning(
            "%s %s：**此 ckpt 無身分資訊**（pre-identity），無法自動比對訓練設定。",
            tagged,
            name,
        )
        return None
    git = identity.get("git", {})
    cfgb = identity.get("config", {})
    logger.info(
        "%s %s：commit=%s dirty=%s seed=%s config_sha=%s model_sha=%s denorm_sha=%s",
        tagged,
        name,
        (git.get("commit") or "unknown")[:12],
        git.get("dirty"),
        (identity.get("seed") or {}).get("seed"),
        (cfgb.get("sha256") or "")[:12],
        (cfgb.get("model_sha256") or "")[:12],
        ((identity.get("denorm") or {}).get("sha256") or "")[:12],
    )
    return identity
