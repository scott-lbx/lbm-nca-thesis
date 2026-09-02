"""early-stop monitor 解析的契約測試（DEEP_AUDIT W4 P0-2）。

守住三件事，每一件都對應本 campaign 實際發生過的缺陷：

1. **解不出即 raise、且錯誤訊息要列出可用 key**。兩支解析器原本在查無 key 時回
   ``None``，呼叫端 ``if x is not None`` 把整段 early stopping 靜默跳過，使
   ``monitor='total'``（預設值，mean 線根本沒有這個 key）這種組態錯誤永遠不可見。
2. **兩份平行實作必須同語義**。W4-B 第一批只修了 ``TESTrainer`` 走的
   ``schedule.select_early_stop_monitor`` 並宣稱「消滅 early_stop 靜默失效」，
   而 ``MeanTrainer``／``TurbTrainer``（``train_mean.py``／``train_turb.py`` 實際
   實例化的類）走的是 ``TrainerBase._select_monitor``——同構缺陷原封不動。
3. **停用時仍寫稽核紀錄**。第一批把 ``as_jsonl_record`` 一起關進 ``enabled``
   分支，等於讓全 repo 的 run 從此不再寫 ``early_stop`` 事件——而那正是唯一能
   事後證明「早停有沒有被啟用／有沒有作用」的紀錄。

再加一道**組態掃描**：raise 的爆炸半徑必須實測，不能靠敘述。W4-B 二輪就是在
docstring 裡寫「實際爆炸半徑為 0」，但當時 7 支 ``turb_redesign`` config 根本
沒有 ``early_stop`` 段落——預設 ``enabled=True`` + ``monitor='total'``，一開跑
就會被新的 raise 打死。那個論證只數了「有寫 early_stop 段的」config。

全部純 CPU，不載 ckpt、不讀 zarr。
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from lbmnca.nca.tes.training.early_stop_dispatch import dispatch_early_stop
from lbmnca.nca.tes.training.schedule import select_early_stop_monitor

#: 只掃 NCA 訓練 config（``configs/nca/``）。合併後 ``configs/`` 底下另有
#: ``configs/bridge/config.yaml``（WebSocket bridge 伺服器設定，非訓練 config，
#: 無 ``early_stop`` 語意），掃到會被誤判為「monitor 解不出」的偽陽性。
CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs" / "nca"

#: ``TESEarlyStopConfig`` 的預設值。config 未寫 ``early_stop`` 段時就是這組。
DEFAULT_ENABLED = True
DEFAULT_MONITOR = "total"

#: mean 線 val **恆會產出**的 key（不含執行期可能因非有限而被聚合器丟掉的）。
#:
#: ★三輪訂正（must_fix）：第一版把 ``ke_ratio``／``ke_converged``／``ke_final``
#: 一併列入，但它們在 KE 曲線退化時為 ``nan``，而 ``aggregate_val_metrics`` 會把
#: 非有限的 key **整個丟掉**——掃描判合法、執行期 KeyError（漏報）。
#: 同時第一版漏了 ``vg_*``（``TESTrainer`` 的視覺門控指標，實際解得出），
#: 造成合法 config 被判違規（誤報）。兩個方向都由 :func:`_monitor_is_resolvable` 收斂。
ALWAYS_PRESENT = frozenset({
    "mean_mse", "mean_rel_l2", "mean_jx_ssim", "mean_mse_2k", "mean_drift_2k",
})

#: 由 VIS-GATE 提供的 monitor 前綴（第三組 metrics）。
VG_PREFIX = "vg_"


def _monitor_is_resolvable(monitor: str) -> bool:
    """判斷一個 monitor 名稱在執行期是否必然解得出。"""
    return monitor in ALWAYS_PRESENT or monitor.startswith(VG_PREFIX)


# ---------------------------------------------------------------------------
# 1. 解不出即 raise，且列出可用 key
# ---------------------------------------------------------------------------

def test_schedule_resolver_raises_with_available_keys() -> None:
    """三組皆無該 key → KeyError，且訊息必須列出三組各自的可用 key。"""
    cold = {"mean_mse": 1.0}
    warm = {"mean_rel_l2": 2.0}
    vg = {"vg_band": 3.0}
    with pytest.raises(KeyError) as ei:
        select_early_stop_monitor("total", cold, warm, vg)
    msg = str(ei.value)
    for token in ("total", "mean_mse", "mean_rel_l2", "vg_band"):
        assert token in msg, f"錯誤訊息未列出 {token!r}，使用者無從得知可用 key"


def test_neither_resolver_returns_none() -> None:
    """兩支解析器都不得回 ``None``——那正是靜默跳過的成因。"""
    with pytest.raises(KeyError):
        select_early_stop_monitor("nope", {}, {}, {})


# ---------------------------------------------------------------------------
# 2. 優先序：cold → warm → vg
# ---------------------------------------------------------------------------

def test_schedule_resolver_priority_cold_first() -> None:
    """同名 key 同時存在時，cold 優先於 warm，warm 優先於 vg。"""
    assert select_early_stop_monitor("k", {"k": 1.0}, {"k": 2.0}, {"k": 3.0}) == 1.0
    assert select_early_stop_monitor("k", {}, {"k": 2.0}, {"k": 3.0}) == 2.0
    assert select_early_stop_monitor("k", {}, {}, {"k": 3.0}) == 3.0


def _iter_configs() -> list[Path]:
    """列出所有非 archive 的 config（訓練實際會載的那些）。"""
    return sorted(
        p
        for p in CONFIG_ROOT.rglob("*.yaml")
        if "archive" not in {q.lower() for q in p.parts}
    )


def test_every_config_either_disables_early_stop_or_names_a_real_metric() -> None:
    """每支 config 必須二選一：明示停用早停，或指定 mean 線真的產得出的 monitor。

    ★這道測試就是 M2 的回歸鎖。缺 ``early_stop`` 段 = 吃預設
    ``enabled=True`` + ``monitor='total'``，而 mean val 指標沒有 ``total``——
    新的 raise 會在第一次驗證炸掉整個 run。二輪的「爆炸半徑為 0」論證漏掉了
    這一類，因為它只統計了「有寫 early_stop 段」的 config。
    """
    offenders: list[str] = []
    for path in _iter_configs():
        try:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        if not isinstance(cfg, dict):
            continue
        es = cfg.get("early_stop")
        enabled = DEFAULT_ENABLED if es is None else es.get("enabled", DEFAULT_ENABLED)
        if not enabled:
            continue
        monitor = DEFAULT_MONITOR if es is None else es.get("monitor", DEFAULT_MONITOR)
        if not _monitor_is_resolvable(monitor):
            rel = path.relative_to(CONFIG_ROOT)
            offenders.append(
                f"{rel}: early_stop {'缺段落' if es is None else '啟用'}"
                f" → monitor={monitor!r} 不是 val 一定產得出的 key"
            )
    assert not offenders, (
        "以下 config 啟用早停但 monitor 解不出，開跑即 KeyError：\n  "
        + "\n  ".join(offenders)
    )


def test_known_set_excludes_metrics_that_can_vanish_at_runtime() -> None:
    """★三輪 must_fix：``known`` 集合必須排除「執行期可能消失」的 key。

    漏報實例：``ke_ratio`` 在 KE 曲線退化時為 ``nan``（``mean_gate_eval``），
    而 ``aggregate_val_metrics`` 會把非有限的 key **整個丟掉**（只留
    ``__n_dropped_samples``／``__n_used_samples``）——於是 monitor 掃描判它合法、
    執行期卻 ``KeyError``。所以「掃描綠 = 不會炸」這個推論本身不成立，
    ``known`` 必須只收恆存在的 key。
    """
    from lbmnca.nca.tes.eval.mean_gate_eval import ke_curve_convergence
    from lbmnca.nca.tes.training.validation.aggregate import aggregate_val_metrics

    degenerate = ke_curve_convergence([1.0, 1.0, 1.0])
    volatile = {k for k, v in degenerate.items() if not math.isfinite(v)}
    assert volatile, "fixture 未造出非有限的 KE 指標，本測試已失去意義"

    out = aggregate_val_metrics(
        [(1, dict(degenerate), {"sq_err": 1.0, "sq_tgt": 1.0, "n_fluid": 1.0})]
    )
    for key in volatile:
        assert key not in out, f"{key} 非有限卻仍出現在聚合結果，測試前提已變"
        assert key not in ALWAYS_PRESENT, (
            f"{key} 執行期會消失，不得列入 ALWAYS_PRESENT（否則掃描漏報）"
        )


def test_vg_prefixed_monitor_is_accepted() -> None:
    """★三輪 must_fix：``vg_*`` 是合法 monitor，不得被掃描誤判為 offender。

    ``TESTrainer`` 會把 VIS-GATE 指標當第三組傳進解析器，``vg_composite``
    實際解得出（``tests/tes/test_m1a_prereqs.py`` 正在斷言這個用法）。
    第一版的 ``known`` 只列 mean gate 指標 → ``vg_composite`` 會被判違規。
    """
    assert _monitor_is_resolvable("vg_composite")
    assert _monitor_is_resolvable("vg_band")
    assert select_early_stop_monitor("vg_composite", {}, {}, {"vg_composite": 0.42}) == 0.42


def test_config_scan_actually_saw_configs() -> None:
    """守住掃描本身——glob 掃到 0 支會讓上面那道測試變成空綠。"""
    found = _iter_configs()
    # ★2026-08-05：configs/ 已重置為 LBM_NCA-1 系列（舊系列在 configs/archive/），
    #   門檻由 >50 降為 >=3。此斷言的用途是「掃描器本身沒壞」，不是「config 很多」。
    assert len(found) >= 3, f"只掃到 {len(found)} 支 config，掃描條件可能壞了"


# ---------------------------------------------------------------------------
# 4. 停用時仍寫 jsonl 稽核紀錄
# ---------------------------------------------------------------------------

class _JsonlSpy:
    """側錄 ``write_jsonl`` 的最小替身。"""

    def __init__(self) -> None:
        self.records: list[dict] = []

    def __call__(self, record: dict, force_flush: bool = False) -> None:
        # 走一次真正的序列化，確保寫進去的東西是合法 JSON
        json.loads(json.dumps(record, default=str, allow_nan=False))
        self.records.append(record)


class _FakeEarlyStop:
    """``TESEarlyStopping`` 的最小替身（只記下被餵了什麼）。"""

    def __init__(self, stop: bool = False) -> None:
        self.stop = stop
        self.seen: list[float] = []
        self.best_state = None

    def update(self, primary, step, state):  # noqa: D102, ANN001
        self.seen.append(primary)
        return self.stop

    def as_jsonl_record(self, step):  # noqa: D102, ANN001
        return {"event": "early_stop", "step": step, "enabled": True}


def _es_cfg(enabled: bool, monitor: str, restore_best: bool = False):
    """造一個 early_stop config 替身。"""
    return SimpleNamespace(
        enabled=enabled, monitor=monitor, restore_best=restore_best
    )


def test_disabled_early_stop_still_writes_audit_record() -> None:
    """``enabled=false`` 時仍必須寫一筆 ``event=early_stop`` 紀錄。

    否則全 repo 的 run 都不再有早停事件，事後無法證明早停有沒有作用。
    ★這裡打的是**生產碼**的 ``dispatch_early_stop``，不是測試檔自建的複製品。
    """
    spy = _JsonlSpy()
    es = _FakeEarlyStop()
    stop = dispatch_early_stop(
        es_cfg=_es_cfg(False, "total"), step=100,
        cold_metrics={}, warm_metrics={}, vg_metrics=None,
        early_stop=es, state_dict_fn=dict, write_jsonl=spy,
    )
    assert stop is False
    assert len(spy.records) == 1
    rec = spy.records[0]
    assert rec["event"] == "early_stop"
    assert rec["enabled"] is False
    assert rec["monitor"] == "total"
    assert es.seen == [], "停用時不該呼叫 early_stop.update"


def test_enabled_early_stop_with_bad_monitor_raises_not_skips() -> None:
    """``enabled=true`` + 解不出 monitor → 必須 raise，不得靜默無紀錄地跳過。

    ★原始缺陷正是「解析器回 None、呼叫端 ``if x is not None`` 靜默跳過」。
    在呼叫端外包 ``try/except KeyError: return`` 也會讓本測試變紅——因為本測試
    執行的就是那條呼叫鏈本身。
    """
    spy = _JsonlSpy()
    with pytest.raises(KeyError):
        dispatch_early_stop(
            es_cfg=_es_cfg(True, "total"), step=100,
            cold_metrics={"mean_mse": 1.0}, warm_metrics={}, vg_metrics=None,
            early_stop=_FakeEarlyStop(), state_dict_fn=dict, write_jsonl=spy,
        )
    assert spy.records == [], "raise 前不該先寫紀錄"


def test_enabled_early_stop_resolves_and_records() -> None:
    """正常路徑：解得出 monitor → 餵給 early_stop 並寫紀錄。"""
    spy = _JsonlSpy()
    es = _FakeEarlyStop(stop=True)
    stop = dispatch_early_stop(
        es_cfg=_es_cfg(True, "mean_rel_l2"), step=500,
        cold_metrics={"mean_rel_l2": 0.42}, warm_metrics={}, vg_metrics=None,
        early_stop=es, state_dict_fn=dict, write_jsonl=spy,
    )
    assert stop is True
    assert es.seen == [0.42]
    assert spy.records[0]["event"] == "early_stop"


def test_dispatch_supports_vg_metrics_for_the_trainer_with_visual_gate() -> None:
    """``TESTrainer`` 路徑會傳 ``vg_*``；共用分派必須支援，不得像舊的第二份實作那樣 KeyError。

    ★舊 ``TrainerBase._select_monitor`` 的簽章根本沒有 ``vg_metrics``，
    卻在註解宣稱與 ``schedule.select_early_stop_monitor``「同語義」。
    """
    spy = _JsonlSpy()
    es = _FakeEarlyStop()
    dispatch_early_stop(
        es_cfg=_es_cfg(True, "vg_composite"), step=1,
        cold_metrics={}, warm_metrics={}, vg_metrics={"vg_composite": 0.42},
        early_stop=es, state_dict_fn=dict, write_jsonl=spy,
    )
    assert es.seen == [0.42]


def test_no_second_monitor_selector_implementation_exists() -> None:
    """全 repo 只能有一份 monitor 解析實作（★平行副本是本 campaign 的復發源）。

    判準：只有 ``schedule.py`` 可以出現「解不出就 raise KeyError」的解析本體。
    ``trainer.py`` 的 ``_select_early_stop_monitor`` 是指向同一支的 staticmethod
    別名（不是第二份實作），允許。
    """
    src_root = (
        Path(__file__).resolve().parents[3] / "src" / "lbmnca" / "nca" / "tes" / "training"
    )
    offenders: list[str] = []
    for path in sorted(src_root.rglob("*.py")):
        if path.name == "schedule.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            body_src = "\n".join(ast.unparse(s) for s in node.body)
            if "early_stop.monitor=" in body_src and "raise KeyError" in body_src:
                offenders.append(f"{path.name}::{node.name}")
    assert not offenders, (
        "出現第二份 monitor 解析實作（應改為委派 schedule.select_early_stop_monitor）："
        f"{offenders}"
    )


def test_both_trainers_delegate_to_the_shared_dispatch() -> None:
    """兩個 trainer 都必須呼叫共用分派，且不得留下自己的分支結構。

    ★缺陷史正是「只修了一邊」。這裡不再比對 ``if not ...enabled:`` 這種
    字串（那種鎖是被測試自己的複製品滿足的），而是斷言兩邊都真的呼叫了
    生產函式，且都不再自建 enabled 分支。
    """
    src_root = (
        Path(__file__).resolve().parents[3] / "src" / "lbmnca" / "nca" / "tes" / "training"
    )
    for name in ("trainer.py", "trainer_base.py"):
        text = (src_root / name).read_text(encoding="utf-8")
        assert "dispatch_early_stop(" in text, f"{name} 未委派共用分派"
        code = "\n".join(ln.split("#")[0] for ln in text.splitlines())
        assert "if not self._cfg.early_stop.enabled:" not in code, (
            f"{name} 仍保有自己的 enabled 分支（應已收進 dispatch_early_stop）"
        )
