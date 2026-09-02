/* app.js — NCA demo 前端全部運算邏輯（幾何編輯／座標轉換／WebSocket 協定／序列播放）。
 *
 * 來源：舊 p5 fake-GH client sketch（2026-07-20 重構移入 demo/）。
 * 設計原則：
 *   1. 本檔是「唯一有邏輯的前端檔」；demo/index.html 只是薄殼（DOM 容器 + 樣式 + 載入）。
 *   2. 所有數值（sim/canvas 尺寸、風力選項、viz 清單、throttle、序列）一律由
 *      demo/demo.config.json 驅動，本檔不寫死任何使用者可調參數。
 *   3. 不用 ES module（無 import/export），純 <script src> + 全域物件 window.NCADemo，
 *      使 file:// 與 http:// 皆可載入。
 *   4. config 缺必要欄位一律 throw（禁止靜默 fallback 預設值）。
 *
 * 兩類 WebSocket 訊息包裝不同（server 端 src/core/server.py::_parse_command 分流）：
 *   analyze  → 頂層扁平 {action:"analyze", buildings, wind_level, mode}
 *   control  → payload 包裹 {action, payload:{...}}（set_viz_mode/seek/set_speed/
 *              save_version/analyze_zarr）
 */
(function (global) {
  'use strict';

  // ── config 嚴格存取：缺欄位直接報錯，不套預設值 ─────────────────────────
  function req(obj, key, where) {
    if (obj === null || typeof obj !== 'object' || !(key in obj)) {
      throw new Error('[NCADemo] config 缺少必要欄位：' + where + '.' + key);
    }
    return obj[key];
  }

  // ── 模組狀態 ─────────────────────────────────────────────────────────────
  let CFG = null;                 // 已驗證的 config 物件
  let SIM_W = 0, SIM_H = 0;       // sim px 座標基準（geometry_ingest 的 1920×1080）
  let CANVAS_W = 0, CANVAS_H = 0; // 畫布 px
  let SX = 1, SYY = 1;            // canvas px per sim px
  let LIVE_THROTTLE_MS = 0;       // 拖曳即時送出節流
  let DEFAULT_WIND = 2;           // wind_levels 中標記 default 的值

  let buildings = [];      // [{cx,cy,w,h}] sim px（一律 canvas top-down 慣例）
  let dragStart = null;    // {x,y} canvas px（畫新建物拖曳起點）
  let dragNow = null;
  let dragMode = null;     // 'draw' | 'move'
  let moveIdx = -1;        // 移動中的建物 index
  let moveOff = null;      // {dx,dy} 游標→建物中心的 canvas px 偏移

  let liveMode = false;    // ON = 每次編輯（畫/移動/刪）自動即時送
  let editMode = 'warm';   // 'warm'(LBM-NCA 接續，本方法) | 'fresh'(每步 cold-start 對照)
  let lastLiveSendMs = 0;

  let ws = null;
  let seq = null;          // 目前選定的序列物件（來自 config.sequences）
  let seqTimers = [];      // 已排程的 setTimeout id
  let seqStartMs = 0;
  let seqCursor = -1;      // 目前 event index（時間軸繪製用）

  let p5i = null;          // p5 instance（instance mode，避免全域函式時序問題）
  const el = {};           // 由 buildUI() 建立的 DOM 參照

  // ── config 驗證 ──────────────────────────────────────────────────────────
  function validateConfig(cfg) {
    if (cfg === null || typeof cfg !== 'object') {
      throw new Error('[NCADemo] config 不是物件');
    }
    req(cfg, 'ws_url', 'config');
    const sim = req(cfg, 'sim', 'config');
    req(sim, 'width', 'config.sim');
    req(sim, 'height', 'config.sim');
    const canvas = req(cfg, 'canvas', 'config');
    req(canvas, 'width', 'config.canvas');
    req(canvas, 'height', 'config.canvas');

    const winds = req(cfg, 'wind_levels', 'config');
    if (!Array.isArray(winds) || winds.length === 0) {
      throw new Error('[NCADemo] config.wind_levels 必須是非空陣列');
    }
    let defaultWind = null;
    winds.forEach(function (w, i) {
      req(w, 'value', 'config.wind_levels[' + i + ']');
      req(w, 'label', 'config.wind_levels[' + i + ']');
      if (w.default === true) { defaultWind = w.value; }   // default 為明確 optional 欄位
    });
    if (defaultWind === null) {
      throw new Error('[NCADemo] config.wind_levels 必須有一項標記 "default": true');
    }

    const vizModes = req(cfg, 'viz_modes', 'config');
    if (!Array.isArray(vizModes) || vizModes.length === 0) {
      throw new Error('[NCADemo] config.viz_modes 必須是非空陣列');
    }
    const defViz = req(cfg, 'default_viz_mode', 'config');
    if (vizModes.indexOf(defViz) < 0) {
      throw new Error('[NCADemo] config.default_viz_mode 不在 config.viz_modes 內：' + defViz);
    }

    const live = req(cfg, 'live', 'config');
    req(live, 'enabled', 'config.live');
    req(live, 'throttle_ms', 'config.live');
    const em = req(live, 'edit_mode', 'config.live');
    if (em !== 'warm' && em !== 'fresh') {
      throw new Error('[NCADemo] config.live.edit_mode 必須是 "warm" 或 "fresh"');
    }

    const seqs = req(cfg, 'sequences', 'config');
    const names = Object.keys(seqs);
    if (names.length === 0) {
      throw new Error('[NCADemo] config.sequences 不可為空');
    }
    names.forEach(function (n) {
      req(seqs[n], 'name', 'config.sequences.' + n);
      const evs = req(seqs[n], 'events', 'config.sequences.' + n);
      if (!Array.isArray(evs)) {
        throw new Error('[NCADemo] config.sequences.' + n + '.events 必須是陣列');
      }
      evs.forEach(function (ev, i) {
        req(ev, 't', 'config.sequences.' + n + '.events[' + i + ']');
        req(ev, 'action', 'config.sequences.' + n + '.events[' + i + ']');
      });
    });
    return defaultWind;
  }

  // ── WebSocket ────────────────────────────────────────────────────────────
  function connectWS() {
    const url = el.url.value.trim();
    try { ws = new WebSocket(url); } catch (e) { logLine('連線例外: ' + e); return; }
    ws.onopen = function () { setStatus(true, '已連線 ' + url); };
    ws.onclose = function () { setStatus(false, '連線關閉'); };
    ws.onerror = function () { setStatus(false, '連線錯誤（main.py 啟了嗎？）'); };
    ws.onmessage = function (ev) {
      try {
        const r = JSON.parse(ev.data);
        logLine('← ' + (r.status || '?') + ' ' + (r.action || '') + ' ' + (r.msg || r.message || ''));
      } catch (e) { logLine('← ' + ev.data); }
    };
  }
  function disconnectWS() { if (ws) { ws.close(); ws = null; } }

  function sendJSON(obj) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      logLine('✗ 未連線，略過: ' + obj.action);
      return false;
    }
    ws.send(JSON.stringify(obj));
    logLine('→ ' + JSON.stringify(obj));
    return true;
  }

  // ── 送指令 ───────────────────────────────────────────────────────────────
  function windLevel() { return parseInt(el.wind.value, 10); }

  // 送出時 y 翻轉：p5 canvas y 向下、GGUI(Taichi) y 向上 → cy' = SIM_H - cy，
  // 使畫布位置 = GGUI 顯示位置（WYSIWYG）。內部 buildings 一律存 canvas(top-down) 慣例。
  function _toSimBuildings(list) {
    return list.map(function (b) {
      return {
        cx: Math.round(b.cx), cy: Math.round(SIM_H - b.cy),
        w: Math.round(b.w), h: Math.round(b.h),
      };
    });
  }

  function sendAnalyze(mode) {
    if (buildings.length === 0) { logLine('✗ 無建物，先畫一棟'); return; }
    sendJSON({
      action: 'analyze',
      buildings: _toSimBuildings(buildings),
      wind_level: windLevel(),
      mode: mode,
    });
  }
  function sendViz(m) { sendJSON({ action: 'set_viz_mode', payload: { viz_mode: m } }); }
  function sendSeek(f) { sendJSON({ action: 'seek', payload: { frame: f } }); }
  function sendSpeed(s) { sendJSON({ action: 'set_speed', payload: { speed: s } }); }
  function sendSave(l) { sendJSON({ action: 'save_version', payload: { label: l } }); }

  function clearBuildings() { buildings = []; }

  // ── 即時 Live / 編輯模式 開關 ────────────────────────────────────────────
  function applyLiveButton() {
    el.liveBtn.textContent = liveMode ? '🔴 即時 Live：ON' : '⚪ 即時 Live：OFF';
    el.liveBtn.style.background = liveMode ? '#c0392b' : '#2a2f3a';
  }
  function toggleLive() {
    liveMode = !liveMode;
    applyLiveButton();
    setArchAttr('live', liveMode);
    logLine('即時 Live ' + (liveMode ? 'ON — 編輯自動送(' + editMode + ')' : 'OFF — 手動按 Send'));
  }
  function applyEditModeButton() {
    el.modeBtn.textContent = editMode === 'warm' ? '編輯模式：warm 接續' : '編輯模式：cold baseline';
    el.modeBtn.style.background = editMode === 'warm' ? '#b9770e' : '#2f6feb';
    el.modeHint.textContent = editMode === 'warm'
      ? 'warm = LBM-NCA 從前次流場接續（本方法，觀察 block 移動→流場重組過程）'
      : 'cold = 每步 baseline 重算 IC（對照，無接續，比較 warm-start 是否有助過程可讀性）';
  }
  function toggleEditMode() {
    editMode = editMode === 'warm' ? 'fresh' : 'warm';
    applyEditModeButton();
    logLine('編輯模式 → ' + editMode);
  }

  // ── 固定序列 ─────────────────────────────────────────────────────────────
  function selectSequence(key) {
    if (!(key in CFG.sequences)) {
      throw new Error('[NCADemo] 序列不存在：' + key);
    }
    seq = CFG.sequences[key];
    el.seqinfo.textContent = seq.name + ' — ' + seq.events.length + ' events';
    logLine('已選序列: ' + key + '（' + seq.name + '）');
  }

  function dispatchSeqEvent(ev, idx) {
    seqCursor = idx;
    const a = ev.action;
    if (a === 'analyze') {
      // 本地預覽同步（序列內 buildings 亦為 canvas 慣例，送出時才翻轉 y）
      buildings = (ev.buildings || []).map(function (b) { return Object.assign({}, b); });
      sendJSON({
        action: 'analyze',
        buildings: _toSimBuildings(ev.buildings || []),
        wind_level: ev.wind_level || seq.wind_level || DEFAULT_WIND,
        mode: ev.mode || 'warm',
      });
    } else if (a === 'analyze_zarr') {
      // 重放真實 dataset 節點（真 mask + graph.json g_vec + zarr mean_flow IC）。
      // root 設 inject_ic:true → IC = zarr nodes/{id}/mean_flow（01 LBM solver 輸出）。
      buildings = [];   // 幾何來自 zarr，清空本地預覽
      sendJSON({
        action: 'analyze_zarr',
        payload: {
          dataset: ev.dataset, case_id: ev.case_id,
          inject_ic: !!ev.inject_ic,
          wind_level: ev.wind_level || seq.wind_level || DEFAULT_WIND,
        },
      });
    } else if (a === 'set_viz_mode') {
      sendViz(ev.viz_mode);
    } else if (a === 'seek') {
      sendSeek(ev.frame);
    } else if (a === 'set_speed') {
      sendSpeed(ev.speed);
    } else if (a === 'save_version') {
      sendSave(ev.label || '');
    } else {
      logLine('✗ 未知序列 action: ' + a);
    }
  }

  function playSequence() {
    if (!seq || !seq.events) { logLine('✗ 未選定序列'); return; }
    stopSequence();
    seqStartMs = p5i.millis();
    seq.events.forEach(function (ev, i) {
      const id = setTimeout(function () { dispatchSeqEvent(ev, i); }, Math.max(0, (ev.t || 0) * 1000));
      seqTimers.push(id);
    });
    const total = Math.max.apply(null, seq.events.map(function (e) { return e.t || 0; }));
    logLine('▶ 播放序列，' + seq.events.length + ' events，時長 ~' + total + 's');
    if (seq.loop) {
      const id = setTimeout(playSequence, (total + 2) * 1000);
      seqTimers.push(id);
    }
  }
  function stopSequence() {
    seqTimers.forEach(clearTimeout);
    seqTimers = [];
    seqCursor = -1;
  }

  // ── 命中測試 ─────────────────────────────────────────────────────────────
  function _hitBuilding(mx, my) {
    // 由上層往下測：回傳建物 index 或 -1。
    for (let i = buildings.length - 1; i >= 0; i--) {
      const b = buildings[i];
      const x0 = (b.cx - b.w / 2) * SX, x1 = (b.cx + b.w / 2) * SX;
      const y0 = (b.cy - b.h / 2) * SYY, y1 = (b.cy + b.h / 2) * SYY;
      if (mx >= x0 && mx <= x1 && my >= y0 && my <= y1) return i;
    }
    return -1;
  }

  // ── p5 sketch（instance mode）────────────────────────────────────────────
  function makeSketch() {
    return function (p) {
      p.setup = function () {
        const c = p.createCanvas(CANVAS_W, CANVAS_H);
        c.parent(el.canvasHolder);
        // 右鍵刪除最上層命中的建物
        c.elt.addEventListener('contextmenu', function (e) {
          e.preventDefault();
          const hit = _hitBuilding(e.offsetX, e.offsetY);
          if (hit >= 0) {
            buildings.splice(hit, 1);
            if (liveMode && buildings.length > 0) sendAnalyze(editMode);
          }
        });
      };

      p.draw = function () {
        p.background('#d7dee6');                 // fluid 淺色
        // 格線
        p.stroke('#c2ccd6'); p.strokeWeight(1);
        for (let x = 0; x <= CANVAS_W; x += CANVAS_W / 8) p.line(x, 0, x, CANVAS_H);
        for (let y = 0; y <= CANVAS_H; y += CANVAS_H / 8) p.line(0, y, CANVAS_W, y);
        // 風向 +X（左緣上游）
        p.stroke('#2f6feb'); p.strokeWeight(2); p.fill('#2f6feb');
        p.line(6, CANVAS_H / 2, 46, CANVAS_H / 2);
        p.triangle(46, CANVAS_H / 2 - 5, 46, CANVAS_H / 2 + 5, 56, CANVAS_H / 2);
        p.noStroke(); p.text('wind +X', 8, CANVAS_H / 2 - 8);

        // 建物 solid 深色；移動中的建物加高亮邊框
        for (let i = 0; i < buildings.length; i++) {
          const b = buildings[i];
          if (i === moveIdx) { p.stroke('#e67e22'); p.strokeWeight(2); }
          else { p.noStroke(); }
          p.fill('#2a2f3a');
          p.rect((b.cx - b.w / 2) * SX, (b.cy - b.h / 2) * SYY, b.w * SX, b.h * SYY);
        }
        // hover 於既有建物上時提示可移動
        if (!dragMode && _hitBuilding(p.mouseX, p.mouseY) >= 0) {
          p.noStroke(); p.fill('#e67e22');
          p.text('拖曳可移動 block（放開自動送 warm）', 8, CANVAS_H - 8);
        }
        // 拖曳預覽
        if (dragStart && dragNow) {
          p.fill(42, 47, 58, 120);
          p.rect(Math.min(dragStart.x, dragNow.x), Math.min(dragStart.y, dragNow.y),
                 Math.abs(dragNow.x - dragStart.x), Math.abs(dragNow.y - dragStart.y));
        }
        // 序列時間軸游標
        if (seqTimers.length > 0 && seq && seq.events) {
          const elapsed = (p.millis() - seqStartMs) / 1000;
          const total = Math.max.apply(null, seq.events.map(function (e) { return e.t || 0; })) + 2;
          const px = Math.min(1, elapsed / total) * CANVAS_W;
          p.stroke('#e67e22'); p.strokeWeight(2); p.line(px, 0, px, 12);
          p.noStroke(); p.fill('#e67e22'); p.text('t=' + elapsed.toFixed(1) + 's', px + 4, 12);
        }
      };

      p.mousePressed = function () {
        if (p.mouseButton !== p.LEFT) return;
        if (p.mouseX < 0 || p.mouseX > CANVAS_W || p.mouseY < 0 || p.mouseY > CANVAS_H) return;
        const hit = _hitBuilding(p.mouseX, p.mouseY);
        if (hit >= 0) {
          // 點在既有建物內 → 移動模式
          dragMode = 'move'; moveIdx = hit;
          const b = buildings[hit];
          moveOff = { dx: b.cx * SX - p.mouseX, dy: b.cy * SYY - p.mouseY };
        } else {
          // 空白處 → 畫新建物
          dragMode = 'draw';
          dragStart = { x: p.mouseX, y: p.mouseY };
          dragNow = { x: p.mouseX, y: p.mouseY };
        }
      };

      p.mouseDragged = function () {
        if (dragMode === 'move' && moveIdx >= 0) {
          const b = buildings[moveIdx];
          // clamp 中心使建物留在 sim 域內
          const hw = b.w / 2, hh = b.h / 2;
          const cx = (p.mouseX + moveOff.dx) / SX, cy = (p.mouseY + moveOff.dy) / SYY;
          b.cx = Math.max(hw, Math.min(SIM_W - hw, cx));
          b.cy = Math.max(hh, Math.min(SIM_H - hh, cy));
          // 即時模式：拖曳中節流連續送（流場隨拖曳更新）
          if (liveMode && p.millis() - lastLiveSendMs > LIVE_THROTTLE_MS) {
            lastLiveSendMs = p.millis();
            sendAnalyze(editMode);
          }
        } else if (dragMode === 'draw' && dragStart) {
          dragNow = { x: p.mouseX, y: p.mouseY };
        }
      };

      p.mouseReleased = function () {
        let edited = false;
        if (dragMode === 'draw' && dragStart && dragNow) {
          const x0 = Math.min(dragStart.x, dragNow.x), x1 = Math.max(dragStart.x, dragNow.x);
          const y0 = Math.min(dragStart.y, dragNow.y), y1 = Math.max(dragStart.y, dragNow.y);
          if (x1 - x0 > 6 && y1 - y0 > 6) {
            buildings.push({
              cx: (x0 + x1) / 2 / SX, cy: (y0 + y1) / 2 / SYY,
              w: (x1 - x0) / SX, h: (y1 - y0) / SYY,
            });
            edited = true;
          }
        } else if (dragMode === 'move' && moveIdx >= 0) {
          edited = true;
        }
        dragMode = null; moveIdx = -1; moveOff = null; dragStart = null; dragNow = null;
        // 即時模式：編輯完成即以當前編輯模式（warm/fresh）自動送
        if (edited && liveMode) sendAnalyze(editMode);
      };
    };
  }

  // ── UI helpers ───────────────────────────────────────────────────────────
  // demo/index.html 的架構圖用 CSS 選擇器讀這兩個 <html data-arch-*> 屬性決定
  // 要不要亮/動；純屬性 bookkeeping，跟下面本來就在做的 el.dot.className 同一類動作，
  // 不構成新決策邏輯（index.html 本身仍是純殼，沒有任何 if/state）。
  function setArchAttr(name, on) {
    document.documentElement.setAttribute('data-arch-' + name, on ? '1' : '0');
  }

  function setStatus(on, msg) {
    el.dot.className = on ? 'on' : '';
    el.status.textContent = msg;
    setArchAttr('connected', on);
  }
  function logLine(s) {
    if (!el.log) { console.log(s); return; }
    el.log.textContent += s + '\n';
    el.log.scrollTop = el.log.scrollHeight;
  }

  // ── DOM 建構（index.html 只提供空的 root 容器）────────────────────────────
  function mk(tag, props, parent) {
    const node = document.createElement(tag);
    if (props) {
      Object.keys(props).forEach(function (k) {
        if (k === 'text') { node.textContent = props[k]; }
        else if (k === 'html') { node.innerHTML = props[k]; }
        else if (k === 'onclick') { node.addEventListener('click', props[k]); }
        else if (k === 'onchange') { node.addEventListener('change', props[k]); }
        else { node.setAttribute(k, props[k]); }
      });
    }
    if (parent) parent.appendChild(node);
    return node;
  }

  function buildUI(root) {
    root.innerHTML = '';
    mk('h1', { text: 'NCA 互動 demo → lbmnca.bridge（WebSocket）' }, root);

    // ── 連線列 ──
    const rowConn = mk('div', { class: 'row' }, root);
    el.url = mk('input', { id: 'url', value: CFG.ws_url, size: '22' }, rowConn);
    mk('button', { class: 'primary', text: 'Connect', onclick: connectWS }, rowConn);
    mk('button', { text: 'Disconnect', onclick: disconnectWS }, rowConn);
    el.dot = mk('span', { id: 'dot' }, rowConn);
    el.status = mk('span', { id: 'status', class: 'hint', text: '未連線' }, rowConn);

    // ── 主體：畫布 + 控制面板 ──
    const wrap = mk('div', { id: 'wrap' }, root);
    const left = mk('div', null, wrap);
    el.canvasHolder = mk('div', { id: 'canvas-holder' }, left);
    mk('div', {
      class: 'hint',
      html: '空白處拖曳＝畫新建物；建物上拖曳＝<b>移動 block</b>（放開自動送 warm）。'
          + '右鍵刪除單棟／Clear 清空。深色=solid，淺色=fluid，左箭頭=風向 +X。',
    }, left);

    const panel = mk('div', { style: 'min-width:260px' }, wrap);

    // 風力下拉（由 config.wind_levels 生成）
    const rowWind = mk('div', { class: 'row' }, panel);
    mk('span', { text: '風力 wind_level：' }, rowWind);
    el.wind = mk('select', { id: 'wind' }, rowWind);
    CFG.wind_levels.forEach(function (w) {
      const o = mk('option', { value: String(w.value), text: w.label }, el.wind);
      if (w.default === true) o.selected = true;
    });

    // Send fresh / warm
    const rowSend = mk('div', { class: 'row' }, panel);
    mk('button', { class: 'primary', text: 'Send (fresh 冷啟)', onclick: function () { sendAnalyze('fresh'); } }, rowSend);
    mk('button', { class: 'warm', text: 'Send (warm 溫啟)', onclick: function () { sendAnalyze('warm'); } }, rowSend);

    const rowClear = mk('div', { class: 'row' }, panel);
    mk('button', { text: 'Clear 建物', onclick: clearBuildings }, rowClear);

    // Live / 編輯模式
    const rowLive = mk('div', {
      class: 'row',
      style: 'border-top:1px solid #2a2f3a;padding-top:8px;margin-top:4px',
    }, panel);
    el.liveBtn = mk('button', { id: 'liveBtn', onclick: toggleLive }, rowLive);
    el.modeBtn = mk('button', { id: 'modeBtn', onclick: toggleEditMode }, rowLive);
    el.modeHint = mk('div', { class: 'row hint', id: 'modeHint' }, panel);

    // viz 模式按鈕（由 config.viz_modes 生成，每列 3 顆）
    mk('div', { class: 'row hint', text: 'viz 模式（WS set_viz_mode）：' }, panel);
    let rowViz = null;
    CFG.viz_modes.forEach(function (m, i) {
      if (i % 3 === 0) rowViz = mk('div', { class: 'row' }, panel);
      const btn = mk('button', { text: m, onclick: function () { sendViz(m); } }, rowViz);
      if (m === CFG.default_viz_mode) btn.classList.add('primary');
    });

    // 序列下拉（由 config.sequences 生成）+ Play/Stop
    mk('div', { class: 'row hint', text: '固定時間序列腳本：' }, panel);
    const rowSeqSel = mk('div', { class: 'row' }, panel);
    el.seqSelect = mk('select', {
      id: 'seqSelect',
      onchange: function (e) { selectSequence(e.target.value); },
    }, rowSeqSel);
    Object.keys(CFG.sequences).forEach(function (k) {
      mk('option', { value: k, text: k + ' — ' + CFG.sequences[k].name }, el.seqSelect);
    });

    const rowSeqBtn = mk('div', { class: 'row' }, panel);
    mk('button', { class: 'primary', text: '▶ Play', onclick: playSequence }, rowSeqBtn);
    mk('button', { text: '■ Stop', onclick: stopSequence }, rowSeqBtn);
    el.seqinfo = mk('span', { id: 'seqinfo', class: 'hint', text: '未選定' }, rowSeqBtn);

    // log
    el.log = mk('div', { id: 'log' }, root);
  }

  // ── 對外入口 ─────────────────────────────────────────────────────────────
  function init(config, rootEl) {
    DEFAULT_WIND = validateConfig(config);
    CFG = config;

    SIM_W = CFG.sim.width;
    SIM_H = CFG.sim.height;
    CANVAS_W = CFG.canvas.width;
    CANVAS_H = CFG.canvas.height;
    SX = CANVAS_W / SIM_W;
    SYY = CANVAS_H / SIM_H;

    liveMode = CFG.live.enabled === true;
    editMode = CFG.live.edit_mode;
    LIVE_THROTTLE_MS = CFG.live.throttle_ms;

    const root = rootEl || document.getElementById('nca-demo-root');
    if (!root) throw new Error('[NCADemo] 找不到掛載容器 #nca-demo-root');

    buildUI(root);

    // launcher 面板可能用 ?ws_port=N 導過來（見 renderLaunchPanel）；有帶就覆蓋
    // Connect 欄位預設值，讓整條鏈路統一用使用者選的 port。沒帶（如直接開 http:// 頁、
    // run_demo.bat 開的頁）就完全比照原行為，用 CFG.ws_url。
    const wsPortOverride = new URLSearchParams(global.location.search).get('ws_port');
    if (wsPortOverride) {
      el.url.value = 'ws://localhost:' + wsPortOverride;
      logLine('WS port 由啟動器面板指定：' + wsPortOverride);
    }

    applyLiveButton();
    applyEditModeButton();
    selectSequence(el.seqSelect.value);

    if (typeof global.p5 !== 'function') {
      throw new Error('[NCADemo] p5.js 未載入（檢查 CDN 或網路）');
    }
    p5i = new global.p5(makeSketch());
    logLine('已載入 config：sim ' + SIM_W + '×' + SIM_H + '，canvas ' + CANVAS_W + '×' + CANVAS_H
            + '，' + Object.keys(CFG.sequences).length + ' 條序列');
  }

  // 顯示致命錯誤（config 載入失敗、file:// CORS…）；不靜默失敗。
  function fatal(root, title, detail) {
    const node = root || document.body;
    node.innerHTML =
      '<div style="font-family:system-ui,sans-serif;color:#e6e6e6;background:#3a1f22;'
      + 'border:1px solid #c0392b;border-radius:8px;padding:16px;margin:12px 0">'
      + '<div style="font-size:15px;font-weight:600;margin-bottom:8px">✗ ' + title + '</div>'
      + '<div style="font-size:13px;white-space:pre-wrap;line-height:1.6">' + detail + '</div>'
      + '</div>';
    console.error('[NCADemo] ' + title + '\n' + detail);
  }

  const HOWTO =
    '請改用本機 HTTP server 開啟（fetch 在 file:// 下會被 CORS 擋）：\n'
    + '  ① 雙擊 docs\\demo\\run_demo.bat（會一併啟動 main.py）\n'
    + '  ② 或手動：在 lbm-nca-thesis 根目錄執行 python -m http.server 8080\n'
    + '     然後開 http://localhost:8080/demo/index.html';

  // ── launcher 面板（file:// 開啟時顯示）──────────────────────────────────
  // 瀏覽器 JS 無法直接 spawn 本機程序；此面板透過 fetch 呼叫常駐的
  // lbmnca.bridge/launcher.py（固定 8090），由它 subprocess 起 main.py + http.server。
  // 就緒後導向 http:// 正式頁，該頁走原本 boot() 的 http 分支顯示完整 demo UI。
  const LAUNCHER_URL = 'http://localhost:8090';
  const DEMO_HTTP_BASE = 'http://localhost:8080/demo/index.html';
  const WS_PORT_STORAGE_KEY = 'nca_ws_port';
  const WS_PORT_DEFAULT = 8765;
  const MIN_PORT = 1024, MAX_PORT = 65535;

  function renderLaunchPanel(root) {
    root.innerHTML = '';
    const wrap = mk('div', {
      style: 'font-family:system-ui,sans-serif;color:#e6e6e6;background:#1c2128;'
           + 'border:1px solid #3a4150;border-radius:8px;padding:16px;margin:12px 0',
    }, root);
    mk('div', { style: 'font-size:15px;font-weight:600;margin-bottom:8px', text: '🚀 NCA demo 啟動器' }, wrap);
    mk('div', {
      class: 'hint',
      style: 'margin-bottom:10px;line-height:1.6',
      text: '偵測到 file:// 開啟。按下方按鈕會透過本機 launcher 服務啟動 main.py + http.server'
          + '（各自獨立視窗）與 GGUI，完成後自動跳轉到正式 demo 頁。',
    }, wrap);

    const rowPort = mk('div', { class: 'row' }, wrap);
    mk('span', { text: 'WS port：' }, rowPort);
    const portInput = mk('input', {
      type: 'number', id: 'launchWsPort', size: '6',
      value: String(parseInt(localStorage.getItem(WS_PORT_STORAGE_KEY), 10) || WS_PORT_DEFAULT),
    }, rowPort);
    mk('span', {
      class: 'hint',
      text: '（main.py 監聽的 port，預設 ' + WS_PORT_DEFAULT + '；跟其他服務衝突才需要改）',
    }, rowPort);

    const statusLine = mk('div', { class: 'hint', style: 'margin:10px 0' }, wrap);
    const btn = mk('button', { class: 'primary', text: '🚀 啟動 demo 後端' }, wrap);
    const log = mk('div', { id: 'launch-log', style: 'margin-top:10px' }, wrap);
    log.className = '';
    log.style.fontFamily = 'ui-monospace, Consolas, monospace';
    log.style.fontSize = '12px';
    log.style.background = '#0d0f12';
    log.style.border = '1px solid #2a2f3a';
    log.style.borderRadius = '6px';
    log.style.padding = '8px';
    log.style.height = '120px';
    log.style.overflow = 'auto';
    log.style.whiteSpace = 'pre-wrap';

    function llog(s) { log.textContent += s + '\n'; log.scrollTop = log.scrollHeight; }

    const MANUAL_HOWTO =
      'launcher 未啟動。請先啟動它（僅需一次；可設開機自動啟動見 demo/install_autostart.ps1）：\n'
      + '  雙擊 docs\\demo\\launcher.bat\n'
      + '啟動後 launcher 會常駐監聽 8090，之後開這頁按按鈕即可，不用再手動跑任何東西。';

    // 讀輸入框並做範圍檢查；不合法回 null（呼叫端顯示錯誤、不打 API）。
    function readPort() {
      const port = parseInt(portInput.value, 10);
      if (!Number.isInteger(port) || port < MIN_PORT || port > MAX_PORT) {
        statusLine.textContent = '✗ WS port 需為 ' + MIN_PORT + '-' + MAX_PORT + ' 的整數';
        return null;
      }
      return port;
    }

    function pollStatus(port, onReady, elapsedMs) {
      elapsedMs = elapsedMs || 0;
      if (elapsedMs > 40000) {
        statusLine.textContent = '逾時：40 秒內後端未就緒，請看 main.py 視窗是否有 traceback。';
        llog('✗ 逾時');
        return;
      }
      fetch(LAUNCHER_URL + '/status?ws_port=' + port)
        .then(function (r) { return r.json(); })
        .then(function (s) {
          statusLine.textContent = 'launcher: 在線 | main.py(ws ' + port + '): ' + (s.ws ? '就緒' : '載入中…')
            + ' | http.server(8080): ' + (s.http ? '就緒' : '啟動中…');
          if (s.ws && s.http) { onReady(); return; }
          setTimeout(function () { pollStatus(port, onReady, elapsedMs + 1000); }, 1000);
        })
        .catch(function (e) {
          statusLine.textContent = 'launcher 連線中斷：' + String(e);
          llog('✗ ' + String(e));
        });
    }

    btn.addEventListener('click', function () {
      const port = readPort();
      if (port === null) return;
      localStorage.setItem(WS_PORT_STORAGE_KEY, String(port));

      btn.disabled = true;
      statusLine.textContent = '檢查 launcher…';
      fetch(LAUNCHER_URL + '/status?ws_port=' + port)
        .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
        .then(function () {
          llog('→ 呼叫 launcher /start?ws_port=' + port);
          return fetch(LAUNCHER_URL + '/start?ws_port=' + port);
        })
        .then(function (r) {
          if (r.status === 409) {
            return r.json().then(function (e) {
              throw { conflict: true, msg: e.error, current_port: e.current_port };
            });
          }
          if (!r.ok) throw new Error('HTTP ' + r.status);
          return r.json();
        })
        .then(function (res) {
          llog('← ' + JSON.stringify(res));
          statusLine.textContent = '等待後端就緒…';
          pollStatus(port, function () {
            llog('✓ 後端就緒，導向正式頁…');
            statusLine.textContent = '就緒，跳轉中…';
            global.location.href = DEMO_HTTP_BASE + '?ws_port=' + port;
          });
        })
        .catch(function (e) {
          if (e && e.conflict) {
            statusLine.textContent = '✗ main.py 已在 port ' + e.current_port + ' 執行中，'
              + '要換到 ' + port + ' 請先關掉那個 main.py 視窗（不會自動幫你關）。';
            llog('✗ 衝突：' + e.msg);
          } else {
            statusLine.textContent = '✗ 無法連上 launcher（' + LAUNCHER_URL + '）';
            llog(MANUAL_HOWTO + '\n\n(' + String(e) + ')');
          }
          btn.disabled = false;
        });
    });

    statusLine.textContent = '尚未啟動';
  }

  /** 載入 config 並初始化；index.html 只需呼叫這一個函式。 */
  function boot(configUrl, rootId) {
    const root = document.getElementById(rootId || 'nca-demo-root');
    if (global.location && global.location.protocol === 'file:') {
      const launchRoot = document.getElementById('nca-launch-bar');
      renderLaunchPanel(launchRoot || root);
      return;
    }
    fetch(configUrl)
      .then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status + ' ' + r.statusText);
        return r.json();
      })
      .then(function (cfg) { init(cfg, root); })
      .catch(function (e) {
        fatal(root, 'config 載入/驗證失敗：' + configUrl, String(e) + '\n\n' + HOWTO);
      });
  }

  global.NCADemo = {
    init: init,
    boot: boot,
    renderLaunchPanel: renderLaunchPanel,
    // 供 console 手動操作／自動化測試
    connectWS: connectWS,
    disconnectWS: disconnectWS,
    sendAnalyze: sendAnalyze,
    sendViz: sendViz,
    sendSeek: sendSeek,
    sendSpeed: sendSpeed,
    sendSave: sendSave,
    clearBuildings: clearBuildings,
    selectSequence: selectSequence,
    playSequence: playSequence,
    stopSequence: stopSequence,
  };
})(window);
