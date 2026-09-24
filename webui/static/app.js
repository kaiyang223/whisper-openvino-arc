/* Whisper 本地转写 Web UI — 前端逻辑 */
'use strict';

const $ = (s) => document.querySelector(s);
const S = {
  state: null,          // /api/state
  files: [],            // 已选文件 {path,name,size}
  job: null,            // 当前任务 id
  es: null,             // EventSource
  running: false,
  browse: { path: '', parent: null, sel: null },
  results: [],          // 本次会话的转写结果
};

/* ---------------- 工具 ---------------- */
function fmtSize(n) {
  if (n == null) return '';
  const u = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return n.toFixed(n < 10 && i > 0 ? 1 : 0) + u[i];
}
function fmtDur(s) {
  if (s == null || isNaN(s)) return '—';
  s = Math.max(0, s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), x = Math.floor(s % 60);
  return (h ? h + ':' + String(m).padStart(2, '0') : String(m)) + ':' + String(x).padStart(2, '0');
}
function fmtTs(t) {
  t = Math.max(0, t || 0);
  const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = Math.floor(t % 60);
  const ms = Math.round((t - Math.floor(t)) * 1000);
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')},${String(ms).padStart(3, '0')}`;
}
function baseName(p) { return (p || '').replace(/[\\/]+$/, '').split(/[\\/]/).pop() || p; }
function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}
function note(kind, html) {
  const n = $('#alert');
  n.className = 'note' + (kind ? ' ' + kind : '');
  n.innerHTML = html || '';
}

/* ---------------- 初始化 ---------------- */
async function init() {
  let st;
  try {
    st = await (await fetch('/api/state')).json();
  } catch (e) {
    note('err', '<b>无法连接服务端。</b>请确认服务仍在运行。');
    return;
  }
  S.state = st;

  // 顶栏徽标
  const g = $('#bGpu');
  if (st.gpu_name) { g.className = 'badge ok'; g.innerHTML = `<span class="k">显卡</span> ${st.gpu_name.replace(/ \(dGPU\)$/, '')}`; }
  else { g.className = 'badge err'; g.innerHTML = '<span class="k">显卡</span> 未检测到 GPU'; }

  const ready = st.models.filter(m => m.ready);
  const mb = $('#bModel');
  mb.className = 'badge ' + (ready.length ? 'ok' : 'err');
  mb.innerHTML = `<span class="k">模型</span> ${ready.length ? ready.map(m => m.key).join(' / ') : '未下载'}`;

  const vb = $('#bVad');
  vb.className = 'badge ' + (st.vad_ready ? 'ok' : 'warn');
  vb.innerHTML = `<span class="k">VAD</span> ${st.vad_ready ? '就绪' : '缺失'}`;

  // 精度档位
  const seg = $('#segAcc');
  seg.innerHTML = '';
  st.presets.forEach(p => {
    const b = el('button');
    b.dataset.key = p.key;
    b.innerHTML = `${p.key}<small>${p.key === 'standard' ? '最快' : p.key === 'high' ? '日常' : '最准'}</small>`;
    b.onclick = () => setAccuracy(p.key);
    seg.appendChild(b);
  });
  setAccuracy('high');

  // 模型
  const ms = $('#model');
  ms.innerHTML = '<option value="">跟随档位</option>';
  st.models.forEach(m => {
    const o = el('option', null, `${m.key} — ${m.label}${m.ready ? '' : '（未下载）'}`);
    o.value = m.key;
    if (!m.ready) o.disabled = true;
    ms.appendChild(o);
  });

  // 设备
  const dv = $('#device');
  dv.innerHTML = '';
  (st.devices || []).forEach(d => {
    const o = el('option', null, d === 'GPU' ? 'GPU（推荐）' : d);
    o.value = d;
    dv.appendChild(o);
  });
  if ([...dv.options].some(o => o.value === 'GPU')) dv.value = 'GPU';

  // 语言
  const lg = $('#language');
  lg.innerHTML = '';
  st.languages.forEach(l => {
    const o = el('option', null, l.code === 'auto' ? '自动检测' : `${l.code} — ${l.label}`);
    o.value = l.code;
    lg.appendChild(o);
  });
  $('#langCount').textContent = (st.languages.length - 1) + ' 种';

  // 输出格式
  const fm = $('#formats');
  fm.innerHTML = '';
  st.formats.forEach((f, i) => {
    const l = el('label');
    const c = document.createElement('input');
    c.type = 'checkbox'; c.value = f; c.checked = (f === 'txt');
    c.onchange = syncFormatAtLeastOne;
    l.appendChild(c);
    l.appendChild(document.createTextNode(f));
    fm.appendChild(l);
  });

  $('#vad').checked = st.vad_ready;
  if (!st.vad_ready) {
    $('#vad').disabled = true;
    $('#vad').closest('.switch').style.opacity = .55;
  }

  bindEvents();
  renderFiles();
  window.__APP_READY__ = true;      // 自动化测试用的就绪标志
}

function setAccuracy(key) {
  [...$('#segAcc').children].forEach(b => b.classList.toggle('active', b.dataset.key === key));
  const map = {
    standard: 'turbo，不做 VAD —— 最快，适合干净单人录音',
    high: 'turbo + VAD —— 日常首选，快且不会在静音处编造文本',
    max: 'large-v3 + VAD —— 最准，适合重要素材，速度约为 turbo 的 1/3',
  };
  $('#accHint').textContent = map[key] || '';
  $('#model').value = '';
}
function accuracy() {
  const b = [...$('#segAcc').children].find(x => x.classList.contains('active'));
  return b ? b.dataset.key : 'high';
}
function syncFormatAtLeastOne(e) {
  const on = [...$('#formats').querySelectorAll('input')].filter(c => c.checked);
  if (!on.length) { e.target.checked = true; }
}

/* ---------------- 文件列表 ---------------- */
function addFiles(items) {
  let added = 0;
  items.forEach(it => {
    if (!it || !it.path) return;
    if (S.files.some(f => f.path === it.path)) return;
    S.files.push(it);
    added++;
  });
  renderFiles();
  if (added) note('ok', `已添加 <b>${added}</b> 个文件，共 <b>${S.files.length}</b> 个待转写。`);
}

function renderFiles() {
  const box = $('#filelist');
  box.innerHTML = '';
  S.files.forEach((f, i) => {
    const row = el('div', 'fileitem');
    row.appendChild(el('span', 'nm', f.name || baseName(f.path)));
    if (f.size) row.appendChild(el('span', 'sz', fmtSize(f.size)));
    const rm = el('button', 'rm', '×');
    rm.title = '移除';
    rm.onclick = () => { S.files.splice(i, 1); renderFiles(); };
    row.appendChild(rm);
    box.appendChild(row);
  });
  $('#btnRun').disabled = S.running || S.files.length === 0;
}

/* ---------------- 目录浏览 ---------------- */
async function browseTo(path) {
  const r = await fetch('/api/browse?path=' + encodeURIComponent(path || ''));
  const d = await r.json();
  S.browse.path = d.path; S.browse.parent = d.parent; S.browse.sel = null;
  $('#crumbPath').value = d.path || '';
  const list = $('#browseList');
  list.innerHTML = '';

  if (d.error) list.appendChild(el('div', 'brow', '⚠ ' + d.error));

  d.dirs.forEach(x => {
    const row = el('div', 'brow');
    row.appendChild(el('span', 'ic', '📁'));
    row.appendChild(el('span', 'nm', x.name));
    row.onclick = () => browseTo(x.path);
    list.appendChild(row);
  });

  d.files.forEach(x => {
    const row = el('div', 'brow');
    row.appendChild(el('span', 'ic', '🎵'));
    row.appendChild(el('span', 'nm', x.name));
    row.appendChild(el('span', 'sz', fmtSize(x.size)));
    row.onclick = () => {
      [...list.querySelectorAll('.brow')].forEach(n => n.classList.remove('sel'));
      row.classList.add('sel');
      S.browse.sel = { path: x.path, name: x.name, size: x.size };
      $('#btnAddSel').disabled = false;
      $('#btnAddSel').textContent = '添加所选';
    };
    row.ondblclick = () => { addFiles([{ path: x.path, name: x.name, size: x.size }]); };
    list.appendChild(row);
  });

  if (!d.dirs.length && !d.files.length && !d.error) {
    list.appendChild(el('div', 'brow', '（此目录下没有子目录或音视频文件）'));
  }
  $('#selInfo').textContent = S.browse.path || '（盘符列表）';
  const addDir = $('#btnAddDir');
  addDir.disabled = !S.browse.path;
  addDir.textContent = '添加整个目录' + ($('#recursive').checked ? '（含子目录）' : '');
  $('#btnAddSel').disabled = true;
}

/* ---------------- 拖拽上传 ---------------- */
async function uploadFiles(fileList) {
  const arr = [...fileList];
  if (!arr.length) return;
  note(null, `正在上传 ${arr.length} 个文件…`);
  for (const f of arr) {
    try {
      const res = await fetch('/api/upload?name=' + encodeURIComponent(f.name), {
        method: 'POST', body: f,
      });
      const d = await res.json();
      if (d.error) { note('err', '上传失败：' + d.error); continue; }
      addFiles([{ path: d.path, name: d.name, size: f.size }]);
    } catch (e) {
      note('err', '上传失败：' + e);
    }
  }
  note('ok', `上传完成，当前 <b>${S.files.length}</b> 个文件待转写。`);
}

/* ---------------- 运行 ---------------- */
function payload() {
  const formats = [...$('#formats').querySelectorAll('input:checked')].map(c => c.value);
  return {
    inputs: S.files.map(f => f.path),
    accuracy: accuracy(),
    model: $('#model').value || null,
    device: $('#device').value || 'GPU',
    language: $('#language').value || 'auto',
    formats: formats.length ? formats : ['txt'],
    prompt: $('#prompt').value,
    hotwords: $('#hotwords').value,
    vad: $('#vad').checked,
    vad_merge_gap: parseInt($('#vadMerge').value, 10) || 2500,
    vad_max_block: parseFloat($('#vadMax').value) || 28,
    output_dir: $('#outputDir').value,
    timestamp_names: $('#tsNames').checked,
    keep_hallucination: $('#keepHallu').checked,
    recursive: $('#recursive').checked,
    timestamps: true,
  };
}

async function run() {
  if (!S.files.length) return;
  setRunning(true);
  note(null, '');
  $('#log').innerHTML = '';
  $('#results').innerHTML = '';
  S.results = [];
  $('#btnCopyAll').style.display = 'none';
  showProgress(true);
  setProgress(0, true);
  $('#stText').textContent = '提交任务…';
  $('#stageTag').textContent = '';

  let d;
  try {
    const r = await fetch('/api/transcribe', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload()),
    });
    d = await r.json();
  } catch (e) {
    note('err', '提交失败：' + e);
    setRunning(false); showProgress(false);
    return;
  }
  if (d.error) {
    note('err', d.error.replace(/\n/g, '<br>').replace(/ /g, '&nbsp;'));
    setRunning(false); showProgress(false);
    return;
  }
  S.job = d.job;
  listen(d.job);
}

function logLine(text, cls) {
  const box = $('#log');
  const line = el('div', cls || null, text);
  box.appendChild(line);
  box.scrollTop = box.scrollHeight;
  while (box.childElementCount > 400) box.removeChild(box.firstChild);
}

let fileTotal = 0, fileIndex = 0;

function listen(job) {
  if (S.es) S.es.close();
  const es = new EventSource('/api/events?job=' + encodeURIComponent(job));
  S.es = es;

  es.onmessage = (ev) => {
    let m;
    try { m = JSON.parse(ev.data); } catch { return; }
    handle(m);
  };
  es.onerror = () => {
    // 任务已结束时服务端会正常关闭流，这里只在仍标记运行时提示
    if (S.running) logLine('（进度连接断开，可刷新页面查看结果）', 't-dim');
  };
}

function handle(m) {
  switch (m.stage) {
    case 'hello':
      logLine('已连接，等待开始…', 't-dim');
      break;

    case 'start':
      fileTotal = m.total; fileIndex = 0;
      logLine(`任务开始：${m.total} 个文件 · 模型 ${m.model} · 设备 ${m.device}` +
        ` · 语言 ${m.language} · VAD ${m.vad ? '开' : '关'}`);
      $('#stageTag').textContent = `${m.total} 个文件`;
      break;

    case 'loading':
      $('#stText').textContent = `加载模型 ${m.model} 到 ${m.device}…`;
      logLine(`加载模型 ${m.model}…`, 't-dim');
      break;

    case 'loaded':
      logLine(`模型就绪（${m.seconds}s）`, 't-ok');
      break;

    case 'file_begin':
      fileIndex = m.index;
      $('#stText').textContent = `[${m.index}/${m.total}] ${m.name}`;
      logLine(`\n▶ [${m.index}/${m.total}] ${m.name}`);
      break;

    case 'decoding':
      logLine('  解码音频…', 't-dim');
      break;

    case 'decoded':
      logLine(`  时长 ${fmtDur(m.duration)}`);
      break;

    case 'slice': {
      const pct = 20 + (fileIndex - 1) / Math.max(fileTotal, 1) * 80 +
        (m.index / Math.max(m.total, 1)) * (80 / Math.max(fileTotal, 1));
      setProgress(Math.min(99, pct), false);
      $('#stText').textContent = `[${fileIndex}/${fileTotal}] ${baseName(m.file)} — 识别第 ${m.index}/${m.total} 段`;
      if (m.index === 1) logLine(`  VAD 切出 ${m.total} 段语音`);
      break;
    }

    case 'done': {
      const pct = 20 + fileIndex / Math.max(fileTotal, 1) * 80;
      setProgress(Math.min(100, pct), false);
      logLine(`  完成 ${m.elapsed}s · ${m.speed ? m.speed.toFixed(1) + '× 实时' : ''} · ` +
        `${m.chars} 字符 · 语种 ${m.language || '—'}`, 't-ok');
      const dropped = (m.perf && m.perf.hallucination_dropped) || [];
      dropped.forEach(d => logLine('  ⚠ 已丢弃疑似幻觉: ' + d, 't-warn'));
      if (m.perf && m.perf.skipped_all) logLine('  ⓘ 整段未检测到语音', 't-dim');
      renderResult(m);
      break;
    }

    case 'file_end':
      if (m.ok === false) logLine(`  ✗ ${m.name} 失败`, 't-err');
      else if (m.partial) logLine(`  ⓘ ${m.name} 已取消，保存的是部分结果`, 't-warn');
      break;

    case 'error':
      logLine(`  ✗ ${m.message}`, 't-err');
      break;

    case 'finish':
      setProgress(100, false);
      onFinish(m);
      break;

    case 'fatal':
      note('err', '<b>任务失败：</b>' + String(m.message).replace(/\n/g, '<br>'));
      logLine('✗ ' + m.message, 't-err');
      setRunning(false); showProgress(false);
      break;

    case 'eof':
      if (S.running) onFinish(m.summary || {});
      break;
  }
}

function onFinish(s) {
  setRunning(false);
  if (S.es) { S.es.close(); S.es = null; }
  $('#stageTag').textContent = '';
  const parts = [
    `成功 <b>${s.ok ?? 0}</b>`,
    s.partial ? `其中 <b>${s.partial}</b> 个为部分结果` : null,
    s.fail ? `失败 <b>${s.fail}</b>` : null,
    s.audio_seconds ? `音频总长 <b>${fmtDur(s.audio_seconds)}</b>` : null,
    s.elapsed ? `耗时 <b>${s.elapsed}s</b>` : null,
    s.speed ? `平均 <b>${s.speed}× 实时</b>` : null,
  ].filter(Boolean);
  if (s.cancelled) note('warn', '<b>任务已取消。</b>' + parts.join(' · '));
  else if (s.fail) note('warn', '<b>部分失败。</b>' + parts.join(' · '));
  else note('ok', parts.join(' · '));
  $('#stText').textContent = s.cancelled ? '已取消' : '全部完成';
  if (S.results.length) $('#btnCopyAll').style.display = '';
  renderFiles();
}

function setRunning(on) {
  S.running = on;
  $('#btnRun').disabled = on || !S.files.length;
  $('#btnRun').textContent = on ? '转写中…' : '开始转写';
  $('#btnCancel').disabled = !on;
  ['#drop', '#btnBrowse', '#language', '#prompt'].forEach(s => {
    const n = $(s); if (n) n.style.pointerEvents = on ? 'none' : '';
  });
}

function showProgress(on) {
  $('#progCard').style.display = on ? '' : 'none';
  $('#progWrap').classList.toggle('show', on);
  setRunning(S.running);
}
function setProgress(pct, indet) {
  const bar = $('#bar');
  bar.classList.toggle('indet', !!indet);
  bar.firstElementChild.style.width = (indet ? 35 : pct) + '%';
  $('#stPct').textContent = indet ? '' : Math.round(pct) + '%';
}

/* ---------------- 结果渲染 ---------------- */
function renderResult(m) {
  S.results.push(m);
  const box = $('#results');
  if (box.querySelector('.empty')) box.innerHTML = '';

  const card = el('div', 'result');

  const head = el('div', 'rh');
  head.appendChild(el('span', 'nm', m.name));
  if (m.partial) {
    const b = el('span', 'badge warn', '部分结果');
    b.style.fontSize = '11px';
    b.style.padding = '1px 7px';
    head.appendChild(b);
  }
  head.appendChild(el('span', 'spacer'));
  const meta = [
    fmtDur(m.audio_seconds),
    m.elapsed != null ? m.elapsed + 's' : null,
    m.speed ? m.speed.toFixed(1) + '×' : null,
    m.language ? m.language : null,
    (m.perf && m.perf.vad_slices != null) ? `VAD ${m.perf.vad_slices} 段` : null,
  ].filter(Boolean).join(' · ');
  head.appendChild(el('span', 'meta', meta));
  card.appendChild(head);

  // 标签页
  const segs = m.segments || [];
  const tabs = el('div', 'tabs');
  const panes = el('div');
  const defs = [
    ['文本', () => {
      const w = el('div');
      if (!m.text) {
        const e2 = el('div', 'empty');
        e2.appendChild(el('div', 'big', '没有识别到内容'));
        e2.appendChild(el('div', 'sub', 'VAD 判定这段音频里没有语音（这是正常的，不是错误）'));
        w.appendChild(e2);
      } else {
        w.appendChild(el('div', 'txt', m.text));
      }
      return w;
    }],
    [`分段 (${segs.length})`, () => {
      const wrap = el('div', 'segwrap');
      if (!segs.length) {
        wrap.appendChild(el('div', 'txt', '（无分段信息）'));
        return wrap;
      }
      const t = el('table', 'seg');
      const thead = el('thead');
      const hr = el('tr');
      ['时间', '内容'].forEach(h => hr.appendChild(el('th', null, h)));
      thead.appendChild(hr); t.appendChild(thead);
      const tb = el('tbody');
      segs.forEach(s => {
        const tr = el('tr');
        tr.appendChild(el('td', 'ts', `${fmtTs(s.start)} → ${fmtTs(s.end)}`));
        tr.appendChild(el('td', null, s.text));
        tb.appendChild(tr);
      });
      t.appendChild(tb);
      wrap.appendChild(t);
      return wrap;
    }],
    ['JSON', () => el('div', 'txt', JSON.stringify(
      { name: m.name, language: m.language, audio_seconds: m.audio_seconds, text: m.text, segments: segs, perf: m.perf },
      null, 2))],
  ];
  defs.forEach(([label, make], i) => {
    const b = el('button', i === 0 ? 'active' : null, label);
    const pane = el('div', 'tabpane' + (i === 0 ? ' active' : ''));
    pane.appendChild(make());
    b.onclick = () => {
      [...tabs.children].forEach(x => x.classList.remove('active'));
      [...panes.children].forEach(x => x.classList.remove('active'));
      b.classList.add('active'); pane.classList.add('active');
    };
    tabs.appendChild(b); panes.appendChild(pane);
  });
  card.appendChild(tabs);
  card.appendChild(panes);

  // 下载
  if (m.outputs && m.outputs.length) {
    const dl = el('div', 'dl');
    const base = m.output_index || 0;
    m.outputs.forEach((p, k) => {
      const ext = (p.split('.').pop() || '').toUpperCase();
      const a = el('a', 'btn sm', '下载 ' + ext);
      a.href = `/api/download?job=${encodeURIComponent(S.job)}&i=${base + k}`;
      a.style.textDecoration = 'none';
      dl.appendChild(a);
    });
    const cp = el('button', 'btn sm ghost', '复制文本');
    cp.onclick = async () => {
      try { await navigator.clipboard.writeText(m.text || ''); cp.textContent = '已复制 ✓'; }
      catch { cp.textContent = '复制失败'; }
      setTimeout(() => { cp.textContent = '复制文本'; }, 1500);
    };
    dl.appendChild(cp);
    card.appendChild(dl);
  }

  box.appendChild(card);
  box.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/* ---------------- 事件绑定 ---------------- */
function bindEvents() {
  // 拖拽
  const dz = $('#drop');
  dz.onclick = () => { $('#mask').classList.add('show'); browseTo(S.browse.path || S.state.root); };
  ['dragenter', 'dragover'].forEach(e => dz.addEventListener(e, ev => {
    ev.preventDefault(); dz.classList.add('over');
  }));
  ['dragleave', 'drop'].forEach(e => dz.addEventListener(e, ev => {
    ev.preventDefault(); dz.classList.remove('over');
  }));
  dz.addEventListener('drop', ev => {
    if (ev.dataTransfer.files.length) uploadFiles(ev.dataTransfer.files);
  });
  document.addEventListener('dragover', e => e.preventDefault());
  document.addEventListener('drop', e => e.preventDefault());

  $('#btnBrowse').onclick = () => { $('#mask').classList.add('show'); browseTo(S.browse.path || S.state.root); };
  $('#btnClearFiles').onclick = () => { S.files = []; renderFiles(); note(null, ''); };
  $('#recursive').onchange = () => { if ($('#mask').classList.contains('show')) browseTo(S.browse.path); };

  // 浏览弹层
  $('#btnClose').onclick = () => $('#mask').classList.remove('show');
  $('#mask').onclick = e => { if (e.target === $('#mask')) $('#mask').classList.remove('show'); };
  $('#btnUp').onclick = () => { if (S.browse.parent) browseTo(S.browse.parent); };
  $('#btnGo').onclick = () => browseTo($('#crumbPath').value.trim());
  $('#crumbPath').onkeydown = e => { if (e.key === 'Enter') browseTo($('#crumbPath').value.trim()); };
  $('#btnAddSel').onclick = () => {
    if (S.browse.sel) { addFiles([S.browse.sel]); $('#mask').classList.remove('show'); }
  };
  $('#btnAddDir').onclick = () => {
    if (S.browse.path) {
      addFiles([{ path: S.browse.path, name: S.browse.path + (S.state.root.endsWith('\\') ? '' : '\\*'), size: 0 }]);
      $('#mask').classList.remove('show');
    }
  };

  // 运行/取消
  $('#btnRun').onclick = run;
  $('#btnCancel').onclick = async () => {
    if (!S.job) return;
    $('#btnCancel').disabled = true;
    await fetch('/api/cancel', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job: S.job }),
    });
    logLine('已请求取消，当前片段跑完后停止…', 't-warn');
  };

  $('#btnCopyAll').onclick = async (e) => {
    const all = S.results.map(r => r.text || '').join('\n\n');
    try { await navigator.clipboard.writeText(all); e.target.textContent = '已复制 ✓'; }
    catch { e.target.textContent = '复制失败'; }
    setTimeout(() => { e.target.textContent = '复制全部文本'; }, 1500);
  };

  // 快捷键
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') $('#mask').classList.remove('show');
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey) && !S.running) run();
  });
}

init();
