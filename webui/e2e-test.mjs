// Whisper Web UI 端到端验证（系统 Edge + CDP，零依赖 npm）
// 用法：先启动 webui.bat，再运行  node webui/e2e-test.mjs
import { spawn } from 'node:child_process';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath } from 'node:url';

// 所有路径从脚本自身位置推导，clone 到任何目录都能跑
const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const SHOT = path.join(ROOT, 'docs', 'screenshots');
const SAMPLES = path.join(ROOT, 'samples');

const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const EDGE2 = 'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe';
const PORT = 9333;
const URL = process.env.WHISPER_UI_URL || 'http://127.0.0.1:8765/';

const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const httpGet = (u) => new Promise((res, rej) => {
  http.get(u, r => { let d = ''; r.on('data', c => d += c); r.on('end', () => res(d)); }).on('error', rej);
});

const results = [];
function check(name, ok, detail = '') {
  results.push({ name, ok, detail });
  console.log(`  ${ok ? '[PASS]' : '[FAIL]'} ${name}${detail ? '   ' + detail : ''}`);
}

let child = null;
let PROF = null;

try {
  const exe = fs.existsSync(EDGE) ? EDGE : EDGE2;
  if (!fs.existsSync(exe)) throw new Error('找不到 Edge: ' + exe);
  PROF = path.join(os.tmpdir(), 'wb-edge-' + Date.now());

  child = spawn(exe, [
    '--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
    '--disable-extensions', '--mute-audio', '--window-size=1560,1100',
    '--remote-debugging-port=' + PORT, '--user-data-dir=' + PROF, 'about:blank',
  ], { stdio: 'ignore' });

  let ver = null;
  for (let i = 0; i < 160; i++) {
    try { ver = JSON.parse(await httpGet(`http://127.0.0.1:${PORT}/json/version`)); break; }
    catch { await sleep(250); }
  }
  if (!ver) throw new Error('CDP 未就绪');
  console.log('  浏览器:', ver['Browser']);

  const targets = JSON.parse(await httpGet(`http://127.0.0.1:${PORT}/json/list`));
  const page = targets.find(t => t.type === 'page');

  const api = await new Promise((resolve, reject) => {
    const ws = new WebSocket(page.webSocketDebuggerUrl);
    const a = { ws, id: 0, waiting: new Map(), events: [] };
    ws.addEventListener('open', () => resolve(a));
    ws.addEventListener('error', () => reject(new Error('ws error')));
    ws.addEventListener('message', (e) => {
      const m = JSON.parse(e.data);
      if (m.id && a.waiting.has(m.id)) {
        const w = a.waiting.get(m.id); a.waiting.delete(m.id);
        m.error ? w.reject(new Error(JSON.stringify(m.error))) : w.resolve(m.result);
      } else if (m.method) a.events.push(m);
    });
  });

  const send = (method, params = {}, timeout = 240000) => {
    const id = ++api.id;
    return new Promise((resolve, reject) => {
      const t = setTimeout(() => { api.waiting.delete(id); reject(new Error('timeout ' + method)); }, timeout);
      api.waiting.set(id, { resolve: v => { clearTimeout(t); resolve(v); }, reject: e => { clearTimeout(t); reject(e); } });
      api.ws.send(JSON.stringify({ id, method, params }));
    });
  };
  const ev = async (expr) => {
    const r = await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true, userGesture: true });
    if (r.exceptionDetails) throw new Error('JS: ' + (r.exceptionDetails.exception?.description || '').slice(0, 500));
    return r.result.value;
  };

  await send('Page.enable'); await send('Runtime.enable');
  await send('DOM.enable'); await send('Log.enable');
  fs.mkdirSync(SHOT, { recursive: true });
  await send('Browser.setDownloadBehavior', { behavior: 'allow', downloadPath: SHOT.replace(/\\/g, '\\\\') });

  console.log('\n1. 加载页面');
  await send('Page.navigate', { url: URL });
  let ready = false;
  for (let i = 0; i < 80; i++) {
    try { if (await ev('!!window.__APP_READY__')) { ready = true; break; } } catch {}
    await sleep(250);
  }
  check('页面加载并初始化完成', ready);

  console.log('\n2. 顶栏状态');
  const top = await ev(`(() => {
    const g = document.querySelector('#bGpu'), m = document.querySelector('#bModel'), v = document.querySelector('#bVad');
    return { gpu: g.textContent.trim(), gcls: g.className, model: m.textContent.trim(), vad: v.textContent.trim() };
  })()`);
  check('显卡徽标显示 Arc A770', /Arc.*A770/.test(top.gpu), top.gpu);
  check('显卡徽标为成功态', top.gcls.includes('ok'), top.gcls);
  check('模型徽标列出已下载模型', /turbo/.test(top.model) && /largev3/.test(top.model), top.model);
  check('VAD 徽标就绪', /就绪/.test(top.vad), top.vad);

  console.log('\n3. 参数控件');
  const ctl = await ev(`(() => {
    const acc = [...document.querySelectorAll('#segAcc button')].map(b => b.dataset.key);
    const langs = document.querySelectorAll('#language option').length;
    const fmts = [...document.querySelectorAll('#formats input')].map(c => c.value);
    const devs = [...document.querySelectorAll('#device option')].map(o => o.value);
    const models = [...document.querySelectorAll('#model option')].map(o => o.value);
    const act = document.querySelector('#segAcc button.active')?.dataset.key;
    return { acc, langs, fmts, devs, models, act, vadOn: document.querySelector('#vad').checked };
  })()`);
  check('精度档位三档齐全', JSON.stringify(ctl.acc) === '["standard","high","max"]', ctl.acc.join(','));
  check('默认选中 high', ctl.act === 'high', ctl.act);
  check('语言下拉 101 项(含自动)', ctl.langs === 101, ctl.langs + ' 项');
  check('输出格式 5 种', ctl.fmts.length === 5, ctl.fmts.join(','));
  check('设备含 GPU', ctl.devs.includes('GPU'), ctl.devs.join(','));
  check('默认勾选 txt', await ev(`document.querySelector('#formats input[value=txt]').checked`));
  check('VAD 默认开启', ctl.vadOn === true);

  console.log('\n4. 截图：初始界面');
  let shot = await send('Page.captureScreenshot', { format: 'png' });
  fs.writeFileSync(path.join(SHOT, '01-initial.png'), Buffer.from(shot.data, 'base64'));
  check('初始截图已保存', true);

  console.log('\n5. 文件浏览弹层');
  await ev(`document.querySelector('#btnBrowse').click()`);
  await sleep(600);
  check('弹层可见', await ev(`getComputedStyle(document.querySelector('#mask')).display !== 'none'`));
  await ev(`(async () => { await browseTo(${JSON.stringify(SAMPLES)}) })()`);
  await sleep(700);
  const br = await ev(`(() => {
    const rows = [...document.querySelectorAll('#browseList .brow')];
    return { n: rows.length, names: rows.map(r => r.querySelector('.nm').textContent),
             crumb: document.querySelector('#crumbPath').value };
  })()`);
  check('列目录成功', path.resolve(br.crumb) === path.resolve(SAMPLES), br.crumb);
  check('列出了样本音频', br.names.some(n => n === 'jfk.wav'), br.names.slice(0, 5).join(' | '));

  shot = await send('Page.captureScreenshot', { format: 'png' });
  fs.writeFileSync(path.join(SHOT, '02-browse.png'), Buffer.from(shot.data, 'base64'));

  // 选中 jfk.wav 并添加
  await ev(`(function(){
    const rows = [...document.querySelectorAll('#browseList .brow')];
    const r = rows.find(x => x.querySelector('.nm').textContent === 'jfk.wav');
    r.click(); return true;
  })()`);
  await sleep(200);
  check('添加按钮已启用', await ev(`!document.querySelector('#btnAddSel').disabled`));
  await ev(`document.querySelector('#btnAddSel').click()`);
  await sleep(400);
  const fl = await ev(`(() => ({
    n: document.querySelectorAll('#filelist .fileitem').length,
    name: document.querySelector('#filelist .fileitem .nm')?.textContent || '',
    runEnabled: !document.querySelector('#btnRun').disabled
  }))()`);
  check('文件已加入列表', fl.n === 1 && fl.name === 'jfk.wav', fl.name);
  check('开始按钮已启用', fl.runEnabled);
  check('弹层已关闭', await ev(`getComputedStyle(document.querySelector('#mask')).display === 'none'`));

  console.log('\n6. 设置参数并开始转写');
  await ev(`document.querySelector('#language').value = 'en'`);
  await ev(`document.querySelector('#prompt').value = 'A famous speech by President John F. Kennedy.'`);
  await ev(`document.querySelector('#formats input[value=srt]').checked = true`);
  shot = await send('Page.captureScreenshot', { format: 'png' });
  fs.writeFileSync(path.join(SHOT, '03-ready.png'), Buffer.from(shot.data, 'base64'));

  const t0 = Date.now();
  // 转写可能不到 1 秒就结束，用定时采样记录"运行中"的按钮状态，避免时序假失败
  await ev(`(function(){
    window.__states = [];
    const rec = () => window.__states.push({
      runDisabled: document.querySelector('#btnRun').disabled,
      cancelDisabled: document.querySelector('#btnCancel').disabled,
      progShown: document.querySelector('#progWrap').classList.contains('show'),
    });
    rec();
    window.__recTimer = setInterval(rec, 25);
    return true;
  })()`);
  await ev(`document.querySelector('#btnRun').click()`);
  await sleep(1200);
  await ev(`clearInterval(window.__recTimer); true`);
  const states = await ev(`window.__states`);
  check('运行中：出现过「开始禁用 + 取消可用」',
    states.some(s => s.runDisabled && !s.cancelDisabled),
    states.length + ' 次采样');
  check('运行中：进度区出现过', states.some(s => s.progShown), '');
  check('运行前按钮为初始态', states[0] && !states[0].runDisabled && states[0].cancelDisabled);

  // 等结果
  let done = false;
  for (let i = 0; i < 120; i++) {
    const st = await ev(`document.querySelector('#results .result') ? 1 : 0`);
    if (st === 1) { done = true; break; }
    await sleep(500);
  }
  const secs = ((Date.now() - t0) / 1000).toFixed(1);
  check('转写完成并出现结果卡片', done, secs + 's');

  console.log('\n7. 结果内容');
  const res = await ev(`(() => {
    const c = document.querySelector('#results .result');
    const txt = c.querySelector('.txt')?.textContent || '';
    const tabs = [...c.querySelectorAll('.tabs button')].map(b => b.textContent);
    const segRows = c.querySelectorAll('table.seg tbody tr').length;
    const links = [...c.querySelectorAll('.dl a')].map(a => a.textContent.trim());
    const btns = [...c.querySelectorAll('.dl button')].map(b => b.textContent.trim());
    return { txt, tabs, segRows, links, btns, dlTotal: c.querySelectorAll('.dl > *').length,
             meta: c.querySelector('.meta')?.textContent || '' };
  })()`);
  check('识别文本正确', /ask not what your country can do for you/i.test(res.txt), res.txt.slice(0, 60));
  check('无幻觉尾巴(无 Thank you)', !/thank you/i.test(res.txt), res.txt.slice(-40));
  check('三个标签页', res.tabs.length === 3, res.tabs.join(' | '));
  check('分段表有数据', res.segRows >= 1, res.segRows + ' 行');
  check('下载链接 TXT/SRT 齐全',
    res.links.some(t => /TXT/.test(t)) && res.links.some(t => /SRT/.test(t)), res.links.join(' | '));
  check('复制按钮存在', res.btns.some(t => /复制/.test(t)), res.btns.join(' | '));
  check('结果元信息含实时倍速', /×/.test(res.meta), res.meta);

  console.log('\n8. 完成状态');
  const fin = await ev(`(() => ({
    alert: document.querySelector('#alert').textContent.trim(),
    pct: document.querySelector('#stPct').textContent,
    runEnabled: !document.querySelector('#btnRun').disabled,
    cancelDisabled: document.querySelector('#btnCancel').disabled
  }))()`);
  check('完成提示出现', /成功/.test(fin.alert), fin.alert);
  check('进度 100%', fin.pct === '100%', fin.pct);
  check('按钮状态已复位', fin.runEnabled && fin.cancelDisabled);

  await sleep(500);
  shot = await send('Page.captureScreenshot', { format: 'png', fullPage: true });
  fs.writeFileSync(path.join(SHOT, '04-result.png'), Buffer.from(shot.data, 'base64'));

  console.log('\n9. 切换标签页');
  await ev(`document.querySelectorAll('#results .tabs button')[1].click()`);
  await sleep(400);
  check('分段标签页激活', await ev(`document.querySelectorAll('#results .tabs button')[1].classList.contains('active')`));
  const segTxt = await ev(`(() => {
    const rows = [...document.querySelectorAll('#results table.seg tbody tr')];
    return rows.map(r => r.children[0].textContent + ' :: ' + r.children[1].textContent).join('\\n');
  })()`);
  check('分段含时间戳', /\d{2}:\d{2}:\d{2},\d{3}/.test(segTxt), segTxt.slice(0, 70).replace(/\n/g, ' / '));
  shot = await send('Page.captureScreenshot', { format: 'png', fullPage: true });
  fs.writeFileSync(path.join(SHOT, '05-segments.png'), Buffer.from(shot.data, 'base64'));

  console.log('\n10. 控制台错误');
  const errs = api.events
    .filter(m => m.method === 'Log.entryAdded' && m.params.entry.level === 'error'
              && !/favicon|net::ERR_/.test(m.params.entry.text))
    .map(m => m.params.entry.text);
  check('无 JS 控制台错误', errs.length === 0, errs.slice(0, 3).join(' | '));

  console.log('\n11. 错误处理：非法语言');
  await ev(`(function(){
    const s = document.querySelector('#language');
    const o = document.createElement('option'); o.value = 'txt'; o.textContent = 'txt(非法)';
    s.appendChild(o); s.value = 'txt'; return true;
  })()`);
  await ev(`document.querySelector('#btnClearFiles').click()`);
  await ev(`window.__f = window.S ? null : null`);   // 无操作，仅保持注入简洁
  await ev(`(function(){
    const rows = document.querySelector('#results');
    return true;
  })()`);
  // 通过接口直接验证校验逻辑（UI 层不允许选非法值，这里走 API）
  const vres = await ev(`(async () => {
    const r = await fetch('/api/transcribe', { method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ inputs:[${JSON.stringify(path.join(SAMPLES, 'jfk.wav'))}], language:'txt' }) });
    return { code: r.status, body: await r.json() };
  })()`);
  check('非法语言被服务端拒绝', vres.code === 400, 'HTTP ' + vres.code);
  check('拒绝信息提到 -f', /-f/.test(vres.body.error || ''), (vres.body.error || '').split('\n')[2] || '');

  // ---------------- 汇总 ----------------
  const pass = results.filter(r => r.ok).length;
  const fail = results.filter(r => !r.ok).length;
  console.log('\n' + '='.repeat(62));
  console.log(`  通过 ${pass}   失败 ${fail}`);
  if (fail) { console.log('  失败项：'); results.filter(r => !r.ok).forEach(r => console.log('    - ' + r.name + '  ' + r.detail)); }
  console.log('='.repeat(62));
  process.exitCode = fail ? 1 : 0;

} catch (e) {
  console.error('\n[异常]', e.message);
  process.exitCode = 2;
} finally {
  try { if (child) child.kill(); } catch {}
  try { if (PROF) fs.rmSync(PROF, { recursive: true, force: true }); } catch {}
}
