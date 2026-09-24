#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Whisper 本地转写 —— Web 界面服务端。

只用 Python 标准库（http.server），不引入任何新依赖。
启动后浏览器打开 http://127.0.0.1:<port> 即可点选操作，命令行用法完全保留。

设计要点
--------
- **只监听 127.0.0.1**，不对外网开放（这是本机 GPU 工具，不是网络服务）。
- **任务串行**：Whisper pipeline 是单例且占显存，同一时刻只跑一个任务。
- **进度用 SSE 推送**：转写线程把事件写进队列，SSE 处理器消费。
- **取消**：置标志位，转写循环在 VAD 分片之间检查（不会硬杀线程）。

接口
----
  GET  /                       页面
  GET  /static/<file>          静态资源
  GET  /api/state              环境信息（设备/模型/档位/语言/默认值）
  GET  /api/browse?path=       目录浏览（给文件选择器用）
  POST /api/upload             上传文件（拖拽用）
  POST /api/transcribe         提交任务
  GET  /api/events?job=<id>    SSE 进度流
  POST /api/cancel             取消任务
  GET  /api/download?job=&i=   下载结果文件
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import queue
import socket
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, quote, unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent          # 项目根目录
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

STATIC = Path(__file__).resolve().parent / "static"
UPLOADS = ROOT / "uploads"

import transcribe as T      # noqa: E402
import vad as VAD           # noqa: E402

HOST = "127.0.0.1"
DEFAULT_PORT = 8765

# ---------------------------------------------------------------- 任务管理
_jobs: dict[str, "Job"] = {}
_jobs_lock = threading.Lock()
_run_lock = threading.Lock()          # 保证同一时刻只有一个转写任务在跑


class Job:
    def __init__(self, payload: dict):
        self.id = uuid.uuid4().hex[:12]
        self.payload = payload
        self.status = "queued"           # queued|running|done|cancelled|error
        self.q: queue.Queue = queue.Queue()
        self.cancelled = False
        self.results: list[dict] = []
        self.summary: dict = {}
        self.created = time.time()
        self.outputs: list[Path] = []
        self.error: str | None = None

    def emit(self, ev: dict) -> bool:
        """由转写线程调用。返回 False 表示要求中止。"""
        if ev.get("stage") == "done" and ev.get("outputs"):
            # 告诉前端这批产出在全局列表里的起始下标，好拼下载链接
            ev["output_index"] = len(self.outputs)
            self.outputs.extend(ev["outputs"])
        self.q.put(ev)
        return not self.cancelled

    def should_stop(self) -> bool:
        return self.cancelled

    def close(self) -> None:
        self.q.put(None)                 # 哨兵，通知 SSE 结束


# ---------------------------------------------------------------- 参数构建
def make_args(p: dict) -> SimpleNamespace:
    """把前端 JSON 转成 process_file 需要的 args 命名空间。"""
    accuracy = p.get("accuracy") or "high"
    preset = T.ACCURACY_PRESETS.get(accuracy, T.ACCURACY_PRESETS["high"])
    model = p.get("model") or preset["model"]
    vad_on = p.get("vad")
    if vad_on is None:
        vad_on = preset["vad"]

    out_dir = (p.get("output_dir") or "").strip()
    formats = p.get("formats") or ["txt"]
    if not isinstance(formats, list):
        formats = [formats]

    lang = (p.get("language") or "auto").strip().lower()
    lang = T.LANG_ALIAS.get(lang, lang)

    return SimpleNamespace(
        model=model,
        device=(p.get("device") or "GPU").strip(),
        language=lang,
        task=p.get("task") or "transcribe",
        accuracy=accuracy,
        vad=bool(vad_on),
        vad_min_silence=int(p.get("vad_min_silence") or 300),
        vad_merge_gap=int(p.get("vad_merge_gap") or 2500),
        vad_max_block=float(p.get("vad_max_block") or 28.0),
        keep_hallucination=bool(p.get("keep_hallucination")),
        prompt=(p.get("prompt") or "").strip() or None,
        hotwords=(p.get("hotwords") or "").strip() or None,
        temperature=None,
        max_length=448,
        no_timestamps=not bool(p.get("timestamps", True)),
        word_timestamps=False,
        output_dir=out_dir or None,
        format=formats,
        timestamp_names=bool(p.get("timestamp_names")),
        recursive=bool(p.get("recursive")),
        show_text=False,
        quiet=True,
    )


def validate(p: dict) -> str | None:
    args = make_args(p)
    err = T.check_language(args.language, args.model)
    if err:
        return err
    err = T.check_device(args.device)
    if err:
        return err
    if args.vad and not (T.VAD_DIR / VAD.MODEL_NAME).exists():
        return ("✗ 找不到 VAD 模型，无法开启静音门控。\n"
                "  获取：venv\\Scripts\\python.exe tools\\download_models.py vad\n"
                "  或在前端关闭 VAD。")
    inputs = p.get("inputs") or []
    if not inputs:
        return "✗ 没有选择任何输入文件。"
    missing = [s for s in inputs if not Path(s).exists()]
    if missing:
        return "✗ 以下路径不存在：\n  " + "\n  ".join(missing[:5])
    return None


# ---------------------------------------------------------------- 转写线程
def run_job(job: Job) -> None:
    p = job.payload
    args = make_args(p)
    with _run_lock:
        if job.cancelled:
            job.status = "cancelled"
            job.close()
            return
        job.status = "running"
        try:
            files = T.collect_inputs(p["inputs"], args.recursive)
            if not files:
                job.emit({"stage": "fatal", "message": "没有找到可处理的音视频文件。"})
                job.status = "error"
                return

            job.emit({"stage": "start", "total": len(files),
                      "model": args.model, "device": args.device,
                      "language": args.language, "vad": args.vad,
                      "files": [str(f) for f in files]})

            t0 = time.perf_counter()
            job.emit({"stage": "loading", "model": args.model, "device": args.device})
            try:
                pipe = T.build_pipeline(args.model, args.device)
            except Exception as e:
                job.emit({"stage": "fatal",
                          "message": f"模型/设备初始化失败：{type(e).__name__}: {e}"})
                job.status = "error"
                return
            job.emit({"stage": "loaded",
                      "seconds": round(time.perf_counter() - t0, 2)})

            stems = T.assign_stems(files)
            ok = fail = 0
            partial_n = 0
            for i, f in enumerate(files, 1):
                if job.cancelled:
                    break
                job.emit({"stage": "file_begin", "index": i, "total": len(files),
                          "name": f.name, "path": str(f)})
                try:
                    r = T.process_file(f, args=args, pipe=pipe, quiet=True,
                                       stem=stems.get(f),
                                       progress=job.emit,
                                       should_stop=job.should_stop)
                except Exception as e:
                    job.emit({"stage": "error", "file": str(f),
                              "message": f"{type(e).__name__}: {e}"})
                    traceback.print_exc()
                    r = None
                if r is None:
                    fail += 1
                    job.emit({"stage": "file_end", "index": i, "ok": False,
                              "name": f.name})
                else:
                    ok += 1
                    partial = bool(r.perf.get("cancelled"))
                    if partial:
                        partial_n += 1
                    job.results.append({
                        "name": f.name, "path": str(f),
                        "text": r.text, "language": r.language,
                        "partial": partial,
                        "audio_seconds": round(r.audio_seconds, 3),
                        "elapsed": round(r.elapsed, 3),
                        "speed": round(r.audio_seconds / r.elapsed, 2)
                        if r.elapsed > 0 else None,
                        "segments": [{"start": round(s.start, 3),
                                      "end": round(s.end, 3), "text": s.text}
                                     for s in r.segments],
                        "perf": {k: v for k, v in r.perf.items()
                                 if not k.startswith("_")},
                    })
                    job.emit({"stage": "file_end", "index": i, "ok": True,
                              "name": f.name, "partial": partial})

            total_audio = sum(r["audio_seconds"] for r in job.results)
            total_time = time.perf_counter() - t0
            job.summary = {
                "ok": ok, "fail": fail, "total": len(files),
                "partial": partial_n,
                "audio_seconds": round(total_audio, 2),
                "elapsed": round(total_time, 2),
                "speed": round(total_audio / total_time, 2) if total_time > 0 else None,
                "cancelled": job.cancelled,
            }
            job.status = "cancelled" if job.cancelled else (
                "done" if fail == 0 else "done")
            job.emit({"stage": "finish", **job.summary})
        except Exception as e:
            traceback.print_exc()
            job.error = f"{type(e).__name__}: {e}"
            job.status = "error"
            job.emit({"stage": "fatal", "message": job.error})
        finally:
            job.close()


# ---------------------------------------------------------------- 目录浏览
AUDIO_EXT = T.AUDIO_EXT


def list_dir(path_str: str) -> dict:
    """目录浏览。path_str 为空时列出所有盘符。"""
    if not path_str:
        drives = []
        for d in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            p = Path(f"{d}:\\")
            try:
                if p.exists():
                    drives.append({"name": f"{d}:", "path": str(p)})
            except OSError:
                pass
        return {"path": "", "parent": None, "dirs": drives, "files": []}

    p = Path(path_str)
    if p.is_file():
        p = p.parent
    if not p.exists():
        return {"path": str(p), "parent": None, "dirs": [], "files": [],
                "error": "路径不存在"}

    dirs, files = [], []
    try:
        for entry in sorted(p.iterdir(), key=lambda x: x.name.lower()):
            if entry.name.startswith("$") or entry.name.startswith("."):
                continue
            try:
                if entry.is_dir():
                    dirs.append({"name": entry.name, "path": str(entry)})
                elif entry.suffix.lower() in AUDIO_EXT:
                    files.append({"name": entry.name, "path": str(entry),
                                  "size": entry.stat().st_size})
            except OSError:
                continue
    except PermissionError:
        return {"path": str(p), "parent": str(p.parent), "dirs": [], "files": [],
                "error": "没有访问权限"}

    return {"path": str(p), "parent": str(p.parent) if p.parent != p else None,
            "dirs": dirs, "files": files}


# ---------------------------------------------------------------- HTTP 处理
class Handler(BaseHTTPRequestHandler):
    server_version = "WhisperUI/1.0"

    def log_message(self, fmt, *a):
        pass                                    # 静音，避免刷屏

    # ---- 工具 ----
    def _send(self, code: int, body: bytes, ctype: str,
              extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            pass

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _err(self, msg: str, code: int = 400) -> None:
        self._json({"error": msg}, code)

    def _body_json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        return json.loads(self.rfile.read(n).decode("utf-8"))

    # ---- GET ----
    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path in ("/", "/index.html"):
                return self._serve_static("index.html")
            if u.path.startswith("/static/"):
                return self._serve_static(u.path[len("/static/"):])
            if u.path == "/api/state":
                return self._json(api_state())
            if u.path == "/api/browse":
                return self._json(list_dir((q.get("path") or [""])[0]))
            if u.path == "/api/events":
                return self._sse((q.get("job") or [""])[0])
            if u.path == "/api/download":
                return self._download((q.get("job") or [""])[0],
                                      int((q.get("i") or ["0"])[0]))
            if u.path == "/api/job":
                return self._job_status((q.get("job") or [""])[0])
            return self._err("未知路径", 404)
        except BrokenPipeError:
            pass
        except Exception as e:
            traceback.print_exc()
            try:
                self._err(f"{type(e).__name__}: {e}", 500)
            except Exception:
                pass

    # ---- POST ----
    def do_POST(self):
        u = urlparse(self.path)
        try:
            if u.path == "/api/transcribe":
                return self._start_job()
            if u.path == "/api/cancel":
                return self._cancel()
            if u.path == "/api/upload":
                return self._upload()
            return self._err("未知路径", 404)
        except Exception as e:
            traceback.print_exc()
            self._err(f"{type(e).__name__}: {e}", 500)

    # ---- 静态文件 ----
    def _serve_static(self, rel: str):
        target = (STATIC / rel).resolve()
        if not str(target).startswith(str(STATIC.resolve())) or not target.is_file():
            return self._err("文件不存在", 404)
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript",):
            ctype += "; charset=utf-8"
        self._send(200, target.read_bytes(), ctype)

    # ---- 任务 ----
    def _start_job(self):
        p = self._body_json()
        err = validate(p)
        if err:
            return self._json({"error": err}, 400)
        job = Job(p)
        with _jobs_lock:
            _jobs[job.id] = job
        threading.Thread(target=run_job, args=(job,), daemon=True).start()
        self._json({"job": job.id})

    def _job_status(self, jid: str):
        job = _jobs.get(jid)
        if not job:
            return self._err("任务不存在", 404)
        self._json({"status": job.status, "results": job.results,
                    "summary": job.summary, "error": job.error})

    def _cancel(self):
        p = self._body_json()
        job = _jobs.get(p.get("job"))
        if not job:
            return self._err("任务不存在", 404)
        job.cancelled = True
        self._json({"ok": True})

    def _upload(self):
        """拖拽上传：body 是原始字节，文件名走查询串或头。"""
        u = urlparse(self.path)
        q = parse_qs(u.query)
        name = (q.get("name") or ["upload.bin"])[0]
        name = Path(unquote(name)).name
        if not name or Path(name).suffix.lower() not in AUDIO_EXT:
            return self._err(f"不支持的文件类型：{name}")
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return self._err("空文件")
        UPLOADS.mkdir(parents=True, exist_ok=True)
        dst = UPLOADS / name
        stem, suf, k = dst.stem, dst.suffix, 1
        while dst.exists():
            dst = UPLOADS / f"{stem}_{k}{suf}"
            k += 1
        remaining = n
        with open(dst, "wb") as f:
            while remaining > 0:
                chunk = self.rfile.read(min(1 << 20, remaining))
                if not chunk:
                    break
                f.write(chunk)
                remaining -= len(chunk)
        self._json({"path": str(dst), "name": dst.name})

    # ---- SSE ----
    def _sse(self, jid: str):
        job = _jobs.get(jid)
        if not job:
            return self._err("任务不存在", 404)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def send(obj):
            data = json.dumps(obj, ensure_ascii=False)
            self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
            self.wfile.flush()

        alive = True
        try:
            send({"stage": "hello", "status": job.status})
            while alive:
                try:
                    ev = job.q.get(timeout=15)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")      # 心跳，防代理断连
                    self.wfile.flush()
                    continue
                if ev is None:
                    alive = False
                    send({"stage": "eof", "status": job.status,
                          "summary": job.summary,
                          "results": [{"name": r["name"]} for r in job.results]})
                    break
                send(ev)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError,
                OSError):
            pass                                  # 浏览器关了页面

    # ---- 下载 ----
    def _download(self, jid: str, idx: int):
        job = _jobs.get(jid)
        if not job:
            return self._err("任务不存在", 404)
        if idx < 0 or idx >= len(job.outputs):
            return self._err("文件序号越界", 404)
        f = Path(job.outputs[idx])
        if not f.is_file():
            return self._err("产出文件已被移动或删除", 404)
        ctype = mimetypes.guess_type(str(f))[0] or "application/octet-stream"
        if ctype.startswith("text/"):
            ctype += "; charset=utf-8"
        ascii_name = f.name.encode("ascii", "ignore").decode() or "result.txt"
        self._send(200, f.read_bytes(), ctype, extra={
            "Content-Disposition":
                f'attachment; filename="{ascii_name}"; '
                f"filename*=UTF-8''{quote(f.name)}"
        })


# ---------------------------------------------------------------- 状态接口
def api_state() -> dict:
    try:
        import openvino as ov
        core = ov.Core()
        devs = list(core.available_devices)
        gpu_name = ""
        if any(d.startswith("GPU") for d in devs):
            try:
                gpu_name = core.get_property("GPU", "FULL_DEVICE_NAME")
            except Exception:
                gpu_name = "GPU"
    except Exception:
        devs, gpu_name = [], ""

    models = []
    for k, spec in T.MODELS.items():
        models.append({
            "key": k, "label": spec.label, "size": spec.size,
            "ready": (T.MODELS_DIR / spec.dirname
                      / "openvino_encoder_model.xml").exists(),
        })

    # 语言表：合并两个模型的语言码（其实一致）
    langs = sorted(T.model_languages("largev3") or T.model_languages("turbo"))
    lang_opts = [{"code": "auto", "label": "自动检测"}] + [
        {"code": c, "label": T.LANG_HINTS.get(c, c)} for c in langs
    ]

    vad_ready = (T.VAD_DIR / VAD.MODEL_NAME).exists()

    return {
        "devices": devs,
        "gpu_name": gpu_name,
        "models": models,
        "presets": [{"key": k, "desc": v["desc"]}
                    for k, v in T.ACCURACY_PRESETS.items()],
        "languages": lang_opts,
        "vad_ready": vad_ready,
        "formats": ["txt", "srt", "vtt", "lrc", "json"],
        "root": str(ROOT),
        "uploads": str(UPLOADS),
        "python": sys.executable.split("\\")[-1],
    }


# ---------------------------------------------------------------- 启动
def find_port(preferred: int) -> int:
    for p in [preferred] + list(range(preferred + 1, preferred + 20)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((HOST, p))
                return p
            except OSError:
                continue
    raise RuntimeError("找不到可用端口")


def main() -> int:
    ap = argparse.ArgumentParser(description="Whisper 本地转写 Web 界面")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = ap.parse_args()

    port = find_port(args.port)
    url = f"http://{HOST}:{port}/"

    # 预热：把 pipeline 用到的东西先导入，避免首次请求卡顿
    st = api_state()

    print()
    print("  " + "=" * 58)
    print("   Whisper 本地转写 · Web 界面")
    print("  " + "=" * 58)
    if st["gpu_name"]:
        print(f"   显卡    {st['gpu_name']}")
    else:
        print("   显卡    未检测到 GPU，将使用 CPU")
    print(f"   设备    {', '.join(st['devices']) or '无'}")
    print(f"   模型    " + ", ".join(
        f"{m['key']}{'' if m['ready'] else '(未下载)'}" for m in st["models"]))
    print(f"   VAD     {'就绪' if st['vad_ready'] else '缺失'}")
    print()
    print(f"   地址    {url}")
    print("   停止    Ctrl+C")
    print("  " + "=" * 58)
    print()

    httpd = ThreadingHTTPServer((HOST, port), Handler)
    httpd.daemon_threads = True
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  已停止。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
