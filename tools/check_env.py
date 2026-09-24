#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""环境自检：确认 OpenVINO、GPU、模型、音频解码链路都正常。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

OK, BAD, WARN = "[OK]  ", "[FAIL]", "[WARN]"


def line(tag: str, msg: str) -> None:
    print(f"  {tag} {msg}")


def section(t: str) -> None:
    print(f"\n{t}")
    print("-" * 62)


def main() -> int:
    problems = 0

    print("=" * 62)
    print("  Whisper / OpenVINO 环境自检")
    print("=" * 62)

    # 1. 基础包
    section("1. Python 依赖")
    line(OK, f"Python {sys.version.split()[0]}  ({sys.executable})")
    for mod in ("numpy", "av", "openvino", "openvino_genai", "requests"):
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            line(OK, f"{mod:<16} {ver}")
        except Exception as e:
            line(BAD, f"{mod:<16} 导入失败: {e}")
            problems += 1

    # 2. OpenVINO 设备
    section("2. OpenVINO 计算设备")
    try:
        import openvino as ov
        core = ov.Core()
        devs = core.available_devices
        line(OK, f"可用设备: {devs}")
        if not any(d.startswith("GPU") for d in devs):
            line(WARN, "没有检测到 GPU 设备 —— 将只能跑 CPU")
            line(WARN, "请安装/更新 Intel 显卡驱动（Arc 需要 31.0.101.4xxx 以上）")
            problems += 1
        for d in devs:
            if not d.startswith("GPU"):
                continue
            name = core.get_property(d, "FULL_DEVICE_NAME")
            line(OK, f"{d}: {name}")
            try:
                caps = core.get_property(d, "OPTIMIZATION_CAPABILITIES")
                line(OK, f"{d} 支持: {', '.join(caps)}")
                if "FP16" not in caps:
                    line(WARN, f"{d} 不支持 FP16，性能会受影响")
            except Exception:
                pass
    except Exception as e:
        line(BAD, f"OpenVINO 初始化失败: {e}")
        problems += 1

    # 3. 模型
    section("3. 本地模型")
    try:
        from transcribe import MODELS  # noqa
        names = MODELS
    except Exception:
        names = {k: None for k in ("turbo", "largev3")}
    mdl_dir = MODELS_DIR
    found = 0
    if mdl_dir.exists():
        for d in sorted(mdl_dir.iterdir()):
            if not d.is_dir():
                continue
            enc = d / "openvino_encoder_model.xml"
            dec = d / "openvino_decoder_model.xml"
            if enc.exists() and dec.exists():
                mb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1048576
                line(OK, f"{d.name:<38} {mb:7.0f} MB")
                found += 1
            elif list(d.glob("openvino_*.xml")):
                # 有部分 IR 文件但不成套 -> 真的不完整
                line(BAD, f"{d.name:<38} 文件不完整")
                problems += 1
            # 其它目录（如 silero-vad）不在这里报，由第 4 节单独检查
    if found == 0:
        line(BAD, "没有任何可用模型")
        line(BAD, "请运行: venv\\Scripts\\python.exe tools\\download_models.py")
        problems += 1

    # 4. VAD
    section("4. VAD 静音门控 (Silero)")
    try:
        import vad
        mp = MODELS_DIR / "silero-vad" / vad.MODEL_NAME
        if mp.exists():
            line(OK, f"{vad.MODEL_NAME}  {mp.stat().st_size / 1024:.0f} KB")
            import numpy as np
            t0 = time.perf_counter()
            sil_segs = vad.speech_segments(np.zeros(16000 * 10, dtype=np.float32),
                                           MODELS_DIR / "silero-vad")
            dt = (time.perf_counter() - t0) * 1000
            if sil_segs:
                line(BAD, f"静音竟被判为 {len(sil_segs)} 段语音，VAD 异常")
                problems += 1
            else:
                line(OK, f"静音判定正常（0 段），10s 耗时 {dt:.0f}ms")
        else:
            line(BAD, f"缺少 {vad.MODEL_NAME}")
            line(BAD, "获取: venv\\Scripts\\python.exe tools\\download_models.py vad")
            problems += 1
    except Exception as e:
        line(BAD, f"VAD 不可用: {type(e).__name__}: {e}")
        problems += 1

    # 5. 音频解码
    section("5. 音频解码 (PyAV)")
    try:
        import av
        line(OK, f"PyAV {av.__version__}")
        names = []
        try:
            names = list(av.formats_available)
        except Exception:
            try:
                names = list(av.format.ContainerFormat.formats_available)
            except Exception:
                names = []
        if names:
            pick = [n for n in ("wav", "mp3", "flac", "mp4", "mkv", "ogg") if n in names]
            line(OK, f"容器格式 {len(names)} 种，常用: {', '.join(pick) or names[:6]}")
        else:
            line(OK, "容器格式列表不可用（不影响转写）")
        # 实测解码
        from transcribe import AUDIO_EXT
        line(OK, f"支持扩展名 {len(AUDIO_EXT)} 种")
    except Exception as e:
        line(BAD, f"PyAV 不可用: {e}")
        problems += 1

    # 6. 端到端冒烟测试
    section("6. 端到端冒烟测试")
    candidate = None
    for d in sorted(mdl_dir.iterdir()) if mdl_dir.exists() else []:
        if (d / "openvino_encoder_model.xml").exists():
            from transcribe import MODELS as _M
            for k, v in _M.items():
                if v.dirname == d.name:
                    candidate = k
                    break
        if candidate:
            break

    if candidate:
        try:
            import numpy as np
            from transcribe import build_pipeline, transcribe_audio
            audio = (np.random.RandomState(0).randn(16000 * 2) * 0.01).astype(np.float32)
            for dev in ("GPU", "CPU"):
                try:
                    pipe = build_pipeline(candidate, dev)
                    txt, segs, _w, perf, _lg = transcribe_audio(
                        audio, model=candidate, device=dev, pipe=pipe,
                        timestamps=False, language="en",
                    )
                    enc = perf.get("encode_ms", 0)
                    line(OK, f"{candidate} @ {dev}: 推理正常 "
                             f"(encode {enc:.0f}ms, 输出 {len(txt)} 字符)")
                    break
                except Exception as e:
                    line(WARN, f"{candidate} @ {dev} 失败: {type(e).__name__}: {e}")
        except Exception as e:
            line(BAD, f"冒烟测试异常: {type(e).__name__}: {e}")
            problems += 1
    else:
        line(WARN, "无模型，跳过冒烟测试")

    print("\n" + "=" * 62)
    if problems == 0:
        print("  结论: 环境正常，可以直接用 whisper.bat 转写。")
    else:
        print(f"  结论: 发现 {problems} 个问题，请按上面提示处理。")
    print("=" * 62)
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
