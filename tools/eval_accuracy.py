#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
精度评测工具 —— 调参用。

用**有已知原文**的样本合成各种劣化（人声嘈杂/混响/窄带/降采样），
客观算 WER，横向比较不同配置。避免"凭感觉调参"。

同时包含静音幻觉对照（这是 Whisper 最致命的坑，见 README）。

用法:
    venv\\Scripts\\python.exe tools\\eval_accuracy.py
    venv\\Scripts\\python.exe tools\\eval_accuracy.py --fast
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

S = ROOT / "samples"
SR = 16000

GROUND = {
    "jfk.wav": ("en", "And so, my fellow Americans, ask not what your country can do for you, "
                      "ask what you can do for your country."),
}


# ---------------------------------------------------------------- 指标
def edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def norm(s: str) -> str:
    s = re.sub(r"[^\w\s']", " ", (s or "").lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def wer(ref: str, hyp: str) -> float:
    r, h = norm(ref).split(), norm(hyp).split()
    return edit_distance(r, h) / len(r) if r else 0.0


# ---------------------------------------------------------------- 劣化
def _p(x):
    return float(np.mean(x ** 2)) or 1e-12


def _mix(sig, noise, snr_db):
    noise = noise * np.sqrt(_p(sig) / (10 ** (snr_db / 10)) / _p(noise))
    out = sig + noise
    return (out / (np.max(np.abs(out)) or 1.0) * 0.9).astype(np.float32)


def babble(n, seed=0, talkers=8):
    """多人同时说话，掩盖语音共振峰，比白噪难得多。"""
    rs = np.random.RandomState(seed)
    acc = np.zeros(n, dtype=np.float32)
    for _ in range(talkers):
        v = rs.randn(n).astype(np.float32)
        X = np.fft.rfft(v)
        f = np.fft.rfftfreq(n, 1 / SR)
        X[(f < 300) | (f > 3400)] = 0
        v = np.fft.irfft(X, n).astype(np.float32)
        env = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * rs.uniform(2, 5) * np.arange(n) / SR))
        acc += v * env
    return acc.astype(np.float32)


def reverb(x, rt60=0.6, seed=0):
    n_ir = int(rt60 * SR)
    t = np.arange(n_ir) / SR
    ir = (np.random.RandomState(seed).randn(n_ir) * np.exp(-6.9 * t / rt60)).astype(np.float32)
    ir[0] = 1.0
    ir /= np.sqrt(np.sum(ir ** 2))
    wet = np.convolve(x, ir)[:len(x)].astype(np.float32)
    out = 0.65 * x + 0.85 * wet
    return (out / (np.max(np.abs(out)) or 1.0) * 0.9).astype(np.float32)


def bandlimit(x, lo=300, hi=3400):
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    h = ((f >= lo) & (f <= hi)).astype(np.float32)
    h[0] = 0
    y = np.fft.irfft(X * h, n).astype(np.float32)
    return (y / (np.max(np.abs(y)) or 1.0) * 0.9).astype(np.float32)


def resample8k(x):
    y = np.fft.rfft(x)[:len(x) // 2]
    z = np.fft.irfft(y, len(x)).astype(np.float32)
    return (z / (np.max(np.abs(z)) or 1.0) * 0.9).astype(np.float32)


def build_variants(clean: np.ndarray, fast: bool) -> list[tuple[str, np.ndarray]]:
    n = len(clean)
    v = [
        ("干净", clean),
        ("babble 5dB", _mix(clean, babble(n, 1), 5)),
        ("babble 0dB", _mix(clean, babble(n, 2), 0)),
        ("babble -5dB", _mix(clean, babble(n, 3), -5)),
        ("混响0.6s+5dB", _mix(reverb(clean, 0.6), babble(n, 4), 5)),
        ("窄带300-3400", bandlimit(clean)),
    ]
    if not fast:
        v += [
            ("窄带+8k降采样", resample8k(bandlimit(clean))),
            ("混响+窄带+0dB", _mix(bandlimit(reverb(clean, 0.8)), babble(n, 5), 0)),
        ]
    return v


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description="Whisper 精度评测")
    ap.add_argument("--fast", action="store_true", help="只跑少量劣化变体")
    ap.add_argument("--device", default="GPU")
    args = ap.parse_args()

    import transcribe as T
    import vad

    src = S / "jfk.wav"
    if not src.exists():
        print("缺少样本 samples/jfk.wav，先跑 tools/get_samples.py")
        return 2
    lang, ref = GROUND["jfk.wav"]
    clean = T.decode_audio(src)
    variants = build_variants(clean, args.fast)

    CONFIGS = [
        ("turbo 无VAD",        dict(model="turbo",   vad=False, prompt=None)),
        ("turbo +VAD",         dict(model="turbo",   vad=True,  prompt=None)),
        ("turbo +VAD +prompt", dict(model="turbo",   vad=True,  prompt="A famous speech by President John F. Kennedy.")),
        ("largev3 无VAD",      dict(model="largev3", vad=False, prompt=None)),
        ("largev3 +VAD",       dict(model="largev3", vad=True,  prompt=None)),
        ("largev3 +VAD +prompt", dict(model="largev3", vad=True,
                                      prompt="A famous speech by President John F. Kennedy.")),
    ]

    pipes: dict[str, object] = {}

    def get_pipe(m):
        if m not in pipes:
            pipes[m] = T.build_pipeline(m, args.device)
        return pipes[m]

    print("=" * 100)
    print(f"  精度评测  ·  {src.name}  ·  设备 {args.device}")
    print(f"  参考: {ref}")
    print("=" * 100)

    table: dict[str, tuple[list[float], float]] = {}
    for cname, cfg in CONFIGS:
        pipe = get_pipe(cfg["model"])
        ws, dur = [], 0.0
        for vname, aud in variants:
            t0 = time.perf_counter()
            try:
                if cfg["vad"]:
                    txt, *_ = T.transcribe_sliced(
                        aud, model=cfg["model"], device=args.device, language=lang,
                        task="transcribe", timestamps=True, prompt=cfg["prompt"],
                        hotwords=None, temperature=None, max_length=448, pipe=pipe)
                else:
                    txt, *_ = T.transcribe_audio(
                        aud, model=cfg["model"], device=args.device, language=lang,
                        task="transcribe", timestamps=True, prompt=cfg["prompt"],
                        hotwords=None, temperature=None, max_length=448, pipe=pipe)
                w = wer(ref, txt)
            except Exception as e:
                w = 1.0
                txt = f"<FAIL {type(e).__name__}>"
            dur += time.perf_counter() - t0
            ws.append(w)
        table[cname] = (ws, dur)

    hdr = f"{'配置':<22}" + "".join(f"{n[:13]:>15}" for n, _ in variants) + f"{'平均':>9}{'耗时':>9}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for cname, (ws, dur) in sorted(table.items(), key=lambda kv: np.mean(kv[1][0])):
        print(f"{cname:<22}" + "".join(f"{w*100:>14.1f}%" for w in ws)
              + f"{np.mean(ws)*100:>8.1f}%{dur:>8.1f}s")
    print("-" * len(hdr))

    best = min(table.items(), key=lambda kv: np.mean(kv[1][0]))
    print(f"\n  最优: {best[0]}  平均 WER {np.mean(best[1][0])*100:.1f}%")

    # ---------------- 静音幻觉对照 ----------------
    print("\n" + "=" * 100)
    print("  静音幻觉对照（30 秒纯静音）—— 这是 VAD 存在的理由")
    print("=" * 100)
    silence = np.zeros(30 * SR, dtype=np.float32)
    for cname, cfg in CONFIGS[:4]:
        if cfg["prompt"]:
            continue
        pipe = get_pipe(cfg["model"])
        try:
            if cfg["vad"]:
                txt, *_ = T.transcribe_sliced(
                    silence, model=cfg["model"], device=args.device, language="zh",
                    task="transcribe", timestamps=True, prompt=None, hotwords=None,
                    temperature=None, max_length=448, pipe=pipe)
            else:
                txt, *_ = T.transcribe_audio(
                    silence, model=cfg["model"], device=args.device, language="zh",
                    task="transcribe", timestamps=True, prompt=None, hotwords=None,
                    temperature=None, max_length=448, pipe=pipe)
            mark = "✔ 无幻觉" if not txt.strip() else "✗ 出现幻觉文本"
            print(f"  {cname:<22} {mark}   {txt[:60]!r}" if txt.strip() else f"  {cname:<22} {mark}")
        except Exception as e:
            print(f"  {cname:<22} FAIL {type(e).__name__}: {str(e)[:50]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
