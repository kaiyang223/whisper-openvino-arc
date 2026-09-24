#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Silero VAD —— 语音活动检测（OpenVINO 版）。

用途：在送进 Whisper 之前先把 **非语音段** 剔掉。
这是抑制 Whisper「静音幻觉」的关键手段——实测纯静音输入会让 Whisper
凭空输出 "Thank you."，中文更会输出
"请不吝点赞 订阅 转发 打赏支持明镜与点点栏目" 这类训练集残留文本，
而且 initial_prompt / temperature=0 都压不住。

模型用的是官方 silero-vad wheel 里专门为 OpenVINO 导出的
`silero_vad_openvino_16k.onnx`（1.2MB，静态形状，无 sr 输入）。
注意：HuggingFace/master 分支那个 `silero_vad.onnx` 是动态量化版，
带 If 分支和动态 shape，OpenVINO 的 ONNX 前端读不了，别用。

调用约定（与官方 utils_vad.OnnxWrapper 一致）：
  input  [1, 576]  = 前 64 点上下文 + 当前 512 点（32ms@16k）
  state  [2, 1, 128]  跨窗隐状态，每段音频开始前清零
  output [1, 1]       语音概率
  stateN [2, 1, 128]  下一窗的 state

设备选择：VAD 放 **CPU**。实测 11 秒音频 CPU 60ms / GPU 448ms，
小模型在 GPU 上被逐次调用开销拖垮，CPU 反而快 7 倍，还省下 GPU 给 Whisper。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SR = 16000
CHUNK = 512           # 32ms @16k，Silero v5 固定窗口
CONTEXT = 64          # 官方 wrapper 的上下文长度
MODEL_NAME = "silero_vad_16k_ov.onnx"

_compiled = None


def _model_path(model_dir: Path) -> Path:
    p = model_dir / MODEL_NAME
    if not p.exists():
        raise FileNotFoundError(
            f"找不到 VAD 模型: {p}\n"
            f"需从 silero-vad wheel 提取 silero_vad_openvino_16k.onnx 并重命名。"
        )
    return p


def _get_compiled(model_dir: Path):
    global _compiled
    if _compiled is None:
        import openvino as ov
        core = ov.Core()
        model = core.read_model(str(_model_path(model_dir)))
        _compiled = core.compile_model(model, "CPU")
    return _compiled


def speech_probs(audio: np.ndarray, model_dir: Path) -> np.ndarray:
    """逐 32ms 窗口返回语音概率。"""
    cm = _get_compiled(model_dir)
    n = len(audio)
    n_chunks = int(np.ceil(n / CHUNK))
    pad = n_chunks * CHUNK - n
    if pad:
        audio = np.concatenate([audio, np.zeros(pad, dtype=np.float32)])

    ctx = np.zeros((1, CONTEXT), dtype=np.float32)
    state = np.zeros((2, 1, 128), dtype=np.float32)
    probs = np.empty(n_chunks, dtype=np.float32)

    for i in range(n_chunks):
        ch = audio[i * CHUNK:(i + 1) * CHUNK].reshape(1, CHUNK)
        x = np.concatenate([ctx, ch], axis=1).astype(np.float32)
        r = cm({"input": x, "state": state})
        probs[i] = float(r["output"][0][0])
        state = r["stateN"]
        ctx = x[:, -CONTEXT:]
    return probs


def speech_segments(
    audio: np.ndarray,
    model_dir: Path,
    threshold: float = 0.5,
    neg_threshold: float | None = None,
    min_speech_ms: int = 250,
    min_silence_ms: int = 180,
    pad_ms: int = 150,
) -> list[tuple[float, float]]:
    """
    返回语音段 [(start_sec, end_sec), ...]：已合并短间隔、丢弃过短段、两侧留白。

      threshold       语音判定门限（越高越保守）
      neg_threshold   退出语音的门限，默认 threshold-0.15，形成迟滞防抖
      min_speech_ms   短于此长度的语音段丢弃
      min_silence_ms  静音持续超过此长度才判为一段结束
      pad_ms          每段前后补留白，避免切掉字头字尾
    """
    if neg_threshold is None:
        neg_threshold = max(threshold - 0.15, 0.01)

    total_sec = len(audio) / SR
    probs = speech_probs(audio, model_dir)

    win_ms = CHUNK / SR * 1000.0                     # 32ms
    min_speech_chunks = max(1, int(min_speech_ms / win_ms))
    min_silence_chunks = max(1, int(min_silence_ms / win_ms))

    raw: list[list[int]] = []
    cur: list[int] | None = None
    silence_run = 0
    for i, p in enumerate(probs):
        if cur is None:
            if p >= threshold:
                cur = [i, i + 1]
                silence_run = 0
        else:
            if p >= neg_threshold:
                cur[1] = i + 1
                silence_run = 0
            else:
                silence_run += 1
                if silence_run >= min_silence_chunks:
                    raw.append(cur)
                    cur = None
                    silence_run = 0
    if cur is not None:
        raw.append(cur)

    merged: list[list[int]] = []
    for s in raw:
        if s[1] - s[0] < min_speech_chunks:
            continue
        if merged and s[0] - merged[-1][1] <= min_silence_chunks:
            merged[-1][1] = s[1]
        else:
            merged.append(s)

    pad_c = int(pad_ms / win_ms)
    out: list[tuple[float, float]] = []
    for s, e in merged:
        a = max(0, s - pad_c) * win_ms / 1000.0
        b = min(len(probs), e + pad_c) * win_ms / 1000.0
        b = min(b, total_sec)
        if b > a:
            out.append((a, b))
    return out


def speech_ratio(audio: np.ndarray, model_dir: Path, **kw) -> float:
    segs = speech_segments(audio, model_dir, **kw)
    return sum(e - s for s, e in segs) / max(len(audio) / SR, 1e-6)


def transcribe_slices(
    audio: np.ndarray,
    model_dir: Path,
    threshold: float = 0.5,
    min_speech_ms: int = 250,
    min_silence_ms: int = 300,
    merge_gap_ms: int = 2500,
    max_block_sec: float = 28.0,
    min_block_sec: float = 0.4,
    pad_ms: int = 150,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """
    返回 (转写块, 纯语音区间)。

    转写块 = 送 Whisper 的**非重叠**切片；纯语音区间 = 未补白的真实语音范围，
    用来做第二道门控：Whisper 在补白区（静音）常会凭空吐字，
    凡落在纯语音区间之外的识别段一律丢弃。

    策略（和单纯切语音段不同，这是为转写质量做的取舍）：
      1. 先按 VAD 找出语音段 —— 目的是**剔除长静音**，避免静音幻觉；
      2. 把间隔 < merge_gap_ms 的相邻段合并成大块 —— 保住句子上下文，
         切太碎会显著掉精度；
      3. 超过 max_block_sec 的块再切开（Whisper 内部是 30s 窗口）；
      4. 两侧补 pad_ms 留白，但**夹在相邻块的中点**，保证绝不重叠。
    """
    raw = speech_segments(audio, model_dir, threshold=threshold,
                          min_speech_ms=min_speech_ms,
                          min_silence_ms=min_silence_ms, pad_ms=0)
    if not raw:
        return [], []

    total = len(audio) / SR
    gap = merge_gap_ms / 1000.0

    # 2) 合并短间隔
    blocks: list[list[float]] = []
    for s, e in raw:
        if blocks and s - blocks[-1][1] < gap:
            blocks[-1][1] = e
        else:
            blocks.append([s, e])

    speech_regions = [(round(s, 4), round(e, 4)) for s, e in blocks]

    # 3) 切超长块
    split: list[list[float]] = []
    for s, e in blocks:
        while e - s > max_block_sec:
            split.append([s, s + max_block_sec])
            s += max_block_sec
        if e - s > 0:
            split.append([s, e])

    # 4) 补留白 + 夹中点防重叠
    pad = pad_ms / 1000.0
    out: list[tuple[float, float]] = []
    for i, (s, e) in enumerate(split):
        a = max(0.0, s - pad)
        b = min(total, e + pad)
        if i > 0:
            a = max(a, (split[i - 1][1] + s) / 2.0)
        if i < len(split) - 1:
            b = min(b, (e + split[i + 1][0]) / 2.0)
        if b - a >= min_block_sec:
            out.append((round(a, 4), round(b, 4)))

    # 最终保险：强制单调不重叠
    fixed: list[tuple[float, float]] = []
    prev_end = -1.0
    for a, b in out:
        a = max(a, prev_end)
        if b > a + 0.05:
            fixed.append((a, b))
            prev_end = b
    return fixed, speech_regions


def overlap_ratio(a: float, b: float, regions: list[tuple[float, float]]) -> float:
    """[a,b] 与 regions 的重叠占自身长度的比例。"""
    if b <= a:
        return 0.0
    total = 0.0
    for s, e in regions:
        lo, hi = max(a, s), min(b, e)
        if hi > lo:
            total += hi - lo
    return total / (b - a)


__all__ = ["speech_probs", "speech_segments", "speech_ratio", "transcribe_slices",
           "overlap_ratio", "SR", "CHUNK", "MODEL_NAME"]
