#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Whisper 高性能本地转写 —— Intel Arc A770 / OpenVINO GenAI
=========================================================

在 Intel Arc 独显上通过 OpenVINO 运行 Whisper large-v3 / large-v3-turbo，
把音频或视频里的话转成文字，支持 txt / srt / vtt / json 输出。

示例
----
    # 转单个文件（自动识别语言，输出同名 .txt）
    python transcribe.py 会议录音.mp3

    # 要字幕，用最高精度模型
    python transcribe.py 讲座.mp4 -m largev3 --format srt

    # 整个目录批量转，中文，输出 srt
    python transcribe.py "D:\\素材" -l zh --format srt -r

    # 把英文音频翻译成英文文本（Whisper 的 translate 任务）
    python transcribe.py talk.wav --task translate

    # 性能基准
    python transcribe.py 样本.wav --bench
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# ---- 让中文在 Windows 控制台正常显示 ----
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
VAD_DIR = MODELS_DIR / "silero-vad"
sys.path.insert(0, str(ROOT / "tools"))

SAMPLE_RATE = 16000

# ---------------------------------------------------------------- 幻觉过滤
# Whisper 在非语音段会凭空编造文本（训练集残留）。VAD 是第一道防线，
# 这里是第二道，用于 VAD 判为语音但其实是音乐/噪声的情况。
_HALLU_MARKERS = (
    "amara.org", "明镜与点点", "点点栏目", "字幕志愿者", "字幕由",
    "subtitle", "subtitles by", "subscribe to my channel",
)
# 整段就是这些内容的，视为幻觉（归一化后精确比较）
_HALLU_WHOLE = {
    "thankyou", "thanks", "youthankyou", "pleasesubscribe", "thankyouforwatching",
    "thanksforwatching", "you", "谢谢大家", "谢谢观看", "谢谢",
    "请不吝点赞订阅转发打赏支持明镜与点点栏目",
    "請不吝點贊訂閱轉發打賞支持明鏡與點點欄目",
    "请不吝点赞订阅转发打赏支持明镜与点点栏目",
}


def _norm_key(t: str) -> str:
    return re.sub(r"[\s,.，。!！?？、:：;；'\"“”‘’]+", "", (t or "").lower())


def is_suspicious(text: str) -> str | None:
    """返回可疑原因；None 表示正常。"""
    t = (text or "").strip()
    if not t:
        return "空输出"
    low = t.lower()
    for m in _HALLU_MARKERS:
        if m in low:
            return f"命中幻觉标记「{m}」"
    if _norm_key(t) in _HALLU_WHOLE:
        return "整段为已知幻觉短句"
    # 重复循环检测（4-gram 去重率过低）
    w = _norm_key(t)
    if len(w) >= 24:
        g = [w[i:i + 4] for i in range(len(w) - 3)]
        if len(set(g)) / len(g) < 0.3:
            return "重复循环"
    return None


# ---------------------------------------------------------------- 精度档位
ACCURACY_PRESETS = {
    "standard": dict(
        model="turbo", vad=False, prompt_from_arg=True, beams=None,
        desc="标准：turbo + 原生整流，最快",
    ),
    "high": dict(
        model="turbo", vad=True, prompt_from_arg=True, beams=None,
        desc="高：turbo + VAD 去静音幻觉（推荐日常用）",
    ),
    "max": dict(
        model="largev3", vad=True, prompt_from_arg=True, beams=None,
        desc="最精确：large-v3 + VAD + 显式语言，慢但最准",
    ),
}

# ---------------------------------------------------------------- 模型注册表
@dataclass
class ModelSpec:
    dirname: str
    label: str
    size: str


MODELS: dict[str, ModelSpec] = {
    "turbo": ModelSpec(
        "whisper-large-v3-turbo-fp16-ov",
        "large-v3-turbo FP16  ·  推荐：速度快 6~8 倍，精度接近 large-v3",
        "约 1.5 GB",
    ),
    "largev3": ModelSpec(
        "whisper-large-v3-fp16-ov",
        "large-v3 FP16  ·  精度最高，99 种语言，速度最慢",
        "约 2.9 GB",
    ),
    "turbo-int8": ModelSpec(
        "whisper-large-v3-turbo-int8-ov",
        "large-v3-turbo INT8  ·  比 FP16 更快，精度略降",
        "约 0.9 GB",
    ),
    "largev3-int8": ModelSpec(
        "whisper-large-v3-int8-ov",
        "large-v3 INT8  ·  大模型 + 量化加速",
        "约 1.6 GB",
    ),
}

AUDIO_EXT = {
    ".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma",
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".ts", ".flv", ".wmv", ".m4v",
}

# 常见语言别名 -> Whisper 语言码
LANG_ALIAS = {
    "chinese": "zh", "中文": "zh", "汉语": "zh", "普通话": "zh", "cmn": "zh",
    "english": "en", "英文": "en", "英语": "en", "eng": "en",
    "japanese": "ja", "日文": "ja", "日语": "ja", "jp": "ja",
    "korean": "ko", "韩语": "ko", "kr": "ko",
    "french": "fr", "法语": "fr", "german": "de", "德语": "de",
    "spanish": "es", "西班牙语": "es", "russian": "ru", "俄语": "ru",
    "cantonese": "yue", "粤语": "yue", "auto": "auto", "自动": "auto",
}


# ---------------------------------------------------------------- 音频解码
def decode_audio(path: Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """用 PyAV 解码任意音视频 -> 单声道 float32 16kHz，取值 [-1,1]。"""
    import av

    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise RuntimeError(f"文件里没有音频轨道: {path.name}")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=sr)

        chunks: list[np.ndarray] = []

        def _push(frames) -> None:
            if frames is None:
                return
            if not isinstance(frames, (list, tuple)):
                frames = [frames]
            for f in frames:
                if f is None:
                    continue
                arr = f.to_ndarray()
                if arr.ndim == 2:          # (channels, samples)
                    arr = arr.mean(axis=0)
                chunks.append(np.asarray(arr, dtype=np.float32).reshape(-1))

        for frame in container.decode(stream):
            _push(resampler.resample(frame))
        _push(resampler.resample(None))     # 冲刷缓冲

    if not chunks:
        raise RuntimeError(f"解码后没有音频数据: {path.name}")

    audio = np.concatenate(chunks).astype(np.float32)
    # 归一化保护（有些解码器会给 int16 量级的值）
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.5:
        audio /= peak
    return np.clip(audio, -1.0, 1.0)


def audio_duration(path: Path) -> float:
    try:
        import av
        with av.open(str(path)) as c:
            if c.duration is not None:
                return c.duration / 1_000_000
            for s in c.streams.audio:
                if s.duration is not None and s.time_base:
                    return float(s.duration * s.time_base)
    except Exception:
        pass
    return float("nan")


# ---------------------------------------------------------------- 模型加载
_PIPE_CACHE: dict[tuple, object] = {}


def model_path_of(key: str) -> Path:
    spec = MODELS[key]
    p = MODELS_DIR / spec.dirname
    if not (p / "openvino_encoder_model.xml").exists():
        raise FileNotFoundError(
            f"找不到模型: {p}\n"
            f"请先运行:  venv\\Scripts\\python.exe tools\\download_models.py {key}"
        )
    return p


@functools.lru_cache(maxsize=None)
def model_languages(key: str) -> frozenset[str]:
    """从模型的 generation_config.json 读出 Whisper 支持的语言码（100 个左右）。"""
    try:
        gc = json.loads(
            (model_path_of(key) / "generation_config.json").read_text(encoding="utf-8"))
        ids = gc.get("lang_to_id") or {}
        return frozenset(k.strip("<>|") for k in ids)
    except Exception:
        return frozenset()


# 输出格式名 -> 提醒用户可能想用 -f 而不是 -l
_FORMAT_NAMES = {"txt", "srt", "vtt", "lrc", "json", "all", "text", "subtitle"}

LANG_HINTS = {
    "zh": "中文普通话", "yue": "粤语", "en": "英语", "ja": "日语", "ko": "韩语",
    "fr": "法语", "de": "德语", "es": "西班牙语", "ru": "俄语", "pt": "葡萄牙语",
    "it": "意大利语", "ar": "阿拉伯语", "th": "泰语", "vi": "越南语",
}


def check_language(lang: str, model_key: str) -> str | None:
    """校验语言码。返回错误提示文本；None 表示通过。"""
    if not lang or lang == "auto":
        return None
    langs = model_languages(model_key)
    if not langs or lang in langs:
        return None

    lines = [f"✗ 语言码 '{lang}' 不是 Whisper 支持的语言（模型 '{model_key}' 共支持 {len(langs)} 个）。"]
    if lang in _FORMAT_NAMES:
        lines += [
            "",
            f"  你是不是想指定**输出格式**？那要用 -f 而不是 -l：",
            f'      whisper.bat "你的文件.mp4" -f {lang}',
        ]
    else:
        import difflib
        near = difflib.get_close_matches(lang, sorted(langs), n=3, cutoff=0.5)
        if near:
            lines += ["", f"  最接近的可用语言码：{', '.join(near)}"]

    common = "  ".join(f"{k} {v}" for k, v in list(LANG_HINTS.items())[:8])
    lines += [
        "",
        f"  常用：{common}",
        "  全部列表：--list-languages",
        "  不想指定就让程序自动检测：去掉 -l 参数（默认 auto）",
    ]
    return "\n".join(lines)


def check_device(device: str) -> str | None:
    """校验设备名。返回错误提示；None 表示通过。"""
    if not device:
        return None
    d = device.strip().upper()
    if ":" in d:                          # MULTI:GPU,CPU 之类，交给 OpenVINO 自己解析
        return None
    try:
        import openvino as ov
        core = ov.Core()
        devs = list(core.available_devices)
    except Exception:
        return None                       # 探测失败就不拦，让后面自然报错

    base = d.split(".")[0]
    if base in ("AUTO",):
        return None
    if base not in devs:
        extra = ""
        if not any(x.startswith("GPU") for x in devs):
            extra = "  （没看到 GPU：装/更新 Intel 显卡驱动，Arc 需 31.0.101.4xxx 以上）\n"
        return (f"✗ 设备 '{device}' 不可用。\n\n"
                f"  本机可用设备：{devs}\n"
                f"{extra}"
                f"  用 GPU 最快；实在不行用 CPU。")

    if "." in d:                          # GPU.0 / GPU.1 这种带序号的写法
        idx = d.split(".", 1)[1]
        try:
            avail = [str(x) for x in core.get_property(base, "AVAILABLE_DEVICES")]
        except Exception:
            avail = []
        avail = [a for a in avail if a != ""]
        if avail and idx not in avail:
            return (f"✗ 设备 '{device}' 不存在。\n\n"
                    f"  {base} 的可用编号：{avail}\n"
                    f"  直接用 '{base}' 即可自动选第一个。")
    return None


def cmd_list_languages(model_key: str) -> int:
    langs = model_languages(model_key)
    if not langs:
        print(f"读不到语言表，请检查模型目录: {MODELS_DIR}")
        return 1
    print(f"\n模型 {model_key} 支持 {len(langs)} 个语言码：\n")
    codes = sorted(langs)
    for i in range(0, len(codes), 10):
        row = codes[i:i + 10]
        print("   " + "".join(f"{c:<6}" for c in row))
    print("\n常用对照：")
    for k, v in LANG_HINTS.items():
        print(f"   {k:<5} {v}")
    print("\n示例：  python transcribe.py 音频.mp3 -l zh\n")
    return 0


def build_pipeline(key: str, device: str = "GPU", *, latency: bool = True):
    cache_key = (key, device, latency)
    if cache_key in _PIPE_CACHE:
        return _PIPE_CACHE[cache_key]

    import openvino_genai as og

    props: dict = {
        "PERFORMANCE_HINT": "LATENCY" if latency else "THROUGHPUT",
        "INFERENCE_PRECISION_HINT": "f16",
    }
    if device.upper().startswith("GPU"):
        props["GPU_ENABLE_SDPA_OPTIMIZATION"] = "YES"

    t0 = time.perf_counter()
    pipe = og.WhisperPipeline(str(model_path_of(key)), device, **props)
    dt = time.perf_counter() - t0

    if os.environ.get("WHISPER_VERBOSE"):
        print(f"  [pipeline] {key} @ {device}  加载耗时 {dt:.2f}s", flush=True)

    _PIPE_CACHE[cache_key] = pipe
    return pipe


# ---------------------------------------------------------------- VAD 切片转写
def transcribe_sliced(
    audio: np.ndarray,
    *,
    model: str,
    device: str,
    language: str | None,
    task: str,
    timestamps: bool,
    prompt: str | None,
    hotwords: str | None,
    temperature: float | None,
    max_length: int,
    pipe,
    min_silence_ms: int = 300,
    merge_gap_ms: int = 2500,
    max_block_sec: float = 28.0,
    drop_hallu: bool = True,
    should_stop=None,
    on_slice=None,
) -> tuple[str, list[Segment], list[dict], dict, str | None]:
    """
    VAD 切片后再转写：剔长静音 → 合并成不重叠大块 → 逐块识别 → 时间戳加偏移。

    这是压制 Whisper「静音幻觉」的关键。实测纯静音会让它输出
    "Thank you."，中文更会输出"请不吝点赞 订阅 转发 打赏支持明镜与点点栏目"。
    """
    import vad as vadmod

    t0 = time.perf_counter()
    slices, speech_regions = vadmod.transcribe_slices(
        audio, VAD_DIR, min_silence_ms=min_silence_ms,
        merge_gap_ms=merge_gap_ms, max_block_sec=max_block_sec,
    )
    vad_ms = (time.perf_counter() - t0) * 1000

    if not slices:
        return "", [], [], {
            "wall_seconds": 0.0, "vad_ms": vad_ms, "vad_slices": 0,
            "skipped_all": 1.0, "note": "VAD 判定整段无语音",
        }, None

    texts: list[str] = []
    segs: list[Segment] = []
    words: list[dict] = []
    detected: str | None = None
    agg: dict = {"encode_ms": 0.0, "decode_ms": 0.0, "inference_ms": 0.0,
                 "tokens": 0, "wall_seconds": 0.0}
    dropped: list[str] = []
    infer_t0 = time.perf_counter()
    # 门控需要真实时间戳，所以内部一律要时间戳（与是否输出无关）
    need_ts = True

    for idx, (a, b) in enumerate(slices, 1):
        if should_stop and should_stop():
            agg["cancelled"] = True
            break
        if on_slice:
            try:
                on_slice(idx, len(slices), a, b)
            except Exception:
                pass
        ia, ib = int(a * SAMPLE_RATE), int(b * SAMPLE_RATE)
        chunk = audio[ia:ib]
        if chunk.size < SAMPLE_RATE * 0.15:      # 太短不值得跑
            continue
        txt, csegs, cwords, perf, lg = transcribe_audio(
            chunk, model=model, device=device, language=language, task=task,
            timestamps=need_ts, prompt=prompt, hotwords=hotwords,
            temperature=temperature, max_length=max_length, pipe=pipe,
        )
        if detected is None and lg:
            detected = lg

        for k in ("encode_ms", "decode_ms", "inference_ms", "tokens"):
            v = perf.get(k)
            if isinstance(v, (int, float)):
                agg[k] = agg.get(k, 0) + v

        kept: list[Segment] = []
        if csegs:
            # ---- 第二道门控：落在纯语音区间之外的段，是补白区的幻觉，丢弃 ----
            for s in csegs:
                gs, ge = s.start + a, s.end + a
                ov = vadmod.overlap_ratio(gs, ge, speech_regions)
                if ov < 0.3:
                    dropped.append(
                        f"[{gs:.2f}-{ge:.2f}s] 落在非语音区(重叠{ov*100:.0f}%): {s.text[:50]}")
                    continue
                reason = is_suspicious(s.text)
                if reason:
                    dropped.append(f"[{gs:.2f}-{ge:.2f}s] {reason}: {s.text[:50]}")
                    if drop_hallu:
                        continue
                kept.append(Segment(gs, ge, s.text))
        else:
            # 没有时间戳（理论上不会走到，need_ts 恒为 True）：退回纯文本过滤
            reason = is_suspicious(txt)
            if reason:
                dropped.append(f"[{a:.2f}-{b:.2f}s] {reason}: {txt[:50]}")
                if not drop_hallu:
                    kept.append(Segment(a, b, txt))
            elif txt:
                kept.append(Segment(a, b, txt))

        for s in kept:
            segs.append(s)
            if s.text:
                texts.append(s.text)
        for w in cwords:
            gs = w["start"] + a
            ge = w["end"] + a
            if vadmod.overlap_ratio(gs, ge, speech_regions) >= 0.3:
                words.append({"word": w["word"], "start": gs, "end": ge})

    agg["wall_seconds"] = time.perf_counter() - infer_t0
    agg["vad_ms"] = vad_ms
    agg["vad_slices"] = len(slices)
    agg["vad_speech_sec"] = round(sum(b - a for a, b in slices), 3)
    if dropped:
        agg["hallucination_dropped"] = dropped

    return _smart_join(texts), segs, words, agg, detected


# ---------------------------------------------------------------- 时间格式
def hhmmss(t: float | None, sep: str = ",") -> str:
    """
    SRT / WebVTT 时间戳：HH:MM:SS<sep>mmm

    ⚠ 实现要点：**先取整到整毫秒，再分解时/分/秒**。
    早期版本写成「先 int(t) 拿秒，再对小数部分 round 毫秒」，
    当 t=59.9999 时毫秒进位到 1000、秒变成 60，产出非法的 "00:00:60,000"。
    现在这样写从数学上不可能出现分/秒 = 60。
    """
    if t is None or t != t:              # None 或 NaN
        t = 0.0
    total_ms = int(round(max(0.0, float(t)) * 1000))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def lrc_stamp(t: float | None) -> str:
    """
    LRC 时间戳：[MM:SS.cc]（**厘秒**两位，这是 LRC 规范）。

    同样先取整到整厘秒再分解，避免 [00:60.00] 这种非法值。
    """
    if t is None or t != t:
        t = 0.0
    total_cs = int(round(max(0.0, float(t)) * 100))
    m, rem = divmod(total_cs, 6000)
    s, cs = divmod(rem, 100)
    return f"[{m:02d}:{s:02d}.{cs:02d}]"


def _jsonable(v):
    """把 OpenVINO 的性能指标对象（如 MeanStdPair）转成可 JSON 化的值。"""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    for attr in ("mean", "avg", "value"):
        if hasattr(v, attr):
            try:
                return float(getattr(v, attr))
            except Exception:
                pass
    try:
        return float(v)
    except Exception:
        pass
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    return str(v)


def human_size(n: int | float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


# ---------------------------------------------------------------- 转写核心
@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Result:
    source: Path
    text: str
    segments: list[Segment] = field(default_factory=list)
    words: list[dict] = field(default_factory=list)
    language: str | None = None
    audio_seconds: float = float("nan")
    elapsed: float = 0.0
    perf: dict = field(default_factory=dict)

    @property
    def rtf(self) -> float:
        if not self.audio_seconds or self.audio_seconds != self.audio_seconds:
            return float("nan")
        return self.elapsed / self.audio_seconds


def _normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())


_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af]")


def _smart_join(parts: Iterable[str]) -> str:
    """拼接分段文本：英文用空格，中日韩之间不留空格。"""
    s = " ".join(p.strip() for p in parts if p and p.strip())
    s = re.sub(r"\s+", " ", s)
    # 去掉 CJK 字符之间的空格（Whisper 分段拼接常见问题）
    s = re.sub(r"(?<=[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af])\s+"
               r"(?=[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af])", "", s)
    return s.strip()


def transcribe_audio(
    audio: np.ndarray,
    *,
    model: str = "turbo",
    device: str = "GPU",
    language: str | None = None,
    task: str = "transcribe",
    timestamps: bool = True,
    word_timestamps: bool = False,
    prompt: str | None = None,
    hotwords: str | None = None,
    beams: int | None = None,
    temperature: float | None = None,
    max_length: int = 448,
    pipe=None,
) -> tuple[str, list[Segment], list[dict], dict, str | None]:
    """跑一次推理，返回 (全文, 分段, 词级, 性能指标, 语种)。

    实现说明（踩过的坑，勿改）
    -------------------------
    1) 必须用 **kwargs** 传参，不能传 WhisperGenerationConfig 对象。
       实测：kwargs 传 return_timestamps=True 能拿到正确时间戳
       （0.00 -> 10.40）；而用 config 对象传，同样的设置会退化成
       start=0.0 / end=-1.0（-1 是"无时间戳"哨兵值）。
    2) max_length 必须显式给。config 对象默认是 2^64-1，会触发
       "vector too long"；kwargs 路径给个正常值（448）最稳。
    3) word_timestamps 对这批 FP16-OV 模型不可用：编码器未做
       cross-attention 分解，会抛
       "Encoder attention heads are not decomposed"。这里做优雅降级。
    """
    if pipe is None:
        pipe = build_pipeline(model, device)

    kw: dict = {
        "max_length": int(max_length),
        "return_timestamps": bool(timestamps or word_timestamps),
    }
    if language and language != "auto":
        kw["language"] = language
    if task:
        kw["task"] = task
    if prompt:
        kw["initial_prompt"] = prompt
    if hotwords:
        kw["hotwords"] = hotwords
    if beams and beams > 1:
        kw["num_beams"] = beams
    if temperature is not None:
        kw["temperature"] = temperature

    if word_timestamps:
        kw["word_timestamps"] = True

    t0 = time.perf_counter()
    try:
        res = pipe.generate(audio, **kw)
    except RuntimeError as e:
        if word_timestamps and "decomposed" in str(e):
            if os.environ.get("WHISPER_VERBOSE"):
                print("  [warn] 该模型不支持词级时间戳，已回退到句级。", flush=True)
            kw.pop("word_timestamps", None)
            res = pipe.generate(audio, **kw)
        else:
            raise
    elapsed = time.perf_counter() - t0

    text = _normalize_text(res.texts[0] if res.texts else "")

    segs: list[Segment] = []
    for c in (res.chunks or []):
        s = float(getattr(c, "start_ts", 0.0) or 0.0)
        e = float(getattr(c, "end_ts", 0.0) or 0.0)
        segs.append(Segment(s, e if e > 0 else s, _normalize_text(getattr(c, "text", ""))))

    words: list[dict] = []
    for w in (getattr(res, "words", None) or []):
        words.append({
            "word": getattr(w, "word", ""),
            "start": float(getattr(w, "start_ts", 0.0) or 0.0),
            "end": float(getattr(w, "end_ts", 0.0) or 0.0),
        })

    detected = getattr(res, "language", None) or None
    if isinstance(detected, str):
        detected = detected.strip("<>|") or None

    perf: dict = {"wall_seconds": elapsed}
    try:
        m = res.perf_metrics
        perf.update({
            "load_ms": m.get_load_time(),
            "encode_ms": m.get_encode_inference_duration(),
            "decode_ms": m.get_decode_inference_duration(),
            "inference_ms": m.get_inference_duration(),
            "infer_total_ms": m.get_generate_duration(),
            "tokens": m.get_num_generated_tokens(),
            "throughput_tps": m.get_throughput(),
        })
    except Exception:
        pass
    perf = {k: _jsonable(v) for k, v in perf.items()}

    return text, segs, words, perf, detected


# ---------------------------------------------------------------- 输出格式
# 单条字幕的最短显示时长（秒）。低于这个长度播放器会一闪而过甚至丢弃。
MIN_CUE_SEC = 0.5


def normalize_segments(segs: Iterable[Segment], min_dur: float = MIN_CUE_SEC):
    """
    把识别分段整理成**可直接写成字幕**的序列。

    做四件事，都是为了产出合规文件：
      1. 丢掉空文本的分段（空 cue 是无效的）；
      2. 按开始时间排序（VAD 切片后拼接可能乱序）；
      3. 保证 end > start —— 零时长 cue 很多播放器会丢弃；
      4. 保证不与下一条重叠 —— 重叠会让播放器显示错乱。
    """
    items = []
    for s in segs or []:
        txt = _normalize_text(getattr(s, "text", "") or "")
        if not txt:
            continue
        try:
            a = float(getattr(s, "start", 0.0) or 0.0)
            b = float(getattr(s, "end", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if a != a or b != b:              # NaN
            continue
        items.append([max(0.0, a), max(0.0, b), txt])
    if not items:
        return []

    items.sort(key=lambda x: (x[0], x[1]))
    out: list[Segment] = []
    for i, (a, b, txt) in enumerate(items):
        end = max(b, a + min_dur)
        if i + 1 < len(items):
            nxt = items[i + 1][0]
            if nxt > a:
                end = min(end, nxt)
        if end <= a:                       # 理论上不会走到，兜底
            end = a + 0.001
        out.append(Segment(a, end, txt))
    return out


def _vtt_escape(t: str) -> str:
    """
    WebVTT 的 cue 正文必须转义 & < >，否则会被解析成实体或标签。
    转义 > 之后正文里也不会再出现字面量 '-->'（规范明令禁止），一举两得。
    注意 & 必须最先替换，否则会把后面生成的实体再转义一次。
    """
    return (t.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;"))


def to_srt(segs: Iterable[Segment]) -> str:
    """
    SubRip (.srt)。规范：序号从 1 连续递增；时间戳 HH:MM:SS,mmm；
    箭头两侧各一个空格；块之间用空行分隔；文件以换行结束。
    """
    items = normalize_segments(segs)
    if not items:
        return ""
    blocks = [
        f"{i}\n{hhmmss(s.start, ',')} --> {hhmmss(s.end, ',')}\n{s.text}"
        for i, s in enumerate(items, 1)
    ]
    return "\n\n".join(blocks) + "\n"


def to_vtt(segs: Iterable[Segment]) -> str:
    """
    WebVTT (.vtt)。规范：首行必须是 WEBVTT；其后必须跟一个空行；
    时间戳用 '.' 分隔毫秒；cue 之间空行分隔；cue 正文需转义。
    """
    items = normalize_segments(segs)
    blocks = [
        f"{hhmmss(s.start, '.')} --> {hhmmss(s.end, '.')}\n{_vtt_escape(s.text)}"
        for s in items
    ]
    if not blocks:
        return "WEBVTT\n"
    return "WEBVTT\n\n" + "\n\n".join(blocks) + "\n"


def to_lrc(segs: Iterable[Segment], meta: dict | None = None) -> str:
    """
    LRC 歌词 (.lrc)。规范：时间戳 [MM:SS.cc]，**厘秒两位**；
    可选元数据标签 [ti:标题] [ar:歌手] [al:专辑] [by:制作者] [offset:毫秒]。

    转写场景下每段文本占一行、只标开始时间 —— 这是 LRC 的通行用法。
    """
    items = normalize_segments(segs)
    meta = meta or {}
    lines: list[str] = []
    for key, tag in (("title", "ti"), ("artist", "ar"), ("album", "al")):
        v = (meta.get(key) or "").strip()
        if v:
            lines.append(f"[{tag}:{v}]")
    lines.append(f"[by:{meta.get('by') or 'Whisper / OpenVINO'}]")
    lines.append("[offset:0]")
    if items:
        lines.append("")
    lines.extend(f"{lrc_stamp(s.start)}{s.text}" for s in items)
    return "\n".join(lines) + "\n"


def write_outputs(r: Result, out_dir: Path, formats: list[str],
                  with_time: bool = False, stem: str | None = None,
                  quiet: bool = False) -> list[Path]:
    name = stem or r.source.stem
    if with_time:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        base = out_dir / f"{name}.{stamp}"
    else:
        base = out_dir / name
    base.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # 统一用 LF 写入。不指定 newline 的话 Windows 上 write_text 会自动转成
    # CRLF，导致产物换行符随操作系统变化 —— 不确定的输出没法做回归测试。
    def w(p: Path, content: str) -> None:
        p.write_text(content, encoding="utf-8", newline="\n")
        written.append(p)

    if "txt" in formats:
        w(base.with_suffix(".txt"), r.text + "\n")

    # 字幕 / 歌词都依赖分段。没有分段时不产出文件，
    # 否则会写出空白的 .srt（不是合法 SRT）这种坏文件。
    timed = [f for f in ("srt", "vtt", "lrc") if f in formats]
    if timed and not r.segments:
        if not quiet:
            print(f"  ⓘ 没有语音分段，已跳过 {'/'.join(timed)} 输出"
                  f"（避免生成不合法的空文件）")
        timed = []

    if "srt" in timed:
        w(base.with_suffix(".srt"), to_srt(r.segments))

    if "vtt" in timed:
        w(base.with_suffix(".vtt"), to_vtt(r.segments))

    if "lrc" in timed:
        meta = {"title": r.source.stem}
        w(base.with_suffix(".lrc"), to_lrc(r.segments, meta=meta))

    if "json" in formats:
        w(base.with_suffix(".json"), json.dumps({
            "source": str(r.source),
            "language": r.language,
            "text": r.text,
            "segments": [vars(s) for s in r.segments],
            "words": r.words,
            "audio_seconds": r.audio_seconds,
            "elapsed_seconds": r.elapsed,
            "rtf": r.rtf,
            "speed_vs_realtime": (1.0 / r.rtf) if r.rtf and r.rtf == r.rtf and r.rtf > 0 else None,
            "perf": r.perf,
            "model": r.perf.get("_model"),
            "device": r.perf.get("_device"),
        }, ensure_ascii=False, indent=2))

    return written


def assign_stems(files: list[Path]) -> dict[Path, str]:
    """批量转写时，若不同路径下有同名文件，给后出现的加序号后缀，避免输出互相覆盖。"""
    from collections import Counter
    cnt = Counter(f.stem for f in files)
    seen: dict[str, int] = {}
    out: dict[Path, str] = {}
    for f in files:
        if cnt[f.stem] <= 1:
            out[f] = f.stem
            continue
        seen[f.stem] = seen.get(f.stem, 0) + 1
        out[f] = f"{f.stem}__{seen[f.stem]}"
    return out


# ---------------------------------------------------------------- 单文件流程
def process_file(
    src: Path,
    *,
    args,
    pipe=None,
    quiet: bool = False,
    stem: str | None = None,
    progress=None,
    should_stop=None,
) -> Result | None:
    """progress(dict) 用于向调用方（如 Web UI）推送进度；返回 False 表示中止。
    should_stop() 返回 True 时中止当前文件（在 VAD 分片之间检查）。"""

    def emit(ev: dict) -> bool:
        if not progress:
            return True
        try:
            return progress(ev) is not False
        except Exception:
            return True

    t_all = time.perf_counter()
    if not quiet:
        print(f"\n▶ {src.name}")
    emit({"stage": "decoding", "file": str(src), "name": src.name})

    try:
        audio = decode_audio(src)
    except Exception as e:
        print(f"  ✗ 解码失败: {e}")
        emit({"stage": "error", "file": str(src), "message": f"解码失败: {e}"})
        return None

    dur = audio.size / SAMPLE_RATE
    if not quiet:
        print(f"  音频 {dur:.1f}s ({hhmmss(dur, '.')})  模型 {args.model} @ {args.device}")
    emit({"stage": "decoded", "file": str(src), "name": src.name,
          "duration": dur, "audio_seconds": round(dur, 3)})

    def _slice_cb(idx, total, a, b):
        emit({"stage": "slice", "file": str(src), "index": idx,
              "total": total, "start": round(a, 2), "end": round(b, 2)})

    # 解析输出格式（提前到这里，因为字幕类格式需要分段，会影响是否要时间戳）
    out_dir = Path(args.output_dir) if args.output_dir else src.parent
    formats = list(args.format or ["txt"])
    if formats == ["all"]:
        formats = ["txt", "srt", "vtt", "json"]

    # srt/vtt/lrc 必须带分段才能产出合法文件，所以即使用户写了
    # --no-timestamps，只要要字幕就强制打开时间戳（否则会写出空白文件）
    wants_timed = any(f in ("srt", "vtt", "lrc") for f in formats)
    need_ts = (not args.no_timestamps) or wants_timed
    if wants_timed and args.no_timestamps and not quiet:
        print("  ⓘ 字幕/歌词格式需要时间戳，已忽略 --no-timestamps")

    try:
        if args.vad:
            text, segs, words, perf, detected = transcribe_sliced(
                audio,
                model=args.model, device=args.device, language=args.language,
                task=args.task, timestamps=need_ts,
                prompt=args.prompt, hotwords=args.hotwords,
                temperature=args.temperature, max_length=args.max_length,
                pipe=pipe,
                min_silence_ms=args.vad_min_silence,
                merge_gap_ms=args.vad_merge_gap,
                max_block_sec=args.vad_max_block,
                drop_hallu=not args.keep_hallucination,
                should_stop=should_stop,
                on_slice=_slice_cb,
            )
        else:
            text, segs, words, perf, detected = transcribe_audio(
                audio,
                model=args.model, device=args.device, language=args.language,
                task=args.task, timestamps=need_ts,
                word_timestamps=args.word_timestamps,
                prompt=args.prompt, hotwords=args.hotwords,
                temperature=args.temperature, max_length=args.max_length,
                pipe=pipe,
            )
    except Exception as e:
        print(f"  ✗ 推理失败: {type(e).__name__}: {e}")
        emit({"stage": "error", "file": str(src),
              "message": f"{type(e).__name__}: {e}"})
        return None

    elapsed = time.perf_counter() - t_all
    r = Result(
        source=src, text=text, segments=segs, words=words,
        language=detected, audio_seconds=dur, elapsed=elapsed, perf=perf,
    )
    r.perf["_model"] = args.model
    r.perf["_device"] = args.device

    if not quiet:
        spd = (dur / elapsed) if elapsed > 0 else 0
        extra = ""
        if args.vad:
            ns = perf.get("vad_slices", 0)
            speech = perf.get("vad_speech_sec")
            extra = f"   VAD {ns} 段"
            if speech is not None:
                extra += f"/语音 {speech:.1f}s ({speech / max(dur, 1e-6) * 100:.0f}%)"
            extra += f" {perf.get('vad_ms', 0):.0f}ms"
        print(f"  完成 {elapsed:.2f}s   速度 {spd:.1f}× 实时   字符 {len(text)}{extra}")
        for d in perf.get("hallucination_dropped", []):
            print(f"  ⚠ 已丢弃疑似幻觉: {d}")
        if perf.get("skipped_all"):
            print("  ⓘ 整段未检测到语音，未产生内容。")
        if args.show_text:
            print("  ─" * 20)
            print("  " + (text[:400] + ("..." if len(text) > 400 else "")))
            print("  ─" * 20)

    written = write_outputs(r, out_dir, formats, with_time=args.timestamp_names,
                            stem=stem, quiet=quiet)
    if not quiet:
        for p in written:
            print(f"  → {p}")

    emit({
        "stage": "done",
        "file": str(src),
        "name": src.name,
        "text": r.text,
        "language": r.language,
        "audio_seconds": round(r.audio_seconds, 3),
        "elapsed": round(r.elapsed, 3),
        "speed": round(r.audio_seconds / r.elapsed, 2) if r.elapsed > 0 else None,
        "chars": len(r.text),
        "segments": [{"start": round(s.start, 3), "end": round(s.end, 3),
                      "text": s.text} for s in r.segments],
        "perf": {k: v for k, v in r.perf.items() if not k.startswith("_")},
        "outputs": [str(p) for p in written],
    })
    return r


# ---------------------------------------------------------------- 基准测试
def run_bench(args) -> int:
    files = collect_inputs(args.inputs, args.recursive)
    if not files:
        print("没有可测试的文件。用 --bench 时请指定一个音频文件。")
        return 2

    src = files[0]
    print("=" * 68)
    print(f"  Whisper / OpenVINO 性能基准   文件: {src.name}")
    print("=" * 68)
    audio = decode_audio(src)
    dur = audio.size / SAMPLE_RATE
    print(f"  音频时长: {dur:.2f}s\n")

    rows = []
    keys = args.models or ["turbo", "largev3"]
    devices = args.devices or ["GPU", "CPU"]

    for key in keys:
        try:
            model_path_of(key)
        except FileNotFoundError as e:
            print(f"  跳过 {key}: {e}")
            continue
        for dev in devices:
            try:
                pipe = build_pipeline(key, dev)
            except Exception as e:
                print(f"  ✗ {key} @ {dev} 加载失败: {e}")
                continue
            # 预热
            try:
                transcribe_audio(audio[:SAMPLE_RATE * 3], model=key, device=dev,
                                 language=args.language, pipe=pipe,
                                 timestamps=False)
            except Exception:
                pass
            # 正式计时（2 次取最快）
            best = None
            for _ in range(2):
                t0 = time.perf_counter()
                txt, segs, _w, perf, _lg = transcribe_audio(
                    audio, model=key, device=dev, language=args.language,
                    pipe=pipe, timestamps=True,
                )
                dt = time.perf_counter() - t0
                best = dt if best is None else min(best, dt)
            rows.append({
                "model": key, "device": dev, "sec": best,
                "x": dur / best, "chars": len(txt),
                "enc_ms": perf.get("encode_ms"), "dec_ms": perf.get("decode_ms"),
                "text": txt,
            })
            print(f"  {key:<14} @ {dev:<3}  {best:7.2f}s   {dur / best:6.1f}× 实时   "
                  f"{len(txt)} 字符")

    if rows:
        print("\n" + "-" * 68)
        print(f"{'模型':<14}{'设备':<6}{'耗时(s)':>10}{'实时倍速':>12}{'字符':>9}")
        print("-" * 68)
        for r in sorted(rows, key=lambda x: x["sec"]):
            print(f"{r['model']:<14}{r['device']:<6}{r['sec']:>10.2f}{r['x']:>11.1f}×{r['chars']:>9}")
        print("-" * 68)
        best = min(rows, key=lambda x: x["sec"])
        print(f"\n最快: {best['model']} @ {best['device']}  —  {best['x']:.1f}× 实时")
        print(f"\n转写样例: {best['text'][:200]}")
    return 0


# ---------------------------------------------------------------- 输入收集
def collect_inputs(raw: list[str], recursive: bool) -> list[Path]:
    out: list[Path] = []
    for item in raw:
        p = Path(item).expanduser()
        if p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            out += sorted(f for f in it if f.is_file() and f.suffix.lower() in AUDIO_EXT)
        elif p.is_file():
            out.append(p)
        else:
            print(f"  ! 路径不存在，已跳过: {item}")
    # 去重保序
    seen, uniq = set(), []
    for f in out:
        k = str(f.resolve()).lower()
        if k not in seen:
            seen.add(k)
            uniq.append(f)
    return uniq


# ---------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="transcribe",
        description="Whisper 高性能本地转写（Intel Arc + OpenVINO）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("inputs", nargs="*", help="音频/视频文件或目录，可多个")

    g = p.add_argument_group("模型与设备")
    g.add_argument("-m", "--model", default=None, choices=list(MODELS),
                   help="模型；不指定则由 --accuracy 决定（high/standard→turbo，max→largev3）")
    g.add_argument("-d", "--device", default="GPU",
                   help="设备：GPU / CPU / AUTO / GPU.0；默认 GPU")
    g.add_argument("-a", "--accuracy", default="high",
                   choices=list(ACCURACY_PRESETS),
                   help="精度档位 standard / high / max；默认 high。"
                        "max = large-v3 + VAD + 建议显式指定语言，最准")

    g2 = p.add_argument_group("识别参数")
    g2.add_argument("-l", "--language", default="auto",
                    help="语言码 zh/en/ja... 或 auto。"
                         "⚠ 要最精确就显式指定，别用 auto")
    g2.add_argument("--task", default="transcribe", choices=["transcribe", "translate"],
                    help="transcribe 转写原文（默认）；translate 翻成英文 —— "
                         "⚠ 当前 OpenVINO 版 Whisper 不支持，见 README「已知限制」")
    g2.add_argument("--prompt", default=None,
                    help="initial_prompt，喂术语/专名可显著提准（最精确必备）")
    g2.add_argument("--hotwords", default=None, help="热词表（逗号分隔）")
    g2.add_argument("--temperature", type=float, default=None,
                    help="采样温度；默认贪心解码（最稳）")
    g2.add_argument("--max-length", type=int, default=448,
                    help="解码器单段最大 token 数；默认 448（Whisper 标准值，不建议改）")
    g2.add_argument("--word-timestamps", action="store_true",
                    help="输出词级时间戳（⚠ 这批 FP16 模型不支持，会自动降级到句级）")
    g2.add_argument("--no-timestamps", action="store_true", help="不生成时间戳，速度略快")

    gv = p.add_argument_group("VAD 静音门控（抑制静音幻觉）")
    gv.add_argument("--vad", dest="vad", action="store_true", default=None,
                    help="开启 VAD：剔掉长静音后再识别（默认开启）")
    gv.add_argument("--no-vad", dest="vad", action="store_false",
                    help="关闭 VAD（整段直接识别；静音处可能凭空冒字）")
    gv.add_argument("--vad-min-silence", type=int, default=300,
                    help="静音超过多少毫秒才算一段结束；默认 300")
    gv.add_argument("--vad-merge-gap", type=int, default=2500,
                    help="语音间隔小于多少毫秒就并成同一块（保上下文）；默认 2500。"
                         "调小会更细碎、可能切掉边界字词；调大会把长静音也并进来")
    gv.add_argument("--vad-max-block", type=float, default=28.0,
                    help="单个识别块最长秒数；默认 28（Whisper 内部窗口 30s）")
    gv.add_argument("--keep-hallucination", action="store_true",
                    help="保留疑似幻觉片段（默认丢弃并告警）")

    g3 = p.add_argument_group("输出")
    g3.add_argument("-f", "--format", nargs="+", default=["txt"],
                    choices=["txt", "srt", "vtt", "lrc", "json", "all"],
                    help="输出格式，可多选；默认 txt")
    g3.add_argument("-o", "--output-dir", default=None, help="输出目录；默认与原文件同目录")
    g3.add_argument("-r", "--recursive", action="store_true", help="目录递归子文件夹")
    g3.add_argument("--timestamp-names", action="store_true", help="输出文件名加时间戳，避免覆盖")
    g3.add_argument("--show-text", action="store_true", help="在终端打印识别结果")
    g3.add_argument("-q", "--quiet", action="store_true")

    g4 = p.add_argument_group("其它")
    g4.add_argument("--bench", action="store_true", help="性能基准模式")
    g4.add_argument("--models", nargs="+", choices=list(MODELS),
                    help="--bench 要对比的模型")
    g4.add_argument("--devices", nargs="+", help="--bench 要对比的设备")
    g4.add_argument("--list-models", action="store_true", help="列出模型")
    g4.add_argument("--list-languages", action="store_true",
                    help="列出当前模型支持的全部语言码")
    return p


def print_welcome() -> int:
    print(f"""
  ╭──────────────────────────────────────────────────────────────╮
  │   Whisper 本地转写  ·  OpenVINO  ·  Intel Arc                │
  ╰──────────────────────────────────────────────────────────────╯

  用法一（最简单）：把音频/视频文件拖到 whisper.bat 上，松手即可。
                    默认 high 档：turbo 模型 + VAD 去静音幻觉

  精度档位 -a（越往下越准、越慢）：
     -a standard    turbo，不做 VAD          最快
     -a high        turbo + VAD              ← 默认，日常首选
     -a max         large-v3 + VAD           最准

  用法二（命令行）：
     whisper.bat "D:\\media\\会议录音.mp3"
     whisper.bat "D:\\media" -r -l zh -f srt
     whisper.bat "D:\\media\\讲座.mp4" -a max -l zh -f txt srt json
     whisper.bat "D:\\media\\访谈.wav" -a max -l zh --prompt "术语：OpenVINO、Tauri"
     whisper.bat "D:\\media\\a.wav" --bench
     whisper.bat "D:\\media\\a.wav" --help

  要最精确，三件事：
     1) -a max                     用 large-v3 大模型
     2) -l zh                      显式指定语言（纯单语素材；混合语言才用 auto）
     3) --prompt "人名、术语、专名"   把词表喂给模型

  其它常用参数：
     -m turbo|largev3    指定模型（覆盖档位）
     -f txt srt vtt json 输出格式，可多选
     -o  输出目录         -r  目录递归
     --no-vad            关闭 VAD（静音处会出现"请不吝点赞"之类幻觉文本）
     --list-models       查看模型与下载状态
     --list-languages    查看支持的 100 个语言码

  注意：-l 是**语言**（zh/en/ja...），-f 才是**输出格式**（txt/srt...），别写反。
        参数写错会立即提示，不会白等模型加载。

  完整参数：python transcribe.py --help
  回归自检：python tools\\selftest.py

  模型目录: {MODELS_DIR}
""")
    return 0


def cmd_list_models() -> int:
    print("\n可用模型：\n")
    for k, s in MODELS.items():
        have = "✔ 已下载" if (MODELS_DIR / s.dirname / "openvino_encoder_model.xml").exists() else "✗ 未下载"
        print(f"  {k:<14} {s.label}")
        print(f"  {'':<14} 体积 {s.size}   {have}")
        print()
    print(f"模型目录: {MODELS_DIR}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_models:
        return cmd_list_models()

    # 精度档位：只有用户没显式指定的项才由档位决定
    preset = ACCURACY_PRESETS[args.accuracy]
    if args.model is None:
        args.model = preset["model"]
    if args.vad is None:
        args.vad = preset["vad"]

    # 语言别名归一
    if args.language:
        lang = str(args.language).strip().lower()
        args.language = LANG_ALIAS.get(lang, lang)

    if args.list_languages:
        return cmd_list_languages(args.model)

    # 语言码前置校验 —— 必须在加载模型之前，否则白等十几秒才报错
    err = check_language(args.language, args.model)
    if err:
        print()
        print(err)
        print()
        return 4

    # 设备前置校验（同理，避免加载模型后才炸）
    if not args.bench:
        err = check_device(args.device)
        if err:
            print()
            print(err)
            print()
            return 4

    if args.bench:
        return run_bench(args)

    if args.task == "translate":
        print("⚠ 注意：当前 OpenVINO 版 Whisper 不支持 translate（翻成英文）任务。")
        print("  实测强制指定 translate 词元也不会翻译，仍按原语言转写，")
        print("  这是 OpenVINO GenAI WhisperPipeline 的实现限制，不是配置问题。")
        print("  要继续将按 transcribe 处理；如确实需要翻译，请另配翻译模型。\n")

    if not args.inputs:
        print_welcome()
        return 0

    files = collect_inputs(args.inputs, args.recursive)
    if not files:
        print("没有找到可处理的音视频文件。")
        return 2

    if args.vad:
        import vad as _vadmod
        if not (VAD_DIR / _vadmod.MODEL_NAME).exists():
            print("⚠ 找不到 VAD 模型，已自动降级为 --no-vad（静音处可能出现幻觉文本）")
            print(f"  缺失路径: {VAD_DIR / _vadmod.MODEL_NAME}")
            print("  获取方式: venv\\Scripts\\python.exe tools\\download_models.py vad\n")
            args.vad = False

    print(f"档位 {args.accuracy}  |  模型 {args.model}  |  设备 {args.device}  |  "
          f"语言 {args.language}  |  VAD {'开' if args.vad else '关'}  |  "
          f"任务 {args.task}  |  共 {len(files)} 个文件")
    if args.accuracy == "max" and args.language == "auto":
        print("提示: 已选 max 档位，再显式指定语言（如 -l zh）可进一步提准。")

    t0 = time.perf_counter()
    try:
        pipe = build_pipeline(args.model, args.device)
    except Exception as e:
        print(f"\n✗ 模型/设备初始化失败: {e}")
        print("  提示：先跑  venv\\Scripts\\python.exe tools\\download_models.py")
        return 3
    print(f"模型加载完成 ({time.perf_counter() - t0:.2f}s)\n")

    ok, fail = 0, 0
    results: list[Result] = []
    stems = assign_stems(files)
    if any(stems.get(f) != f.stem for f in files):
        print("提示: 检测到同名文件，已自动加序号后缀避免互相覆盖。\n")
    for f in files:
        r = process_file(f, args=args, pipe=pipe, quiet=args.quiet,
                         stem=stems.get(f))
        if r is None:
            fail += 1
        else:
            ok += 1
            results.append(r)

    total_audio = sum(r.audio_seconds for r in results if r.audio_seconds == r.audio_seconds)
    total_time = sum(r.elapsed for r in results)
    print("\n" + "=" * 60)
    print(f"完成: {ok} 成功, {fail} 失败")
    if total_audio and total_time:
        print(f"音频总长 {hhmmss(total_audio)}  总耗时 {total_time:.1f}s  "
              f"平均 {total_audio / total_time:.1f}× 实时")
    print("=" * 60)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
