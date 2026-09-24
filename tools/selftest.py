#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
端到端回归自检 —— 改完代码后跑这个，确认没有退化。

覆盖：
  1. 环境与设备
  2. VAD 行为（静音/噪声/纯音必须 0 段；语音必须有段）
  3. 转写正确性（有 ground truth 的样本逐字比对）
  4. 静音幻觉抑制（VAD 开必须为空，关必须出幻觉——用来说明这道防线的价值）
  5. VAD 切片不重叠
  6. 速度

用法:
    venv\\Scripts\\python.exe tools\\selftest.py
"""
from __future__ import annotations

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

PASS, FAIL, SKIP = "[PASS]", "[FAIL]", "[SKIP]"
results: list[tuple[str, bool | None, str]] = []


def check(name: str, ok: bool | None, detail: str = "") -> None:
    results.append((name, ok, detail))
    tag = PASS if ok else (SKIP if ok is None else FAIL)
    print(f"  {tag} {name}" + (f"   {detail}" if detail else ""))


def edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def wer(ref: str, hyp: str) -> float:
    import re
    n = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()
    r, h = n(ref).split(), n(hyp).split()
    return edit_distance(r, h) / len(r) if r else 0.0


def main() -> int:
    print("=" * 74)
    print("  Whisper / OpenVINO 端到端回归自检")
    print("=" * 74)

    import openvino_genai as og
    import transcribe as T
    import vad

    # ---------------- 1. 环境 ----------------
    print("\n1. 环境与设备")
    try:
        import openvino as ov
        devs = ov.Core().available_devices
        check("OpenVINO 设备", "GPU" in devs, f"{devs}")
    except Exception as e:
        check("OpenVINO 设备", False, str(e)[:60])
    for k in ("turbo", "largev3"):
        try:
            T.model_path_of(k)
            check(f"模型 {k}", True)
        except Exception as e:
            check(f"模型 {k}", False, str(e)[:60])
    vad_ok = (T.VAD_DIR / vad.MODEL_NAME).exists()
    check("VAD 模型", vad_ok, vad.MODEL_NAME)

    # ---------------- 2. VAD 行为 ----------------
    print("\n2. VAD 行为")
    n30 = 30 * SR
    cases = [
        ("纯静音 -> 0 段", np.zeros(n30, np.float32), True),
        ("底噪 -> 0 段", (np.random.RandomState(0).randn(n30) * 0.0008).astype(np.float32), True),
        ("440Hz 纯音 -> 0 段", (0.1 * np.sin(2 * np.pi * 440 * np.arange(10 * SR) / SR)).astype(np.float32), True),
        ("英文语音 -> 有段", None, False),
        ("中文语音 -> 有段", None, False),
    ]
    jfk = T.decode_audio(S / "jfk.wav") if (S / "jfk.wav").exists() else None
    zh = T.decode_audio(S / "zh_long.wav") if (S / "zh_long.wav").exists() else None
    for name, aud, expect_empty in cases:
        if aud is None:
            aud = jfk if "英文" in name else zh
            if aud is None:
                check(name, None, "缺样本")
                continue
        try:
            t0 = time.perf_counter()
            segs = vad.speech_segments(aud, T.VAD_DIR)
            dt = (time.perf_counter() - t0) * 1000
            got_empty = len(segs) == 0
            check(name, got_empty == expect_empty,
                  f"{len(segs)} 段, {dt:.0f}ms")
        except Exception as e:
            check(name, False, str(e)[:60])

    # ---------------- 3. 切片不重叠 ----------------
    print("\n3. VAD 切片完整性")
    if jfk is not None:
        long_a = np.concatenate([jfk] * 4).astype(np.float32)
        blocks, regions = vad.transcribe_slices(long_a, T.VAD_DIR)
        ov = sum(1 for i in range(len(blocks) - 1) if blocks[i][1] > blocks[i + 1][0] + 1e-6)
        cov = sum(b - a for a, b in blocks) / (len(long_a) / SR)
        check("切片无重叠", ov == 0, f"{len(blocks)} 块, 重叠 {ov} 处")
        check("切片覆盖率 > 90%", cov > 0.90, f"{cov*100:.1f}%")
        sil_blocks, _ = vad.transcribe_slices(np.zeros(n30, np.float32), T.VAD_DIR)
        check("静音切片为空", len(sil_blocks) == 0, f"{len(sil_blocks)} 块")

    # ---------------- 4. 转写正确性 ----------------
    print("\n4. 转写正确性（有 ground truth）")
    GROUND = {
        "jfk.wav": ("en", "And so, my fellow Americans, ask not what your country can do for you, "
                          "ask what you can do for your country."),
    }
    pipe = None
    try:
        pipe = T.build_pipeline("turbo", "GPU")
    except Exception as e:
        check("pipeline 构建", False, str(e)[:60])

    if pipe is not None:
        for fn, (lang, ref) in GROUND.items():
            p = S / fn
            if not p.exists():
                check(f"{fn} 转写", None, "缺样本")
                continue
            aud = T.decode_audio(p)
            txt, segs, _w, perf, _lg = T.transcribe_sliced(
                aud, model="turbo", device="GPU", language=lang, task="transcribe",
                timestamps=True, prompt=None, hotwords=None, temperature=None,
                max_length=448, pipe=pipe,
            )
            w = wer(ref, txt)
            check(f"{fn} 逐字正确 (WER={w*100:.0f}%)", w == 0.0, f"{txt[:60]!r}")

    # ---------------- 5. 静音幻觉抑制 ----------------
    print("\n5. 静音幻觉抑制")
    if pipe is not None:
        sil = np.zeros(n30, np.float32)
        txt_v, *_ = T.transcribe_sliced(
            sil, model="turbo", device="GPU", language="zh", task="transcribe",
            timestamps=True, prompt=None, hotwords=None, temperature=None,
            max_length=448, pipe=pipe)
        ok_v = txt_v.strip() == ""
        check("VAD 开：静音输出为空", ok_v, f"{txt_v[:50]!r}" if txt_v else "空")

        txt_n, *_ = T.transcribe_audio(
            sil, model="turbo", device="GPU", language="zh", task="transcribe",
            timestamps=True, prompt=None, hotwords=None, temperature=None,
            max_length=448, pipe=pipe)
        check("VAD 关：静音会出幻觉（对照，预期为真）",
              txt_n.strip() != "", f"{txt_n[:50]!r}")

    # ---------------- 6. 速度 ----------------
    print("\n6. 速度")
    if pipe is not None and (S / "bench_en_44s.wav").exists():
        aud = T.decode_audio(S / "bench_en_44s.wav")
        for tag, use_vad in (("无 VAD", False), ("有 VAD", True)):
            T.transcribe_audio(aud[:SR * 3], model="turbo", device="GPU",
                               language="en", task="transcribe", timestamps=True,
                               prompt=None, hotwords=None, temperature=None,
                               max_length=448, pipe=pipe)   # 预热
            t0 = time.perf_counter()
            if use_vad:
                T.transcribe_sliced(aud, model="turbo", device="GPU", language="en",
                                    task="transcribe", timestamps=True, prompt=None,
                                    hotwords=None, temperature=None, max_length=448,
                                    pipe=pipe)
            else:
                T.transcribe_audio(aud, model="turbo", device="GPU", language="en",
                                   task="transcribe", timestamps=True, prompt=None,
                                   hotwords=None, temperature=None, max_length=448,
                                   pipe=pipe)
            dt = time.perf_counter() - t0
            check(f"44s 音频 {tag} 速度", dt < 8.0, f"{dt:.2f}s ({44/dt:.1f}× 实时)")

    # ---------------- 7. 参数前置校验 ----------------
    print("\n7. 参数前置校验（必须在加载模型前拦住）")
    try:
        check("合法语言 zh 通过", T.check_language("zh", "largev3") is None)
        check("合法语言 yue 通过", T.check_language("yue", "largev3") is None)
        check("auto 通过", T.check_language("auto", "largev3") is None)
        e = T.check_language("txt", "largev3")
        check("非法语言 txt 被拦", e is not None, (e or "").splitlines()[0][:56] if e else "")
        check("错误提示里提到 -f", e is not None and "-f" in e)
        e2 = T.check_language("zhh", "largev3")
        check("拼错 zhh 给出近似建议", e2 is not None and "zh" in e2)
        langs = T.model_languages("largev3")
        check("语言表读到 100 个码", len(langs) == 100, f"{len(langs)} 个")
        check("设备 GPU 通过", T.check_device("GPU") is None)
        check("设备 GPU.0 通过", T.check_device("GPU.0") is None)
        check("设备 AUTO 通过", T.check_device("AUTO") is None)
        check("设备 GPU.9 被拦", T.check_device("GPU.9") is not None)
        de = T.check_device("NPU")
        check("非法设备 NPU 被拦", de is not None, (de or "").splitlines()[0][:56] if de else "")
    except Exception as e:
        check("参数校验", False, f"{type(e).__name__}: {e}")

    # ---------------- 8. 字幕/歌词格式合规 ----------------
    print("\n8. 字幕 / 歌词格式合规")
    import shutil
    import tempfile
    try:
        import subtitle_check as SC
    except Exception as e:
        check("导入 subtitle_check", False, str(e)[:60])
        SC = None

    # 8a. 时间戳进位边界（曾经产出非法的 00:00:60,000）
    bad_ts = []
    for t in (0.0, 2.5, 59.999, 59.9999, 59.99999, 3599.9999, 3600.0, 7322.123):
        srt, vtt, lrc = T.hhmmss(t, ","), T.hhmmss(t, "."), T.lrc_stamp(t)
        if ":60," in srt or ":60." in vtt or ":60." in lrc:
            bad_ts.append((t, srt, vtt, lrc))
    check("时间戳无 :60 越界（8 个边界值）", not bad_ts,
          f"{bad_ts[0]}" if bad_ts else "")
    check("59.9999s 正确进位到 00:01:00,000",
          T.hhmmss(59.9999, ",") == "00:01:00,000", T.hhmmss(59.9999, ","))
    check("LRC 59.999s 正确进位到 [01:00.00]",
          T.lrc_stamp(59.999) == "[01:00.00]", T.lrc_stamp(59.999))
    check("LRC 用厘秒两位（规范要求）",
          T.lrc_stamp(2.5) == "[00:02.50]", T.lrc_stamp(2.5))

    # 8b. 用刁钻分段实际生成文件，交给校验器体检
    if SC is not None:
        tmp = Path(tempfile.mkdtemp(prefix="wsfmt-"))
        try:
            segs = [
                T.Segment(0.0, 2.5, "正常一句"),
                T.Segment(2.5, 2.5, "零时长"),              # 零时长
                T.Segment(59.9999, 61.0, "进位边界"),        # 触发 :60
                T.Segment(61.0, 63.0, "a & b < c > d"),     # 需转义
                T.Segment(63.0, 63.0, "含 --> 箭头"),        # VTT 禁止
                T.Segment(65.0, 66.0, "   "),               # 空文本
                T.Segment(64.0, 67.0, "与上一条重叠"),        # 重叠+乱序
            ]
            (tmp / "t.srt").write_text(T.to_srt(segs), encoding="utf-8", newline="\n")
            (tmp / "t.vtt").write_text(T.to_vtt(segs), encoding="utf-8", newline="\n")
            (tmp / "t.lrc").write_text(T.to_lrc(segs, meta={"title": "测试"}),
                                       encoding="utf-8", newline="\n")

            for name in ("t.srt", "t.vtt", "t.lrc"):
                rep = SC.check_file(tmp / name)
                check(f"{name} 格式合规", rep.ok,
                      rep.errors[0] if rep.errors else
                      f"{rep.stats} ffmpeg={'ok' if rep.ff else '-'}")

            # 具体条款
            srt_txt = (tmp / "t.srt").read_text(encoding="utf-8")
            vtt_txt = (tmp / "t.vtt").read_text(encoding="utf-8")
            lrc_txt = (tmp / "t.lrc").read_text(encoding="utf-8")

            check("SRT 序号从 1 连续", "\n1\n" in "\n" + srt_txt
                  and "\n2\n" in srt_txt, "")
            check("SRT 用逗号分隔毫秒", "00:00:00,000 -->" in srt_txt, "")
            check("SRT 块间空行", "\n\n" in srt_txt, "")
            check("SRT 末尾有换行", srt_txt.endswith("\n"), "")
            check("VTT 首行为 WEBVTT", vtt_txt.startswith("WEBVTT\n"), vtt_txt[:12])
            check("VTT 头后有空行", vtt_txt.startswith("WEBVTT\n\n"), "")
            check("VTT 用点分隔毫秒", "00:00:00.000 -->" in vtt_txt, "")
            check("VTT 已转义 & < >",
                  "&amp;" in vtt_txt and "&lt;" in vtt_txt and "&gt;" in vtt_txt, "")
            check("VTT 把 '-->' 转义成 '--&gt;'（规范禁止字面量）",
                  "--&gt;" in vtt_txt, "")
            check("LRC 有时间戳且为厘秒", "[00:02.50]" in lrc_txt, "")
            check("LRC 有元数据标签", "[ti:" in lrc_txt and "[offset:0]" in lrc_txt, "")
            check("零时长 cue 已补最小时长", "00:00:02,500 --> 00:00:03,000" in srt_txt, "")

            # 解析出所有时间戳，验证单调不重叠、时长恒正
            # 注意正则要锚定 SRT 时间戳格式，否则正文里的 "含 --> 箭头" 也会被匹配
            TS = r"(\d{2}:\d{2}:\d{2},\d{3})"
            pairs = [(SC._ms(*SC.SRT_TS.match(a).groups()),
                      SC._ms(*SC.SRT_TS.match(b).groups()))
                     for a, b in re.findall(rf"^{TS}\s+-->\s+{TS}$", srt_txt, re.M)]
            check("SRT 时间戳可被解析", len(pairs) == 6, f"{len(pairs)} 条")
            check("SRT 时长恒正", all(b > a for a, b in pairs), "")
            check("SRT 单调不重叠",
                  all(pairs[i][1] <= pairs[i + 1][0] for i in range(len(pairs) - 1)),
                  str(pairs))
            check("空文本 cue 已丢弃（仍为 6 条）", len(pairs) == 6, "")

            # 8c. ffmpeg 交叉验证
            ff_ok = ff_bad = 0
            for name in ("t.srt", "t.vtt"):
                rep = SC.check_file(tmp / name)
                if rep.ff is True:
                    ff_ok += 1
                elif rep.ff is False:
                    ff_bad += 1
            check("ffmpeg 可解析生成的字幕", ff_bad == 0,
                  f"{ff_ok} 个通过" if ff_ok else "PyAV 不可用")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # 8d. 无分段时不产出空白字幕文件（空白 .srt 不是合法 SRT）
    if SC is not None:
        tmp2 = Path(tempfile.mkdtemp(prefix="wsempty-"))
        try:
            class _A:  # 最小 args
                pass
            r = T.Result(source=Path("x.wav"), text="", segments=[], words=[],
                         language=None, audio_seconds=1.0, elapsed=0.1, perf={})
            written = T.write_outputs(r, tmp2, ["txt", "srt", "vtt", "lrc"],
                                      quiet=True)
            names = sorted(p.suffix for p in written)
            check("无分段时不写 srt/lrc（避免非法空文件）",
                  ".srt" not in names and ".lrc" not in names, str(names))
            check("无分段时 txt 仍产出", ".txt" in names, str(names))
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)

    # ---------------- 汇总 ----------------
    n_pass = sum(1 for _, ok, _ in results if ok is True)
    n_fail = sum(1 for _, ok, _ in results if ok is False)
    n_skip = sum(1 for _, ok, _ in results if ok is None)
    print("\n" + "=" * 74)
    print(f"  通过 {n_pass}   失败 {n_fail}   跳过 {n_skip}")
    if n_fail:
        print("  失败项：")
        for name, ok, detail in results:
            if ok is False:
                print(f"    - {name}  {detail}")
    print("=" * 74)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
